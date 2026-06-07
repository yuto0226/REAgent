from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.key_binding.bindings.emacs import load_emacs_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput, Processor, Transformation
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.style import Style as RichStyle

from reagent.compact import make_compact_fn
from reagent.rendering import (
    ASSISTANT_BULLET_STYLE,
    GUIDE_STYLE,
    TOOL_BULLET_STYLE,
    RichRenderer,
    SPINNER_FRAMES,
    TERMINAL_THEME,
    _fmt_elapsed,
)
from reagent.results import ErrorResult, ToolResult
from reagent.session import Session
from reagent.session.turn import MODEL, run_turn
from reagent.slash_commands import SlashCommand, SlashRender, SlashResult, completions, dispatch

# Map Shift+Enter escape sequences to F20 as a proxy key.
# Terminals supporting kitty keyboard protocol send \x1b[13;2u;
# xterm modifyOtherKeys sends \x1b[27;2;13~.
ANSI_SEQUENCES.setdefault("\x1b[13;2u", Keys.F20)
ANSI_SEQUENCES.setdefault("\x1b[27;2;13~", Keys.F20)

_SPINNER_MS = 120
_SPIN_INTERVAL = _SPINNER_MS / 1000
_CTRL_C_WINDOW = 1.5
_FLASH_INTERVAL = 0.4
_INPUT_PREFIX = "> "
_BUSY_HINT = "Wait for current response to finish"
_EXIT_HINT = "Press Ctrl+C again to exit"
_ERROR_BULLET_STYLE = RichStyle.parse("red")

_STYLE = PTStyle.from_dict(
    {
        "separator": "fg:ansibrightblack",
        "prompt-prefix": "fg:ansicyan",
        "thinking-frame": "fg:ansimagenta",
        "thinking": "fg:ansibrightblack",
        "thinking-for": "fg:ansibrightblack",
        "status": "fg:ansiyellow",
        "hint": "fg:ansibrightblack",
        "completion": "fg:ansibrightblack",
        "completion-selected": "fg:ansibrightmagenta bold",
        "completion-meta": "fg:ansibrightblack",
    }
)


@dataclass
class _Call:
    name: str
    args: dict
    started_at: float
    state: Literal["pending", "success", "error"] = "pending"


@dataclass
class _ReplState:
    thinking_at: float | None = None
    think_phase: str = ""
    think_tokens: int = 0
    status_text: str = ""
    status_style: str = "status"
    hint: str = ""
    active_turn: asyncio.Task | None = None
    last_ctrl_c: float = 0.0
    slash_cmds: list[SlashCommand] = field(default_factory=list)
    slash_idx: int = 0


@dataclass(frozen=True)
class _SlashRoute:
    action: Literal["exit", "handled", "submit"]
    prompt: str = ""
    message: str = ""
    render: SlashRender = "default"


class _Outbox:
    """Commits stable output above the active prompt."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Callable[[], None] | None] = asyncio.Queue()

    def submit(self, print_fn: Callable[[], None]) -> None:
        self._queue.put_nowait(print_fn)

    async def run(self) -> None:
        while True:
            print_fn = await self._queue.get()
            try:
                if print_fn is None:
                    return
                await run_in_terminal(print_fn)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    def stop(self) -> None:
        self._queue.put_nowait(None)


class _PendingCalls:
    def __init__(self) -> None:
        self._cap_console = Console(force_terminal=True, theme=TERMINAL_THEME, highlight=False)
        self._cap_renderer = RichRenderer(console=self._cap_console, use_live=False)
        self.calls: dict[str, _Call] = {}

    def render(self, *, width: int) -> str:
        self._cap_console.width = width
        chunks: list[str] = []

        for tool_call_id, call in self.calls.items():
            elapsed = time.monotonic() - call.started_at
            bullet = _tool_call_bullet_style(call.state, elapsed=elapsed)
            with self._cap_console.capture() as capture:
                self._cap_renderer.tool_call(
                    call.name,
                    call.args,
                    tool_call_id=tool_call_id,
                    bullet_style=bullet,
                )
            raw = capture.get().strip("\n")
            if raw:
                chunks.append(raw)

        return "\n".join(chunks)


def _tool_call_bullet_style(
    state: Literal["pending", "success", "error"],
    *,
    elapsed: float,
) -> RichStyle:
    if state == "success":
        return TOOL_BULLET_STYLE
    if state == "error":
        return _ERROR_BULLET_STYLE
    if int(elapsed / _FLASH_INTERVAL) % 2:
        return ASSISTANT_BULLET_STYLE
    return GUIDE_STYLE


class _ContinuationIndent(Processor):
    _N = len(_INPUT_PREFIX)

    def apply_transformation(self, transformation_input) -> Transformation:
        if transformation_input.lineno == 0:
            return Transformation(transformation_input.fragments)
        n = self._N
        return Transformation(
            [("class:prompt-prefix", "  "), *transformation_input.fragments],
            source_to_display=lambda i: i + n,
            display_to_source=lambda i: max(0, i - n),
        )


def _sep_text(width: int) -> FormattedText:
    return FormattedText([("class:separator", "─" * width)])


def _fmt_status(status_text: str, *, style_class: str) -> FormattedText:
    return FormattedText([(f"class:{style_class}", status_text)])


def _fmt_hint(hint: str) -> FormattedText:
    return FormattedText([("class:hint", f" {hint}" if hint else "")])


def _fmt_thinking(frame: str, *, elapsed: float, token_part: str) -> FormattedText:
    return FormattedText(
        [
            ("class:thinking-frame", frame),
            ("class:thinking", f" thinking  ({_fmt_elapsed(elapsed)}{token_part})"),
        ]
    )


def _enter_action(text: str, *, is_running: bool) -> Literal["newline", "hint", "submit", "ignore"]:
    if text.endswith("\\"):
        return "newline"
    stripped = text.strip()
    if not stripped:
        return "ignore"
    if is_running:
        return "hint"
    return "submit"


def _route_slash_result(user_input: str, result: SlashResult) -> _SlashRoute:
    if result.outcome == "not_slash":
        return _SlashRoute(action="submit", prompt=user_input)
    if result.outcome == "submit_prompt":
        return _SlashRoute(action="submit", prompt=result.prompt)
    if result.outcome == "exit":
        return _SlashRoute(action="exit")
    # covers "handled" and "unknown"
    return _SlashRoute(action="handled", message=result.message, render=result.render)


def _exit_hint_expired(*, now: float, last_ctrl_c: float, hint: str, active_turn: bool) -> bool:
    return not active_turn and hint == _EXIT_HINT and last_ctrl_c > 0.0 and now - last_ctrl_c >= _CTRL_C_WINDOW


def _fmt_usage(
    *,
    total: int,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
) -> str:
    input_part = f"input={input_tokens:,}"
    if cached_tokens:
        input_part += f" (+ {cached_tokens:,} cached)"

    output_part = f"output={output_tokens:,}"
    if reasoning_tokens:
        output_part += f" (reasoning {reasoning_tokens:,})"

    return f"Usage: total={total:,} {input_part} {output_part}"


def _fmt_usage_line(text: str) -> str:
    return f"\n{text}"


def _print_usage(console: Console, text: str) -> None:
    console.print(_fmt_usage_line(text), highlight=False)


def _make_app(*, layout: Layout | None, key_bindings: KeyBindingsBase | None) -> Application[None]:
    return Application(
        layout=layout,
        key_bindings=key_bindings,
        style=_STYLE,
        full_screen=False,
        erase_when_done=True,
        refresh_interval=_SPIN_INTERVAL,
    )


async def run(session: Session) -> None:
    terminal_console = Console(force_terminal=True, theme=TERMINAL_THEME)
    terminal_renderer = RichRenderer(console=terminal_console, use_live=False)
    state = _ReplState()
    calls = _PendingCalls()
    outbox = _Outbox()

    def _invalidate() -> None:
        if output_app is not None:
            output_app.invalidate()

    def _set_status(msg: str, *, style_class: str = "status") -> None:
        state.status_text = msg
        state.status_style = style_class
        _invalidate()

    def _commit(render: Callable[[str], None], text: str) -> None:
        if text:
            outbox.submit(lambda: render(text))

    def _show_call(tool_call_id: str, name: str, args: dict) -> None:
        calls.calls[tool_call_id] = _Call(name=name, args=args, started_at=time.monotonic())
        _invalidate()

    def _commit_result(tool_call_id: str, result: ToolResult) -> None:
        call = calls.calls.pop(tool_call_id, None)

        def _print() -> None:
            if call is not None:
                style = _ERROR_BULLET_STYLE if isinstance(result, ErrorResult) else TOOL_BULLET_STYLE
                terminal_renderer.tool_call(call.name, call.args, tool_call_id=tool_call_id, bullet_style=style)
            terminal_renderer.tool_result(tool_call_id, result)

        outbox.submit(_print)
        _invalidate()

    class _Sink:
        def on_assistant(self, text: str) -> None:
            _commit(terminal_renderer.assistant, text)

        def on_think(self, text: str) -> None:
            _commit(terminal_renderer.think, text)

        def on_tool_call(self, tool_call_id: str, name: str, args: dict) -> None:
            _show_call(tool_call_id, name, args)

        def on_tool_result(self, tool_call_id: str, result) -> None:
            _commit_result(tool_call_id, result)

        def on_user(self, text: str) -> None:
            _commit(terminal_renderer.user, text)

        def on_status(self, msg: str) -> None:
            _set_status(msg)

        def on_prompt(self, text: str) -> None:
            pass

        def on_thinking_start(self) -> None:
            pass

        def on_thinking_stop(self) -> None:
            pass

        def on_thinking_update(self, phase: str, tokens: int) -> None:
            state.think_phase = phase
            state.think_tokens = tokens
            _invalidate()

    session._sink = _Sink()

    buf = Buffer(name="input", multiline=True)
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    output_app: Application[None] | None = None
    hint_clear_task: asyncio.Task | None = None

    def _is_running() -> bool:
        return state.active_turn is not None

    def _is_thinking() -> bool:
        return state.thinking_at is not None

    def _get_calls() -> ANSI:
        return ANSI(calls.render(width=terminal_console.width))

    def _get_status() -> FormattedText:
        t = state.thinking_at
        if t is not None:
            elapsed = time.monotonic() - t
            frame = SPINNER_FRAMES[int(elapsed * 1000 / _SPINNER_MS) % len(SPINNER_FRAMES)]
            arrow = "↑" if state.think_phase == "up" else "↓"
            token_part = f"  {arrow}{state.think_tokens}" if state.think_tokens else ""
            return _fmt_thinking(frame, elapsed=elapsed, token_part=token_part)
        if state.status_text:
            return _fmt_status(state.status_text, style_class=state.status_style)
        return FormattedText([])

    def _has_calls() -> bool:
        return bool(calls.calls)

    def _has_status() -> bool:
        return _is_thinking() or bool(state.status_text)

    def _has_gap() -> bool:
        return _has_calls() or _has_status()

    def _get_hint() -> FormattedText:
        return _fmt_hint(state.hint)

    def _has_completions() -> bool:
        return bool(state.slash_cmds)

    def _get_completion_panel() -> FormattedText:
        cmds = state.slash_cmds
        idx = state.slash_idx
        name_w = max(len(c.name) for c in cmds) + 1  # +1 for /
        lines: list[tuple[str, str]] = []
        for i, cmd in enumerate(cmds):
            sel = i == idx
            base = "class:completion-selected" if sel else ""
            meta = "class:completion-selected" if sel else "class:completion-meta"
            name = f"/{cmd.name}".ljust(name_w + 1)
            if i > 0:
                lines.append(("", "\n"))
            lines += [(base, f"  {name}  "), (meta, cmd.description)]
        return FormattedText(lines)

    def _update_completions(_buf: Buffer) -> None:
        text = _buf.text
        if text.startswith("/") and " " not in text and "\n" not in text:
            cmds = list(completions(text))
            if cmds != state.slash_cmds:
                state.slash_cmds = cmds
                state.slash_idx = 0
                _invalidate()
        elif state.slash_cmds:
            state.slash_cmds = []
            state.slash_idx = 0
            _invalidate()

    buf.on_text_changed += _update_completions

    def _schedule_ctrl_c_hint_clear() -> None:
        nonlocal hint_clear_task
        if hint_clear_task is not None:
            hint_clear_task.cancel()

        async def _clear_after_window() -> None:
            await asyncio.sleep(_CTRL_C_WINDOW)
            if _exit_hint_expired(
                now=time.monotonic(),
                last_ctrl_c=state.last_ctrl_c,
                hint=state.hint,
                active_turn=state.active_turn is not None,
            ):
                state.hint = ""
                state.last_ctrl_c = 0.0
                _invalidate()

        hint_clear_task = asyncio.create_task(_clear_after_window())

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        if state.slash_cmds:
            cmd = state.slash_cmds[state.slash_idx]
            state.slash_cmds = []
            text = f"/{cmd.name}"
            buf.reset()
            state.hint = ""
            input_queue.put_nowait(text)
            return
        action = _enter_action(buf.text, is_running=_is_running())
        if action == "newline":
            event.current_buffer.delete_before_cursor()
            event.current_buffer.insert_text("\n")
        elif action == "hint":
            state.hint = _BUSY_HINT
            _invalidate()
        elif action == "submit":
            text = buf.text
            event.current_buffer.reset()
            state.hint = ""
            input_queue.put_nowait(text)

    @kb.add("down", filter=Condition(_has_completions), eager=True)
    @kb.add("tab", filter=Condition(_has_completions), eager=True)
    def _completion_down(event) -> None:
        state.slash_idx = (state.slash_idx + 1) % len(state.slash_cmds)
        _invalidate()

    @kb.add("up", filter=Condition(_has_completions), eager=True)
    def _completion_up(event) -> None:
        state.slash_idx = (state.slash_idx - 1) % len(state.slash_cmds)
        _invalidate()

    @kb.add("escape", filter=Condition(_has_completions), eager=True)
    def _dismiss_completions(event) -> None:
        state.slash_cmds = []
        _invalidate()

    @kb.add("f20")
    @kb.add("escape", "enter", eager=True)
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-c", eager=True)
    def _ctrl_c(event) -> None:
        t = state.active_turn
        if t is not None:
            t.cancel()
            state.hint = ""
            state.last_ctrl_c = 0.0
        else:
            now = time.monotonic()
            if now - state.last_ctrl_c <= _CTRL_C_WINDOW:
                input_queue.put_nowait(None)
            else:
                state.last_ctrl_c = now
                event.current_buffer.reset()
                state.hint = _EXIT_HINT
                _schedule_ctrl_c_hint_clear()
                _invalidate()

    @kb.add("escape", filter=Condition(_is_thinking), eager=True)
    def _interrupt(event) -> None:
        t = state.active_turn
        if t is not None:
            t.cancel()

    @kb.add("c-d")
    def _eof(event) -> None:
        if not buf.text:
            input_queue.put_nowait(None)

    layout = Layout(
        # Active tail: gap, pending calls, status, input separator, input,
        # slash completion panel, hint separator, hint/blank.
        HSplit(
            [
                ConditionalContainer(
                    Window(height=1),
                    filter=Condition(_has_gap),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(_get_calls), dont_extend_height=True),
                    filter=Condition(_has_calls),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(_get_status), height=1),
                    filter=Condition(_has_status),
                ),
                ConditionalContainer(
                    Window(height=1),
                    filter=Condition(_has_status),
                ),
                Window(FormattedTextControl(lambda: _sep_text(terminal_console.width)), height=1),
                Window(
                    BufferControl(
                        buffer=buf,
                        input_processors=[
                            BeforeInput(_INPUT_PREFIX, style="class:prompt-prefix"),
                            _ContinuationIndent(),
                        ],
                        include_default_input_processors=True,
                    ),
                    wrap_lines=True,
                    height=Dimension(min=1, max=10),
                    dont_extend_height=True,
                ),
                Window(FormattedTextControl(lambda: _sep_text(terminal_console.width)), height=1),
                ConditionalContainer(
                    Window(FormattedTextControl(_get_completion_panel), dont_extend_height=True),
                    filter=Condition(_has_completions),
                ),
                Window(FormattedTextControl(_get_hint), height=1),
            ]
        ),
        focused_element=buf,
    )

    output_app = _make_app(
        layout=layout,
        key_bindings=merge_key_bindings([load_emacs_bindings(), kb]),
    )

    async def _process_turns() -> None:
        compact_fn = make_compact_fn(MODEL)
        while True:
            user_input = await input_queue.get()
            if user_input is None:
                output_app.exit()
                return

            route = _route_slash_result(user_input, dispatch(user_input, session, compact_fn=compact_fn))
            if route.action == "exit":
                output_app.exit()
                return
            if route.action == "handled":
                if route.render == "panel":
                    outbox.submit(lambda: terminal_renderer.status_panel(session))
                elif route.message:
                    render_fn = {
                        "error": terminal_renderer.error,
                        "notice": terminal_renderer.notice,
                    }.get(route.render, terminal_renderer.status)
                    _commit(render_fn, route.message)
                await outbox.drain()
                continue

            user_input = route.prompt
            session.emit_user(user_input)
            await outbox.drain()
            session.add_user(user_input)

            turn_task = asyncio.create_task(run_turn(session))
            state.active_turn = turn_task
            loop.add_signal_handler(signal.SIGINT, turn_task.cancel)
            state.thinking_at = time.monotonic()
            state.think_phase = ""
            state.think_tokens = 0
            session.emit_thinking_start()
            _invalidate()

            interrupted = False
            try:
                await turn_task
            except asyncio.CancelledError:
                interrupted = True
            finally:
                elapsed = time.monotonic() - (state.thinking_at or 0.0)
                state.thinking_at = None
                state.think_phase = ""
                state.think_tokens = 0
                _set_status(f"• thinking for {_fmt_elapsed(elapsed)}", style_class="thinking-for")
                state.active_turn = None
                session.emit_thinking_stop()
                loop.remove_signal_handler(signal.SIGINT)
                await outbox.drain()
                output_app.invalidate()

            if interrupted:
                _set_status("■ Conversation interrupted")

    process_task = asyncio.create_task(_process_turns())
    outbox_task = asyncio.create_task(outbox.run())
    try:
        await output_app.run_async()
    finally:
        process_task.cancel()
        await asyncio.gather(process_task, return_exceptions=True)
        if hint_clear_task is not None:
            hint_clear_task.cancel()
            await asyncio.gather(hint_clear_task, return_exceptions=True)
        await outbox.drain()
        outbox.stop()
        await asyncio.gather(outbox_task, return_exceptions=True)
        _print_usage(
            terminal_console,
            _fmt_usage(
                total=session.total_tokens,
                input_tokens=session.prompt_tokens,
                output_tokens=session.completion_tokens,
                cached_tokens=session.cached_tokens,
                reasoning_tokens=session.reasoning_tokens,
            ),
        )


def start(session: Session) -> None:
    try:
        asyncio.run(run(session))
    except KeyboardInterrupt:
        pass
