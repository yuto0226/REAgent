from __future__ import annotations

import asyncio
import io
import shutil
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
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

from reagent.protocol import TerminalSink
from reagent.rendering import (
    ASSISTANT_BULLET_STYLE,
    GUIDE_STYLE,
    TOOL_BULLET_STYLE,
    RichRenderer,
    SPINNER_FRAMES,
    TERMINAL_THEME,
    _fmt_elapsed,
)
from reagent.results import ErrorResult
from reagent.session import Session
from reagent.session.turn import run_turn

# Map Shift+Enter escape sequences to F20 as a proxy key.
# Terminals supporting kitty keyboard protocol send \x1b[13;2u;
# xterm modifyOtherKeys sends \x1b[27;2;13~.
ANSI_SEQUENCES.setdefault("\x1b[13;2u", Keys.F20)
ANSI_SEQUENCES.setdefault("\x1b[27;2;13~", Keys.F20)

_SPINNER_MS = 120
_SPIN_INTERVAL = _SPINNER_MS / 1000
_DOUBLE_CTRL_C_WINDOW = 1.5  # seconds between two idle Ctrl+C presses to exit
_TOOL_CALL_FLASH_INTERVAL = 0.4

_STYLE = PTStyle.from_dict(
    {
        "separator": "fg:ansibrightblack",
        "prompt-prefix": "fg:ansicyan",
        "thinking-frame": "fg:ansimagenta",
        "thinking": "fg:ansibrightblack",
        "thinking-for": "fg:ansibrightblack",
        "status": "fg:ansiyellow",
        "hint": "fg:ansibrightblack",
    }
)


@dataclass
class _ReplaceableBlock:
    key: str
    render: Callable[[], str]


@dataclass
class _ToolCallDisplay:
    name: str
    args: dict
    started_at: float
    state: Literal["pending", "success", "error"] = "pending"


class _OutputBuffer(io.TextIOBase):
    """Captures Rich console output as a line list for O(tail) display slicing."""

    def __init__(self) -> None:
        self._lines: list[str | _ReplaceableBlock] = []
        self._partial: str = ""  # content after the last newline
        self._app: Application | None = None

    def write(self, s: str) -> int:
        combined = self._partial + s
        *complete, self._partial = combined.split("\n")
        self._lines.extend(complete)
        if self._app is not None:
            self._app.invalidate()
        return len(s)

    def add_replaceable_block(self, key: str, render: Callable[[], str]) -> None:
        self._lines.append(_ReplaceableBlock(key, render))
        if self._app is not None:
            self._app.invalidate()

    def flush(self) -> None:
        pass

    def _rendered_lines(self) -> list[str]:
        lines: list[str] = []
        for line in self._lines:
            if isinstance(line, _ReplaceableBlock):
                lines.extend(line.render().splitlines())
            else:
                lines.append(line)
        return lines

    def get_tail(self, n: int) -> str:
        """Return the last n lines as a joined string — O(n), not O(total)."""
        if n <= 0:
            return ""

        tail: list[str] = []
        if self._partial:
            tail.append(self._partial)

        for line in reversed(self._lines):
            if len(tail) >= n:
                break
            rendered_lines = line.render().splitlines() if isinstance(line, _ReplaceableBlock) else [line]
            tail.extend(reversed(rendered_lines[-(n - len(tail)) :]))

        return "\n".join(reversed(tail))

    def getvalue(self) -> str:
        """Full content for terminal replay after app exit."""
        result = "\n".join(self._rendered_lines())
        if self._partial:
            return result + "\n" + self._partial if result else self._partial
        return result + "\n" if result else ""


def _tool_call_bullet_style(
    state: Literal["pending", "success", "error"],
    *,
    elapsed: float,
) -> RichStyle:
    if state == "success":
        return TOOL_BULLET_STYLE
    if state == "error":
        return RichStyle.parse("red")
    if int(elapsed / _TOOL_CALL_FLASH_INTERVAL) % 2:
        return ASSISTANT_BULLET_STYLE
    return GUIDE_STYLE


class _ContinuationIndent(Processor):
    """Indent continuation lines to align with the first line after '> '.

    Must shift cursor mappings by 2 to account for the added indent characters.
    """

    _N = 2

    def apply_transformation(self, transformation_input) -> Transformation:
        if transformation_input.lineno == 0:
            return Transformation(transformation_input.fragments)
        n = self._N
        return Transformation(
            [("class:prompt-prefix", "  "), *transformation_input.fragments],
            source_to_display=lambda i: i + n,
            display_to_source=lambda i: max(0, i - n),
        )


def _sep_text() -> FormattedText:
    return FormattedText([("class:separator", "─" * shutil.get_terminal_size().columns)])


def _status_needs_spacer(*, is_thinking: bool, status_text: str) -> bool:
    return is_thinking or bool(status_text)


def _status_needs_input_spacer(*, is_thinking: bool, status_text: str) -> bool:
    return _status_needs_spacer(is_thinking=is_thinking, status_text=status_text)


def _format_status(status_text: str, *, style_class: str) -> FormattedText:
    return FormattedText([(f"class:{style_class}", status_text)])


def _format_thinking(frame: str, *, elapsed: float, token_part: str) -> FormattedText:
    return FormattedText(
        [
            ("class:thinking-frame", frame),
            ("class:thinking", f" thinking  ({_fmt_elapsed(elapsed)}{token_part})"),
        ]
    )


async def run(session: Session) -> None:
    output = _OutputBuffer()

    # use_live=False prevents Rich Live from emitting cursor-movement escape codes
    # that would corrupt the static ANSI buffer.
    renderer = RichRenderer(
        console=Console(file=output, force_terminal=True, theme=TERMINAL_THEME),  # type: ignore[arg-type]
        use_live=False,
    )

    # Layout state — mutable list containers so the nested _Sink class methods
    # (which can't use 'nonlocal') share state with key bindings and layout controls.
    _thinking_at: list[float | None] = [None]
    _thinking_phase: list[str] = [""]
    _thinking_tokens: list[int] = [0]
    _status_text: list[str] = [""]
    _status_style: list[str] = ["status"]
    _hint_text: list[str] = [""]
    _active_turn: list[asyncio.Task | None] = [None]
    _last_ctrl_c: list[float] = [0.0]
    _tool_call_displays: dict[str, _ToolCallDisplay] = {}

    # Custom sink: inherits pass-through forwarders from TerminalSink but overrides
    # status/thinking methods to update layout state instead of writing to the buffer.
    def _set_status(msg: str, *, style_class: str = "status") -> None:
        _status_text[0] = msg
        _status_style[0] = style_class
        if output._app:
            output._app.invalidate()

    def _render_tool_call(tool_call_id: str) -> str:
        display = _tool_call_displays[tool_call_id]
        bullet_style = _tool_call_bullet_style(
            display.state,
            elapsed=time.monotonic() - display.started_at,
        )
        with renderer.console.capture() as capture:
            renderer.tool_call(
                display.name,
                display.args,
                tool_call_id=tool_call_id,
                bullet_style=bullet_style,
            )
        return capture.get()

    class _Sink(TerminalSink):
        def on_tool_call(self, tool_call_id: str, name: str, args: dict) -> None:
            _tool_call_displays[tool_call_id] = _ToolCallDisplay(
                name=name,
                args=args,
                started_at=time.monotonic(),
            )
            output.add_replaceable_block(tool_call_id, lambda: _render_tool_call(tool_call_id))

        def on_tool_result(self, tool_call_id: str, result) -> None:
            display = _tool_call_displays.get(tool_call_id)
            if display is not None:
                display.state = "error" if isinstance(result, ErrorResult) else "success"
            super().on_tool_result(tool_call_id, result)

        def on_status(self, msg: str) -> None:
            _set_status(msg)

        def on_thinking_start(self) -> None:
            pass  # spinner driven by _thinking_at set in _process_turns

        def on_thinking_stop(self) -> None:
            pass  # "thinking for Xs" written by _process_turns after clearing _thinking_at

        def on_thinking_update(self, phase: str, tokens: int) -> None:
            _thinking_phase[0] = phase
            _thinking_tokens[0] = tokens
            if output._app:
                output._app.invalidate()

    session._sink = _Sink(renderer)

    buf = Buffer(name="input", multiline=True)
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _is_thinking() -> bool:
        return _thinking_at[0] is not None

    def _get_status() -> FormattedText:
        t = _thinking_at[0]
        if t is not None:
            elapsed = time.monotonic() - t
            frame = SPINNER_FRAMES[int(elapsed * 1000 / _SPINNER_MS) % len(SPINNER_FRAMES)]
            phase = _thinking_phase[0]
            tokens = _thinking_tokens[0]
            arrow = "↑" if phase == "up" else "↓"
            token_part = f"  {arrow}{tokens}" if tokens else ""
            return _format_thinking(frame, elapsed=elapsed, token_part=token_part)
        s = _status_text[0]
        if s:
            return _format_status(s, style_class=_status_style[0])
        return FormattedText([])

    def _has_status() -> bool:
        return _is_thinking() or bool(_status_text[0])

    def _has_status_spacer() -> bool:
        return _status_needs_spacer(
            is_thinking=_is_thinking(),
            status_text=_status_text[0],
        )

    def _has_status_input_spacer() -> bool:
        return _status_needs_input_spacer(
            is_thinking=_is_thinking(),
            status_text=_status_text[0],
        )

    def _get_hint() -> FormattedText:
        s = _hint_text[0]
        return FormattedText([("class:hint", f" {s}")]) if s else FormattedText([])

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        text = buf.text
        if text.endswith("\\"):
            event.current_buffer.delete_before_cursor()
            event.current_buffer.insert_text("\n")
            return
        event.current_buffer.reset()
        stripped = text.strip()
        if not stripped:
            return
        if stripped.lower() in ("/quit", "/exit"):
            input_queue.put_nowait(None)
        else:
            _hint_text[0] = ""
            input_queue.put_nowait(text)

    @kb.add("f20")
    @kb.add("escape", "enter", eager=True)
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-c", eager=True)
    def _ctrl_c(event) -> None:
        t = _active_turn[0]
        if t is not None:
            t.cancel()
            _hint_text[0] = ""
            _last_ctrl_c[0] = 0.0
        else:
            now = time.monotonic()
            if now - _last_ctrl_c[0] <= _DOUBLE_CTRL_C_WINDOW:
                input_queue.put_nowait(None)
            else:
                _last_ctrl_c[0] = now
                event.current_buffer.reset()
                _hint_text[0] = "Press Ctrl+C again to exit"
                if output._app:
                    output._app.invalidate()

    # ESC cancels the running turn; filtered out when idle so emacs bindings work normally.
    @kb.add("escape", filter=Condition(_is_thinking), eager=True)
    def _interrupt(event) -> None:
        t = _active_turn[0]
        if t is not None:
            t.cancel()

    @kb.add("c-d")
    def _eof(event) -> None:
        if not buf.text:
            input_queue.put_nowait(None)

    def _get_output() -> ANSI:
        reserved = (
            4
            + (1 if _has_status() else 0)
            + (1 if _has_status_spacer() else 0)
            + (1 if _has_status_input_spacer() else 0)
            + (1 if _hint_text[0] else 0)
        )
        available = max(1, shutil.get_terminal_size().lines - reserved)
        return ANSI(output.get_tail(available))

    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl(_get_output), dont_extend_height=True),
                ConditionalContainer(
                    Window(height=1),
                    filter=Condition(_has_status_spacer),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(_get_status), height=1),
                    filter=Condition(_has_status),
                ),
                ConditionalContainer(
                    Window(height=1),
                    filter=Condition(_has_status_input_spacer),
                ),
                Window(FormattedTextControl(_sep_text), height=1),
                Window(
                    BufferControl(
                        buffer=buf,
                        input_processors=[BeforeInput("> ", style="class:prompt-prefix"), _ContinuationIndent()],
                        include_default_input_processors=True,
                    ),
                    wrap_lines=True,
                    height=Dimension(min=1, max=10),
                    dont_extend_height=True,
                ),
                Window(FormattedTextControl(_sep_text), height=1),
                ConditionalContainer(
                    Window(FormattedTextControl(_get_hint), height=1),
                    filter=Condition(lambda: bool(_hint_text[0])),
                ),
                Window(),  # spacer: absorbs remaining space below the input
            ]
        )
    )

    app: Application[None] = Application(
        layout=layout,
        key_bindings=merge_key_bindings([load_emacs_bindings(), kb]),
        style=_STYLE,
        full_screen=True,
    )
    output._app = app

    async def _process_turns() -> None:
        while True:
            user_input = await input_queue.get()
            if user_input is None:
                app.exit()
                return

            session.emit_user(user_input)
            session.add_user(user_input)

            turn_task = asyncio.create_task(run_turn(session))
            _active_turn[0] = turn_task
            loop.add_signal_handler(signal.SIGINT, turn_task.cancel)
            _thinking_at[0] = time.monotonic()
            _thinking_phase[0] = ""
            _thinking_tokens[0] = 0
            session.emit_thinking_start()

            async def _spin() -> None:
                while True:
                    await asyncio.sleep(_SPIN_INTERVAL)
                    app.invalidate()

            spin_task = asyncio.create_task(_spin())
            interrupted = False
            try:
                await turn_task
            except asyncio.CancelledError:
                interrupted = True
            finally:
                spin_task.cancel()
                elapsed = time.monotonic() - _thinking_at[0]  # always set before try
                _thinking_at[0] = None
                _thinking_phase[0] = ""
                _thinking_tokens[0] = 0
                _set_status(f"• thinking for {_fmt_elapsed(elapsed)}", style_class="thinking-for")
                _active_turn[0] = None
                session.emit_thinking_stop()
                loop.remove_signal_handler(signal.SIGINT)
                app.invalidate()

            if interrupted:
                _set_status("■ Conversation interrupted")

    process_task = asyncio.create_task(_process_turns())
    try:
        await app.run_async()
    finally:
        process_task.cancel()
        await asyncio.gather(process_task, return_exceptions=True)

    # full_screen uses the alternate screen buffer; on exit replay to the main screen
    # so the conversation history remains visible.
    content = output.getvalue()
    if content:
        sys.stdout.write(content)
        sys.stdout.flush()


def start(session: Session) -> None:
    try:
        asyncio.run(run(session))
    except KeyboardInterrupt:
        pass
