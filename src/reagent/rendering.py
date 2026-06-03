from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment, Segments
from rich.style import Style
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text

from reagent.results import DiffResult, ErrorResult, ReadResult, ShellResult, ToolResult


SPINNER_FRAMES = "☰☱☲☳☴☵☶☷"
_SPINNER_MS = 80


def _fmt_elapsed(elapsed: float) -> str:
    if elapsed >= 60:
        m = int(elapsed // 60)
        s = elapsed % 60
        return f"{m}m {s:.1f}s"
    return f"{elapsed:.1f}s"


class _ThinkingStatus:
    """Live-renderable that updates elapsed time and token counts on every refresh."""

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._phase: str = "up"
        self._tokens: int = 0

    def update(self, phase: str, tokens: int) -> None:
        self._phase = phase
        self._tokens = tokens

    def __rich_console__(self, console, options):
        elapsed = time.monotonic() - self._started_at
        frame = SPINNER_FRAMES[int(elapsed * 1000 / _SPINNER_MS) % len(SPINNER_FRAMES)]
        arrow = "↑" if self._phase == "up" else "↓"
        token_part = f"  {arrow}{self._tokens}" if self._tokens else ""
        stats = f"({elapsed:.1f}s{token_part})"

        yield Segment.line()
        yield Text.assemble((frame, "reagent.spinner_frame"), (f" thinking  {stats}", "dim"))


TERMINAL_THEME = Theme(
    {
        "reagent.assistant_bullet": "white",
        "reagent.think": "dim italic",
        "reagent.tool_bullet": "green",
        "reagent.tool_call": "bold",
        "reagent.guide": "dim",
        "reagent.success": "green",
        "reagent.error": "red",
        "reagent.spinner_frame": "light_slate_blue",
        "reagent.status": "yellow",
        "reagent.prompt": "cyan",
        "reagent.user": "bold grey100 on #554d57",
    }
)

ASSISTANT_BULLET_STYLE = Style.parse("white")
TOOL_BULLET_STYLE = Style.parse("green")
GUIDE_STYLE = Style.parse("dim")


class RichRenderer:
    def __init__(self, console: Console | None = None, max_lines: int = 40, use_live: bool = True) -> None:
        self.console = console if console is not None else Console(theme=TERMINAL_THEME)
        self.max_lines = max_lines
        self._use_live = use_live
        self._live: Live | None = None
        self._thinking_status: _ThinkingStatus | None = None

    def assistant(self, text: str) -> None:
        if not text:
            return

        self.console.print()

        try:
            rendered = self.console.render_lines(
                Markdown(text), self.console.options.update(width=max(20, self.console.width - 2))
            )
            for index, line in enumerate(rendered):
                prefix = "• " if index == 0 else "  "
                self.console.print(
                    Segments([Segment(prefix, ASSISTANT_BULLET_STYLE), *self._rstrip_segments(line), Segment.line()])
                )
        except Exception:
            self._print_hanging_lines(text.splitlines(), bullet_style=ASSISTANT_BULLET_STYLE, content_style="")

    def think(self, text: str) -> None:
        if not text:
            return

        self.console.print()
        self._print_hanging_lines(text.splitlines(), bullet_style=GUIDE_STYLE, content_style="reagent.think")

    def tool_call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        tool_call_id: str = "",
        bullet_style: Style = TOOL_BULLET_STYLE,
    ) -> None:
        del tool_call_id
        self.console.print()

        if name == "shell" and isinstance(args.get("command"), str):
            self._print_shell_call(args["command"], bullet_style=bullet_style)
            return

        formatted_args = self._fmt_args(name, args)
        lines = self._wrap_call(name, formatted_args)
        self._print_tool_call_lines(lines, bullet_style=bullet_style)

    def tool_result(self, tool_call_id: str, result: ToolResult) -> None:
        del tool_call_id
        match result:
            case ErrorResult(message=msg):
                self._print_tree(self._clip_lines(msg), style="reagent.error")
            case ShellResult(output=output):
                self._print_tree(self._clip_lines(output), style="reagent.guide")
            case ReadResult(content=content, path=path, start_line=start_line):
                if content:
                    self._print_read(content, path, start_line)
                else:
                    self._print_tree(["(empty file)"], style="reagent.guide")
            case DiffResult(diff=diff, path=path, message=msg):
                if diff:
                    self._print_diff(diff, path)
                else:
                    self._print_tree([msg], style="reagent.success")

    def _print_read(self, content: str, path: str, start_line: int) -> None:
        lines = content.splitlines()
        omitted = 0
        if len(lines) > self.max_lines:
            omitted = len(lines) - self.max_lines
            lines = lines[: self.max_lines]
        ext = Path(path).suffix.lstrip(".")
        self.console.print(
            Syntax(
                "\n".join(lines),
                ext or "text",
                line_numbers=True,
                start_line=start_line,
                theme="ansi_dark",
                background_color="default",
            )
        )
        if omitted:
            self.console.print(Text(f"    ... +{omitted} lines omitted", style="dim"))

    def _print_diff(self, unified_diff: str, path: str) -> None:
        old_ln = new_ln = 0
        printed = 0
        total = 0

        lines_to_print: list[tuple[str, str, str, str]] = []
        for raw in unified_diff.splitlines():
            if raw.startswith(("---", "+++")):
                continue
            if raw.startswith("@@"):
                m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
                if m:
                    old_ln = int(m.group(1))
                    new_ln = int(m.group(2))
                continue
            if raw.startswith("+"):
                num, marker, style = f"{new_ln:>4}", "+", "reagent.success"
                new_ln += 1
            elif raw.startswith("-"):
                num, marker, style = "    ", "-", "reagent.error"
                old_ln += 1
            else:
                num, marker, style = f"{new_ln:>4}", " ", "dim"
                old_ln += 1
                new_ln += 1
            total += 1
            if printed < self.max_lines:
                lines_to_print.append((num, marker, raw[1:], style))
                printed += 1

        for i, (num, marker, content, style) in enumerate(lines_to_print):
            prefix = "  ⎿ " if i == 0 else "    "
            self.console.print(
                Text.assemble((prefix, GUIDE_STYLE), (num, GUIDE_STYLE), (f" {marker} ", style), (content, style))
            )

        if total > self.max_lines:
            self.console.print(Text(f"    ... +{total - self.max_lines} lines omitted", style="dim"))

    def user(self, text: str) -> None:
        if not text:
            return
        self.console.print()
        width = self.console.width
        for index, line in enumerate(text.splitlines() or [""]):
            prefix = "> " if index == 0 else "  "
            self.console.print(Text(f"{prefix}{line}".ljust(width), style="reagent.user"))

    def status(self, msg: str) -> None:
        self.console.print()
        self.console.print(Text(msg, style="reagent.status"))

    def prompt(self, text: str) -> None:
        self.console.print(Text(text, style="reagent.prompt"), end="")
        self.console.file.flush()

    def thinking_start(self) -> None:
        if not self._use_live:
            self._thinking_status = _ThinkingStatus()
            return
        if self._live is None:
            self._thinking_status = _ThinkingStatus()
            self._live = Live(
                self._thinking_status,
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            self._live.start()

    def thinking_update(self, phase: str, tokens: int) -> None:
        if self._thinking_status is not None:
            self._thinking_status.update(phase, tokens)

    def thinking_stop(self) -> None:
        if self._thinking_status is None:
            return
        elapsed = time.monotonic() - self._thinking_status._started_at
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._thinking_status = None
        self.console.print(Text(f"• thinking for {_fmt_elapsed(elapsed)}", style="dim"))

    def _print_hanging_lines(self, lines: list[str], bullet_style: Style, content_style: str) -> None:
        if not lines:
            return

        for index, line in enumerate(lines):
            prefix = "• " if index == 0 else "  "
            self.console.print(Text.assemble((prefix, bullet_style), (line, content_style)))

    def _print_shell_call(self, command: str, *, bullet_style: Style) -> None:
        logical_lines = command.splitlines() or [""]
        indent = " " * len("• shell(")

        for index, logical_line in enumerate(logical_lines):
            is_first_logical = index == 0
            is_last_logical = index == len(logical_lines) - 1

            prefix_segments = (
                [Segment("• ", bullet_style), Segment("shell(")] if is_first_logical else [Segment(indent, GUIDE_STYLE)]
            )
            suffix_segments = [Segment(")")] if is_last_logical else []
            prefix_width = len("• shell(") if is_first_logical else len(indent)
            suffix_width = 1 if is_last_logical else 0

            width = max(20, self.console.width - prefix_width - suffix_width)
            syntax = Syntax(logical_line, "bash", theme="ansi_dark", background_color="default", word_wrap=True)
            rendered_lines = self.console.render_lines(syntax, self.console.options.update(width=width))

            for rendered_index, rendered_line in enumerate(rendered_lines):
                line_prefix = prefix_segments if rendered_index == 0 else [Segment(indent, GUIDE_STYLE)]
                line_suffix = suffix_segments if rendered_index == len(rendered_lines) - 1 else []

                self.console.print(
                    Segments(
                        [
                            *line_prefix,
                            *self._rstrip_segments(rendered_line),
                            *line_suffix,
                            Segment.line(),
                        ]
                    )
                )

    def _print_tool_call_lines(self, lines: list[str], *, bullet_style: Style) -> None:
        first, *rest = lines
        self.console.print(Text.assemble(("•", bullet_style), " ", (first, "reagent.tool_call")))

        for line in rest:
            self.console.print(Text.assemble(("  ", "reagent.guide"), (line, "reagent.tool_call")))

    def _rstrip_segments(self, segments: list[Segment]) -> list[Segment]:
        stripped = list(segments)
        while stripped and not stripped[-1].text.rstrip():
            stripped.pop()

        if stripped:
            stripped[-1] = Segment(stripped[-1].text.rstrip(), stripped[-1].style)

        return stripped

    def _fmt_args(self, name: str, args: dict[str, Any]) -> str:
        if not args:
            return ""

        if name in {"write_file", "read_file", "edit_file"} and isinstance(args.get("path"), str):
            return self._fmt_file(args)

        try:
            return json.dumps(args, ensure_ascii=False)
        except TypeError:
            return repr(args)

    def _fmt_file(self, args: dict[str, Any]) -> str:
        path = args["path"]
        start = args.get("start_line")
        end = args.get("end_line")

        if isinstance(start, int) and isinstance(end, int):
            return f"{path} {start}:{end}"

        if isinstance(start, int):
            return f"{path} {start}:"

        if isinstance(end, int):
            return f"{path} :{end}"

        return path

    def _wrap_call(self, name: str, formatted_args: str) -> list[str]:
        logical_lines = formatted_args.splitlines() or [""]
        logical_lines[0] = f"{name}({logical_lines[0]}"
        logical_lines[-1] = f"{logical_lines[-1]})"

        first_width = max(20, self.console.width - 2)
        rest_width = max(20, self.console.width - 4)
        lines: list[str] = []

        for logical_line in logical_lines:
            width = first_width if not lines else rest_width
            if not logical_line:
                lines.append("")
                continue
            lines.extend(
                textwrap.wrap(
                    logical_line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

        return lines or [f"{name}()"]

    def _clip_lines(self, content: str) -> list[str]:
        lines = content.splitlines()
        if len(lines) <= self.max_lines:
            return lines

        if self.max_lines < 3:
            return lines[: self.max_lines]

        head_count = max(1, self.max_lines // 2)
        tail_count = self.max_lines - head_count - 1
        omitted = len(lines) - head_count - tail_count

        return [*lines[:head_count], f"... +{omitted} lines omitted", *lines[-tail_count:]]

    def _print_tree(self, lines: list[str], style: str) -> None:
        if not lines:
            return

        wrapped_lines = self._wrap_result_lines(lines)
        first, *rest = wrapped_lines
        self.console.print(Text.assemble(("  ⎿ ", GUIDE_STYLE), (first, style)))

        for line in rest:
            self.console.print(Text.assemble(("    ", GUIDE_STYLE), (line, style)))

    def _wrap_result_lines(self, lines: list[str]) -> list[str]:
        width = max(20, self.console.width - 4)
        wrapped: list[str] = []

        for line in lines:
            if not line:
                wrapped.append("")
                continue

            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )

        return wrapped
