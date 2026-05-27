from __future__ import annotations

import json
import textwrap
from typing import Any

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.segment import Segment, Segments
from rich.style import Style
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text

TERMINAL_THEME = Theme(
    {
        "reagent.assistant_bullet": "white",
        "reagent.think": "dim italic",
        "reagent.tool_bullet": "green",
        "reagent.tool_call": "bold",
        "reagent.guide": "dim",
        "reagent.success": "green",
        "reagent.error": "red",
        "reagent.status": "yellow",
    }
)

ASSISTANT_BULLET_STYLE = Style.parse("white")
TOOL_BULLET_STYLE = Style.parse("green")
GUIDE_STYLE = Style.parse("dim")


class RichRenderer:
    def __init__(self, console: Console | None = None, max_lines: int = 40) -> None:
        self.console = console if console is not None else Console(theme=TERMINAL_THEME)
        self.max_lines = max_lines

    def assistant(self, text: str) -> None:
        if not text:
            return

        self.console.print()

        try:
            self._print_hanging_renderable(Markdown(text), bullet_style=ASSISTANT_BULLET_STYLE)
        except Exception:
            self._print_hanging_lines(text.splitlines(), bullet_style=ASSISTANT_BULLET_STYLE, content_style="")

    def think(self, text: str) -> None:
        if not text:
            return

        self.console.print()
        self._print_hanging_lines(text.splitlines(), bullet_style=GUIDE_STYLE, content_style="reagent.think")

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.console.print()

        if name == "shell" and isinstance(args.get("command"), str):
            self._print_shell_call(args["command"])
            return

        formatted_args = self._fmt_args(name, args)
        lines = self._wrap_call(name, formatted_args)
        self._print_tool_call_lines(lines)

    def tool_result(self, tool_call_id: str, content: str, *, tool_name: str | None = None) -> None:
        del tool_call_id

        lines = self._clip_lines(content)

        if content.startswith("Error:"):
            self._print_tree(lines, style="reagent.error")
            return

        if tool_name in {"write_file", "edit_file"}:
            self._print_tree(lines, style="reagent.success")
            return

        self._print_tree(lines, style="reagent.guide")

    def status(self, msg: str) -> None:
        self.console.print(Text(msg, style="reagent.status"))

    def _print_hanging_renderable(self, renderable: RenderableType, bullet_style: Style) -> None:
        lines = self.console.render_lines(
            renderable, self.console.options.update(width=max(20, self.console.width - 2))
        )

        for index, line in enumerate(lines):
            prefix = "• " if index == 0 else "  "
            self.console.print(Segments([Segment(prefix, bullet_style), *self._rstrip_segments(line), Segment.line()]))

    def _print_hanging_lines(self, lines: list[str], bullet_style: Style, content_style: str) -> None:
        if not lines:
            return

        for index, line in enumerate(lines):
            prefix = "• " if index == 0 else "  "
            self.console.print(Text.assemble((prefix, bullet_style), (line, content_style)))

    def _print_shell_call(self, command: str) -> None:
        logical_lines = command.splitlines() or [""]
        indent = " " * len("• shell(")

        for index, logical_line in enumerate(logical_lines):
            is_first_logical = index == 0
            is_last_logical = index == len(logical_lines) - 1

            prefix_segments = (
                [Segment("• ", TOOL_BULLET_STYLE), Segment("shell(")]
                if is_first_logical
                else [Segment(indent, GUIDE_STYLE)]
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

    def _print_tool_call_lines(self, lines: list[str]) -> None:
        first, *rest = lines
        self.console.print(Text.assemble(("•", "reagent.tool_bullet"), " ", (first, "reagent.tool_call")))

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
