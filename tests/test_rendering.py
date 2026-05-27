from rich.console import Console
from rich.color import ColorType

from reagent.rendering import RichRenderer


def make_renderer(max_lines: int = 80, width: int = 100) -> tuple[RichRenderer, Console]:
    console = Console(record=True, force_terminal=False, width=width)
    return RichRenderer(console=console, max_lines=max_lines), console


def test_assistant_renders_markdown_text():
    renderer, console = make_renderer()

    renderer.assistant("# Heading\n\n- item\n\n```python\nprint('hi')\n```")

    output = console.export_text()
    assert "Heading" in output
    assert "item" in output
    assert "print" in output


def test_assistant_output_uses_white_bullet_prefix():
    renderer, console = make_renderer(width=28)

    renderer.assistant("hello world this should wrap")

    segments = console._record_buffer
    assert any(
        segment.text == "• " and segment.style and segment.style.color and segment.style.color.name == "white"
        for segment in segments
    )
    output = console.export_text()
    assert "• hello world" in output
    assert "\n  " in output


def test_assistant_markdown_wraps_inside_console_width():
    renderer, console = make_renderer(width=36)

    renderer.assistant("This is a long markdown paragraph that should wrap inside the configured console width.")

    output = console.export_text()
    lines = [line for line in output.splitlines() if line]
    assert len(lines) > 1
    assert all(len(line) <= 36 for line in lines)
    assert all(line.startswith(("• ", "  ")) for line in lines)


def test_thinking_is_rendered_with_content():
    renderer, console = make_renderer()

    renderer.think("considering options")

    assert "considering options" in console.export_text()


def test_thinking_output_uses_dim_bullet_prefix():
    renderer, console = make_renderer()

    renderer.think("thinking\nmore")

    segments = console._record_buffer
    assert any(segment.text == "• " and segment.style and segment.style.dim for segment in segments)
    output = console.export_text()
    assert "• thinking" in output
    assert "  more" in output


def test_tool_call_renders_name_and_arguments():
    renderer, console = make_renderer()

    renderer.tool_call("shell", {"command": "git status --short"})

    output = console.export_text()
    assert "• shell(git status --short)" in output
    assert "command" not in output


def test_shell_tool_call_highlights_command_segments():
    renderer, console = make_renderer()

    renderer.tool_call("shell", {"command": "echo '$HOME' && git status --short"})

    segments = console._record_buffer
    assert any(segment.text == "echo" and segment.style and segment.style.color for segment in segments)


def test_shell_tool_call_uses_ansi_colors_not_truecolor():
    renderer, console = make_renderer()

    renderer.tool_call("shell", {"command": "echo '$HOME' && git status --short"})

    styled_segments = [
        segment for segment in console._record_buffer if segment.style is not None and segment.style.color is not None
    ]
    assert styled_segments
    assert all(
        segment.style is not None
        and segment.style.color is not None
        and segment.style.color.type != ColorType.TRUECOLOR
        for segment in styled_segments
    )


def test_long_tool_call_wraps_aligned_to_open_paren():
    renderer, console = make_renderer(width=58)

    renderer.tool_call(
        "shell",
        {
            "command": (
                "rtk uv run pytest tests/test_rendering.py::test_read_file_tool_call_renders_path_and_range "
                "tests/test_rendering.py::test_write_file_tool_call_renders_path -q"
            )
        },
    )

    output = console.export_text()
    assert "• shell(rtk uv run pytest" in output
    assert "\n        " in output
    assert "\n  │ " not in output


def test_tool_call_continuation_indent_is_dim():
    renderer, console = make_renderer(width=30)

    renderer.tool_call("shell", {"command": "python3 - <<'PY'\nprint('hello world')\nPY"})

    segments = console._record_buffer
    assert any(segment.text == "        " and segment.style and segment.style.dim for segment in segments)


def test_shell_tool_call_parens_are_not_shell_highlighted():
    renderer, console = make_renderer()

    renderer.tool_call("shell", {"command": "git status --short"})

    segments = console._record_buffer
    paren_segments = [segment for segment in segments if segment.text in {"shell(", ")"}]
    assert paren_segments
    assert all(not (segment.style and segment.style.color) for segment in paren_segments)


def test_multiline_tool_call_preserves_logical_lines():
    renderer, console = make_renderer(width=80)

    renderer.tool_call("shell", {"command": "python3 - <<'PY'\nprint('hello world')\nPY"})

    output = console.export_text()
    assert "• shell(python3 - <<'PY'" in output
    assert "        print('hello world')" in output
    assert "        PY)" in output
    assert "\\n" not in output


def test_read_file_tool_call_renders_path_and_range():
    renderer, console = make_renderer()

    renderer.tool_call("read_file", {"path": "src/reagent/protocol.py", "start_line": 10, "end_line": 20})

    output = console.export_text()
    assert "• read_file(src/reagent/protocol.py 10:20)" in output
    assert "start_line" not in output
    assert "end_line" not in output


def test_write_file_tool_call_renders_path():
    renderer, console = make_renderer()

    renderer.tool_call("write_file", {"path": "src/reagent/rendering.py", "content": "secret content"})

    output = console.export_text()
    assert "• write_file(src/reagent/rendering.py)" in output
    assert "secret content" not in output


def test_edit_file_tool_call_renders_path_and_range():
    renderer, console = make_renderer()

    renderer.tool_call("edit_file", {"path": "src/reagent/protocol.py", "start_line": 10, "end_line": 12})

    output = console.export_text()
    assert "• edit_file(src/reagent/protocol.py 10:12)" in output
    assert "start_line" not in output
    assert "end_line" not in output


def test_tool_result_renders_generic_output():
    renderer, console = make_renderer()

    renderer.tool_result("call-1", "line one\nline two")

    output = console.export_text()
    assert "  ⎿ line one" in output
    assert "    line two" in output


def test_tool_result_wraps_long_lines_with_content_indent():
    renderer, console = make_renderer(width=34)

    renderer.tool_result("call-1", "abcdefghijklmnopqrstuvwxyz0123456789")

    output = console.export_text()
    lines = output.splitlines()
    assert lines[0].startswith("  ⎿ ")
    assert lines[1].startswith("    ")
    assert all(len(line) <= 34 for line in lines)


def test_tool_result_tree_prefix_is_dim():
    renderer, console = make_renderer()

    renderer.tool_result("call-1", "line one\nline two")

    segments = console._record_buffer
    assert any(segment.text == "  ⎿ " and segment.style and segment.style.dim for segment in segments)


def test_tool_result_renders_error_output():
    renderer, console = make_renderer()

    renderer.tool_result("call-1", "Error: failed")

    assert "Error: failed" in console.export_text()


def test_tool_result_truncates_middle_lines():
    renderer, console = make_renderer(max_lines=5)

    renderer.tool_result("call-1", "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7")

    output = console.export_text()
    assert "  ⎿ line 1" in output
    assert "    line 2" in output
    assert "    ... +3 lines omitted" in output
    assert "    line 6" in output
    assert "    line 7" in output
    assert "line 3" not in output


def test_status_renders_message():
    renderer, console = make_renderer()

    renderer.status("compacting...")

    assert "compacting..." in console.export_text()


def test_read_file_result_renders_numbered_content():
    renderer, console = make_renderer()

    renderer.tool_result("call-1", "1: from __future__ import annotations\n2: print('hi')", tool_name="read_file")

    output = console.export_text()
    assert "from __future__ import annotations" in output
    assert "print" in output


def test_write_file_result_renders_success_message():
    renderer, console = make_renderer()

    renderer.tool_result("call-1", "Written 12 bytes to /tmp/example.txt", tool_name="write_file")

    assert "Written 12 bytes" in console.export_text()


def test_tool_result_empty_content_produces_no_output():
    renderer, console = make_renderer()

    renderer.tool_result("call-1", "")

    assert console.export_text().strip() == ""


def test_tool_result_truncates_to_max_lines_exactly():
    renderer, console = make_renderer(max_lines=3)

    renderer.tool_result("call-1", "a\nb\nc\nd\ne")

    output = console.export_text()
    assert "  ⎿ a" in output
    assert "... +3 lines omitted" in output
    assert "    e" in output
    assert "b" not in output


def test_write_file_tool_call_uses_fmt_file_path():
    renderer, console = make_renderer()

    renderer.tool_call("write_file", {"path": "out.txt"})

    assert "• write_file(out.txt)" in console.export_text()
