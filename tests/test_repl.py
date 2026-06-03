from reagent.repl import (
    _OutputBuffer,
    _format_status,
    _format_thinking,
    _status_needs_input_spacer,
    _status_needs_spacer,
    _tool_call_bullet_style,
)


def test_regular_status_is_separated_from_message_by_blank_line():
    assert _status_needs_spacer(is_thinking=False, status_text="compacting...")


def test_regular_status_is_separated_from_input_by_blank_line():
    assert _status_needs_input_spacer(is_thinking=False, status_text="compacting...")


def test_thinking_status_is_separated_from_message_by_blank_line():
    assert _status_needs_spacer(is_thinking=True, status_text="")


def test_thinking_status_is_separated_from_input_by_blank_line():
    assert _status_needs_input_spacer(is_thinking=True, status_text="")


def test_interrupt_status_is_separated_from_message_by_blank_line():
    assert _status_needs_spacer(
        is_thinking=False,
        status_text="■ Conversation interrupted",
    )


def test_thinking_for_text_is_gray_and_not_indented():
    assert _format_status("• thinking for 1.0s", style_class="thinking-for") == [
        ("class:thinking-for", "• thinking for 1.0s")
    ]


def test_interrupt_text_is_yellow_and_not_indented():
    assert _format_status("■ Conversation interrupted", style_class="status") == [
        ("class:status", "■ Conversation interrupted")
    ]


def test_thinking_frame_is_purple_and_text_is_gray():
    assert _format_thinking("☰", elapsed=1.0, token_part="") == [
        ("class:thinking-frame", "☰"),
        ("class:thinking", " thinking  (1.0s)"),
    ]


def test_output_buffer_replaceable_block_updates_in_place():
    output = _OutputBuffer()
    state = ["pending"]
    output.write("before\n")
    output.add_replaceable_block("call-1", lambda: f"• shell(pwd) [{state[0]}]\n")
    output.write("after\n")

    state[0] = "success"

    assert output.getvalue() == "before\n• shell(pwd) [success]\nafter\n"


def test_output_buffer_replaceable_block_counts_rendered_lines_in_tail():
    output = _OutputBuffer()
    output.write("before\n")
    output.add_replaceable_block("call-1", lambda: "\n• shell(echo hi)\n        continued)\n")
    output.write("after\n")

    assert output.get_tail(3) == "• shell(echo hi)\n        continued)\nafter"


def test_pending_tool_call_bullet_flashes_between_gray_and_white():
    assert _tool_call_bullet_style("pending", elapsed=0.0).dim
    style = _tool_call_bullet_style("pending", elapsed=0.4)
    assert style.color is not None and style.color.name == "white"


def test_completed_tool_call_bullet_is_green_or_red():
    success = _tool_call_bullet_style("success", elapsed=0.0)
    error = _tool_call_bullet_style("error", elapsed=0.0)
    assert success.color is not None and success.color.name == "green"
    assert error.color is not None and error.color.name == "red"
