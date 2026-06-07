import time

from rich.console import Console

from reagent.rendering import TERMINAL_THEME
from reagent.repl import (
    _Call,
    _PendingCalls,
    _SlashRoute,
    _enter_action,
    _exit_hint_expired,
    _fmt_hint,
    _fmt_status,
    _fmt_thinking,
    _fmt_usage,
    _make_app,
    _print_usage,
    _route_slash_result,
    _tool_call_bullet_style,
)
from reagent.slash_commands import SlashResult


def test_thinking_for_text_is_gray_and_not_indented():
    assert _fmt_status("• thinking for 1.0s", style_class="thinking-for") == [
        ("class:thinking-for", "• thinking for 1.0s")
    ]


def test_interrupt_text_is_yellow_and_not_indented():
    assert _fmt_status("■ Conversation interrupted", style_class="status") == [
        ("class:status", "■ Conversation interrupted")
    ]


def test_thinking_frame_is_purple_and_text_is_gray():
    assert _fmt_thinking("☰", elapsed=1.0, token_part="") == [
        ("class:thinking-frame", "☰"),
        ("class:thinking", " thinking  (1.0s)"),
    ]


def test_pending_tool_call_bullet_flashes_between_gray_and_white():
    assert _tool_call_bullet_style("pending", elapsed=0.0).dim
    style = _tool_call_bullet_style("pending", elapsed=0.4)
    assert style.color is not None and style.color.name == "white"


def test_completed_tool_call_bullet_is_green_or_red():
    success = _tool_call_bullet_style("success", elapsed=0.0)
    error = _tool_call_bullet_style("error", elapsed=0.0)
    assert success.color is not None and success.color.name == "green"
    assert error.color is not None and error.color.name == "red"


def test_pending_calls_render_tool_call():
    display = _PendingCalls()
    display.calls["call-1"] = _Call(name="shell", args={"command": "ls"}, started_at=time.monotonic())

    output = display.render(width=80)

    assert "• shell(ls)" in Console(force_terminal=False, width=80, theme=TERMINAL_THEME).render_str(output).plain


def test_enter_submits_when_idle():
    assert _enter_action("hello", is_running=False) == "submit"


def test_enter_keeps_draft_when_running():
    assert _enter_action("next prompt", is_running=True) == "hint"


def test_enter_backslash_adds_newline_even_when_running():
    assert _enter_action("draft\\", is_running=True) == "newline"


def test_enter_exit_command_only_exits_when_idle():
    assert _enter_action("/exit", is_running=False) == "submit"
    assert _enter_action("/exit", is_running=True) == "hint"


def test_route_slash_result_submits_normal_input():
    result = _route_slash_result("hello", SlashResult(outcome="not_slash"))

    assert result == _SlashRoute(action="submit", prompt="hello")


def test_route_slash_result_submits_expanded_prompt():
    result = _route_slash_result("/draft topic", SlashResult(outcome="submit_prompt", prompt="expanded prompt"))

    assert result == _SlashRoute(action="submit", prompt="expanded prompt")


def test_route_slash_result_exits():
    result = _route_slash_result("/exit", SlashResult(outcome="exit"))

    assert result == _SlashRoute(action="exit")


def test_route_slash_result_keeps_local_message_out_of_prompt_submission():
    result = _route_slash_result("/status", SlashResult(outcome="handled", message="status text"))

    assert result == _SlashRoute(action="handled", message="status text")


def test_route_slash_result_treats_unknown_as_local_message():
    result = _route_slash_result("/nope", SlashResult(outcome="unknown", message="Unknown command: /nope"))

    assert result == _SlashRoute(action="handled", message="Unknown command: /nope")


def test_empty_hint_still_renders_reserved_line():
    assert _fmt_hint("") == [("class:hint", "")]


def test_ctrl_c_hint_clears_after_exit_window():
    assert _exit_hint_expired(
        now=3.0,
        last_ctrl_c=1.0,
        hint="Press Ctrl+C again to exit",
        active_turn=False,
    )


def test_ctrl_c_hint_does_not_clear_while_turn_is_active():
    assert not _exit_hint_expired(
        now=3.0,
        last_ctrl_c=1.0,
        hint="Press Ctrl+C again to exit",
        active_turn=True,
    )


def test_repl_app_erases_managed_tail_on_exit():
    app = _make_app(layout=None, key_bindings=None)

    assert not app.full_screen
    assert app.erase_when_done


def test_usage_summary_omits_missing_optional_counts():
    assert _fmt_usage(total=1500, input_tokens=1000, output_tokens=500, cached_tokens=0, reasoning_tokens=0) == (
        "Usage: total=1,500 input=1,000 output=500"
    )


def test_usage_summary_includes_cached_and_reasoning_counts():
    assert (
        _fmt_usage(
            total=107505,
            input_tokens=92781,
            output_tokens=14724,
            cached_tokens=2585600,
            reasoning_tokens=4631,
        )
        == "Usage: total=107,505 input=92,781 (+ 2,585,600 cached) output=14,724 (reasoning 4,631)"
    )


def test_usage_output_is_separated_from_previous_message():
    console = Console(record=True, force_terminal=False)

    _print_usage(console, "Usage: total=1 input=1 output=0")

    assert console.export_text() == "\nUsage: total=1 input=1 output=0\n"


def test_usage_output_is_not_highlighted():
    console = Console(record=True, force_terminal=False)

    _print_usage(console, "Usage: total=1 input=1 output=0")

    usage_segments = [
        segment for segment in console._record_buffer if "Usage" in segment.text or "total" in segment.text
    ]
    assert usage_segments
    assert all(segment.style is None for segment in usage_segments)
