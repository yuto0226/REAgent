import time

from rich.console import Console

from reagent.rendering import TERMINAL_THEME
from reagent.repl import (
    _Call,
    _PendingCalls,
    _enter_action,
    _exit_hint_expired,
    _fmt_hint,
    _fmt_status,
    _fmt_thinking,
    _tool_call_bullet_style,
)


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
    assert _enter_action("/exit", is_running=False) == "exit"
    assert _enter_action("/exit", is_running=True) == "hint"


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
