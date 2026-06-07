from reagent.protocol import SilentSink
from reagent.session import Session
from reagent.slash_commands import SlashCommand, builtins, dispatch, find, parse


def test_parse_ignores_normal_input():
    assert parse("hello") is None


def test_parse_extracts_command_without_args():
    parsed = parse("/status")

    assert parsed is not None
    assert parsed.name == "status"
    assert parsed.args == ""


def test_parse_preserves_raw_args():
    parsed = parse("  /compact   now please  ")

    assert parsed is not None
    assert parsed.name == "compact"
    assert parsed.args == "now please"


def test_find_resolves_exit_alias():
    command = find("quit")

    assert command is not None
    assert command.name == "exit"


def test_help_is_not_registered():
    assert find("help") is None


def test_builtin_metadata_reserves_extension_fields():
    command = find("status")

    assert command == SlashCommand(
        name="status",
        aliases=(),
        description="Show local session stats",
        action="local",
        origin="builtin",
        accepts_args=False,
        arg_hint="",
        available="idle",
        visible=True,
        template=None,
    )


def test_builtins_store_aliases_on_canonical_commands():
    names = [command.name for command in builtins()]

    assert names == ["exit", "status", "compact"]
    assert all(command.name != "quit" for command in builtins())


def test_dispatch_returns_not_slash_for_normal_input():
    result = dispatch("hello", Session(sink=SilentSink()))

    assert result.outcome == "not_slash"
    assert result.message == ""
    assert result.prompt == ""


def test_dispatch_returns_unknown_for_unregistered_slash():
    result = dispatch("/missing", Session(sink=SilentSink()))

    assert result.outcome == "unknown"
    assert result.message == "Unknown command: /missing"


def test_dispatch_rejects_args_for_status():
    result = dispatch("/status now", Session(sink=SilentSink()))

    assert result.outcome == "handled"
    assert result.message == "Command /status does not accept arguments"


def test_dispatch_exits_for_exit_alias():
    result = dispatch("/quit", Session(sink=SilentSink()))

    assert result.outcome == "exit"


def test_dispatch_status_reports_session_counters_without_history_change():
    session = Session(sink=SilentSink())
    session.add_user("hello")
    session.llm_calls = 2
    session.tool_calls = 3
    session.prompt_tokens = 1000
    session.completion_tokens = 250
    session.cached_tokens = 75
    session.reasoning_tokens = 30
    before = session.messages

    result = dispatch("/status", session)

    assert result.outcome == "handled"
    assert result.message == (
        "Turns: 1\nLLM calls: 2\nTool calls: 3\nTokens: total=1,250 input=1,000 output=250 cached=75 reasoning=30"
    )
    assert session.messages == before
