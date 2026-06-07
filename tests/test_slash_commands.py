from reagent.slash_commands import SlashCommand, builtins, find, parse


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
