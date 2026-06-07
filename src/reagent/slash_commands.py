from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SlashAction = Literal["local", "prompt"]
SlashOrigin = Literal["builtin"]
SlashAvailability = Literal["idle", "running", "any"]


@dataclass(frozen=True)
class SlashCommand:
    """A normalized command record used by lookup and dispatch."""

    name: str
    aliases: tuple[str, ...]
    description: str
    action: SlashAction
    origin: SlashOrigin
    accepts_args: bool = False
    arg_hint: str = ""
    available: SlashAvailability = "idle"
    visible: bool = True
    template: str | None = None


@dataclass(frozen=True)
class ParsedSlash:
    """A syntactic slash command before registry resolution."""

    name: str
    args: str


def parse(text: str) -> ParsedSlash | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    body = stripped[1:]
    if not body:
        return ParsedSlash(name="", args="")

    parts = body.split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) == 2 else ""
    return ParsedSlash(name=name, args=args)


def builtins() -> tuple[SlashCommand, ...]:
    return (
        SlashCommand(
            name="exit",
            aliases=("quit",),
            description="Exit the REPL",
            action="local",
            origin="builtin",
        ),
        SlashCommand(
            name="status",
            aliases=(),
            description="Show local session stats",
            action="local",
            origin="builtin",
        ),
        SlashCommand(
            name="compact",
            aliases=(),
            description="Compact the current session context",
            action="local",
            origin="builtin",
        ),
    )


def find(name: str) -> SlashCommand | None:
    normalized = name.lower()
    for command in builtins():
        if normalized == command.name or normalized in command.aliases:
            return command
    return None
