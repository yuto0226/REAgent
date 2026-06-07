from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from reagent.session.session import Message, Session


SlashOrigin = Literal["builtin"]
SlashOutcome = Literal["not_slash", "unknown", "exit", "handled", "submit_prompt"]
SlashRender = Literal["default", "notice", "error", "panel"]


@dataclass(frozen=True)
class SlashCommand:
    """A normalized command record used by lookup and dispatch."""

    name: str
    aliases: tuple[str, ...]
    description: str
    origin: SlashOrigin
    accepts_args: bool = False
    visible: bool = True


@dataclass(frozen=True)
class ParsedSlash:
    """A syntactic slash command before registry resolution."""

    name: str
    args: str


@dataclass(frozen=True)
class SlashResult:
    """The REPL-facing result of slash dispatch."""

    outcome: SlashOutcome
    message: str = ""
    prompt: str = ""
    render: SlashRender = "default"
    command_name: str = ""


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


BUILTINS: tuple[SlashCommand, ...] = (
    SlashCommand(
        name="exit",
        aliases=("quit",),
        description="Exit the REPL",
        origin="builtin",
    ),
    SlashCommand(
        name="status",
        aliases=(),
        description="Show local session stats",
        origin="builtin",
    ),
    SlashCommand(
        name="compact",
        aliases=(),
        description="Compact the current session context",
        origin="builtin",
    ),
)


def find(name: str) -> SlashCommand | None:
    normalized = name.lower()
    for command in BUILTINS:
        if normalized == command.name or normalized in command.aliases:
            return command
    return None


def completions(prefix: str) -> tuple[SlashCommand, ...]:
    normalized = prefix.removeprefix("/").lower()
    return tuple(
        c
        for c in BUILTINS
        if c.visible and (c.name.startswith(normalized) or any(a.startswith(normalized) for a in c.aliases))
    )


def dispatch(
    text: str,
    session: Session,
    compact_fn: Callable[[list[Message]], str] | None = None,
) -> SlashResult:
    parsed = parse(text)
    if parsed is None:
        return SlashResult(outcome="not_slash")

    command = find(parsed.name)
    if command is None:
        return SlashResult(outcome="unknown", message=f"Unknown command: /{parsed.name}", render="error")

    if parsed.args and not command.accepts_args:
        return SlashResult(outcome="handled", message=f"Command /{command.name} does not accept arguments")

    if command.name == "exit":
        return SlashResult(outcome="exit")

    if command.name == "status":
        return SlashResult(outcome="handled", render="panel", command_name="status")

    if command.name == "compact":
        if compact_fn is None:
            return SlashResult(
                outcome="handled", message="Compact is unavailable", render="error", command_name="compact"
            )

        try:
            changed = session.compact(compact_fn, force=True)
        except OSError as exc:
            return SlashResult(
                outcome="handled", message=f"Compact failed: {exc}", render="error", command_name="compact"
            )

        if changed:
            return SlashResult(outcome="handled", message="Context compacted", render="notice", command_name="compact")

        return SlashResult(outcome="handled", message="Nothing to compact yet", command_name="compact")

    return SlashResult(outcome="handled", message=f"Command /{command.name} is not available yet")
