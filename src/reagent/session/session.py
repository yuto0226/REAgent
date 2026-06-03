from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from reagent.protocol import OutputSink, TerminalSink
from reagent.results import ToolResult

TOKEN_LIMIT = 60_000


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None


class AssistantToolCallMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


Message = UserMessage | AssistantMessage | AssistantToolCallMessage | ToolMessage


def _usage_detail(usage: Any, details_name: str, token_name: str) -> int:
    details = getattr(usage, details_name, None)
    if details is None:
        return 0
    if isinstance(details, dict):
        return int(details.get(token_name, 0) or 0)
    return int(getattr(details, token_name, 0) or 0)


class Session:
    def __init__(self, sink: OutputSink | None = None) -> None:
        self._sink: OutputSink = sink if sink is not None else TerminalSink()
        self._history: list[Message] = []
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.tool_calls = 0
        self.turns = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def add_user(self, content: str) -> None:
        self._history.append(UserMessage(role="user", content=content))
        self.turns += 1

    def add_assistant(self, content: str) -> None:
        self._history.append(AssistantMessage(role="assistant", content=content))
        if content:
            self._sink.on_assistant(content)

    def add_think(self, content: str) -> None:
        self._history.append(AssistantMessage(role="assistant", content=content))
        if content:
            self._sink.on_think(content)

    def add_tool_calls(self, raw_message: Any) -> None:
        self._history.append(
            AssistantToolCallMessage(
                role="assistant",
                content=raw_message.content,
                tool_calls=raw_message.tool_calls,
            )
        )
        self.tool_calls += len(raw_message.tool_calls or [])

    def add_tool_result(self, tool_call_id: str, result: ToolResult) -> None:
        self._history.append(ToolMessage(role="tool", tool_call_id=tool_call_id, content=result.text))
        self._sink.on_tool_result(tool_call_id, result)

    def emit_tool_call(self, tool_call_id: str, name: str, args: dict) -> None:
        self._sink.on_tool_call(tool_call_id, name, args)

    def emit_user(self, text: str) -> None:
        self._sink.on_user(text)

    def emit_status(self, msg: str) -> None:
        self._sink.on_status(msg)

    def emit_prompt(self, text: str) -> None:
        self._sink.on_prompt(text)

    def emit_thinking_start(self) -> None:
        self._sink.on_thinking_start()

    def emit_thinking_update(self, phase: str, tokens: int) -> None:
        self._sink.on_thinking_update(phase, tokens)

    def emit_thinking_stop(self) -> None:
        self._sink.on_thinking_stop()

    def record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self.llm_calls += 1
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.cached_tokens += _usage_detail(usage, "prompt_tokens_details", "cached_tokens")
        self.reasoning_tokens += _usage_detail(usage, "completion_tokens_details", "reasoning_tokens")

    def _estimate_tokens(self) -> int:
        return len(json.dumps(list(self._history), default=str)) // 4

    def compact(self, completion_fn: Callable[[list[Message]], str]) -> None:
        tokens = self._estimate_tokens()
        if tokens <= TOKEN_LIMIT:
            return

        self.emit_status(f"[!] compacting ... ({tokens})\n")
        turn_starts = [i for i, m in enumerate(self._history) if m["role"] == "user"]
        if len(turn_starts) < 2:
            self.truncate()
            return

        compact_end = turn_starts[-1]
        to_compact = list(self._history[1:compact_end])
        try:
            summary = completion_fn(to_compact)
            self._history[1:compact_end] = [UserMessage(role="user", content=f"[Context summary: {summary}]")]
        except Exception:
            self.truncate()

    def truncate(self) -> None:
        # Drop oldest turns (from index 1) until history fits within TOKEN_LIMIT.
        # Each turn spans from a UserMessage to just before the next UserMessage.
        while self._estimate_tokens() > TOKEN_LIMIT and len(self._history) > 1:
            end = 2
            while end < len(self._history) and self._history[end]["role"] != "user":
                end += 1
            del self._history[1:end]
