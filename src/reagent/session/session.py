from __future__ import annotations

import bisect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from reagent.protocol import OutputSink, TerminalSink
from reagent.results import DiffResult, ErrorResult, ReadResult, ShellResult, ToolResult
from reagent.session.recorder import SessionRecorder, jsonable, read_entries, to_provider_message

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
    def __init__(self, sink: OutputSink | None = None, recorder: SessionRecorder | None = None) -> None:
        self._sink: OutputSink = sink if sink is not None else TerminalSink()
        self._recorder = recorder
        # Each entry is (message, jsonl_seq). seq is None when the message has no
        # independent JSONL record (e.g. a compact replacement injected during load).
        self._history: list[tuple[Message, int | None]] = []
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
        return tuple(m for m, _ in self._history)

    def _log(self, message: Message) -> int | None:
        if self._recorder is None:
            return None
        return self._recorder.record_message(message)["seq"]

    def _append(self, message: Message) -> None:
        self._history.append((message, self._log(message)))

    def add_user(self, content: str) -> None:
        self._append(UserMessage(role="user", content=content))
        self.turns += 1

    def add_assistant(self, content: str) -> None:
        message = AssistantMessage(role="assistant", content=content)
        self._append(message)
        if content:
            self._sink.on_assistant(content)

    def add_think(self, content: str) -> None:
        message = AssistantMessage(role="assistant", content=content)
        seq: int | None = None
        if self._recorder is not None:
            seq = self._recorder.record_message({**message, "is_think": True})["seq"]
        self._history.append((message, seq))
        if content:
            self._sink.on_think(content)

    def add_tool_calls(self, raw_message: Any) -> None:
        message = AssistantToolCallMessage(
            role="assistant",
            content=raw_message.content,
            tool_calls=jsonable(raw_message.tool_calls or []),
        )
        self._append(message)
        self.tool_calls += len(raw_message.tool_calls or [])

    def add_tool_result(self, tool_call_id: str, result: ToolResult) -> None:
        message = ToolMessage(role="tool", tool_call_id=tool_call_id, content=result.text)
        seq: int | None = None
        if self._recorder is not None:
            extra = {"result_type": type(result).__name__, "result_data": jsonable(vars(result))}
            seq = self._recorder.record_message({**message, **extra})["seq"]

        self._history.append((message, seq))
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

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached_tokens = _usage_detail(usage, "prompt_tokens_details", "cached_tokens")
        reasoning_tokens = _usage_detail(usage, "completion_tokens_details", "reasoning_tokens")

        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_tokens += cached_tokens
        self.reasoning_tokens += reasoning_tokens

        if self._recorder is not None:
            self._recorder.record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            )

    def _estimate_tokens(self) -> int:
        # ~4 chars per token; rough estimate, not precise.
        return len(json.dumps([m for m, _ in self._history], default=str)) // 4

    def compact(self, completion_fn: Callable[[list[Message]], str]) -> None:
        tokens = self._estimate_tokens()
        if tokens <= TOKEN_LIMIT:
            return

        self.emit_status(f"[!] compacting ... ({tokens})\n")
        turn_starts = [i for i, (m, _) in enumerate(self._history) if m["role"] == "user"]

        if len(turn_starts) < 2:
            self.truncate()
            return

        compact_end = turn_starts[-1]
        # _history[0] is the seed user message; preserve it and the latest turn.
        to_compact = [m for m, _ in self._history[1:compact_end]]
        compacted_seqs = [seq for _, seq in self._history[1:compact_end] if seq is not None]

        try:
            summary = completion_fn(to_compact)
        except Exception:
            self.truncate()
            return

        replacement = UserMessage(role="user", content=f"[Context summary: {summary}]")
        entry_seq: int | None = None
        if self._recorder is not None and compacted_seqs:
            entry = self._recorder.record_compact(
                start_seq=min(compacted_seqs),
                end_seq=max(compacted_seqs),
                replacement_message=dict(replacement),
            )
            # Store the compact entry's own seq so a subsequent compact can
            # identify and replace this summary by seq range during replay.
            entry_seq = entry["seq"]
        self._history[1:compact_end] = [(replacement, entry_seq)]

    def truncate(self) -> None:
        # Drop oldest turns (from index 1) until history fits within TOKEN_LIMIT.
        # Each turn spans from a UserMessage to just before the next UserMessage.
        while self._estimate_tokens() > TOKEN_LIMIT and len(self._history) > 1:
            end = 2
            while end < len(self._history) and self._history[end][0]["role"] != "user":
                end += 1
            del self._history[1:end]


def _apply_compact(
    replayed: list[tuple[int, dict[str, Any]]],
    entry_seq: int,
    data: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    start_seq = data.get("start_seq")
    end_seq = data.get("end_seq")
    replacement = data.get("replacement_message")

    if not (isinstance(start_seq, int) and isinstance(end_seq, int) and isinstance(replacement, dict)):
        return replayed

    # Sort before filtering: chained compacts can leave replayed out-of-order
    # (a prior compact's entry_seq may exceed later message seqs), which would
    # make bisect_left produce the wrong insertion index.
    filtered = sorted(
        [(s, m) for s, m in replayed if not start_seq <= s <= end_seq],
        key=lambda t: t[0],
    )
    insert_at = bisect.bisect_left([s for s, _ in filtered], start_seq)
    # Use entry_seq (the compact entry's own seq) so a subsequent compact can
    # identify and replace this summary by seq range during replay.
    filtered.insert(insert_at, (entry_seq, cast(dict[str, Any], replacement)))
    return filtered


def _reconstruct_result(result_type: str, content: str, result_data: Any) -> ToolResult:
    if isinstance(result_data, dict):
        try:
            if result_type == "ShellResult":
                return ShellResult(**result_data)
            if result_type == "ReadResult":
                return ReadResult(**result_data)
            if result_type == "DiffResult":
                return DiffResult(**result_data)
            if result_type == "ErrorResult":
                return ErrorResult(**result_data)
        except TypeError:
            pass
    return ShellResult(content)


def _parse_tool_call(tool_call: Any) -> tuple[str, str, dict[str, Any]] | None:
    """Extract (id, name, args) from a provider tool_call dict, or None if malformed."""
    if not isinstance(tool_call, dict):
        return None

    call_id = tool_call.get("id")
    function = tool_call.get("function")
    if not isinstance(call_id, str) or not isinstance(function, dict):
        return None

    name = function.get("name")
    raw_args = function.get("arguments")
    if not isinstance(name, str):
        return None

    # arguments may arrive as a JSON string or already-parsed dict
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    return call_id, name, args


def replay_sink(sink: OutputSink, data: dict[str, Any]) -> None:
    """Replay one stored JSONL message to the UI sink."""
    role = data.get("role")

    if role == "user":
        content = data.get("content")
        if isinstance(content, str):
            sink.on_user(content)
        return

    if role == "assistant":
        content = data.get("content")
        if isinstance(content, str) and content:
            if data.get("is_think"):
                sink.on_think(content)
            else:
                sink.on_assistant(content)

        tool_calls = data.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                parsed = _parse_tool_call(tool_call)
                if parsed is not None:
                    sink.on_tool_call(*parsed)
        return

    if role == "tool":
        call_id = data.get("tool_call_id")
        content = data.get("content", "")
        if isinstance(call_id, str) and isinstance(content, str):
            result = _reconstruct_result(
                data.get("result_type", "ShellResult"),
                content,
                data.get("result_data"),
            )
            sink.on_tool_result(call_id, result)


def load_session(path: str | Path, sink: OutputSink | None = None) -> Session:
    resolved = Path(path).expanduser()
    entries, _ = read_entries(resolved)

    # Prefer session_id from the meta entry; fall back to first entry's session_id
    # so a renamed file (no matching stem) doesn't silently drop all messages.
    session_id = next(
        (e["session_id"] for e in entries if e["type"] == "meta"),
        next((e["session_id"] for e in entries), resolved.stem),
    )
    entries = [entry for entry in entries if entry["session_id"] == session_id]
    session = Session(sink=sink, recorder=SessionRecorder.resume(path=resolved, session_id=session_id, entries=entries))
    replayed_messages: list[tuple[int, dict[str, Any]]] = []

    for entry in entries:
        event_type = entry["type"]
        data = entry["data"]

        if event_type == "message":
            # Keep raw data so replay_sink can access is_think / result_type / result_data.
            replayed_messages.append((entry["seq"], data))

        elif event_type == "compact":
            replayed_messages = _apply_compact(replayed_messages, entry["seq"], data)

        elif event_type == "usage":
            session.llm_calls += 1
            session.prompt_tokens += int(data.get("prompt_tokens", 0) or 0)
            session.completion_tokens += int(data.get("completion_tokens", 0) or 0)
            session.cached_tokens += int(data.get("cached_tokens", 0) or 0)
            session.reasoning_tokens += int(data.get("reasoning_tokens", 0) or 0)

    for seq, raw_data in replayed_messages:
        # to_provider_message strips internal fields (is_think, result_type, etc.)
        # so the LLM never sees them.
        provider_msg = cast(Message, to_provider_message(raw_data))
        session._history.append((provider_msg, seq))
        role = raw_data.get("role")

        if role == "user":
            session.turns += 1

        tool_calls = raw_data.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            session.tool_calls += len(tool_calls)

        replay_sink(session._sink, raw_data)

    return session
