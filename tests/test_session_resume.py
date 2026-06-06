from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

from reagent.protocol import SilentSink
from reagent.results import ReadResult, ShellResult
import reagent.session.session as session_module
from reagent.session import Session, load_session
from reagent.session.recorder import SessionEntry, SessionRecorder, to_provider_message


def test_load_session_replays_messages_usage_and_attaches_recorder(tmp_path):
    recorder_session = Session(
        sink=SilentSink(),
        recorder=SessionRecorder.create(
            root=tmp_path,
            cwd="/repo",
            model="model",
        ),
    )
    recorder_session.add_user("hello")
    recorder_session.add_tool_calls(
        SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                }
            ],
        )
    )
    recorder_session.add_tool_result("call-1", ShellResult("out"))
    recorder_session.record_usage(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details={"cached_tokens": 3},
            completion_tokens_details={"reasoning_tokens": 2},
        )
    )

    assert recorder_session._recorder is not None
    loaded = load_session(recorder_session._recorder.path, sink=SilentSink())

    assert loaded._recorder is not None
    assert loaded._recorder.path == recorder_session._recorder.path
    assert loaded.turns == 1
    assert loaded.llm_calls == 1
    assert loaded.prompt_tokens == 10
    assert loaded.completion_tokens == 5
    assert loaded.cached_tokens == 3
    assert loaded.reasoning_tokens == 2
    assert list(loaded.messages) == [to_provider_message(message) for message in recorder_session.messages]
    assert all("id" not in message and "parent_id" not in message for message in loaded.messages)


class RecordingSink:
    def __init__(self):
        self.calls: list = []

    def on_assistant(self, text: str) -> None:
        self.calls.append(("on_assistant", text))

    def on_think(self, text: str) -> None:
        self.calls.append(("on_think", text))

    def on_tool_call(self, tool_call_id: str, name: str, args: dict) -> None:
        self.calls.append(("on_tool_call", tool_call_id, name, args))

    def on_tool_result(self, tool_call_id: str, result) -> None:
        self.calls.append(("on_tool_result", tool_call_id, result.text))

    def on_user(self, text: str) -> None:
        self.calls.append(("on_user", text))

    def on_status(self, msg: str) -> None:
        self.calls.append(("on_status", msg))

    def on_prompt(self, text: str) -> None:
        self.calls.append(("on_prompt", text))

    def on_thinking_start(self) -> None:
        self.calls.append(("on_thinking_start",))

    def on_thinking_update(self, phase: str, tokens: int) -> None:
        self.calls.append(("on_thinking_update", phase, tokens))

    def on_thinking_stop(self) -> None:
        self.calls.append(("on_thinking_stop",))


def test_load_session_reprints_visible_history_to_sink(tmp_path):
    recorder_session = Session(
        sink=SilentSink(),
        recorder=SessionRecorder.create(root=tmp_path, cwd="/repo", model="model"),
    )
    recorder_session.add_user("hello")
    recorder_session.add_assistant("hi")
    recorder_session.add_tool_calls(
        SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": {"cmd": "pwd"}},
                }
            ],
        )
    )
    recorder_session.add_tool_result("call-1", ShellResult("out"))

    sink = RecordingSink()
    assert recorder_session._recorder is not None
    load_session(recorder_session._recorder.path, sink=sink)

    assert sink.calls == [
        ("on_user", "hello"),
        ("on_assistant", "hi"),
        ("on_tool_call", "call-1", "shell", {"cmd": "pwd"}),
        ("on_tool_result", "call-1", "out"),
    ]


def test_load_session_explicit_path_uses_embedded_session_id_when_file_is_renamed(tmp_path):
    recorder = SessionRecorder.create(root=tmp_path, cwd="/repo", model="model")
    recorder.record_message({"role": "user", "content": "hello"})
    renamed = recorder.path.with_name("renamed.jsonl")
    recorder.path.rename(renamed)

    loaded = load_session(renamed, sink=SilentSink())

    assert loaded.messages == ({"role": "user", "content": "hello"},)


def test_load_session_replays_think_via_on_think(tmp_path):
    recorder_session = Session(
        sink=SilentSink(),
        recorder=SessionRecorder.create(root=tmp_path, cwd="/repo", model="model"),
    )
    recorder_session.add_user("question")
    recorder_session.add_think("reasoning")
    recorder_session.add_assistant("answer")

    sink = RecordingSink()
    assert recorder_session._recorder is not None
    load_session(recorder_session._recorder.path, sink=sink)

    assert sink.calls == [
        ("on_user", "question"),
        ("on_think", "reasoning"),
        ("on_assistant", "answer"),
    ]


def test_load_session_reconstructs_read_result(tmp_path):
    recorder_session = Session(
        sink=SilentSink(),
        recorder=SessionRecorder.create(root=tmp_path, cwd="/repo", model="model"),
    )
    recorder_session.add_user("read")
    recorder_session.add_tool_result("call-r", ReadResult(path="/file.py", content="x = 1"))

    sink = RecordingSink()
    assert recorder_session._recorder is not None
    load_session(recorder_session._recorder.path, sink=sink)

    on_tool_result_call = next(c for c in sink.calls if c[0] == "on_tool_result")
    assert on_tool_result_call[1] == "call-r"
    # result.text should match the original ReadResult rendering (numbered lines)
    assert "1: x = 1" in on_tool_result_call[2]


def test_load_session_applies_compact_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_LIMIT", 1)
    recorder_session = Session(
        sink=SilentSink(),
        recorder=SessionRecorder.create(root=tmp_path, cwd="/repo", model="model"),
    )
    recorder_session.add_user("keep")
    recorder_session.add_user("old")
    recorder_session.add_assistant("old response")
    recorder_session.add_user("latest")

    recorder_session.compact(lambda messages: "summarized old context")

    assert recorder_session._recorder is not None
    loaded = load_session(recorder_session._recorder.path, sink=SilentSink())

    assert list(loaded.messages) == [
        {"role": "user", "content": "keep"},
        {"role": "user", "content": "[Context summary: summarized old context]"},
        {"role": "user", "content": "latest"},
    ]


def test_load_session_applies_chained_compact_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_LIMIT", 1)
    recorder_session = Session(
        sink=SilentSink(),
        recorder=SessionRecorder.create(root=tmp_path, cwd="/repo", model="model"),
    )
    recorder_session.add_user("keep")
    recorder_session.add_user("old one")
    recorder_session.add_assistant("old response")
    recorder_session.add_user("latest")
    recorder_session.compact(lambda messages: "summary one")

    recorder_session.add_user("newest")
    recorder_session.compact(lambda messages: "summary two")

    assert recorder_session._recorder is not None
    loaded = load_session(recorder_session._recorder.path, sink=SilentSink())

    assert list(loaded.messages) == [
        {"role": "user", "content": "keep"},
        {"role": "user", "content": "[Context summary: summary two]"},
        {"role": "user", "content": "newest"},
    ]


class FailingCompactRecorder:
    def __init__(self):
        self.path = Path("session.jsonl")
        self._seq = 0

    def record_message(self, message: Mapping[str, Any]) -> SessionEntry:
        self._seq += 1
        return cast(SessionEntry, {"seq": self._seq})

    def record_usage(self, **_: Any) -> None:
        pass

    def record_compact(self, *, start_seq: int, end_seq: int, replacement_message: dict[str, Any]) -> SessionEntry:
        raise OSError("disk full")


def test_compact_propagates_recorder_failures(monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_LIMIT", 1)
    session = Session(sink=SilentSink(), recorder=FailingCompactRecorder())
    session.add_user("keep")
    session.add_user("old")
    session.add_assistant("old response")
    session.add_user("latest")

    try:
        session.compact(lambda messages: "summary")
    except OSError as exc:
        assert str(exc) == "disk full"
    else:
        raise AssertionError("compact should propagate recorder persistence failures")
