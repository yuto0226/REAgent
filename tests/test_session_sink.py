# tests/test_session_sink.py
from reagent.protocol import SilentSink
from reagent.results import ErrorResult, ShellResult
from reagent.session import Session


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
        self.calls.append(("on_tool_result", tool_call_id, result))

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


def test_session_defaults_to_terminal_sink():
    from reagent.protocol import TerminalSink

    s = Session()
    assert isinstance(s._sink, TerminalSink)


def test_session_accepts_custom_sink():
    sink = SilentSink()
    s = Session(sink=sink)
    assert s._sink is sink


def test_add_assistant_calls_sink(capsys):
    s = Session(sink=SilentSink())
    s.add_assistant("hello")
    assert capsys.readouterr().out == ""


def test_add_think_calls_sink(capsys):
    s = Session(sink=SilentSink())
    s.add_think("thinking...")
    assert capsys.readouterr().out == ""


def test_add_tool_result_calls_sink(capsys):
    s = Session(sink=SilentSink())
    s.add_tool_result("id1", ShellResult("output"))
    assert capsys.readouterr().out == ""


def test_emit_tool_call_calls_sink(capsys):
    s = Session(sink=SilentSink())
    s.emit_tool_call("id1", "bash", {"cmd": "ls"})
    assert capsys.readouterr().out == ""


def test_emit_status_calls_sink(capsys):
    s = Session(sink=SilentSink())
    s.emit_status("compacting...")
    assert capsys.readouterr().out == ""


def test_add_assistant_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).add_assistant("hello")
    assert ("on_assistant", "hello") in sink.calls


def test_add_think_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).add_think("reasoning")
    assert ("on_think", "reasoning") in sink.calls


def test_add_tool_result_dispatches_to_sink():
    sink = RecordingSink()
    result = ShellResult("output")
    Session(sink=sink).add_tool_result("call1", result)
    assert ("on_tool_result", "call1", result) in sink.calls


def test_add_tool_result_stores_text_in_history():
    s = Session(sink=SilentSink())
    s.add_tool_result("call1", ShellResult("hello"))
    assert s.messages[-1]["content"] == "hello"


def test_add_tool_result_error_stores_message_in_history():
    s = Session(sink=SilentSink())
    s.add_tool_result("call1", ErrorResult("Error: oops"))
    assert s.messages[-1]["content"] == "Error: oops"


def test_emit_tool_call_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_tool_call("id1", "bash", {"cmd": "ls"})
    assert ("on_tool_call", "id1", "bash", {"cmd": "ls"}) in sink.calls


def test_emit_status_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_status("compacting...")
    assert ("on_status", "compacting...") in sink.calls


def test_emit_thinking_start_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_thinking_start()
    assert ("on_thinking_start",) in sink.calls


def test_emit_thinking_stop_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_thinking_stop()
    assert ("on_thinking_stop",) in sink.calls


def test_emit_thinking_update_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_thinking_update("up", 1234)
    assert ("on_thinking_update", "up", 1234) in sink.calls


def test_emit_user_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_user("hello from user")
    assert ("on_user", "hello from user") in sink.calls
