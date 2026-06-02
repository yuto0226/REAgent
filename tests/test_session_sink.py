# tests/test_session_sink.py
from reagent.protocol import SilentSink
from reagent.session import Session


class RecordingSink:
    def __init__(self):
        self.calls: list = []

    def on_assistant(self, text: str) -> None:
        self.calls.append(("on_assistant", text))

    def on_think(self, text: str) -> None:
        self.calls.append(("on_think", text))

    def on_tool_call(self, name: str, args: dict) -> None:
        self.calls.append(("on_tool_call", name, args))

    def on_tool_result(self, tool_call_id: str, content: str, tool_name: str | None = None) -> None:
        self.calls.append(("on_tool_result", tool_call_id, content, tool_name))

    def on_status(self, msg: str) -> None:
        self.calls.append(("on_status", msg))

    def on_prompt(self, text: str) -> None:
        self.calls.append(("on_prompt", text))


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
    s.add_tool_result("id1", "output")
    assert capsys.readouterr().out == ""


def test_emit_tool_call_calls_sink(capsys):
    s = Session(sink=SilentSink())
    s.emit_tool_call("bash", {"cmd": "ls"})
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
    Session(sink=sink).add_tool_result("call1", "output")
    assert ("on_tool_result", "call1", "output", None) in sink.calls


def test_add_tool_result_dispatches_tool_name_to_sink():
    sink = RecordingSink()
    Session(sink=sink).add_tool_result("call1", "output", tool_name="shell")
    assert ("on_tool_result", "call1", "output", "shell") in sink.calls


def test_emit_tool_call_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_tool_call("bash", {"cmd": "ls"})
    assert ("on_tool_call", "bash", {"cmd": "ls"}) in sink.calls


def test_emit_status_dispatches_to_sink():
    sink = RecordingSink()
    Session(sink=sink).emit_status("compacting...")
    assert ("on_status", "compacting...") in sink.calls
