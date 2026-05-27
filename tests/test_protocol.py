# tests/test_protocol.py
from reagent.protocol import OutputSink, SilentSink, TerminalSink


class RecordingRenderer:
    def __init__(self):
        self.calls = []

    def assistant(self, text: str) -> None:
        self.calls.append(("assistant", text))

    def think(self, text: str) -> None:
        self.calls.append(("think", text))

    def tool_call(self, name: str, args: dict) -> None:
        self.calls.append(("tool_call", name, args))

    def tool_result(self, tool_call_id: str, content: str, *, tool_name: str | None = None) -> None:
        self.calls.append(("tool_result", tool_call_id, content, tool_name))

    def status(self, msg: str) -> None:
        self.calls.append(("status", msg))


def test_terminal_sink_implements_protocol():
    assert isinstance(TerminalSink(), OutputSink)


def test_silent_sink_implements_protocol():
    assert isinstance(SilentSink(), OutputSink)


def test_silent_sink_produces_no_output(capsys):
    sink = SilentSink()
    sink.on_assistant("hello")
    sink.on_think("thinking")
    sink.on_tool_call("bash", {"cmd": "ls"})
    sink.on_tool_result("id1", "result")
    sink.on_status("status msg")
    assert capsys.readouterr().out == ""


def test_terminal_sink_on_assistant(capsys):
    TerminalSink().on_assistant("hello")
    assert "hello" in capsys.readouterr().out


def test_terminal_sink_on_assistant_empty_no_output(capsys):
    TerminalSink().on_assistant("")
    assert capsys.readouterr().out == ""


def test_terminal_sink_on_think(capsys):
    TerminalSink().on_think("reasoning")
    assert "reasoning" in capsys.readouterr().out


def test_terminal_sink_on_tool_call(capsys):
    TerminalSink().on_tool_call("bash", {"cmd": "ls"})
    assert "bash" in capsys.readouterr().out


def test_terminal_sink_on_tool_result(capsys):
    TerminalSink().on_tool_result("id1", "some output")
    assert "some output" in capsys.readouterr().out


def test_terminal_sink_on_status(capsys):
    TerminalSink().on_status("compacting...")
    assert "compacting..." in capsys.readouterr().out


def test_terminal_sink_delegates_to_renderer():
    renderer = RecordingRenderer()
    sink = TerminalSink(renderer=renderer)

    sink.on_assistant("hello")
    sink.on_think("thinking")
    sink.on_tool_call("shell", {"command": "pwd"})
    sink.on_tool_result("call-1", "output")
    sink.on_status("status")

    assert renderer.calls == [
        ("assistant", "hello"),
        ("think", "thinking"),
        ("tool_call", "shell", {"command": "pwd"}),
        ("tool_result", "call-1", "output", None),
        ("status", "status"),
    ]
