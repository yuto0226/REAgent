# tests/test_protocol.py
from reagent.protocol import OutputSink, SilentSink, TerminalSink
from reagent.results import ShellResult


class RecordingRenderer:
    def __init__(self):
        self.calls = []

    def assistant(self, text: str) -> None:
        self.calls.append(("assistant", text))

    def think(self, text: str) -> None:
        self.calls.append(("think", text))

    def tool_call(self, name: str, args: dict, *, tool_call_id: str = "") -> None:
        self.calls.append(("tool_call", tool_call_id, name, args))

    def tool_result(self, tool_call_id: str, result) -> None:
        self.calls.append(("tool_result", tool_call_id, result))

    def user(self, text: str) -> None:
        self.calls.append(("user", text))

    def status(self, msg: str) -> None:
        self.calls.append(("status", msg))

    def prompt(self, text: str) -> None:
        self.calls.append(("prompt", text))

    def thinking_start(self) -> None:
        self.calls.append(("thinking_start",))

    def thinking_update(self, phase: str, tokens: int) -> None:
        self.calls.append(("thinking_update", phase, tokens))

    def thinking_stop(self) -> None:
        self.calls.append(("thinking_stop",))


def test_terminal_sink_implements_protocol():
    assert isinstance(TerminalSink(), OutputSink)


def test_silent_sink_implements_protocol():
    assert isinstance(SilentSink(), OutputSink)


def test_silent_sink_produces_no_output(capsys):
    sink = SilentSink()
    sink.on_assistant("hello")
    sink.on_think("thinking")
    sink.on_tool_call("id1", "bash", {"cmd": "ls"})
    sink.on_tool_result("id1", ShellResult("result"))
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
    TerminalSink().on_tool_call("id1", "bash", {"cmd": "ls"})
    assert "bash" in capsys.readouterr().out


def test_terminal_sink_on_tool_result(capsys):
    TerminalSink().on_tool_result("id1", ShellResult("some output"))
    assert "some output" in capsys.readouterr().out


def test_terminal_sink_on_status(capsys):
    TerminalSink().on_status("compacting...")
    assert "compacting..." in capsys.readouterr().out


def test_terminal_sink_delegates_to_renderer():
    renderer = RecordingRenderer()
    sink = TerminalSink(renderer=renderer)

    result = ShellResult("output")
    sink.on_assistant("hello")
    sink.on_think("thinking")
    sink.on_tool_call("call-1", "shell", {"command": "pwd"})
    sink.on_tool_result("call-1", result)
    sink.on_user("hi there")
    sink.on_status("status")

    assert renderer.calls == [
        ("assistant", "hello"),
        ("think", "thinking"),
        ("tool_call", "call-1", "shell", {"command": "pwd"}),
        ("tool_result", "call-1", result),
        ("user", "hi there"),
        ("status", "status"),
    ]
