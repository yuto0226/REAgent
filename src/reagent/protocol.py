from __future__ import annotations

from typing import Protocol, runtime_checkable


class Renderer(Protocol):
    def assistant(self, text: str) -> None: ...
    def think(self, text: str) -> None: ...
    def tool_call(self, name: str, args: dict) -> None: ...
    def tool_result(self, tool_call_id: str, content: str, *, tool_name: str | None = None) -> None: ...
    def status(self, msg: str) -> None: ...
    def prompt(self, text: str) -> None: ...


@runtime_checkable
class OutputSink(Protocol):
    def on_assistant(self, text: str) -> None: ...
    def on_think(self, text: str) -> None: ...
    def on_tool_call(self, name: str, args: dict) -> None: ...
    def on_tool_result(self, tool_call_id: str, content: str, tool_name: str | None = None) -> None: ...
    def on_status(self, msg: str) -> None: ...
    def on_prompt(self, text: str) -> None: ...


class TerminalSink:
    def __init__(self, renderer: Renderer | None = None) -> None:
        if renderer is None:
            from reagent.rendering import RichRenderer

            renderer = RichRenderer()

        self._renderer = renderer

    def on_assistant(self, text: str) -> None:
        self._renderer.assistant(text)

    def on_think(self, text: str) -> None:
        self._renderer.think(text)

    def on_tool_call(self, name: str, args: dict) -> None:
        self._renderer.tool_call(name, args)

    def on_tool_result(self, tool_call_id: str, content: str, tool_name: str | None = None) -> None:
        self._renderer.tool_result(tool_call_id, content, tool_name=tool_name)

    def on_status(self, msg: str) -> None:
        self._renderer.status(msg)

    def on_prompt(self, text: str) -> None:
        self._renderer.prompt(text)


class SilentSink:
    def on_assistant(self, text: str) -> None:
        pass

    def on_think(self, text: str) -> None:
        pass

    def on_tool_call(self, name: str, args: dict) -> None:
        pass

    def on_tool_result(self, tool_call_id: str, content: str, tool_name: str | None = None) -> None:
        pass

    def on_status(self, msg: str) -> None:
        pass

    def on_prompt(self, text: str) -> None:
        pass
