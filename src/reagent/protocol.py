from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from reagent.results import ToolResult


class Renderer(Protocol):
    def assistant(self, text: str) -> None: ...
    def think(self, text: str) -> None: ...
    def tool_call(self, name: str, args: dict) -> None: ...
    def tool_result(self, tool_call_id: str, result: "ToolResult") -> None: ...
    def status(self, msg: str) -> None: ...
    def prompt(self, text: str) -> None: ...
    def thinking_start(self) -> None: ...
    def thinking_update(self, phase: str, tokens: int) -> None: ...
    def thinking_stop(self) -> None: ...


@runtime_checkable
class OutputSink(Protocol):
    def on_assistant(self, text: str) -> None: ...
    def on_think(self, text: str) -> None: ...
    def on_tool_call(self, name: str, args: dict) -> None: ...
    def on_tool_result(self, tool_call_id: str, result: "ToolResult") -> None: ...
    def on_status(self, msg: str) -> None: ...
    def on_prompt(self, text: str) -> None: ...
    def on_thinking_start(self) -> None: ...
    def on_thinking_update(self, phase: str, tokens: int) -> None: ...
    def on_thinking_stop(self) -> None: ...


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

    def on_tool_result(self, tool_call_id: str, result: "ToolResult") -> None:
        self._renderer.tool_result(tool_call_id, result)

    def on_status(self, msg: str) -> None:
        self._renderer.status(msg)

    def on_prompt(self, text: str) -> None:
        self._renderer.prompt(text)

    def on_thinking_start(self) -> None:
        self._renderer.thinking_start()

    def on_thinking_update(self, phase: str, tokens: int) -> None:
        self._renderer.thinking_update(phase, tokens)

    def on_thinking_stop(self) -> None:
        self._renderer.thinking_stop()


class SilentSink:
    def on_assistant(self, text: str) -> None:
        pass

    def on_think(self, text: str) -> None:
        pass

    def on_tool_call(self, name: str, args: dict) -> None:
        pass

    def on_tool_result(self, tool_call_id: str, result: "ToolResult") -> None:
        pass

    def on_status(self, msg: str) -> None:
        pass

    def on_prompt(self, text: str) -> None:
        pass

    def on_thinking_start(self) -> None:
        pass

    def on_thinking_update(self, phase: str, tokens: int) -> None:
        pass

    def on_thinking_stop(self) -> None:
        pass
