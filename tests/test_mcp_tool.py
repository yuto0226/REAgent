from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from reagent.mcp.tool import MCPTool, build_mcp_tools
from reagent.results import ErrorResult, MCPResult


class _StubManager:
    def __init__(self, result: Any = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, args: dict) -> Any:
        self.calls.append((name, args))
        if self._raises is not None:
            raise self._raises
        return self._result

    def list_tools(self) -> list[tuple[str, str, dict]]:
        return [("ida__decompile", "Decompile", {"type": "object", "properties": {}})]


@pytest.fixture
def bg_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


def test_mcptool_exposes_tool_schema(bg_loop):
    tool = MCPTool(
        "ida__decompile",
        "Decompile a function",
        {"type": "object", "properties": {"address": {"type": "string"}}},
        _StubManager(),
        bg_loop,
    )
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ida__decompile"
    assert schema["function"]["description"] == "Decompile a function"
    assert schema["function"]["parameters"]["properties"]["address"]["type"] == "string"


def test_mcptool_run_returns_manager_result(bg_loop):
    manager: Any = _StubManager(result=MCPResult(content='{"ok": true}'))
    tool = MCPTool("ida__decompile", "d", {}, manager, bg_loop)
    out = tool.run({"address": "0x401000"})
    assert isinstance(out, MCPResult)
    assert out.text == '{"ok": true}'
    assert manager.calls == [("ida__decompile", {"address": "0x401000"})]


def test_mcptool_run_wraps_bridge_errors(bg_loop):
    manager: Any = _StubManager(raises=RuntimeError("boom"))
    out = MCPTool("ida__x", "d", {}, manager, bg_loop).run({})
    assert isinstance(out, ErrorResult)
    assert "boom" in out.text


def test_build_mcp_tools_from_manager(bg_loop):
    manager: Any = _StubManager()
    tools = build_mcp_tools(manager, bg_loop)
    assert [t.name for t in tools] == ["ida__decompile"]
    assert isinstance(tools[0], MCPTool)
