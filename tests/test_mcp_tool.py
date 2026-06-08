from __future__ import annotations

from typing import Any

from reagent.mcp.tool import MCPTool, build_mcp_tools
from reagent.results import MCPResult


class _StubManager:
    def __init__(self, result: Any = None) -> None:
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, args: dict) -> Any:
        self.calls.append((name, args))
        return self._result

    def list_tools(self) -> list[tuple[str, str, dict]]:
        return [("ida__decompile", "Decompile", {"type": "object", "properties": {}})]


def test_mcptool_exposes_tool_schema():
    tool = MCPTool(
        "ida__decompile",
        "Decompile a function",
        {"type": "object", "properties": {"address": {"type": "string"}}},
        _StubManager(),
    )
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ida__decompile"
    assert schema["function"]["description"] == "Decompile a function"
    assert schema["function"]["parameters"]["properties"]["address"]["type"] == "string"


async def test_mcptool_invoke_returns_manager_result():
    manager: Any = _StubManager(result=MCPResult(content='{"ok": true}'))
    tool = MCPTool("ida__decompile", "d", {}, manager)
    out = await tool.invoke({"address": "0x401000"})
    assert isinstance(out, MCPResult)
    assert out.text == '{"ok": true}'
    assert manager.calls == [("ida__decompile", {"address": "0x401000"})]


def test_build_mcp_tools_from_manager():
    manager: Any = _StubManager()
    tools = build_mcp_tools(manager)
    assert [t.name for t in tools] == ["ida__decompile"]
    assert isinstance(tools[0], MCPTool)
