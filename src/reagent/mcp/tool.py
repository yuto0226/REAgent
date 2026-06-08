from __future__ import annotations

from typing import Any

from reagent.mcp.client import MCPManager
from reagent.results import ToolResult
from reagent.tools.base import Tool


class MCPTool(Tool):
    def __init__(self, name: str, description: str, parameters: dict[str, Any], mcp: MCPManager) -> None:
        self.name = name
        self.description = description
        self._parameters = parameters
        self._mcp = mcp

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        return await self._mcp.call(self.name, params)


def build_mcp_tools(mcp: MCPManager) -> list[MCPTool]:
    return [MCPTool(name, description, parameters, mcp) for name, description, parameters in mcp.list_tools()]
