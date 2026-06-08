from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from typing import Any

from reagent.mcp.client import MCPManager
from reagent.results import ErrorResult, ToolResult
from reagent.tools.base import Tool


class MCPTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        mcp: MCPManager,
        loop: AbstractEventLoop,
    ) -> None:
        self.name = name
        self.description = description
        self._parameters = parameters
        self._mcp = mcp
        self._loop = loop

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def run(self, params: dict[str, Any]) -> ToolResult:
        future = asyncio.run_coroutine_threadsafe(self._mcp.call(self.name, params), self._loop)
        try:
            return future.result()
        except Exception as exc:
            return ErrorResult(f"Error: {exc}")


def build_mcp_tools(mcp: MCPManager, loop: AbstractEventLoop) -> list[MCPTool]:
    return [MCPTool(name, description, parameters, mcp, loop) for name, description, parameters in mcp.list_tools()]
