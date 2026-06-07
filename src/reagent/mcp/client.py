from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from types import TracebackType
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

from reagent.mcp.types import ServerSpec
from reagent.results import ErrorResult, MCPResult, ToolResult


@dataclass
class _Entry:
    spec: ServerSpec
    session: ClientSession
    tool: str
    input_schema: dict[str, Any]
    description: str


class MCPManager:
    def __init__(self, specs: list[ServerSpec], emit: Callable[[str], None] | None = None) -> None:
        self._specs = list(specs)
        self._emit = emit or (lambda _msg: None)
        self._stack = AsyncExitStack()
        self._tools: dict[str, _Entry] = {}

    async def __aenter__(self) -> MCPManager:
        await self._stack.__aenter__()
        try:
            for spec in self._specs:
                await self._connect(spec)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return await self._stack.__aexit__(exc_type, exc, tb)

    async def _connect(self, spec: ServerSpec) -> None:
        try:
            session, tools = await self._open(spec)
        except Exception as exc:
            self._emit(f"[mcp] {spec.name}: connection failed, skipping ({exc})")
            return

        count = 0
        for tool in tools:
            if not spec.allows(tool.name):
                continue
            self._tools[f"{spec.name}__{tool.name}"] = _Entry(
                spec=spec,
                session=session,
                tool=tool.name,
                input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                description=tool.description or "",
            )
            count += 1
        self._emit(f"[mcp] {spec.name}: {count} tool(s) ready")

    async def _open(self, spec: ServerSpec) -> tuple[ClientSession, list[Tool]]:
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(
                spec.url,
                headers=dict(spec.headers),
                timeout=timedelta(seconds=spec.connect_timeout),
            )
        )
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session, (await session.list_tools()).tools

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": entry.description,
                    "parameters": entry.input_schema,
                },
            }
            for name, entry in self._tools.items()
        ]

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            return ErrorResult(f"Error: unknown MCP tool {name!r}")
        try:
            result = await asyncio.wait_for(
                entry.session.call_tool(entry.tool, arguments=args),
                timeout=entry.spec.call_timeout,
            )
        except asyncio.TimeoutError:
            return ErrorResult(f"Error: MCP tool {name!r} timed out after {entry.spec.call_timeout}s")
        except Exception as exc:
            return ErrorResult(f"Error: {exc}")
        return MCPResult(name=name, content=result.model_dump_json())
