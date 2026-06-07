from __future__ import annotations

import asyncio
from typing import Any

from reagent.mcp.client import MCPManager, _Entry
from reagent.mcp.types import ServerSpec
from reagent.results import ErrorResult, MCPResult


class _Block:
    def __init__(self, type: str, text: str = "", mimeType: str = "") -> None:
        self.type = type
        self.text = text
        self.mimeType = mimeType


class _CallResult:
    def __init__(self, content: list[Any], isError: bool = False, structuredContent: Any = None) -> None:
        self.content = content
        self.isError = isError
        self.structuredContent = structuredContent


class _StubSession:
    def __init__(self, result: Any = None, delay: float = 0.0, raises: Exception | None = None) -> None:
        self._result = result
        self._delay = delay
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        self.calls.append((name, arguments or {}))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return self._result


def _manager_with(
    session: Any, *, tool: str = "decompile", call_timeout: float = 60.0, schema: dict | None = None
) -> MCPManager:
    spec = ServerSpec(name="ida", url="http://x/mcp", call_timeout=call_timeout)
    m = MCPManager([spec])
    m._tools[f"ida__{tool}"] = _Entry(
        spec=spec,
        session=session,
        tool=tool,
        input_schema=schema or {"type": "object", "properties": {}},
        description="Decompile a function",
    )
    return m


def test_allows_everything_when_allowlist_is_none():
    assert ServerSpec(name="x", url="u").allows("anything")


def test_allowlist_filters_tools():
    spec = ServerSpec(name="x", url="u", allow=("decompile", "xrefs_to"))
    assert spec.allows("decompile")
    assert not spec.allows("delete_database")


def test_from_call_flattens_text_blocks():
    out = MCPResult.from_call("ida__decompile", _CallResult([_Block("text", "a"), _Block("text", "b")]))
    assert isinstance(out, MCPResult)
    assert out.text == "a\nb"


def test_from_call_marks_non_text_blocks():
    out = MCPResult.from_call("t", _CallResult([_Block("image", mimeType="image/png")]))
    assert isinstance(out, MCPResult)
    assert "image" in out.text.lower()


def test_from_call_empty_content_reads_as_no_content():
    out = MCPResult.from_call("t", _CallResult([]))
    assert isinstance(out, MCPResult)
    assert out.text == "(no content)"


def test_from_call_iserror_maps_to_error_result():
    out = MCPResult.from_call("t", _CallResult([_Block("text", "kaboom")], isError=True))
    assert isinstance(out, ErrorResult)
    assert "kaboom" in out.text


def test_schemas_emit_openai_function_shape():
    m = _manager_with(
        _StubSession(),
        schema={"type": "object", "properties": {"address": {"type": "string"}}, "required": ["address"]},
    )
    schemas = m.schemas()
    assert len(schemas) == 1
    fn = schemas[0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "ida__decompile"
    assert fn["function"]["parameters"]["properties"]["address"]["type"] == "string"


def test_has_and_tool_names():
    m = _manager_with(_StubSession())
    assert m.has("ida__decompile")
    assert not m.has("ida__missing")
    assert m.tool_names == ["ida__decompile"]


async def test_call_success_returns_mcpresult():
    session = _StubSession(result=_CallResult([_Block("text", "int main() {}")]))
    m = _manager_with(session)
    out = await m.call("ida__decompile", {"address": "0x401000"})
    assert isinstance(out, MCPResult)
    assert "int main" in out.text
    assert session.calls == [("decompile", {"address": "0x401000"})]


async def test_call_unknown_tool_returns_error():
    out = await _manager_with(_StubSession()).call("ida__nope", {})
    assert isinstance(out, ErrorResult)
    assert "unknown" in out.text.lower()


async def test_call_timeout_returns_error():
    session = _StubSession(result=_CallResult([_Block("text", "slow")]), delay=0.2)
    out = await _manager_with(session, call_timeout=0.01).call("ida__decompile", {})
    assert isinstance(out, ErrorResult)
    assert "timed out" in out.text.lower()


async def test_call_client_exception_maps_to_error():
    session = _StubSession(raises=RuntimeError("disconnected"))
    out = await _manager_with(session).call("ida__decompile", {})
    assert isinstance(out, ErrorResult)
    assert "disconnected" in out.text


async def test_call_server_iserror_maps_to_error():
    session = _StubSession(result=_CallResult([_Block("text", "boom")], isError=True))
    out = await _manager_with(session).call("ida__decompile", {})
    assert isinstance(out, ErrorResult)
    assert "boom" in out.text
