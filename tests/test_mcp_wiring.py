from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import reagent.tools as tools_mod
from reagent.config import load_layers
from reagent.repl import _mcp_specs
from reagent.results import ErrorResult, ToolResult
from reagent.tools.base import Tool


def write_toml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def restore_tools():
    all_tools = list(tools_mod._ALL_TOOLS)
    schemas = list(tools_mod.TOOLS)
    handlers = dict(tools_mod.TOOL_HANDLERS)
    yield
    tools_mod._ALL_TOOLS[:] = all_tools
    tools_mod.TOOLS[:] = schemas
    tools_mod.TOOL_HANDLERS.clear()
    tools_mod.TOOL_HANDLERS.update(handlers)


class _FakeTool(Tool):
    name = "fake__ping"
    description = "fake"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def run(self, params: dict[str, Any]) -> ToolResult:
        return ErrorResult("noop")


def test_register_tools_adds_schema_and_handler(restore_tools):
    before = len(tools_mod.TOOLS)
    tools_mod.register_tools([_FakeTool()])
    assert len(tools_mod.TOOLS) == before + 1
    assert tools_mod.TOOLS[-1]["function"]["name"] == "fake__ping"
    assert "fake__ping" in tools_mod.TOOL_HANDLERS
    assert tools_mod.TOOL_HANDLERS["fake__ping"]({}).text == "noop"


def test_mcp_specs_selects_enabled_http_servers_with_url(tmp_path):
    config_path = write_toml(
        tmp_path / "config.toml",
        """
        [mcp.servers.ida]
        transport = "http"
        url = "http://127.0.0.1:14542/mcp"
        headers = { Authorization = "Bearer t" }

        [mcp.servers.local]
        transport = "stdio"
        command = "run"

        [mcp.servers.off]
        enabled = false
        transport = "http"
        url = "http://127.0.0.1:1/mcp"
        """,
    )
    config = load_layers(cwd=tmp_path, env={}, extra_config_paths=[config_path]).config

    specs = _mcp_specs(config)

    assert [s.name for s in specs] == ["ida"]
    assert specs[0].url == "http://127.0.0.1:14542/mcp"
    assert specs[0].headers == {"Authorization": "Bearer t"}
