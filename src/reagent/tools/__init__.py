from collections.abc import Callable
from typing import Any

from reagent.results import ToolResult
from reagent.tools.base import Tool
from reagent.tools.edit_file import EditFileTool
from reagent.tools.read_file import ReadFileTool
from reagent.tools.shell import ShellTool
from reagent.tools.write_file import WriteFileTool

_ALL_TOOLS: list[Tool] = [ShellTool(), ReadFileTool(), WriteFileTool(), EditFileTool()]

TOOLS: list[dict[str, Any]] = [t.to_schema() for t in _ALL_TOOLS]
TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], ToolResult]] = {t.name: t.run for t in _ALL_TOOLS}

__all__ = ["Tool", "TOOLS", "TOOL_HANDLERS"]
