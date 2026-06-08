from collections.abc import Sequence
from typing import Any

from reagent.tools.base import Tool
from reagent.tools.edit_file import EditFileTool
from reagent.tools.read_file import ReadFileTool
from reagent.tools.shell import ShellTool
from reagent.tools.write_file import WriteFileTool

_ALL_TOOLS: list[Tool] = [ShellTool(), ReadFileTool(), WriteFileTool(), EditFileTool()]

TOOLS: list[dict[str, Any]] = [t.to_schema() for t in _ALL_TOOLS]
TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in _ALL_TOOLS}


def register_tools(extra: Sequence[Tool]) -> None:
    _ALL_TOOLS.extend(extra)
    TOOLS[:] = [t.to_schema() for t in _ALL_TOOLS]
    TOOLS_BY_NAME.clear()
    TOOLS_BY_NAME.update({t.name: t for t in _ALL_TOOLS})


__all__ = ["TOOLS", "TOOLS_BY_NAME", "Tool", "register_tools"]
