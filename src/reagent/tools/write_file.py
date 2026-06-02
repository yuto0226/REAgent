import difflib
import os
from typing import Any

from reagent.results import DiffResult, ErrorResult, ToolResult
from reagent.tools.base import Tool, params, prop, resolve_path


def write_file(path: str, content: str) -> ToolResult:
    try:
        path = resolve_path(path)
    except PermissionError as e:
        return ErrorResult(f"Error: {e}")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            with open(path) as f:
                old_content = f.read()
        except FileNotFoundError:
            old_content = ""

        with open(path, "w") as f:
            f.write(content)

        diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            )
        )
        return DiffResult(
            path=path,
            diff=diff,
            message=f"Written {len(content)} bytes to {path}",
        )

    except (PermissionError, OSError) as e:
        return ErrorResult(f"Error: {e}")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a file with the given content."

    @property
    def parameters(self) -> dict[str, Any]:
        return params(
            {"path": prop("string"), "content": prop("string")},
            required=["path", "content"],
        )

    def run(self, params: dict[str, Any]) -> ToolResult:
        return write_file(params["path"], params["content"])
