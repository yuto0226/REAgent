from typing import Any

from reagent.results import ErrorResult, ReadResult, ToolResult
from reagent.tools.base import MAX_OUTPUT, Tool, params, prop, resolve_path


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
    try:
        path = resolve_path(path)
        with open(path) as f:
            lines = f.readlines()

    except (FileNotFoundError, PermissionError, OSError) as e:
        return ErrorResult(f"Error: {e}")

    total = len(lines)
    s = (start_line or 1) - 1
    end = end_line or total
    chunk = lines[s:end]

    raw = "".join(chunk)

    if not raw:
        return ReadResult(path=path, content="", start_line=s + 1)

    if len(raw) > MAX_OUTPUT:
        raw = raw[:MAX_OUTPUT]
        last_newline = raw.rfind("\n")
        if last_newline != -1:
            raw = raw[: last_newline + 1]

        next_line = s + raw.count("\n") + 1
        raw += f"(truncated, use start_line={next_line} to continue, file has {total} lines total)\n"

    return ReadResult(path=path, content=raw, start_line=s + 1)


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Optionally specify a 1-indexed line range."

    @property
    def parameters(self) -> dict[str, Any]:
        return params(
            {
                "path": prop("string"),
                "start_line": prop("integer", "First line to read (1-indexed, inclusive)."),
                "end_line": prop("integer", "Last line to read (1-indexed, inclusive)."),
            },
            required=["path"],
        )

    def run(self, params: dict[str, Any]) -> ToolResult:
        return read_file(params["path"], params.get("start_line"), params.get("end_line"))
