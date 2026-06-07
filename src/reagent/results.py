from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass
class ShellResult:
    output: str

    @property
    def text(self) -> str:
        return self.output


@dataclass
class ReadResult:
    path: str
    content: str  # raw file content, no line-number prefix
    start_line: int = 1

    @property
    def text(self) -> str:
        if not self.content:
            return "(empty file)"
        lines = self.content.splitlines()
        return "\n".join(f"{self.start_line + i}: {line}" for i, line in enumerate(lines))


@dataclass
class DiffResult:
    path: str
    diff: str
    message: str
    kind: Literal["write", "edit"] = "edit"

    @property
    def text(self) -> str:
        return self.message


@dataclass
class ErrorResult:
    message: str

    @property
    def text(self) -> str:
        return self.message


@dataclass
class MCPResult:
    name: str
    content: str
    structured: dict | None = None

    @property
    def text(self) -> str:
        return self.content or "(no content)"

    @classmethod
    def from_call(cls, name: str, result: object) -> ToolResult:
        parts: list[str] = []
        for block in getattr(result, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                parts.append(getattr(block, "text", "") or "")
            elif kind == "image":
                parts.append(f"[image: {getattr(block, 'mimeType', 'image')}]")
            elif kind == "audio":
                parts.append(f"[audio: {getattr(block, 'mimeType', 'audio')}]")
            elif kind == "resource":
                parts.append("[embedded resource]")
            else:
                parts.append(str(block))
        text = "\n".join(p for p in parts if p)

        if getattr(result, "isError", False):
            return ErrorResult(text or f"Error: MCP tool {name!r} reported an error")
        return cls(name=name, content=text, structured=getattr(result, "structuredContent", None))


ToolResult: TypeAlias = ShellResult | ReadResult | DiffResult | ErrorResult | MCPResult
