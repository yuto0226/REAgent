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
    content: str

    @property
    def text(self) -> str:
        return self.content


ToolResult: TypeAlias = ShellResult | ReadResult | DiffResult | ErrorResult | MCPResult
