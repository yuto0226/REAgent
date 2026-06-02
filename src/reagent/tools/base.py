from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from reagent.results import ToolResult


MAX_OUTPUT = 50_000

# Snapshot at import time — intentionally fixed to the workspace root at startup.
_ALLOWED_ROOTS: tuple[str, ...] = (
    os.path.realpath(os.getcwd()),
    os.path.realpath("/tmp"),
)


def resolve_path(path: str) -> str:
    """Resolve path and verify it's within an allowed root."""
    resolved = os.path.realpath(path)
    if not any(resolved.startswith(root + os.sep) or resolved == root for root in _ALLOWED_ROOTS):
        allowed = ", ".join(_ALLOWED_ROOTS)
        raise PermissionError(f"Path '{resolved}' is outside allowed roots: {allowed}")
    return resolved


PropType = Literal["string", "integer", "number", "boolean", "array", "object", "null"]


def prop(type: PropType, description: str = "") -> dict[str, Any]:
    return {"type": type, "description": description} if description else {"type": type}


def params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


class Tool(ABC):
    name: str
    description: str

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self, params: dict[str, Any]) -> "ToolResult": ...

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
