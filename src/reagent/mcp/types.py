from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServerSpec:
    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    allow: tuple[str, ...] | None = None
    connect_timeout: float = 10.0
    call_timeout: float = 60.0

    def allows(self, tool: str) -> bool:
        return self.allow is None or tool in self.allow
