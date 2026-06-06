from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, TypedDict, cast, get_args


EventType = Literal["meta", "message", "usage", "compact"]
EVENT_TYPES: frozenset[str] = frozenset(get_args(EventType))


class SessionEntry(TypedDict):
    version: int
    seq: int
    timestamp: str
    session_id: str
    type: EventType
    data: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _client_version() -> str:
    try:
        return version("reagent")
    except PackageNotFoundError:
        return "dev"


def _data_root() -> Path:
    return Path(os.environ.get("REAGENT_HOME", "~/.reagent")).expanduser()


_INTERNAL_FIELDS = frozenset({"id", "parent_id", "is_think", "result_type", "result_data"})


def to_provider_message(message: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if key not in _INTERNAL_FIELDS}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return {key: jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def read_entries(path: str | os.PathLike[str], session_id: str | None = None) -> tuple[list[SessionEntry], int]:
    entries: list[SessionEntry] = []
    skipped = 0

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        if not isinstance(entry, dict):
            skipped += 1
            continue

        if (
            entry.get("version") != 1
            or not isinstance(entry.get("seq"), int)
            or not isinstance(entry.get("timestamp"), str)
            or not isinstance(entry.get("session_id"), str)
            or entry.get("type") not in EVENT_TYPES
            or not isinstance(entry.get("data"), dict)
        ):
            skipped += 1
            continue

        if session_id is not None and entry.get("session_id") != session_id:
            skipped += 1
            continue

        entries.append(cast(SessionEntry, entry))

    return entries, skipped


def find_file(value: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate

    session_id = str(value)
    try:
        uuid.UUID(session_id)
    except ValueError as exc:
        raise ValueError(f"not a valid session path or UUID: {value}") from exc

    data_root = Path(root).expanduser() if root is not None else _data_root()
    matches = sorted((data_root / "sessions").glob(f"**/{session_id}.jsonl"))

    if not matches:
        raise FileNotFoundError(f"session not found: {session_id}")
    return matches[-1]


class SessionRecorder:
    def __init__(self, path: Path, session_id: str, seq: int) -> None:
        self.path = path
        self.session_id = session_id
        self._seq = seq
        self._last_message_id: str | None = None
        self._dir_ensured = False

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str] | None = None,
        *,
        cwd: str,
        model: str,
        python_version: str | None = None,
    ) -> SessionRecorder:
        created_at = utc_now()
        session_id = str(uuid.uuid4())
        date = created_at[:10].split("-")

        data_root = Path(root).expanduser() if root is not None else _data_root()
        path = data_root / "sessions" / date[0] / date[1] / date[2] / f"{session_id}.jsonl"

        recorder = cls(path=path, session_id=session_id, seq=0)
        recorder._write(
            "meta",
            {
                "created_at": created_at,
                "cwd": cwd,
                "model": model,
                "client": {"name": "reagent", "version": _client_version()},
                "python_version": python_version if python_version is not None else sys.version.split()[0],
            },
            ts=created_at,
        )
        return recorder

    @classmethod
    def resume(
        cls,
        path: str | os.PathLike[str],
        session_id: str | None = None,
        *,
        entries: list[SessionEntry] | None = None,
    ) -> SessionRecorder:
        resolved = Path(path).expanduser()
        resolved_session_id = session_id if session_id is not None else resolved.stem

        if entries is None:
            entries, _ = read_entries(resolved, session_id=resolved_session_id)

        next_seq = max((entry["seq"] for entry in entries), default=-1) + 1
        recorder = cls(path=resolved, session_id=resolved_session_id, seq=next_seq)
        recorder._dir_ensured = True  # resume path already exists

        for entry in entries:
            if entry["type"] == "message":
                message_id = entry["data"].get("id")
                if isinstance(message_id, str):
                    recorder._last_message_id = message_id

        return recorder

    def record_message(self, message: Mapping[str, Any]) -> SessionEntry:
        data = jsonable(message)
        data.setdefault("id", str(uuid.uuid4()))
        data.setdefault("parent_id", self._last_message_id)

        entry = self._write("message", data)
        message_id = data.get("id")

        if isinstance(message_id, str):
            self._last_message_id = message_id
        return entry

    def record_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
    ) -> None:
        self._write(
            "usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
        )

    def record_compact(self, *, start_seq: int, end_seq: int, replacement_message: dict[str, Any]) -> SessionEntry:
        return self._write(
            "compact",
            {
                "start_seq": start_seq,
                "end_seq": end_seq,
                "replacement_message": jsonable(replacement_message),
            },
        )

    def _write(self, event_type: EventType, data: dict[str, Any], ts: str | None = None) -> SessionEntry:
        if not self._dir_ensured:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._dir_ensured = True
        entry: SessionEntry = {
            "version": 1,
            "seq": self._seq,
            "timestamp": ts if ts is not None else utc_now(),
            "session_id": self.session_id,
            "type": event_type,
            "data": data,
        }

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._seq += 1

        return entry
