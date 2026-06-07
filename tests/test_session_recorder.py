from __future__ import annotations

import json
import uuid

from reagent.session.recorder import (
    SessionRecorder,
    find_file,
    read_entries,
    to_provider_message,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recorder_creates_uuid_named_jsonl_under_date_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("reagent.session.recorder.utc_now", lambda: "2026-06-06T12:34:56.789Z")
    monkeypatch.setattr("reagent.session.recorder._client_version", lambda: "9.9.9")

    recorder = SessionRecorder.create(
        root=tmp_path,
        cwd="/repo",
        model="test-model",
        python_version="3.test",
    )

    assert recorder.path.parent == tmp_path / "sessions" / "2026" / "06" / "06"
    assert recorder.path.name == f"{recorder.session_id}.jsonl"
    uuid.UUID(recorder.session_id)

    entries = read_jsonl(recorder.path)
    assert entries == [
        {
            "version": 1,
            "seq": 0,
            "timestamp": "2026-06-06T12:34:56.789Z",
            "session_id": recorder.session_id,
            "type": "meta",
            "data": {
                "created_at": "2026-06-06T12:34:56.789Z",
                "cwd": "/repo",
                "model": "test-model",
                "client": {"name": "reagent", "version": "9.9.9"},
                "python_version": "3.test",
            },
        }
    ]


def test_recorder_writes_message_usage_and_compact_entries(tmp_path, monkeypatch):
    timestamps = iter(
        [
            "2026-06-06T00:00:00.000Z",
            "2026-06-06T00:00:01.000Z",
            "2026-06-06T00:00:02.000Z",
            "2026-06-06T00:00:03.000Z",
        ]
    )
    monkeypatch.setattr("reagent.session.recorder.utc_now", lambda: next(timestamps))
    recorder = SessionRecorder.create(root=tmp_path, cwd="/repo", model="model")

    recorder.record_message({"role": "user", "content": "hello", "id": "m1", "parent_id": None})
    recorder.record_usage(
        prompt_tokens=10,
        completion_tokens=5,
        cached_tokens=3,
        reasoning_tokens=2,
    )
    recorder.record_compact(tail_start_seq=2, summary_seq=3)

    entries = read_jsonl(recorder.path)
    assert [entry["seq"] for entry in entries] == [0, 1, 2, 3]
    assert [entry["type"] for entry in entries] == ["meta", "message", "usage", "compact"]
    assert entries[1]["data"]["id"] == "m1"
    assert entries[1]["data"]["role"] == "user"
    assert entries[2]["data"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cached_tokens": 3,
        "reasoning_tokens": 2,
    }
    assert entries[3]["data"]["tail_start_seq"] == 2
    assert entries[3]["data"]["summary_seq"] == 3
    assert "start_seq" not in entries[3]["data"]
    assert "end_seq" not in entries[3]["data"]


def test_recorder_generates_message_id_and_parent_chain(tmp_path):
    recorder = SessionRecorder.create(root=tmp_path, cwd="/repo", model="model")

    first = recorder.record_message({"role": "user", "content": "hello"})
    second = recorder.record_message({"role": "assistant", "content": "hi"})

    entries = read_jsonl(recorder.path)
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert entries[1]["data"]["id"]
    assert entries[1]["data"]["parent_id"] is None
    assert entries[2]["data"]["parent_id"] == entries[1]["data"]["id"]


def test_read_entries_skips_malformed_json_and_wrong_session(tmp_path):
    session_id = str(uuid.uuid4())
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "version": 1,
                        "seq": 0,
                        "timestamp": "2026-06-06T00:00:00.000Z",
                        "session_id": session_id,
                        "type": "meta",
                        "data": {},
                    }
                ),
                "{not-json",
                json.dumps(
                    {
                        "version": 1,
                        "seq": 1,
                        "timestamp": "2026-06-06T00:00:00.000Z",
                        "session_id": str(uuid.uuid4()),
                        "type": "message",
                        "data": {},
                    }
                ),
                json.dumps(
                    {
                        "version": 1,
                        "seq": 2,
                        "timestamp": "2026-06-06T00:00:01.000Z",
                        "session_id": session_id,
                        "type": "usage",
                        "data": {},
                    }
                ),
            ]
        )
        + "\n"
    )

    entries, skipped = read_entries(path, session_id=session_id)

    assert skipped == 2
    assert [entry["seq"] for entry in entries] == [0, 2]


def test_read_entries_skips_json_entries_with_missing_required_fields(tmp_path):
    session_id = str(uuid.uuid4())
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"version": 1, "session_id": session_id, "type": "message", "data": {}}),
                json.dumps({"version": 1, "seq": 1, "session_id": session_id, "data": {}}),
                json.dumps({"version": 1, "seq": 2, "type": "message", "data": {}}),
                json.dumps(
                    {
                        "version": 1,
                        "seq": 3,
                        "timestamp": "2026-06-06T00:00:00.000Z",
                        "session_id": session_id,
                        "type": "message",
                        "data": {"role": "user", "content": "kept"},
                    }
                ),
            ]
        )
        + "\n"
    )

    entries, skipped = read_entries(path, session_id=session_id)

    assert skipped == 3
    assert [entry["seq"] for entry in entries] == [3]


def test_to_provider_message_strips_local_ids_without_mutating_original():
    message = {"role": "assistant", "content": "ok", "id": "local", "parent_id": "parent", "is_summary": True}

    sanitized = to_provider_message(message)

    assert sanitized == {"role": "assistant", "content": "ok"}
    assert message["id"] == "local"


def test_find_file_accepts_path_or_uuid_scan(tmp_path):
    recorder = SessionRecorder.create(root=tmp_path, cwd="/repo", model="model")

    assert find_file(str(recorder.path), root=tmp_path) == recorder.path
    assert find_file(recorder.session_id, root=tmp_path) == recorder.path
