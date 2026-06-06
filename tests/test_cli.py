from __future__ import annotations

import os

from reagent.cli import build_session


def test_build_session_creates_recorder_under_reagent_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("MODEL_ID", "test-model")

    session = build_session()

    assert session._recorder is not None
    assert session._recorder.path.is_file()
    assert session._recorder.path.is_relative_to(tmp_path / "sessions")


def test_build_session_resumes_existing_path(tmp_path, monkeypatch):
    monkeypatch.setenv("REAGENT_HOME", str(tmp_path))
    monkeypatch.setenv("MODEL_ID", "test-model")
    session = build_session()
    session.add_user("hello")

    assert session._recorder is not None
    resumed = build_session(resume=os.fspath(session._recorder.path))

    assert resumed._recorder is not None
    assert resumed._recorder.path == session._recorder.path
    assert resumed.messages == ({"role": "user", "content": "hello"},)
