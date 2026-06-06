from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ.setdefault("LITELLM_LOG", "ERROR")

from reagent.repl import start  # noqa: E402
from reagent.session import Session, SessionRecorder, find_file, load_session  # noqa: E402


def build_session(resume: str | None = None, sink=None) -> Session:
    if resume is not None:
        path = find_file(resume)
        return load_session(path, sink=sink)

    recorder = SessionRecorder.create(
        cwd=os.getcwd(),
        model=os.environ["MODEL_ID"],
    )
    return Session(sink=sink, recorder=recorder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Resume a session by JSONL path or session UUID")
    args = parser.parse_args(argv)

    session = build_session(resume=args.resume)
    start(session)
    return 0
