from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ.setdefault("LITELLM_LOG", "ERROR")

from reagent.config import Config, apply_provider_env, load  # noqa: E402
from reagent.repl import start  # noqa: E402
from reagent.session import Session, SessionRecorder, find_file, load_session  # noqa: E402


def build_session(resume: str | None = None, sink=None, config: Config | None = None) -> Session:
    resolved = config if config is not None else load()
    if resume is not None:
        path = find_file(resume)
        return load_session(path, sink=sink)

    recorder = SessionRecorder.create(
        cwd=os.getcwd(),
        model=resolved.llm.model,
    )
    return Session(sink=sink, recorder=recorder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Resume a session by JSONL path or session UUID")
    args = parser.parse_args(argv)

    config = load()
    apply_provider_env(config)
    session = build_session(resume=args.resume, config=config)
    start(session, config)
    return 0
