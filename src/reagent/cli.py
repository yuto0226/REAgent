from __future__ import annotations

import argparse
import getpass
import os

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ.setdefault("LITELLM_LOG", "ERROR")

from reagent.config import Config, apply_provider_env, load, load_layers, remove_provider_key, set_provider_key  # noqa: E402
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
    subparsers = parser.add_subparsers(dest="command")

    providers = subparsers.add_parser("providers", help="Manage provider credentials")
    provider_subparsers = providers.add_subparsers(dest="provider_command", required=True)

    provider_subparsers.add_parser("list", help="List configured providers")

    login = provider_subparsers.add_parser("login", help="Store a provider API key")
    login.add_argument("provider")
    login.add_argument("--key", help="API key to store; prompts when omitted")

    logout = provider_subparsers.add_parser("logout", help="Remove a stored provider API key")
    logout.add_argument("provider")

    args = parser.parse_args(argv)

    if args.command == "providers":
        return _run_providers(args)

    config = load()
    apply_provider_env(config)
    session = build_session(resume=args.resume, config=config)
    start(session, config)
    return 0


def _run_providers(args: argparse.Namespace) -> int:
    if args.provider_command == "list":
        layers = load_layers()
        for name, provider in sorted(layers.config.providers.items()):
            status = "logged-in" if provider.key else "logged-out"
            print(f"{name}\t{status}")
        return 0

    if args.provider_command == "login":
        key = args.key if args.key is not None else getpass.getpass(f"{args.provider} API key: ")
        set_provider_key(args.provider, key)
        print(f"{args.provider}\tlogged-in")
        return 0

    if args.provider_command == "logout":
        remove_provider_key(args.provider)
        print(f"{args.provider}\tlogged-out")
        return 0

    raise AssertionError(f"unknown provider command: {args.provider_command}")
