from __future__ import annotations

import asyncio
import os
import signal
import sys

from reagent.session import Session
from reagent.session.turn import run_turn


async def _get_input() -> str:
    """Read one stdin line via add_reader — no blocking thread."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    fd = sys.stdin.fileno()

    def _on_readable() -> None:
        loop.remove_reader(fd)
        try:
            data = os.read(fd, 4096)
        except OSError:
            if not fut.done():
                fut.set_exception(EOFError())
            return
        if not data:
            if not fut.done():
                fut.set_exception(EOFError())
            return
        if not fut.done():
            fut.set_result(data.decode(errors="replace").rstrip("\n\r"))

    loop.add_reader(fd, _on_readable)
    try:
        return await fut
    except asyncio.CancelledError:
        loop.remove_reader(fd)
        raise


async def run(session: Session) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            session.emit_prompt("> ")
            prompt = await _get_input()
        except EOFError:
            break

        stripped = prompt.strip()
        if not stripped:
            continue

        if stripped.lower() in ("/quit", "/exit"):
            break

        session.add_user(prompt)

        turn_task = asyncio.create_task(run_turn(session))
        loop.add_signal_handler(signal.SIGINT, turn_task.cancel)
        session.emit_thinking_start()

        interrupted = False
        try:
            await turn_task
        except asyncio.CancelledError:
            interrupted = True
        finally:
            session.emit_thinking_stop()
            loop.remove_signal_handler(signal.SIGINT)
        if interrupted:
            session.emit_status("■ Conversation interrupted")

        print()


def start(session: Session) -> None:
    try:
        asyncio.run(run(session))
    except KeyboardInterrupt:
        print()
