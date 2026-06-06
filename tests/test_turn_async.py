import inspect
import asyncio

import pytest

import reagent.session.turn as turn
from reagent.session.turn import provider_messages, run_turn


def test_run_turn_is_coroutine():
    assert inspect.iscoroutinefunction(run_turn)


def test_provider_messages_strips_local_ids():
    messages = (
        {"role": "user", "content": "hello", "id": "m1", "parent_id": None},
        {"role": "assistant", "content": "hi", "id": "m2", "parent_id": "m1"},
    )

    assert provider_messages(messages) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


async def test_llm_call_finishes_in_background_after_caller_is_cancelled(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def fake_acompletion(**kwargs):
        started.set()
        await release.wait()
        finished.set()
        return object()

    monkeypatch.setattr(turn, "acompletion", fake_acompletion)

    task = asyncio.create_task(turn._call_llm(model="test", messages=[]))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not finished.is_set()

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=1)
