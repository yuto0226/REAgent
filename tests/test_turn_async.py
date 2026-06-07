import inspect
import asyncio
from types import SimpleNamespace

import pytest

from reagent.config import AgentConfig, Config, LLMConfig, LLMModelsConfig, MCPConfig, SkillsConfig
import reagent.session.turn as turn
from reagent.session import Session
from reagent.session.turn import to_provider_messages, run_turn


def test_run_turn_is_coroutine():
    assert inspect.iscoroutinefunction(run_turn)


def test_to_provider_messages_strips_local_ids():
    messages = (
        {"role": "user", "content": "hello", "id": "m1", "parent_id": None},
        {"role": "assistant", "content": "hi", "id": "m2", "parent_id": "m1"},
    )

    assert to_provider_messages(messages) == [
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


async def test_run_turn_uses_runtime_config(monkeypatch):
    calls = []
    session = Session()
    session.add_user("hello")
    config = Config(
        llm=LLMConfig(
            model="test-model",
            reasoning_effort="high",
            thinking_budget_tokens=1234,
            models=LLMModelsConfig(available=[]),
        ),
        agent=AgentConfig(max_turns=1),
        providers={},
        mcp=MCPConfig(servers={}),
        skills=SkillsConfig(enabled=True, paths=[]),
    )

    async def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="done", reasoning_content=""),
                )
            ],
        )

    monkeypatch.setattr(turn, "_call_llm", fake_call_llm)
    monkeypatch.setattr(turn, "make_compact_fn", lambda model: lambda messages: "")

    await run_turn(session, config)

    assert calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": turn.system_prompt()},
                {"role": "user", "content": "hello"},
            ],
            "tools": turn.TOOLS,
            "reasoning_effort": "high",
            "thinking": {"type": "enabled", "budget_tokens": 1234},
            "num_retries": 10,
        }
    ]
