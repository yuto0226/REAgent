from __future__ import annotations

import json
import os
from typing import Any, cast

import litellm

litellm.drop_params = True
from litellm import acompletion  # noqa: E402
from litellm.exceptions import BadRequestError  # noqa: E402
from litellm.types.utils import ModelResponse  # noqa: E402

from reagent.compact import make_compact_fn  # noqa: E402
from reagent.results import ErrorResult  # noqa: E402
from reagent.session.prompt import system_prompt  # noqa: E402
from reagent.session.session import Session  # noqa: E402
from reagent.tools import TOOLS, TOOL_HANDLERS  # noqa: E402

MODEL = os.environ["MODEL_ID"]
MAX_ITERATIONS = 50
THINKING_BUDGET = 8192


def extract_reasoning(message: Any) -> str:
    rc = getattr(message, "reasoning_content", None)
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    return ""


def extract_text(message: Any) -> str:
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if text:
            texts.append(str(text))

    return "\n".join(texts).strip()


async def run_turn(session: Session) -> None:
    compact_fn = make_compact_fn(MODEL)  # TODO: make_compact_fn should use acompletion; sync call blocks event loop
    sys_prompt = system_prompt()

    for _ in range(MAX_ITERATIONS):
        before = session._estimate_tokens()
        session.compact(compact_fn)
        after = session._estimate_tokens()

        if after < before:
            session.emit_status(f"[compact: {before} → {after} tokens]")

        messages = [{"role": "system", "content": sys_prompt}, *session.messages]
        session.emit_thinking_update("up", session._estimate_tokens())
        try:
            resp = cast(
                ModelResponse,
                await acompletion(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    reasoning_effort="medium",
                    thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
                    num_retries=10,
                ),
            )
        except BadRequestError as exc:
            session.add_assistant(f"Stopped: request rejected by API - {exc}")
            return

        usage = getattr(resp, "usage", None)
        down_tokens = getattr(usage, "completion_tokens", 0) or 0
        session.emit_thinking_update("down", down_tokens)
        session.record_usage(usage)
        choice0 = resp.choices[0]
        message = choice0.message
        text = extract_text(message)

        if choice0.finish_reason == "length":
            session.add_assistant("Stopped: response hit max tokens. The output may be incomplete.")
            return

        thought = extract_reasoning(choice0.message) or text
        if thought and choice0.finish_reason == "tool_calls":
            session.add_think(thought)

        if choice0.finish_reason != "tool_calls":
            session.add_assistant(text)
            return

        if not message.tool_calls:
            raise RuntimeError(f"finish_reason=tool_calls but tool_calls is empty: {message}")

        session.add_tool_calls(message)

        for tc in message.tool_calls:
            name = tc.function.name or ""

            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                session.add_tool_result(tc.id, ErrorResult(f"Error: invalid tool arguments: {exc}"))
                continue

            session.emit_tool_call(name, tool_input)
            handler = TOOL_HANDLERS.get(name)
            # TODO: wrap handler in run_in_executor; subprocess.run() blocks event loop
            result = handler(tool_input) if handler else ErrorResult(f"Error: unknown tool {name!r}")

            session.add_tool_result(tc.id, result)

    session.add_assistant(f"Stopped: reached iteration limit of {MAX_ITERATIONS}.")
