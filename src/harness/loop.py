"""The agent loop: ask the model, run the tools it requests, repeat.

The model only produces text. This loop is the code around it: it sends the
task and the tool menu, runs whatever tools the model asks for, sends the
results back, and stops when the model is finished or a limit is hit.

The loop never imports ``anthropic``. It talks to a ``ModelClient`` (real or
fake) and a ``ToolRegistry``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from harness.model import ModelClient, Usage
from harness.tools import ToolRegistry

StopReason = Literal["final_answer", "max_steps", "max_tokens", "budget", "unexpected"]
EventHandler = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 10  # most model calls allowed in one run
    max_total_tokens: int | None = None  # input + output tokens summed over all steps
    system_prompt: str | None = None


@dataclass
class RunResult:
    stop_reason: StopReason  # why the run ended
    messages: list[dict[str, Any]]  # the full conversation, as sent to the model
    steps: int  # how many times the model was called
    usage: Usage  # tokens used, summed over every step
    final_text: str | None = None  # the model's last words; None if it never finished
    detail: str | None = None  # extra explanation for "unexpected" stops


def run_agent(
    task: str,
    client: ModelClient,
    registry: ToolRegistry,
    config: RunConfig | None = None,
    on_event: EventHandler | None = None,
) -> RunResult:
    if config is None:
        config = RunConfig()

    def emit(event: dict[str, Any]) -> None:
        # Report what is happening (the trajectory log plugs in here later).
        if on_event is not None:
            on_event(event)

    # The conversation so far. The API is stateless, so every call resends it all.
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    total_usage = Usage(input_tokens=0, output_tokens=0)
    steps = 0

    def finish(
        stop_reason: StopReason, final_text: str | None = None, detail: str | None = None
    ) -> RunResult:
        # Every exit goes through here so each run ends with exactly one run_end event.
        emit({"type": "run_end", "stop_reason": stop_reason, "steps": steps, "detail": detail})
        return RunResult(
            stop_reason=stop_reason,
            messages=messages,
            steps=steps,
            usage=total_usage,
            final_text=final_text,
            detail=detail,
        )

    emit({"type": "run_start", "task": task})

    # Each pass through this loop is one step: one model call, plus any tools it asked for.
    for step in range(1, config.max_steps + 1):
        # 1. Ask the model, sending the history and the tool menu.
        turn = client.step(messages, registry.definitions(), config.system_prompt)
        steps = step

        # 2. Remember what the model said. Its reply is part of the history now.
        messages.append(turn.as_assistant_message())

        # 3. Add this call's tokens to the running total (used for the budget).
        total_usage = Usage(
            input_tokens=total_usage.input_tokens + turn.usage.input_tokens,
            output_tokens=total_usage.output_tokens + turn.usage.output_tokens,
        )
        emit(
            {
                "type": "model_turn",
                "step": step,
                "text": turn.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "input": call.input}
                    for call in turn.tool_calls
                ],
                "stop_reason": turn.stop_reason,
                "usage": {
                    "input_tokens": turn.usage.input_tokens,
                    "output_tokens": turn.usage.output_tokens,
                },
            }
        )

        # 4. The model finished on its own: its text is the answer. This wins even
        #    over the budget, because the answer is already in hand.
        if turn.stop_reason in ("end_turn", "stop_sequence"):
            return finish("final_answer", turn.text)

        # 5. The reply was cut off by the output limit. A half-written tool call could
        #    be wrong, so stop here without running anything.
        if turn.stop_reason == "max_tokens":
            return finish("max_tokens", turn.text)

        # 6. Anything other than a clean tool request (a refusal, a pause, or a
        #    "tool_use" that names no tools) is not something this loop can continue from.
        if turn.stop_reason != "tool_use" or not turn.tool_calls:
            detail = (
                "The model stopped for tool use but requested no tools."
                if turn.stop_reason == "tool_use"
                else f"Unexpected stop_reason {turn.stop_reason!r}."
            )
            return finish("unexpected", turn.text, detail)

        # 7. The model asked for tools. Run each one and collect the results.
        results: list[dict[str, Any]] = []
        for call in turn.tool_calls:
            # The registry never raises: a bad call comes back as an error result.
            result = registry.run(call.name, call.input)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,  # tells the model which request this answers
                "content": result.content,
            }
            if result.is_error:
                block["is_error"] = True  # lets the model know it should correct itself
            results.append(block)
            emit(
                {
                    "type": "tool_result",
                    "step": step,
                    "tool_use_id": call.id,
                    "name": call.name,
                    "input": call.input,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )

        # 8. Send all of this turn's results back in ONE user message. Splitting them
        #    across messages teaches the model to stop making parallel calls.
        messages.append({"role": "user", "content": results})

        # 9. Check the budget after the tools ran, so the history stays well formed
        #    (every tool request has its result) and the run could be resumed.
        if (
            config.max_total_tokens is not None
            and total_usage.input_tokens + total_usage.output_tokens >= config.max_total_tokens
        ):
            return finish("budget")

    # 10. Ran out of steps while the model still wanted to work.
    return finish("max_steps")
