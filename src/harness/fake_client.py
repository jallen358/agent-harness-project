"""A scripted stand-in for the model, so loop tests need no API key.

``FakeClient`` implements the ``ModelClient`` protocol. Instead of calling
Claude it replays a list of ``ModelTurn``s you wrote in advance, one per call
to ``step``, and records what the loop sent it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from harness.model import ModelTurn, ToolCall, Usage


class FakeClientExhaustedError(RuntimeError):
    """The loop asked for more turns than the script contained."""


@dataclass(frozen=True)
class RecordedCall:
    """What the loop passed to one ``step`` call, frozen at call time."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    system: str | None


class FakeClient:
    def __init__(self, script: Sequence[ModelTurn]) -> None:
        self._script = list(script)
        self._next = 0
        self.calls: list[RecordedCall] = []

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str | None = None,
    ) -> ModelTurn:
        # Snapshot, because the loop keeps appending to this same list.
        self.calls.append(
            RecordedCall(
                messages=copy.deepcopy(messages),
                tools=copy.deepcopy(tools),
                system=system,
            )
        )
        if self._next >= len(self._script):
            raise FakeClientExhaustedError(
                f"The script had {len(self._script)} scripted turn(s), but the agent "
                f"asked for turn {self._next + 1}. Add a turn to the script, or the "
                "loop is not stopping when it should."
            )
        turn = self._script[self._next]
        self._next += 1
        return turn

    def assert_finished(self) -> None:
        """Fail if the run ended before using every scripted turn."""
        unused = len(self._script) - self._next
        if unused:
            raise AssertionError(
                f"{unused} scripted turn(s) were never used; the run stopped early."
            )


def final_turn(text: str, usage: Usage | None = None) -> ModelTurn:
    """A turn where the model gives its final answer."""
    return ModelTurn(
        text=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=usage or Usage(input_tokens=10, output_tokens=5),
    )


def tool_use_turn(*calls: ToolCall, text: str = "", usage: Usage | None = None) -> ModelTurn:
    """A turn where the model asks for one or more tools to be run."""
    return ModelTurn(
        text=text,
        tool_calls=list(calls),
        stop_reason="tool_use",
        usage=usage or Usage(input_tokens=10, output_tokens=5),
    )
