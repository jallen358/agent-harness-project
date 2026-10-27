from __future__ import annotations

from typing import Any

import pytest

from harness.fake_client import (
    FakeClient,
    FakeClientExhaustedError,
    final_turn,
    tool_use_turn,
)
from harness.model import ModelClient, ToolCall, Usage


def test_replays_scripted_turns_in_order() -> None:
    client = FakeClient([final_turn("first"), final_turn("second")])

    assert client.step([], []).text == "first"
    assert client.step([], []).text == "second"


def test_satisfies_the_model_client_protocol() -> None:
    # mypy checks this assignment; the loop will only ever see a ModelClient.
    client: ModelClient = FakeClient([final_turn("hi")])

    assert client.step([], []).text == "hi"


def test_records_what_it_was_asked() -> None:
    client = FakeClient([final_turn("ok")])
    tools = [{"name": "echo", "description": "d", "input_schema": {"type": "object"}}]

    client.step([{"role": "user", "content": "go"}], tools, system="be brief")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.messages == [{"role": "user", "content": "go"}]
    assert call.tools == tools
    assert call.system == "be brief"


def test_recorded_messages_are_a_snapshot() -> None:
    # The loop keeps appending to one list. Without a snapshot, every recorded
    # call would show the final conversation instead of what the model saw.
    client = FakeClient([final_turn("a"), final_turn("b")])
    messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]

    client.step(messages, [])
    messages.append({"role": "assistant", "content": "later"})
    client.step(messages, [])

    assert len(client.calls[0].messages) == 1
    assert len(client.calls[1].messages) == 2


def test_raises_a_clear_error_when_the_script_runs_out() -> None:
    client = FakeClient([final_turn("only")])
    client.step([], [])

    with pytest.raises(FakeClientExhaustedError, match="1 scripted turn"):
        client.step([], [])


def test_assert_finished_catches_unused_turns() -> None:
    client = FakeClient([final_turn("a"), final_turn("b")])
    client.step([], [])

    with pytest.raises(AssertionError, match="1 scripted turn"):
        client.assert_finished()

    client.step([], [])
    client.assert_finished()


def test_final_turn_shape() -> None:
    turn = final_turn("done", usage=Usage(input_tokens=7, output_tokens=3))

    assert turn.text == "done"
    assert turn.tool_calls == []
    assert turn.stop_reason == "end_turn"
    assert turn.usage == Usage(input_tokens=7, output_tokens=3)


def test_tool_use_turn_shape() -> None:
    call = ToolCall(id="toolu_1", name="echo", input={"text": "hi"})

    turn = tool_use_turn(call, text="Calling echo.")

    assert turn.text == "Calling echo."
    assert turn.tool_calls == [call]
    assert turn.stop_reason == "tool_use"


def test_tool_use_turn_supports_parallel_calls() -> None:
    first = ToolCall(id="toolu_1", name="echo", input={"text": "a"})
    second = ToolCall(id="toolu_2", name="echo", input={"text": "b"})

    assert tool_use_turn(first, second).tool_calls == [first, second]
