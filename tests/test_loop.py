from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from harness.demo_tools import build_demo_registry
from harness.fake_client import FakeClient, final_turn, tool_use_turn
from harness.loop import RunConfig, run_agent
from harness.model import ModelTurn, ToolCall, Usage
from harness.tools import ToolRegistry


def echo_call(id: str = "toolu_1", text: str = "hi") -> ToolCall:
    return ToolCall(id=id, name="echo", input={"text": text})


class NoArgs(BaseModel):
    pass


def registry_with_recorder() -> tuple[ToolRegistry, list[str]]:
    """A registry whose one tool records every time it actually runs."""
    ran: list[str] = []
    registry = ToolRegistry()

    @registry.register("record", "Record that the tool ran.", NoArgs)
    def record(args: NoArgs) -> str:
        ran.append("ran")
        return "recorded"

    return registry, ran


# ---- finishing -----------------------------------------------------------


def test_final_answer_on_first_turn() -> None:
    client = FakeClient([final_turn("All done.", usage=Usage(10, 5))])

    result = run_agent("Say done.", client, build_demo_registry())

    assert result.stop_reason == "final_answer"
    assert result.final_text == "All done."
    assert result.steps == 1
    assert result.usage == Usage(10, 5)
    assert result.messages == [
        {"role": "user", "content": "Say done."},
        {"role": "assistant", "content": [{"type": "text", "text": "All done."}]},
    ]


def test_tool_call_then_final_answer() -> None:
    client = FakeClient(
        [
            tool_use_turn(echo_call("toolu_1", "hi"), usage=Usage(10, 5)),
            final_turn("echo said hi", usage=Usage(20, 4)),
        ]
    )

    result = run_agent("Echo hi.", client, build_demo_registry())

    assert result.stop_reason == "final_answer"
    assert result.final_text == "echo said hi"
    assert result.steps == 2
    assert result.usage == Usage(30, 9)  # summed across both steps
    assert result.messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "hi"}}
        ],
    }
    assert result.messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi"}],
    }
    client.assert_finished()


def test_parallel_tool_calls_share_one_user_message() -> None:
    client = FakeClient(
        [
            tool_use_turn(echo_call("toolu_1", "a"), echo_call("toolu_2", "b")),
            final_turn("done"),
        ]
    )

    result = run_agent("Echo twice.", client, build_demo_registry())

    tool_messages = [m for m in result.messages if m["role"] == "user" and m is not result.messages[0]]
    assert len(tool_messages) == 1
    assert [block["tool_use_id"] for block in tool_messages[0]["content"]] == ["toolu_1", "toolu_2"]
    assert [block["content"] for block in tool_messages[0]["content"]] == ["a", "b"]


def test_first_request_carries_task_tools_and_system_prompt() -> None:
    client = FakeClient([final_turn("ok")])
    registry = build_demo_registry()

    run_agent("Do it.", client, registry, RunConfig(system_prompt="Be brief."))

    call = client.calls[0]
    assert call.messages == [{"role": "user", "content": "Do it."}]
    assert call.tools == registry.definitions()
    assert call.system == "Be brief."


# ---- tool failures are fed back, not raised ------------------------------


def test_unknown_tool_error_is_returned_to_the_model() -> None:
    client = FakeClient(
        [
            tool_use_turn(ToolCall(id="toolu_1", name="nope", input={})),
            final_turn("recovered"),
        ]
    )

    result = run_agent("Try it.", client, build_demo_registry())

    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "nope" in block["content"] and "echo" in block["content"]
    # The model saw the error on its next turn, and the run carried on.
    assert client.calls[1].messages[-1]["content"][0]["is_error"] is True
    assert result.stop_reason == "final_answer"


def test_invalid_arguments_error_is_returned_to_the_model() -> None:
    client = FakeClient(
        [
            tool_use_turn(ToolCall(id="toolu_1", name="echo", input={})),
            final_turn("recovered"),
        ]
    )

    result = run_agent("Try it.", client, build_demo_registry())

    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "text" in block["content"]  # names the bad field
    assert result.stop_reason == "final_answer"


def test_handler_exception_does_not_crash_the_run() -> None:
    registry = ToolRegistry()

    @registry.register("boom", "Always fails.", NoArgs)
    def boom(args: NoArgs) -> str:
        raise RuntimeError("kaboom")

    client = FakeClient(
        [
            tool_use_turn(ToolCall(id="toolu_1", name="boom", input={})),
            final_turn("recovered"),
        ]
    )

    result = run_agent("Try it.", client, registry)

    block = result.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "kaboom" in block["content"]
    assert result.stop_reason == "final_answer"


# ---- stop conditions -----------------------------------------------------


def test_stops_at_max_steps_with_well_formed_history() -> None:
    client = FakeClient([tool_use_turn(echo_call(f"toolu_{i}")) for i in range(1, 4)])

    result = run_agent("Loop.", client, build_demo_registry(), RunConfig(max_steps=2))

    assert result.stop_reason == "max_steps"
    assert result.final_text is None
    assert result.steps == 2
    assert len(client.calls) == 2  # never asked for a third turn
    # The last tool results were appended, so the history ends with a user message.
    assert result.messages[-1]["role"] == "user"


def test_max_tokens_stops_without_running_truncated_tool_calls() -> None:
    registry, ran = registry_with_recorder()
    truncated = ModelTurn(
        text="I will call",
        tool_calls=[ToolCall(id="toolu_1", name="record", input={})],
        stop_reason="max_tokens",
        usage=Usage(10, 5),
    )
    client = FakeClient([truncated])

    result = run_agent("Go.", client, registry)

    assert result.stop_reason == "max_tokens"
    assert result.final_text == "I will call"
    assert ran == []  # the cut-off call was not executed
    assert result.messages[-1]["role"] == "assistant"


def test_stops_when_token_budget_is_reached() -> None:
    client = FakeClient(
        [
            tool_use_turn(echo_call("toolu_1"), usage=Usage(100, 50)),
            final_turn("never reached"),
        ]
    )

    result = run_agent(
        "Go.", client, build_demo_registry(), RunConfig(max_total_tokens=120)
    )

    assert result.stop_reason == "budget"
    assert result.usage == Usage(100, 50)
    assert result.steps == 1
    assert len(client.calls) == 1


def test_final_answer_wins_over_budget() -> None:
    # The answer is already in hand, so going over budget on the last turn is not a failure.
    client = FakeClient([final_turn("done", usage=Usage(500, 500))])

    result = run_agent("Go.", client, build_demo_registry(), RunConfig(max_total_tokens=10))

    assert result.stop_reason == "final_answer"


def test_unexpected_stop_reason_is_reported() -> None:
    refusal = ModelTurn(text="", tool_calls=[], stop_reason="refusal", usage=Usage(1, 1))
    client = FakeClient([refusal])

    result = run_agent("Go.", client, build_demo_registry())

    assert result.stop_reason == "unexpected"
    assert result.detail is not None and "refusal" in result.detail


def test_tool_use_without_tool_calls_is_unexpected() -> None:
    odd = ModelTurn(text="", tool_calls=[], stop_reason="tool_use", usage=Usage(1, 1))
    client = FakeClient([odd])

    result = run_agent("Go.", client, build_demo_registry())

    assert result.stop_reason == "unexpected"
    assert result.detail is not None and "tool" in result.detail


# ---- events --------------------------------------------------------------


def test_on_event_reports_each_step_in_order() -> None:
    events: list[dict[str, Any]] = []
    client = FakeClient([tool_use_turn(echo_call("toolu_1", "hi")), final_turn("done")])

    run_agent("Echo hi.", client, build_demo_registry(), on_event=events.append)

    assert [e["type"] for e in events] == [
        "run_start",
        "model_turn",
        "tool_result",
        "model_turn",
        "run_end",
    ]
    tool_event = events[2]
    assert tool_event["name"] == "echo"
    assert tool_event["tool_use_id"] == "toolu_1"
    assert tool_event["content"] == "hi"
    assert tool_event["is_error"] is False
    assert events[-1]["stop_reason"] == "final_answer"
