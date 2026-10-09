from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from harness.model import (
    AnthropicClient,
    ModelTurn,
    ToolCall,
    Usage,
    turn_from_response,
)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(id: str, name: str, input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def response(
    content: list[SimpleNamespace],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_text_only() -> None:
    turn = turn_from_response(response([text_block("Hello there.")]))

    assert turn.text == "Hello there."
    assert turn.tool_calls == []
    assert turn.stop_reason == "end_turn"
    assert turn.usage == Usage(input_tokens=10, output_tokens=5)


def test_tool_use_only() -> None:
    turn = turn_from_response(
        response(
            [tool_block("toolu_1", "echo", {"text": "hello"})],
            stop_reason="tool_use",
            input_tokens=20,
            output_tokens=8,
        )
    )

    assert turn.text == ""
    assert turn.tool_calls == [ToolCall(id="toolu_1", name="echo", input={"text": "hello"})]
    assert turn.stop_reason == "tool_use"
    assert turn.usage == Usage(input_tokens=20, output_tokens=8)


def test_text_plus_tool_use() -> None:
    turn = turn_from_response(
        response(
            [
                text_block("Let me echo that."),
                tool_block("toolu_1", "echo", {"text": "a"}),
                tool_block("toolu_2", "echo", {"text": "b"}),
            ],
            stop_reason="tool_use",
        )
    )

    assert turn.text == "Let me echo that."
    assert [call.id for call in turn.tool_calls] == ["toolu_1", "toolu_2"]
    assert [call.input for call in turn.tool_calls] == [{"text": "a"}, {"text": "b"}]


def test_multiple_text_blocks_are_concatenated() -> None:
    turn = turn_from_response(response([text_block("Hello, "), text_block("world.")]))

    assert turn.text == "Hello, world."


def test_unknown_block_type_raises_clear_error() -> None:
    unknown = SimpleNamespace(type="server_tool_use", id="srvtoolu_1")

    with pytest.raises(ValueError, match="server_tool_use"):
        turn_from_response(response([text_block("hi"), unknown]))


def test_tool_call_input_is_copied() -> None:
    sdk_input = {"text": "hello"}
    turn = turn_from_response(response([tool_block("toolu_1", "echo", sdk_input)]))

    sdk_input["text"] = "mutated"

    assert turn.tool_calls[0].input == {"text": "hello"}


def test_as_assistant_message_text_and_tool_use() -> None:
    turn = ModelTurn(
        text="Calling echo.",
        tool_calls=[ToolCall(id="toolu_1", name="echo", input={"text": "hello"})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=2),
    )

    assert turn.as_assistant_message() == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Calling echo."},
            {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "hello"}},
        ],
    }


def test_as_assistant_message_text_only() -> None:
    turn = ModelTurn(
        text="Done.",
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=2),
    )

    assert turn.as_assistant_message() == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
    }


def test_as_assistant_message_omits_empty_text_block() -> None:
    # The API rejects empty text blocks, so a tool-only turn must not emit one.
    turn = ModelTurn(
        text="",
        tool_calls=[ToolCall(id="toolu_1", name="echo", input={})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=2),
    )

    assert turn.as_assistant_message() == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {}}],
    }


def test_round_trip_through_response_preserves_content() -> None:
    blocks = [text_block("Hi."), tool_block("toolu_1", "echo", {"text": "x"})]

    message = turn_from_response(response(blocks, stop_reason="tool_use")).as_assistant_message()

    assert message["content"] == [
        {"type": "text", "text": "Hi."},
        {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"text": "x"}},
    ]


def test_from_env_reads_harness_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_MODEL", "some-model")

    client = AnthropicClient.from_env()

    assert client.model == "some-model"
    assert client.max_tokens == 1024


@pytest.mark.parametrize("value", [None, ""])
def test_from_env_requires_harness_model(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    if value is None:
        monkeypatch.delenv("HARNESS_MODEL", raising=False)
    else:
        monkeypatch.setenv("HARNESS_MODEL", value)

    with pytest.raises(ValueError, match="HARNESS_MODEL"):
        AnthropicClient.from_env()


class FakeMessages:
    def __init__(self, reply: SimpleNamespace) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.reply


def client_with_fake(
    monkeypatch: pytest.MonkeyPatch, reply: SimpleNamespace
) -> tuple[AnthropicClient, FakeMessages]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient(model="some-model", max_tokens=256)
    fake = FakeMessages(reply)
    client._client = SimpleNamespace(messages=fake)  # type: ignore[assignment]
    return client, fake


def test_step_sends_request_and_converts_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake = client_with_fake(monkeypatch, response([text_block("ok")]))
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"name": "echo", "description": "d", "input_schema": {"type": "object"}}]

    turn = client.step(messages, tools, system="be brief")

    assert turn.text == "ok"
    assert fake.calls == [
        {
            "model": "some-model",
            "max_tokens": 256,
            "messages": messages,
            "tools": tools,
            "system": "be brief",
        }
    ]


def test_step_omits_system_and_empty_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake = client_with_fake(monkeypatch, response([text_block("ok")]))

    client.step([{"role": "user", "content": "hi"}], [])

    assert "system" not in fake.calls[0]
    assert "tools" not in fake.calls[0]
