"""Thin wrapper hiding the Anthropic SDK behind simple types.

This is the only module allowed to import ``anthropic``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: Usage = field(default_factory=lambda: Usage(0, 0))

    def as_assistant_message(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if self.text:  # the API rejects empty text blocks
            content.append({"type": "text", "text": self.text})
        for call in self.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            )
        return {"role": "assistant", "content": content}


class ModelClient(Protocol):
    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str | None = None,
    ) -> ModelTurn: ...


def turn_from_response(response: Any) -> ModelTurn:
    """Convert an SDK ``Message`` into a ``ModelTurn``.

    Raises ``ValueError`` on content block types it does not understand rather
    than silently dropping them.
    """
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        else:
            raise ValueError(f"Unsupported content block type: {block.type!r}")
    return ModelTurn(
        text="".join(texts),
        tool_calls=tool_calls,
        stop_reason=response.stop_reason or "",
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
    )


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # the SDK reads ANTHROPIC_API_KEY itself

    @classmethod
    def from_env(cls) -> AnthropicClient:
        model = os.environ.get("HARNESS_MODEL")
        if not model:
            raise ValueError("HARNESS_MODEL is not set")
        return cls(model=model)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str | None = None,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if system is not None:
            kwargs["system"] = system
        response: Any = self._client.messages.create(**kwargs)
        return turn_from_response(response)
