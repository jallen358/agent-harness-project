"""Tool registry.

Each tool is defined once. That single definition serves two purposes:

1. The menu: ``definitions()`` returns what we send to the model so it knows
   which tools exist and what inputs they take.
2. The kitchen: ``run()`` looks a tool up by name, validates the model's input,
   runs the handler, and packages the result.

``run()`` never raises. Every failure comes back as a ``ToolResult`` with
``is_error=True`` and a message written for the model to read and act on.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
Handler = Callable[[Any], str]

@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler

@dataclass
class ToolResult:
    content:str 
    is_error: bool = False

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, args_model: type[BaseModel]
    ) -> Callable[[Handler], Handler]:
        """Decorator that registers a handler as a tool."""
        if not _NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid tool name {name!r}: use 1-128 letters, digits, "
                "underscores, or hyphens."
            )
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered.")

        def decorator(handler: Handler) -> Handler:
            self._tools[name] = Tool(name, description, args_model, handler)
            return handler

        return decorator
    # ---- the menu -----------------------------------------------------

    def names(self) -> list[str]:
        """Return the tool names in alphabetical order."""
        return sorted(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions in the shape the Messages API expects."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.args_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]
    
    def run(self, name: str, raw_input: Any) -> ToolResult:
        """Run a tool by name. Never raises; failures become error results."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                f"Unknown tool {name!r}. Available tools: {', '.join(self.names())}.",
                is_error=True,
            )

        try:
            args = tool.args_model.model_validate(raw_input)
        except ValidationError as exc:
            return ToolResult(
                _describe_validation_error(name, exc),
                is_error=True,
            )

        try:
            return ToolResult(tool.handler(args))
        except Exception as exc:  # noqa: BLE001 - the loop must survive any tool bug
            return ToolResult(
                f"Tool {name!r} failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )

def _describe_validation_error(tool_name: str, exc: ValidationError) -> str:
    """Turn pydantic's verbose error into a short message the model can act on."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "(whole input)"
        problems.append(f"{field}: {err['msg']}")
    return (
        f"Invalid arguments for tool {tool_name!r}. "
        + "; ".join(problems)
        + ". Fix the arguments and try again."
    )