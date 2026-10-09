import pytest
from pydantic import BaseModel, ConfigDict

from harness.tools import ToolRegistry


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    times: int = 1


def _echo(args: EchoArgs) -> str:
    return " ".join([args.text] * args.times)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", "Repeat text back.", EchoArgs)(_echo)
    return registry


def test_valid_call() -> None:
    result = make_registry().run("echo", {"text": "hi", "times": 2})
    assert not result.is_error
    assert result.content == "hi hi"


def test_default_argument_used() -> None:
    assert make_registry().run("echo", {"text": "hi"}).content == "hi"


def test_missing_required_argument() -> None:
    result = make_registry().run("echo", {})
    assert result.is_error
    assert "text" in result.content
    assert "echo" in result.content


def test_wrong_type() -> None:
    result = make_registry().run("echo", {"text": "hi", "times": "lots"})
    assert result.is_error
    assert "times" in result.content


def test_extra_field_rejected_when_model_forbids_it() -> None:
    result = make_registry().run("echo", {"text": "hi", "loud": True})
    assert result.is_error
    assert "loud" in result.content


def test_input_that_is_not_an_object() -> None:
    result = make_registry().run("echo", "hello")
    assert result.is_error


def test_unknown_tool_lists_valid_names() -> None:
    result = make_registry().run("ecko", {"text": "hi"})
    assert result.is_error
    assert "ecko" in result.content
    assert "echo" in result.content


def test_handler_exception_becomes_error_result() -> None:
    registry = ToolRegistry()

    def boom(args: EchoArgs) -> str:
        raise RuntimeError("disk on fire")

    registry.register("boom", "Always fails.", EchoArgs)(boom)
    result = registry.run("boom", {"text": "x"})
    assert result.is_error
    assert "disk on fire" in result.content


def test_duplicate_registration_raises() -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register("echo", "Again.", EchoArgs)(_echo)


def test_invalid_tool_name_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Invalid tool name"):
        registry.register("has space", "Bad name.", EchoArgs)(_echo)


def test_definitions_shape() -> None:
    (definition,) = make_registry().definitions()
    assert definition["name"] == "echo"
    assert definition["description"] == "Repeat text back."
    schema = definition["input_schema"]
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"text", "times"}
    assert schema["required"] == ["text"]