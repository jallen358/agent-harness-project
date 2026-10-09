from pydantic import BaseModel, ConfigDict

from harness.tools import ToolRegistry


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def build_demo_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register("echo", "Repeat the given text back.", EchoArgs)
    def echo(args: EchoArgs) -> str:
        return args.text

    return registry