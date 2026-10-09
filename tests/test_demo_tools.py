from harness.demo_tools import build_demo_registry


def test_demo_registry_has_echo() -> None:
    assert build_demo_registry().names() == ["echo"]


def test_echo_runs() -> None:
    result = build_demo_registry().run("echo", {"text": "hello"})
    assert not result.is_error
    assert result.content == "hello"


def test_registries_are_independent() -> None:
    first = build_demo_registry()
    second = build_demo_registry()  # would raise if registration leaked into a global
    assert first is not second