from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from harness.demo_tools import build_demo_registry
from harness.fake_client import FakeClient, final_turn, tool_use_turn
from harness.loop import run_agent
from harness.model import ToolCall
from harness.trajectory import TrajectoryLogger, new_run_path, read_trajectory


def fixed_clock() -> datetime:
    return datetime(2026, 10, 27, 15, 30, 12, 123000, tzinfo=UTC)


def test_writes_one_json_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    log = TrajectoryLogger(path, clock=fixed_clock)

    log({"type": "run_start", "task": "go"})
    log({"type": "run_end", "stop_reason": "final_answer"})

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "ts": "2026-10-27T15:30:12.123+00:00",
        "type": "run_start",
        "task": "go",
    }
    assert json.loads(lines[1])["type"] == "run_end"


def test_each_event_is_on_disk_before_the_next_one(tmp_path: Path) -> None:
    # If the run crashes, everything logged so far must already be in the file.
    path = tmp_path / "run.jsonl"
    log = TrajectoryLogger(path, clock=fixed_clock)

    log({"type": "run_start"})
    assert len(path.read_text().splitlines()) == 1

    log({"type": "model_turn"})
    assert len(path.read_text().splitlines()) == 2


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "nested" / "run.jsonl"

    TrajectoryLogger(path, clock=fixed_clock)({"type": "run_start"})

    assert path.exists()


def test_appends_instead_of_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    TrajectoryLogger(path, clock=fixed_clock)({"type": "first"})
    TrajectoryLogger(path, clock=fixed_clock)({"type": "second"})

    assert [e["type"] for e in read_trajectory(path)] == ["first", "second"]


def test_non_ascii_text_is_kept_readable(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    TrajectoryLogger(path, clock=fixed_clock)({"type": "model_turn", "text": "café €5"})

    assert "café €5" in path.read_text(encoding="utf-8")
    assert read_trajectory(path)[0]["text"] == "café €5"


def test_unserializable_event_raises_instead_of_dropping_it(tmp_path: Path) -> None:
    log = TrajectoryLogger(tmp_path / "run.jsonl", clock=fixed_clock)

    with pytest.raises(TypeError):
        log({"type": "model_turn", "text": object()})


def test_read_trajectory_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"type": "a"}\n\n{"type": "b"}\n')

    assert [e["type"] for e in read_trajectory(path)] == ["a", "b"]


def test_read_trajectory_names_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"type": "a"}\n{not json}\n')

    with pytest.raises(ValueError, match="line 2"):
        read_trajectory(path)


def test_read_trajectory_rejects_lines_that_are_not_objects(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("[1, 2, 3]\n")

    with pytest.raises(ValueError, match="line 1"):
        read_trajectory(path)


def test_new_run_path_is_timestamped_and_unique(tmp_path: Path) -> None:
    first = new_run_path(tmp_path, clock=fixed_clock)
    second = new_run_path(tmp_path, clock=fixed_clock)

    assert first.parent == tmp_path
    assert first.name.startswith("20261027T153012Z-")
    assert first.suffix == ".jsonl"
    assert first != second  # two runs in the same second must not collide


def test_logger_plugs_into_the_loop(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    client = FakeClient(
        [
            tool_use_turn(ToolCall(id="toolu_1", name="echo", input={"text": "hi"})),
            final_turn("echo said hi"),
        ]
    )

    run_agent("Echo hi.", client, build_demo_registry(), on_event=TrajectoryLogger(path))

    events: list[dict[str, Any]] = read_trajectory(path)
    assert [e["type"] for e in events] == [
        "run_start",
        "model_turn",
        "tool_result",
        "model_turn",
        "run_end",
    ]
    assert all("ts" in e for e in events)
    assert events[-1]["final_text"] == "echo said hi"
