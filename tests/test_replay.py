from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.demo_tools import build_demo_registry
from harness.fake_client import FakeClient, final_turn, tool_use_turn
from harness.loop import run_agent
from harness.model import ToolCall
from harness.replay import format_trajectory, main
from harness.trajectory import TrajectoryLogger

RUN_START: dict[str, Any] = {
    "type": "run_start",
    "task": "Echo hi.",
    "system_prompt": None,
    "tool_names": ["echo"],
    "max_steps": 10,
    "max_total_tokens": None,
}
TOOL_TURN: dict[str, Any] = {
    "type": "model_turn",
    "step": 1,
    "text": "Calling echo.",
    "tool_calls": [{"id": "toolu_1", "name": "echo", "input": {"text": "hi"}}],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}
TOOL_RESULT: dict[str, Any] = {
    "type": "tool_result",
    "step": 1,
    "tool_use_id": "toolu_1",
    "name": "echo",
    "input": {"text": "hi"},
    "content": "hi",
    "is_error": False,
}
FINAL_TURN: dict[str, Any] = {
    "type": "model_turn",
    "step": 2,
    "text": "echo said hi",
    "tool_calls": [],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 20, "output_tokens": 4},
}
RUN_END: dict[str, Any] = {
    "type": "run_end",
    "stop_reason": "final_answer",
    "steps": 2,
    "detail": None,
    "final_text": "echo said hi",
    "usage": {"input_tokens": 30, "output_tokens": 9},
}


def test_formats_a_whole_run() -> None:
    lines = format_trajectory([RUN_START, TOOL_TURN, TOOL_RESULT, FINAL_TURN, RUN_END])

    assert lines == [
        "RUN START  Echo hi.",
        "  tools: echo  max_steps: 10  budget: none",
        "STEP 1  model  stop_reason=tool_use  tokens in=10 out=5",
        "  says: Calling echo.",
        '  calls: echo {"text": "hi"} [toolu_1]',
        "STEP 1  tool   echo -> hi",
        "STEP 2  model  stop_reason=end_turn  tokens in=20 out=4",
        "  says: echo said hi",
        "RUN END  final_answer after 2 steps  tokens in=30 out=9",
    ]


def test_tool_errors_are_marked() -> None:
    error = {**TOOL_RESULT, "content": "Unknown tool 'nope'.", "is_error": True}

    lines = format_trajectory([error])

    assert lines[0] == "STEP 1  tool   echo -> ERROR: Unknown tool 'nope'."


def test_long_content_is_clipped_unless_full() -> None:
    long_result = {**TOOL_RESULT, "content": "x" * 500}

    clipped = format_trajectory([long_result], limit=50)[0]
    full = format_trajectory([long_result], limit=None)[0]

    assert len(clipped) < 100
    assert "more chars" in clipped
    assert full.endswith("x" * 500)


def test_newlines_stay_on_one_line() -> None:
    result = {**TOOL_RESULT, "content": "line one\nline two"}

    assert format_trajectory([result])[0] == "STEP 1  tool   echo -> line one\\nline two"


def test_flags_a_log_that_never_finished() -> None:
    lines = format_trajectory([RUN_START, TOOL_TURN])

    assert "without a run_end" in lines[-1]


def test_run_end_detail_and_budget_are_shown() -> None:
    start = {**RUN_START, "max_total_tokens": 500, "system_prompt": "Be brief."}
    end = {**RUN_END, "stop_reason": "unexpected", "detail": "Unexpected stop_reason 'refusal'."}

    lines = format_trajectory([start, end])

    assert "budget: 500" in lines[1]
    assert lines[2] == "  system: Be brief."
    assert lines[-1] == "  detail: Unexpected stop_reason 'refusal'."


def test_unknown_event_types_are_shown_not_dropped() -> None:
    lines = format_trajectory([{"type": "something_new", "x": 1}])

    assert "something_new" in lines[0] and '"x": 1' in lines[0]


def test_empty_log() -> None:
    assert format_trajectory([]) == ["(empty log)"]


def test_main_prints_a_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.jsonl"
    client = FakeClient(
        [
            tool_use_turn(ToolCall(id="toolu_1", name="echo", input={"text": "hi"})),
            final_turn("echo said hi"),
        ]
    )
    run_agent("Echo hi.", client, build_demo_registry(), on_event=TrajectoryLogger(path))

    code = main([str(path)])

    out = capsys.readouterr().out
    assert code == 0
    assert "RUN START  Echo hi." in out
    assert "STEP 1  tool   echo -> hi" in out
    assert "RUN END  final_answer after 2 steps" in out


def test_main_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(tmp_path / "nope.jsonl")])

    captured = capsys.readouterr()
    assert code == 2
    assert "nope.jsonl" in captured.err
    assert captured.out == ""


def test_main_reports_a_corrupt_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")

    code = main([str(path)])

    assert code == 1
    assert "line 1" in capsys.readouterr().err


def test_main_full_flag_disables_clipping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.jsonl"
    TrajectoryLogger(path)({**TOOL_RESULT, "content": "y" * 500})

    main([str(path), "--full"])

    assert "y" * 500 in capsys.readouterr().out
