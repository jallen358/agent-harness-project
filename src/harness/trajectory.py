"""Trajectory log: a record of everything that happened during a run.

The loop reports each step through its ``on_event`` callback. ``TrajectoryLogger``
is a callback that writes each event to a file as one line of JSON (JSONL).
``read_trajectory`` loads such a file back; ``harness.replay`` prints it.

One JSON object per line means the file is append-only, survives a crash up to
the last event, and can be searched with ordinary tools such as ``grep``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TrajectoryLogger:
    """An ``on_event`` callback that appends each event to a JSONL file."""

    def __init__(self, path: Path, clock: Clock = _utc_now) -> None:
        self.path = path
        self._clock = clock  # injectable so tests get predictable timestamps

    def __call__(self, event: dict[str, Any]) -> None:
        # The logger stamps the time, not the loop, so the loop stays deterministic.
        record = {"ts": self._clock().isoformat(timespec="milliseconds"), **event}
        # Serialize first. An event that can't be written raises instead of being
        # silently dropped: a log with holes is worse than a run that stops.
        line = json.dumps(record, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open, write and close on every event, so everything logged so far is on
        # disk even if the run crashes on the next step.
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def new_run_path(runs_dir: Path = Path("runs"), clock: Clock = _utc_now) -> Path:
    """A fresh log path such as ``runs/20261027T153012Z-3fa9c1.jsonl``."""
    stamp = clock().strftime("%Y%m%dT%H%M%SZ")
    # The random suffix keeps two runs that start in the same second apart.
    return runs_dir / f"{stamp}-{uuid.uuid4().hex[:6]}.jsonl"


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    """Load a log file. Raises ``ValueError`` naming the line if one is not a JSON object."""
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {number} is not valid JSON ({exc.msg}).") from exc
            if not isinstance(event, dict):
                # ValueError, not TypeError: this is bad file contents, which replay reports.
                raise ValueError(  # noqa: TRY004
                    f"{path}: line {number} is not a JSON object."
                )
            events.append(event)
    return events
