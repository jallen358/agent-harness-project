"""Replay: print a trajectory log as a readable, step-by-step account of a run.

Usage:  uv run replay runs/<file>.jsonl [--full]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from harness.trajectory import read_trajectory

DEFAULT_LIMIT = 200  # characters shown per piece of text before it is clipped


def _clip(text: str, limit: int | None) -> str:
    # Keep every entry on one line, and shorten very long text (such as a CSV a tool returned).
    text = text.replace("\n", "\\n")
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text) - limit} more chars)"


def _tokens(usage: dict[str, Any]) -> str:
    return f"in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}"


def format_trajectory(events: list[dict[str, Any]], limit: int | None = DEFAULT_LIMIT) -> list[str]:
    """Turn logged events into display lines. ``limit=None`` shows text in full."""
    if not events:
        return ["(empty log)"]

    lines: list[str] = []
    for event in events:
        kind = event.get("type")

        if kind == "run_start":
            budget = event.get("max_total_tokens")
            lines.append(f"RUN START  {_clip(str(event.get('task', '')), limit)}")
            lines.append(
                f"  tools: {', '.join(event.get('tool_names', [])) or '(none)'}"
                f"  max_steps: {event.get('max_steps')}"
                f"  budget: {'none' if budget is None else budget}"
            )
            if event.get("system_prompt"):
                lines.append(f"  system: {_clip(event['system_prompt'], limit)}")

        elif kind == "model_turn":
            lines.append(
                f"STEP {event.get('step')}  model  stop_reason={event.get('stop_reason')}"
                f"  tokens {_tokens(event.get('usage', {}))}"
            )
            if event.get("text"):
                lines.append(f"  says: {_clip(event['text'], limit)}")
            for call in event.get("tool_calls", []):
                arguments = json.dumps(call.get("input", {}), ensure_ascii=False)
                lines.append(
                    f"  calls: {call.get('name')} {_clip(arguments, limit)} [{call.get('id')}]"
                )

        elif kind == "tool_result":
            content = _clip(str(event.get("content", "")), limit)
            marker = "ERROR: " if event.get("is_error") else ""
            lines.append(f"STEP {event.get('step')}  tool   {event.get('name')} -> {marker}{content}")

        elif kind == "run_end":
            steps = event.get("steps")
            lines.append(
                f"RUN END  {event.get('stop_reason')} after {steps} step{'' if steps == 1 else 's'}"
                f"  tokens {_tokens(event.get('usage', {}))}"
            )
            if event.get("detail"):
                lines.append(f"  detail: {event['detail']}")

        else:
            # An event type this version doesn't know: show it rather than hide it.
            lines.append(f"?? {kind}: {_clip(json.dumps(event, ensure_ascii=False), limit)}")

    if not any(event.get("type") == "run_end" for event in events):
        lines.append("(log ends without a run_end event: the run crashed or was killed)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay", description="Replay a trajectory log.")
    parser.add_argument("path", type=Path, help="a .jsonl log written by TrajectoryLogger")
    parser.add_argument("--full", action="store_true", help="don't clip long text")
    args = parser.parse_args(argv)

    try:
        events = read_trajectory(args.path)
    except FileNotFoundError:
        print(f"No such log file: {args.path}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for line in format_trajectory(events, limit=None if args.full else DEFAULT_LIMIT):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
