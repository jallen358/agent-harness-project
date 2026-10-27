"""Run the reconciliation agent on a case, using the real model.

Needs ANTHROPIC_API_KEY and HARNESS_MODEL in the environment. Usage:

    uv run --env-file .env python scripts/run_reconcile.py [--data data/toy] [--max-steps 25]

Every step is logged to runs/<timestamp>-<id>.jsonl; read it back with
`uv run replay <that file>`.
"""

import argparse
import json
from pathlib import Path

from harness.loop import RunConfig, run_agent
from harness.model import AnthropicClient
from harness.reconcile_tools import SYSTEM_PROMPT, TASK, build_reconciliation_registry
from harness.reconciliation import ReconciliationState, load_inputs
from harness.trajectory import TrajectoryLogger, new_run_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/toy"))
    parser.add_argument("--max-steps", type=int, default=25)
    args = parser.parse_args()

    state = ReconciliationState(*load_inputs(args.data))
    registry = build_reconciliation_registry(state)
    client = AnthropicClient.from_env()
    log_path = new_run_path()

    result = run_agent(
        TASK,
        client,
        registry,
        RunConfig(max_steps=args.max_steps, system_prompt=SYSTEM_PROMPT),
        on_event=TrajectoryLogger(log_path),
    )

    print(f"Stopped: {result.stop_reason} after {result.steps} steps")
    print(f"Tokens: in={result.usage.input_tokens} out={result.usage.output_tokens}")
    print(f"Log: {log_path}   (replay with: uv run replay {log_path})")
    if result.detail:
        print(f"Detail: {result.detail}")
    print(f"\nAgent's summary:\n{result.final_text}")

    # A quick side-by-side with the answer key. Step 7 replaces this with a real grader.
    key = json.loads((args.data / "answer_key.json").read_text())
    print("\nProposals vs answer key:")
    for bank_line in state.bank_lines:
        expected = key[bank_line.id]
        proposal = state.proposals.get(bank_line.id)
        got = f"{proposal.status} -> {proposal.entry_id}" if proposal else "(no proposal)"
        want = f"{expected['status']} -> {expected['entry_id']}"
        # entry_id null in the key means "any entry is acceptable", so only compare the status.
        same_status = proposal is not None and proposal.status == expected["status"]
        same_entry = expected["entry_id"] is None or (
            proposal is not None and proposal.entry_id == expected["entry_id"]
        )
        mark = "ok  " if same_status and same_entry else "DIFF"
        print(f"  {mark} {bank_line.id}: got {got:<32} want {want}")
    return 0 if result.stop_reason == "final_answer" else 1


if __name__ == "__main__":
    raise SystemExit(main())
