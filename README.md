Agent harness for Anthropic Messages API tool-calling experiments.

Run tests with `uv run pytest`.

## Logging and replay

Pass a `TrajectoryLogger` to `run_agent` to record every step of a run as one
JSON line per event:

```python
from harness.trajectory import TrajectoryLogger, new_run_path

run_agent(task, client, registry, on_event=TrajectoryLogger(new_run_path()))
```

Logs go to `runs/` (gitignored, because they can contain task data). Read one
back as a step-by-step account with:

```
uv run replay runs/<file>.jsonl          # long text is clipped
uv run replay runs/<file>.jsonl --full   # show everything
```

## The reconciliation workload

The practice task is bank reconciliation: match a bank statement to the
ledger and explain every difference. The toy case in `data/toy/` has six bank
lines and five ledger entries with a duplicate charge, a timing difference and
an amount mismatch planted in it (`answer_key.json` holds the right answers).

```
uv run python scripts/generate_data.py                              # rewrite data/toy from code
uv run --env-file .env python scripts/run_reconcile.py              # run the agent on it
uv run replay runs/<file>.jsonl                                     # read what it did
```

`run_reconcile.py` needs `ANTHROPIC_API_KEY` and `HARNESS_MODEL` in `.env`
(see `.env.example`). It prints the agent's summary and compares its proposals
with the answer key. The tests never need a key.
