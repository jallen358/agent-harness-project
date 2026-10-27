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
