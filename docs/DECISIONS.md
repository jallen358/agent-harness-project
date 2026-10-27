# Design decisions

Each entry says what was chosen, what was rejected, and why. Entries marked
**Planned** are decided but not built yet. Entries under **Open** are not
decided.

## Model client (`src/harness/model.py`)

### Only `model.py` imports `anthropic`
- **Chose:** one module owns the SDK. The rest of the code sees `ToolCall`,
  `Usage`, `ModelTurn` and the `ModelClient` protocol.
- **Rejected:** passing SDK response objects through the loop.
- **Why:** swapping in a fake model for tests, or changing SDK versions, then
  touches one file. `FakeClient` depends on this.

### Plain dicts for content blocks
- **Chose:** `as_assistant_message()` returns `{"role": "assistant", "content": [...]}`
  with plain dict blocks.
- **Rejected:** keeping the SDK's block objects.
- **Why:** the loop can append the message directly and tests can compare it
  with `==`.

### An empty text block is omitted
- **Chose:** a turn with no text (a tool-only turn) emits no `text` block.
- **Why:** the API rejects empty text blocks.

### Multiple text blocks are joined with no separator
- **Chose:** `"".join(...)`, in order.
- **Rejected:** keeping a list of text blocks, or joining with newlines.
- **Why:** `ModelTurn.text` stays a single string. Blocks from one turn are
  continuous prose, so inserting separators would change the text. Revisit if
  a real response shows otherwise.

### Unknown content block types raise `ValueError`
- **Chose:** `turn_from_response` raises, naming the block type.
- **Rejected:** silently skipping unknown blocks.
- **Why:** a dropped block would corrupt the conversation history without any
  signal. See Open: thinking blocks.

### `from_env` requires `HARNESS_MODEL`
- **Chose:** an unset or empty `HARNESS_MODEL` raises `ValueError("HARNESS_MODEL is not set")`.
- **Why:** `.env.example` ships the variable empty, so "empty" is a likely
  state. Failing here is clearer than an API error about an empty model name.

### The SDK client is built in the constructor
- **Chose:** `AnthropicClient.__init__` creates `anthropic.Anthropic()`; the
  SDK reads `ANTHROPIC_API_KEY` itself.
- **Why:** matches the project notes, and the API key never appears in our code.
- **Cost:** constructing the client can raise if no credentials are found.
  Tests set a dummy key.

### Request fields are only sent when set
- **Chose:** `tools` is omitted when empty and `system` is omitted when `None`.
- **Why:** passing `system=None` through is not the same as omitting it.

## Tool registry (`src/harness/tools.py`)

Recorded from the code as written. Rationale is inferred from the module
docstring and the project notes.

### Each tool is defined once
- **Chose:** a tool is a name, a description, a pydantic `args_model` and a
  handler. `definitions()` (the menu) and `run()` (the kitchen) both read
  that one entry.
- **Rejected:** a hand-written schema list plus an `if name == ...` chain.
- **Why:** the two copies drift apart, and every new tool edits both.

### `run()` never raises
- **Chose:** unknown tool, failed validation and handler exceptions all return
  `ToolResult(..., is_error=True)`.
- **Why:** a crash ends the run. An error the model can read lets it correct
  itself and continue.

### Error messages are written for the model
- **Unknown tool:** names the bad tool and lists the valid ones.
- **Validation failure:** names each bad field and its problem, then says to
  fix the arguments and try again.
- **Handler exception:** reports the exception type and message.
- **Why:** "No bank line B99. Valid IDs: B1-B6" helps recovery; "Error 500" does not.

### Tool names are validated at registration
- **Chose:** names must match `[a-zA-Z0-9_-]{1,128}`, and a duplicate name raises
  `ValueError`.
- **Why:** a bad or duplicate name should fail at startup, not mid-run.

## FakeClient (`src/harness/fake_client.py`)

### Lives in `src/`, not `tests/`
- **Why:** the Step 7 eval will reuse it.

### Records a deep copy of each call's messages
- **Chose:** `copy.deepcopy(messages)` at call time.
- **Rejected:** storing the list itself.
- **Why:** the loop keeps appending to one list, so every recorded call would
  otherwise show the final conversation.

### Running out of script raises an error
- **Chose:** `FakeClientExhaustedError`, saying how many turns the script had
  and which one was asked for.
- **Rejected:** returning a default final answer.
- **Why:** a default would hide a loop that fails to stop.

### `assert_finished()` for the opposite failure
- **Why:** a run that stops before using every scripted turn is usually a bug
  in the loop or in the test.

## Loop (`src/harness/loop.py`)

### Branch on `stop_reason`
- `end_turn` / `stop_sequence`: stop with `final_answer`.
- `tool_use`: run the tools and loop again.
- `max_tokens`: stop with `max_tokens`. **Do not run its tool calls**, because
  the output was cut off and a call may be incomplete.
- Anything else (`refusal`, `pause_turn`, ...), or `tool_use` with no tool
  calls: stop with `unexpected`, with the reason in `RunResult.detail`.

### All tool results from one turn go in one user message
- **Why:** splitting them across messages teaches the model to stop making
  parallel calls. Failures carry `is_error: true`; successes omit the key.

### Token budget counts total usage across steps
- **Chose:** sum `input_tokens + output_tokens` over every step.
- **Why:** each call resends the full history, so this is the real spend, not
  just the final conversation size.
- **Chose:** the budget is checked *after* the tools run, and a final answer
  wins over the budget.
- **Why:** the history stays well formed (every tool request has its result),
  and an answer already in hand is not thrown away.
- **Cost:** tools with side effects still run on the step that exhausts the
  budget.

### `max_steps` and `budget` stops leave a well-formed history
- **Chose:** the check happens after the step's tool results are appended, so
  the history ends with a user `tool_result` message.
- **Why:** the run could be resumed by calling the model again. Resuming
  itself belongs to Week 2.
- (An earlier draft of this file said the history would end in a dangling
  `tool_use`. That is not how the loop was built.)

### `final_text` is `None` unless the model said something final
- **Chose:** `final_answer`, `max_tokens` and `unexpected` carry the model's
  text; `max_steps` and `budget` leave it `None`.
- **Why:** after those two stops the model was mid-task, so its last text is
  not an answer and must not be mistaken for one.

### Logging goes through an `on_event` callback
- **Chose:** four event types: `run_start`, `model_turn`, `tool_result`,
  `run_end`. Every exit goes through one `finish()` so each run emits exactly
  one `run_end`.
- **Why:** the loop stays free of logging code, and Step 4 plugs the
  trajectory logger in without rewriting the loop.

### `registry` comes before `config` in `run_agent`
- **Chose:** `run_agent(task, client, registry, config=None, on_event=None)`.
- **Why:** parameters with defaults must come last. `config=None` instead of a
  `RunConfig()` default avoids a shared default object.

### Default `max_steps` is 10
- **Why:** arbitrary. Revisit after the first real runs.

## Trajectory log (`src/harness/trajectory.py`, `src/harness/replay.py`)

### One JSON line per event (JSONL)
- **Chose:** each event is one line in `runs/<timestamp>-<id>.jsonl`.
- **Rejected:** one JSON document per run, written at the end.
- **Why:** append-only, so a run that crashes still leaves everything up to its
  last event, and the file can be searched with `grep`. A single document would
  be lost with the run.

### The log is complete; replay does the clipping
- **Chose:** the file keeps full text and tool output. `replay` clips long text
  to 200 characters unless `--full` is given.
- **Why:** the notes warn against logging only the final answer. Shortening at
  write time would destroy the detail needed to diagnose a bad run.
- **Cost:** logs can be large, and they can contain task data (for this
  project, bank and ledger lines). `runs/` is gitignored.

### Every event is written to disk immediately
- **Chose:** open, append and close on each event.
- **Rejected:** keeping one file handle open for the whole run.
- **Why:** nothing buffered is lost on a crash, and there is no handle to close
  or leak. Volume is a few events per step, so the cost is negligible.

### The logger adds the timestamp, not the loop
- **Why:** the loop stays deterministic and testable. The clock is injectable.

### A log failure stops the run
- **Chose:** an event that cannot be serialized raises.
- **Rejected:** `default=str`, or swallowing the error.
- **Why:** a log with silent holes is worse than a run that stops loudly.

### `run_start` and `run_end` carry the setup and the outcome
- **Chose:** `run_start` records the system prompt, tool names and limits.
  `run_end` records `final_text` and total usage.
- **Why:** the notes' question is "why did this run go wrong?" That needs to know
  which tools and limits were in force, and how the run ended.

### Replay shows what it does not understand
- **Chose:** unknown event types are printed as `??`, and a log with no
  `run_end` is flagged as crashed or killed.
- **Why:** silently skipping events would hide exactly the cases replay exists
  for.

### `replay` is a console script
- **Chose:** `[project.scripts] replay = "harness.replay:main"`, so
  `uv run replay <log>` works. `python -m harness.replay` also works.

## Reconciliation workload (`src/harness/reconciliation.py`, `src/harness/reconcile_tools.py`)

### Domain and tools are separate modules
- **Chose:** `reconciliation.py` holds the data, answer key, CSV reading/writing
  and `ReconciliationState`. `reconcile_tools.py` holds the pydantic argument
  models, the prompts and the three tools, which are thin wrappers over the state.
- **Why:** the Week 2 coarse `auto_reconcile` tool and the Step 7 grader can
  reuse the domain code without the tools.

### Money is `Decimal`
- **Why:** float arithmetic on currency produces rounding errors, and the
  agent's whole job is spotting small differences.

### The data is generated from code, checked in, and tested for drift
- **Chose:** `toy_case()` is the source of truth. `data/toy/*.csv` and
  `answer_key.json` are committed, and a test fails if they stop matching the
  generator.
- **Why:** the repo shows what the agent sees, and the generator will make the
  10 eval cases in Step 7.

### `find_candidate_entries` returns near amounts too (resolves a notes question)
- **Chose:** every ledger entry within the date window, ranked by amount
  closeness and then date, each marked "amount matches" or "amount differs by X".
- **Rejected:** exact-amount matches only.
- **Why:** with exact-only the agent could never discover the 25.00 vs 52.00
  mismatch; it would just see "no candidates".
- **Also shown:** whether a candidate is already matched to another bank line.
  That is how the agent finds the duplicate: the repeat charge's only exact
  candidate is already taken.

### `ToolError` carries messages to the model verbatim
- **Chose:** `tools.ToolError`. `run()` returns its message unchanged with
  `is_error=True`. Any other exception is still wrapped with the tool name and
  exception type.
- **Rejected:** letting handlers raise `ValueError`.
- **Why:** "Tool 'propose_match' failed: ValueError: ..." buries the message.
  The notes' goal is "No bank line B99. Valid IDs: B1-B6".

### Error messages name the problem and the valid options (resolves a notes question)
- Unknown IDs list every valid ID. A status that needs an entry says to use
  `no_match` if nothing fits. Matching an entry that is already taken names the
  bank line that owns it and suggests `duplicate_suspected`.

### One `propose_match` with five statuses (resolves a notes question)
- **Chose:** one tool; the status is an enum whose meanings are in the schema.
- **Rejected:** a separate tool per outcome.
- **Why:** one place to validate and one record shape. **Untested hypothesis:**
  this was decided before any real run. The Week 2 comparison against a coarse
  `auto_reconcile` tool, and the first logs, may overturn it.

### An entry can be matched once; a duplicate does not claim it
- **Chose:** `matched`, `timing_difference` and `amount_mismatch` claim their
  ledger entry; `duplicate_suspected` does not, and its `entry_id` is optional.
- **Why:** two bank lines matched to one entry is the error being hunted. The
  answer key leaves the duplicate's entry unspecified.

### Re-proposing replaces the earlier proposal
- **Chose:** the reply says "Replaced the earlier proposal for B1 (was no_match)."
- **Why:** lets the agent correct itself. Making this idempotent and
  checkpointed is Week 2.

### `entry_id` defaults to null and `note` is required
- **Chose:** omitting `entry_id` means null (the notes' model had it required).
  `note` must be non-empty.
- **Why:** the model should not have to write `null` for a no-match, and a
  required note puts the agent's reasoning in the log.

### The system prompt defines the statuses but not the planted errors
- **Why:** it is generic guidance the agent would get on any case. Hints about
  this case's specific problems would make the eval meaningless.

### `run_reconcile.py` prints a side-by-side with the answer key
- **Why:** to read a first run quickly. Step 7 replaces it with a real grader.

## Project

### Python 3.12
- **Chose:** `.python-version` is 3.12 and `requires-python` is `>=3.12`.
- **Rejected:** staying on 3.9 and writing `Optional[str]` in the pydantic models.
- **Why:** pydantic evaluates annotations at runtime, and `str | None` raises
  `TypeError` on 3.9. The tool argument models in the project notes use it
  (`ProposeMatch.entry_id`).
- **Side effect:** `uv.lock` shrank because the old lock carried a second,
  older resolution for Python 3.9. No package moved to a version that 3.12 was
  not already using (pytest 9.1.1, pydantic 2.14.0, anthropic 1.12.1).

## Open

- **API errors leave a log without `run_end`.** The loop does not catch
  exceptions from `client.step`, so a network or API failure ends the run with
  no closing event. Replay flags this, but the cause is not logged. Candidate
  for Step 5: catch it in the loop, emit a `run_error` event, then re-raise.
- **Thinking blocks.** Newer models may return `thinking` blocks by default,
  and `turn_from_response` would raise on them. Options: ignore them, or turn
  thinking off in the request. Unconfirmed until the first real run.
- **`scripts/raw_call.py` still imports `anthropic`.** This breaks the
  "only `model.py`" rule. Switch it to `AnthropicClient`.
- **Module name.** The notes plan `client.py`; the module is `model.py`.
  Rename it or update the notes.
- **Stacked PRs.** `main` is the default branch. Each step lives on its own
  branch stacked on the previous one (`claude/optimistic-fermat-1t62q8` for
  Steps 3 and the Python bump, `step-4-trajectory-logging`,
  `step-5-reconciliation-workload`). As each PR merges, retarget the next to
  `main`; after a squash merge, rebase the next branch first.
- **The first real run has not happened.** Everything so far is tested against
  `FakeClient`. `scripts/run_reconcile.py` is ready, and the first log will show
  how the tool descriptions and error messages actually land.
- **Coarse tool comparison (Week 2).** Build an `auto_reconcile` tool next to
  the three and compare.
