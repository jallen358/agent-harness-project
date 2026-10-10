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

## Loop (`src/harness/loop.py`): Planned

### Branch on `stop_reason`
- `end_turn` / `stop_sequence`: stop with `final_answer`.
- `tool_use`: run the tools and loop again.
- `max_tokens`: stop with `max_tokens`. **Do not run its tool calls**, because
  the output was cut off and a call may be incomplete.
- Anything else (`refusal`, `pause_turn`, ...): stop with `unexpected`, keeping
  the raw reason.

### All tool results from one turn go in one user message
- **Why:** splitting them across messages teaches the model to stop making
  parallel calls. Failures use `is_error: true`.

### Token budget counts total usage across steps
- **Chose:** sum `input_tokens + output_tokens` over every step.
- **Why:** each call resends the full history, so this is the real spend, not
  just the final conversation size.

### Hitting `max_steps` leaves a dangling `tool_use`
- **Consequence:** the final assistant message has a tool call with no result,
  so that run cannot simply be resumed. Resuming belongs to Week 2.

### Logging goes through an `on_event` callback
- **Why:** the loop stays free of logging code, and Step 4 plugs the
  trajectory logger in without rewriting the loop.

### Default `max_steps` is 10
- **Why:** arbitrary. Revisit after the first real runs.

## Open

- **Python version.** `.python-version` and `requires-python` say 3.9. Pydantic
  fails on `str | None` there (`TypeError: unsupported operand type(s) for |`),
  and the notes' `ProposeMatch.entry_id: str | None` needs it. Recommended:
  bump to 3.12. Alternative: `Optional[str]` throughout.
- **Thinking blocks.** Newer models may return `thinking` blocks by default,
  and `turn_from_response` would raise on them. Options: ignore them, or turn
  thinking off in the request. Unconfirmed until the first real run.
- **`scripts/raw_call.py` still imports `anthropic`.** This breaks the
  "only `model.py`" rule. Switch it to `AnthropicClient`.
- **Module name.** The notes plan `client.py`; the module is `model.py`.
  Rename it or update the notes.
- **Branches.** PR #1 merged into `step-1`. `main` was fast-forwarded to the
  mypy fix, and `step-1` is still GitHub's default branch. Decide where
  feature branches start from, and change the default branch in GitHub's
  settings.
- **From the notes, still undecided:** whether `find_candidate_entries` returns
  only exact amount matches or near ones too; what error messages say in the
  reconciliation tools; one `propose_match` with five statuses versus a tool
  per outcome; a coarse `auto_reconcile` tool as a Week 2 comparison.
