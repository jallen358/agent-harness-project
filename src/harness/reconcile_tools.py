"""The reconciliation tools the agent uses, plus the prompts that frame the task.

Three tools, each a thin wrapper over ``ReconciliationState``:

- ``list_unmatched_bank_lines``: what is left to do.
- ``find_candidate_entries``: which ledger entries could belong to a bank line.
- ``propose_match``: record the decision for one bank line.

Errors are written for the model to read: they say what was wrong and list the
valid options, so the agent can correct itself and carry on.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from harness.reconciliation import (
    ENTRY_STATUSES,
    Proposal,
    ReconciliationState,
    Status,
)
from harness.tools import ToolError, ToolRegistry

SYSTEM_PROMPT = """\
You are a bookkeeping assistant reconciling a bank statement against a general ledger.

Work through every bank line. For each one, find the ledger entry it corresponds to \
(if any) and record exactly one proposal with propose_match. Use the tools to look at \
the data; do not guess amounts or dates.

Statuses:
- matched: the same transaction, with the same amount and close dates.
- timing_difference: the same transaction and the same amount, but the bank and the \
books dated it on opposite sides of a period end.
- amount_mismatch: the same transaction, but the amounts differ. Say by how much in \
the note.
- duplicate_suspected: the bank line repeats another bank line that is already matched \
to the only ledger entry for that transaction.
- no_match: no ledger entry corresponds to the bank line.

When every bank line has a proposal, reply with a short summary: how many of each \
status, and anything a person should look at.
"""

TASK = (
    "Reconcile the March 2025 bank statement against the ledger. "
    "Record a proposal for every bank line, then summarize what you found."
)


class ListUnmatchedBankLines(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=50, ge=1, le=500, description="Most bank lines to show.")


class FindCandidateEntries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bank_line_id: str = Field(description="The bank line to find ledger entries for, e.g. 'B3'.")
    date_window_days: int = Field(
        default=5,
        ge=0,
        le=366,
        description="How many days either side of the bank line's date to search.",
    )


class ProposeMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bank_line_id: str = Field(description="The bank line this decision is about, e.g. 'B3'.")
    entry_id: str | None = Field(
        default=None,
        description=(
            "The ledger entry it corresponds to, e.g. 'L3'. Required for matched, "
            "timing_difference and amount_mismatch. Must be null for no_match."
        ),
    )
    status: Status = Field(
        description=(
            "matched: same transaction, same amount, close dates. "
            "timing_difference: same transaction and amount, dated on opposite sides of a "
            "period end. "
            "amount_mismatch: same transaction, different amounts. "
            "duplicate_suspected: repeats another bank line already matched to the entry. "
            "no_match: no ledger entry fits."
        )
    )
    note: str = Field(
        min_length=1, description="Why. For a mismatch, say by how much the amounts differ."
    )


def _count(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {plural or noun + 's'}"


def _days(n: int) -> str:
    return _count(n, "day")


def _unknown_bank_line(state: ReconciliationState, bank_line_id: str) -> ToolError:
    valid = ", ".join(line.id for line in state.bank_lines)
    return ToolError(f"No bank line {bank_line_id!r}. Valid IDs: {valid}.")


def _unknown_entry(state: ReconciliationState, entry_id: str) -> ToolError:
    valid = ", ".join(entry.id for entry in state.ledger_entries)
    return ToolError(f"No ledger entry {entry_id!r}. Valid IDs: {valid}.")


def build_reconciliation_registry(state: ReconciliationState) -> ToolRegistry:
    """A registry whose tools read and write ``state``."""
    registry = ToolRegistry()

    @registry.register(
        "list_unmatched_bank_lines",
        "List the bank lines that have no proposal yet, in statement order. "
        "Each line shows its ID, date, description and amount.",
        ListUnmatchedBankLines,
    )
    def list_unmatched_bank_lines(args: ListUnmatchedBankLines) -> str:
        pending = state.unproposed()
        if not pending:
            total = len(state.bank_lines)
            return f"Every bank line has a proposal ({total} of {total})."

        verb = "has" if len(pending) == 1 else "have"
        header = f"{_count(len(pending), 'bank line')} {verb} no proposal yet"
        if len(pending) > args.limit:
            header += f" (showing the first {args.limit})"
        rows = [
            f"{line.id} | {line.date} | {line.description} | {line.amount:.2f}"
            for line in pending[: args.limit]
        ]
        return "\n".join([header + ":", *rows])

    @registry.register(
        "find_candidate_entries",
        "Find ledger entries dated near a bank line, best amount match first. Entries "
        "whose amount differs are included on purpose, because a near match may be a real "
        "discrepancy. Shows when an entry is already matched to another bank line.",
        FindCandidateEntries,
    )
    def find_candidate_entries(args: FindCandidateEntries) -> str:
        line = state.bank_line(args.bank_line_id)
        if line is None:
            raise _unknown_bank_line(state, args.bank_line_id)

        window = args.date_window_days
        nearby = [
            entry
            for entry in state.ledger_entries
            if abs((entry.date - line.date).days) <= window
        ]
        if not nearby:
            return (
                f"No ledger entries within {_days(window)} of {line.id} ({line.date}). "
                "Try a larger date_window_days."
            )

        # Exact amounts first, then the closest amounts, then the closest dates.
        nearby.sort(key=lambda e: (abs(e.amount - line.amount), abs((e.date - line.date).days)))

        rows = []
        for entry in nearby:
            offset = (entry.date - line.date).days
            if offset == 0:
                when = "same day"
            else:
                when = f"{_days(abs(offset))} {'after' if offset > 0 else 'before'}"

            difference = abs(entry.amount - line.amount)
            amount_note = "amount matches" if difference == 0 else f"amount differs by {difference:.2f}"

            row = f"{entry.id} | {entry.date} | {entry.memo} | {entry.amount:.2f} | {amount_note} | {when}"
            owner = state.owner_of_entry(entry.id)
            if owner == line.id:
                row += " | already proposed for this line"
            elif owner is not None:
                row += f" | already matched to {owner}"
            rows.append(row)

        header = (
            f"{_count(len(nearby), 'ledger entry', 'ledger entries')} "
            f"within {_days(window)} of {line.id} ({line.date}, {line.amount:.2f}), "
            "best amount match first:"
        )
        return "\n".join([header, *rows])

    @registry.register(
        "propose_match",
        "Record your decision for one bank line. Use status matched, timing_difference or "
        "amount_mismatch with the ledger entry's ID; duplicate_suspected when the line "
        "repeats another bank line; no_match (with entry_id null) when nothing fits. "
        "Proposing again for the same bank line replaces the earlier proposal.",
        ProposeMatch,
    )
    def propose_match(args: ProposeMatch) -> str:
        if state.bank_line(args.bank_line_id) is None:
            raise _unknown_bank_line(state, args.bank_line_id)

        if args.status in ENTRY_STATUSES and args.entry_id is None:
            raise ToolError(
                f"Status {args.status!r} needs an entry_id. If no ledger entry fits, "
                "use status 'no_match' with entry_id null."
            )
        if args.status == "no_match" and args.entry_id is not None:
            raise ToolError(
                "Status 'no_match' means no ledger entry fits, so entry_id must be null. "
                "To link an entry, use 'matched', 'timing_difference' or 'amount_mismatch'."
            )

        if args.entry_id is not None:
            if state.ledger_entry(args.entry_id) is None:
                raise _unknown_entry(state, args.entry_id)
            owner = state.owner_of_entry(args.entry_id)
            if args.status in ENTRY_STATUSES and owner is not None and owner != args.bank_line_id:
                raise ToolError(
                    f"Ledger entry {args.entry_id} is already matched to {owner}. Each ledger "
                    f"entry can match only one bank line. If {args.bank_line_id} repeats "
                    f"{owner}'s transaction, propose status 'duplicate_suspected' for it instead."
                )

        previous = state.record(
            Proposal(args.bank_line_id, args.entry_id, args.status, args.note)
        )

        target = args.entry_id if args.entry_id is not None else "no entry"
        message = f"Recorded {args.bank_line_id} -> {target} ({args.status})."
        if previous is not None:
            message = (
                f"Replaced the earlier proposal for {args.bank_line_id} (was {previous.status}). "
                + message
            )

        remaining = len(state.unproposed())
        if remaining == 0:
            message += " All bank lines now have a proposal."
        else:
            verb = "has" if remaining == 1 else "have"
            message += f" {_count(remaining, 'bank line')} still {verb} no proposal."
        return message

    return registry
