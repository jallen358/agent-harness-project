"""The toy reconciliation workload: its data, its answer key, and the state tools act on.

Bank reconciliation means matching a bank statement against the company's own
records (the ledger) and explaining every difference. It is accounting work with
checkable answers, which makes it a good first task for the agent.

This module knows nothing about the model or the tool registry. The tools in
``harness.reconcile_tools`` are thin wrappers over ``ReconciliationState``.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

Status = Literal[
    "matched", "timing_difference", "amount_mismatch", "duplicate_suspected", "no_match"
]

# Statuses that say "this bank line corresponds to this ledger entry". Only these
# claim the entry, so an entry can't be matched to two bank lines. A suspected
# duplicate points at the same entry as the original without claiming it.
ENTRY_STATUSES: frozenset[str] = frozenset({"matched", "timing_difference", "amount_mismatch"})


@dataclass(frozen=True)
class BankLine:
    id: str
    date: dt.date
    description: str
    amount: Decimal  # money is Decimal, never float


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    date: dt.date
    memo: str
    amount: Decimal


@dataclass(frozen=True)
class Expected:
    """The right answer for one bank line. ``entry_id=None`` means no entry is required."""

    status: Status
    entry_id: str | None


@dataclass(frozen=True)
class Proposal:
    """What the agent decided about one bank line."""

    bank_line_id: str
    entry_id: str | None
    status: Status
    note: str


@dataclass(frozen=True)
class Case:
    name: str
    bank_lines: tuple[BankLine, ...]
    ledger_entries: tuple[LedgerEntry, ...]
    answer_key: dict[str, Expected]  # bank line id -> the right answer


def toy_case() -> Case:
    """The six-line case from the project notes, with one of each planted problem."""

    def bank(id: str, date: str, description: str, amount: str) -> BankLine:
        return BankLine(id, dt.date.fromisoformat(date), description, Decimal(amount))

    def entry(id: str, date: str, memo: str, amount: str) -> LedgerEntry:
        return LedgerEntry(id, dt.date.fromisoformat(date), memo, Decimal(amount))

    return Case(
        name="toy",
        bank_lines=(
            bank("B1", "2025-03-02", "ACH DEPOSIT ACME CORP", "5000.00"),
            bank("B2", "2025-03-03", "CHECK 1041", "-1200.00"),
            bank("B3", "2025-03-05", "STAPLES 0334", "-84.50"),
            bank("B4", "2025-03-05", "STAPLES 0334", "-84.50"),  # repeat of B3
            bank("B5", "2025-03-09", "MONTHLY SERVICE FEE", "-25.00"),  # books say 52.00
            bank("B6", "2025-03-31", "ACH DEPOSIT GLOBEX", "3200.00"),  # books dated Apr 2
        ),
        ledger_entries=(
            entry("L1", "2025-03-01", "Invoice 2201 Acme", "5000.00"),
            entry("L2", "2025-03-02", "Rent check 1041", "-1200.00"),
            entry("L3", "2025-03-05", "Office supplies Staples", "-84.50"),
            entry("L4", "2025-04-02", "Invoice 2208 Globex", "3200.00"),
            entry("L5", "2025-03-09", "Bank fee", "-52.00"),
        ),
        answer_key={
            "B1": Expected("matched", "L1"),
            "B2": Expected("matched", "L2"),
            "B3": Expected("matched", "L3"),
            "B4": Expected("duplicate_suspected", None),
            "B5": Expected("amount_mismatch", "L5"),
            "B6": Expected("timing_difference", "L4"),
        },
    )


# ---- CSV files -------------------------------------------------------------


def write_case(case: Case, directory: Path) -> None:
    """Write ``bank.csv``, ``ledger.csv`` and ``answer_key.json`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "bank.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["bank_line_id", "date", "description", "amount"])
        for line in case.bank_lines:
            writer.writerow([line.id, line.date.isoformat(), line.description, f"{line.amount:.2f}"])

    with (directory / "ledger.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["entry_id", "date", "memo", "amount"])
        for entry in case.ledger_entries:
            writer.writerow([entry.id, entry.date.isoformat(), entry.memo, f"{entry.amount:.2f}"])

    key = {
        line_id: {"status": expected.status, "entry_id": expected.entry_id}
        for line_id, expected in case.answer_key.items()
    }
    (directory / "answer_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")


def load_inputs(directory: Path) -> tuple[list[BankLine], list[LedgerEntry]]:
    """Read ``bank.csv`` and ``ledger.csv`` (not the answer key) from ``directory``."""
    with (directory / "bank.csv").open(newline="", encoding="utf-8") as file:
        bank_lines = [
            BankLine(
                id=row["bank_line_id"],
                date=dt.date.fromisoformat(row["date"]),
                description=row["description"],
                amount=Decimal(row["amount"]),
            )
            for row in csv.DictReader(file)
        ]
    with (directory / "ledger.csv").open(newline="", encoding="utf-8") as file:
        ledger_entries = [
            LedgerEntry(
                id=row["entry_id"],
                date=dt.date.fromisoformat(row["date"]),
                memo=row["memo"],
                amount=Decimal(row["amount"]),
            )
            for row in csv.DictReader(file)
        ]
    return bank_lines, ledger_entries


# ---- the state the agent works on ------------------------------------------


class ReconciliationState:
    """The bank lines, the ledger, and the proposals made so far."""

    def __init__(self, bank_lines: list[BankLine], ledger_entries: list[LedgerEntry]) -> None:
        self.bank_lines = bank_lines
        self.ledger_entries = ledger_entries
        self._bank_by_id = {line.id: line for line in bank_lines}
        self._entry_by_id = {entry.id: entry for entry in ledger_entries}
        self._proposals: dict[str, Proposal] = {}

    @property
    def proposals(self) -> dict[str, Proposal]:
        return dict(self._proposals)

    def bank_line(self, bank_line_id: str) -> BankLine | None:
        return self._bank_by_id.get(bank_line_id)

    def ledger_entry(self, entry_id: str) -> LedgerEntry | None:
        return self._entry_by_id.get(entry_id)

    def unproposed(self) -> list[BankLine]:
        """Bank lines that have no proposal yet, in statement order."""
        return [line for line in self.bank_lines if line.id not in self._proposals]

    def owner_of_entry(self, entry_id: str) -> str | None:
        """The bank line this ledger entry is matched to, if any."""
        for proposal in self._proposals.values():
            if proposal.entry_id == entry_id and proposal.status in ENTRY_STATUSES:
                return proposal.bank_line_id
        return None

    def record(self, proposal: Proposal) -> Proposal | None:
        """Save a proposal, replacing any earlier one. Returns the one it replaced."""
        previous = self._proposals.get(proposal.bank_line_id)
        self._proposals[proposal.bank_line_id] = proposal
        return previous
