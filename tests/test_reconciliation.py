from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

from harness.reconciliation import (
    Expected,
    Proposal,
    ReconciliationState,
    load_inputs,
    toy_case,
    write_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_toy_case_matches_the_project_notes() -> None:
    case = toy_case()

    assert [line.id for line in case.bank_lines] == ["B1", "B2", "B3", "B4", "B5", "B6"]
    assert [entry.id for entry in case.ledger_entries] == ["L1", "L2", "L3", "L4", "L5"]
    assert case.bank_lines[2] == case.bank_lines[3].__class__(
        id="B3", date=dt.date(2025, 3, 5), description="STAPLES 0334", amount=Decimal("-84.50")
    )
    assert case.answer_key == {
        "B1": Expected("matched", "L1"),
        "B2": Expected("matched", "L2"),
        "B3": Expected("matched", "L3"),
        "B4": Expected("duplicate_suspected", None),
        "B5": Expected("amount_mismatch", "L5"),
        "B6": Expected("timing_difference", "L4"),
    }


def test_csv_files_use_the_documented_columns(tmp_path: Path) -> None:
    write_case(toy_case(), tmp_path)

    bank = (tmp_path / "bank.csv").read_text().splitlines()
    ledger = (tmp_path / "ledger.csv").read_text().splitlines()
    assert bank[0] == "bank_line_id,date,description,amount"
    assert bank[1] == "B1,2025-03-02,ACH DEPOSIT ACME CORP,5000.00"
    assert ledger[0] == "entry_id,date,memo,amount"
    assert ledger[5] == "L5,2025-03-09,Bank fee,-52.00"


def test_csv_round_trip_is_lossless(tmp_path: Path) -> None:
    case = toy_case()
    write_case(case, tmp_path)

    bank_lines, ledger_entries = load_inputs(tmp_path)

    assert bank_lines == list(case.bank_lines)
    assert ledger_entries == list(case.ledger_entries)


def test_committed_data_matches_the_generator(tmp_path: Path) -> None:
    # data/toy is checked in so the repo shows the data; this keeps it from drifting.
    write_case(toy_case(), tmp_path)

    for name in ("bank.csv", "ledger.csv", "answer_key.json"):
        committed = (REPO_ROOT / "data" / "toy" / name).read_text()
        assert committed == (tmp_path / name).read_text(), name


def test_state_starts_with_no_proposals() -> None:
    state = ReconciliationState(*load_inputs(REPO_ROOT / "data" / "toy"))

    assert [line.id for line in state.unproposed()] == ["B1", "B2", "B3", "B4", "B5", "B6"]
    assert state.proposals == {}


def test_proposing_removes_a_line_from_the_unproposed_list() -> None:
    state = ReconciliationState(*load_inputs(REPO_ROOT / "data" / "toy"))

    state.record(Proposal("B1", "L1", "matched", "same amount"))

    assert "B1" not in [line.id for line in state.unproposed()]
    assert state.proposals["B1"].entry_id == "L1"
    assert state.owner_of_entry("L1") == "B1"
    assert state.owner_of_entry("L2") is None


def test_a_duplicate_does_not_claim_the_entry() -> None:
    state = ReconciliationState(*load_inputs(REPO_ROOT / "data" / "toy"))
    state.record(Proposal("B3", "L3", "matched", "ok"))

    state.record(Proposal("B4", "L3", "duplicate_suspected", "repeat of B3"))

    assert state.owner_of_entry("L3") == "B3"
