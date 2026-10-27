from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.fake_client import FakeClient, final_turn, tool_use_turn
from harness.loop import RunConfig, run_agent
from harness.model import ToolCall
from harness.reconcile_tools import SYSTEM_PROMPT, TASK, build_reconciliation_registry
from harness.reconciliation import ReconciliationState, load_inputs, toy_case
from harness.tools import ToolRegistry

DATA = Path(__file__).resolve().parents[1] / "data" / "toy"


def setup() -> tuple[ToolRegistry, ReconciliationState]:
    state = ReconciliationState(*load_inputs(DATA))
    return build_reconciliation_registry(state), state


def propose(
    registry: ToolRegistry,
    bank_line_id: str,
    entry_id: str | None,
    status: str = "matched",
    note: str = "checked",
) -> Any:
    return registry.run(
        "propose_match",
        {"bank_line_id": bank_line_id, "entry_id": entry_id, "status": status, "note": note},
    )


# ---- the menu ------------------------------------------------------------


def test_menu_lists_the_three_tools() -> None:
    registry, _ = setup()

    definitions = {d["name"]: d for d in registry.definitions()}

    assert set(definitions) == {"list_unmatched_bank_lines", "find_candidate_entries", "propose_match"}
    assert all(d["description"] for d in definitions.values())


def test_menu_spells_out_the_allowed_statuses() -> None:
    registry, _ = setup()
    schema = next(d for d in registry.definitions() if d["name"] == "propose_match")["input_schema"]

    assert set(schema["properties"]["status"]["enum"]) == {
        "matched",
        "timing_difference",
        "amount_mismatch",
        "duplicate_suspected",
        "no_match",
    }
    assert set(schema["required"]) == {"bank_line_id", "status", "note"}


# ---- list_unmatched_bank_lines ------------------------------------------


def test_list_shows_every_line_at_first() -> None:
    registry, _ = setup()

    result = registry.run("list_unmatched_bank_lines", {})

    assert not result.is_error
    assert "6 bank lines have no proposal yet" in result.content
    assert "B1 | 2025-03-02 | ACH DEPOSIT ACME CORP | 5000.00" in result.content
    assert "B6 | 2025-03-31 | ACH DEPOSIT GLOBEX | 3200.00" in result.content


def test_list_respects_limit_and_says_so() -> None:
    registry, _ = setup()

    result = registry.run("list_unmatched_bank_lines", {"limit": 2})

    assert "showing the first 2" in result.content
    assert "B2" in result.content and "B3" not in result.content


def test_list_drops_lines_once_proposed() -> None:
    registry, _ = setup()
    propose(registry, "B1", "L1")

    content = registry.run("list_unmatched_bank_lines", {}).content

    assert "5 bank lines have no proposal yet" in content
    assert "B1 |" not in content


def test_list_says_when_nothing_is_left() -> None:
    registry, state = setup()
    for line in state.unproposed():
        propose(registry, line.id, None, "no_match")

    content = registry.run("list_unmatched_bank_lines", {}).content

    assert "Every bank line has a proposal" in content


def test_list_rejects_a_zero_limit() -> None:
    registry, _ = setup()

    result = registry.run("list_unmatched_bank_lines", {"limit": 0})

    assert result.is_error and "limit" in result.content


# ---- find_candidate_entries ----------------------------------------------


def test_candidates_put_the_exact_amount_first() -> None:
    registry, _ = setup()

    content = registry.run("find_candidate_entries", {"bank_line_id": "B3"}).content

    first = next(line for line in content.splitlines() if line.startswith("L"))
    assert first.startswith("L3 | 2025-03-05 | Office supplies Staples | -84.50")
    assert "amount matches" in first


def test_candidates_include_near_amounts_so_mismatches_can_be_found() -> None:
    # An exact-only search could never surface the 25.00 vs 52.00 discrepancy.
    registry, _ = setup()

    content = registry.run("find_candidate_entries", {"bank_line_id": "B5"}).content

    assert "L5 | 2025-03-09 | Bank fee | -52.00" in content
    assert "amount differs by 27.00" in content


def test_candidates_reach_across_a_period_end() -> None:
    registry, _ = setup()

    content = registry.run("find_candidate_entries", {"bank_line_id": "B6"}).content

    assert "L4 | 2025-04-02" in content
    assert "2 days after" in content


def test_window_narrows_the_search() -> None:
    registry, _ = setup()

    narrow = registry.run("find_candidate_entries", {"bank_line_id": "B6", "date_window_days": 1}).content

    assert "No ledger entries within 1 day of B6" in narrow


def test_candidates_show_which_entries_are_already_taken() -> None:
    registry, _ = setup()
    propose(registry, "B3", "L3")

    content = registry.run("find_candidate_entries", {"bank_line_id": "B4"}).content

    assert "already matched to B3" in content


def test_candidates_mark_the_entry_already_proposed_for_this_line() -> None:
    registry, _ = setup()
    propose(registry, "B3", "L3")

    content = registry.run("find_candidate_entries", {"bank_line_id": "B3"}).content

    assert "already proposed for this line" in content
    assert "already matched to B3" not in content


def test_unknown_bank_line_lists_the_valid_ids() -> None:
    registry, _ = setup()

    result = registry.run("find_candidate_entries", {"bank_line_id": "B99"})

    assert result.is_error
    assert result.content == "No bank line 'B99'. Valid IDs: B1, B2, B3, B4, B5, B6."


# ---- propose_match -------------------------------------------------------


def test_propose_records_the_match_and_reports_progress() -> None:
    registry, state = setup()

    result = propose(registry, "B1", "L1", "matched", "same amount, one day apart")

    assert not result.is_error
    assert "Recorded B1 -> L1 (matched)" in result.content
    assert "5 bank lines still have no proposal" in result.content
    assert state.proposals["B1"].note == "same amount, one day apart"


def test_propose_no_match_records_no_entry() -> None:
    registry, state = setup()

    result = propose(registry, "B5", None, "no_match", "nothing fits")

    assert not result.is_error
    assert "Recorded B5 -> no entry (no_match)" in result.content
    assert state.proposals["B5"].entry_id is None


def test_propose_unknown_bank_line() -> None:
    registry, _ = setup()

    result = propose(registry, "B99", "L1")

    assert result.is_error
    assert result.content.startswith("No bank line 'B99'. Valid IDs: B1,")


def test_propose_unknown_entry() -> None:
    registry, _ = setup()

    result = propose(registry, "B1", "L99")

    assert result.is_error
    assert result.content == "No ledger entry 'L99'. Valid IDs: L1, L2, L3, L4, L5."


def test_matched_status_needs_an_entry() -> None:
    registry, _ = setup()

    result = propose(registry, "B1", None, "matched")

    assert result.is_error
    assert "needs an entry_id" in result.content and "no_match" in result.content


def test_no_match_must_not_name_an_entry() -> None:
    registry, _ = setup()

    result = propose(registry, "B1", "L1", "no_match")

    assert result.is_error
    assert "entry_id must be null" in result.content


def test_an_entry_cannot_be_matched_twice_and_the_error_points_to_duplicates() -> None:
    registry, state = setup()
    propose(registry, "B3", "L3")

    result = propose(registry, "B4", "L3", "matched")

    assert result.is_error
    assert "L3 is already matched to B3" in result.content
    assert "duplicate_suspected" in result.content
    assert "B4" not in state.proposals


def test_a_suspected_duplicate_may_name_the_entry_or_not() -> None:
    registry, state = setup()
    propose(registry, "B3", "L3")

    with_entry = propose(registry, "B4", "L3", "duplicate_suspected", "repeat of B3")
    without_entry = propose(registry, "B4", None, "duplicate_suspected", "repeat of B3")

    assert not with_entry.is_error and not without_entry.is_error
    assert state.proposals["B4"].entry_id is None


def test_a_second_proposal_replaces_the_first_and_says_so() -> None:
    registry, state = setup()
    propose(registry, "B1", None, "no_match", "first guess")

    result = propose(registry, "B1", "L1", "matched", "found it")

    assert not result.is_error
    assert "Replaced the earlier proposal for B1 (was no_match)" in result.content
    assert state.proposals["B1"].status == "matched"


def test_a_bad_status_is_rejected_by_name() -> None:
    registry, _ = setup()

    result = propose(registry, "B1", "L1", "probably")

    assert result.is_error and "status" in result.content


def test_an_empty_note_is_rejected_by_name() -> None:
    registry, _ = setup()

    result = propose(registry, "B1", "L1", "matched", "")

    assert result.is_error and "note" in result.content


# ---- the whole thing through the loop ------------------------------------


def call(id: str, name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=id, name=name, input=arguments)


def test_a_correct_scripted_run_reproduces_the_answer_key() -> None:
    registry, state = setup()
    case = toy_case()
    proposals = [
        call("t1", "propose_match", bank_line_id="B1", entry_id="L1", status="matched", note="ok"),
        call("t2", "propose_match", bank_line_id="B2", entry_id="L2", status="matched", note="ok"),
        call("t3", "propose_match", bank_line_id="B3", entry_id="L3", status="matched", note="ok"),
        call("t4", "propose_match", bank_line_id="B4", entry_id=None,
             status="duplicate_suspected", note="repeat of B3"),
        call("t5", "propose_match", bank_line_id="B5", entry_id="L5", status="amount_mismatch",
             note="25.00 vs 52.00"),
        call("t6", "propose_match", bank_line_id="B6", entry_id="L4", status="timing_difference",
             note="Mar 31 vs Apr 2"),
    ]
    client = FakeClient(
        [
            tool_use_turn(call("t0", "list_unmatched_bank_lines")),
            tool_use_turn(*proposals),
            final_turn("Matched 3, 1 timing difference, 1 mismatch, 1 suspected duplicate."),
        ]
    )

    result = run_agent(TASK, client, registry, RunConfig(system_prompt=SYSTEM_PROMPT))

    assert result.stop_reason == "final_answer"
    assert {
        line_id: (p.status, p.entry_id) for line_id, p in state.proposals.items()
    } == {
        line_id: (expected.status, expected.entry_id)
        for line_id, expected in case.answer_key.items()
    }


def test_the_agent_can_recover_from_a_domain_error() -> None:
    registry, state = setup()
    client = FakeClient(
        [
            tool_use_turn(call("t1", "propose_match", bank_line_id="B3", entry_id="L3",
                               status="matched", note="ok")),
            # The mistake: calling the repeat charge a plain match for the same entry.
            tool_use_turn(call("t2", "propose_match", bank_line_id="B4", entry_id="L3",
                               status="matched", note="ok")),
            # The fix, using what the error message said.
            tool_use_turn(call("t3", "propose_match", bank_line_id="B4", entry_id=None,
                               status="duplicate_suspected", note="repeat of B3")),
            final_turn("done"),
        ]
    )

    result = run_agent(TASK, client, registry)

    tool_results = [
        block
        for message in result.messages
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
    ]
    assert [bool(block.get("is_error")) for block in tool_results] == [False, True, False]
    assert state.proposals["B4"].status == "duplicate_suspected"
    assert result.stop_reason == "final_answer"


def test_the_prompts_define_every_status() -> None:
    for status in ("matched", "timing_difference", "amount_mismatch", "duplicate_suspected", "no_match"):
        assert status in SYSTEM_PROMPT
    assert "bank" in TASK.lower() and "ledger" in TASK.lower()
