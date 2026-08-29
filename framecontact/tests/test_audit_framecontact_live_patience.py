from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_framecontact_live_patience.py"
)
SPEC = importlib.util.spec_from_file_location("audit_framecontact_live_patience", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(epoch: int, margin: float, contact_macro: float) -> dict:
    return {
        "epoch": epoch,
        "elapsed_seconds": epoch * 120.0,
        "validation": {
            "selection_metric": margin,
            "contact": {"mean_per_complex_auprc": contact_macro},
        },
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_replay_selection_matches_primary_and_tie_break_rule() -> None:
    records = {
        1: _record(1, 0.10, 0.20),
        2: _record(2, 0.10005, 0.21),
        3: _record(3, 0.10004, 0.19),
        4: _record(4, 0.10020, 0.18),
        5: _record(5, 0.10011, 0.30),
    }
    state = MODULE.replay_selection(records, minimum_improvement=1e-4)
    assert state["best_epoch"] == 5
    assert state["best_validation_worst_directional_margin"] == pytest.approx(0.10011)
    assert state["best_contact_macro_auprc_tie_break"] == pytest.approx(0.30)
    assert state["improvement_epochs"] == [1, 2, 4, 5]
    assert state["stale_epochs"] == 0


def test_audit_reports_conditional_patience_timing_and_hash(tmp_path: Path) -> None:
    _write(
        tmp_path / "fc-123_0.out",
        [
            _record(1, 0.10, 0.20),
            _record(2, 0.11, 0.20),
            _record(3, 0.09, 0.30),
            _record(4, 0.08, 0.40),
        ],
    )
    result = MODULE.audit_live_patience(
        log_root=tmp_path,
        log_prefix="fc",
        job_id="123",
        condition_index=0,
        condition_label="candidate",
        seeds=[17, 42],
        patience=5,
    )

    seed = result["states_by_seed"]["17"]
    assert seed["best_epoch"] == 2
    assert seed["stale_epochs"] == 2
    assert seed["patience_remaining_if_no_new_best"] == 3
    assert seed["recent_mean_seconds_per_epoch"] == pytest.approx(120.0)
    assert seed["conditional_hours_to_early_stop_if_no_new_best"] == pytest.approx(
        0.1
    )
    assert len(seed["log_sha256_at_audit"]) == 64
    assert result["states_by_seed"]["42"]["status"] == (
        "not_started_or_no_epoch_record"
    )
    assert "No checkpoint" in result["claim_boundary"]


def test_audit_rejects_invalid_patience(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="patience"):
        MODULE.audit_live_patience(
            log_root=tmp_path,
            log_prefix="fc",
            job_id="123",
            condition_index=0,
            condition_label="candidate",
            seeds=[17],
            patience=0,
        )
