"""Contract tests for the additive partner-specific 2026 authority ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict:
    return json.loads(
        (ROOT / "baselines" / "BASELINE_LOCK_V6.json").read_text(
            encoding="utf-8"
        )
    )


def test_v6_partner_specific_lock_is_additive_and_fail_closed() -> None:
    payload = _payload()
    parent = ROOT / payload["parent"]["path"]
    assert payload["format_version"] == "framecontact_baseline_lock_v6"
    assert _sha256(parent) == payload["parent"]["sha256"]
    assert payload["comparison_contract"]["contact_cutoff_axes_angstrom"] == [
        4.5,
        6.0,
        8.0,
    ]
    assert payload["comparison_contract"][
        "full_sequence_retrieval_must_be_separated_from_true_interface_mask_retrieval"
    ] is True
    assert payload["comparison_contract"][
        "same_antigen_different_antibody_counterfactual_required"
    ] is True
    assert set(payload["frontier_additions"]) == {
        "aamfm",
        "calm",
        "pepnet",
        "sppider_seq",
    }
    for authority in payload["frontier_additions"].values():
        assert authority["identical_split_runs"] == 0
        assert authority["result_status"] == "not_run"
        assert authority["comparability_boundary"]
        assert authority["license_boundary"]


def test_v6_sources_weights_and_absence_boundaries_are_pinned() -> None:
    additions = _payload()["frontier_additions"]
    assert additions["aamfm"]["source_revision"] == (
        "36036588de55f9bd1675dbbf92d726e3ce8db941"
    )
    assert additions["calm"]["source_revision"] is None
    assert additions["pepnet"]["source_revision"] == (
        "fa0fb4002c5294a094bf4d050098692067e545f4"
    )
    assert additions["pepnet"]["official_checkpoint_present"] is False
    assert additions["sppider_seq"]["source_revision"] == (
        "9c12e8c826ad4e47b00477cab56f6d58715d8f60"
    )
    assert additions["sppider_seq"]["checkpoints"][
        "models/crossattn_pep_run07_best.pt"
    ]["sha256"] == (
        "c359fcdb5751d94d3a1cdc01bb285cb06ff2aade604a992ecdc9e8642252cfe0"
    )
    assert additions["sppider_seq"]["checkpoints"][
        "models/crossattn_rec_run27_best.pt"
    ]["sha256"] == (
        "de00e13dd1edbe7a4eb76e3cbf298ba65abe2ef44cd85d9ce12dff5b5e01c113"
    )
