"""Audit the compact checkpoint RNG-stashing negative result."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_3b_checkpoint_rng_job_1459049_rejected.json"
)


def test_checkpoint_rng_candidate_is_fail_closed() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["job"]["job_id"] == 1_459_049
    assert result["job"]["world_size"] == 8
    assert result["decision"] == "REJECT_NO_RNG_STASH"
    assert result["status"] == "PASS_REJECTED_CANDIDATE"
    assert result["gates"]["throughput_speedup"] is True
    assert result["gates"]["loss_parity"] is False
    assert result["gates"]["gradient_norm_parity"] is False
    assert result["authorization"]["real_data_gate"] is False
    assert result["authorization"]["production_pretraining"] is False
    assert all(len(digest) == 64 for digest in result["raw_receipts"].values())
