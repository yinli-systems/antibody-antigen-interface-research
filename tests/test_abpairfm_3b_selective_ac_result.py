"""Audit the compact selective-checkpoint negative result."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_3b_selective_ac_job_1459029_rejected.json"
)


def test_selective_checkpoint_candidate_is_fail_closed() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["job"]["job_id"] == 1_459_029
    assert result["job"]["world_size"] == 8
    assert result["decision"] == "REJECT_SELECTIVE_ATTENTION"
    assert result["status"] == "PASS_REJECTED_CANDIDATE"
    assert result["gates"]["flash_attention_policy_exercised"] is True
    assert result["gates"]["throughput_speedup"] is False
    assert result["gates"]["loss_parity"] is False
    assert result["gates"]["gradient_norm_parity"] is False
    assert result["authorization"]["real_data_gate"] is False
    assert result["authorization"]["production_pretraining"] is False
    assert all(
        len(digest) == 64 for digest in result["raw_receipts"].values()
    )
