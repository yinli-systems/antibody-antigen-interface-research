"""Audit the compact MINT checkpoint compatibility result."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_mint_650m_checkpoint_compatibility_v1.json"
)


def test_mint_checkpoint_is_admitted_only_for_pair_mlm_and_retrieval() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["architecture"]["unique_parameter_elements"] == 813_371_969
    assert result["compatibility"]["architecture_compatible"] is True
    assert result["compatibility"]["missing_keys"] == []
    assert result["compatibility"]["unexpected_keys"] == []
    assert result["compatibility"]["shape_mismatches"] == {}
    assert result["compatibility"]["tied_lm_head_duplicate_verified"] is True
    assert result["authorization"]["frozen_development_pair_mlm_baseline"] is True
    assert result["authorization"]["frozen_development_retrieval_baseline"] is True
    assert result["authorization"]["frozen_development_contact_baseline"] is False
    assert result["authorization"]["production_training"] is False
    assert result["authorization"]["test_access"] is False
    assert all(
        len(value) == 64
        for key, value in result["provenance"].items()
        if key.endswith("sha256")
    )
