from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_mint_development_retrieval_v1.json"


def test_mint_retrieval_protocol_freezes_test_and_contact_boundaries() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["authorization"]["development_pair_retrieval"] is True
    assert protocol["authorization"]["development_contact"] is False
    assert protocol["authorization"]["production_training"] is False
    assert protocol["authorization"]["test_access"] is False
    assert protocol["input"]["test_roles_permitted"] is False
    assert protocol["test_access"] is False


def test_mint_retrieval_protocol_fixes_lengths_duplicates_and_units() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["length_policy"]["maximum_residues_per_endpoint_window"] == 510
    duplicate = protocol["metrics"]["duplicate_policy"]
    assert duplicate["primary"].startswith("collapse exact candidate endpoint")
    assert duplicate["secondary"].startswith("retain all 31")
    assert protocol["metrics"]["bootstrap_unit"] == "canonical_pair_sha256"
    assert protocol["metrics"]["rank_tie_policy"].startswith("average rank")


def test_mint_retrieval_protocol_pins_audited_public_artifacts() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = protocol["model"]
    assert model["backbone_unique_parameters"] == 813_371_969
    assert model["head_parameters"] == 6_558_721
    assert len(model["backbone_checkpoint_sha256"]) == 64
    assert len(model["head_checkpoint_sha256"]) == 64
    assert len(model["retrieval_head_audit_sha256"]) == 64
