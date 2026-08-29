from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/prospective_epitope_design_v1"
SPEC = importlib.util.spec_from_file_location("prospective_study", STUDY / "prospective_study.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def completed_protocol(tmp_path: Path) -> Path:
    protocol = json.loads((STUDY / "protocol/PROTOCOL_DRAFT_V1.json").read_text())
    protocol["target"] = {
        "target_id": "synthetic-safe-target",
        "molecule_type": "synthetic-test-only",
        "antigen_artifact_sha256": "a" * 64,
        "epitope_definition": "synthetic-region-1",
        "bound_state_or_structure_source_disclosure": "synthetic fixture; no biological structure",
    }
    protocol["safety"] = {
        "approved": True,
        "reviewer": "independent-synthetic-reviewer",
        "review_date": "2026-08-27",
        "target_is_pathogen_or_toxin": False,
        "gain_of_function_in_scope": False,
    }
    protocol["computational_authority"] = {
        "generator_id": "synthetic-generator",
        "generator_revision_or_weight_sha256": "b" * 64,
        "generation_protocol_sha256": "c" * 64,
        "proposal_pool_manifest_sha256": "d" * 64,
        "eligibility_rule": "synthetic finite fixtures only",
        "baseline_score_definition": "synthetic baseline column",
        "partner_specific_model_id": "synthetic-partner-ranker",
        "partner_specific_checkpoint_sha256": "e" * 64,
        "partner_score_definition": "synthetic partner column",
        "acquisition_score_implementation_sha256": "f" * 64,
        "acquisition_score_definition": "synthetic acquisition column",
        "diversity_cluster_implementation_sha256": "1" * 64,
        "diversity_cluster_definition": "synthetic cluster label",
        "candidate_outcomes_used_to_fit_or_select_any_authority": False,
    }
    protocol["assay"] = {
        "primary_name": "synthetic binary assay",
        "primary_hit_rule": "primary_value >= 1.0",
        "threshold_value": 1.0,
        "threshold_unit": "synthetic-unit",
        "direction": "higher_is_better",
        "positive_control": "synthetic-positive",
        "negative_control": "synthetic-negative",
        "confirmatory_assay": "synthetic confirmation",
        "lab_blinded_to_selection_arm": True,
    }
    protocol["design"]["candidate_budget_per_arm"] = 3
    protocol["design"]["bootstrap"]["replicates"] = 1000
    protocol["role_separation"] = {
        "private_mapping_custodian": "synthetic-mapping-role",
        "lab_results_custodian": "synthetic-results-role",
        "same_person_allowed": False,
    }
    path = tmp_path / "completed.json"
    path.write_text(json.dumps(protocol))
    return path


def freeze_fixture(tmp_path: Path) -> tuple[Path, str, Path, Path, Path]:
    completed = completed_protocol(tmp_path)
    protocol = tmp_path / "authority/protocol.json"
    receipt = tmp_path / "authority/receipt.json"
    MODULE.freeze_protocol(
        argparse.Namespace(
            input=completed,
            output=protocol,
            receipt=receipt,
            frozen_at_utc="2026-08-27T00:00:00+00:00",
        )
    )
    protocol_sha = MODULE.sha256_file(protocol)
    key = tmp_path / "private/blinding.key"
    key.parent.mkdir(parents=True)
    key.write_bytes(b"synthetic-test-key-not-for-production" * 2)
    public = tmp_path / "blinded/manifest.json"
    private = tmp_path / "private/mapping.json"
    candidate_receipt = tmp_path / "blinded/receipt.json"
    MODULE.freeze_slate(
        argparse.Namespace(
            protocol=protocol,
            protocol_sha256=protocol_sha,
            candidates=STUDY / "examples/SYNTHETIC_CANDIDATES.tsv",
            blinding_key_file=key,
            public_manifest=public,
            private_mapping=private,
            receipt=candidate_receipt,
        )
    )
    return protocol, protocol_sha, public, private, candidate_receipt


def test_incomplete_protocol_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target.target_id"):
        MODULE.freeze_protocol(
            argparse.Namespace(
                input=STUDY / "protocol/PROTOCOL_DRAFT_V1.json",
                output=tmp_path / "protocol.json",
                receipt=tmp_path / "receipt.json",
                frozen_at_utc="2026-08-27T00:00:00+00:00",
            )
        )


def test_blinded_slate_is_deterministic_and_hides_policy(tmp_path: Path) -> None:
    _, _, public_path, private_path, _ = freeze_fixture(tmp_path)
    public = json.loads(public_path.read_text())
    private = json.loads(private_path.read_text())
    public_text = public_path.read_text()
    assert public["candidate_budget_per_arm"] == 3
    assert public["arm_membership_disclosed"] is False
    assert "generator_baseline" not in public_text
    assert "partner_score" not in public_text
    assert "mock-" not in public_text
    assert len(private["records"]) == public["unique_candidate_count"]
    assert all(len(row["blinded_id"]) == 18 for row in private["records"])


def test_candidate_outcome_column_is_rejected(tmp_path: Path) -> None:
    source = STUDY / "examples/SYNTHETIC_CANDIDATES.tsv"
    bad = tmp_path / "bad.tsv"
    lines = source.read_text().splitlines()
    lines[0] += "\tassay_hit"
    lines[1] += "\t1"
    for index in range(2, len(lines)):
        lines[index] += "\t0"
    bad.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="columns differ"):
        MODULE.validate_candidates(bad)


def test_candidate_receipt_detects_mapping_change(tmp_path: Path) -> None:
    protocol, protocol_sha, public, private, candidate_receipt = freeze_fixture(tmp_path)
    mapping = json.loads(private.read_text())
    mapping["records"][0]["scores"]["partner_score"] += 0.01
    private.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="candidate receipt"):
        MODULE.analyze(
            argparse.Namespace(
                protocol=protocol,
                protocol_sha256=protocol_sha,
                public_manifest=public,
                private_mapping=private,
                candidate_receipt=candidate_receipt,
                assay_results=tmp_path / "unused-assay.tsv",
                control_results=tmp_path / "unused-controls.tsv",
                output=tmp_path / "unused-result.json",
                receipt=tmp_path / "unused-receipt.json",
            )
        )


def test_primary_hit_must_match_frozen_threshold(tmp_path: Path) -> None:
    protocol, protocol_sha, public, private, candidate_receipt = freeze_fixture(tmp_path)
    manifest = json.loads(public.read_text())
    results = tmp_path / "assay.tsv"
    with results.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MODULE.RESULT_COLUMNS)
        for row in manifest["candidates"]:
            writer.writerow([row["blinded_id"], 0, 2.0, "", ""])
    controls = tmp_path / "controls.tsv"
    with controls.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MODULE.CONTROL_COLUMNS)
        writer.writerow(["synthetic-positive", "positive", 2.0, 1])
        writer.writerow(["synthetic-negative", "negative", 0.0, 1])
    with pytest.raises(ValueError, match="primary_hit differs"):
        MODULE.analyze(
            argparse.Namespace(
                protocol=protocol,
                protocol_sha256=protocol_sha,
                public_manifest=public,
                private_mapping=private,
                candidate_receipt=candidate_receipt,
                assay_results=results,
                control_results=controls,
                output=tmp_path / "unused-result.json",
                receipt=tmp_path / "unused-receipt.json",
            )
        )


def test_complete_blinded_analysis(tmp_path: Path) -> None:
    protocol, protocol_sha, public, private, candidate_receipt = freeze_fixture(tmp_path)
    manifest = json.loads(public.read_text())
    results = tmp_path / "assay.tsv"
    with results.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MODULE.RESULT_COLUMNS)
        for index, row in enumerate(manifest["candidates"]):
            hit = index % 2
            writer.writerow([row["blinded_id"], hit, float(hit), hit, 1])
    controls = tmp_path / "controls.tsv"
    with controls.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MODULE.CONTROL_COLUMNS)
        writer.writerow(["synthetic-positive", "positive", 2.0, 1])
        writer.writerow(["synthetic-negative", "negative", 0.0, 1])
    output = tmp_path / "results/result.json"
    receipt = tmp_path / "results/receipt.json"
    MODULE.analyze(
        argparse.Namespace(
            protocol=protocol,
            protocol_sha256=protocol_sha,
            public_manifest=public,
            private_mapping=private,
            candidate_receipt=candidate_receipt,
            assay_results=results,
            control_results=controls,
            output=output,
            receipt=receipt,
        )
    )
    result = json.loads(output.read_text())
    assert result["status"] == "PASS_BLINDED_ASSAY_ANALYSIS"
    assert result["audit"]["all_blinded_ids_accounted_for"] is True
    assert result["audit"]["positive_and_negative_controls_passed"] is True
    assert set(result["arm_metrics"]) == set(MODULE.ARMS)
    assert all(result["arm_metrics"][arm]["n"] == 3 for arm in MODULE.ARMS)
    assert "superior" not in output.read_text().lower()
