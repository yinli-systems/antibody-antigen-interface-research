#!/usr/bin/env python3
"""Synthetic tests for the authenticated retrieval-control comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sabdab2_asymmetric_partner_retrieval import (
    _score_array_sha256,
    canonical_sha256,
    write_score_artifact,
)
from sabdab2_asymmetric_retrieval_statistics import (
    AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
    AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
    CONTROL_NAMES,
    CONTROL_SUITE_VERSION,
    load_score_artifact,
)
from sabdab2_retrieval_control_comparator import (
    CONTROL_PROTOCOL_SHA256,
    BOUNDARY_VERIFICATION_SHA256,
    EXECUTION_PROTOCOL_SHA256,
    EXECUTION_HARDENING_PROTOCOL_SHA256,
    EXPECTED_FIXED_EVALUATION,
    GATE_NAME,
    LOCKED_BOOTSTRAP_COUNT,
    LOCKED_BOOTSTRAP_SEED,
    _authenticate_pair,
    _authenticate_control_bundle,
    _authenticate_locked_submission_evidence,
    _authenticate_control_suite_completion,
    _authenticate_protocol,
    _authenticate_suite,
    _paired_comparison,
    _validate_control_arguments,
    _write_exclusive_json,
    run_comparison,
)
from sabdab2_retrieval_controls import (
    _source_hashes as control_source_hashes,
    _write_bundle,
    control_evaluation_identity,
)


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_PROTOCOL = (
    ROOT
    / "outputs/interaction_first/research/PARTNER_RETRIEVAL_PROTOCOL_V1_5_EXECUTION_LOCK.md"
)
CONTROL_PROTOCOL = (
    ROOT
    / "outputs/interaction_first/research/PARTNER_RETRIEVAL_CONTROLS_PROTOCOL_V1.md"
)
EXECUTION_HARDENING_PROTOCOL = (
    ROOT
    / "outputs/interaction_first/research/PARTNER_RETRIEVAL_PROTOCOL_V1_6_EXECUTION_HARDENING.md"
)
BOUNDARY_VERIFICATION = (
    ROOT
    / "outputs/interaction_first/research/authority_validation_nw_boundaries/independent_verification_v2.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_reference() -> dict[str, str]:
    return {
        "path": str(EXECUTION_PROTOCOL.resolve()),
        "sha256": EXECUTION_PROTOCOL_SHA256,
    }


def _control_protocol_reference() -> dict[str, str]:
    return {
        "path": str(CONTROL_PROTOCOL.resolve()),
        "sha256": CONTROL_PROTOCOL_SHA256,
    }


def _artifact(name: str, scores: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    count = scores.shape[0]
    antibody_keys = sorted(_digest(f"ab-{index}") for index in range(count))
    antigen_keys = sorted(_digest(f"ag-{index}") for index in range(scores.shape[1]))
    antibody_array = np.asarray(antibody_keys, dtype="S64")
    antigen_array = np.asarray(antigen_keys, dtype="S64")
    return {
        "path": Path(f"/synthetic/{name}.scores.npz"),
        "sha256": _digest(f"{name}-score-file"),
        "similarities": np.asarray(scores, dtype=np.float32),
        "positive_mask": np.asarray(mask, dtype=bool),
        "antibody_keys": antibody_keys,
        "antigen_keys": antigen_keys,
        "array_sha256": {
            "similarity_matrix": _score_array_sha256(
                np.asarray(scores, dtype=np.float32)
            ),
            "positive_mask": _score_array_sha256(np.asarray(mask, dtype=bool)),
            "antibody_keys": _score_array_sha256(antibody_array),
            "antigen_keys": _score_array_sha256(antigen_array),
        },
    }


def _dataset() -> dict[str, object]:
    return {
        "split_manifest_sha256": _digest("boundary"),
        "split_format_version": AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
        "split_intended_use": AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
        "strict_nw_constraint_mode": "concat_antibody_and_antigen",
        "instances_sha256": _digest("instances"),
        "split_membership_sha256": {
            "train": _digest("train-membership"),
            "validation": _digest("validation-membership"),
            "excluded_validation": _digest("excluded-membership"),
        },
        "raw_instance_counts": {
            "train": 20,
            "validation": 8,
            "excluded_validation": 2,
        },
        "authority": {"sha256": _digest("authority")},
    }


def _verified(
    name: str,
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    primary: bool,
) -> dict[str, object]:
    evaluation: dict[str, object] = {
        "checkpoint_run_mode": (
            "final_refit_fixed_epochs" if primary else "train_only_control_bundle"
        ),
        "one_shot_evaluation": True,
        "checkpoint_sha256": _digest("primary-checkpoint" if primary else "control-bundle"),
        "dataset": _dataset(),
    }
    ledger = {
        "required": True,
        "status": "completed",
        "canonical_key": _digest("primary-ledger" if primary else "control-ledger"),
        "path": (
            "/synthetic/primary-ledger.json"
            if primary
            else "/synthetic/control-ledger.json"
        ),
        "sha256": _digest("primary-ledger-file" if primary else "control-ledger-file"),
    }
    if not primary:
        evaluation["control"] = {
            "name": name,
            "protocol_version": CONTROL_SUITE_VERSION,
        }
        evaluation["execution_protocol_v1_6"] = {
            "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
            "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
        }
        evaluation["control_protocol_v1"] = _control_protocol_reference()
        ledger["control_name"] = name
    return {
        "evaluation": evaluation,
        "evaluation_path": Path(f"/synthetic/{name}.evaluation.json"),
        "evaluation_sha256": _digest(f"{name}-evaluation-file"),
        "artifact": _artifact(name, scores, mask),
        "evidence_scope": "reused_development_confirmation",
        "boundary_verification": {
            "format_version": AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
            "excluded_validation_rows_read_for_statistics": False,
        },
        "one_shot_verification": ledger,
    }


def _raw_control_specs() -> list[list[str]]:
    return [
        [
            name,
            f"/synthetic/{name}.evaluation.json",
            _digest(f"{name}-evaluation-file"),
            f"/synthetic/{name}.scores.npz",
            _digest(f"{name}-score-file"),
        ]
        for name in CONTROL_NAMES
    ]


def _primary_spec() -> tuple[Path, str, Path, str]:
    return (
        Path("/synthetic/primary.evaluation.json"),
        _digest("primary-evaluation-file"),
        Path("/synthetic/primary.scores.npz"),
        _digest("primary-score-file"),
    )


def _arguments() -> argparse.Namespace:
    return argparse.Namespace(
        primary_evaluation_json=Path("/synthetic/primary.evaluation.json"),
        primary_evaluation_sha256=_digest("primary-evaluation-file"),
        primary_score_artifact=Path("/synthetic/primary.scores.npz"),
        primary_score_artifact_sha256=_digest("primary-score-file"),
        control=_raw_control_specs(),
        execution_protocol=EXECUTION_PROTOCOL,
        execution_protocol_sha256=EXECUTION_PROTOCOL_SHA256,
        control_protocol=CONTROL_PROTOCOL,
        control_protocol_sha256=CONTROL_PROTOCOL_SHA256,
        output=Path("/synthetic/comparison.json"),
    )


class ProtocolAndInputTests(unittest.TestCase):
    def test_physical_protocols_match_the_locked_digests(self) -> None:
        execution = _authenticate_protocol(
            EXECUTION_PROTOCOL,
            EXECUTION_PROTOCOL_SHA256,
            EXECUTION_PROTOCOL_SHA256,
            "execution",
        )
        controls = _authenticate_protocol(
            CONTROL_PROTOCOL,
            CONTROL_PROTOCOL_SHA256,
            CONTROL_PROTOCOL_SHA256,
            "controls",
        )
        hardening = _authenticate_protocol(
            EXECUTION_HARDENING_PROTOCOL,
            EXECUTION_HARDENING_PROTOCOL_SHA256,
            EXECUTION_HARDENING_PROTOCOL_SHA256,
            "hardening",
        )
        self.assertEqual(execution["sha256"], EXECUTION_PROTOCOL_SHA256)
        self.assertEqual(controls["sha256"], CONTROL_PROTOCOL_SHA256)
        self.assertEqual(
            hardening["sha256"], EXECUTION_HARDENING_PROTOCOL_SHA256
        )

    def test_substituted_protocol_digest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked digest"):
            _authenticate_protocol(
                EXECUTION_PROTOCOL,
                _digest("substitution"),
                EXECUTION_PROTOCOL_SHA256,
                "execution",
            )

    def test_exact_control_inventory_is_required(self) -> None:
        controls = _validate_control_arguments(_raw_control_specs())
        self.assertEqual(tuple(controls), CONTROL_NAMES)
        with self.assertRaisesRegex(ValueError, "Exactly 5"):
            _validate_control_arguments(_raw_control_specs()[:-1])
        duplicate = _raw_control_specs()
        duplicate[-1][0] = duplicate[0][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _validate_control_arguments(duplicate)

    def test_comparison_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            _write_exclusive_json(output, {"status": "first"})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                _write_exclusive_json(output, {"status": "replacement"})
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_duplicate_json_keys_are_rejected_before_shared_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "evaluation.json"
            evaluation.write_text('{"split":"validation","split":"test"}', encoding="utf-8")
            with patch(
                "sabdab2_retrieval_control_comparator.verify_evaluation_and_artifact"
            ) as verifier, self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _authenticate_pair(
                    evaluation,
                    _digest("evaluation"),
                    Path(directory) / "scores.npz",
                    _digest("scores"),
                )
            verifier.assert_not_called()


class LockedSubmissionEvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, object]:
        mask = np.eye(3, dtype=bool)
        scores = np.eye(3, dtype=np.float32)
        verified = _verified("primary", scores, mask, primary=True)
        evaluation_path = (root / "primary.evaluation.json").resolve()
        score_path = (root / "primary.scores.npz").resolve()
        verified["evaluation_path"] = evaluation_path
        verified["artifact"]["path"] = score_path
        selected_boundary = "concat_boundary"
        protocol_reference = _protocol_reference()
        hardening_protocol_reference = {
            "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
            "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
        }
        boundary_reference = {
            "path": str(BOUNDARY_VERIFICATION.resolve()),
            "sha256": BOUNDARY_VERIFICATION_SHA256,
        }
        dataset = verified["evaluation"]["dataset"]
        manifest = {
            "format_version": "sabdab2_asymmetric_final_refit_submission_manifest_v1",
            "intended_use": "development_only",
            "frozen_test_used": False,
            "fixed_evaluation": EXPECTED_FIXED_EVALUATION,
            "canonical_refit_boundary": selected_boundary,
            "upstream_evidence": {
                "protocol_v1_5_execution_lock": protocol_reference,
                "protocol_v1_6_execution_hardening": hardening_protocol_reference,
                "boundary_verification_v2": boundary_reference,
            },
            "boundaries": {
                selected_boundary: {
                    "split_manifest": {
                        "path": "/synthetic/boundary.json",
                        "sha256": dataset["split_manifest_sha256"],
                    },
                    "strict_nw_constraint_mode": dataset[
                        "strict_nw_constraint_mode"
                    ],
                    "evaluation_output": str(evaluation_path),
                    "score_artifact": str(score_path),
                }
            },
        }
        manifest_path = root / "submission.json"
        registry = (root / "registry").resolve()
        manifest["one_shot_ledger_registry_dir"] = str(registry)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        submission_reference = {
            "path": str(manifest_path.resolve()),
            "sha256": _file_digest(manifest_path),
        }
        checkpoint_path = (root / "final.pt").resolve()
        checkpoint_sha256 = _digest("final-checkpoint")
        claim_path = (
            registry
            / "checkpoint-suite-claims"
            / f"{submission_reference['sha256']}.json"
        )
        claim_path.parent.mkdir(parents=True)
        claim = {
            "format_version": "sabdab2_final_checkpoint_suite_claim_v1",
            "stage": "final_checkpoint_suite_claim",
            "intended_use": "two_boundary_one_shot_development_confirmation",
            "status": "immutable",
            "submission_manifest": submission_reference,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "one_shot_ledger_registry_dir": str(registry),
            "boundaries": {
                selected_boundary: {
                    "split_manifest_sha256": dataset["split_manifest_sha256"],
                    "strict_nw_constraint_mode": dataset[
                        "strict_nw_constraint_mode"
                    ],
                },
                "chain_boundary": {
                    "split_manifest_sha256": _digest("chain-boundary"),
                    "strict_nw_constraint_mode": "chain_aware_four_view",
                },
            },
        }
        manifest["boundaries"]["chain_boundary"] = {
            "split_manifest": {
                "path": "/synthetic/chain-boundary.json",
                "sha256": _digest("chain-boundary"),
            },
            "strict_nw_constraint_mode": "chain_aware_four_view",
            "evaluation_output": str((root / "chain.evaluation.json").resolve()),
            "score_artifact": str((root / "chain.scores.npz").resolve()),
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        submission_reference["sha256"] = _file_digest(manifest_path)
        claim_path = (
            registry
            / "checkpoint-suite-claims"
            / f"{submission_reference['sha256']}.json"
        )
        claim["submission_manifest"] = submission_reference
        claim_path.write_text(
            json.dumps(claim, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        claim_reference = {
            "path": str(claim_path.resolve()),
            "sha256": _file_digest(claim_path),
        }
        evidence = {
            "submission_manifest": submission_reference,
            "protocol_v1_5_execution_lock": protocol_reference,
            "protocol_v1_6_execution_hardening": hardening_protocol_reference,
            "boundary_verification_v2": boundary_reference,
            "fixed_evaluation": EXPECTED_FIXED_EVALUATION,
            "selected_boundary": selected_boundary,
        }
        verified["evaluation"]["locked_submission_evidence"] = evidence
        verified["evaluation"]["predeclared_evaluation_boundary"] = selected_boundary
        verified["evaluation"]["checkpoint"] = str(checkpoint_path)
        verified["evaluation"]["checkpoint_sha256"] = checkpoint_sha256
        verified["evaluation"]["one_shot_ledger"] = {
            "registry_dir": str(registry),
            "canonical_path": str(registry / f"{selected_boundary}.ledger.json"),
        }
        verified["evaluation"]["checkpoint_suite_claim"] = claim_reference
        verified["artifact"]["metadata"] = {
            "contract": {
                "evaluation_context": {
                    "submission_manifest_sha256": submission_reference["sha256"],
                    "protocol_v1_5_execution_lock_sha256": EXECUTION_PROTOCOL_SHA256,
                    "protocol_v1_6_execution_hardening_sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    "boundary_verification_v2_sha256": BOUNDARY_VERIFICATION_SHA256,
                    "fixed_evaluation_sha256": canonical_sha256(
                        EXPECTED_FIXED_EVALUATION
                    ),
                    "selected_boundary": selected_boundary,
                    "checkpoint_suite_claim_sha256": claim_reference["sha256"],
                }
            }
        }
        primary_score_reference = write_score_artifact(
            score_path,
            {
                "similarity_matrix": scores,
                "positive_mask": mask,
                "antibody_keys": verified["artifact"]["antibody_keys"],
                "antigen_keys": verified["artifact"]["antigen_keys"],
            },
            verified["artifact"]["metadata"]["contract"]["evaluation_context"],
        )
        verified["artifact"]["sha256"] = primary_score_reference["sha256"]
        physical_evaluation = {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_suite_claim": claim_reference,
            "predeclared_evaluation_boundary": selected_boundary,
            "score_artifact": primary_score_reference,
            "one_shot_ledger": {
                "canonical_path": str(registry / f"{selected_boundary}.ledger.json")
            },
        }
        evaluation_path.write_text(
            json.dumps(physical_evaluation, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        verified["evaluation_sha256"] = _file_digest(evaluation_path)
        chain_evaluation_path = (root / "chain.evaluation.json").resolve()
        chain_score_path = (root / "chain.scores.npz").resolve()
        chain_score_context = {
            **verified["artifact"]["metadata"]["contract"]["evaluation_context"],
            "selected_boundary": "chain_boundary",
        }
        chain_score_reference = write_score_artifact(
            chain_score_path,
            {
                "similarity_matrix": scores,
                "positive_mask": mask,
                "antibody_keys": verified["artifact"]["antibody_keys"],
                "antigen_keys": verified["artifact"]["antigen_keys"],
            },
            chain_score_context,
        )
        chain_evaluation_path.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_suite_claim": claim_reference,
                    "predeclared_evaluation_boundary": "chain_boundary",
                    "score_artifact": chain_score_reference,
                    "one_shot_ledger": {
                        "canonical_path": str(registry / "chain_boundary.ledger.json")
                    },
                },
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        completed_outputs = {
            selected_boundary: {
                "evaluation": evaluation_path,
                "score": score_path,
            },
            "chain_boundary": {
                "evaluation": chain_evaluation_path,
                "score": chain_score_path,
            },
        }
        completion_boundaries = {}
        for boundary_name, outputs in completed_outputs.items():
            ledger_path = registry / f"{boundary_name}.ledger.json"
            ledger = {
                "status": "completed",
                "checkpoint_suite_claim": claim_reference,
                "evaluation_json_sha256": _file_digest(outputs["evaluation"]),
                "score_artifact_sha256": _file_digest(outputs["score"]),
            }
            ledger_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            if boundary_name == selected_boundary:
                verified["one_shot_verification"] = {
                    "path": str(ledger_path.resolve()),
                    "sha256": _file_digest(ledger_path),
                }
            completion_boundaries[boundary_name] = {
                **claim["boundaries"][boundary_name],
                "one_shot_ledger": {
                    "path": str(ledger_path.resolve()),
                    "sha256": _file_digest(ledger_path),
                },
                "evaluation_json": {
                    "path": str(outputs["evaluation"]),
                    "sha256": _file_digest(outputs["evaluation"]),
                },
                "score_artifact": {
                    "path": str(outputs["score"]),
                    "sha256": _file_digest(outputs["score"]),
                },
            }
        completion_path = (
            registry
            / "checkpoint-suite-completions"
            / f"{submission_reference['sha256']}.json"
        )
        completion_path.parent.mkdir(parents=True)
        completion = {
            "format_version": "sabdab2_final_checkpoint_suite_completion_v1",
            "stage": "final_checkpoint_suite_completion",
            "intended_use": "two_boundary_one_shot_development_confirmation",
            "status": "completed",
            "checkpoint_suite_claim": claim_reference,
            "submission_manifest": submission_reference,
            "checkpoint": claim["checkpoint"],
            "one_shot_ledger_registry_dir": str(registry),
            "boundaries": completion_boundaries,
        }
        completion_path.write_text(
            json.dumps(completion, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        return verified

    def test_primary_evaluation_and_score_contract_bind_the_submission_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            result = _authenticate_locked_submission_evidence(
                verified, _protocol_reference()
            )
        self.assertEqual(
            result["boundary_verification_v2"]["sha256"],
            BOUNDARY_VERIFICATION_SHA256,
        )
        self.assertEqual(result["selected_boundary"], "concat_boundary")

    def test_missing_locked_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            del verified["evaluation"]["locked_submission_evidence"]
            with self.assertRaisesRegex(PermissionError, "locked-submission evidence"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_substituted_protocol_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            evidence = verified["evaluation"]["locked_submission_evidence"]
            evidence["protocol_v1_5_execution_lock"] = {
                "path": str(CONTROL_PROTOCOL.resolve()),
                "sha256": CONTROL_PROTOCOL_SHA256,
            }
            with self.assertRaisesRegex(ValueError, "different protocol v1.5"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_substituted_v1_6_protocol_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            evidence = verified["evaluation"]["locked_submission_evidence"]
            evidence["protocol_v1_6_execution_hardening"] = {
                "path": str(CONTROL_PROTOCOL.resolve()),
                "sha256": CONTROL_PROTOCOL_SHA256,
            }
            with self.assertRaisesRegex(ValueError, "different protocol v1.6"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_missing_checkpoint_suite_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            del verified["evaluation"]["checkpoint_suite_claim"]
            with self.assertRaisesRegex(ValueError, "path-and-SHA"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_checkpoint_suite_claim_status_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            reference = verified["evaluation"]["checkpoint_suite_claim"]
            claim_path = Path(reference["path"])
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim["status"] = "replaceable"
            claim_path.write_text(
                json.dumps(claim, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            reference["sha256"] = _file_digest(claim_path)
            verified["artifact"]["metadata"]["contract"]["evaluation_context"][
                "checkpoint_suite_claim_sha256"
            ] = reference["sha256"]
            with self.assertRaisesRegex(ValueError, "immutable identity"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_checkpoint_suite_claim_must_bind_both_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            reference = verified["evaluation"]["checkpoint_suite_claim"]
            claim_path = Path(reference["path"])
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            del claim["boundaries"]["chain_boundary"]
            claim_path.write_text(
                json.dumps(claim, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            reference["sha256"] = _file_digest(claim_path)
            verified["artifact"]["metadata"]["contract"]["evaluation_context"][
                "checkpoint_suite_claim_sha256"
            ] = reference["sha256"]
            with self.assertRaisesRegex(ValueError, "both manifest boundaries"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_missing_primary_suite_completion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            evidence = verified["evaluation"]["locked_submission_evidence"]
            registry = Path(verified["evaluation"]["one_shot_ledger"]["registry_dir"])
            completion = (
                registry
                / "checkpoint-suite-completions"
                / f"{evidence['submission_manifest']['sha256']}.json"
            )
            completion.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "completion receipt is missing"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_primary_suite_completion_status_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            evidence = verified["evaluation"]["locked_submission_evidence"]
            registry = Path(verified["evaluation"]["one_shot_ledger"]["registry_dir"])
            completion_path = (
                registry
                / "checkpoint-suite-completions"
                / f"{evidence['submission_manifest']['sha256']}.json"
            )
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["status"] = "partial"
            completion_path.write_text(
                json.dumps(completion, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "violates its identity"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_primary_suite_completion_rehashes_the_other_boundary_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verified = self._fixture(root)
            (root / "chain.scores.npz").write_bytes(b"tampered-chain-score")
            with self.assertRaisesRegex(ValueError, "physical SHA-256 mismatch"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )

    def test_missing_score_context_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified = self._fixture(Path(directory))
            del verified["artifact"]["metadata"]["contract"][
                "evaluation_context"
            ]["boundary_verification_v2_sha256"]
            with self.assertRaisesRegex(ValueError, "score contract lacks"):
                _authenticate_locked_submission_evidence(
                    verified, _protocol_reference()
                )


class ControlSuiteCompletionTests(unittest.TestCase):
    def test_control_bundle_binds_the_exact_physical_controls_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "wrong-protocol-bundle.npz"
            wrong_protocol = {
                "path": str(EXECUTION_PROTOCOL.resolve()),
                "sha256": EXECUTION_PROTOCOL_SHA256,
            }
            hardening = {
                "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
            }
            config_without_hash = {
                "protocol_version": CONTROL_SUITE_VERSION,
                "control_protocol_v1": wrong_protocol,
                "execution_protocol_v1_6": hardening,
                "source_sha256": control_source_hashes(),
                "validation_rows_materialized": False,
                "frozen_test_used": False,
            }
            config = {
                **config_without_hash,
                "config_sha256": canonical_sha256(config_without_hash),
            }
            written = _write_bundle(
                bundle_path,
                {"synthetic": np.asarray([1.0], dtype=np.float32)},
                config,
            )
            with self.assertRaisesRegex(ValueError, "authenticated provenance"):
                _authenticate_control_bundle(
                    {"path": written["path"], "sha256": written["sha256"]},
                    config["config_sha256"],
                    _control_protocol_reference(),
                    hardening,
                )

    def _fixture(
        self, root: Path
    ) -> tuple[dict[str, dict[str, object]], Path, Path]:
        protocol_reference = {
            "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
            "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
        }
        control_protocol_reference = _control_protocol_reference()
        bundle_path = (root / "control-bundle.npz").resolve()
        bundle_config_without_hash = {
            "protocol_version": CONTROL_SUITE_VERSION,
            "control_protocol_v1": control_protocol_reference,
            "execution_protocol_v1_6": protocol_reference,
            "source_sha256": control_source_hashes(),
            "validation_rows_materialized": False,
            "frozen_test_used": False,
        }
        bundle_config = {
            **bundle_config_without_hash,
            "config_sha256": canonical_sha256(bundle_config_without_hash),
        }
        _write_bundle(
            bundle_path,
            {"synthetic": np.asarray([1.0], dtype=np.float32)},
            bundle_config,
        )
        bundle_sha256 = _file_digest(bundle_path)
        bundle_config_sha256 = bundle_config["config_sha256"]
        registry = (root / "control-registry").resolve()
        registry.mkdir()
        boundary_paths = {
            "concat_antibody_and_antigen": (
                ROOT
                / "outputs/interaction_first/research/authority_validation_nw_boundaries/concat_boundary_v2.json"
            ),
            "concat_antibody_antigen_plus_vh_vl": (
                ROOT
                / "outputs/interaction_first/research/authority_validation_nw_boundaries/chain_boundary_v2.json"
            ),
        }
        boundaries = {}
        current_controls: dict[str, dict[str, object]] = {}
        other_score_path: Path | None = None
        for mode, split_path in boundary_paths.items():
            split_payload = json.loads(split_path.read_text(encoding="utf-8"))
            output_dir = (root / f"outputs-{mode}").resolve()
            output_dir.mkdir()
            split_sha256 = _file_digest(split_path)
            authority_reference = {
                "path": str((ROOT / "outputs/data_snapshot/double_remote_split.json").resolve()),
                "sha256": _file_digest(
                    ROOT / "outputs/data_snapshot/double_remote_split.json"
                ),
            }
            prepared_reference = {
                "path": str((ROOT / "outputs/data_snapshot/instances.parquet").resolve()),
                "sha256": _file_digest(ROOT / "outputs/data_snapshot/instances.parquet"),
            }
            canonical_reference = {
                "path": str(
                    (ROOT / "outputs/interaction_first/remote/canonical_manifest.csv").resolve()
                ),
                "sha256": _file_digest(
                    ROOT / "outputs/interaction_first/remote/canonical_manifest.csv"
                ),
            }
            ledger_key, ledger_contract = control_evaluation_identity(
                bundle_sha256,
                split_sha256,
                authority_reference["sha256"],
            )
            ledger_path = (registry / f"{ledger_key}.json").resolve()
            locked_evidence = {
                "format_version": "sabdab2_control_locked_submission_evidence_v1",
                "control_protocol_v1": control_protocol_reference,
                "execution_protocol_v1_6": protocol_reference,
                "prepared_instances": prepared_reference,
                "canonical_sequence_manifest": canonical_reference,
                "training_embedding_content_manifest_sha256": {
                    "primary_antibody": "9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e",
                    "shared_esm_antibody": "03a9e0e4b4acf8b3b76cca651d65c4f278f5ac5a8985d3b12fe001da7809bf36",
                    "esm_antigen": "d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e",
                },
                "boundary_verification_v2": {
                    "path": str(BOUNDARY_VERIFICATION.resolve()),
                    "sha256": BOUNDARY_VERIFICATION_SHA256,
                },
                "authority_split": authority_reference,
                "selected_boundary": {
                    "constraint_mode": mode,
                    "path": str(split_path.resolve()),
                    "sha256": split_sha256,
                    "scientific_payload_sha256": split_payload[
                        "scientific_payload_sha256"
                    ],
                },
                "required_boundary_modes": [
                    "concat_antibody_and_antigen",
                    "concat_antibody_antigen_plus_vh_vl",
                ],
                "control_suite_completion_required": True,
                "frozen_test_used": False,
            }
            ledger_outputs = {}
            control_records = {}
            for name in CONTROL_NAMES:
                evaluation_path = output_dir / f"{name}.evaluation.json"
                score_path = output_dir / f"{name}.scores.npz"
                score_values = np.eye(3, dtype=np.float32)
                positive_mask = np.eye(3, dtype=bool)
                score_reference = write_score_artifact(
                    score_path,
                    {
                        "similarity_matrix": score_values,
                        "positive_mask": positive_mask,
                        "antibody_keys": sorted(
                            _digest(f"fixture-ab-{index}") for index in range(3)
                        ),
                        "antigen_keys": sorted(
                            _digest(f"fixture-ag-{index}") for index in range(3)
                        ),
                    },
                    {
                        "control_protocol_v1": control_protocol_reference,
                        "control_protocol_v1_sha256": CONTROL_PROTOCOL_SHA256,
                    },
                )
                contract_sha256 = score_reference["contract_sha256"]
                evaluation = {
                    "checkpoint": str(bundle_path),
                    "checkpoint_sha256": bundle_sha256,
                    "control_bundle_sha256": bundle_sha256,
                    "control_protocol_v1": control_protocol_reference,
                    "execution_protocol_v1_6": protocol_reference,
                    "locked_submission_evidence": locked_evidence,
                    "dataset": {
                        "split_manifest_sha256": _file_digest(split_path),
                        "strict_nw_constraint_mode": mode,
                    },
                    "control": {
                        "name": name,
                        "protocol_version": CONTROL_SUITE_VERSION,
                        "bundle_config_sha256": bundle_config_sha256,
                    },
                    "one_shot_ledger": {
                        "canonical_key": ledger_key,
                        "canonical_path": str(ledger_path),
                        "registry_dir": str(registry),
                    },
                    "score_artifact": {
                        "sha256": _file_digest(score_path),
                        "contract_sha256": contract_sha256,
                    },
                }
                evaluation_path.write_text(
                    json.dumps(evaluation, sort_keys=True, allow_nan=False),
                    encoding="utf-8",
                )
                ledger_outputs[name] = {
                    "evaluation_json": str(evaluation_path),
                    "evaluation_json_sha256": _file_digest(evaluation_path),
                    "score_artifact": str(score_path),
                    "score_artifact_sha256": _file_digest(score_path),
                    "score_contract_sha256": contract_sha256,
                    "bidirectional_mrr": 1.0,
                }
                control_records[name] = {
                    "evaluation_json": {
                        "path": str(evaluation_path),
                        "sha256": _file_digest(evaluation_path),
                    },
                    "score_artifact": {
                        "path": str(score_path),
                        "sha256": _file_digest(score_path),
                        "contract_sha256": contract_sha256,
                    },
                }
                if mode == "concat_antibody_and_antigen":
                    verified = _verified(
                        name,
                        np.eye(3, dtype=np.float32),
                        np.eye(3, dtype=bool),
                        primary=False,
                    )
                    verified["evaluation"] = evaluation
                    verified["evaluation_path"] = evaluation_path.resolve()
                    verified["evaluation_sha256"] = _file_digest(evaluation_path)
                    verified["artifact"]["path"] = score_path.resolve()
                    verified["artifact"]["sha256"] = _file_digest(score_path)
                    current_controls[name] = verified
                elif name == CONTROL_NAMES[-1]:
                    other_score_path = score_path
            ledger = {
                **ledger_contract,
                "status": "completed",
                "evaluation_identity_sha256": ledger_key,
                "requested_output_dir": str(output_dir),
                "control_protocol_v1": control_protocol_reference,
                "execution_protocol_v1_6": protocol_reference,
                "locked_submission_evidence_sha256": canonical_sha256(
                    locked_evidence
                ),
                "control_outputs": ledger_outputs,
            }
            ledger_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            if mode == "concat_antibody_and_antigen":
                for name in CONTROL_NAMES:
                    current_controls[name]["one_shot_verification"] = {
                        "path": str(ledger_path),
                        "sha256": _file_digest(ledger_path),
                    }
            boundaries[mode] = {
                "split_manifest": {
                    "path": str(split_path.resolve()),
                    "sha256": _file_digest(split_path),
                },
                "boundary_scientific_payload_sha256": split_payload[
                    "scientific_payload_sha256"
                ],
                "ledger": {
                    "path": str(ledger_path),
                    "sha256": _file_digest(ledger_path),
                    "canonical_key": ledger_key,
                },
                "output_dir": str(output_dir),
                "controls": control_records,
            }
        receipt = {
            "format_version": "sabdab2_control_two_boundary_suite_completion_v1",
            "status": "completed",
            "control_suite_version": CONTROL_SUITE_VERSION,
            "control_bundle": {
                "path": str(bundle_path),
                "sha256": bundle_sha256,
                "config_sha256": bundle_config_sha256,
            },
            "control_protocol_v1": _control_protocol_reference(),
            "execution_protocol_v1_6": protocol_reference,
            "boundary_verification_v2": {
                "path": str(BOUNDARY_VERIFICATION.resolve()),
                "sha256": BOUNDARY_VERIFICATION_SHA256,
            },
            "authority_split": {
                "path": str((ROOT / "outputs/data_snapshot/double_remote_split.json").resolve()),
                "sha256": _file_digest(
                    ROOT / "outputs/data_snapshot/double_remote_split.json"
                ),
            },
            "immutable_inputs": {
                "prepared_instances": {
                    "path": str((ROOT / "outputs/data_snapshot/instances.parquet").resolve()),
                    "sha256": _file_digest(
                        ROOT / "outputs/data_snapshot/instances.parquet"
                    ),
                },
                "canonical_sequence_manifest": {
                    "path": str(
                        (ROOT / "outputs/interaction_first/remote/canonical_manifest.csv").resolve()
                    ),
                    "sha256": _file_digest(
                        ROOT / "outputs/interaction_first/remote/canonical_manifest.csv"
                    ),
                },
                "training_embedding_content_manifest_sha256": {
                    "primary_antibody": "9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e",
                    "shared_esm_antibody": "03a9e0e4b4acf8b3b76cca651d65c4f278f5ac5a8985d3b12fe001da7809bf36",
                    "esm_antigen": "d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e",
                },
            },
            "required_boundary_modes": [
                "concat_antibody_and_antigen",
                "concat_antibody_antigen_plus_vh_vl",
            ],
            "boundaries": boundaries,
            "frozen_test_used": False,
        }
        receipt["completion_identity_sha256"] = canonical_sha256(receipt)
        receipt_path = registry / "suite-completions" / f"{bundle_sha256}.json"
        receipt_path.parent.mkdir()
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        if other_score_path is None:
            raise AssertionError("Synthetic control completion lacks the second boundary")
        return current_controls, receipt_path, other_score_path

    def test_both_control_boundaries_and_all_five_controls_authenticate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, _, _ = self._fixture(Path(directory))
            result = _authenticate_control_suite_completion(
                controls,
                {
                    "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                    "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                },
                _control_protocol_reference(),
            )
        self.assertTrue(result["both_mandatory_boundaries_completed"])
        self.assertTrue(result["all_five_controls_completed_per_boundary"])

    def test_missing_control_completion_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, receipt_path, _ = self._fixture(Path(directory))
            receipt_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "completion receipt is missing"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )

    def test_control_completion_rehashes_the_other_boundary_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, _, other_score = self._fixture(Path(directory))
            other_score.write_bytes(b"tampered-other-boundary-control-score")
            with self.assertRaisesRegex(ValueError, "physical SHA-256 mismatch"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )

    def test_control_completion_identity_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, receipt_path, _ = self._fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["status"] = "partial"
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "violates its identity"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )

    def test_control_completion_receipt_cannot_substitute_the_controls_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, receipt_path, _ = self._fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("completion_identity_sha256")
            receipt["control_protocol_v1"] = {
                "path": str(EXECUTION_PROTOCOL.resolve()),
                "sha256": EXECUTION_PROTOCOL_SHA256,
            }
            receipt["completion_identity_sha256"] = canonical_sha256(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different controls protocol"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )

    def test_other_boundary_ledger_cannot_substitute_the_controls_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, receipt_path, _ = self._fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("completion_identity_sha256")
            boundary = receipt["boundaries"][
                "concat_antibody_antigen_plus_vh_vl"
            ]
            ledger_path = Path(boundary["ledger"]["path"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["control_protocol_v1"] = {
                "path": str(EXECUTION_PROTOCOL.resolve()),
                "sha256": EXECUTION_PROTOCOL_SHA256,
            }
            ledger_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            boundary["ledger"]["sha256"] = _file_digest(ledger_path)
            receipt["completion_identity_sha256"] = canonical_sha256(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ledger.*inconsistent"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )

    def test_other_boundary_evaluation_cannot_substitute_the_controls_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, receipt_path, _ = self._fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("completion_identity_sha256")
            boundary = receipt["boundaries"][
                "concat_antibody_antigen_plus_vh_vl"
            ]
            control_name = CONTROL_NAMES[0]
            evaluation_reference = boundary["controls"][control_name][
                "evaluation_json"
            ]
            evaluation_path = Path(evaluation_reference["path"])
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["control_protocol_v1"] = {
                "path": str(EXECUTION_PROTOCOL.resolve()),
                "sha256": EXECUTION_PROTOCOL_SHA256,
            }
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            evaluation_sha256 = _file_digest(evaluation_path)
            evaluation_reference["sha256"] = evaluation_sha256
            ledger_path = Path(boundary["ledger"]["path"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["control_outputs"][control_name][
                "evaluation_json_sha256"
            ] = evaluation_sha256
            ledger_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            boundary["ledger"]["sha256"] = _file_digest(ledger_path)
            receipt["completion_identity_sha256"] = canonical_sha256(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output.*inconsistent"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )

    def test_other_boundary_score_context_cannot_substitute_the_controls_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controls, receipt_path, _ = self._fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("completion_identity_sha256")
            boundary = receipt["boundaries"][
                "concat_antibody_antigen_plus_vh_vl"
            ]
            control_name = CONTROL_NAMES[0]
            control_record = boundary["controls"][control_name]
            score_path = Path(control_record["score_artifact"]["path"])
            score = load_score_artifact(score_path)
            score_path.unlink()
            replacement = write_score_artifact(
                score_path,
                {
                    "similarity_matrix": score["similarities"],
                    "positive_mask": score["positive_mask"],
                    "antibody_keys": score["antibody_keys"],
                    "antigen_keys": score["antigen_keys"],
                },
                {
                    "control_protocol_v1": {
                        "path": str(EXECUTION_PROTOCOL.resolve()),
                        "sha256": EXECUTION_PROTOCOL_SHA256,
                    },
                    "control_protocol_v1_sha256": CONTROL_PROTOCOL_SHA256,
                },
            )
            control_record["score_artifact"] = {
                "path": replacement["path_at_creation"],
                "sha256": replacement["sha256"],
                "contract_sha256": replacement["contract_sha256"],
            }
            evaluation_path = Path(control_record["evaluation_json"]["path"])
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["score_artifact"] = replacement
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            evaluation_sha256 = _file_digest(evaluation_path)
            control_record["evaluation_json"]["sha256"] = evaluation_sha256
            ledger_path = Path(boundary["ledger"]["path"])
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger_record = ledger["control_outputs"][control_name]
            ledger_record["evaluation_json_sha256"] = evaluation_sha256
            ledger_record["score_artifact_sha256"] = replacement["sha256"]
            ledger_record["score_contract_sha256"] = replacement[
                "contract_sha256"
            ]
            ledger_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            boundary["ledger"]["sha256"] = _file_digest(ledger_path)
            receipt["completion_identity_sha256"] = canonical_sha256(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output.*inconsistent"):
                _authenticate_control_suite_completion(
                    controls,
                    {
                        "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                        "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                    },
                    _control_protocol_reference(),
                )


class AuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        submission_patcher = patch(
            "sabdab2_retrieval_control_comparator._authenticate_locked_submission_evidence",
            return_value={
                "authenticated": True,
                "protocol_v1_6_execution_hardening": {
                    "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                    "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                },
            },
        )
        submission_patcher.start()
        self.addCleanup(submission_patcher.stop)
        completion_patcher = patch(
            "sabdab2_retrieval_control_comparator._authenticate_control_suite_completion",
            return_value={"both_mandatory_boundaries_completed": True},
        )
        completion_patcher.start()
        self.addCleanup(completion_patcher.stop)
        self.mask = np.eye(8, dtype=bool)
        self.primary_scores = np.eye(8, dtype=np.float32)
        self.control_scores = np.ones((8, 8), dtype=np.float32)
        np.fill_diagonal(self.control_scores, -1.0)
        self.primary = _verified(
            "primary", self.primary_scores, self.mask, primary=True
        )
        self.controls = {
            name: _verified(name, self.control_scores, self.mask, primary=False)
            for name in CONTROL_NAMES
        }
        self.specs = _validate_control_arguments(_raw_control_specs())

    def _side_effect(
        self, evaluation: Path, evaluation_sha: str, score: Path, score_sha: str
    ) -> dict[str, object]:
        del evaluation_sha, score, score_sha
        if evaluation.name == "primary.evaluation.json":
            return self.primary
        name = evaluation.name.removesuffix(".evaluation.json")
        return self.controls[name]

    def test_full_suite_authentication_accepts_one_common_identity(self) -> None:
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ) as verifier:
            primary, controls, authentication = _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )
        self.assertEqual(verifier.call_count, 6)
        self.assertIs(primary, self.primary)
        self.assertEqual(tuple(controls), CONTROL_NAMES)
        self.assertEqual(
            authentication["control_bundle_sha256"], _digest("control-bundle")
        )

    def test_boundary_or_library_mismatch_is_rejected(self) -> None:
        self.controls[CONTROL_NAMES[-1]]["evaluation"]["dataset"][
            "split_manifest_sha256"
        ] = _digest("different-boundary")
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(ValueError, "does not share the primary boundary"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )

    def test_positive_graph_mismatch_is_rejected(self) -> None:
        target = self.controls[CONTROL_NAMES[-1]]["artifact"]
        changed = np.roll(self.mask, 1, axis=1)
        target["positive_mask"] = changed
        target["array_sha256"]["positive_mask"] = _score_array_sha256(changed)
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(ValueError, "positive graph"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )

    def test_wrong_primary_role_is_rejected(self) -> None:
        self.primary["evaluation"]["checkpoint_run_mode"] = "train_only_control_bundle"
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(PermissionError, "not the locked final refit"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )

    def test_wrong_control_identity_or_protocol_binding_is_rejected(self) -> None:
        target = self.controls[CONTROL_NAMES[0]]["evaluation"]
        target["control"]["name"] = CONTROL_NAMES[1]
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(ValueError, "identity mismatch"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )
        target["control"]["name"] = CONTROL_NAMES[0]
        target["execution_protocol_v1_6"] = {
            "path": str(EXECUTION_PROTOCOL.resolve()),
            "sha256": _digest("wrong-protocol"),
        }
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(ValueError, "not bound"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )
        target["execution_protocol_v1_6"] = {
            "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
            "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
        }
        target["control_protocol_v1"] = {
            "path": str(EXECUTION_PROTOCOL.resolve()),
            "sha256": EXECUTION_PROTOCOL_SHA256,
        }
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(ValueError, "controls protocol"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )

    def test_controls_must_share_one_completed_central_ledger(self) -> None:
        self.controls[CONTROL_NAMES[-1]]["one_shot_verification"]["path"] = (
            "/synthetic/alternate-ledger.json"
        )
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._side_effect,
        ), self.assertRaisesRegex(ValueError, "one completed central ledger"):
            _authenticate_suite(
                _primary_spec(),
                self.specs,
                _protocol_reference(),
                _control_protocol_reference(),
            )


class StatisticalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        submission_patcher = patch(
            "sabdab2_retrieval_control_comparator._authenticate_locked_submission_evidence",
            return_value={
                "authenticated": True,
                "protocol_v1_6_execution_hardening": {
                    "path": str(EXECUTION_HARDENING_PROTOCOL.resolve()),
                    "sha256": EXECUTION_HARDENING_PROTOCOL_SHA256,
                },
            },
        )
        submission_patcher.start()
        self.addCleanup(submission_patcher.stop)
        completion_patcher = patch(
            "sabdab2_retrieval_control_comparator._authenticate_control_suite_completion",
            return_value={"both_mandatory_boundaries_completed": True},
        )
        completion_patcher.start()
        self.addCleanup(completion_patcher.stop)
        self.mask = np.eye(8, dtype=bool)
        self.primary_scores = np.eye(8, dtype=np.float32)
        self.lower_scores = np.ones((8, 8), dtype=np.float32)
        np.fill_diagonal(self.lower_scores, -1.0)

    def _suite_side_effect(
        self,
        controls: dict[str, dict[str, object]],
        primary: dict[str, object],
    ):
        def side_effect(
            evaluation: Path, evaluation_sha: str, score: Path, score_sha: str
        ) -> dict[str, object]:
            del evaluation_sha, score, score_sha
            if evaluation.name == "primary.evaluation.json":
                return primary
            return controls[evaluation.name.removesuffix(".evaluation.json")]

        return side_effect

    def test_all_five_positive_paired_lower_bounds_pass_the_gate(self) -> None:
        primary = _verified("primary", self.primary_scores, self.mask, primary=True)
        controls = {
            name: _verified(name, self.lower_scores, self.mask, primary=False)
            for name in CONTROL_NAMES
        }
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._suite_side_effect(controls, primary),
        ):
            report = run_comparison(_arguments())
        self.assertEqual(LOCKED_BOOTSTRAP_COUNT, 10_000)
        self.assertEqual(LOCKED_BOOTSTRAP_SEED, 42)
        self.assertEqual(report["preregistered_gate_update"]["gate"], GATE_NAME)
        self.assertEqual(report["preregistered_gate_update"]["status"], "passed")
        self.assertFalse(report["authentication"]["excluded_validation_rows_read"])
        for comparison in report["paired_primary_minus_control"].values():
            self.assertEqual(comparison["gate_status"], "passed")
            self.assertEqual(
                comparison["selected_interval_source"],
                "joint_bipartite_component_bootstrap",
            )
            for direction in (
                "antibody_to_antigen",
                "antigen_to_antibody",
                "bidirectional",
            ):
                self.assertIn("mrr", comparison["gate_relevant_effects"][direction])
                self.assertIn(
                    "recall_at_1", comparison["gate_relevant_effects"][direction]
                )

    def test_one_nonpositive_lower_bound_fails_the_global_gate(self) -> None:
        primary = _verified("primary", self.primary_scores, self.mask, primary=True)
        controls = {
            name: _verified(name, self.lower_scores, self.mask, primary=False)
            for name in CONTROL_NAMES
        }
        controls[CONTROL_NAMES[-1]] = _verified(
            CONTROL_NAMES[-1], self.primary_scores, self.mask, primary=False
        )
        with patch(
            "sabdab2_retrieval_control_comparator._authenticate_pair",
            side_effect=self._suite_side_effect(controls, primary),
        ):
            report = run_comparison(_arguments())
        self.assertEqual(report["preregistered_gate_update"]["status"], "failed")
        self.assertEqual(
            report["preregistered_gate_update"]["per_control_status"][
                CONTROL_NAMES[-1]
            ],
            "failed",
        )

    def test_anchor_bootstrap_is_used_only_when_joint_components_are_unavailable(self) -> None:
        metric_names = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20")
        primary = {
            direction: {metric: np.asarray([1.0, 1.0]) for metric in metric_names}
            for direction in ("antibody_to_antigen", "antigen_to_antibody")
        }
        control = {
            direction: {metric: np.asarray([0.0, 0.0]) for metric in metric_names}
            for direction in ("antibody_to_antigen", "antigen_to_antibody")
        }
        comparison = _paired_comparison(
            primary,
            control,
            np.ones((2, 2), dtype=bool),
        )
        self.assertEqual(
            comparison["selected_interval_source"],
            "paired_anchor_bootstrap_independent_directions",
        )
        self.assertIsNotNone(comparison["paired_anchor_bootstrap_sensitivity"])
        self.assertEqual(comparison["gate_status"], "passed")

    def test_directional_mrr_and_r_at_1_are_reported_but_not_gate_criteria(self) -> None:
        metric_names = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20")
        primary = {
            "antibody_to_antigen": {
                metric: np.full(8, 0.0 if metric == "recall_at_1" else 0.4)
                for metric in metric_names
            },
            "antigen_to_antibody": {
                metric: np.full(8, 1.0)
                for metric in metric_names
            },
        }
        control = {
            "antibody_to_antigen": {
                metric: np.full(8, 1.0 if metric == "recall_at_1" else 0.5)
                for metric in metric_names
            },
            "antigen_to_antibody": {
                metric: np.full(8, 0.0)
                for metric in metric_names
            },
        }
        comparison = _paired_comparison(
            primary,
            control,
            np.eye(8, dtype=bool),
        )
        self.assertLess(
            comparison["gate_relevant_effects"]["antibody_to_antigen"]["mrr"][
                "lower"
            ],
            0.0,
        )
        self.assertEqual(
            comparison["gate_relevant_effects"]["bidirectional"]["recall_at_1"][
                "lower"
            ],
            0.0,
        )
        self.assertEqual(comparison["gate_status"], "passed")


if __name__ == "__main__":
    unittest.main()
