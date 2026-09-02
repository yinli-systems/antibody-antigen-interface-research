#!/usr/bin/env python3
"""Independently audit one PINDER-antibody train/validation S2 assignment.

The auditor authenticates the create-only provisional assignment, rebuilds all
train/validation/discarded ledgers from the canonical train-authority manifest,
and exhaustively aligns every train-versus-validation unique sequence pair in
VH, non-empty VL, concatenated-antibody, and antigen views.  It authorizes the
split only when every view has zero inclusive 80/80 crossing edges.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import audit_pinder_antibody_s2_exact_four_view_v1 as parent


FORMAT_VERSION = "pinder_antibody_s2_train_validation_exact_audit_v1"
ASSIGNMENT_FORMAT_VERSION = "pinder_antibody_s2_train_validation_assignment_v1"
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")


class AuditError(RuntimeError):
    """Raised when an authenticated input or audit invariant fails."""


def _membership(values: Iterable[str]) -> str:
    return parent.membership_sha256(values)


def _validate_ledger(
    ledger: Any,
    bucket: str,
    source_by_id: Mapping[str, parent.SourceRow],
) -> set[str]:
    if not isinstance(ledger, Mapping) or ledger.get("bucket") != bucket:
        raise AuditError(f"{bucket} ledger is missing or mislabeled")

    def values(name: str, count_name: str, sha_name: str) -> list[str]:
        observed = ledger.get(name)
        if not isinstance(observed, list) or not all(
            isinstance(value, str) for value in observed
        ):
            raise AuditError(f"{bucket} {name} is not a string list")
        if observed != sorted(observed) or len(observed) != len(set(observed)):
            raise AuditError(f"{bucket} {name} is not sorted and unique")
        if ledger.get(count_name) != len(observed):
            raise AuditError(f"{bucket} {name} count mismatch")
        if ledger.get(sha_name) != _membership(observed):
            raise AuditError(f"{bucket} {name} membership SHA mismatch")
        return observed

    system_ids = values(
        "system_ids", "system_count", "system_membership_sha256"
    )
    pair_ids = values(
        "exact_pair_ids",
        "unique_exact_pair_count",
        "exact_pair_membership_sha256",
    )
    pdb_ids = values("pdb_ids", "unique_pdb_count", "pdb_membership_sha256")
    if not set(system_ids).issubset(source_by_id):
        raise AuditError(f"{bucket} contains an unknown system")
    observed_pairs = {source_by_id[value].exact_pair_id for value in system_ids}
    observed_pdbs = {source_by_id[value].pdb_id for value in system_ids}
    if observed_pairs != set(pair_ids):
        raise AuditError(f"{bucket} exact-pair ledger does not reconstruct")
    if observed_pdbs != set(pdb_ids):
        raise AuditError(f"{bucket} PDB ledger does not reconstruct")
    return set(system_ids)


def validate_assignment(
    assignment: Mapping[str, Any],
    rows: Sequence[parent.SourceRow],
    expected_scientific_sha256: str,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    if assignment.get("format_version") != ASSIGNMENT_FORMAT_VERSION:
        raise AuditError("assignment format mismatch")
    if assignment.get("status") != (
        "PASS_PROVISIONAL_TRAIN_VALIDATION_S2_80_ASSIGNMENT"
    ):
        raise AuditError("assignment did not pass its provisional gate")
    if assignment.get("approval_status") != (
        "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_ZERO_CROSSING_AUDIT"
    ):
        raise AuditError("assignment approval boundary mismatch")
    if assignment.get("authorized_for_training") is not False:
        raise AuditError("provisional assignment unexpectedly authorized training")
    if assignment.get("authorized_for_model_selection") is not False:
        raise AuditError("provisional assignment unexpectedly authorized selection")
    claimed = assignment.get("scientific_payload_sha256")
    scientific = dict(assignment)
    scientific.pop("scientific_payload_sha256", None)
    observed = parent.canonical_payload_sha256(scientific)
    if claimed != expected_scientific_sha256 or observed != expected_scientific_sha256:
        raise AuditError("assignment scientific payload SHA mismatch")

    source_by_id = {row.system_id: row for row in rows}
    ledgers = assignment.get("bucket_ledgers")
    if not isinstance(ledgers, Mapping) or set(ledgers) != {
        "train",
        "validation",
        "discarded",
    }:
        raise AuditError("assignment does not contain exactly three bucket ledgers")
    bucket_ids = {
        bucket: _validate_ledger(ledgers[bucket], bucket, source_by_id)
        for bucket in ("train", "validation", "discarded")
    }
    if any(
        bucket_ids[left] & bucket_ids[right]
        for left, right in (
            ("train", "validation"),
            ("train", "discarded"),
            ("validation", "discarded"),
        )
    ):
        raise AuditError("bucket system ledgers overlap")
    if set().union(*bucket_ids.values()) != set(source_by_id):
        raise AuditError("bucket system ledgers are not exhaustive")

    bucket_by_system = {
        system_id: bucket
        for bucket, identifiers in bucket_ids.items()
        for system_id in identifiers
    }
    pair_buckets: dict[str, set[str]] = {}
    pdb_buckets: dict[str, set[str]] = {}
    for row in rows:
        pair_buckets.setdefault(row.exact_pair_id, set()).add(
            bucket_by_system[row.system_id]
        )
        pdb_buckets.setdefault(row.pdb_id, set()).add(bucket_by_system[row.system_id])
    if any(len(buckets) != 1 for buckets in pair_buckets.values()):
        raise AuditError("an exact pair is split across buckets")
    if any(len(buckets) != 1 for buckets in pdb_buckets.values()):
        raise AuditError("a PDB is split across buckets")

    checks = assignment.get("checks")
    if not isinstance(checks, Mapping):
        raise AuditError("assignment checks are missing")
    forbidden_true_or_nonfalse = (
        "validation_rows_read",
        "test_rows_read",
        "task_labels_read",
        "structures_read",
        "predictions_or_scores_read",
    )
    if any(checks.get(name) is not False for name in forbidden_true_or_nonfalse):
        raise AuditError("assignment data firewall is not closed")
    return bucket_ids, {
        "all_source_systems_accounted_once": True,
        "exact_pairs_indivisible_across_all_buckets": True,
        "pdbs_indivisible_across_all_buckets": True,
        "train_system_count": len(bucket_ids["train"]),
        "validation_system_count": len(bucket_ids["validation"]),
        "discarded_system_count": len(bucket_ids["discarded"]),
    }


def build_inventories(
    rows: Sequence[parent.SourceRow], bucket_ids: Mapping[str, set[str]]
) -> dict[str, tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]]:
    bucket_index = {"train": 0, "validation": 1}
    inventory: dict[str, list[dict[str, str]]] = {
        view: [dict(), dict()] for view in REQUIRED_VIEWS
    }
    bucket_by_system = {
        system_id: bucket
        for bucket in ("train", "validation")
        for system_id in bucket_ids[bucket]
    }
    for row in rows:
        bucket = bucket_by_system.get(row.system_id)
        if bucket is None:
            continue
        values = {
            "vh": row.vh,
            "vl": row.vl,
            "concatenated_antibody": row.concatenated_antibody,
            "antigen": row.antigen,
        }
        for view, sequence in values.items():
            if sequence:
                inventory[view][bucket_index[bucket]][
                    parent.sequence_sha256(sequence)
                ] = sequence
    result = {
        view: (tuple(sorted(values[0].items())), tuple(sorted(values[1].items())))
        for view, values in inventory.items()
    }
    for view, (train, validation) in result.items():
        if {digest for digest, _ in train} & {digest for digest, _ in validation}:
            raise AuditError(f"exact sequence overlap across train/validation in {view}")
    return result


def _write_jsonl(handle: BinaryIO, payload: Mapping[str, Any]) -> None:
    handle.write(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    )


def audit_view(
    view: str,
    inventory: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
    target_path: Path,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], parent.StagedArtifact]:
    train, validation = inventory
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".staged", dir=target_path.parent
    )
    temporary = Path(temporary_name)
    counts = {"candidate": 0, "aligned": 0, "pruned": 0, "crossing": 0}
    expected_candidates = len(train) * len(validation)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for partial in parent._iter_partial_results(
                train,
                validation,
                view,
                0,
                1,
                workers,
                chunk_size,
            ):
                counts["candidate"] += partial["candidate_pair_count"]
                counts["aligned"] += partial["dynamic_programming_alignment_count"]
                counts["pruned"] += partial[
                    "provably_pruned_by_length_upper_bound_count"
                ]
                for row in partial["crossing_rows"]:
                    row = dict(row)
                    row["left_bucket"] = "train"
                    row["right_bucket"] = "validation"
                    _write_jsonl(handle, row)
                    counts["crossing"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        if counts["candidate"] != expected_candidates:
            raise AuditError("Cartesian-product enumeration incomplete")
        if counts["aligned"] + counts["pruned"] != expected_candidates:
            raise AuditError("alignment/prune accounting incomplete")
        artifact = parent.StagedArtifact(
            temporary_path=temporary,
            target_path=target_path,
            sha256=parent.sha256_file(temporary),
            size_bytes=temporary.stat().st_size,
            edge_count=counts["crossing"],
        )
        return {
            "sequence_view": view,
            "train_unique_sequence_count": len(train),
            "validation_unique_sequence_count": len(validation),
            "train_validation_unique_sequence_pair_count": counts["candidate"],
            "dynamic_programming_alignment_count": counts["aligned"],
            "provably_pruned_by_length_upper_bound_count": counts["pruned"],
            "threshold_crossing_edge_count": counts["crossing"],
            "valid_zero_crossing_edges": counts["crossing"] == 0,
        }, artifact
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _stage_bytes(path: Path, payload: bytes) -> parent.StagedArtifact:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staged", dir=path.parent
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return parent.StagedArtifact(
        temporary_path=temporary,
        target_path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        edge_count=0,
    )


def _publish(artifact: parent.StagedArtifact) -> None:
    if parent.sha256_file(artifact.temporary_path) != artifact.sha256:
        raise AuditError("staged artifact changed before publication")
    try:
        os.link(artifact.temporary_path, artifact.target_path)
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise FileExistsError(artifact.target_path) from error
        raise


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_targets = {
        view: output_dir / f"{view}.identity80_coverage80.crossing_edges.jsonl"
        for view in REQUIRED_VIEWS
    }
    result_target = output_dir / "PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RESULT_V1.json"
    receipt_target = output_dir / "PINDER_ANTIBODY_S2_TRAIN_VALIDATION_EXACT_AUDIT_RECEIPT_V1.json"
    if any(path.exists() for path in (*edge_targets.values(), result_target, receipt_target)):
        raise FileExistsError("create-only audit output already exists")

    inputs = {
        "assignment": parent.verify_file(
            arguments.assignment, arguments.assignment_sha256, "provisional assignment"
        ),
        "canonical_manifest": parent.verify_file(
            arguments.canonical_manifest,
            arguments.canonical_manifest_sha256,
            "canonical manifest",
        ),
        "auditor_source": parent.verify_file(
            Path(__file__), arguments.auditor_source_sha256, "auditor source"
        ),
    }
    assignment = parent.strict_json(Path(inputs["assignment"]["path"]))
    rows = parent.load_source_rows(Path(inputs["canonical_manifest"]["path"]))
    bucket_ids, ledger_checks = validate_assignment(
        assignment, rows, arguments.assignment_scientific_sha256
    )
    inventories = build_inventories(rows, bucket_ids)
    parent.build_locked_aligner()
    runtime = parent.runtime_provenance(arguments.workers, arguments.chunk_size)

    staged_edges: list[parent.StagedArtifact] = []
    staged_result: parent.StagedArtifact | None = None
    staged_receipt: parent.StagedArtifact | None = None
    try:
        results: dict[str, dict[str, Any]] = {}
        for view in REQUIRED_VIEWS:
            result, artifact = audit_view(
                view,
                inventories[view],
                edge_targets[view],
                arguments.workers,
                arguments.chunk_size,
            )
            results[view] = result
            staged_edges.append(artifact)
            print(
                f"completed view={view} pairs={result['train_validation_unique_sequence_pair_count']} "
                f"crossings={result['threshold_crossing_edge_count']}",
                flush=True,
            )
        approved = all(results[view]["valid_zero_crossing_edges"] for view in REQUIRED_VIEWS)
        total_candidates = sum(
            result["train_validation_unique_sequence_pair_count"]
            for result in results.values()
        )
        total_aligned = sum(
            result["dynamic_programming_alignment_count"] for result in results.values()
        )
        total_pruned = sum(
            result["provably_pruned_by_length_upper_bound_count"]
            for result in results.values()
        )
        total_crossings = sum(
            result["threshold_crossing_edge_count"] for result in results.values()
        )
        if total_aligned + total_pruned != total_candidates:
            raise AuditError("four-view exhaustive accounting mismatch")
        for label, reference in inputs.items():
            if parent.sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise AuditError(f"{label} changed during audit")

        result_payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "status": (
                "PASS_EXACT_FOUR_VIEW_TRAIN_VALIDATION_S2_80_ZERO_CROSSING_AUDIT"
                if approved
                else "FAIL_EXACT_FOUR_VIEW_TRAIN_VALIDATION_S2_80_CROSSINGS_DETECTED"
            ),
            "approval_status": (
                "APPROVED_PINDER_TRAIN_ONLY_S2_TRAIN_VALIDATION_SPLIT"
                if approved
                else "NOT_APPROVED_REBUILD_WITH_NEW_EXACT_EDGES"
            ),
            "authorized_for_training": approved,
            "authorized_for_model_selection": approved,
            "inputs": inputs,
            "assignment_scientific_payload_sha256": arguments.assignment_scientific_sha256,
            "ledger_checks": ledger_checks,
            "policy": {
                "algorithm": "exact global dynamic programming (Needleman-Wunsch family)",
                "backend": f"Biopython PairwiseAligner {parent.EXPECTED_BIOPYTHON_VERSION}",
                "substitution_matrix": "BLOSUM62",
                "substitution_matrix_payload_sha256": parent.EXPECTED_BLOSUM62_PAYLOAD_SHA256,
                "gap_open_penalty": parent.GAP_OPEN_PENALTY,
                "gap_extend_penalty": parent.GAP_EXTEND_PENALTY,
                "aligned_region_identity_minimum": 0.8,
                "reciprocal_coverage_minimum": 0.8,
                "threshold_equality_included": True,
                "required_sequence_views": list(REQUIRED_VIEWS),
                "missing_vl_policy": "exclude genuinely empty VL from VL inventory",
                "only_pruning_rule": "raw length ratio proves reciprocal coverage below 0.8",
            },
            "inventory": {
                view: {
                    "train_unique_sequence_count": len(inventories[view][0]),
                    "validation_unique_sequence_count": len(inventories[view][1]),
                    "exact_train_validation_overlap_count": 0,
                }
                for view in REQUIRED_VIEWS
            },
            "results": results,
            "exhaustive_accounting": {
                "total_train_validation_unique_sequence_pair_count": total_candidates,
                "total_dynamic_programming_alignment_count": total_aligned,
                "total_provably_pruned_by_length_upper_bound_count": total_pruned,
                "total_threshold_crossing_edge_count": total_crossings,
                "alignments_plus_safe_prunes_equal_all_candidates": True,
            },
            "crossing_artifacts": {
                view: {
                    "path": str(edge_targets[view]),
                    "sha256": staged_edges[index].sha256,
                    "size_bytes": staged_edges[index].size_bytes,
                    "edge_count": staged_edges[index].edge_count,
                    "raw_sequences_or_system_ids_included": False,
                }
                for index, view in enumerate(REQUIRED_VIEWS)
            },
            "runtime": runtime,
            "data_firewall": {
                "source_pool": "PINDER_train_only",
                "external_validation_rows_read": False,
                "test_rows_read": False,
                "task_labels_read": False,
                "structures_read": False,
                "predictions_or_scores_read": False,
                "discarded_sequences_excluded_from_boundary_audit": len(
                    bucket_ids["discarded"]
                ),
            },
            "claim_boundary": (
                "This exact audit only authorizes one PINDER-train-only internal "
                "train/validation S2 split. It is not a performance evaluation, "
                "temporal test, structural split, or PLM pretraining leakage audit."
            ),
        }
        result_payload["scientific_payload_sha256"] = parent.canonical_payload_sha256(
            result_payload
        )
        staged_result = _stage_bytes(
            result_target, parent.canonical_json_bytes(result_payload)
        )
        receipt_payload = {
            "format_version": "pinder_antibody_s2_train_validation_exact_audit_receipt_v1",
            "result": {
                "path": str(result_target),
                "sha256": staged_result.sha256,
                "size_bytes": staged_result.size_bytes,
                "scientific_payload_sha256": result_payload[
                    "scientific_payload_sha256"
                ],
            },
            "crossing_artifacts": result_payload["crossing_artifacts"],
            "all_outputs_create_only": True,
            "all_inputs_rehashed_before_publication": True,
        }
        staged_receipt = _stage_bytes(
            receipt_target, parent.canonical_json_bytes(receipt_payload)
        )
        for artifact in (*staged_edges, staged_result, staged_receipt):
            _publish(artifact)
        print(
            json.dumps(
                {
                    "status": result_payload["status"],
                    "authorized_for_training": approved,
                    "total_threshold_crossing_edge_count": total_crossings,
                    "result_sha256": staged_result.sha256,
                    "receipt_sha256": staged_receipt.sha256,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return result_payload
    finally:
        for artifact in (*staged_edges, staged_result, staged_receipt):
            if artifact is not None:
                artifact.temporary_path.unlink(missing_ok=True)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--assignment-scientific-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--auditor-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=256)
    arguments = parser.parse_args(argv)
    if arguments.workers < 1 or arguments.chunk_size < 1:
        parser.error("workers and chunk size must be positive")
    for name in (
        "assignment_sha256",
        "assignment_scientific_sha256",
        "canonical_manifest_sha256",
        "auditor_source_sha256",
    ):
        value = getattr(arguments, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name} is not a lowercase SHA-256")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
