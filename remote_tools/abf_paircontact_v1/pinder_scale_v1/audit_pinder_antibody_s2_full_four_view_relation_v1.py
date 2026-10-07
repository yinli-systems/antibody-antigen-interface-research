#!/usr/bin/env python3
"""Materialize the complete exact 80/80 relation on PINDER-train sequences.

Every unordered pair of unique sequences is covered in each of four views:
VH, non-empty VL, concatenated antibody, and antigen.  The implementation
reuses the locked Biopython/BLOSUM62 global aligner from the independent split
auditor.  The emitted ledgers contain digests and alignment statistics only;
raw sequences, system identifiers, structures, labels, predictions, external
validation rows, and test rows are never written or read.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import audit_pinder_antibody_s2_exact_four_view_v1 as parent


FORMAT_VERSION = "pinder_antibody_s2_full_four_view_relation_v1"
REQUIRED_VIEWS = parent.REQUIRED_VIEWS


class FullRelationError(RuntimeError):
    """Raised when a complete-relation input or accounting invariant fails."""


def build_complete_inventories(
    rows: Sequence[parent.SourceRow],
) -> dict[str, tuple[tuple[str, str], ...]]:
    inventories: dict[str, dict[str, str]] = {
        view: {} for view in REQUIRED_VIEWS
    }
    for row in rows:
        values = {
            "vh": row.vh,
            "vl": row.vl,
            "concatenated_antibody": row.concatenated_antibody,
            "antigen": row.antigen,
        }
        for view, sequence in values.items():
            if sequence:
                inventories[view][parent.sequence_sha256(sequence)] = sequence
    result = {
        view: tuple(sorted(inventory.items()))
        for view, inventory in inventories.items()
    }
    if any(not result[view] for view in REQUIRED_VIEWS):
        raise FullRelationError("a required sequence view is empty")
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


def _unordered_edge_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    left = row.get("left_sequence_sha256")
    right = row.get("right_sequence_sha256")
    if not isinstance(left, str) or not isinstance(right, str):
        raise FullRelationError("crossing row has invalid digests")
    if left == right:
        return None
    return (left, right) if left < right else None


def audit_complete_view(
    view: str,
    inventory: tuple[tuple[str, str], ...],
    target_path: Path,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], parent.StagedArtifact]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".staged", dir=target_path.parent
    )
    temporary = Path(temporary_name)
    count = len(inventory)
    expected_computations = count * count
    expected_unordered_pairs = count * (count - 1) // 2
    counts = {"candidate": 0, "aligned": 0, "pruned": 0, "edge": 0}
    seen: set[tuple[str, str]] = set()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for partial in parent._iter_partial_results(
                inventory,
                inventory,
                view,
                -1,
                -1,
                workers,
                chunk_size,
            ):
                counts["candidate"] += partial["candidate_pair_count"]
                counts["aligned"] += partial["dynamic_programming_alignment_count"]
                counts["pruned"] += partial[
                    "provably_pruned_by_length_upper_bound_count"
                ]
                for row in partial["crossing_rows"]:
                    key = _unordered_edge_key(row)
                    if key is None:
                        continue
                    if key in seen:
                        raise FullRelationError("duplicate canonical unordered edge")
                    seen.add(key)
                    canonical_row = dict(row)
                    canonical_row["complete_relation_unordered_pair"] = True
                    _write_jsonl(handle, canonical_row)
                    counts["edge"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        if counts["candidate"] != expected_computations:
            raise FullRelationError("complete Cartesian enumeration is incomplete")
        if counts["aligned"] + counts["pruned"] != expected_computations:
            raise FullRelationError("alignment/prune accounting is incomplete")
        if counts["edge"] != len(seen):
            raise FullRelationError("edge accounting mismatch")
        artifact = parent.StagedArtifact(
            temporary_path=temporary,
            target_path=target_path,
            sha256=parent.sha256_file(temporary),
            size_bytes=temporary.stat().st_size,
            edge_count=counts["edge"],
        )
        return {
            "sequence_view": view,
            "unique_sequence_count": count,
            "unique_unordered_sequence_pair_count": expected_unordered_pairs,
            "executed_ordered_pair_alignment_or_prune_count": counts["candidate"],
            "dynamic_programming_alignment_count": counts["aligned"],
            "provably_pruned_by_length_upper_bound_count": counts["pruned"],
            "qualifying_unordered_edge_count": counts["edge"],
            "all_unordered_pairs_covered": True,
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
        sha256=parent.sha256_file(temporary),
        size_bytes=temporary.stat().st_size,
        edge_count=0,
    )


def _publish(artifact: parent.StagedArtifact) -> None:
    if parent.sha256_file(artifact.temporary_path) != artifact.sha256:
        raise FullRelationError("staged artifact changed before publication")
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
        view: output_dir / f"{view}.identity80_coverage80.complete_edges.jsonl"
        for view in REQUIRED_VIEWS
    }
    result_target = output_dir / "PINDER_ANTIBODY_S2_FULL_FOUR_VIEW_RELATION_RESULT_V1.json"
    receipt_target = output_dir / "PINDER_ANTIBODY_S2_FULL_FOUR_VIEW_RELATION_RECEIPT_V1.json"
    if any(path.exists() for path in (*edge_targets.values(), result_target, receipt_target)):
        raise FileExistsError("create-only full-relation output already exists")

    inputs = {
        "canonical_manifest": parent.verify_file(
            arguments.canonical_manifest,
            arguments.canonical_manifest_sha256,
            "canonical manifest",
        ),
        "auditor_source": parent.verify_file(
            Path(__file__), arguments.auditor_source_sha256, "full-relation source"
        ),
    }
    rows = parent.load_source_rows(Path(inputs["canonical_manifest"]["path"]))
    inventories = build_complete_inventories(rows)
    parent.build_locked_aligner()
    runtime = parent.runtime_provenance(arguments.workers, arguments.chunk_size)

    staged_edges: list[parent.StagedArtifact] = []
    staged_result: parent.StagedArtifact | None = None
    staged_receipt: parent.StagedArtifact | None = None
    try:
        results: dict[str, dict[str, Any]] = {}
        for view in REQUIRED_VIEWS:
            result, artifact = audit_complete_view(
                view,
                inventories[view],
                edge_targets[view],
                arguments.workers,
                arguments.chunk_size,
            )
            results[view] = result
            staged_edges.append(artifact)
            print(
                f"completed view={view} unique_pairs="
                f"{result['unique_unordered_sequence_pair_count']} "
                f"edges={result['qualifying_unordered_edge_count']}",
                flush=True,
            )
        for label, reference in inputs.items():
            if parent.sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise FullRelationError(f"{label} changed during full audit")
        result_payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "status": "PASS_COMPLETE_EXACT_FOUR_VIEW_S2_80_RELATION_MATERIALIZED",
            "authorized_for_training": False,
            "authorized_for_model_selection": False,
            "inputs": inputs,
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
            "results": results,
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
            "checks": {
                "all_four_views_complete": True,
                "all_unordered_unique_sequence_pairs_covered": True,
                "canonical_edge_orientation_is_digest_ascending": True,
                "raw_sequences_or_system_ids_written": False,
            },
            "runtime": runtime,
            "data_firewall": {
                "source_pool": "PINDER_train_only",
                "external_validation_rows_read": False,
                "test_rows_read": False,
                "task_labels_read": False,
                "structures_read": False,
                "predictions_or_scores_read": False,
            },
            "required_next_gate": (
                "rebuild a provisional train-validation split from these complete "
                "components, then independently verify zero boundary crossings"
            ),
            "claim_boundary": (
                "This artifact is a complete sequence-similarity relation for split "
                "construction. It is not a performance result or training authorization."
            ),
        }
        result_payload["scientific_payload_sha256"] = parent.canonical_payload_sha256(
            result_payload
        )
        staged_result = _stage_bytes(
            result_target, parent.canonical_json_bytes(result_payload)
        )
        receipt_payload = {
            "format_version": "pinder_antibody_s2_full_four_view_relation_receipt_v1",
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
                    "edge_counts": {
                        view: results[view]["qualifying_unordered_edge_count"]
                        for view in REQUIRED_VIEWS
                    },
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
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--auditor-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=256)
    arguments = parser.parse_args(argv)
    if arguments.workers < 1 or arguments.chunk_size < 1:
        parser.error("workers and chunk size must be positive")
    for name in ("canonical_manifest_sha256", "auditor_source_sha256"):
        value = getattr(arguments, name)
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            parser.error(f"{name} is not a lowercase SHA-256")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
