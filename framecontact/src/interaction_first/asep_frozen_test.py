"""Receipt-gated, label-free preparation for the frozen AsEP test role."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any

import torch

from .asep_authority import INNER_AUDIT_FORMAT
from .asep_external import (
    AsepExternalError,
    _load_split_roles,
    adapt_asep_structure_bytes,
    extract_asep_antigen_pdb_bytes,
    extract_asep_label_free_structure_metadata,
    load_asep_graph_bytes,
)
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


SELECTION_FORMAT = "framecontact_asep_development_selection_v1"
ENSEMBLE_FORMAT = "framecontact_asep_development_ensemble_v1"
LABEL_FREE_TEST_FORMAT = "framecontact_asep_label_free_test_metadata_v1"
OPENING_FORMAT = "framecontact_asep_official_test_opening_v1"
FROZEN_TEST_CACHE_FORMAT = "framecontact_asep_frozen_test_cache_v1"
FROZEN_TEST_ARTIFACT_FORMAT = "framecontact_asep_frozen_test_artifact_v1"
GRAPH_OPEN_CONSUMPTION_FORMAT = "framecontact_asep_test_graph_open_consumption_v1"
ASEP_ARCHITECTURE_MATRIX = (
    "sequence_dense",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
    "frozen_frame_ipa",
    "framecontact",
    "framecontact_interleaved",
)
EVALUATION_DEPENDENCY_PATHS = (
    "scripts/evaluate_asep_frozen_test.py",
    "scripts/evaluate_framecontact_sabdab2_ensemble_test.py",
    "scripts/freeze_asep_development_ensemble.py",
    "scripts/train_framecontact.py",
    "src/interaction_first/asep_frozen_test.py",
    "src/interaction_first/framecontact_ensemble.py",
    "src/interaction_first/framecontact_interventions.py",
    "src/interaction_first/framecontact_metrics.py",
)


def asep_evaluation_source_hashes(
    repository_root: Path | None = None,
) -> dict[str, str]:
    """Hash every direct source dependency used after the test opening."""
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else repository_root.resolve()
    )
    result = {}
    for relative in EVALUATION_DEPENDENCY_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise AsepExternalError(f"Missing regular evaluation dependency: {relative}")
        result[relative] = sha256_file(path)
    return result


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AsepExternalError(f"Cannot parse {name}") from error
    if not isinstance(payload, dict):
        raise AsepExternalError(f"{name} must be a JSON object")
    return payload


def authenticate_asep_development_selection(
    selection_root: Path,
    *,
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    """Authenticate the unique validation-only winner before test-feature access."""
    result_path = selection_root / "ASEP_DEVELOPMENT_SELECTION.json"
    receipt_path = selection_root / "SELECTION_RECEIPT.json"
    authenticate_file(receipt_path, selection_receipt_sha256, "AsEP selection receipt")
    receipt = _json(receipt_path, "AsEP selection receipt")
    authenticate_file(
        result_path, str(receipt.get("result_sha256", "")), "AsEP selection result"
    )
    result = _json(result_path, "AsEP selection result")
    selected = result.get("selected_ensemble", {})
    candidates = result.get("all_candidates", [])
    selection_rule = result.get("selection_rule", {})
    candidate_experiments = [
        candidate.get("experiment")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    selected_metric = selected.get("selection_metric")
    selected_contact = selected.get("contact_mean_per_complex_auprc")
    candidate_scores_valid = all(
        isinstance(candidate.get("selection_metric"), (int, float))
        and math.isfinite(float(candidate["selection_metric"]))
        and isinstance(
            candidate.get("contact_mean_per_complex_auprc"), (int, float)
        )
        and math.isfinite(float(candidate["contact_mean_per_complex_auprc"]))
        for candidate in candidates
        if isinstance(candidate, dict)
    )
    architecture_index = {
        experiment: index
        for index, experiment in enumerate(ASEP_ARCHITECTURE_MATRIX)
    }
    candidate_order_valid = all(
        (
            (
                float(left["selection_metric"]),
                float(left["contact_mean_per_complex_auprc"]),
            )
            > (
                float(right["selection_metric"]),
                float(right["contact_mean_per_complex_auprc"]),
            )
            or (
                (
                    float(left["selection_metric"]),
                    float(left["contact_mean_per_complex_auprc"]),
                )
                == (
                    float(right["selection_metric"]),
                    float(right["contact_mean_per_complex_auprc"]),
                )
                and architecture_index.get(left.get("experiment"), len(architecture_index))
                < architecture_index.get(
                    right.get("experiment"), len(architecture_index)
                )
            )
        )
        for left, right in zip(candidates, candidates[1:], strict=False)
        if isinstance(left, dict) and isinstance(right, dict)
    )
    if (
        receipt.get("format_version") != SELECTION_FORMAT
        or result.get("format_version") != SELECTION_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or receipt.get("selected_experiment") != result.get("selected_experiment")
        or receipt.get("selected_ensemble_result_sha256")
        != selected.get("result_sha256")
        or selection_rule.get("primary")
        != "maximize minimum_direction_cluster_weighted_margin"
        or selection_rule.get("first_tie_break")
        != "maximize mean_per_complex_contact_auprc"
        or selection_rule.get("test_role_forbidden") is not True
        or len(candidates) != 8
        or len(candidate_experiments) != 8
        or set(candidate_experiments) != set(ASEP_ARCHITECTURE_MATRIX)
        or not candidate_scores_valid
        or not candidate_order_valid
        or not isinstance(selected_metric, (int, float))
        or not math.isfinite(float(selected_metric))
        or not isinstance(selected_contact, (int, float))
        or not math.isfinite(float(selected_contact))
        or candidates[0].get("experiment") != result.get("selected_experiment")
        or candidates[0].get("root") != selected.get("root")
        or candidates[0].get("result_sha256") != selected.get("result_sha256")
        or candidates[0].get("receipt_sha256") != selected.get("receipt_sha256")
        or candidates[0].get("selection_metric") != selected_metric
        or candidates[0].get("contact_mean_per_complex_auprc") != selected_contact
    ):
        raise AsepExternalError("AsEP development selection is not pre-opening eligible")
    ensemble_root = Path(str(selected.get("root", "")))
    ensemble_result_path = (
        ensemble_root / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    )
    ensemble_receipt_path = ensemble_root / "ENSEMBLE_RECEIPT.json"
    authenticate_file(
        ensemble_result_path,
        str(selected.get("result_sha256", "")),
        "selected AsEP ensemble result",
    )
    authenticate_file(
        ensemble_receipt_path,
        str(selected.get("receipt_sha256", "")),
        "selected AsEP ensemble receipt",
    )
    ensemble_result = _json(ensemble_result_path, "selected AsEP ensemble result")
    ensemble_receipt = _json(ensemble_receipt_path, "selected AsEP ensemble receipt")
    if (
        ensemble_result.get("format_version") != ENSEMBLE_FORMAT
        or ensemble_receipt.get("format_version") != ENSEMBLE_FORMAT
        or ensemble_receipt.get("result_sha256") != selected.get("result_sha256")
        or ensemble_result.get("experiment") != result.get("selected_experiment")
        or ensemble_result.get("seeds") != [17, 42, 73, 101, 137]
        or ensemble_result.get("ensemble_rule", {}).get("member_weighting")
        != "uniform 1/5"
        or ensemble_result.get("authority", {}).get("dataset_backend") != "asep"
        or len(ensemble_result.get("authority", {}).get("members", [])) != 5
        or ensemble_result.get("authority", {}).get("test_loaded") is not False
        or ensemble_result.get("authority", {}).get("external_test_loaded") is not False
        or ensemble_receipt.get("test_loaded") is not False
        or ensemble_receipt.get("external_test_loaded") is not False
        or ensemble_result.get("validation", {}).get("selection_metric_rule")
        != "minimum_direction_cluster_weighted_margin"
        or ensemble_result.get("validation", {}).get("specificity_score_source")
        != "interaction_only"
        or ensemble_result.get("validation", {}).get("selection_metric")
        != selected_metric
        or ensemble_result.get("validation", {})
        .get("contact", {})
        .get("mean_per_complex_auprc")
        != selected_contact
    ):
        raise AsepExternalError("Selected AsEP ensemble is not pre-opening eligible")
    return {
        "selection_result": result,
        "selection_receipt_sha256": selection_receipt_sha256,
        "selection_result_sha256": receipt["result_sha256"],
        "selected_ensemble_root": str(ensemble_root),
        "selected_ensemble_result_sha256": selected["result_sha256"],
        "selected_ensemble_receipt_sha256": selected["receipt_sha256"],
    }


def materialize_asep_label_free_test_metadata(
    *,
    authority_root: Path,
    inner_audit_path: Path,
    inner_audit_sha256: str,
    split_root: Path,
    split_receipt_sha256: str,
    split_method: str,
    selection_root: Path,
    selection_receipt_sha256: str,
    output_root: Path,
    maximum_pdb_member_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Open test PDB inputs once, leaving graph tensors and task labels sealed."""
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite AsEP label-free test metadata: {output_root}"
        )
    selection = authenticate_asep_development_selection(
        selection_root, selection_receipt_sha256=selection_receipt_sha256
    )
    authenticate_file(inner_audit_path, inner_audit_sha256, "AsEP inner audit")
    inner = _json(inner_audit_path, "AsEP inner audit")
    if (
        inner.get("format_version") != INNER_AUDIT_FORMAT
        or inner.get("authority", {}).get("task_labels_read") is not False
        or inner.get("authority", {}).get("model_predictions_or_scores_read") is not False
        or inner.get("checks", {}).get("split_payload_opened") is not False
    ):
        raise AsepExternalError("AsEP inner audit is not label-free eligible")
    roles, split_receipt, split_manifest_sha256 = _load_split_roles(
        split_root, split_receipt_sha256, split_method
    )
    selected = {identifier for identifier, role in roles.items() if role == "test"}
    expected_count = int(
        split_receipt.get("manifests", {})
        .get(split_method, {})
        .get("role_counts", {})
        .get("test", -1)
    )
    if not selected or len(selected) != expected_count:
        raise AsepExternalError("AsEP official test selection is incomplete")
    index_by_id = {identifier: index for index, identifier in enumerate(roles)}
    structure_archive = authority_root / "payload/asepv1.2.0/structures.tar.gz"
    structure_archive_sha256 = authenticate_file(
        structure_archive,
        str(inner.get("structure_archive", {}).get("sha256", "")),
        "AsEP structure archive",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with tarfile.open(structure_archive, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or len(path.parts) != 2
                    or path.parts[0] != "structures"
                    or path.suffix != ".pdb"
                ):
                    raise AsepExternalError("AsEP structure tar member schema differs")
                identifier = path.stem
                if identifier not in selected:
                    continue
                if identifier in seen:
                    raise AsepExternalError("AsEP structure tar repeats a test member")
                if member.size <= 0 or member.size > maximum_pdb_member_bytes:
                    raise AsepExternalError("AsEP test PDB member size is outside limit")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read selected AsEP test PDB")
                raw = handle.read(maximum_pdb_member_bytes + 1)
                if len(raw) != member.size:
                    raise AsepExternalError("AsEP test PDB length differs from tar metadata")
                metadata = extract_asep_label_free_structure_metadata(
                    raw, expected_instance_id=identifier
                )
                antigen_pdb = extract_asep_antigen_pdb_bytes(
                    raw, antigen_chain=str(metadata["antigen_chain"])
                )
                foldseek_structure_id = hashlib.sha256(
                    f"asep:{identifier}".encode("ascii")
                ).hexdigest()
                relative = (
                    Path("antigen_structures") / f"{foldseek_structure_id}.pdb"
                )
                antigen_path = temporary / relative
                antigen_path.parent.mkdir(parents=True, exist_ok=True)
                antigen_path.write_bytes(antigen_pdb)
                records.append(
                    {
                        "dataset_index": index_by_id[identifier],
                        "instance_id": identifier,
                        "role": "test",
                        "foldseek_structure_id": foldseek_structure_id,
                        "pdb_id": metadata["pdb_id"],
                        "antigen_chain": metadata["antigen_chain"],
                        "antigen_pdb_seqres_sha256": metadata[
                            "antigen_pdb_seqres_sha256"
                        ],
                        "antibody_pdb_seqres_sha256": metadata[
                            "antibody_pdb_seqres_sha256"
                        ],
                        "antigen_pdb_seqres_length": metadata[
                            "antigen_pdb_seqres_length"
                        ],
                        "antibody_pdb_seqres_length": metadata[
                            "antibody_pdb_seqres_length"
                        ],
                        "antigen_observed_residue_count": metadata[
                            "antigen_observed_residue_count"
                        ],
                        "antibody_observed_residue_count": metadata[
                            "antibody_observed_residue_count"
                        ],
                        "source_pdb_member_sha256": hashlib.sha256(raw).hexdigest(),
                        "antigen_structure_path": relative.as_posix(),
                        "antigen_structure_sha256": sha256_file(antigen_path),
                    }
                )
                seen.add(identifier)
        if seen != selected:
            raise AsepExternalError(
                f"AsEP structure archive lacks test member {sorted(selected - seen)[:1]}"
            )
        records.sort(key=lambda record: int(record["dataset_index"]))
        manifest_path = temporary / "ASEP_LABEL_FREE_TEST_METADATA.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=tuple(records[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(records)
        receipt = {
            "format_version": LABEL_FREE_TEST_FORMAT,
            "authority": {
                "inner_audit_sha256": inner_audit_sha256,
                "split_receipt_sha256": split_receipt_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "split_method": split_method,
                "structure_archive_sha256": structure_archive_sha256,
                "selection_receipt_sha256": selection[
                    "selection_receipt_sha256"
                ],
                "selection_result_sha256": selection[
                    "selection_result_sha256"
                ],
                "selected_ensemble_result_sha256": selection[
                    "selected_ensemble_result_sha256"
                ],
            },
            "manifest_sha256": sha256_file(manifest_path),
            "test_instance_count": len(records),
            "label_free_test_feature_open_count": 1,
            "test_label_open_count": 0,
            "checks": {
                "development_selection_frozen_before_test_feature_access": True,
                "only_test_pdb_input_members_opened": True,
                "official_test_graph_tensors_opened": False,
                "official_test_contact_or_residue_labels_read": False,
                "model_predictions_or_scores_read": False,
                "antigen_structures_exported_for_decontamination_only": True,
            },
            "sequence_digest_boundary": (
                "Exact-sequence hashes use PDB SEQRES axes, which may be ordered "
                "deletion-only views of official graph sequences. PDB identity and "
                "structural similarity remain independent contamination controls."
            ),
            "claim_boundary": (
                "One-time label-free official-test input preparation after development "
                "selection. No graph tensor, task label, model prediction, score, or "
                "performance metric was opened."
            ),
        }
        (temporary / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _authenticate_official_test_opening(
    opening_receipt_path: Path,
    opening_receipt_sha256: str,
) -> dict[str, Any]:
    authenticate_file(
        opening_receipt_path,
        opening_receipt_sha256,
        "AsEP official-test opening receipt",
    )
    opening = _json(opening_receipt_path, "AsEP official-test opening receipt")
    observed_dependencies = asep_evaluation_source_hashes()
    state_root = Path(str(opening.get("test_state_root", "")))
    if (
        opening.get("format_version") != OPENING_FORMAT
        or opening.get("eligible_to_open") is not True
        or opening.get("test_feature_open_count_before") != 1
        or opening.get("test_graph_or_label_open_count_before") != 0
        or opening.get("test_labels_read_at_freeze") is not False
        or opening.get("model_predictions_or_scores_read_at_freeze") is not False
        or opening.get("test_adapter_source_sha256") != sha256_file(Path(__file__))
        or opening.get("test_evaluator_source_sha256")
        != observed_dependencies["scripts/evaluate_asep_frozen_test.py"]
        or opening.get("evaluation_source_dependencies") != observed_dependencies
        or not state_root.is_absolute()
        or state_root.is_symlink()
        or not state_root.is_dir()
        or opening.get("graph_open_consumption_token_name")
        != "ASEP_TEST_GRAPH_OPEN_CONSUMED.json"
        or opening.get("score_consumption_token_name")
        != "ASEP_TEST_SCORE_CONSUMED.json"
    ):
        raise AsepExternalError("AsEP official-test opening receipt is ineligible")
    return opening


def _consume_graph_open_once(
    opening: dict[str, Any],
    *,
    opening_receipt_sha256: str,
    output_root: Path,
) -> Path:
    """Atomically consume the only authorized official graph-label opening."""
    state_root = Path(str(opening["test_state_root"]))
    token_path = state_root / str(opening["graph_open_consumption_token_name"])
    payload = canonical_json_bytes(
        {
            "format_version": GRAPH_OPEN_CONSUMPTION_FORMAT,
            "opening_receipt_sha256": opening_receipt_sha256,
            "selected_ensemble_result_sha256": opening[
                "selected_ensemble_result_sha256"
            ],
            "requested_output_root": str(output_root.resolve()),
            "state": "official_graph_label_open_started_fail_closed",
            "recovery_boundary": (
                "If materialization crashes, preserve this token and freeze a "
                "recovery receipt; never delete it and reopen silently."
            ),
        }
    )
    descriptor = os.open(
        token_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return token_path


def materialize_asep_frozen_test_cache(
    *,
    authority_root: Path,
    inner_audit_path: Path,
    inner_audit_sha256: str,
    split_root: Path,
    split_receipt_sha256: str,
    split_method: str,
    opening_receipt_path: Path,
    opening_receipt_sha256: str,
    output_root: Path,
    maximum_member_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Open official test graph labels once under the frozen opening receipt."""
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP frozen test cache: {output_root}")
    opening = _authenticate_official_test_opening(
        opening_receipt_path, opening_receipt_sha256
    )
    if (
        opening.get("inner_audit_sha256") != inner_audit_sha256
        or opening.get("split_receipt_sha256") != split_receipt_sha256
        or opening.get("split_method") != split_method
    ):
        raise AsepExternalError("Opening receipt differs from requested AsEP authorities")
    authenticate_file(inner_audit_path, inner_audit_sha256, "AsEP inner audit")
    inner = _json(inner_audit_path, "AsEP inner audit")
    if (
        inner.get("format_version") != INNER_AUDIT_FORMAT
        or inner.get("authority", {}).get("task_labels_read") is not False
        or inner.get("authority", {}).get("model_predictions_or_scores_read") is not False
    ):
        raise AsepExternalError("AsEP inner audit is not opening eligible")
    roles, split_receipt, split_manifest_sha256 = _load_split_roles(
        split_root, split_receipt_sha256, split_method
    )
    selected = {identifier for identifier, role in roles.items() if role == "test"}
    expected_count = int(
        split_receipt.get("manifests", {})
        .get(split_method, {})
        .get("role_counts", {})
        .get("test", -1)
    )
    if not selected or len(selected) != expected_count:
        raise AsepExternalError("AsEP official test selection is incomplete")
    if opening.get("official_test_instance_count") != len(selected):
        raise AsepExternalError("Opening receipt official-test count differs")
    payload_root = authority_root / "payload/asepv1.2.0"
    graph_archive = payload_root / "asepv1_interim_graphs.tar.gz"
    structure_archive = payload_root / "structures.tar.gz"
    graph_sha = authenticate_file(
        graph_archive,
        str(inner.get("graph_archive", {}).get("sha256", "")),
        "AsEP graph archive",
    )
    structure_sha = authenticate_file(
        structure_archive,
        str(inner.get("structure_archive", {}).get("sha256", "")),
        "AsEP structure archive",
    )
    if (
        opening.get("graph_archive_sha256") != graph_sha
        or opening.get("structure_archive_sha256") != structure_sha
    ):
        raise AsepExternalError("Opening receipt archive authority differs")
    graph_open_token = _consume_graph_open_once(
        opening,
        opening_receipt_sha256=opening_receipt_sha256,
        output_root=output_root,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    stage_root = temporary / ".graph-stage"
    stage_root.mkdir()
    records: list[dict[str, Any]] = []
    try:
        staged: set[str] = set()
        with tarfile.open(graph_archive, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or len(path.parts) != 2
                    or path.parts[0] != "asepv1_interim_graphs"
                    or path.suffix != ".pt"
                ):
                    raise AsepExternalError("AsEP graph tar member schema differs")
                identifier = path.stem
                if identifier not in selected:
                    continue
                if identifier in staged or member.size <= 0 or member.size > maximum_member_bytes:
                    raise AsepExternalError("AsEP test graph member is invalid")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read selected AsEP test graph")
                raw = handle.read(maximum_member_bytes + 1)
                if len(raw) != member.size:
                    raise AsepExternalError("AsEP test graph member length differs")
                graph = load_asep_graph_bytes(
                    raw,
                    expected_instance_id=identifier,
                    maximum_bytes=maximum_member_bytes,
                )
                stage = {
                    key: graph[key]
                    for key in (
                        "instance_id",
                        "heavy_sequence",
                        "light_sequence",
                        "antigen_sequence_full",
                        "antigen_chain",
                        "antibody_sequence",
                        "antigen_sequence",
                        "antibody_full_selection",
                        "antigen_full_selection",
                        "antibody_chain_ids",
                        "antigen_chain_ids",
                        "antibody_esm2_35m",
                        "antibody_igfold",
                        "antigen_esm2_35m",
                        "pair_positive_indices",
                        "antibody_labels",
                        "antigen_labels",
                    )
                }
                torch.save(stage, stage_root / f"{identifier}.pt")
                staged.add(identifier)
        if staged != selected:
            raise AsepExternalError("AsEP official test graph stage is incomplete")

        materialized: set[str] = set()
        with tarfile.open(structure_archive, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or len(path.parts) != 2
                    or path.parts[0] != "structures"
                    or path.suffix != ".pdb"
                ):
                    raise AsepExternalError("AsEP structure tar member schema differs")
                identifier = path.stem
                if identifier not in selected:
                    continue
                if (
                    identifier in materialized
                    or member.size <= 0
                    or member.size > maximum_member_bytes
                ):
                    raise AsepExternalError("AsEP test structure member is invalid")
                handle = archive.extractfile(member)
                if handle is None:
                    raise AsepExternalError("Cannot read selected AsEP test structure")
                raw = handle.read(maximum_member_bytes + 1)
                if len(raw) != member.size:
                    raise AsepExternalError("AsEP test structure member length differs")
                graph = torch.load(
                    stage_root / f"{identifier}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                structure = adapt_asep_structure_bytes(raw, graph)
                artifact = {
                    "format_version": FROZEN_TEST_ARTIFACT_FORMAT,
                    "instance_id": identifier,
                    "role": "test",
                    **graph,
                    "antibody_backbone": structure["antibody_backbone"],
                    "antigen_backbone": structure["antigen_backbone"],
                    "antibody_frame_valid_mask": structure[
                        "antibody_frame_valid_mask"
                    ],
                    "antigen_frame_valid_mask": structure[
                        "antigen_frame_valid_mask"
                    ],
                    "antibody_missing_frame_mask": structure[
                        "antibody_missing_frame_mask"
                    ],
                    "antigen_missing_frame_mask": structure[
                        "antigen_missing_frame_mask"
                    ],
                }
                relative = Path("artifacts") / identifier[:2] / f"{identifier}.pt"
                artifact_path = temporary / relative
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(artifact, artifact_path)
                records.append(
                    {
                        "instance_id": identifier,
                        "role": "test",
                        "artifact_path": relative.as_posix(),
                        "artifact_sha256": sha256_file(artifact_path),
                        "antibody_length": len(graph["antibody_sequence"]),
                        "antigen_length": len(graph["antigen_sequence"]),
                        "positive_pair_count": int(
                            graph["pair_positive_indices"].shape[1]
                        ),
                    }
                )
                materialized.add(identifier)
        if materialized != selected:
            raise AsepExternalError("AsEP official test cache is incomplete")
        shutil.rmtree(stage_root)
        order = {identifier: index for index, identifier in enumerate(roles)}
        records.sort(key=lambda record: order[str(record["instance_id"])])
        manifest = {
            "format_version": FROZEN_TEST_CACHE_FORMAT,
            "authority": {
                "opening_receipt_sha256": opening_receipt_sha256,
                "inner_audit_sha256": inner_audit_sha256,
                "split_receipt_sha256": split_receipt_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "split_method": split_method,
                "graph_archive_sha256": graph_sha,
                "structure_archive_sha256": structure_sha,
                "selected_ensemble_result_sha256": opening[
                    "selected_ensemble_result_sha256"
                ],
                "graph_open_consumption_token_sha256": sha256_file(
                    graph_open_token
                ),
            },
            "feature_modes": {
                "official_esm2_35m": {"antibody_dim": 480, "antigen_dim": 480},
                "official_igfold_esm2": {"antibody_dim": 512, "antigen_dim": 480},
            },
            "record_count": len(records),
            "role_counts": {"test": len(records)},
            "records": records,
            "checks": {
                "official_test_graph_and_labels_opened_once": True,
                "development_selection_and_opening_frozen_first": True,
                "contact_marginals_validated": True,
                "missing_frames_explicit": True,
                "artifacts_weights_only_loadable": True,
                "model_predictions_or_scores_read": False,
            },
            "claim_boundary": (
                "Authenticated one-time official-test cache after frozen opening. "
                "No model prediction or performance score has yet been computed."
            ),
        }
        manifest_path = temporary / "ASEP_FROZEN_TEST_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        receipt = {
            "format_version": FROZEN_TEST_CACHE_FORMAT,
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": len(records),
            "test_graph_or_label_open_count": 1,
            "model_predictions_or_scores_read": False,
            "graph_open_consumption_token_sha256": sha256_file(
                graph_open_token
            ),
        }
        (temporary / "ASEP_FROZEN_TEST_CACHE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class AsepFrozenTestDataset(torch.utils.data.Dataset):
    """Load the once-opened official test cache on complete pair grids."""

    def __init__(
        self,
        cache_root: Path,
        *,
        manifest_sha256: str,
        feature_mode: str = "official_esm2_35m",
        instance_ids: set[str] | None = None,
    ) -> None:
        manifest_path = cache_root / "ASEP_FROZEN_TEST_CACHE_MANIFEST.json"
        authenticate_file(manifest_path, manifest_sha256, "AsEP frozen test manifest")
        manifest = _json(manifest_path, "AsEP frozen test manifest")
        if (
            manifest.get("format_version") != FROZEN_TEST_CACHE_FORMAT
            or manifest.get("checks", {}).get(
                "official_test_graph_and_labels_opened_once"
            )
            is not True
            or manifest.get("checks", {}).get("model_predictions_or_scores_read")
            is not False
            or feature_mode not in manifest.get("feature_modes", {})
        ):
            raise AsepExternalError("AsEP frozen test cache is ineligible")
        records = list(manifest.get("records", []))
        if instance_ids is not None:
            records = [record for record in records if record["instance_id"] in instance_ids]
            if {record["instance_id"] for record in records} != instance_ids:
                raise AsepExternalError("Requested AsEP frozen-test subset is incomplete")
        if not records:
            raise AsepExternalError("AsEP frozen test dataset is empty")
        self.cache_root = cache_root
        self.records = tuple(records)
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.cache_root / record["artifact_path"]
        authenticate_file(path, record["artifact_sha256"], "AsEP frozen test artifact")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (
            artifact.get("format_version") != FROZEN_TEST_ARTIFACT_FORMAT
            or artifact.get("instance_id") != record["instance_id"]
            or artifact.get("role") != "test"
        ):
            raise AsepExternalError("AsEP frozen test artifact metadata differs")
        antibody_features = (
            artifact["antibody_esm2_35m"]
            if self.feature_mode == "official_esm2_35m"
            else artifact["antibody_igfold"]
        )
        antigen_features = artifact["antigen_esm2_35m"]
        pair_labels = torch.zeros(
            int(antibody_features.shape[0]),
            int(antigen_features.shape[0]),
            dtype=torch.float32,
        )
        positive = artifact["pair_positive_indices"]
        if (
            not torch.is_tensor(positive)
            or positive.dtype != torch.long
            or positive.ndim != 2
            or positive.shape[0] != 2
        ):
            raise AsepExternalError("AsEP frozen test contacts are malformed")
        pair_labels[positive[0], positive[1]] = 1.0
        if not torch.equal(
            pair_labels.bool().any(dim=1).to(torch.float32),
            artifact["antibody_labels"],
        ) or not torch.equal(
            pair_labels.bool().any(dim=0).to(torch.float32),
            artifact["antigen_labels"],
        ):
            raise AsepExternalError("AsEP frozen test contacts do not reproduce marginals")
        return {
            "instance_id": artifact["instance_id"],
            "antibody_features": antibody_features,
            "antigen_features": antigen_features,
            "antibody_backbone": artifact["antibody_backbone"],
            "antigen_backbone": artifact["antigen_backbone"],
            "antibody_chain_ids": artifact["antibody_chain_ids"],
            "antigen_chain_ids": artifact["antigen_chain_ids"],
            "antibody_frame_valid_mask": artifact["antibody_frame_valid_mask"],
            "antigen_frame_valid_mask": artifact["antigen_frame_valid_mask"],
            "antibody_missing_frame_mask": artifact["antibody_missing_frame_mask"],
            "antigen_missing_frame_mask": artifact["antigen_missing_frame_mask"],
            "antibody_labels": artifact["antibody_labels"],
            "antigen_labels": artifact["antigen_labels"],
            "pair_labels": pair_labels,
            "pair_candidate_mask": torch.ones_like(pair_labels, dtype=torch.bool),
        }


__all__ = [
    "AsepFrozenTestDataset",
    "ASEP_ARCHITECTURE_MATRIX",
    "FROZEN_TEST_CACHE_FORMAT",
    "LABEL_FREE_TEST_FORMAT",
    "OPENING_FORMAT",
    "EVALUATION_DEPENDENCY_PATHS",
    "GRAPH_OPEN_CONSUMPTION_FORMAT",
    "asep_evaluation_source_hashes",
    "authenticate_asep_development_selection",
    "materialize_asep_label_free_test_metadata",
    "materialize_asep_frozen_test_cache",
]
