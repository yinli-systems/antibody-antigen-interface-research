"""Authenticated US-align spot checks for Foldseek split-boundary pairs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file
from .structural_novelty import STRUCTURAL_AUDIT_FORMAT, symmetric_tm_score


FORMAT = "framecontact_usalign_boundary_spot_check_v1"
_TM_SCORE = re.compile(
    r"TM-score=\s*([0-9]+(?:\.[0-9]+)?)\s+\(normalized by length of Structure_([12])"
)


def parse_usalign_tm_scores(stdout: str) -> tuple[float, float]:
    """Parse the two length-normalized TM-scores from US-align stdout."""
    values: dict[int, float] = {}
    for score_text, structure_text in _TM_SCORE.findall(stdout):
        score = float(score_text)
        structure = int(structure_text)
        if structure in values:
            raise ValueError("US-align output repeats a normalized TM-score")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("US-align TM-score lies outside [0, 1]")
        values[structure] = score
    if set(values) != {1, 2}:
        raise ValueError("US-align output lacks both length-normalized TM-scores")
    return values[1], values[2]


def _csv_index(path: Path, key: str, required: set[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns {sorted(required)}")
        result = {}
        for row in reader:
            identifier = str(row[key]).strip()
            if not identifier or identifier in result:
                raise ValueError(f"{path} has an invalid or duplicate {key}")
            result[identifier] = {
                str(name): "" if value is None else str(value).strip()
                for name, value in row.items()
            }
        return result


def _align_one(
    task: tuple[str, str, str, str, str, float, float, str]
) -> dict[str, Any]:
    (
        binary,
        query,
        target,
        query_path,
        target_path,
        foldseek_score,
        threshold,
        score_rule,
    ) = task
    completed = subprocess.run(
        [binary, query_path, target_path],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    query_tm, target_tm = parse_usalign_tm_scores(completed.stdout)
    usalign_score = symmetric_tm_score(query_tm, target_tm, score_rule)
    return {
        "query": query,
        "target": target,
        "score_rule": score_rule,
        "foldseek_symmetric_tm_score": foldseek_score,
        "usalign_query_normalized_tm_score": query_tm,
        "usalign_target_normalized_tm_score": target_tm,
        "usalign_symmetric_tm_score": usalign_score,
        "foldseek_above_threshold": foldseek_score >= threshold,
        "usalign_above_threshold": usalign_score >= threshold,
        "classification_agreement": (foldseek_score >= threshold)
        == (usalign_score >= threshold),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
    }


def run_usalign_boundary_spot_checks(
    *,
    structural_audit_path: Path,
    structural_audit_sha256: str,
    split_manifest_path: Path,
    split_manifest_sha256: str,
    export_manifest_path: Path,
    export_manifest_sha256: str,
    structure_root: Path,
    usalign_binary_path: Path,
    usalign_binary_sha256: str,
    usalign_lock_path: Path,
    usalign_lock_sha256: str,
    output_root: Path,
    workers: int = 8,
) -> dict[str, Any]:
    """Run a deterministic, atomic spot-check set and expose any split violation."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    authority = {
        "structural_audit_sha256": authenticate_file(
            structural_audit_path, structural_audit_sha256, "structural audit"
        ),
        "split_manifest_sha256": authenticate_file(
            split_manifest_path, split_manifest_sha256, "split manifest"
        ),
        "export_manifest_sha256": authenticate_file(
            export_manifest_path, export_manifest_sha256, "antigen export manifest"
        ),
        "usalign_binary_sha256": authenticate_file(
            usalign_binary_path, usalign_binary_sha256, "US-align binary"
        ),
        "usalign_lock_sha256": authenticate_file(
            usalign_lock_path, usalign_lock_sha256, "US-align tool lock"
        ),
        "task_labels_read": False,
    }
    audit = json.loads(structural_audit_path.read_text(encoding="utf-8"))
    lock = json.loads(usalign_lock_path.read_text(encoding="utf-8"))
    if audit.get("format_version") != STRUCTURAL_AUDIT_FORMAT or audit.get("valid") is not True:
        raise ValueError("Only a passing structural audit may be spot-checked")
    if (
        lock.get("format_version") != "framecontact_usalign_tool_lock_v1"
        or lock.get("remote_binary_sha256") != usalign_binary_sha256
    ):
        raise ValueError("US-align tool lock does not bind the requested binary")
    threshold = float(audit["thresholds"]["hard_edge_symmetric_tm_score"])
    score_rule = str(audit["thresholds"]["foldseek_score_rule"])
    symmetric_tm_score(0.0, 0.0, score_rule)
    split = _csv_index(
        split_manifest_path,
        "antigen_structure_id",
        {"antigen_structure_id", "role"},
    )
    exported = _csv_index(
        export_manifest_path,
        "antigen_structure_id",
        {"antigen_structure_id", "foldseek_structure_id"},
    )
    if set(split) != set(exported):
        raise ValueError("Split and antigen export structure membership differ")
    pairs = audit.get("boundary_spot_check_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Structural audit contains no boundary pairs to spot-check")
    tasks = []
    requested_strata: dict[str, int] = {}
    for pair in pairs:
        query, target = str(pair["query"]), str(pair["target"])
        if query not in split or target not in split or query >= target:
            raise ValueError("Structural audit contains an invalid canonical boundary pair")
        query_path = structure_root / f"{exported[query]['foldseek_structure_id']}.pdb"
        target_path = structure_root / f"{exported[target]['foldseek_structure_id']}.pdb"
        if not query_path.is_file() or not target_path.is_file():
            raise FileNotFoundError("A boundary-pair antigen structure is missing")
        stratum = str(pair.get("pair_stratum", "legacy_unstratified"))
        if stratum not in {"cross_role", "within_role", "legacy_unstratified"}:
            raise ValueError("Structural audit contains an invalid pair stratum")
        requested_strata[stratum] = requested_strata.get(stratum, 0) + 1
        tasks.append(
            (
                str(usalign_binary_path),
                query,
                target,
                str(query_path),
                str(target_path),
                float(pair["symmetric_tm_score"]),
                threshold,
                score_rule,
            )
        )
    if workers == 1:
        results = [_align_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_align_one, tasks))
    results.sort(key=lambda value: (value["query"], value["target"]))
    for result in results:
        result["query_role"] = split[result["query"]]["role"]
        result["target_role"] = split[result["target"]]["role"]
        result["cross_role_usalign_hard_edge"] = (
            result["query_role"] != result["target_role"]
            and result["usalign_above_threshold"]
        )
    violation_count = sum(value["cross_role_usalign_hard_edge"] for value in results)
    disagreement_count = sum(not value["classification_agreement"] for value in results)
    sampled_cross_role = sum(
        value["query_role"] != value["target_role"] for value in results
    )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite spot-check output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        table_path = temporary / "USALIGN_BOUNDARY_SPOT_CHECKS.tsv"
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(results[0]),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(results)
        receipt = {
            "format_version": FORMAT,
            "valid": violation_count == 0,
            "authority": authority,
            "tm_threshold": threshold,
            "tm_score_rule": score_rule,
            "spot_check_count": len(results),
            "requested_pair_strata": requested_strata,
            "sampled_cross_role_pair_count": sampled_cross_role,
            "sampled_within_role_pair_count": len(results) - sampled_cross_role,
            "available_boundary_pair_count": int(
                audit["counts"]["boundary_pair_count"]
            ),
            "audit_boundary_set_truncated": bool(
                audit.get("boundary_spot_check_pairs_truncated")
            ),
            "foldseek_usalign_classification_disagreement_count": disagreement_count,
            "cross_role_usalign_hard_edge_count": violation_count,
            "table_sha256": sha256_file(table_path),
            "claim_boundary": (
                "Independent structure-alignment spot checks only. A passing sample does "
                "not prove model generalization or benchmark performance."
            ),
        }
        (temporary / "USALIGN_BOUNDARY_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = ["FORMAT", "parse_usalign_tm_scores", "run_usalign_boundary_spot_checks"]
