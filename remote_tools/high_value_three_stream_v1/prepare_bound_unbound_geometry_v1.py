#!/usr/bin/env python3
"""Acquire and freeze a label-free bound/unbound antibody benchmark inventory."""

from __future__ import annotations

import argparse
from itertools import permutations
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile


REPOSITORY = "Zhaonan99/Antibody-antigen-complex-structure-benchmark-dataset"
SOURCE_COMMIT = "6891e6e6790c7a898a08471193f494adf6f16b48"
ARCHIVE_URL = f"https://codeload.github.com/{REPOSITORY}/zip/{SOURCE_COMMIT}"
DATA_ROOT = "ABAG-Docking_benchmark_dataset"
SELECTED_SUBSETS = ("87cases", "25cases_Truncated_file")


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def acquire_archive(path: Path) -> str:
    if path.is_file() and not path.is_symlink():
        return sha256_file(path)
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "FrameContact-bound-unbound-audit-v1"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        raw = response.read()
    if not raw.startswith(b"PK\x03\x04"):
        raise ValueError("GitHub codeload response is not a ZIP archive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(raw)
    os.replace(temporary, path)
    return sha256_bytes(raw)


def safe_extract(archive: Path, destination: Path) -> Path:
    marker = destination / "EXTRACTED_COMMIT"
    if marker.is_file() and marker.read_text().strip() == SOURCE_COMMIT:
        roots = [path for path in destination.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ValueError("resumed archive extraction has ambiguous root")
        return roots[0]
    if destination.exists():
        raise FileExistsError(f"partial extraction is not reusable: {destination}")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
        handle.extractall(temporary)
    (temporary / "EXTRACTED_COMMIT").write_text(SOURCE_COMMIT + "\n")
    os.replace(temporary, destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("archive has ambiguous repository root")
    return roots[0]


def protein_chains(path: Path) -> list[dict[str, object]]:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    structure = PDBParser(QUIET=True).get_structure(path.stem, path)
    model = next(structure.get_models())
    chains: list[dict[str, object]] = []
    for chain in model:
        sequence: list[str] = []
        complete = 0
        residue_ids: list[str] = []
        for residue in chain:
            if not is_aa(residue, standard=False):
                continue
            amino_acid = seq1(
                residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"
            ).upper()
            sequence.append(amino_acid if amino_acid in "ACDEFGHIKLMNPQRSTVWY" else "X")
            residue_ids.append(f"{residue.id[1]}{str(residue.id[2]).strip()}")
            if all(name in residue for name in ("N", "CA", "C")):
                coords = [residue[name].coord for name in ("N", "CA", "C")]
                if all(math.isfinite(float(value)) for xyz in coords for value in xyz):
                    complete += 1
        if sequence:
            text = "".join(sequence)
            chains.append(
                {
                    "chain_id": str(chain.id),
                    "sequence": text,
                    "sequence_sha256": sha256_bytes(text.encode("ascii")),
                    "length": len(text),
                    "backbone_complete_count": complete,
                    "residue_ids": residue_ids,
                }
            )
    if not chains:
        raise ValueError(f"{path} contains no protein chains")
    return chains


def alignment_metrics(query: str, target: str) -> dict[str, object]:
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query, target)[0]
    qblocks, tblocks = alignment.aligned
    aligned = 0
    matches = 0
    mapping: list[list[int]] = []
    for (q0, q1), (t0, t1) in zip(qblocks, tblocks, strict=True):
        length = int(q1 - q0)
        aligned += length
        matches += sum(a == b for a, b in zip(query[q0:q1], target[t0:t1]))
        mapping.extend([[int(q0 + offset), int(t0 + offset)] for offset in range(length)])
    return {
        "identity": matches / aligned if aligned else 0.0,
        "bound_coverage": aligned / len(query),
        "unbound_coverage": aligned / len(target),
        "aligned_residue_count": aligned,
        "bound_to_unbound_index_pairs": mapping,
    }


def match_chains(bound: list[dict[str, object]], unbound: list[dict[str, object]]) -> dict[str, object]:
    if max(len(bound), len(unbound)) > 8:
        raise ValueError("unexpectedly many chains for exact assignment")
    metrics = {
        (i, j): alignment_metrics(str(left["sequence"]), str(right["sequence"]))
        for i, left in enumerate(bound)
        for j, right in enumerate(unbound)
    }
    best: tuple[float, tuple[tuple[int, int], ...]] | None = None
    if len(bound) <= len(unbound):
        for targets in permutations(range(len(unbound)), len(bound)):
            pairs = tuple((index, targets[index]) for index in range(len(bound)))
            score = sum(float(metrics[pair]["identity"]) for pair in pairs)
            candidate = (score, pairs)
            if best is None or candidate[0] > best[0]:
                best = candidate
    else:
        for sources in permutations(range(len(bound)), len(unbound)):
            pairs = tuple((sources[index], index) for index in range(len(unbound)))
            score = sum(float(metrics[pair]["identity"]) for pair in pairs)
            candidate = (score, pairs)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    assignments = [
        {
            "bound_chain_id": bound[i]["chain_id"],
            "unbound_chain_id": unbound[j]["chain_id"],
            **metrics[(i, j)],
        }
        for i, j in best[1]
    ]
    eligible = (
        len(best[1]) == len(bound) == len(unbound)
        and all(
            float(row["identity"]) >= 0.90
            and float(row["bound_coverage"]) >= 0.90
            and float(row["unbound_coverage"]) >= 0.90
            for row in assignments
        )
    )
    return {
        "assignments": assignments,
        "bound_chain_count": len(bound),
        "unbound_chain_count": len(unbound),
        "eligible_for_residue_paired_robustness": eligible,
    }


def locate_cases(repository_root: Path) -> list[tuple[str, Path]]:
    data_root = repository_root / DATA_ROOT
    cases: list[tuple[str, Path]] = []
    for subset in SELECTED_SUBSETS:
        subset_root = data_root / subset
        if not subset_root.is_dir():
            raise FileNotFoundError(subset_root)
        for case_root in sorted(path for path in subset_root.iterdir() if path.is_dir()):
            cases.append((subset, case_root))
    identifiers = [root.name for _, root in cases]
    if len(cases) != 112 or len(set(identifiers)) != 112:
        raise ValueError(f"expected 112 unique cases, observed {len(cases)}")
    return cases


def case_files(case_root: Path) -> dict[str, Path]:
    stems = {
        "antibody_bound": "_r_b.pdb",
        "antibody_unbound": "_r_u.pdb",
        "antigen_bound": "_l_b.pdb",
        "antigen_unbound": "_l_u.pdb",
    }
    result: dict[str, Path] = {}
    files = list(case_root.glob("*.pdb"))
    for role, suffix in stems.items():
        matches = [path for path in files if path.name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"{case_root.name} has {len(matches)} {role} files")
        result[role] = matches[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    require_sha(args.protocol, args.protocol_sha256, "protocol")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("format_version") != "framecontact_bound_unbound_geometry_protocol_v1"
        or protocol.get("status") != "FROZEN_BEFORE_DATA_ACQUISITION"
        or protocol.get("source", {}).get("commit") != SOURCE_COMMIT
        or protocol.get("label_free_gate", {}).get("script_sha256")
        != sha256_file(Path(__file__))
    ):
        raise ValueError("bound/unbound protocol differs")
    result_path = args.output_root / "BOUND_UNBOUND_LABEL_FREE_MANIFEST_V1.json"
    receipt_path = args.output_root / "BOUND_UNBOUND_LABEL_FREE_RECEIPT_V1.json"
    if result_path.exists() or receipt_path.exists():
        raise FileExistsError("bound/unbound label-free result already exists")

    work = args.output_root / "label-free-work-v1"
    archive = work / f"source-{SOURCE_COMMIT}.zip"
    archive_sha = acquire_archive(archive)
    repository_root = safe_extract(archive, work / "extracted")
    cases: list[dict[str, object]] = []
    for index, (subset, root) in enumerate(locate_cases(repository_root), start=1):
        files = case_files(root)
        structures = {role: protein_chains(path) for role, path in files.items()}
        antibody_mapping = match_chains(
            structures["antibody_bound"], structures["antibody_unbound"]
        )
        antigen_mapping = match_chains(
            structures["antigen_bound"], structures["antigen_unbound"]
        )
        cases.append(
            {
                "case_id": root.name,
                "subset": subset,
                "files": {
                    role: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "chains": structures[role],
                    }
                    for role, path in files.items()
                },
                "antibody_mapping": antibody_mapping,
                "antigen_mapping": antigen_mapping,
                "eligible_for_residue_paired_robustness": bool(
                    antibody_mapping["eligible_for_residue_paired_robustness"]
                    and antigen_mapping["eligible_for_residue_paired_robustness"]
                ),
            }
        )
        print(json.dumps({"phase": "bound_unbound_mapping", "done": index, "total": 112, "case": root.name}), flush=True)
    eligible = [row["case_id"] for row in cases if row["eligible_for_residue_paired_robustness"]]
    payload = {
        "format_version": "abag_bound_unbound_label_free_manifest_v1",
        "status": "PASS_BOUND_UNBOUND_LABEL_FREE_FREEZE",
        "protocol_sha256": args.protocol_sha256,
        "source": {
            "repository": REPOSITORY,
            "commit": SOURCE_COMMIT,
            "archive_url": ARCHIVE_URL,
            "archive_sha256": archive_sha,
            "license_file_present_at_commit": False,
            "redistribution_authorized_by_this_receipt": False,
        },
        "selection": {
            "subsets": list(SELECTED_SUBSETS),
            "uses_truncated_variant_for_25_declared_cases": True,
            "sequence_mapping_identity_minimum": 0.90,
            "sequence_mapping_reciprocal_coverage_minimum": 0.90,
        },
        "counts": {
            "cases": len(cases),
            "eligible_cases": len(eligible),
        },
        "eligible_case_ids": eligible,
        "cases": sorted(cases, key=lambda value: str(value["case_id"])),
        "audit": {
            "contact_labels_computed": False,
            "cross_chain_distances_computed": False,
            "task_labels_read": False,
            "training_or_development_overlap_used_for_selection": False,
        },
    }
    atomic_write(result_path, canonical_bytes(payload))
    receipt = {
        "format_version": "abag_bound_unbound_label_free_receipt_v1",
        "status": payload["status"],
        "manifest_sha256": sha256_file(result_path),
        "protocol_sha256": args.protocol_sha256,
        "archive_sha256": archive_sha,
        "eligible_cases": len(eligible),
        "contact_labels_computed": False,
        "selection_performed": False,
    }
    atomic_write(receipt_path, canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
