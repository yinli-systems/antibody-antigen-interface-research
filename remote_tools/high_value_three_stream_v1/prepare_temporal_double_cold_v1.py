#!/usr/bin/env python3
"""Freeze a label-free, strict-temporal, sequence-double-cold SAbDab2 cohort.

This phase downloads only cropped mmCIF structures, parses intra-chain sequences
and backbone completeness, and compares antibody/antigen sequences with the
frozen development corpus.  It never computes cross-chain distances or labels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile


API_URL = "https://sabdab.opig.stats.ox.ac.uk/api/download/structures"
IDENTITY_THRESHOLD = 0.80
RECIPROCAL_COVERAGE_THRESHOLD = 0.80
_REFERENCES: dict[str, tuple[tuple[str, str], ...]] = {}


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


def split_field(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split("|") if part.strip())


def protein_antigen_chains(record: dict[str, object]) -> tuple[str, ...]:
    chains = split_field(str(record["antigen_chain"]))
    types = split_field(str(record["antigen_type"]))
    if not chains:
        raise ValueError(f"{record['instance']} has no antigen chain")
    if len(types) == 1 and types[0].upper() == "PROTEIN":
        return chains
    if len(types) != len(chains):
        raise ValueError(f"{record['instance']} antigen chain/type arity differs")
    selected = tuple(
        chain for chain, kind in zip(chains, types, strict=True)
        if kind.upper() == "PROTEIN"
    )
    if not selected:
        raise ValueError(f"{record['instance']} has no protein antigen chain")
    return selected


def _request_payload(record: dict[str, object], *, include_empty_light: bool) -> bytes:
    antibody: dict[str, object] = {
        "heavy_chain": str(record["hchain"]),
        "include_antigen": True,
    }
    light = str(record["lchain"])
    if light not in {"", "NA", "None", "nan"}:
        antibody["light_chain"] = light
    elif include_empty_light:
        antibody["light_chain"] = ""
    return json.dumps(
        {
            "structures": [
                {
                    "pdb_id": str(record["pdb"]),
                    "full_structure": False,
                    "antibody_instances": [antibody],
                }
            ]
        }
    ).encode("utf-8")


def download_cropped_cif(record: dict[str, object], output: Path) -> str:
    if output.is_file() and not output.is_symlink():
        return sha256_file(output)
    errors: list[str] = []
    for include_empty_light in (False, True):
        for attempt in range(4):
            request = urllib.request.Request(
                API_URL,
                data=_request_payload(record, include_empty_light=include_empty_light),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = response.read()
                if raw.startswith(b"PK\x03\x04"):
                    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                        members = sorted(
                            name for name in archive.namelist()
                            if name.lower().endswith((".cif", ".mmcif"))
                        )
                        if len(members) != 1:
                            raise ValueError(f"API archive has {len(members)} mmCIF files")
                        raw = archive.read(members[0])
                if not raw.startswith((b"data_", b"#")) or b"_atom_site." not in raw:
                    raise ValueError("SAbDab2 response is not an mmCIF payload")
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
                with temporary.open("xb") as handle:
                    handle.write(raw)
                os.replace(temporary, output)
                return sha256_bytes(raw)
            except (urllib.error.URLError, TimeoutError, ValueError) as error:
                errors.append(f"empty_light={include_empty_light} attempt={attempt}: {error}")
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Cannot download {record['instance']}: {'; '.join(errors)}")


def observed_chain(model: object, chain_id: str) -> tuple[str, int, int]:
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    try:
        chain = model[chain_id]
    except KeyError as error:
        raise ValueError(f"mmCIF lacks declared chain {chain_id!r}") from error
    sequence: list[str] = []
    complete = 0
    residue_count = 0
    for residue in chain:
        if not is_aa(residue, standard=False):
            continue
        residue_count += 1
        amino_acid = seq1(
            residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X"
        ).upper()
        sequence.append(amino_acid if amino_acid in "ACDEFGHIKLMNPQRSTVWY" else "X")
        if all(name in residue for name in ("N", "CA", "C")):
            coordinates = [residue[name].coord for name in ("N", "CA", "C")]
            if all(math.isfinite(float(value)) for xyz in coordinates for value in xyz):
                complete += 1
    if not sequence:
        raise ValueError(f"chain {chain_id!r} has no protein residues")
    return "".join(sequence), complete, residue_count


def parse_instance(record: dict[str, object], cif_path: Path) -> dict[str, object]:
    from Bio.PDB import MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure(str(record["instance"]), cif_path)
    model = next(structure.get_models())
    antibody_chains = [str(record["hchain"])]
    light = str(record["lchain"])
    if light not in {"", "NA", "None", "nan"}:
        antibody_chains.append(light)
    antigen_chains = list(protein_antigen_chains(record))
    ab_parts = [observed_chain(model, chain) for chain in antibody_chains]
    ag_parts = [observed_chain(model, chain) for chain in antigen_chains]
    ab_sequence = "".join(part[0] for part in ab_parts)
    ag_sequence = "".join(part[0] for part in ag_parts)
    return {
        "instance_id": str(record["instance"]),
        "pdb_id": str(record["pdb"]),
        "antibody_chain_ids": antibody_chains,
        "antigen_chain_ids": antigen_chains,
        "ab_sequence": ab_sequence,
        "ag_sequence": ag_sequence,
        "ab_sequence_sha256": sha256_bytes(ab_sequence.encode("ascii")),
        "ag_sequence_sha256": sha256_bytes(ag_sequence.encode("ascii")),
        "ab_length": len(ab_sequence),
        "ag_length": len(ag_sequence),
        "ab_backbone_complete_count": sum(part[1] for part in ab_parts),
        "ag_backbone_complete_count": sum(part[1] for part in ag_parts),
        "cif_path": str(cif_path),
        "cif_sha256": sha256_file(cif_path),
    }


def load_references(path: Path) -> dict[str, tuple[tuple[str, str], ...]]:
    references: dict[str, list[tuple[str, str]]] = {"antibody": [], "antigen": []}
    seen: dict[str, set[str]] = {"antibody": set(), "antigen": set()}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for role, key in (("antibody", "ab_seq"), ("antigen", "ag_seq")):
                sequence = str(row[key]).upper()
                digest = sha256_bytes(sequence.encode("ascii"))
                if digest not in seen[role]:
                    seen[role].add(digest)
                    references[role].append((str(row["instance_id"]), sequence))
    return {role: tuple(values) for role, values in references.items()}


def init_worker(references: dict[str, tuple[tuple[str, str], ...]]) -> None:
    global _REFERENCES
    _REFERENCES = references


def kmer_set(sequence: str, k: int = 3) -> set[str]:
    if len(sequence) < k:
        return {sequence}
    return {sequence[index:index + k] for index in range(len(sequence) - k + 1)}


def alignment_metrics(query: str, target: str) -> tuple[float, float, float]:
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query, target)[0]
    query_blocks, target_blocks = alignment.aligned
    aligned = 0
    matches = 0
    for (q0, q1), (t0, t1) in zip(query_blocks, target_blocks, strict=True):
        length = int(q1 - q0)
        if length != int(t1 - t0):
            raise RuntimeError("alignment block lengths differ")
        aligned += length
        matches += sum(a == b for a, b in zip(query[q0:q1], target[t0:t1]))
    identity = matches / aligned if aligned else 0.0
    return identity, aligned / len(query), aligned / len(target)


def audit_query(task: tuple[str, str, str]) -> dict[str, object]:
    instance_id, role, query = task
    qkmers = kmer_set(query)
    candidates: list[tuple[int, int, str, str]] = []
    length_pruned = 0
    for target_id, target in _REFERENCES[role]:
        length_ratio = min(len(query), len(target)) / max(len(query), len(target))
        if length_ratio + 1e-12 < RECIPROCAL_COVERAGE_THRESHOLD:
            length_pruned += 1
            continue
        shared = len(qkmers.intersection(kmer_set(target)))
        candidates.append((-shared, abs(len(query) - len(target)), target_id, target))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    best = {"identity": 0.0, "query_coverage": 0.0, "target_coverage": 0.0}
    aligned_count = 0
    for _, _, target_id, target in candidates:
        identity, query_coverage, target_coverage = alignment_metrics(query, target)
        aligned_count += 1
        if identity > best["identity"]:
            best = {
                "identity": identity,
                "query_coverage": query_coverage,
                "target_coverage": target_coverage,
                "target_instance_id": target_id,
            }
        if (
            identity + 1e-12 >= IDENTITY_THRESHOLD
            and query_coverage + 1e-12 >= RECIPROCAL_COVERAGE_THRESHOLD
            and target_coverage + 1e-12 >= RECIPROCAL_COVERAGE_THRESHOLD
        ):
            return {
                "instance_id": instance_id,
                "role": role,
                "cold": False,
                "violating_hit": {
                    "target_instance_id": target_id,
                    "identity": identity,
                    "query_coverage": query_coverage,
                    "target_coverage": target_coverage,
                },
                "exact_alignments_run": aligned_count,
                "length_pruned_references": length_pruned,
            }
    return {
        "instance_id": instance_id,
        "role": role,
        "cold": True,
        "maximum_observed_identity": best,
        "exact_alignments_run": aligned_count,
        "length_pruned_references": length_pruned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--identity-manifest-sha256", required=True)
    parser.add_argument("--reference-sequences", type=Path, required=True)
    parser.add_argument("--reference-sequences-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    require_sha(args.identity_manifest, args.identity_manifest_sha256, "identity manifest")
    require_sha(args.reference_sequences, args.reference_sequences_sha256, "reference sequences")
    require_sha(args.protocol, args.protocol_sha256, "protocol")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("format_version") != "framecontact_temporal_double_cold_protocol_v1"
        or protocol.get("status") != "FROZEN_BEFORE_STRUCTURE_OR_LABEL_ACCESS"
        or protocol.get("selection", {}).get("allowed") is not False
        or protocol.get("label_free_gate", {}).get("script_sha256")
        != sha256_file(Path(__file__))
    ):
        raise ValueError("temporal protocol differs")
    for final_name in (
        "TEMPORAL_DOUBLE_COLD_LABEL_FREE_MANIFEST_V1.json",
        "TEMPORAL_DOUBLE_COLD_LABEL_FREE_RECEIPT_V1.json",
    ):
        if (args.output_root / final_name).exists():
            raise FileExistsError(args.output_root / final_name)

    identity = json.loads(args.identity_manifest.read_text())
    records = identity["records"]
    raw_root = args.output_root / "label-free-work-v1" / "cropped-cif"
    parsed: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        safe_id = str(record["instance"]).replace("/", "_")
        cif_path = raw_root / f"{safe_id}.cif"
        download_cropped_cif(record, cif_path)
        parsed.append(parse_instance(record, cif_path))
        print(json.dumps({"phase": "download_parse", "done": index, "total": len(records), "instance": safe_id}), flush=True)

    references = load_references(args.reference_sequences)
    tasks = [
        (row["instance_id"], role, row["ab_sequence" if role == "antibody" else "ag_sequence"])
        for row in parsed
        for role in ("antibody", "antigen")
    ]
    audits: dict[str, dict[str, dict[str, object]]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(references,),
    ) as executor:
        futures = {executor.submit(audit_query, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            audits.setdefault(str(result["instance_id"]), {})[str(result["role"])] = result
            completed += 1
            print(json.dumps({"phase": "exact_nw", "done": completed, "total": len(tasks), "instance": result["instance_id"], "role": result["role"], "cold": result["cold"]}), flush=True)

    double_cold_ids = sorted(
        row["instance_id"] for row in parsed
        if audits[str(row["instance_id"])]["antibody"]["cold"]
        and audits[str(row["instance_id"])]["antigen"]["cold"]
    )
    payload = {
        "format_version": "sabdab2_temporal_double_cold_label_free_manifest_v1",
        "status": "PASS_LABEL_FREE_DOUBLE_COLD_FREEZE",
        "protocol_sha256": args.protocol_sha256,
        "identity_manifest_sha256": args.identity_manifest_sha256,
        "reference_sequences_sha256": args.reference_sequences_sha256,
        "definition": {
            "alignment": "global Needleman-Wunsch; BLOSUM62; gap-open=-10; gap-extend=-0.5",
            "identity_threshold": IDENTITY_THRESHOLD,
            "query_coverage_threshold": RECIPROCAL_COVERAGE_THRESHOLD,
            "target_coverage_threshold": RECIPROCAL_COVERAGE_THRESHOLD,
            "double_cold": "no qualifying antibody hit and no qualifying antigen hit to frozen train+validation",
        },
        "counts": {
            "identity_instances": len(records),
            "unique_pdbs": len({row["pdb_id"] for row in parsed}),
            "double_cold_instances": len(double_cold_ids),
            "double_cold_unique_pdbs": len({row["pdb_id"] for row in parsed if row["instance_id"] in double_cold_ids}),
            "reference_antibody_sequences": len(references["antibody"]),
            "reference_antigen_sequences": len(references["antigen"]),
        },
        "double_cold_instance_ids": double_cold_ids,
        "instances": [
            {
                **row,
                "antibody_cold_audit": audits[str(row["instance_id"])]["antibody"],
                "antigen_cold_audit": audits[str(row["instance_id"])]["antigen"],
                "double_cold": str(row["instance_id"]) in double_cold_ids,
            }
            for row in sorted(parsed, key=lambda value: str(value["instance_id"]))
        ],
        "audit": {
            "task_labels_read": False,
            "cross_chain_distances_computed": False,
            "contact_labels_computed": False,
            "old_test_accessed": False,
            "selection_performed": False,
        },
    }
    result_path = args.output_root / "TEMPORAL_DOUBLE_COLD_LABEL_FREE_MANIFEST_V1.json"
    atomic_write(result_path, canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_temporal_double_cold_label_free_receipt_v1",
        "status": payload["status"],
        "manifest_sha256": sha256_file(result_path),
        "protocol_sha256": args.protocol_sha256,
        "double_cold_instances": len(double_cold_ids),
        "task_labels_read": False,
        "cross_chain_distances_computed": False,
        "selection_performed": False,
    }
    receipt_path = args.output_root / "TEMPORAL_DOUBLE_COLD_LABEL_FREE_RECEIPT_V1.json"
    atomic_write(receipt_path, canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
