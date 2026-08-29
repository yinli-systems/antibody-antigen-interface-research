#!/usr/bin/env python3
"""Small end-to-end contract test for resumable paired decontamination."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
COLS = (
    "heavy_sequence",
    "light_sequence",
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
    "study",
    "run",
    "species",
    "donor_hash",
    "row_hash",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_gzip(path: Path, payload: bytes) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            out.write(payload)


class PairedDecontaminationContractTest(unittest.TestCase):
    def test_protocol_v3_compound_exclusion_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = [f"{index:064x}" for index in range(4)]
            lines = [
                f"{hashes[1]}|heavy\tsabdab2|dev1|heavy|sha256={'b'*64}\t0.850\t100\t0.900\t0.900\t0\t10\n",
                f"{hashes[2]}|heavy\tsabdab2|dev2|heavy|sha256={'d'*64}\t0.850\t100\t0.900\t0.900\t0\t10\n",
                f"{hashes[0]}|heavy\tsabdab2|dev0|heavy|sha256={'a'*64}\t0.900\t100\t0.800\t0.800\t0\t10\n",
                f"{hashes[3]}|heavy\tsabdab2|dev4|heavy|sha256={'f'*64}\t0.950\t100\t1.000\t0.700\t0\t10\n",
                f"{hashes[1]}|light\tsabdab2|dev1|light|sha256={'c'*64}\t0.820\t100\t0.900\t0.900\t0\t10\n",
                f"{hashes[2]}|light\tsabdab2|dev3|light|sha256={'e'*64}\t0.850\t100\t0.900\t0.900\t0\t10\n",
            ]
            shard_root = root / "shards"
            shard_root.mkdir()
            shard = shard_root / "hits-shard-000.tsv.gz"
            write_gzip(shard, "".join(lines).encode())
            manifest = shard_root / "manifest.json"
            write_json(
                manifest,
                {
                    "shard_count": 1,
                    "total_hit_lines": len(lines),
                    "shards": [
                        {
                            "index": 0,
                            "path": shard.name,
                            "line_count": len(lines),
                            "sha256": sha256_file(shard),
                        }
                    ],
                },
            )
            script = HERE / "reduce_paired_hit_shard_protocol_v3_unordered_v2.py"
            plan = root / "plan.json"
            write_json(
                plan,
                {
                    "format_version": "paired_esmc_cdr_decontamination_plan_v2",
                    "status": "FROZEN_PROTOCOL_V3_EXCLUSION_BEFORE_REDUCTION",
                    "shards": {
                        "count": 1,
                        "manifest": str(manifest),
                        "manifest_sha256": sha256_file(manifest),
                    },
                    "scripts": {script.name: sha256_file(script)},
                    "recovery": {"thresholds_changed": False},
                    "audit": {"test_access": False},
                },
            )
            output_root = root / "reduced"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--plan",
                    str(plan),
                    "--plan-sha256",
                    sha256_file(plan),
                    "--shard-index",
                    "0",
                    "--output-root",
                    str(output_root),
                ],
                check=True,
                capture_output=True,
            )
            result = json.loads(
                (output_root / "shard-000/SHARD_REDUCTION_RESULT_V2.json").read_text()
            )
            self.assertEqual(result["counts"]["single_chain_90pct_pairs"], 1)
            self.assertEqual(result["counts"]["paired_same_instance_80pct_pairs"], 1)
            self.assertEqual(result["counts"]["excluded_union_pairs"], 2)
            with gzip.open(
                output_root / "shard-000/excluded-row-hashes.tsv.gz", "rt"
            ) as handle:
                self.assertEqual(handle.read().splitlines(), hashes[:2])

    def test_two_shard_pipeline_excludes_complete_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = ("00" + "0" * 62, "01" + "0" * 62, "ff" + "f" * 62)
            source = root / "pairs.tsv.gz"
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(
                buffer, fieldnames=COLS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for index, row_hash in enumerate(hashes):
                writer.writerow(
                    {
                        "heavy_sequence": "ACDE",
                        "light_sequence": "FGHI",
                        "cdr1_aa_heavy": "A",
                        "cdr2_aa_heavy": "C",
                        "cdr3_aa_heavy": "D",
                        "cdr1_aa_light": "F",
                        "cdr2_aa_light": "G",
                        "cdr3_aa_light": "H",
                        "study": "s",
                        "run": str(index),
                        "species": "human",
                        "donor_hash": f"{index + 10:064x}",
                        "row_hash": row_hash,
                    }
                )
            write_gzip(source, buffer.getvalue().encode())
            source_receipt = root / "source.receipt.json"
            write_json(
                source_receipt,
                {
                    "format_version": "abpairfm_oas_stage_b_candidates_v2",
                    "counts": {"unique_exact_pairs": 3},
                    "test_access": False,
                },
            )

            shard_root = root / "shards"
            shard_root.mkdir()
            records = []
            lines = {
                0: [
                    f"{hashes[0]}|heavy\tdev|heavy\t0.500\t4\t0.800\t1.000\t0\t10\n",
                    f"{hashes[0]}|light\tdev|light\t0.900\t4\t1.000\t1.000\t0\t20\n",
                ],
                1: [
                    f"{hashes[2]}|light\tdev|light\t0.700\t4\t0.900\t1.000\t0\t15\n"
                ],
            }
            for index in range(2):
                path = shard_root / f"hits-shard-{index:03d}.tsv.gz"
                write_gzip(path, "".join(lines[index]).encode())
                records.append(
                    {
                        "index": index,
                        "path": path.name,
                        "line_count": len(lines[index]),
                        "sha256": sha256_file(path),
                    }
                )
            manifest_path = shard_root / "manifest.json"
            write_json(
                manifest_path,
                {
                    "status": "PASS_BOUNDED_SHARD_STAGING",
                    "shard_count": 2,
                    "total_hit_lines": 3,
                    "shards": records,
                    "audit": {"test_access": False},
                },
            )
            plan_path = root / "plan.json"
            write_json(
                plan_path,
                {
                    "format_version": "paired_esmc_cdr_decontamination_plan_v1",
                    "status": "FROZEN_BEFORE_SHARD_REDUCTION",
                    "hits": {"sha256": "frozen-hits"},
                    "shards": {
                        "count": 2,
                        "manifest": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                    },
                    "source_pairs": {
                        "path": str(source),
                        "sha256": sha256_file(source),
                        "receipt": str(source_receipt),
                        "receipt_sha256": sha256_file(source_receipt),
                        "count": 3,
                    },
                    "decontamination": {"minimum_sequence_identity": 0.5},
                    "scripts": {
                        name: sha256_file(HERE / name)
                        for name in (
                            "reduce_paired_hit_shard_v1.py",
                            "merge_paired_hit_shards_v1.py",
                            "materialize_paired_devclean_v1.py",
                        )
                    },
                    "audit": {"test_access": False},
                },
            )
            plan_sha = sha256_file(plan_path)
            reductions = root / "reductions"
            for index in range(2):
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "reduce_paired_hit_shard_v1.py"),
                        "--plan",
                        str(plan_path),
                        "--plan-sha256",
                        plan_sha,
                        "--shard-index",
                        str(index),
                        "--output-root",
                        str(reductions),
                    ],
                    check=True,
                    capture_output=True,
                )
            merged = root / "merged"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "merge_paired_hit_shards_v1.py"),
                    "--plan",
                    str(plan_path),
                    "--plan-sha256",
                    plan_sha,
                    "--reductions-root",
                    str(reductions),
                    "--output-root",
                    str(merged),
                ],
                check=True,
                capture_output=True,
            )
            output = root / "filtered.tsv.gz"
            receipt = root / "filtered.receipt.json"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "materialize_paired_devclean_v1.py"),
                    "--plan",
                    str(plan_path),
                    "--plan-sha256",
                    plan_sha,
                    "--merged-root",
                    str(merged),
                    "--output",
                    str(output),
                    "--receipt",
                    str(receipt),
                ],
                check=True,
                capture_output=True,
            )
            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                kept = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["row_hash"] for row in kept], [hashes[1]])
            final = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(final["counts"]["excluded_pairs"], 2)
            self.assertEqual(final["counts"]["kept_pairs"], 1)
            self.assertFalse(final["audit"]["test_access"])


if __name__ == "__main__":
    unittest.main()
