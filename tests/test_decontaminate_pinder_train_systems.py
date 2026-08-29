"""Unit contracts for PINDER strict-train development decontamination."""

from __future__ import annotations

import csv
import gzip
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/decontaminate_pinder_train_systems.py"
SBATCH = ROOT / "scripts/decontaminate_pinder_train_systems.sbatch"


def _module():
    specification = importlib.util.spec_from_file_location("pinder_decontam", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_excluded_endpoints_enforces_identity_and_coverage(tmp_path: Path) -> None:
    module = _module()
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "dev\ta-R\t0.5\t100\t0.8\t0.8\t1e-20\t100\n",
        encoding="utf-8",
    )
    excluded, counts = module.excluded_endpoints(hits, {"a-R", "b-L"})
    assert excluded == {"a-R"}
    assert counts == {
        "development_queries_with_hits": 1,
        "excluded_endpoint_sequences": 1,
        "homology_hit_rows": 1,
    }


def test_filter_manifest_removes_either_endpoint_deterministically(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "train.tsv.gz"
    fields = ("split", "id", "cluster_id", "holo_R_pdb", "holo_L_pdb")
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "id": "remove",
                "cluster_id": "c1",
                "holo_R_pdb": "a-R.pdb",
                "holo_L_pdb": "b-L.pdb",
            }
        )
        writer.writerow(
            {
                "split": "train",
                "id": "keep",
                "cluster_id": "c2",
                "holo_R_pdb": "c-R.pdb",
                "holo_L_pdb": "d-L.pdb",
            }
        )
    output = tmp_path / "clean.tsv.gz"
    counts = module.filter_manifest(
        source,
        output,
        {"a-R", "b-L", "c-R", "d-L"},
        {"a-R"},
    )
    assert counts == {
        "input_clusters": 2,
        "input_systems": 2,
        "output_clusters": 1,
        "output_systems": 1,
        "removed_systems": 1,
    }
    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        assert [row["id"] for row in csv.DictReader(handle, delimiter="\t")] == [
            "keep"
        ]


def test_sbatch_freezes_exact_mmseqs_gate_on_gpu_4090() -> None:
    source = SBATCH.read_text(encoding="ascii")
    assert "#SBATCH -p gpu_4090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "hp_4090" not in source
    assert "--min-seq-id 0.5" in source
    assert "-c 0.8 --cov-mode 0 --alignment-mode 3" in source
    assert "/envs/contact/bin/python3.10" in source
    assert "pinder_holo_sequences_v3.fasta" in source
