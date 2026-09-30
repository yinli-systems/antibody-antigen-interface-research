from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from extensions.frontier_multimodel_v1.build_af3_structure_completion import (
    SEEDS,
    main as build_main,
    sequence_sha,
)
from extensions.frontier_multimodel_v1.materialize_af3_structure_completion import (
    ranking_rows,
    selected_top_sample,
)


def write_ranking(path: Path, top_seed: int = 42, top_sample: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "sample", "ranking_score"])
        for seed in SEEDS:
            for sample in range(5):
                score = 0.9 if (seed, sample) == (top_seed, top_sample) else 0.1
                writer.writerow([seed, sample, score])


def test_selected_top_sample_requires_top_level_model_to_match_maximum(tmp_path: Path) -> None:
    result = tmp_path / "result"
    ranking = result / "ranking_scores.csv"
    write_ranking(ranking)
    for seed in SEEDS:
        for sample in range(5):
            source = result / f"seed-{seed}_sample-{sample}" / "model.cif"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{seed}:{sample}".encode())
    top = result / "name_model.cif"
    top.write_bytes((result / "seed-42_sample-3/model.cif").read_bytes())
    score, seed, sample, source = selected_top_sample(result, ranking, top)
    assert score == pytest.approx(0.9)
    assert (seed, sample) == (42, 3)
    assert source == result / "seed-42_sample-3/model.cif"


def test_selected_top_sample_rejects_unrelated_top_level_model(tmp_path: Path) -> None:
    result = tmp_path / "result"
    ranking = result / "ranking_scores.csv"
    write_ranking(ranking)
    for seed in SEEDS:
        for sample in range(5):
            source = result / f"seed-{seed}_sample-{sample}" / "model.cif"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{seed}:{sample}".encode())
    top = result / "name_model.cif"
    top.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="maximum-score"):
        selected_top_sample(result, ranking, top)


def test_ranking_rows_requires_all_25_samples(tmp_path: Path) -> None:
    ranking = tmp_path / "ranking_scores.csv"
    write_ranking(ranking)
    rows = ranking_rows(ranking)
    assert len(rows) == 25
    ranking.write_text("seed,sample,ranking_score\n42,0,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 25"):
        ranking_rows(ranking)


def test_builder_freezes_exact_40_missing_sequences(tmp_path: Path, monkeypatch) -> None:
    canonical = tmp_path / "canonical.csv"
    existing_heavy = "ACDEFGHIK"
    existing_antigen = "LMNPQRSTV"
    with canonical.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["INSTANCE", "vh_seq", "vl_seq", "ag_seq"])
        writer.writeheader()
        for index in range(217):
            heavy = f"ACDEFGHIKW{index:03d}" if index < 40 else existing_heavy
            writer.writerow({
                "INSTANCE": f"v{index:03d}",
                "vh_seq": heavy,
                "vl_seq": "",
                "ag_seq": existing_antigen,
            })
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "splits": {
            "train": ["train"],
            "validation": [f"v{index:03d}" for index in range(217)],
            "test": ["test"],
        }
    }), encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "contact_labels_loaded": False,
        "test_loaded": False,
        "records": [
            {"sequence_sha256": sequence_sha(existing_heavy)},
            {"sequence_sha256": sequence_sha(existing_antigen)},
        ],
    }), encoding="utf-8")
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/bin/bash\n", encoding="utf-8")
    output = tmp_path / "campaign"
    monkeypatch.setattr(sys, "argv", [
        "build_af3_structure_completion.py",
        "--canonical-manifest", str(canonical),
        "--split-manifest", str(split),
        "--base-structure-manifest", str(base),
        "--af3-runner", str(runner),
        "--source-commit", "commit",
        "--output-root", str(output),
    ])
    build_main()
    plan = json.loads((output / "PLAN.json").read_text())
    assert plan["task_count"] == 40
    assert plan["missing_role_counts"] == {"heavy": 40}
    assert plan["contact_labels_read"] is False
    assert plan["test_labels_read"] is False
    assert len(list((output / "inputs").glob("batch-*/*.json"))) == 40
    assert plan["tasks"][0]["length"] <= plan["tasks"][-1]["length"]


def test_completion_protocol_forbids_method_substitution() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "extensions/frontier_multimodel_v1/AF3_STRUCTURE_AUTHORITY_COMPLETION_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "different predictor" in text
    assert "zero coordinates" in text
    assert "No SOTA" in text
