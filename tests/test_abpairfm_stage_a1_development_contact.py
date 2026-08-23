from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_abpairfm_stage_a1_frozen_development_contact as module


ROOT = Path(__file__).resolve().parents[1]


def test_window_spans_cover_long_endpoints_deterministically() -> None:
    spans = module.window_spans(1006, 510)
    assert spans == [(0, 510), (248, 758), (496, 1006)]
    counts = np.zeros(1006, dtype=np.uint8)
    for start, stop in spans:
        counts[start:stop] += 1
    assert counts.min() == 1
    assert counts.max() == 3


def test_contact_batch_preserves_residue_offsets() -> None:
    item = module.WindowWork(
        instance_id="x",
        antibody_start=2,
        antibody_stop=5,
        antigen_start=4,
        antigen_stop=8,
        antibody_sequence="ACD",
        antigen_sequence="EFGH",
    )
    tokens, chain_ids, offsets = module.build_contact_batch(
        [item], torch.device("cpu")
    )
    assert tokens.shape == chain_ids.shape == (1, 11)
    assert chain_ids[0, :5].tolist() == [0] * 5
    assert chain_ids[0, 5:].tolist() == [1] * 6
    assert offsets == [(1, 4, 6, 10)]


def test_average_precision_groups_ties() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.bool_)
    scores = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float32)
    assert module.average_precision(labels, scores) == pytest.approx(5.0 / 6.0)


def test_fixed_raw_ece_is_zero_for_balanced_perfect_bins() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.bool_)
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert module.fixed_raw_ece(labels, probabilities, 2) == 0.0


def test_protocol_freezes_contact_ood_without_test() -> None:
    path = ROOT / "protocols" / "abpairfm_stage_a1_development_contact_v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_GPU_EVALUATION"
    assert protocol["authorization"]["development_contact"] is True
    assert protocol["authorization"]["test_access"] is False
    assert protocol["test_access"] is False
    assert protocol["calibration"]["fitted_parameters"] is False
    assert protocol["split_slices"]["standard_validation"] == 549
    assert protocol["split_slices"]["double_cold_validation"] == 324
    assert protocol["split_slices"]["temporal_validation"] == 337
    assert protocol["compute"]["partition"] == "gpu_4090"


def test_contact_sbatch_uses_only_gpu_4090_and_fixed_python() -> None:
    text = (
        ROOT
        / "scripts"
        / "submit_abpairfm_stage_a1_frozen_development_contact.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in text
    assert "#SBATCH --gres=gpu:8" in text
    assert "hp_4090" not in text
    assert "#SBATCH --mem" not in text
    assert "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10" in text
