from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_abpairfm_checkpoint_transfer_probe.py"
PROTOCOL = ROOT / "protocols" / "abpairfm_checkpoint_transfer_probe_v1.json"
SCALE_PROTOCOL = ROOT / "protocols" / "abpairfm_650m_scale_control_v1.json"
SUBMISSION = ROOT / "scripts" / "submit_abpairfm_checkpoint_transfer_probe.sbatch"


def _module():
    specification = importlib.util.spec_from_file_location("checkpoint_transfer_probe", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_protocol_freezes_development_only_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["format_version"] == "abpairfm_checkpoint_transfer_probe_v1"
    assert protocol["status"] == "FROZEN_BEFORE_GPU_EVALUATION"
    assert protocol["authorization"] == {
        "development_contact": False,
        "development_pair_retrieval": True,
        "production_training": False,
        "test_access": False,
    }
    assert protocol["checkpoint_ladder"]["targets"] == [
        250_000_000,
        500_000_000,
        1_000_000_000,
    ]
    assert protocol["score"]["probability_interpretation"] is False
    assert protocol["compute"]["partition"] == "gpu_4090"
    assert protocol["compute"]["explicit_mem_argument_permitted"] is False


def test_submission_can_bind_an_immutable_project_root() -> None:
    submission = SUBMISSION.read_text(encoding="utf-8")
    assert "${ABPAIRFM_PROBE_PROJECT:-" in submission
    assert "#SBATCH -p gpu_4090" in submission
    assert "#SBATCH --mem" not in submission


def test_deterministic_masks_are_nonempty_and_endpoint_specific() -> None:
    module = _module()
    key = ("ordered", "a" * 64, "b" * 64)
    first = module._selected_positions(
        sequence="ACDEFGHIKLMNPQRSTVWY",
        pair_key=key,
        endpoint="left",
        mask_seed=1701,
    )
    repeated = module._selected_positions(
        sequence="ACDEFGHIKLMNPQRSTVWY",
        pair_key=key,
        endpoint="left",
        mask_seed=1701,
    )
    right = module._selected_positions(
        sequence="ACDEFGHIKLMNPQRSTVWY",
        pair_key=key,
        endpoint="right",
        mask_seed=1701,
    )
    assert first == repeated
    assert len(first) == 3
    assert first != right
    assert min(first) >= 0 and max(first) < 20


def test_650m_control_is_frozen_to_official_family_dimensions() -> None:
    protocol = json.loads(SCALE_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["format_version"] == "abpairfm_650m_scale_control_v1"
    assert protocol["authorization"]["test_access"] is False
    assert protocol["model_control"]["dimensions"] == {
        "embed_dimension": 1280,
        "ffn_dimension": 5120,
        "head_count": 20,
        "layer_count": 33,
    }
    assert protocol["model_control"]["task_head"] is None


def test_score_part_aggregation_preserves_window_means() -> None:
    module = _module()
    key_a = ("ordered", "a", "b")
    key_b = ("ordered", "c", "d")
    scores = module._aggregate_score_parts(
        [
            {key_a: (-4.0, 2)},
            {key_a: (-3.0, 1), key_b: (-2.0, 1)},
        ]
    )
    assert scores[key_a] == -7.0 / 3.0
    assert scores[key_b] == -2.0
