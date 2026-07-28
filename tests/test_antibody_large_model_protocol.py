import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parameter_count(preset: dict[str, int]) -> int:
    dimension = preset["d_model"]
    hidden = preset["ffn_hidden_size"]
    layers = preset["n_layers"]
    per_block = 4 * dimension * dimension + 3 * dimension * hidden + 2 * dimension
    embeddings = (26 + 7) * dimension
    final_norm = dimension
    return layers * per_block + embeddings + final_norm


def test_large_model_has_exact_source_derived_parameter_count() -> None:
    family = json.loads(
        (ROOT / "protocols/antibody_model_family_large_v1.json").read_text(
            encoding="ascii"
        )
    )
    pilot = json.loads(
        (ROOT / "protocols/antibody_large_model_pilot_v1.json").read_text(
            encoding="ascii"
        )
    )
    preset = family["presets"]["abf_1_3b"]
    assert preset["d_model"] // preset["n_heads"] == 64
    assert _parameter_count(preset) == 1_307_632_000
    assert pilot["model"]["total_parameter_count"] == _parameter_count(preset)


def test_large_model_execution_and_production_are_still_locked() -> None:
    pilot = json.loads(
        (ROOT / "protocols/antibody_large_model_pilot_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert pilot["authorization"] == {
        "eight_gpu_systems_pilot": False,
        "production_pretraining": False,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }
    assert pilot["execution_gate"]["backend"] == "fsdp2"
    assert pilot["execution_gate"]["exact_650m_recovery_required"] is True
    assert pilot["production_gate"]["test_set_selection_permitted"] is False
    assert pilot["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }


def test_large_training_schedule_is_token_based_and_not_epoch_inflated() -> None:
    training = json.loads(
        (ROOT / "protocols/antibody_training_large_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert training["schedule"]["total_residue_tokens"] == 30_000_000_000
    assert training["schedule"]["warmup_residue_tokens"] == 300_000_000
    assert training["presets"]["abf_1_3b"]["learning_rate"] == 0.00012


def test_large_model_verification_is_source_bound_and_exact() -> None:
    receipt_path = (
        ROOT / "results/pretraining/antibody_model_1_3b_verification_v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    parameter_receipt = receipt["parameter_receipts"]["abf_1_3b"]
    assert receipt["checks"] == {
        "cpu_forward_replay_bitwise": True,
        "forward_loss_backward": True,
        "mask_replay_bitwise": True,
        "meta_parameter_count": True,
        "tied_embedding_gradient": True,
    }
    assert parameter_receipt["total_parameter_count"] == 1_307_632_000
    assert parameter_receipt["trainable_parameter_count"] == 1_307_632_000
    assert receipt["family_file_sha256"] == hashlib.sha256(
        (ROOT / "protocols/antibody_model_family_large_v1.json").read_bytes()
    ).hexdigest()
    assert receipt["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
