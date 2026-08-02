import hashlib
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from antibody_pretraining.model import (  # noqa: E402
    AntibodyMaskedLanguageModel,
    AntibodyModelConfig,
    config_from_preset,
    load_model_family,
    parameter_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = ROOT / "protocols/antibody_model_family_v1.json"


def test_small_model_forward_loss_and_tied_output() -> None:
    config = AntibodyModelConfig(
        d_model=64,
        ffn_hidden_size=128,
        maximum_sequence_tokens=16,
        n_heads=4,
        n_layers=2,
    )
    model = AntibodyMaskedLanguageModel(config)
    input_ids = torch.tensor([[2, 5, 6, 7, 3, 0], [2, 8, 9, 10, 3, 0]])
    type_ids = torch.ones_like(input_ids)
    attention_mask = input_ids.ne(0)
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]

    output = model(input_ids, type_ids, attention_mask, labels)

    assert output.logits.shape == (2, 6, config.vocabulary_size)
    assert output.target_count.item() == 2
    assert output.loss_sum is not None and torch.isfinite(output.loss_sum)
    output.loss_sum.backward()
    assert model.token_embedding.weight.grad is not None


def test_named_presets_have_exact_meta_parameter_receipts() -> None:
    family = load_model_family(FAMILY_PATH)
    family_sha256 = hashlib.sha256(FAMILY_PATH.read_bytes()).hexdigest()
    counts = {}
    with torch.device("meta"):
        for preset in ("abf_35m", "abf_300m", "abf_650m"):
            model = AntibodyMaskedLanguageModel(config_from_preset(family, preset))
            receipt = parameter_receipt(
                model,
                preset=preset,
                family_file_sha256=family_sha256,
            )
            counts[preset] = receipt["total_parameter_count"]
            assert receipt["total_parameter_count"] == receipt[
                "trainable_parameter_count"
            ]
            assert sum(receipt["parameter_count_by_module"].values()) == receipt[
                "total_parameter_count"
            ]

    assert 25_000_000 <= counts["abf_35m"] <= 40_000_000
    assert 275_000_000 <= counts["abf_300m"] <= 340_000_000
    assert 600_000_000 <= counts["abf_650m"] <= 700_000_000
