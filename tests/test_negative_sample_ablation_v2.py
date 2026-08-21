"""Validate the raw-Brier paired negative-sample ablation closure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "remote_tools" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = load_tool("patch_framecontact_raw_brier", "patch_framecontact_raw_brier.py")
freezer = load_tool(
    "freeze_negative_sample_ablation_v2",
    "freeze_negative_sample_ablation_v2.py",
)
analyzer = load_tool(
    "analyze_negative_sample_ablation",
    "analyze_negative_sample_ablation.py",
)


TRAINER_FIXTURE = '''\
def _sigmoid_numpy(value):
    return value


def binary_classification_metrics(labels, probabilities, *, threshold):
    return {"brier": 0.0}


def evaluate(contact_logits, contact_labels, temperature, contact_threshold):
    contact_probabilities = _sigmoid_numpy(contact_logits / temperature)
    result = {
        "contact": {
            **binary_classification_metrics(
                contact_labels, contact_probabilities, threshold=contact_threshold
            ),
        }
    }
    return result
'''


def test_trainer_instrumentation_is_exact_and_idempotent(tmp_path: Path) -> None:
    trainer = tmp_path / "train_framecontact.py"
    trainer.write_text(TRAINER_FIXTURE, encoding="utf-8")
    first_sha = patcher.instrument_trainer(trainer)
    first = trainer.read_text(encoding="utf-8")
    second_sha = patcher.instrument_trainer(trainer)
    assert first_sha == second_sha
    assert trainer.read_text(encoding="utf-8") == first
    assert first.count("fixed_raw_logit_threshold_zero") == 1
    assert "raw_contact_probabilities = _sigmoid_numpy(contact_logits)" in first
    assert "probability_temperature\": 1.0" in first
    assert "threshold=0.5" in first


def make_source_root(root: Path) -> Path:
    for directory in freezer.TREE_ROOTS:
        (root / directory).mkdir(parents=True)
    (root / "scripts/train_framecontact.py").write_text(
        TRAINER_FIXTURE + '# "fixed_raw_logit_threshold_zero"\n',
        encoding="utf-8",
    )
    (root / "src/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "protocols/base.json").write_text("{}\n", encoding="utf-8")
    return root


def test_v2_freezer_authenticates_arm_and_source_closure(tmp_path: Path) -> None:
    source_root = make_source_root(tmp_path / "source")
    base_protocol = tmp_path / "base.json"
    base_protocol.write_text(
        json.dumps(
            {
                "data": {"dataset_backend": "asep"},
                "model_selection": {
                    "early_stopping_patience": 15,
                    "maximum_epochs": 100,
                    "seeds": [17, 42, 73, 101, 137],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "protocol"
    receipt = freezer.freeze(
        dataset="asep",
        arm="with_negative",
        base_protocol=base_protocol,
        source_root=source_root,
        output_root=output,
    )
    protocol = output / "framecontact_asep_with_negative_v2.json"
    verified = freezer.verify(
        receipt_path=output / "PROTOCOL_RECEIPT.json",
        protocol_path=protocol,
        source_root=source_root,
    )
    assert verified == receipt
    assert receipt["specificity_loss_weight"] == 0.25
    assert receipt["test_loaded"] is False
    assert receipt["external_test_loaded"] is False
    (source_root / "src/model.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source inventory differs"):
        freezer.verify(
            receipt_path=output / "PROTOCOL_RECEIPT.json",
            protocol_path=protocol,
            source_root=source_root,
        )


def test_analyzer_reports_missing_raw_metric_explicitly() -> None:
    with pytest.raises(KeyError, match="fixed_raw_logit_threshold_zero"):
        analyzer.nested(
            {"contact": {"brier": 0.1}},
            ("contact", "fixed_raw_logit_threshold_zero", "brier"),
        )


def test_v2_array_is_paired_and_gpu_constrained() -> None:
    script = (
        ROOT / "remote_tools/framecontact_negative_ablation_v2.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_4090" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --array=0-19%20" in script
    assert "hp_4090" not in script
    assert "negative-sample-ablation-v2" in script
    assert "with_negative" in script and "no_negative" in script
    assert "/test" not in script and "external-test" not in script
