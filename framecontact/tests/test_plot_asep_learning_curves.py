from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_asep_learning_curves.py"
SPEC = importlib.util.spec_from_file_location("plot_asep_learning_curves", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_run(root: Path, seed: int, *, test_loaded: bool = False) -> Path:
    run = root / "condition" / f"seed-{seed}"
    run.mkdir(parents=True)
    result = {
        "experiment": "condition",
        "configuration": {"seed": seed},
        "authority": {"test_loaded": test_loaded, "external_test_loaded": False},
        "selection": {"best_epoch": 1},
        "history": [
            {
                "epoch": epoch,
                "train": {"loss": 1.0 / epoch + seed / 1000},
                "validation": {
                    group: {"auprc": 0.1 * epoch, "mcc": 0.01, "f1": 0.02}
                    for group in MODULE.GROUPS
                },
            }
            for epoch in (1, 2)
        ],
    }
    result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (run / "RUN_RECEIPT.json").write_text(
        json.dumps(
            {
                "result_sha256": digest,
                "seed": seed,
                "test_loaded": test_loaded,
                "external_test_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_summary_preserves_loss_metric_boundary(tmp_path: Path) -> None:
    runs = [MODULE.load_run(_write_run(tmp_path, seed)) for seed in (17, 42)]
    summary = MODULE.summarize(runs)
    assert summary["matched_epoch_count"] == 2
    assert summary["validation_objective_available"] is False
    assert summary["trajectory"][0]["seed_count"] == 2
    assert summary["trajectory"][0]["train_loss"]["mean"] == pytest.approx(1.0295)
    assert summary["trajectory"][1]["validation"]["epitope"]["auprc_mean"] == pytest.approx(0.2)


def test_rejects_test_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Non-development"):
        MODULE.load_run(_write_run(tmp_path, 17, test_loaded=True))


def test_rejects_hash_mismatch(tmp_path: Path) -> None:
    run = _write_run(tmp_path, 17)
    (run / "FRAMECONTACT_DEVELOPMENT_RESULT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.load_run(run)
