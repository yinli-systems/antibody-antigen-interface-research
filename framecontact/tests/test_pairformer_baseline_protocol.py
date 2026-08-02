"""Isolation and hash-binding checks for the adapted Pairformer baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "protocols/framecontact_pairformer_baseline_v1.json"
RECEIPT_PATH = (
    REPOSITORY_ROOT / "protocols/FRAMECONTACT_PAIRFORMER_BASELINE_V1_RECEIPT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_pairformer_protocol_is_hash_bound_and_test_sealed() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    lock_path = REPOSITORY_ROOT / protocol["baseline_lock"]["path"]
    assert _sha256(PROTOCOL_PATH) == receipt["protocol_file_sha256"]
    assert _canonical_sha256(protocol) == receipt["protocol_canonical_sha256"]
    assert _sha256(lock_path) == protocol["baseline_lock"]["sha256"]
    assert receipt["baseline_lock_sha256"] == protocol["baseline_lock"]["sha256"]
    assert protocol["condition"] == "adapted_boltz1_pairformer"
    assert protocol["frozen_test_action"].startswith("forbidden")
    assert receipt["frozen_test_authorized"] is False
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]


def test_pairformer_condition_is_absent_from_confirmatory_registry() -> None:
    script = (
        "import json, pathlib, runpy; "
        "d=runpy.run_path('scripts/train_framecontact.py', run_name='not_main'); "
        "experiments=d['EXPERIMENTS']; "
        "primary=json.loads(pathlib.Path('protocols/framecontact_primary_v5.json').read_text()); "
        "assert 'adapted_boltz1_pairformer' not in experiments; "
        "assert len(primary['development_matrix']) == 8; "
        "assert set(primary['development_matrix']).issubset(experiments); "
        "assert not ({'framecontact_unary_only', 'framecontact_contact_marginal', "
        "'framecontact_contact_hybrid', 'framecontact_distogram', "
        "'framecontact_distogram_triangle'} & set(primary['development_matrix']))"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_pairformer_training_entry_exposes_only_external_condition() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_pairformer_baseline.py",
            "--pairformer-artifact",
            "/not/read/by/help/artifact.pt",
            "--pairformer-receipt",
            "/not/read/by/help/receipt.json",
            "--boltz-source-root",
            "/not/read/by/help/boltz",
            "--help",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "adapted_boltz1_pairformer" in result.stdout
    assert "sequence_dense" not in result.stdout
