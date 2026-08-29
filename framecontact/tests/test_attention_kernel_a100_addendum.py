"""Hash and claim-boundary checks for the AB03 A100 kernel addendum."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = ROOT / "protocols/framecontact_attention_kernel_a100_addendum_v1.json"
RECEIPT = (
    ROOT
    / "protocols/FRAMECONTACT_ATTENTION_KERNEL_A100_ADDENDUM_V1_RECEIPT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_a100_addendum_is_hash_bound_and_claim_safe() -> None:
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    benchmark_path = ROOT / addendum["execution"]["benchmark_receipt"]["path"]
    run_receipt_path = ROOT / addendum["execution"]["run_receipt"]["path"]
    source_manifest_path = (
        ROOT / "receipts/attention-kernel-a100-v3-ab03/SOURCE_MANIFEST.json"
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))

    assert _sha256(ADDENDUM) == receipt["addendum_file_sha256"]
    assert _canonical_sha256(addendum) == receipt["addendum_canonical_sha256"]
    assert _sha256(benchmark_path) == receipt["benchmark_sha256"]
    assert _sha256(run_receipt_path) == receipt["run_receipt_sha256"]
    assert _sha256(source_manifest_path) == receipt["source_manifest_sha256"]
    assert run_receipt["benchmark_sha256"] == receipt["benchmark_sha256"]
    assert run_receipt["source_tree_sha256"] == receipt["source_tree_sha256"]
    assert all(row["passed"] for row in benchmark["correctness"])

    boundary = addendum["promotion_boundary"]
    assert boundary["a100_kernel_correctness_claim_authorized"] is True
    assert boundary["a100_shape_specific_latency_and_allocator_claim_authorized"] is True
    assert boundary["model_quality_claim_authorized"] is False
    assert boundary["frozen_test_claim_authorized"] is False
    assert boundary["external_generalization_claim_authorized"] is False
    assert boundary["custom_fused_kernel_claim_authorized"] is False
    assert boundary["state_of_the_art_claim_authorized"] is False
    assert boundary["current_frozen_training_changed"] is False
    assert boundary["default_kernel_changed"] is False
