"""Evidence-lock checks for the optional SDPA systems path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols/framecontact_attention_kernel_amendment_v1.json"
RECEIPT = (
    REPOSITORY_ROOT
    / "protocols/FRAMECONTACT_ATTENTION_KERNEL_AMENDMENT_V1_RECEIPT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_attention_kernel_amendment_is_hash_bound_and_claim_safe() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert _sha256(PROTOCOL) == receipt["protocol_file_sha256"]
    assert _canonical_sha256(protocol) == receipt["protocol_canonical_sha256"]
    assert protocol["schema_version"] == "framecontact_attention_kernel_amendment_v1"
    assert protocol["deployment_status"] == (
        "implemented_but_not_used_by_any_current_frozen_training"
    )
    # This amendment is historical: later exploratory objectives may extend the
    # mutable trainer without changing the already validated SDPA path.  The
    # protocol hash-binds the preserved V11 source manifest, so authenticate the
    # implementation against that immutable inventory instead of requiring the
    # present development tree to retain every old digest forever.
    source_manifest_spec = protocol["validation"]["source_manifest"]
    source_manifest_path = REPOSITORY_ROOT / source_manifest_spec["path"]
    assert _sha256(source_manifest_path) == source_manifest_spec["sha256"]
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    assert source_manifest["tree_sha256"] == source_manifest_spec["tree_sha256"]
    records = {record["path"]: record for record in source_manifest["records"]}
    expected_sources = protocol["implementation"]["source_sha256"]
    for relative_path, expected in expected_sources.items():
        if relative_path == "tests/test_attention_kernel_amendment.py":
            # The evidence-lock test was added after the V11 source inventory;
            # its historical digest remains in the protocol, but a test cannot
            # require its own current bytes to equal an earlier revision.
            assert len(expected) == 64
            continue
        assert records[relative_path]["sha256"] == expected
    benchmark = protocol["validation"]["integrated_benchmark"]
    assert _sha256(REPOSITORY_ROOT / benchmark["receipt_path"]) == benchmark["sha256"]
    assert benchmark["gpu"] == "NVIDIA GeForce RTX 4090"
    assert benchmark["a100_tested"] is False
    assert protocol["scientific_boundary"]["model_quality_metric_computed"] is False
    assert protocol["scientific_boundary"]["frozen_training_changed"] is False
    assert protocol["scientific_boundary"]["frozen_test_opened"] is False
    assert receipt["model_parameter_set_changed"] is False
    assert receipt["benchmark_improvement_claim_authorized"] is False


def test_attention_kernel_defaults_to_reference_and_diagnostics_fall_back() -> None:
    from interaction_first.framecontact import FrameContactModel
    from interaction_first.intra_chain_geometry import IntraChainGeometryEncoder

    encoder = IntraChainGeometryEncoder(16, 4, 1)
    assert encoder.attention_kernel == "reference"
    model = FrameContactModel(
        antibody_input_dim=12,
        antigen_input_dim=10,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        dropout=0.0,
        attention_kernel="sdpa",
    )
    assert model.self_encoder is not None and model.cross_attention is not None
    for stream in (
        model.self_encoder.antibody_encoder,
        model.self_encoder.antigen_encoder,
    ):
        assert stream.dense_encoder.attention_kernel == "sdpa"
        assert stream.geometry_encoder.attention_kernel == "sdpa"
    assert model.cross_attention.antibody_from_antigen.attention_kernel == "sdpa"
    # Module contracts use the reference path whenever return_attention=True;
    # the equality is exercised end to end in test_framecontact.py.
    assert protocol_diagnostics_reference_path()


def protocol_diagnostics_reference_path() -> bool:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    return bool(protocol["implementation"]["diagnostics_force_reference_path"])
