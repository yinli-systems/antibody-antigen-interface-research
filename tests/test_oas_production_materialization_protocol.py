"""Validate the frozen OAS production-materialization protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/oas_production_materialization_v1.json"


def test_protocol_preserves_input_authority_and_test_boundary() -> None:
    """Bind the audited raw inventory without opening downstream held-out data."""
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["format_version"] == "oas_production_materialization_protocol_v1"
    assert value["inputs"] == {
        "download_completion_audit_file_sha256": (
            "58557ba1ff7b1f30b48a0d5ac167b2ceb734770ced4d38ec4bf77bb25d55dbd2"
        ),
        "download_completion_audit_payload_sha256": (
            "f767d026f17e06a8532a7d687605a3cee2d4eaad0b406f52e8a34e441458683b"
        ),
        "download_receipt_count": 6_373,
        "filter_policy_file_sha256": (
            "5f7e360ff731805fd87f33297c4cba09b9f5c07ddf73c96d4850db54b09106e7"
        ),
        "plan_file_sha256": (
            "45f7d002400dd81d9afa5f6b7daff6b6e67d7557aa2e379a031a5ac3b1f2b1b0"
        ),
        "plan_payload_sha256": (
            "fba741bc6666067f1601f2096d31507c1b38edb8af95118110eaf1cef7e059e0"
        ),
        "raw_compressed_bytes": 120_714_440_652,
        "raw_inventory_sha256": (
            "9bf131fd0e79e3c3a0f112a087fabc441010924dff773b67752ae191ce11ce44"
        ),
        "source_record_count": 265_834_820,
    }
    assert value["test_access"] == {
        "downstream_metrics_used": False,
        "sabdab2_heldout_accessed": False,
    }
    assert value["authorization"]["large_scale_pretraining"] is False
    assert value["authorization"]["production_corpus_use"] is False


def test_protocol_requires_validation_precedence_and_independent_replay() -> None:
    """Prevent leakage or a mapper-generated receipt from authorizing training."""
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["deduplication"]["cross_split_rule"] == (
        "validation_wins_and_all_matching_training_records_are_removed"
    )
    assert value["deduplication"]["hash_collision_policy"].startswith("fail_")
    assert value["deduplication"]["sample_key_collision_policy"].startswith(
        "fail_"
    )
    assert value["quality_gates"]["independent_auditor_imports_no_mapper_or_reducer"]
    assert value["quality_gates"]["validation_studies_absent_from_training_units"]
    assert value["execution"]["reduce_task_count"] == value["execution"][
        "bucket_count"
    ]
    assert value["map_record"]["fixed_record_bytes"] == 240


def test_protocol_does_not_allocate_gpus_for_cpu_materialization() -> None:
    """Reserve GPU allocations for model work rather than CSV preprocessing."""
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["resource_policy"]["gpu_allocation_during_materialization"] is False
    assert value["resource_policy"]["shared_storage_minimum_free_bytes"] >= 10**12
