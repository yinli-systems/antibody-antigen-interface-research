from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interaction_first.sabdab2_structure import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_topology_routing_registry_is_isolated_and_hash_bound() -> None:
    from interaction_first.exploratory_experiments import EXPLORATORY_EXPERIMENTS
    from interaction_first.topology_routing_experiments import (
        TOPOLOGY_ROUTING_EXPERIMENTS,
    )

    confirmatory_names = {
        "sequence_dense",
        "spatial_topology",
        "distance",
        "distance_direction",
        "full_pose",
        "frozen_frame_ipa",
        "framecontact",
        "framecontact_interleaved",
    }
    assert confirmatory_names.isdisjoint(TOPOLOGY_ROUTING_EXPERIMENTS)
    assert set(EXPLORATORY_EXPERIMENTS).isdisjoint(
        TOPOLOGY_ROUTING_EXPERIMENTS
    )
    assert tuple(TOPOLOGY_ROUTING_EXPERIMENTS) == (
        "topology_routed_multiscale",
    )

    protocol_path = (
        REPOSITORY_ROOT
        / "protocols/framecontact_topology_routing_exploratory_v1.json"
    )
    receipt_path = (
        REPOSITORY_ROOT
        / "protocols/FRAMECONTACT_TOPOLOGY_ROUTING_EXPLORATORY_V1_RECEIPT.json"
    )
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["protocol_file_sha256"] == hashlib.sha256(
        protocol_raw
    ).hexdigest()
    assert receipt["protocol_canonical_sha256"] == hashlib.sha256(
        canonical_json_bytes(protocol)
    ).hexdigest()
    assert protocol["condition"] == "topology_routed_multiscale"
    assert protocol["evidence_boundary_at_freeze"][
        "topology_routing_training_started"
    ] is False
    assert receipt["frozen_test_authorized"] is False
    assert receipt["official_test_materialization_authorized"] is False


def test_topology_routing_scales_match_disclosed_validation_diagnostic() -> None:
    protocol = json.loads(
        (
            REPOSITORY_ROOT
            / "protocols/framecontact_topology_routing_exploratory_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["architecture"]["registered_cumulative_radii_angstrom"] == [
        6.0,
        8.0,
        12.0,
    ]
    evidence = protocol["adaptation_disclosure"]["observed_scale_evidence"]
    assert evidence["spatial_8a_minus_sequence_window_8_ci95"][0] > 0
    assert evidence["spatial_16a_minus_spatial_8a_ci95"][1] < 0
