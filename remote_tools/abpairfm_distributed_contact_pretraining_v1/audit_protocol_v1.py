#!/usr/bin/env python3
"""Fail-closed static audit for the distributed pretraining protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    if protocol.get("format_version") != "abpairfm_distributed_contact_pretraining_v1":
        raise SystemExit("Protocol format differs")
    if protocol.get("status") != "FROZEN_B32K_AUTHORIZED":
        raise SystemExit("Protocol status differs")
    if protocol.get("test_access") is not False:
        raise SystemExit("Test access must remain false")
    authorization = protocol.get("authorization", {})
    if (
        authorization.get("stage_a0_continuation") is not True
        or authorization.get("systems_canary") is not True
        or authorization.get("general_ppi_biological_training") is not True
        or authorization.get("B32K_training") is not True
        or authorization.get("B500K_training") is not False
        or authorization.get("antibody_adaptation") is not False
        or authorization.get("external_test_access") is not False
    ):
        raise SystemExit("Authorization boundary differs")
    if protocol.get("data", {}).get("antibody_split", {}).get("status") != "PENDING_INDEPENDENT_ZERO_CROSSING_AUDIT":
        raise SystemExit("Antibody split must remain pending")
    if protocol.get("systems", {}).get("production") != {
        "gpu": "RTX_5090",
        "single_node_world_size": 8,
    }:
        raise SystemExit("Production topology differs")
    if protocol.get("systems", {}).get("b32k") != {
        "gpu": "RTX_4090",
        "single_node_world_size": 8,
    }:
        raise SystemExit("B32K topology differs")

    source_paths = {
        "curriculum_builder": args.repository_root / "remote_tools/abpairfm_distributed_contact_pretraining_v1/build_pinder_curriculum_manifests_v1.py",
        "contact_label_materializer": args.repository_root / "remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py",
        "contact_label_shard_materializer": args.repository_root / "remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_label_shard_v1.py",
        "contact_label_shard_merger": args.repository_root / "remote_tools/abpairfm_distributed_contact_pretraining_v1/merge_pinder_contact_label_shards_v1.py",
        "b_canary_runner": args.repository_root / "remote_tools/abpairfm_distributed_contact_pretraining_v1/run_pinder_b_canary_v1.py",
        "frontier_fsdp2": args.repository_root / "src/interaction_first/frontier_fsdp2.py",
        "objective": args.repository_root / "src/antibody_pretraining/paircontact_pretraining.py",
        "pair_aware_attention": args.repository_root / "src/antibody_pretraining/pair_aware_attention.py",
        "pair_aware_distributed": args.repository_root / "src/antibody_pretraining/pair_aware_distributed.py",
        "pair_aware_esm2": args.repository_root / "src/antibody_pretraining/pair_aware_esm2.py",
    }
    expected = protocol.get("source_sha256", {})
    observed = {name: sha256(path) for name, path in source_paths.items()}
    if observed != expected:
        raise SystemExit(f"Frozen source hashes differ: {observed}")
    print(
        json.dumps(
            {
                "protocol_sha256": sha256(args.protocol),
                "source_sha256": observed,
                "status": "PASS_PROTOCOL_AND_SOURCE_FROZEN",
                "test_access": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
