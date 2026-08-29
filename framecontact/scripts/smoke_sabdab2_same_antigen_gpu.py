#!/usr/bin/env python3
"""GPU smoke the same-antigen scorer on a label-allowed validation subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_contacts import (  # noqa: E402
    Sabdab2ContactBatchCollator,
    Sabdab2ContactDataset,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from evaluate_framecontact_sabdab2_ensemble_test import (  # noqa: E402
    _evaluate_same_antigen_challenge,
    _independent_se3_prediction_invariance,
)
from train_framecontact import _model_from_configuration  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--validation-challenge", type=Path, required=True)
    parser.add_argument("--validation-challenge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-groups", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError("Refusing to overwrite smoke output")
    if sha256_file(arguments.validation_challenge) != arguments.validation_challenge_sha256:
        raise ValueError("Validation challenge SHA-256 differs")
    panel = json.loads(arguments.validation_challenge.read_text(encoding="utf-8"))
    if (
        panel.get("role") != "validation"
        or panel.get("authority", {}).get("task_labels_read") is not False
    ):
        raise ValueError("GPU smoke requires a label-free validation challenge")
    selected = []
    observed_groups = set()
    for entry in panel["entries"]:
        group = str(entry["antigen_sequence_sha256"])
        if group in observed_groups:
            continue
        selected.append({**entry, "alternates": entry["alternates"][:1]})
        observed_groups.add(group)
        if len(observed_groups) >= arguments.maximum_groups:
            break
    if len(selected) < 2:
        raise ValueError("Validation smoke needs at least two exact-antigen groups")
    small_panel = {
        **panel,
        "entries": selected,
        "focal_instance_count": len(selected),
        "ordered_pair_count": len(selected),
        "exact_antigen_group_count": len(selected),
    }
    base = Sabdab2StructureEmbeddingDataset(
        arguments.structure_cache_root,
        arguments.embedding_root,
        None,
        split_role="validation",
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
    )
    dataset = Sabdab2ContactDataset(base, arguments.contact_cache_root)
    first = dataset[0]
    configuration = {
        "d_model": 128,
        "num_heads": 8,
        "num_self_layers": 2,
        "num_rbf": 16,
        "neighbor_k": 32,
        "neighbor_radius": 16.0,
        "sequence_window": 32,
        "dropout": 0.1,
    }
    torch.manual_seed(20_260_730)
    torch.cuda.manual_seed_all(20_260_730)
    model = _model_from_configuration(
        first, configuration, "framecontact_interleaved"
    ).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = _evaluate_same_antigen_challenge(
        model,
        dataset,
        small_panel,
        torch.device("cuda"),
        use_bf16=True,
        maximum_epitope_jaccard=0.50,
        bootstrap_replicates=100,
        bootstrap_seed=20_260_730,
    )
    invariance = _independent_se3_prediction_invariance(
        model,
        [Sabdab2ContactBatchCollator()([dataset[0]])],
        torch.device("cuda"),
        use_bf16=True,
        seed=20_260_731,
    )
    payload = {
        "format_version": "framecontact_same_antigen_gpu_smoke_v1",
        "claim_boundary": (
            "Randomly initialized model on a deterministic validation-only subset; "
            "validates execution and schema, not performance."
        ),
        "validation_challenge_sha256": arguments.validation_challenge_sha256,
        "split_manifest_sha256": arguments.split_manifest_sha256,
        "exact_antigen_groups_requested": len(selected),
        "scored_ordered_pair_count": result["scored_ordered_pair_count"],
        "primary_ordered_pair_count": result["primary_ordered_pair_count"],
        "ineligible_ordered_pair_count": result["ineligible_ordered_pair_count"],
        "result_schema_keys": sorted(result),
        "independent_se3_prediction_invariance": invariance,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "optimizer_step_performed": False,
        "test_role_loaded": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(json.dumps({**payload, "output": str(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
