#!/usr/bin/env python3
"""Audit learned topology gates in one selected development checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.topology_routed_attention import (  # noqa: E402
    TopologyRoutedMultiScaleHeadBias,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _entropy(values: np.ndarray) -> float:
    probabilities = values / values.sum() if values.sum() > 0 else values
    if not probabilities.any():
        return 0.0
    selected = probabilities[probabilities > 0]
    return float(-(selected * np.log(selected)).sum())


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite gate audit: {arguments.output}")
    protocol_sha256 = authenticate_file(
        arguments.protocol, arguments.protocol_sha256, "topology protocol"
    )
    receipt_sha256 = authenticate_file(
        arguments.protocol_receipt,
        arguments.protocol_receipt_sha256,
        "topology receipt",
    )
    result_path = arguments.run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = arguments.run_root / "FRAMECONTACT_SELECTED.pt"
    result_sha256 = authenticate_file(
        result_path, arguments.result_sha256, "development result"
    )
    checkpoint_sha256 = authenticate_file(
        checkpoint_path, arguments.checkpoint_sha256, "selected checkpoint"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("experiment") != "topology_routed_multiscale"
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or result.get("authority", {}).get("protocol_file_sha256")
        != protocol_sha256
        or result.get("authority", {}).get("protocol_receipt_sha256")
        != receipt_sha256
    ):
        raise ValueError("Development result authority is not eligible")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Selected checkpoint lacks a model_state_dict")
    suffix = ".geometry_bias.gate_unconstrained"
    gates = {
        key: torch.tanh(value.detach().float()).numpy()
        for key, value in state.items()
        if key.endswith(suffix)
    }
    if len(gates) != 4:
        raise ValueError(f"Expected four routed layers, found {len(gates)}")
    expected_shape = (
        len(TopologyRoutedMultiScaleHeadBias.RELATION_NAMES),
        len(TopologyRoutedMultiScaleHeadBias.COMPONENT_NAMES),
        8,
    )
    if any(values.shape != expected_shape for values in gates.values()):
        raise ValueError("A topology gate tensor has the wrong shape")

    layers: list[dict[str, Any]] = []
    stacked = []
    for key in sorted(gates):
        values = gates[key]
        stacked.append(values)
        absolute = np.abs(values)
        layers.append(
            {
                "state_key": key,
                "gate_shape": list(values.shape),
                "gate_abs_mean": float(absolute.mean()),
                "gate_abs_max": float(absolute.max()),
                "nonzero_fraction_gt_1e_8": float((absolute > 1e-8).mean()),
                "per_relation_component_head": {
                    relation_name: {
                        component_name: [
                            float(value)
                            for value in values[
                                relation_index, component_index
                            ].tolist()
                        ]
                        for component_index, component_name in enumerate(
                            TopologyRoutedMultiScaleHeadBias.COMPONENT_NAMES
                        )
                    }
                    for relation_index, relation_name in enumerate(
                        TopologyRoutedMultiScaleHeadBias.RELATION_NAMES
                    )
                },
            }
        )
    all_values = np.stack(stacked)
    absolute = np.abs(all_values)
    per_relation_head = absolute.mean(axis=(0, 2))
    per_component_head = absolute.mean(axis=(0, 1))
    head_rows = []
    for head_index in range(expected_shape[-1]):
        relation_values = per_relation_head[:, head_index]
        component_values = per_component_head[:, head_index]
        head_rows.append(
            {
                "head": head_index,
                "dominant_relation": (
                    TopologyRoutedMultiScaleHeadBias.RELATION_NAMES[
                        int(relation_values.argmax())
                    ]
                ),
                "dominant_component": (
                    TopologyRoutedMultiScaleHeadBias.COMPONENT_NAMES[
                        int(component_values.argmax())
                    ]
                ),
                "relation_abs_mean": {
                    name: float(relation_values[index])
                    for index, name in enumerate(
                        TopologyRoutedMultiScaleHeadBias.RELATION_NAMES
                    )
                },
                "component_abs_mean": {
                    name: float(component_values[index])
                    for index, name in enumerate(
                        TopologyRoutedMultiScaleHeadBias.COMPONENT_NAMES
                    )
                },
                "normalized_relation_entropy": (
                    _entropy(relation_values) / np.log(len(relation_values))
                ),
                "normalized_component_entropy": (
                    _entropy(component_values) / np.log(len(component_values))
                ),
            }
        )
    validation = result["selected_checkpoint_validation"]
    payload = {
        "format_version": "topology_routing_checkpoint_audit_v1",
        "authority": {
            "protocol_file_sha256": protocol_sha256,
            "protocol_receipt_sha256": receipt_sha256,
            "development_result_sha256": result_sha256,
            "selected_checkpoint_sha256": checkpoint_sha256,
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "experiment": result["experiment"],
        "seed": result["configuration"]["seed"],
        "selected_epoch": result["selection"]["best_epoch"],
        "validation_summary": {
            "selection_metric": validation["selection_metric"],
            "contact_macro_auprc": validation["contact"][
                "mean_per_complex_auprc"
            ],
            "contact_pooled_auprc": validation["contact"]["auprc"],
            "epitope_macro_auprc": validation["epitope"][
                "mean_per_complex_auprc"
            ],
            "epitope_pooled_auprc": validation["epitope"]["auprc"],
            "paratope_macro_auprc": validation["paratope"][
                "mean_per_complex_auprc"
            ],
        },
        "aggregate": {
            "gate_tensor_count": len(gates),
            "gate_scalar_count": int(absolute.size),
            "gate_abs_mean": float(absolute.mean()),
            "gate_abs_max": float(absolute.max()),
            "nonzero_fraction_gt_1e_8": float((absolute > 1e-8).mean()),
        },
        "head_routing": head_rows,
        "layers": layers,
        "claim_boundary": (
            "Descriptive audit of one validation-selected development checkpoint. "
            "It is not an attribution proof, frozen-test result, external result, "
            "generalization result, or benchmark claim."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "gate_abs_mean": payload["aggregate"]["gate_abs_mean"],
                "gate_abs_max": payload["aggregate"]["gate_abs_max"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
