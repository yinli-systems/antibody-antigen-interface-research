#!/usr/bin/env python3
"""Run the adapted Boltz-1 Pairformer baseline with the development-only trainer.

This wrapper replaces the process-local model factory and experiment registry;
it never mutates the frozen eight-condition FrameContact registry on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import train_framecontact as trainer  # noqa: E402
from interaction_first.pairformer_framecontact import (  # noqa: E402
    AdaptedBoltzPairformerContactModel,
)


def _adapter_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pairformer-artifact", type=Path, required=True)
    parser.add_argument("--pairformer-receipt", type=Path, required=True)
    parser.add_argument("--boltz-source-root", type=Path, required=True)
    return parser.parse_known_args()


ADAPTER_ARGUMENTS, REMAINING_ARGUMENTS = _adapter_arguments()
sys.argv = [sys.argv[0], *REMAINING_ARGUMENTS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


trainer.EXPERIMENTS = {
    "adapted_boltz1_pairformer": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "official_boltz1_pairformer",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
    }
}


def _model_from_configuration(
    first: Mapping[str, Any],
    configuration: Mapping[str, Any],
    experiment: str,
) -> AdaptedBoltzPairformerContactModel:
    if experiment != "adapted_boltz1_pairformer":
        raise ValueError("Unknown adapted Pairformer baseline")
    receipt = json.loads(
        ADAPTER_ARGUMENTS.pairformer_receipt.read_text(encoding="utf-8")
    )
    trainer.EXTERNAL_BASELINE_AUTHORITY = {
        "baseline_family": "adapted_boltz1_pairformer_contact",
        "artifact_path": str(ADAPTER_ARGUMENTS.pairformer_artifact.resolve()),
        "artifact_sha256": receipt["artifact_sha256"],
        "extraction_receipt_path": str(
            ADAPTER_ARGUMENTS.pairformer_receipt.resolve()
        ),
        "extraction_receipt_sha256": _sha256(
            ADAPTER_ARGUMENTS.pairformer_receipt
        ),
        "boltz_source_root": str(ADAPTER_ARGUMENTS.boltz_source_root.resolve()),
        "boltz_source_revision": receipt["boltz_source_revision"],
        "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
        "state_tensor_count": receipt["state_tensor_count"],
        "native_boltz_pipeline_reproduced": False,
    }
    return AdaptedBoltzPairformerContactModel(
        artifact_path=ADAPTER_ARGUMENTS.pairformer_artifact,
        receipt_path=ADAPTER_ARGUMENTS.pairformer_receipt,
        boltz_source_root=ADAPTER_ARGUMENTS.boltz_source_root,
        antibody_input_dim=int(first["antibody_features"].shape[-1]),
        antigen_input_dim=int(first["antigen_features"].shape[-1]),
        d_model=int(configuration["d_model"]),
        num_heads=int(configuration["num_heads"]),
        num_rbf=int(configuration["num_rbf"]),
        neighbor_k=int(configuration["neighbor_k"]),
        neighbor_radius=float(configuration["neighbor_radius"]),
        sequence_window=int(configuration["sequence_window"]),
        dropout=float(configuration["dropout"]),
    )


trainer._model_from_configuration = _model_from_configuration
_ORIGINAL_EVALUATE = trainer._evaluate


def _evaluate_without_internal_attention(*args: Any, **kwargs: Any) -> Any:
    # Official Boltz Pairformer does not expose the FrameContact-specific
    # head diagnostics contract.  Keep task metrics identical and record this
    # architectural non-comparability instead of fabricating diagnostics.
    kwargs["collect_attention"] = False
    return _ORIGINAL_EVALUATE(*args, **kwargs)


trainer._evaluate = _evaluate_without_internal_attention


if __name__ == "__main__":
    trainer.main()
