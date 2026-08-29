#!/usr/bin/env python3
"""One real development-complex GPU step for distogram + triangle training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from interaction_first.framecontact import (  # noqa: E402
    FrameContactModel,
    contact_logit_from_distogram,
)
from interaction_first.framecontact_objective import (  # noqa: E402
    FrameContactLossWeights,
    FrameContactObjective,
)
from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: E402
from interaction_first.sabdab2_pair_geometry import (  # noqa: E402
    Sabdab2PairGeometryBatchCollator,
    Sabdab2PairGeometryDataset,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureEmbeddingDataset,
    authenticate_file,
    canonical_json_bytes,
)


class _SingleInstanceView:
    def __init__(self, base: object, index: int) -> None:
        self.base = base
        self.cache_root = base.cache_root
        self.records = (base.records[index],)
        self.index = int(index)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        if index != 0:
            raise IndexError(index)
        return self.base[self.index]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--pair-geometry-cache-root", type=Path, required=True)
    parser.add_argument("--pair-geometry-manifest-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-self-layers", type=int, default=2)
    return parser.parse_args()


def _role(split_path: Path, identifier: str) -> str:
    with split_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["instance_id"] == identifier]
    if len(rows) != 1 or rows[0].get("role") not in {"train", "validation", "test"}:
        raise ValueError("instance must occur exactly once in the authenticated split")
    role = rows[0]["role"]
    if role == "test":
        raise ValueError("GPU smoke refuses to access a frozen-test instance")
    return role


def _atomic_receipt(payload: dict[str, object], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite smoke receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    arguments = _arguments()
    authorities = {
        "structure_manifest_sha256": authenticate_file(
            arguments.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json",
            arguments.structure_manifest_sha256,
            "structure manifest",
        ),
        "contact_manifest_sha256": authenticate_file(
            arguments.contact_cache_root / "CONTACT_CACHE_MANIFEST.json",
            arguments.contact_manifest_sha256,
            "contact manifest",
        ),
        "pair_geometry_manifest_sha256": authenticate_file(
            arguments.pair_geometry_cache_root / "PAIR_GEOMETRY_CACHE_MANIFEST.json",
            arguments.pair_geometry_manifest_sha256,
            "pair geometry manifest",
        ),
        "source_manifest_sha256": authenticate_file(
            arguments.source_manifest,
            arguments.source_manifest_sha256,
            "training source manifest",
        ),
        "embedding_manifest_sha256": authenticate_file(
            arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json",
            arguments.embedding_manifest_sha256,
            "embedding manifest",
        ),
        "split_manifest_sha256": authenticate_file(
            arguments.split_manifest,
            arguments.split_manifest_sha256,
            "split manifest",
        ),
    }
    role = _role(arguments.split_manifest, arguments.instance_id)
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    if arguments.precision == "bf16" and arguments.device != "cuda":
        raise ValueError("bf16 smoke requires CUDA")
    if arguments.d_model % arguments.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    base = Sabdab2StructureEmbeddingDataset(
        arguments.structure_cache_root,
        arguments.embedding_root,
        None,
        split_role=role,
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
    )
    matches = [
        index
        for index, record in enumerate(base.records)
        if record["instance_id"] == arguments.instance_id
    ]
    if len(matches) != 1:
        raise ValueError("authenticated development split lacks the smoke instance")
    contact = Sabdab2ContactDataset(
        _SingleInstanceView(base, matches[0]), arguments.contact_cache_root
    )
    geometry = Sabdab2PairGeometryDataset(
        contact,
        arguments.pair_geometry_cache_root,
        manifest_sha256=arguments.pair_geometry_manifest_sha256,
    )
    batch = Sabdab2PairGeometryBatchCollator()([geometry[0]])
    derived_labels = (batch["pair_distances"] < 4.5) & batch["pair_distance_mask"]
    if not torch.equal(derived_labels, batch["pair_labels"].bool()):
        raise RuntimeError("real batch distances do not exactly reproduce pair labels")

    device = torch.device(arguments.device)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = FrameContactModel(
        antibody_input_dim=int(batch["antibody_features"].shape[-1]),
        antigen_input_dim=int(batch["antigen_features"].shape[-1]),
        d_model=arguments.d_model,
        num_heads=arguments.num_heads,
        num_self_layers=arguments.num_self_layers,
        geometry_feature_mode="full_pose",
        geometry_bias_mode="pair_mlp",
        neighbor_k=32,
        neighbor_radius=16.0,
        sequence_window=32,
        dropout=0.1,
        fusion_mode="late",
        pair_output_mode="distogram",
        residue_output_mode="gated_hybrid",
        residue_unary_source="intra_chain",
        contact_marginalization="noisy_or",
    ).to(device)
    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            contact=1.0,
            paratope=0.25,
            epitope=0.25,
            specificity=0.0,
            distogram=1.0,
            triangle=0.1,
        ),
        triangle_neighbor_radius=12.0,
        triangle_neighbor_k=8,
        triangle_slack=0.5,
        antigen_patch_top_seeds=1,
        antigen_patch_radius=12.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if arguments.precision == "bf16"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with autocast:
        output = model(batch)
        losses = objective(output, batch)
    projected = contact_logit_from_distogram(
        output["distogram_logits"],
        contact_bin_count=model.contact_head.contact_bin_count,
    )
    projection_error = float(
        (projected.float() - output["contact_logits"].float()).abs().max().detach()
    )
    if projection_error > 1e-5:
        raise RuntimeError("distogram-to-contact projection is inconsistent")
    losses["loss"].float().backward()
    parameters_with_gradient = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    if not parameters_with_gradient or any(
        not bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters_with_gradient
    ):
        raise RuntimeError("model gradients are empty or non-finite")
    gradient_norm = math.sqrt(
        sum(float(parameter.grad.float().pow(2).sum()) for parameter in parameters_with_gradient)
    )
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    receipt = {
        "format_version": "framecontact_real_distogram_triangle_smoke_v1",
        "status": "pass",
        "instance_id": arguments.instance_id,
        "split_role": role,
        "frozen_test_accessed": False,
        "authorities": authorities,
        "model": {
            "geometry_feature_mode": "full_pose",
            "fusion_mode": "late",
            "pair_output_mode": "distogram",
            "residue_output_mode": "gated_hybrid",
            "residue_unary_source": "intra_chain",
            "contact_marginalization": "noisy_or",
            "antigen_triangle_patch_seed_source": (
                "isolated-monomer intra-chain epitope unary logits"
            ),
            "d_model": arguments.d_model,
            "num_heads": arguments.num_heads,
            "num_self_layers": arguments.num_self_layers,
        },
        "batch": {
            "antibody_residues": int(batch["antibody_mask"].sum()),
            "antigen_residues": int(batch["antigen_mask"].sum()),
            "candidate_pairs": int(batch["pair_candidate_mask"].sum()),
            "positive_contacts": int(batch["pair_labels"].sum()),
            "distance_contact_equivalence": True,
        },
        "optimization": {
            "losses": {
                key: float(value.detach().float())
                for key, value in losses.items()
                if key.endswith("loss") and torch.is_tensor(value)
            },
            "finite_gradient_parameter_tensors": len(parameters_with_gradient),
            "global_gradient_norm": gradient_norm,
            "optimizer_step_completed": True,
            "distogram_contact_projection_max_abs_error": projection_error,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "precision": arguments.precision,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
        "claim_boundary": (
            "One authenticated development-complex forward, backward, and optimizer "
            "step only; no validation selection, frozen-test result, benchmark, speedup, "
            "or generalization claim."
        ),
    }
    _atomic_receipt(receipt, arguments.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
