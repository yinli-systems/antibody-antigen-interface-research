#!/usr/bin/env python3
"""Run the frozen synthetic one-GPU ABF-PairContact systems canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from antibody_pretraining.stage_a1_objective import StageA1Heads


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(bytes(value.untyped_storage()))
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-objective-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(arguments: argparse.Namespace) -> dict:
    if _sha256(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("PairContact systems-canary protocol SHA differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version")
        != "abf_paircontact_one_gpu_systems_canary_v1"
        or protocol.get("status") != "FROZEN_BEFORE_ONE_GPU_SYSTEMS_CANARY"
        or protocol.get("authorization")
        != {
            "one_gpu_systems_canary": True,
            "production_training": False,
            "test_access": False,
        }
        or protocol.get("systems", {}).get("partition") != "gpu_4090"
        or protocol.get("systems", {}).get("world_size") != 1
        or protocol.get("optimization", {}).get("updates") != 20
        or protocol.get("data")
        != {
            "external_files": False,
            "synthetic_only": True,
            "test_access": False,
        }
    ):
        raise RuntimeError("PairContact systems-canary authority differs")
    return protocol


def main() -> None:
    arguments = _arguments()
    protocol = _load_protocol(arguments)
    objective = Path(__file__).resolve().parents[1] / "src" / "antibody_pretraining" / "stage_a1_objective.py"
    if _sha256(objective) != arguments.expected_objective_sha256:
        raise RuntimeError("PairContact objective SHA differs")
    if arguments.checkpoint.exists() or arguments.result.exists():
        raise FileExistsError("PairContact systems-canary output already exists")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PairContact canary requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if properties.name != "NVIDIA GeForce RTX 4090":
        raise RuntimeError(f"Forbidden GPU model: {properties.name}")

    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    torch.use_deterministic_algorithms(True)
    configuration = protocol["model"]
    heads = StageA1Heads(
        embedding_dimension=configuration["embedding_dimension"],
        projection_dimension=configuration["projection_dimension"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=protocol["optimization"]["learning_rate"],
        weight_decay=protocol["optimization"]["weight_decay"],
    )

    hidden = torch.randn(
        2,
        5,
        configuration["embedding_dimension"],
        device=device,
    )
    chain_ids = torch.tensor(
        [[0, 0, 1, 1, -1], [0, 0, 1, 1, -1]], device=device
    )
    pair_batch = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    pair_left = torch.tensor([0, 0, 1, 0, 0, 1], device=device)
    pair_right = torch.tensor([2, 3, 2, 2, 3, 2], device=device)
    pair_labels = torch.tensor([1, 0, 0, 1, 0, 0], device=device)
    retrieval_allowed = torch.ones((2, 2), dtype=torch.bool, device=device)

    initial_state_sha256 = _state_sha256(heads)
    losses: list[float] = []
    maximum_gradient_norm = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(protocol["optimization"]["updates"]):
        optimizer.zero_grad(set_to_none=True)
        output = heads.loss(
            hidden=hidden,
            chain_ids=chain_ids,
            mlm_loss=torch.zeros((), device=device),
            retrieval_allowed=retrieval_allowed,
            pair_batch_indices=pair_batch,
            pair_left_token_indices=pair_left,
            pair_right_token_indices=pair_right,
            pair_labels=pair_labels,
            mlm_weight=0.0,
            retrieval_weight=0.0,
            contact_weight=1.0,
            contact_ranking_weight=protocol["loss"]["ranking_weight"],
            contact_centering_weight=protocol["loss"]["centering_weight"],
            contact_ranking_margin=protocol["loss"]["ranking_margin"],
        )
        if not bool(torch.isfinite(output.total)):
            raise RuntimeError("PairContact systems-canary loss is nonfinite")
        output.total.backward()
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in heads.parameters()
        ):
            raise RuntimeError("PairContact systems-canary gradient is invalid")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                heads.parameters(), protocol["optimization"]["gradient_clip_norm"]
            )
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError("PairContact systems-canary gradient norm is nonfinite")
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        losses.append(float(output.total.detach()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if losses[-1] >= losses[0]:
        raise RuntimeError("PairContact systems-canary loss did not decrease")

    final_state_sha256 = _state_sha256(heads)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "protocol_sha256": arguments.expected_protocol_sha256,
            "test_access": False,
        },
        arguments.checkpoint,
    )
    checkpoint_sha256 = _sha256(arguments.checkpoint)
    restored = StageA1Heads(
        embedding_dimension=configuration["embedding_dimension"],
        projection_dimension=configuration["projection_dimension"],
    ).to(device)
    payload = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(payload["model"], strict=True)
    restore_state_sha256 = _state_sha256(restored)
    if restore_state_sha256 != final_state_sha256:
        raise RuntimeError("PairContact checkpoint restore differs")

    result = {
        "authorization": protocol["authorization"],
        "checkpoint": {
            "file_sha256": checkpoint_sha256,
            "final_state_sha256": final_state_sha256,
            "restore_matches_save": True,
            "restore_state_sha256": restore_state_sha256,
        },
        "claim_boundary": protocol["claim_boundary"],
        "device": properties.name,
        "format_version": "abf_paircontact_one_gpu_systems_canary_result_v1",
        "loss": {
            "final": losses[-1],
            "first": losses[0],
            "relative_reduction": (losses[0] - losses[-1]) / losses[0],
        },
        "optimization": {
            "elapsed_seconds": elapsed,
            "maximum_gradient_norm_before_clipping": maximum_gradient_norm,
            "updates": len(losses),
        },
        "provenance": {
            "initial_state_sha256": initial_state_sha256,
            "objective_sha256": arguments.expected_objective_sha256,
            "protocol_sha256": arguments.expected_protocol_sha256,
        },
        "status": "PASS_ONE_GPU_GRAPH_OPTIMIZER_AND_CHECKPOINT_GATE",
        "systems": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch": torch.__version__,
        },
        "test_access": False,
    }
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    with arguments.result.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
