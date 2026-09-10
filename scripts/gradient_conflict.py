#!/usr/bin/env python3
"""Measure whether the node and pair objectives pull the shared trunk apart.

    python3 scripts/gradient_conflict.py --checkpoint best.pt --prepare-dir ... \
        --contact-dir ... --embedding-dir ... --split-manifest ... --json out.json

Removing pair supervision leaves the node metrics unchanged or slightly better,
at ten seeds out of ten. "The two tasks compete for capacity" is one explanation
of that and not the only one: gradient conflict, a compromised shared
representation, loss-scale imbalance, plain optimisation interference, and pair
supervision learning fine-grained relations irrelevant to node classification
would all produce the same observation.

This separates the first from the rest. For each instance it computes the
gradient of the node objective and the gradient of the pair objective with
respect to the **shared** parameters -- everything that is not a node head and
not the pair decoder -- and reports the cosine between them.

    cos < 0   the objectives pull the trunk in opposing directions; the negative
              transfer has a mechanism and it is gradient conflict
    cos ~ 0   the objectives are close to orthogonal; they are not fighting, so
              a capacity or loss-scale account survives and conflict does not
    cos > 0   they agree, and the observed negative transfer needs a different
              explanation entirely

Gradient magnitudes are reported alongside, because a loss-scale imbalance shows
up there rather than in the angle: two objectives can agree in direction while
one contributes a hundred times more gradient than the other.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

# Verified against the checkpoint's own parameter names. Getting these wrong is
# not a cosmetic error: a pair-branch parameter counted as shared would put the
# pair gradient partly against itself and manufacture a positive cosine.
NODE_HEAD_PREFIXES = ("epitope_head.", "paratope_head.")
PAIR_HEAD_PREFIXES = ("decoder.", "pair_decoder.")


def _load_training_module():
    spec = importlib.util.spec_from_file_location(
        "sabdab2_for_gradient_conflict",
        REPOSITORY_ROOT / "scripts" / "sabdab2_interaction_first.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(name: str) -> str:
    """Which part of the model a parameter belongs to."""
    if name.startswith(PAIR_HEAD_PREFIXES):
        return "pair_head"
    if name.startswith(NODE_HEAD_PREFIXES):
        return "node_head"
    return "shared"


def flat_gradient(loss, parameters):
    """Gradient of one loss with respect to a parameter list, flattened."""
    import torch

    grads = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True, create_graph=False
    )
    pieces = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, grads)
    ]
    return torch.cat(pieces)


def cosine(left, right) -> float:
    import torch

    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--instances", type=int, default=60)
    parser.add_argument("--representation-manifest", type=Path, default=None)
    parser.add_argument("--esm-if1-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import pandas as pd
    import torch

    from interaction_first.contacts import load_contact_map
    from interaction_first.losses import InteractionLossWeights, interaction_first_loss
    from interaction_first.model import PartnerSpecificInteractionModel
    from interaction_first.runtime import EmbeddingStore, safe_torch_load

    # The training script is imported for the split loader and the tensor
    # assembly, which live only there. Everything the library already exports is
    # taken from the library: the script does not re-export those names, and
    # reaching for them through it fails at the first call.
    training = _load_training_module()

    frame = pd.read_parquet(arguments.prepare_dir / "instances.parquet").set_index(
        "instance_id", verify_integrity=True
    )
    assignments = training.load_split_assignments(arguments.split_manifest.expanduser().resolve())
    frame = frame.loc[[i for i in frame.index if assignments.get(str(i)) == arguments.split]]
    frame = frame.sort_index().iloc[: arguments.instances]

    device = torch.device(arguments.device)
    checkpoint = safe_torch_load(arguments.checkpoint, map_location=device)
    config = checkpoint["config"]

    store = EmbeddingStore(arguments.embedding_dir.expanduser().resolve(), 2.0)
    if arguments.esm_if1_root is not None:
        from interaction_first.composite_embeddings import (
            CompositeEmbeddingStore,
            build_sources,
        )

        store = CompositeEmbeddingStore(
            store,
            build_sources(
                arguments.representation_manifest.expanduser().resolve(),
                esm_if1_root=arguments.esm_if1_root.expanduser().resolve(),
            ),
        )

    model = PartnerSpecificInteractionModel(
        antigen_input_dim=int(config.get("antigen_input_dim", config["input_dim"])),
        antibody_input_dim=int(config.get("antibody_input_dim", config["input_dim"])),
        d_model=int(config["d_model"]),
        pair_dim=int(config["pair_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        marginal_pooling=str(config["marginal_pooling"]),
        marginal_top_k=int(config["marginal_top_k"]),
        marginal_temperature=float(config["marginal_temperature"]),
        score_top_k=int(config["score_top_k"]),
        score_pooling=str(config["score_pooling"]),
        score_temperature=float(config["score_temperature"]),
        score_top_fraction=float(config["score_top_fraction"]),
        center_partner_score=bool(config["center_partner_score"]),
        cross_attention_gate_initial_logit=float(config["cross_attention_gate_initial_logit"]),
        pair_decoder=str(config.get("pair_decoder", "bilinear")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.train()

    groups: dict[str, list] = {"shared": [], "node_head": [], "pair_head": []}
    names: dict[str, list[str]] = {"shared": [], "node_head": [], "pair_head": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        where = classify(name)
        groups[where].append(parameter)
        names[where].append(name)

    node_weights = InteractionLossWeights(
        edge=0.0, partner_rank=0.0,
        epitope_bce=float(config["loss_weights"]["epitope_bce"]),
        paratope_bce=float(config["loss_weights"]["paratope_bce"]),
        epitope_dice=float(config["loss_weights"]["epitope_dice"]),
        paratope_dice=float(config["loss_weights"]["paratope_dice"]),
    )
    pair_weights = InteractionLossWeights(
        edge=1.0, partner_rank=0.0,
        epitope_bce=0.0, paratope_bce=0.0, epitope_dice=0.0, paratope_dice=0.0,
    )

    per_instance = []
    for instance_id in frame.index:
        row = frame.loc[instance_id]
        contact = load_contact_map(
            arguments.contact_dir / f"{instance_id}.npz", str(instance_id)
        )
        tensors = training._example_tensors(contact, row, store, device)
        antigen, antibody, pair_labels, pair_mask, epitope_labels, paratope_labels = tensors[:6]
        antigen_mask, antibody_mask = tensors[6], tensors[7]
        outputs = model(antigen, antibody, antigen_mask, antibody_mask)

        node_loss, _ = interaction_first_loss(
            outputs, pair_labels, epitope_labels, paratope_labels,
            antigen_mask, antibody_mask, weights=node_weights,
        )
        pair_loss, _ = interaction_first_loss(
            outputs, pair_labels, epitope_labels, paratope_labels,
            antigen_mask, antibody_mask, weights=pair_weights,
        )

        record = {"instance_id": str(instance_id)}
        for where in ("shared", "node_head"):
            if not groups[where]:
                continue
            node_gradient = flat_gradient(node_loss, groups[where])
            pair_gradient = flat_gradient(pair_loss, groups[where])
            record[f"cosine_{where}"] = cosine(node_gradient, pair_gradient)
            record[f"node_norm_{where}"] = float(node_gradient.norm())
            record[f"pair_norm_{where}"] = float(pair_gradient.norm())
        per_instance.append(record)
        model.zero_grad(set_to_none=True)

    if not groups["pair_head"] or not groups["node_head"]:
        print(
            "the pair or node head matched no parameters; the classification is "
            f"wrong for this checkpoint. Counts: "
            f"{ {k: len(v) for k, v in groups.items()} }",
            file=sys.stderr,
        )
        return 2

    report = {
        "format_version": "gradient_conflict_v1",
        "checkpoint": str(arguments.checkpoint),
        "split": arguments.split,
        "instances": len(per_instance),
        "parameter_counts": {k: sum(p.numel() for p in v) for k, v in groups.items()},
        "parameter_names": {k: v[:6] for k, v in names.items()},
        "per_instance": per_instance,
        "claim_boundary": (
            "Gradients at one trained checkpoint on one split. A cosine near zero "
            "rules gradient conflict out as the mechanism; it does not identify "
            "which of the remaining explanations is right."
        ),
    }
    for where in ("shared", "node_head"):
        values = [r[f"cosine_{where}"] for r in per_instance if f"cosine_{where}" in r]
        values = [v for v in values if v == v]
        if not values:
            continue
        report[f"cosine_{where}_mean"] = statistics.mean(values)
        report[f"cosine_{where}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        report[f"cosine_{where}_negative_fraction"] = sum(1 for v in values if v < 0) / len(values)
        node_norms = [r[f"node_norm_{where}"] for r in per_instance]
        pair_norms = [r[f"pair_norm_{where}"] for r in per_instance]
        report[f"node_norm_{where}_mean"] = statistics.mean(node_norms)
        report[f"pair_norm_{where}_mean"] = statistics.mean(pair_norms)
        report[f"pair_over_node_norm_{where}"] = (
            statistics.mean(pair_norms) / statistics.mean(node_norms)
            if statistics.mean(node_norms) else None
        )
        print(f"{where}: {len(values)} instances, {report['parameter_counts'][where]} parameters")
        print(f"  cos(node, pair)      mean {report[f'cosine_{where}_mean']:+.4f}"
              f"  sd {report[f'cosine_{where}_sd']:.4f}"
              f"  negative in {100 * report[f'cosine_{where}_negative_fraction']:.0f}%")
        print(f"  gradient norms       node {report[f'node_norm_{where}_mean']:.4g}"
              f"  pair {report[f'pair_norm_{where}_mean']:.4g}"
              f"  ratio {report[f'pair_over_node_norm_{where}']:.3g}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
