"""Train-only PAIRFormer overfit gate for authenticated sparse contacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .development_authority import canonical_json_bytes
from .development_contact_dataset import (
    DevelopmentContactDataset,
    DevelopmentContactExample,
)
from .frontier_checkpoint import (
    create_checkpoint_generation,
    restore_checkpoint_generation,
    sha256_file,
)
from .metrics import average_precision, contact_map_metrics
from .pair_geometry import (
    ProspectivePairGeometryModel,
    double_centered_partner_score,
)
from .pair_geometry_losses import masked_contact_bce_with_logits
from .pair_geometry_schema import PairGeometrySchema


REPORT_FORMAT = "pairscale_train_only_overfit_report_v1"
METRICS_FORMAT = "pairscale_train_only_epoch_metrics_v1"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for PAIRScale metrics") from error
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for PAIRScale training") from error
    return torch


def _finite_positive(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class PairscaleOverfitConfig:
    """Locked hyperparameters for one non-confirmatory overfit run."""

    seed: int = 17
    epochs: int = 200
    learning_rate: float = 0.001
    minimum_learning_rate_fraction: float = 0.05
    weight_decay: float = 0.0
    positive_weight: float = 1024.0
    contact_ranking_weight: float = 1.0
    hard_negative_ratio: int = 8
    contact_ranking_temperature: float = 1.0
    partner_ranking_weight: float = 0.25
    partner_ranking_margin: float = 0.10
    partner_ranking_temperature: float = 1.0
    gradient_clip_norm: float = 5.0
    d_model: int = 192
    pair_dim: int = 96
    num_heads: int = 6
    num_fusion_layers: int = 2
    ffn_dim: int = 384
    pair_chunk_size: int = 64
    dropout: float = 0.0
    eval_every_epochs: int = 10
    checkpoint_every_epochs: int = 10
    precision: str = "bf16"
    gate_auprc: float = 0.95
    gate_precision_at_true_count: float = 0.90
    gate_partner_win_rate: float = 0.95

    def validate(self) -> None:
        """Reject ambiguous, unsafe, or scientifically inconsistent settings."""
        for name in (
            "seed",
            "epochs",
            "d_model",
            "pair_dim",
            "num_heads",
            "num_fusion_layers",
            "ffn_dim",
            "pair_chunk_size",
            "eval_every_epochs",
            "checkpoint_every_epochs",
            "hard_negative_ratio",
        ):
            value = getattr(self, name)
            if name == "seed":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("seed must be a nonnegative integer")
            else:
                _positive_integer(value, name)
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        for name in (
            "learning_rate",
            "minimum_learning_rate_fraction",
            "positive_weight",
            "gradient_clip_norm",
            "contact_ranking_temperature",
            "partner_ranking_margin",
            "partner_ranking_temperature",
        ):
            _finite_positive(getattr(self, name), name)
        for name in (
            "weight_decay",
            "contact_ranking_weight",
            "partner_ranking_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in (
            "gate_auprc",
            "gate_precision_at_true_count",
            "gate_partner_win_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def contact_only_geometry_schema() -> PairGeometrySchema:
    """Return explicit decoder semantics for the binary contact-only gate."""
    return PairGeometrySchema(
        distance_bin_edges_angstrom=(2.0, 4.5, 8.0, 16.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=("placeholder_orientation",),
        orientation_axis_is_cyclic=(False,),
        orientation_bin_edges_radians=((0.0,),),
        label_format_version="pairscale_contact_only_targets_v1",
        orientation_schema_version="pairscale_orientation_not_executed_v1",
        primary_contact_definition=(
            "minimum_observed_residue_non_hydrogen_atom_distance_le_4.5_angstrom"
        ),
        distance_definition="distance_bins_are_decoder_latents_without_targets",
        orientation_definition="orientation_head_not_executed",
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def build_contact_only_model(
    dataset: DevelopmentContactDataset,
    config: PairscaleOverfitConfig,
) -> ProspectivePairGeometryModel:
    """Construct a PAIRFormer whose contact probability is distance-mass based."""
    config.validate()
    return ProspectivePairGeometryModel(
        antibody_input_dim=dataset.antibody_input_dim,
        antigen_input_dim=dataset.antigen_input_dim,
        d_model=config.d_model,
        pair_dim=config.pair_dim,
        num_heads=config.num_heads,
        num_fusion_layers=config.num_fusion_layers,
        ffn_dim=config.ffn_dim,
        geometry_schema=contact_only_geometry_schema(),
        pair_chunk_size=config.pair_chunk_size,
        dropout=config.dropout,
        centering_iterations=32,
    )


def configure_determinism(seed: int) -> None:
    """Lock all in-process generators and deterministic CUDA behavior."""
    torch = _require_torch()
    random.seed(seed)
    _require_numpy().random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda,
        "matmul",
    ):
        torch.backends.cuda.matmul.allow_tf32 = False


def deterministic_epoch_order(count: int, seed: int, epoch: int) -> tuple[int, ...]:
    """Return a stateless order so epoch-boundary resume cannot alter sampling."""
    _positive_integer(count, "count")
    _positive_integer(epoch, "epoch")
    generator = random.Random(f"pairscale-overfit-v1:{seed}:{epoch}")
    order = list(range(count))
    generator.shuffle(order)
    return tuple(order)


def _device_example(example: DevelopmentContactExample, device: Any) -> tuple[Any, ...]:
    torch = _require_torch()
    antibody = example.antibody_embeddings.to(device=device, non_blocking=True).unsqueeze(0)
    antigen = example.antigen_embeddings.to(device=device, non_blocking=True).unsqueeze(0)
    targets = example.contact_targets.to(device=device, non_blocking=True).unsqueeze(0)
    pair_mask = example.contact_mask.to(device=device, non_blocking=True).unsqueeze(0)
    antibody_mask = torch.ones(
        (1, int(antibody.shape[1])),
        dtype=torch.bool,
        device=device,
    )
    antigen_mask = torch.ones(
        (1, int(antigen.shape[1])),
        dtype=torch.bool,
        device=device,
    )
    return antibody, antigen, antibody_mask, antigen_mask, targets, pair_mask


def _forward_contact(model: Any, example: DevelopmentContactExample, device: Any) -> tuple[Any, Any]:
    antibody, antigen, antibody_mask, antigen_mask, targets, pair_mask = _device_example(
        example,
        device,
    )
    outputs = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        pair_mask,
        return_pair_representations=False,
        enabled_heads={"contact", "distance"},
    )
    return outputs, targets


def _forward_cross_pair(
    model: Any,
    antibody_example: DevelopmentContactExample,
    antigen_example: DevelopmentContactExample,
    device: Any,
) -> Mapping[str, Any]:
    """Score an unknown cross-pair without creating a binary contact target."""
    torch = _require_torch()
    antibody = antibody_example.antibody_embeddings.to(device).unsqueeze(0)
    antigen = antigen_example.antigen_embeddings.to(device).unsqueeze(0)
    antibody_mask = torch.ones(
        (1, int(antibody.shape[1])),
        dtype=torch.bool,
        device=device,
    )
    antigen_mask = torch.ones(
        (1, int(antigen.shape[1])),
        dtype=torch.bool,
        device=device,
    )
    pair_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    return model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        pair_mask,
        return_pair_representations=False,
        enabled_heads={"contact", "distance"},
    )


def hard_negative_contact_ranking_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    negative_ratio: int,
    temperature: float,
) -> Any:
    """Rank every positive above the current hardest within-complex negatives."""
    torch = _require_torch()
    _positive_integer(negative_ratio, "negative_ratio")
    temperature = _finite_positive(temperature, "temperature")
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, antibody, antigen]")
    if not torch.is_tensor(targets) or tuple(targets.shape) != tuple(logits.shape):
        raise ValueError("targets must align with logits")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(logits.shape):
        raise ValueError("mask must be boolean and align with logits")
    losses: list[Any] = []
    for batch_index in range(int(logits.shape[0])):
        selected_logits = logits[batch_index][mask[batch_index]].float()
        selected_targets = targets[batch_index][mask[batch_index]]
        if not bool(torch.isfinite(selected_logits).all()):
            raise ValueError("Candidate logits must be finite")
        if not bool(((selected_targets == 0) | (selected_targets == 1)).all()):
            raise ValueError("Candidate targets must be binary")
        positive_logits = selected_logits[selected_targets == 1]
        negative_logits = selected_logits[selected_targets == 0]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            raise ValueError("Ranking requires positive and negative candidates")
        hard_count = min(
            int(negative_logits.numel()),
            int(positive_logits.numel()) * negative_ratio,
        )
        hard_negatives = torch.topk(
            negative_logits,
            k=hard_count,
            sorted=False,
        ).values
        differences = (
            hard_negatives.unsqueeze(0) - positive_logits.unsqueeze(1)
        ) / temperature
        losses.append(torch.nn.functional.softplus(differences).mean() * temperature)
    return torch.stack(losses).mean()


def matched_unknown_partner_ranking_loss(
    matched_outputs: Mapping[str, Any],
    decoy_outputs: Mapping[str, Any],
    *,
    positive_pair_count: int,
    margin: float,
    temperature: float,
) -> Any:
    """Rank a cognate pair above one unknown cross-pair without binary labeling."""
    torch = _require_torch()
    margin = _finite_positive(margin, "margin")
    temperature = _finite_positive(temperature, "temperature")
    matched_logits = matched_outputs["contact_logits"].float()
    decoy_logits = decoy_outputs["contact_logits"].float()
    matched_top = _top_k_mean(matched_logits[0], positive_pair_count)
    decoy_top = _top_k_mean(decoy_logits[0], positive_pair_count)
    _, matched_centered = double_centered_partner_score(
        matched_logits,
        matched_outputs["candidate_pair_mask"],
        centering_iterations=32,
    )
    _, decoy_centered = double_centered_partner_score(
        decoy_logits,
        decoy_outputs["candidate_pair_mask"],
        centering_iterations=32,
    )
    top_loss = torch.nn.functional.softplus(
        (margin - matched_top + decoy_top) / temperature
    ) * temperature
    centered_loss = torch.nn.functional.softplus(
        (margin - matched_centered[0] + decoy_centered[0]) / temperature
    ) * temperature
    return 0.5 * (top_loss + centered_loss)


def train_one_epoch(
    *,
    model: Any,
    optimizer: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
    config: PairscaleOverfitConfig,
    epoch: int,
) -> tuple[Mapping[str, float], int]:
    """Train once over every authenticated complex with no pair subsampling."""
    torch = _require_torch()
    model.train()
    loss_sums = {
        "total": 0.0,
        "contact_bce": 0.0,
        "contact_ranking": 0.0,
        "partner_ranking": 0.0,
    }
    step_count = 0
    use_bf16 = config.precision == "bf16" and device.type == "cuda"
    decoy_indices = _select_cyclic_decoys(examples)
    for index in deterministic_epoch_order(len(examples), config.seed, epoch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            enabled=use_bf16,
        ):
            outputs, targets = _forward_contact(model, examples[index], device)
            contact_bce = masked_contact_bce_with_logits(
                outputs["contact_logits"],
                targets,
                outputs["candidate_pair_mask"],
                positive_weight=config.positive_weight,
            )
            contact_ranking = hard_negative_contact_ranking_loss(
                outputs["contact_logits"],
                targets,
                outputs["candidate_pair_mask"],
                negative_ratio=config.hard_negative_ratio,
                temperature=config.contact_ranking_temperature,
            )
            if config.partner_ranking_weight > 0.0:
                decoy_outputs = _forward_cross_pair(
                    model,
                    examples[index],
                    examples[decoy_indices[index]],
                    device,
                )
                partner_ranking = matched_unknown_partner_ranking_loss(
                    outputs,
                    decoy_outputs,
                    positive_pair_count=examples[index].positive_pair_count,
                    margin=config.partner_ranking_margin,
                    temperature=config.partner_ranking_temperature,
                )
            else:
                partner_ranking = contact_bce.new_zeros(())
            loss = (
                contact_bce
                + config.contact_ranking_weight * contact_ranking
                + config.partner_ranking_weight * partner_ranking
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, example {index}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("Non-finite gradient norm")
        optimizer.step()
        for name, value in (
            ("total", loss),
            ("contact_bce", contact_bce),
            ("contact_ranking", contact_ranking),
            ("partner_ranking", partner_ranking),
        ):
            loss_sums[name] += float(value.detach().cpu())
        step_count += 1
    return {
        name: value / step_count for name, value in loss_sums.items()
    }, step_count


def _top_k_mean(values: Any, count: int) -> Any:
    torch = _require_torch()
    flattened = values.reshape(-1)
    selected = min(_positive_integer(count, "top-k count"), int(flattened.numel()))
    return torch.topk(flattened, k=selected, sorted=False).values.float().mean()


def _model_contact_marginal_outer(probabilities: Any) -> Any:
    """Build a separable baseline from the model's own pair marginals."""
    np = _require_numpy()
    matrix = np.asarray(probabilities, dtype=np.float64)
    antibody_marginal = matrix.max(axis=1)
    antigen_marginal = matrix.max(axis=0)
    return np.multiply.outer(antibody_marginal, antigen_marginal)


def _label_marginal_outer(labels: Any) -> Any:
    """Build an explicitly label-derived oracle, never a deployable baseline."""
    np = _require_numpy()
    matrix = np.asarray(labels, dtype=np.uint8)
    return np.multiply.outer(matrix.max(axis=1), matrix.max(axis=0)).astype(np.float64)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def _select_cyclic_decoys(
    examples: Sequence[DevelopmentContactExample],
) -> tuple[int, ...]:
    """Choose one deterministic non-cognate antigen for every complex."""
    count = len(examples)
    if count < 2:
        raise ValueError("Partner controls require at least two complexes")
    result: list[int] = []
    for index, example in enumerate(examples):
        for offset in range(1, count):
            candidate = (index + offset) % count
            if examples[candidate].antigen_id != example.antigen_id:
                result.append(candidate)
                break
        else:
            raise ValueError("No non-cognate antigen is available")
    return tuple(result)


def evaluate_overfit(
    *,
    model: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
    config: PairscaleOverfitConfig,
) -> dict[str, Any]:
    """Evaluate train-only memorization and partner controls without selection."""
    torch = _require_torch()
    np = _require_numpy()
    model.eval()
    per_complex: list[dict[str, Any]] = []
    pooled_labels: list[Any] = []
    pooled_probabilities: list[Any] = []
    matched_top_scores: list[float] = []
    matched_centered_scores: list[float] = []
    with torch.inference_mode():
        for example in examples:
            outputs, targets = _forward_contact(model, example, device)
            probabilities = outputs["contact_probabilities"][0].float().cpu().numpy()
            labels = targets[0].cpu().numpy()
            mask = outputs["candidate_pair_mask"][0].cpu().numpy()
            selected_labels = labels[mask]
            selected_probabilities = probabilities[mask]
            metrics = contact_map_metrics(selected_labels, selected_probabilities)
            model_outer = _model_contact_marginal_outer(probabilities)[mask]
            label_outer = _label_marginal_outer(labels)[mask]
            metrics["model_marginal_outer_auprc"] = average_precision(
                selected_labels,
                model_outer,
            )
            metrics["label_marginal_oracle_auprc"] = average_precision(
                selected_labels,
                label_outer,
            )
            metrics["instance_id"] = example.instance_id
            per_complex.append(metrics)
            pooled_labels.append(selected_labels.astype(np.uint8, copy=False))
            pooled_probabilities.append(selected_probabilities.astype(np.float32, copy=False))
            logits = outputs["contact_logits"]
            matched_top_scores.append(
                float(_top_k_mean(logits[0], example.positive_pair_count).cpu())
            )
            _, centered_score = double_centered_partner_score(
                logits.float(),
                outputs["candidate_pair_mask"],
                centering_iterations=32,
            )
            matched_centered_scores.append(float(centered_score[0].cpu()))

        decoy_top_scores: list[float] = []
        decoy_centered_scores: list[float] = []
        for matched_index, decoy_index in enumerate(_select_cyclic_decoys(examples)):
            matched = examples[matched_index]
            decoy = examples[decoy_index]
            outputs = _forward_cross_pair(model, matched, decoy, device)
            pair_mask = outputs["candidate_pair_mask"]
            logits = outputs["contact_logits"]
            decoy_top_scores.append(
                float(_top_k_mean(logits[0], matched.positive_pair_count).cpu())
            )
            _, centered_score = double_centered_partner_score(
                logits.float(),
                pair_mask,
                centering_iterations=32,
            )
            decoy_centered_scores.append(float(centered_score[0].cpu()))

    macro_auprc = _mean([float(value["auprc"]) for value in per_complex])
    macro_precision_at_k = _mean(
        [float(value["precision_at_true_contact_count"]) for value in per_complex]
    )
    top_wins = _mean(
        [
            float(matched > decoy)
            for matched, decoy in zip(matched_top_scores, decoy_top_scores)
        ]
    )
    centered_wins = _mean(
        [
            float(matched > decoy)
            for matched, decoy in zip(matched_centered_scores, decoy_centered_scores)
        ]
    )
    top_delta = _mean(
        [matched - decoy for matched, decoy in zip(matched_top_scores, decoy_top_scores)]
    )
    centered_delta = _mean(
        [
            matched - decoy
            for matched, decoy in zip(matched_centered_scores, decoy_centered_scores)
        ]
    )
    gate_checks = {
        "macro_pair_auprc": macro_auprc >= config.gate_auprc,
        "macro_precision_at_true_contact_count": (
            macro_precision_at_k >= config.gate_precision_at_true_count
        ),
        "matched_partner_topk_win_rate": top_wins >= config.gate_partner_win_rate,
        "partner_permutation_mean_decline": top_delta > 0.0,
    }
    return {
        "scope": "authenticated_train_only_overfit_nonconfirmatory",
        "model_selection_authorized": False,
        "held_out_evaluation_performed": False,
        "complete_candidate_domain": True,
        "negative_pair_subsampling": False,
        "unknown_cross_pair_semantics": (
            "ranking_control_only_not_a_measured_nonbinder"
        ),
        "macro_pair_auprc": macro_auprc,
        "pooled_pair_auprc": average_precision(
            np.concatenate(pooled_labels),
            np.concatenate(pooled_probabilities),
        ),
        "macro_precision_at_true_contact_count": macro_precision_at_k,
        "macro_model_marginal_outer_auprc": _mean(
            [float(value["model_marginal_outer_auprc"]) for value in per_complex]
        ),
        "macro_label_marginal_oracle_auprc": _mean(
            [float(value["label_marginal_oracle_auprc"]) for value in per_complex]
        ),
        "partner_controls": {
            "decoy_policy": "deterministic_cyclic_non_cognate_antigen",
            "matched_partner_topk_win_rate": top_wins,
            "matched_partner_topk_mean_delta": top_delta,
            "double_centered_partner_win_rate": centered_wins,
            "double_centered_partner_mean_delta": centered_delta,
            "matched_topk_scores": matched_top_scores,
            "decoy_topk_scores": decoy_top_scores,
            "matched_double_centered_scores": matched_centered_scores,
            "decoy_double_centered_scores": decoy_centered_scores,
        },
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "per_complex": per_complex,
    }


def module_source_hashes() -> dict[str, str]:
    """Bind every directly executed scientific module to exact source bytes."""
    names = (
        "interaction_first.contacts",
        "interaction_first.development_contact_dataset",
        "interaction_first.metrics",
        "interaction_first.pair_geometry",
        "interaction_first.pair_geometry_losses",
        "interaction_first.pair_geometry_schema",
        "interaction_first.pairscale_overfit",
    )
    result: dict[str, str] = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(str(module.__file__)).resolve()
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(result) != len(names):
        raise RuntimeError("Executed module basenames are not unique")
    return dict(sorted(result.items()))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(payload))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(payload)))
        handle.flush()


def run_overfit_training(
    *,
    dataset: DevelopmentContactDataset,
    output_root: Path,
    config: PairscaleOverfitConfig,
    resume_checkpoint_root: Path | None = None,
    resume_generation: str | None = None,
    expected_resume_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Run a create-only training attempt and publish immutable evidence."""
    torch = _require_torch()
    config.validate()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    metrics_path = output_root / "epoch_metrics.jsonl"
    checkpoints_root = output_root / "checkpoints"
    examples = tuple(dataset[index] for index in range(len(dataset)))
    if len(examples) != 32:
        raise ValueError("The first PAIRScale overfit gate requires exactly 32 complexes")
    configure_determinism(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This real-data overfit run requires CUDA")
    device = torch.device("cuda", 0)
    model = build_contact_only_model(dataset, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * config.minimum_learning_rate_fraction,
    )
    provenance = {
        "cache_manifest_sha256": dataset.manifest_sha256,
        "cache_receipt_file_sha256": dataset.receipt_file_sha256,
        **{f"source_{name}": digest for name, digest in module_source_hashes().items()},
    }
    config_payload = config.canonical_payload()
    start_epoch = 1
    global_step = 0
    resume_evidence: Mapping[str, Any] | None = None
    resume_requested = any(
        value is not None
        for value in (
            resume_checkpoint_root,
            resume_generation,
            expected_resume_receipt_sha256,
        )
    )
    if resume_requested:
        if not all(
            value is not None
            for value in (
                resume_checkpoint_root,
                resume_generation,
                expected_resume_receipt_sha256,
            )
        ):
            raise ValueError("Resume root, generation, and receipt anchor are inseparable")
        resume_evidence = restore_checkpoint_generation(
            Path(resume_checkpoint_root),
            str(resume_generation),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            map_location=device,
            expected_provenance_hashes=provenance,
            expected_receipt_sha256=str(expected_resume_receipt_sha256),
            expected_rng_world_size=1,
        )
        if resume_evidence["config"] != config_payload:
            raise ValueError("Resume checkpoint configuration mismatch")
        start_epoch = int(resume_evidence["epoch"]) + 1
        global_step = int(resume_evidence["global_step"])
    run_manifest = {
        "format_version": "pairscale_train_only_overfit_run_manifest_v1",
        "scope": "authenticated_train_only_overfit_nonconfirmatory",
        "validation_or_test_access_authorized": False,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_receipt_file_sha256": dataset.receipt_file_sha256,
        "instance_count": len(examples),
        "candidate_pair_count": sum(int(value.contact_mask.sum()) for value in examples),
        "positive_pair_count": sum(value.positive_pair_count for value in examples),
        "config": config_payload,
        "source_sha256": module_source_hashes(),
        "resume": None if resume_evidence is None else dict(resume_evidence),
    }
    _atomic_json(output_root / "RUN_MANIFEST.json", run_manifest)
    last_evaluation: Mapping[str, Any] | None = None
    last_receipt: Mapping[str, Any] | None = None
    for epoch in range(start_epoch, config.epochs + 1):
        losses, steps = train_one_epoch(
            model=model,
            optimizer=optimizer,
            examples=examples,
            device=device,
            config=config,
            epoch=epoch,
        )
        global_step += steps
        scheduler.step()
        should_evaluate = epoch == 1 or epoch % config.eval_every_epochs == 0 or epoch == config.epochs
        if should_evaluate:
            last_evaluation = evaluate_overfit(
                model=model,
                examples=examples,
                device=device,
                config=config,
            )
        epoch_metrics = {
            "format_version": METRICS_FORMAT,
            "epoch": epoch,
            "global_step": global_step,
            "mean_training_loss": losses["total"],
            "mean_loss_components": dict(losses),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "evaluation": last_evaluation if should_evaluate else None,
        }
        _append_jsonl(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        should_checkpoint = (
            epoch % config.checkpoint_every_epochs == 0
            or epoch == config.epochs
            or (last_evaluation is not None and bool(last_evaluation["gate_passed"]))
        )
        if should_checkpoint:
            generation = f"epoch-{epoch:04d}-step-{global_step:08d}"
            last_receipt = create_checkpoint_generation(
                checkpoints_root,
                generation,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=epoch,
                global_step=global_step,
                provenance_hashes=provenance,
                config=config_payload,
            )
            last_receipt = {
                **dict(last_receipt),
                "receipt_file_sha256": sha256_file(
                    checkpoints_root / generation / "receipt.json"
                ),
            }
        if last_evaluation is not None and bool(last_evaluation["gate_passed"]):
            break
    if last_evaluation is None or last_receipt is None:
        raise RuntimeError("Training ended without evaluation and checkpoint evidence")
    report = {
        "format_version": REPORT_FORMAT,
        "scope": "authenticated_train_only_overfit_nonconfirmatory",
        "claim_boundary": (
            "training-set memorization and engineering controls only; no held-out, "
            "confirmatory, comparative, or state-of-the-art claim"
        ),
        "completed_epoch": int(last_receipt["epoch"]),
        "global_step": int(last_receipt["global_step"]),
        "gate_passed": bool(last_evaluation["gate_passed"]),
        "evaluation": dict(last_evaluation),
        "final_checkpoint": dict(last_receipt),
        "resume_exactness_status": (
            "checkpoint_restored_from_external_receipt_anchor"
            if resume_evidence is not None
            else "not_yet_tested_by_forked_replay"
        ),
    }
    _atomic_json(output_root / "FINAL_REPORT.json", report)
    return report
