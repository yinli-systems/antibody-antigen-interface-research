"""Fixed staged training step for the PAIRScale E1/E2 repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .development_contact_dataset import (
    DevelopmentContactDataset,
    DevelopmentContactExample,
)
from .pair_geometry import ProspectivePairGeometryModel
from .pair_geometry_losses import masked_contact_bce_with_logits
from .pairscale_overfit import (
    contact_only_geometry_schema,
    deterministic_epoch_order,
    hard_negative_contact_ranking_loss,
)
from .pairscale_partner_specificity import (
    BidirectionalDecoyPanel,
    bidirectional_specificity_losses,
    build_single_anchor_score_row,
)


PHASE_UNARY_WARMUP = "unary_warmup"
PHASE_RESIDUAL_SPECIFICITY = "frozen_unary_residual_specificity"
PHASE_JOINT_FINETUNE = "low_rate_joint_finetune"

UNARY_PATH_MODULES = (
    "antibody_projection",
    "antigen_projection",
    "antibody_final_norm",
    "antigen_final_norm",
    "antibody_unary_head",
    "antigen_unary_head",
)
PAIR_PATH_MODULES = (
    "fusion_layers",
    "antibody_pair_add",
    "antigen_pair_add",
    "antibody_pair_gate",
    "antigen_pair_gate",
    "pair_refinement",
    "interaction_head",
    "specificity_head",
)
UNUSED_FROZEN_MODULES = (
    "distance_head",
    "orientation_head",
)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return converted


def _finite_positive(value: Any, name: str) -> float:
    converted = _finite_nonnegative(value, name)
    if converted == 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


@dataclass(frozen=True)
class PairscaleE1E2Config:
    """Freeze E1/E2 architecture, objectives, and phase boundaries."""

    seed: int = 17
    epochs: int = 260
    unary_warmup_end_epoch: int = 20
    residual_phase_end_epoch: int = 220
    freeze_complete_unary_path_during_residual_phase: bool = True
    learning_rate: float = 0.001
    joint_learning_rate_multiplier: float = 0.1
    weight_decay: float = 0.0
    positive_weight: float = 1024.0
    contact_ranking_weight: float = 1.0
    hard_negative_ratio: int = 8
    contact_ranking_temperature: float = 1.0
    retrieval_weight: float = 0.25
    retrieval_temperature: float = 0.5
    cognate_margin_weight: float = 0.25
    cognate_margin: float = 0.1
    cognate_margin_temperature: float = 0.5
    decoy_count_per_direction: int = 4
    gradient_clip_norm: float = 5.0
    d_model: int = 192
    pair_dim: int = 96
    num_heads: int = 6
    num_fusion_layers: int = 2
    ffn_dim: int = 384
    pair_chunk_size: int = 64
    dropout: float = 0.0
    eval_every_epochs: int = 20
    checkpoint_every_epochs: int = 20
    precision: str = "bf16"

    def validate(self) -> None:
        """Reject settings that weaken the staged protocol."""
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.freeze_complete_unary_path_during_residual_phase is not True:
            raise ValueError(
                "freeze_complete_unary_path_during_residual_phase must be true"
            )
        for name in (
            "epochs",
            "unary_warmup_end_epoch",
            "residual_phase_end_epoch",
            "hard_negative_ratio",
            "decoy_count_per_direction",
            "d_model",
            "pair_dim",
            "num_heads",
            "num_fusion_layers",
            "ffn_dim",
            "pair_chunk_size",
            "eval_every_epochs",
            "checkpoint_every_epochs",
        ):
            _positive_integer(getattr(self, name), name)
        if not (
            self.unary_warmup_end_epoch
            < self.residual_phase_end_epoch
            < self.epochs
        ):
            raise ValueError(
                "Phase boundaries must satisfy unary < residual < epochs"
            )
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        for name in (
            "learning_rate",
            "joint_learning_rate_multiplier",
            "positive_weight",
            "contact_ranking_temperature",
            "retrieval_temperature",
            "cognate_margin",
            "cognate_margin_temperature",
            "gradient_clip_norm",
        ):
            _finite_positive(getattr(self, name), name)
        for name in (
            "weight_decay",
            "contact_ranking_weight",
            "retrieval_weight",
            "cognate_margin_weight",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if not 0.0 < self.joint_learning_rate_multiplier <= 1.0:
            raise ValueError("joint_learning_rate_multiplier must be in (0, 1]")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")

    def canonical_payload(self) -> Mapping[str, Any]:
        self.validate()
        return asdict(self)


def training_phase(epoch: int, config: PairscaleE1E2Config) -> str:
    """Return the only authorized phase for a one-indexed epoch."""
    config.validate()
    _positive_integer(epoch, "epoch")
    if epoch > config.epochs:
        raise ValueError("epoch exceeds the fixed training horizon")
    if epoch <= config.unary_warmup_end_epoch:
        return PHASE_UNARY_WARMUP
    if epoch <= config.residual_phase_end_epoch:
        return PHASE_RESIDUAL_SPECIFICITY
    return PHASE_JOINT_FINETUNE


def build_e1_e2_model(
    dataset: DevelopmentContactDataset,
    config: PairscaleE1E2Config,
) -> ProspectivePairGeometryModel:
    """Construct the fixed decomposed model without changing the backbone."""
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
        contact_logit_mode="unary_plus_interaction_residual",
    )


def _device_streams(
    antibody_example: DevelopmentContactExample,
    antigen_example: DevelopmentContactExample,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    antibody = antibody_example.antibody_embeddings.to(
        device=device,
        non_blocking=True,
    ).unsqueeze(0)
    antigen = antigen_example.antigen_embeddings.to(
        device=device,
        non_blocking=True,
    ).unsqueeze(0)
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
    complete_pair_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    return antibody, antigen, antibody_mask, antigen_mask, complete_pair_mask


def _forward_unknown_pair(
    model: Any,
    antibody_example: DevelopmentContactExample,
    antigen_example: DevelopmentContactExample,
    device: Any,
) -> Mapping[str, Any]:
    inputs = _device_streams(antibody_example, antigen_example, device)
    return model(
        *inputs,
        return_pair_representations=False,
        enabled_heads={"interaction"},
    )


def _forward_cognate(
    model: Any,
    example: DevelopmentContactExample,
    device: Any,
) -> tuple[Mapping[str, Any], Any, Any]:
    import torch

    inputs = _device_streams(example, example, device)
    outputs = model(
        *inputs,
        return_pair_representations=False,
        enabled_heads={"contact", "interaction"},
    )
    targets = example.contact_targets.to(
        device=device,
        non_blocking=True,
    ).unsqueeze(0)
    supervision_mask = example.contact_mask.to(
        device=device,
        non_blocking=True,
    ).unsqueeze(0)
    if tuple(targets.shape) != tuple(outputs["contact_logits"].shape):
        raise ValueError("Contact targets do not align with model outputs")
    if supervision_mask.dtype != torch.bool:
        raise ValueError("Contact supervision mask must be boolean")
    return outputs, targets, supervision_mask


def _phase_contact_logits(outputs: Mapping[str, Any], phase: str) -> Any:
    """Expose exactly one gradient path for each frozen training phase."""
    mask = outputs["candidate_pair_mask"]
    antibody_unary = outputs["antibody_unary_logits"].unsqueeze(2)
    antigen_unary = outputs["antigen_unary_logits"].unsqueeze(1)
    residual = outputs["contact_interaction_residual_logits"]
    if phase == PHASE_UNARY_WARMUP:
        return (antibody_unary + antigen_unary).masked_fill(~mask, 0.0)
    if phase == PHASE_RESIDUAL_SPECIFICITY:
        return (
            antibody_unary.detach() + antigen_unary.detach() + residual
        ).masked_fill(~mask, 0.0)
    if phase == PHASE_JOINT_FINETUNE:
        return outputs["contact_logits"]
    raise ValueError("Unknown E1/E2 training phase")


def set_phase_learning_rate(
    optimizer: Any,
    config: PairscaleE1E2Config,
    phase: str,
) -> float:
    """Apply the frozen low-rate multiplier only during joint fine-tuning."""
    learning_rate = config.learning_rate
    if phase == PHASE_JOINT_FINETUNE:
        learning_rate *= config.joint_learning_rate_multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _parameter_ownership(model: Any) -> Mapping[str, Mapping[str, Any]]:
    """Assign every parameter to exactly one locked training path."""
    named_parameters = dict(model.named_parameters())
    name_by_identity = {id(value): name for name, value in named_parameters.items()}
    if len(name_by_identity) != len(named_parameters):
        raise RuntimeError("Model exposes aliased named parameters")
    groups: dict[str, dict[str, Any]] = {
        "unary": {},
        "pair": {},
        "unused": {},
    }
    seen: set[int] = set()
    for group, module_names in (
        ("unary", UNARY_PATH_MODULES),
        ("pair", PAIR_PATH_MODULES),
        ("unused", UNUSED_FROZEN_MODULES),
    ):
        for module_name in module_names:
            module = getattr(model, module_name, None)
            if module is None:
                raise RuntimeError(
                    f"Locked E1/E2 module {module_name} is unavailable"
                )
            for parameter in module.parameters():
                identity = id(parameter)
                if identity in seen:
                    raise RuntimeError("A parameter belongs to multiple training paths")
                name = name_by_identity.get(identity)
                if name is None:
                    raise RuntimeError("A module parameter is absent from the model")
                groups[group][name] = parameter
                seen.add(identity)
    missing = sorted(
        name for name, parameter in named_parameters.items() if id(parameter) not in seen
    )
    if missing:
        raise RuntimeError(f"Unclassified E1/E2 parameters: {missing}")
    return groups


def set_phase_trainable_parameters(model: Any, phase: str) -> Mapping[str, Any]:
    """Freeze complete paths so a fixed unary head cannot drift indirectly."""
    groups = _parameter_ownership(model)
    if phase == PHASE_UNARY_WARMUP:
        trainable_groups = frozenset({"unary"})
    elif phase == PHASE_RESIDUAL_SPECIFICITY:
        trainable_groups = frozenset({"pair"})
    elif phase == PHASE_JOINT_FINETUNE:
        trainable_groups = frozenset({"unary", "pair"})
    else:
        raise ValueError("Unknown E1/E2 training phase")
    for group_name, parameters in groups.items():
        trainable = group_name in trainable_groups
        for parameter in parameters.values():
            parameter.requires_grad_(trainable)
    trainable = [
        parameter
        for parameters in groups.values()
        for parameter in parameters.values()
        if parameter.requires_grad
    ]
    frozen = [
        parameter
        for parameters in groups.values()
        for parameter in parameters.values()
        if not parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("The selected E1/E2 phase has no trainable parameters")
    return {
        "trainable_parameter_count": sum(item.numel() for item in trainable),
        "trainable_parameter_tensor_count": len(trainable),
        "frozen_parameter_count": sum(item.numel() for item in frozen),
        "frozen_parameter_tensor_count": len(frozen),
        "unary_path_trainable": "unary" in trainable_groups,
        "pair_path_trainable": "pair" in trainable_groups,
        "unused_geometry_heads_trainable": False,
    }


def parameter_path_state_sha256(model: Any, path: str) -> str:
    """Hash one complete parameter path with dtype and shape metadata."""
    import torch

    groups = _parameter_ownership(model)
    if path not in groups:
        raise ValueError("Parameter path must be unary, pair, or unused")
    digest = hashlib.sha256()
    for name, parameter in sorted(groups[path].items()):
        tensor = parameter.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for block in (metadata, raw):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()


def train_one_epoch_e1_e2(
    *,
    model: Any,
    optimizer: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
    config: PairscaleE1E2Config,
    epoch: int,
    decoy_panel: BidirectionalDecoyPanel,
) -> tuple[Mapping[str, float | str | int | bool], int]:
    """Train one fixed epoch with separate contact and specificity paths."""
    import torch

    config.validate()
    decoy_panel.validate(len(examples))
    phase = training_phase(epoch, config)
    phase_parameters = set_phase_trainable_parameters(model, phase)
    learning_rate = set_phase_learning_rate(optimizer, config, phase)
    model.train()
    loss_sums = {
        "total": 0.0,
        "contact_bce": 0.0,
        "contact_ranking": 0.0,
        "bidirectional_retrieval": 0.0,
        "bidirectional_cognate_margin": 0.0,
    }
    step_count = 0
    use_bf16 = config.precision == "bf16" and device.type == "cuda"
    for index in deterministic_epoch_order(len(examples), config.seed, epoch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            enabled=use_bf16,
        ):
            matched, targets, supervision_mask = _forward_cognate(
                model,
                examples[index],
                device,
            )
            contact_logits = _phase_contact_logits(matched, phase)
            contact_bce = masked_contact_bce_with_logits(
                contact_logits,
                targets,
                supervision_mask,
                positive_weight=config.positive_weight,
            )
            contact_ranking = hard_negative_contact_ranking_loss(
                contact_logits,
                targets,
                supervision_mask,
                negative_ratio=config.hard_negative_ratio,
                temperature=config.contact_ranking_temperature,
            )
            if phase == PHASE_UNARY_WARMUP:
                retrieval = contact_bce.new_zeros(())
                cognate_margin = contact_bce.new_zeros(())
            else:
                antigen_decoys = [
                    _forward_unknown_pair(
                        model,
                        examples[index],
                        examples[decoy_index],
                        device,
                    )
                    for decoy_index in decoy_panel.antibody_to_antigen[index]
                ]
                antibody_decoys = [
                    _forward_unknown_pair(
                        model,
                        examples[decoy_index],
                        examples[index],
                        device,
                    )
                    for decoy_index in decoy_panel.antigen_to_antibody[index]
                ]
                ab_scores = build_single_anchor_score_row(
                    matched,
                    antigen_decoys,
                )
                ag_scores = build_single_anchor_score_row(
                    matched,
                    antibody_decoys,
                )
                positive_mask = torch.zeros_like(ab_scores, dtype=torch.bool)
                positive_mask[:, 0] = True
                specificity = bidirectional_specificity_losses(
                    antibody_to_antigen_scores=ab_scores,
                    antibody_to_antigen_positive_mask=positive_mask,
                    antigen_to_antibody_scores=ag_scores,
                    antigen_to_antibody_positive_mask=positive_mask,
                    retrieval_temperature=config.retrieval_temperature,
                    margin=config.cognate_margin,
                    margin_temperature=config.cognate_margin_temperature,
                )
                retrieval = specificity.retrieval
                cognate_margin = specificity.margin
            loss = (
                contact_bce
                + config.contact_ranking_weight * contact_ranking
                + config.retrieval_weight * retrieval
                + config.cognate_margin_weight * cognate_margin
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, example {index}")
        loss.backward()
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
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
            ("bidirectional_retrieval", retrieval),
            ("bidirectional_cognate_margin", cognate_margin),
        ):
            loss_sums[name] += float(value.detach().cpu())
        step_count += 1
    result: dict[str, float | str | int | bool] = {
        name: value / step_count for name, value in loss_sums.items()
    }
    result["phase"] = phase
    result["learning_rate"] = learning_rate
    result.update(phase_parameters)
    return result, step_count


__all__ = [
    "PHASE_JOINT_FINETUNE",
    "PHASE_RESIDUAL_SPECIFICITY",
    "PHASE_UNARY_WARMUP",
    "PairscaleE1E2Config",
    "PAIR_PATH_MODULES",
    "UNARY_PATH_MODULES",
    "UNUSED_FROZEN_MODULES",
    "build_e1_e2_model",
    "parameter_path_state_sha256",
    "set_phase_trainable_parameters",
    "set_phase_learning_rate",
    "train_one_epoch_e1_e2",
    "training_phase",
]
