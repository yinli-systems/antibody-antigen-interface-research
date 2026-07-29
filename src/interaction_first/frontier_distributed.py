"""Distributed primitives for partner-aware contrastive training."""

from __future__ import annotations

from typing import Any


LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND = "local_slice_ddp_exact"
EXACT_ALL_REDUCE_GRADIENT_BACKEND = "exact_all_reduce"
LEGACY_AUTOGRAD_GRADIENT_BACKEND = "legacy_autograd"
SUPPORTED_GATHER_GRADIENT_BACKENDS = frozenset(
    {
        LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
        EXACT_ALL_REDUCE_GRADIENT_BACKEND,
        LEGACY_AUTOGRAD_GRADIENT_BACKEND,
    }
)


def _require_torch() -> Any:
    """Import PyTorch lazily so metadata-only workflows remain usable."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for distributed training") from error
    return torch


def _distributed_context(group: Any = None) -> tuple[Any, int, int]:
    """Return the initialized process group, world size, and rank."""
    torch = _require_torch()
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return distributed, 1, 0
    return (
        distributed,
        int(distributed.get_world_size(group=group)),
        int(distributed.get_rank(group=group)),
    )


def _validate_equal_shape(tensor: Any, group: Any = None) -> None:
    """Fail collectively when ranks provide tensors with unequal shapes."""
    torch = _require_torch()
    distributed, world_size, _ = _distributed_context(group)
    if world_size == 1:
        return
    ndim = torch.tensor([tensor.ndim], dtype=torch.int64, device=tensor.device)
    gathered_ndim = [torch.empty_like(ndim) for _ in range(world_size)]
    distributed.all_gather(gathered_ndim, ndim, group=group)
    dimensions = [int(value.item()) for value in gathered_ndim]
    if len(set(dimensions)) != 1:
        raise ValueError(f"All-gather rank mismatch: tensor ranks are {dimensions}")
    shape = torch.tensor(tuple(tensor.shape), dtype=torch.int64, device=tensor.device)
    gathered_shapes = [torch.empty_like(shape) for _ in range(world_size)]
    distributed.all_gather(gathered_shapes, shape, group=group)
    shapes = [tuple(int(value) for value in item.tolist()) for item in gathered_shapes]
    if len(set(shapes)) != 1:
        raise ValueError(f"All-gather shape mismatch: tensor shapes are {shapes}")


def _validate_gather_gradient_backend(gradient_backend: str) -> str:
    """Return one explicitly supported gather-gradient implementation."""
    if not isinstance(gradient_backend, str):
        raise TypeError("gradient_backend must be a string")
    if gradient_backend not in SUPPORTED_GATHER_GRADIENT_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_GATHER_GRADIENT_BACKENDS))
        raise ValueError(
            f"Unsupported gather gradient backend {gradient_backend!r}; "
            f"expected one of: {supported}"
        )
    return gradient_backend


def _gather_with_local_slice_ddp_exact(tensor: Any, group: Any = None) -> Any:
    """Gather detached values while retaining one DDP-scaled local slice.

    This backend is exact only when all of the following invariants hold:

    * every rank computes the same full-global scalar objective;
    * the gather group is the same fixed group used by standard DDP;
    * DDP averages parameter gradients by that fixed world size; and
    * Join, uneven effective-world-size semantics, and communication hooks that
      change DDP's averaging rule are not used.

    Ordinary all-gather supplies identical detached values on every rank. The
    local rank's slice is replaced by a live identity whose backward multiplies
    only that embedding gradient by the fixed world size. DDP's subsequent
    parameter-gradient average cancels this factor. Parameters used after the
    gather, such as a learnable temperature, are deliberately not scaled.
    """
    torch = _require_torch()
    distributed, world_size, rank = _distributed_context(group)

    class _DDPExactLocalIdentity(torch.autograd.Function):
        @staticmethod
        def forward(context: Any, local_tensor: Any) -> Any:
            context.world_size = world_size
            return local_tensor.clone()

        @staticmethod
        def backward(context: Any, gradient: Any) -> tuple[Any]:
            return (gradient * context.world_size,)

    contiguous_detached = tensor.detach().contiguous()
    gathered = [
        torch.empty_like(contiguous_detached) for _ in range(world_size)
    ]
    distributed.all_gather(gathered, contiguous_detached, group=group)
    gathered[rank] = _DDPExactLocalIdentity.apply(tensor)
    return torch.cat(gathered, dim=0)


def _gather_with_exact_all_reduce_backward(tensor: Any, group: Any = None) -> Any:
    """Gather tensors with an exact SUM-all-reduce backward.

    Let rank ``k`` evaluate a loss ``L_k`` from the same concatenated output
    ``Y = concat(X_0, ..., X_{W-1})``. The correct gradient for ``X_r`` is
    ``sum_k dL_k/dY[r]``. A SUM all-reduce of the complete upstream gradient
    followed by the local-rank slice computes exactly that quantity while
    avoiding the reduce-scatter path used by PyTorch's legacy autograd gather.

    Reducing only each process's differently indexed local slice would mix
    gradients from different examples and is therefore intentionally avoided.
    """
    torch = _require_torch()
    distributed, world_size, rank = _distributed_context(group)
    local_batch_size = int(tensor.shape[0])

    class _ExactAllGather(torch.autograd.Function):
        @staticmethod
        def forward(context: Any, local_tensor: Any) -> Any:
            contiguous = local_tensor.contiguous()
            gathered = [torch.empty_like(contiguous) for _ in range(world_size)]
            distributed.all_gather(gathered, contiguous, group=group)
            context.local_batch_size = local_batch_size
            context.rank = rank
            return torch.cat(gathered, dim=0)

        @staticmethod
        def backward(context: Any, gradient: Any) -> tuple[Any]:
            expected_leading_size = world_size * context.local_batch_size
            if int(gradient.shape[0]) != expected_leading_size:
                raise RuntimeError(
                    "Gather backward received an invalid leading dimension: "
                    f"expected {expected_leading_size}, received "
                    f"{int(gradient.shape[0])}"
                )
            reduced_gradient = gradient.contiguous().clone()
            distributed.all_reduce(
                reduced_gradient,
                op=distributed.ReduceOp.SUM,
                group=group,
            )
            local_gradient = reduced_gradient.narrow(
                0,
                context.rank * context.local_batch_size,
                context.local_batch_size,
            )
            return (local_gradient.contiguous(),)

    return _ExactAllGather.apply(tensor)


def _gather_with_legacy_autograd(tensor: Any, group: Any = None) -> Any:
    """Use PyTorch's autograd-aware all-gather for controlled comparisons."""
    torch = _require_torch()
    try:
        from torch.distributed.nn.functional import all_gather
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "The legacy backend requires PyTorch's autograd-aware all_gather"
        ) from error
    gathered = all_gather(tensor.contiguous(), group=group)
    return torch.cat(tuple(gathered), dim=0)


def gather_equal_with_grad(
    tensor: Any,
    group: Any = None,
    *,
    gradient_backend: str = LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
) -> Any:
    """Concatenate equal-shape tensors while preserving cross-rank gradients.

    Every rank receives the same rank-major concatenated tensor. The production
    default, ``local_slice_ddp_exact``, has no backward collective and is exact
    under the fixed standard-DDP invariants documented by
    :func:`_gather_with_local_slice_ddp_exact`. ``exact_all_reduce`` preserves
    general distributed-autograd semantics for diagnostics. ``legacy_autograd``
    remains available only for controlled compatibility checks.
    """
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError("tensor must be a PyTorch tensor")
    if tensor.ndim == 0:
        raise ValueError("tensor must have a leading batch dimension")
    selected_backend = _validate_gather_gradient_backend(gradient_backend)
    _, world_size, _ = _distributed_context(group)
    if world_size == 1:
        return tensor
    _validate_equal_shape(tensor, group)
    if selected_backend == LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND:
        return _gather_with_local_slice_ddp_exact(tensor, group=group)
    if selected_backend == EXACT_ALL_REDUCE_GRADIENT_BACKEND:
        return _gather_with_exact_all_reduce_backward(tensor, group=group)
    return _gather_with_legacy_autograd(tensor, group=group)


def _gather_equal_without_grad(tensor: Any, group: Any = None) -> Any:
    """Concatenate equal-shape metadata tensors without autograd tracking."""
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError("tensor must be a PyTorch tensor")
    if tensor.ndim == 0:
        raise ValueError("tensor must have a leading batch dimension")
    distributed, world_size, _ = _distributed_context(group)
    if world_size == 1:
        return tensor
    _validate_equal_shape(tensor, group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    distributed.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=0)


def _validate_boolean_mask(
    value: Any,
    expected_shape: tuple[int, ...],
    expected_device: Any,
    name: str,
) -> Any:
    """Require an exact boolean tensor instead of coercing untrusted values."""
    torch = _require_torch()
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if value.device != expected_device:
        raise ValueError(f"{name} and score tensors must share one device")
    return value


def _validate_replicated_objective_masks(
    positive_mask: Any,
    exclusion_mask: Any,
    group: Any = None,
) -> None:
    """Fail collectively when ranks do not use one identical global objective."""
    torch = _require_torch()
    distributed, world_size, _ = _distributed_context(group)
    if world_size == 1:
        return
    _validate_equal_shape(positive_mask, group)
    _validate_equal_shape(exclusion_mask, group)
    encoded = (
        positive_mask.to(dtype=torch.uint8)
        + exclusion_mask.to(dtype=torch.uint8) * 2
    ).contiguous()
    minimum = encoded.clone()
    maximum = encoded.clone()
    distributed.all_reduce(minimum, op=distributed.ReduceOp.MIN, group=group)
    distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX, group=group)
    if not torch.equal(minimum, maximum):
        raise ValueError(
            "Every rank must use identical observed-positive and exclusion masks"
        )


def exact_distributed_mean(
    local_sum: Any,
    local_count: Any,
    group: Any = None,
) -> Any:
    """Return the exact global mean with DDP-correct backward scaling.

    The forward scalar is identical on every rank. The custom backward includes
    the world-size factor that is subsequently cancelled by DistributedDataParallel's
    gradient averaging, yielding the gradient of the true sample-weighted mean.
    """
    torch = _require_torch()
    if (
        not torch.is_tensor(local_sum)
        or local_sum.layout != torch.strided
        or local_sum.ndim != 0
    ):
        raise ValueError("local_sum must be a scalar PyTorch tensor")
    if not local_sum.is_floating_point() or local_sum.is_complex():
        raise TypeError("local_sum must use a real floating-point dtype")
    if local_sum.dtype not in (torch.float32, torch.float64):
        raise TypeError("local_sum must use float32 or float64 for stable reduction")
    if not bool(torch.isfinite(local_sum.detach()).item()):
        raise ValueError("local_sum must be finite")
    if torch.is_tensor(local_count):
        if local_count.layout != torch.strided or local_count.ndim != 0:
            raise ValueError("local_count must be scalar")
        if local_count.dtype == torch.bool or local_count.is_complex():
            raise TypeError("local_count must use a real numeric dtype")
        count = local_count.detach().to(device=local_sum.device, dtype=torch.float64)
    else:
        if isinstance(local_count, bool):
            raise TypeError("local_count cannot be boolean")
        count = torch.tensor(
            float(local_count),
            dtype=torch.float64,
            device=local_sum.device,
        )
    if not bool(torch.isfinite(count).item()) or float(count.item()) < 0:
        raise ValueError("local_count must be finite and non-negative")
    distributed, world_size, _ = _distributed_context(group)
    if world_size == 1:
        if float(count.item()) <= 0:
            raise ValueError("The global mean denominator must be positive")
        return local_sum / count.to(dtype=local_sum.dtype)

    class _ExactDistributedMean(torch.autograd.Function):
        @staticmethod
        def forward(context: Any, value: Any, sample_count: Any) -> Any:
            total_value = value.detach().clone()
            total_count = sample_count.detach().clone()
            distributed.all_reduce(
                total_value,
                op=distributed.ReduceOp.SUM,
                group=group,
            )
            distributed.all_reduce(
                total_count,
                op=distributed.ReduceOp.SUM,
                group=group,
            )
            if (
                not bool(torch.isfinite(total_count).item())
                or float(total_count.item()) <= 0
            ):
                raise ValueError("The global mean denominator must be positive")
            context.scale = world_size / float(total_count.item())
            return total_value / total_count.to(dtype=total_value.dtype)

        @staticmethod
        def backward(context: Any, gradient: Any) -> tuple[Any, None]:
            return gradient * context.scale, None

    return _ExactDistributedMean.apply(local_sum, count)


def _directional_multi_positive_loss(
    logits: Any,
    positive_mask: Any,
    exclusion_mask: Any,
    direction_name: str,
) -> Any:
    """Compute one direction of multi-positive InfoNCE."""
    torch = _require_torch()
    positive_counts = positive_mask.sum(dim=1)
    allowed_mask = ~exclusion_mask
    denominator_counts = allowed_mask.sum(dim=1)
    if not bool((positive_counts > 0).all().item()):
        missing = torch.nonzero(positive_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Every {direction_name} query must have an observed positive; "
            f"missing indices: {missing}"
        )
    if not bool((denominator_counts > 0).all().item()):
        missing = (
            torch.nonzero(denominator_counts == 0, as_tuple=False).flatten().tolist()
        )
        raise ValueError(
            f"Every {direction_name} query must have a denominator; "
            f"missing indices: {missing}"
        )
    negative_infinity = torch.tensor(
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    positive_logits = torch.where(positive_mask, logits, negative_infinity)
    denominator_logits = torch.where(allowed_mask, logits, negative_infinity)
    return (
        torch.logsumexp(denominator_logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def symmetric_score_matrix_info_nce(
    score_matrix: Any,
    observed_positive_mask: Any,
    exclusion_mask: Any | None = None,
    temperature: float | Any = 0.07,
    return_details: bool = False,
) -> Any:
    """Compute symmetric multi-positive InfoNCE from an explicit score matrix."""
    torch = _require_torch()
    if not torch.is_tensor(score_matrix) or score_matrix.ndim != 2:
        raise ValueError("score_matrix must be a two-dimensional tensor")
    if not torch.is_floating_point(score_matrix):
        raise ValueError("score_matrix must use a floating-point dtype")
    if score_matrix.shape[0] == 0 or score_matrix.shape[1] == 0:
        raise ValueError("score_matrix cannot contain an empty axis")
    if not bool(torch.isfinite(score_matrix).all().item()):
        raise ValueError("score_matrix must contain only finite values")
    positive = _validate_boolean_mask(
        observed_positive_mask,
        tuple(score_matrix.shape),
        score_matrix.device,
        "observed_positive_mask",
    )
    if exclusion_mask is None:
        excluded = torch.zeros_like(positive)
    else:
        excluded = _validate_boolean_mask(
            exclusion_mask,
            tuple(score_matrix.shape),
            score_matrix.device,
            "exclusion_mask",
        )
    if bool((positive & excluded).any().item()):
        raise ValueError("Observed positives cannot be excluded from denominators")
    calculation_dtype = (
        torch.float32
        if score_matrix.dtype in (torch.float16, torch.bfloat16)
        else score_matrix.dtype
    )
    if torch.is_tensor(temperature):
        if temperature.ndim != 0:
            raise ValueError("temperature must be scalar")
        if not temperature.is_floating_point() or temperature.is_complex():
            raise TypeError("temperature tensor must use a real floating-point dtype")
        temperature_tensor = temperature.to(
            device=score_matrix.device,
            dtype=calculation_dtype,
        )
    else:
        temperature_tensor = torch.tensor(
            float(temperature),
            device=score_matrix.device,
            dtype=calculation_dtype,
        )
    if not bool(torch.isfinite(temperature_tensor).item()) or float(
        temperature_tensor.detach().item()
    ) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = score_matrix.to(dtype=calculation_dtype) / temperature_tensor.to(
        dtype=calculation_dtype
    )
    antibody_to_antigen = _directional_multi_positive_loss(
        logits,
        positive,
        excluded,
        "antibody-to-antigen",
    )
    antigen_to_antibody = _directional_multi_positive_loss(
        logits.transpose(0, 1),
        positive.transpose(0, 1),
        excluded.transpose(0, 1),
        "antigen-to-antibody",
    )
    loss = 0.5 * (antibody_to_antigen + antigen_to_antibody)
    if return_details:
        return {
            "loss": loss,
            "antibody_to_antigen": antibody_to_antigen,
            "antigen_to_antibody": antigen_to_antibody,
        }
    return loss


def symmetric_multi_positive_info_nce(
    local_antibody_embeddings: Any,
    local_antigen_embeddings: Any,
    observed_positive_mask: Any,
    exclusion_mask: Any | None = None,
    temperature: float | Any = 0.07,
    local_valid_mask: Any | None = None,
    group: Any = None,
    return_details: bool = False,
    gather_gradient_backend: str = LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
) -> Any:
    """Compute symmetric global multi-positive InfoNCE.

    ``observed_positive_mask`` and ``exclusion_mask`` must follow the global
    rank-major order produced by :func:`gather_equal_with_grad`. An exclusion
    marks an ambiguous or forbidden candidate that must not enter a query's
    denominator. Observed positives may never be excluded.

    The default gather backend requires a fixed standard-DDP process group and
    an identical full-global objective on every rank. It must not be combined
    with Join or any effective-world-size or gradient-averaging override.
    """
    torch = _require_torch()
    antibody = local_antibody_embeddings
    antigen = local_antigen_embeddings
    if not torch.is_tensor(antibody) or not torch.is_tensor(antigen):
        raise TypeError("Embedding inputs must be PyTorch tensors")
    if antibody.ndim != 2 or antigen.ndim != 2:
        raise ValueError("Embedding inputs must have shape [batch, features]")
    if antibody.shape[0] != antigen.shape[0]:
        raise ValueError("Each rank must provide equal antibody and antigen batches")
    if antibody.shape[0] == 0 or antibody.shape[1] == 0:
        raise ValueError("Embedding batches cannot be empty")
    if antibody.shape[1] != antigen.shape[1]:
        raise ValueError("Antibody and antigen embedding dimensions must match")
    if antibody.device != antigen.device:
        raise ValueError("Antibody and antigen embeddings must share a device")
    if antibody.dtype != antigen.dtype:
        raise ValueError("Antibody and antigen embeddings must share a dtype")
    if not bool(torch.isfinite(antibody).all().item()) or not bool(
        torch.isfinite(antigen).all().item()
    ):
        raise ValueError("Embedding inputs must be finite")
    if local_valid_mask is None:
        local_valid = torch.ones(
            antibody.shape[0],
            dtype=torch.bool,
            device=antibody.device,
        )
    else:
        if (
            not torch.is_tensor(local_valid_mask)
            or local_valid_mask.dtype != torch.bool
            or local_valid_mask.ndim != 1
            or int(local_valid_mask.shape[0]) != int(antibody.shape[0])
        ):
            raise ValueError(
                "local_valid_mask must be boolean with shape [local_batch]"
            )
        if local_valid_mask.device != antibody.device:
            raise ValueError(
                "local_valid_mask and embedding inputs must share a device"
            )
        local_valid = local_valid_mask

    temperature_dtype = (
        torch.float64 if antibody.dtype == torch.float64 else torch.float32
    )
    if torch.is_tensor(temperature):
        if temperature.ndim != 0:
            raise ValueError("temperature must be scalar")
        if not temperature.is_floating_point() or temperature.is_complex():
            raise TypeError("temperature tensor must use a real floating-point dtype")
        temperature_tensor = temperature.to(
            device=antibody.device,
            dtype=temperature_dtype,
        )
    else:
        temperature_tensor = torch.tensor(
            float(temperature),
            device=antibody.device,
            dtype=temperature_dtype,
        )
    if not bool(torch.isfinite(temperature_tensor).item()) or float(
        temperature_tensor.detach().item()
    ) <= 0:
        raise ValueError("temperature must be finite and positive")

    selected_gather_backend = _validate_gather_gradient_backend(
        gather_gradient_backend
    )
    _, world_size, rank = _distributed_context(group)
    global_antibody = gather_equal_with_grad(
        antibody,
        group=group,
        gradient_backend=selected_gather_backend,
    )
    global_antigen = gather_equal_with_grad(
        antigen,
        group=group,
        gradient_backend=selected_gather_backend,
    )
    global_valid = _gather_equal_without_grad(local_valid, group=group)
    expected_shape = (global_antibody.shape[0], global_antigen.shape[0])
    positive = _validate_boolean_mask(
        observed_positive_mask,
        expected_shape,
        antibody.device,
        "observed_positive_mask",
    )
    if exclusion_mask is None:
        excluded = torch.zeros_like(positive)
    else:
        excluded = _validate_boolean_mask(
            exclusion_mask,
            expected_shape,
            antibody.device,
            "exclusion_mask",
        )
    if bool((positive & excluded).any().item()):
        raise ValueError("Observed positives cannot be excluded from denominators")
    _validate_replicated_objective_masks(positive, excluded, group=group)

    valid_indices = torch.nonzero(global_valid, as_tuple=False).flatten()
    if int(valid_indices.numel()) == 0:
        raise ValueError("At least one globally valid example is required")
    global_antibody = global_antibody.index_select(0, valid_indices)
    global_antigen = global_antigen.index_select(0, valid_indices)
    positive = positive.index_select(0, valid_indices).index_select(1, valid_indices)
    excluded = excluded.index_select(0, valid_indices).index_select(1, valid_indices)
    calculation_dtype = (
        torch.float64
        if global_antibody.dtype == torch.float64
        else torch.float32
    )
    with torch.autocast(device_type=global_antibody.device.type, enabled=False):
        scores = global_antibody.to(dtype=calculation_dtype) @ global_antigen.to(
            dtype=calculation_dtype
        ).transpose(0, 1)
    details = symmetric_score_matrix_info_nce(
        scores,
        positive,
        excluded,
        temperature=temperature_tensor,
        return_details=True,
    )
    loss = details["loss"]
    if return_details:
        return {
            "loss": loss,
            "antibody_to_antigen": details["antibody_to_antigen"],
            "antigen_to_antibody": details["antigen_to_antibody"],
            "global_antibody_count": int(global_antibody.shape[0]),
            "global_antigen_count": int(global_antigen.shape[0]),
            "global_padded_count": int(global_valid.numel() - valid_indices.numel()),
            "local_slot_count_observed": int(antibody.shape[0]),
            "global_slot_count_observed": int(global_valid.numel()),
            "distributed_world_size_observed": world_size,
            "distributed_rank_observed": rank,
            "gather_gradient_backend": selected_gather_backend,
            "ddp_averaging_contract": (
                "fixed_world_size_standard_ddp_gradient_average"
            ),
            "similarity_dtype": str(scores.dtype),
        }
    return loss
