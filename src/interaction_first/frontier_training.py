"""Deterministic real-data training utilities for frontier models."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

from .frontier_batching import (
    FrontierBatch,
    FrontierBatchCollator,
    FrontierExample,
    canonical_identifier,
)
from .frontier_checkpoint import (
    _restore_verified_checkpoint_payload,
    capture_rng_state,
    create_checkpoint_generation,
    load_checkpoint_generation,
)
from .frontier_data import (
    FoldLocalFrontierDataset,
    RESERVED_PADDING_PREFIX,
)
from .frontier_distributed import (
    LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
    symmetric_multi_positive_info_nce,
)


SAMPLER_FORMAT = "frontier_exact_distributed_batch_sampler_v2"
CURSOR_FORMAT = "frontier_training_cursor_v2"
CHECKPOINT_CONFIG_FORMAT = "frontier_training_checkpoint_config_v2"
MATERIALIZED_PERMUTATION_STRATEGY = (
    "materialized_cpu_int64_epoch_permutation"
)
MATERIALIZED_PERMUTATION_INDEX_BYTES = 8
MAX_MATERIALIZED_PERMUTATION_INDICES = 16_777_216
SAMPLER_SCALE_SCOPE = "finite_authority_train_cache_only"
UNSUPPORTED_SCALE_ACTION = (
    "billion_scale_requires_independently_audited_stateless_block_shuffle_"
    "or_sharded_index"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_torch() -> Any:
    """Import PyTorch lazily so metadata utilities remain importable."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for frontier training") from error
    return torch


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON metadata deterministically."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    """Return a deterministic digest for finite JSON metadata."""
    try:
        payload = _canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Value must be finite and JSON-compatible") from error
    return hashlib.sha256(payload).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    """Return a strictly positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a non-negative non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _epoch_seed(seed: int, epoch: int, dataset_size: int) -> int:
    """Derive a platform-stable 63-bit seed without Python hash randomization."""
    payload = f"{seed}:{epoch}:{dataset_size}:frontier".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _materialized_permutation_dataset_size(value: Any) -> int:
    """Validate the finite-cache size before any permutation can be allocated."""
    dataset_size = _positive_integer(value, "dataset_size")
    if dataset_size > MAX_MATERIALIZED_PERMUTATION_INDICES:
        raise ValueError(
            "dataset_size exceeds the audited materialized-permutation "
            f"limit ({MAX_MATERIALIZED_PERMUTATION_INDICES} indices); "
            "billion-scale pretraining requires an independently audited "
            "stateless/block shuffle or sharded index"
        )
    return dataset_size


@dataclass(frozen=True)
class DistributedSampleReference:
    """One local slot plus the exact rank-major global batch plan."""

    index: int
    valid: bool
    epoch: int
    step: int
    global_position: int
    global_indices: tuple[int, ...]
    global_valid_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Reject malformed or ambiguous sampler metadata at the data boundary."""
        for name, value in (
            ("index", self.index),
            ("epoch", self.epoch),
            ("step", self.step),
            ("global_position", self.global_position),
        ):
            _nonnegative_integer(value, name)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be boolean")
        if (
            not isinstance(self.global_indices, tuple)
            or not self.global_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.global_indices
            )
        ):
            raise ValueError(
                "global_indices must be a nonempty tuple of non-negative integers"
            )
        if (
            not isinstance(self.global_valid_mask, tuple)
            or len(self.global_valid_mask) != len(self.global_indices)
            or any(not isinstance(value, bool) for value in self.global_valid_mask)
        ):
            raise ValueError(
                "global_valid_mask must be a boolean tuple aligned to global_indices"
            )


class ExactDistributedBatchSampler:
    """Shard a finite cached permutation and mask, rather than train on, its tail.

    This sampler deliberately materializes one CPU int64 permutation per rank and
    epoch. It is bounded to finite authority-train caches. Billion-sequence
    pretraining requires a separately audited stateless/block shuffle or sharded
    index implementation.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        local_batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self._dataset_size = _materialized_permutation_dataset_size(dataset_size)
        self.local_batch_size = _positive_integer(
            local_batch_size,
            "local_batch_size",
        )
        self.world_size = _positive_integer(world_size, "world_size")
        self.rank = _nonnegative_integer(rank, "rank")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be boolean")
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0
        self._cached_permutation_epoch: int | None = None
        self._cached_permutation: Any | None = None

    @property
    def dataset_size(self) -> int:
        """Return the immutable finite-cache inventory size."""
        return self._dataset_size

    @property
    def global_batch_size(self) -> int:
        """Return the all-rank batch width."""
        return self.local_batch_size * self.world_size

    def __len__(self) -> int:
        """Return the exact number of synchronized steps in one epoch."""
        return math.ceil(self.dataset_size / self.global_batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic permutation for one epoch."""
        selected_epoch = _nonnegative_integer(epoch, "epoch")
        if selected_epoch != self.epoch:
            self._cached_permutation_epoch = None
            self._cached_permutation = None
        self.epoch = selected_epoch

    def _permutation(self) -> Any:
        """Return one cached CPU int64 permutation shared by all epoch steps."""
        torch = _require_torch()
        if (
            self._cached_permutation_epoch == self.epoch
            and self._cached_permutation is not None
        ):
            return self._cached_permutation
        if self.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _epoch_seed(self.seed, self.epoch, self.dataset_size)
            )
            permutation = torch.randperm(
                self.dataset_size,
                generator=generator,
                dtype=torch.int64,
                device="cpu",
            )
        else:
            permutation = torch.arange(
                self.dataset_size,
                dtype=torch.int64,
                device="cpu",
            )
        self._cached_permutation_epoch = self.epoch
        self._cached_permutation = permutation
        return permutation

    def batch_for_step(self, step: int) -> tuple[DistributedSampleReference, ...]:
        """Return one rank's slots and the shared rank-major global plan."""
        step = _nonnegative_integer(step, "step")
        if step >= len(self):
            raise IndexError("step is outside the sampler epoch")
        permutation = self._permutation()
        global_start = step * self.global_batch_size
        global_indices: list[int] = []
        global_valid: list[bool] = []
        for offset in range(self.global_batch_size):
            position = global_start + offset
            valid = position < self.dataset_size
            source_position = position if valid else position % self.dataset_size
            global_indices.append(int(permutation[source_position].item()))
            global_valid.append(valid)
        global_indices_tuple = tuple(global_indices)
        global_valid_tuple = tuple(global_valid)
        local_start = self.rank * self.local_batch_size
        references = []
        for local_offset in range(self.local_batch_size):
            global_offset = local_start + local_offset
            references.append(
                DistributedSampleReference(
                    index=global_indices_tuple[global_offset],
                    valid=global_valid_tuple[global_offset],
                    epoch=self.epoch,
                    step=step,
                    global_position=global_start + global_offset,
                    global_indices=global_indices_tuple,
                    global_valid_mask=global_valid_tuple,
                )
            )
        return tuple(references)

    def __iter__(self) -> Iterator[list[DistributedSampleReference]]:
        """Yield equal-shape local batches in synchronized step order."""
        for step in range(len(self)):
            yield list(self.batch_for_step(step))

    def contract(self) -> dict[str, Any]:
        """Return the rank-independent exact-resume sampler contract."""
        return {
            "format_version": SAMPLER_FORMAT,
            "dataset_size": self.dataset_size,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "steps_per_epoch": len(self),
            "permutation_strategy": MATERIALIZED_PERMUTATION_STRATEGY,
            "permutation_index_bytes": MATERIALIZED_PERMUTATION_INDEX_BYTES,
            "permutation_memory_bytes_per_rank": (
                self.dataset_size * MATERIALIZED_PERMUTATION_INDEX_BYTES
            ),
            "maximum_materialized_permutation_indices": (
                MAX_MATERIALIZED_PERMUTATION_INDICES
            ),
            "scale_scope": SAMPLER_SCALE_SCOPE,
            "billion_scale_supported": False,
            "unsupported_scale_action": UNSUPPORTED_SCALE_ACTION,
        }

    @property
    def contract_sha256(self) -> str:
        """Return the deterministic rank-independent sampler digest."""
        return _json_sha256(self.contract())

    def state_dict(self) -> dict[str, Any]:
        """Return a rank-specific state for local validation."""
        return {
            **self.contract(),
            "rank": self.rank,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate a saved local state and restore only its epoch."""
        if not isinstance(state, Mapping):
            raise ValueError("sampler state must be a mapping")
        expected_keys = set(self.contract()) | {"rank", "epoch"}
        if set(state) != expected_keys:
            raise ValueError("sampler state fields do not match the locked schema")
        for key, expected in self.contract().items():
            if state[key] != expected:
                raise ValueError(f"sampler state field {key!r} changed")
        if state["rank"] != self.rank:
            raise ValueError("sampler state rank differs from the current process")
        self.set_epoch(state["epoch"])


@dataclass(frozen=True)
class SampledFrontierExample:
    """One loaded slot and its immutable sampler reference."""

    reference: DistributedSampleReference
    example: FrontierExample


@dataclass(frozen=True)
class D0FrontierBatch:
    """A compact padded batch containing only tensors consumed by D0."""

    instance_ids: tuple[str, ...]
    antibody_ids: tuple[str, ...]
    antigen_ids: tuple[str, ...]
    antibody_embeddings: Any
    antigen_embeddings: Any
    antibody_chain_ids: Any
    antibody_lengths: Any
    antigen_lengths: Any
    antibody_mask: Any
    antigen_mask: Any

    def __post_init__(self) -> None:
        """Validate the compact D0 tensor and mask contract."""
        torch = _require_torch()
        batch_size = len(self.instance_ids)
        if batch_size <= 0 or len(set(self.instance_ids)) != batch_size:
            raise ValueError("D0 batch instance_ids must be nonempty and unique")
        if (
            len(self.antibody_ids) != batch_size
            or len(self.antigen_ids) != batch_size
        ):
            raise ValueError("D0 batch identifiers must align to the batch axis")
        for name, values in (
            ("instance_id", self.instance_ids),
            ("antibody_id", self.antibody_ids),
            ("antigen_id", self.antigen_ids),
        ):
            if any(canonical_identifier(value, name) != value for value in values):
                raise ValueError(f"D0 {name} values must use canonical form")
        if (
            self.antibody_embeddings.ndim != 3
            or self.antigen_embeddings.ndim != 3
            or int(self.antibody_embeddings.shape[0]) != batch_size
            or int(self.antigen_embeddings.shape[0]) != batch_size
        ):
            raise ValueError("D0 embeddings must use [batch, residue, feature]")
        if (
            not self.antibody_embeddings.is_floating_point()
            or not self.antigen_embeddings.is_floating_point()
        ):
            raise TypeError("D0 embeddings must use floating-point dtypes")
        if (
            not bool(torch.isfinite(self.antibody_embeddings).all().item())
            or not bool(torch.isfinite(self.antigen_embeddings).all().item())
        ):
            raise ValueError("D0 embeddings must contain only finite values")
        device = self.antibody_embeddings.device
        if self.antigen_embeddings.device != device:
            raise ValueError("D0 embedding streams must share one device")
        maximum_antibody = int(self.antibody_embeddings.shape[1])
        maximum_antigen = int(self.antigen_embeddings.shape[1])
        expected_shapes = {
            "antibody_chain_ids": (batch_size, maximum_antibody),
            "antibody_lengths": (batch_size,),
            "antigen_lengths": (batch_size,),
            "antibody_mask": (batch_size, maximum_antibody),
            "antigen_mask": (batch_size, maximum_antigen),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{name} has an invalid D0 batch shape")
            if value.device != device:
                raise ValueError("Every D0 batch tensor must share one device")
        if (
            self.antibody_chain_ids.dtype != torch.long
            or self.antibody_lengths.dtype != torch.long
            or self.antigen_lengths.dtype != torch.long
        ):
            raise TypeError("D0 chain identifiers and lengths must use int64")
        if (
            self.antibody_mask.dtype != torch.bool
            or self.antigen_mask.dtype != torch.bool
        ):
            raise TypeError("D0 residue masks must use bool")
        expected_antibody_mask = (
            torch.arange(maximum_antibody, device=device).unsqueeze(0)
            < self.antibody_lengths.unsqueeze(1)
        )
        expected_antigen_mask = (
            torch.arange(maximum_antigen, device=device).unsqueeze(0)
            < self.antigen_lengths.unsqueeze(1)
        )
        if not torch.equal(self.antibody_mask, expected_antibody_mask):
            raise ValueError("D0 antibody_mask must be a contiguous prefix mask")
        if not torch.equal(self.antigen_mask, expected_antigen_mask):
            raise ValueError("D0 antigen_mask must be a contiguous prefix mask")
        if bool(
            torch.logical_or(
                self.antibody_lengths <= 0,
                self.antibody_lengths > maximum_antibody,
            ).any().item()
        ):
            raise ValueError("D0 antibody lengths are outside the padded axis")
        if bool(
            torch.logical_or(
                self.antigen_lengths <= 0,
                self.antigen_lengths > maximum_antigen,
            ).any().item()
        ):
            raise ValueError("D0 antigen lengths are outside the padded axis")
        if bool(
            self.antibody_embeddings.masked_select(
                ~self.antibody_mask.unsqueeze(-1)
            ).ne(0).any().item()
        ):
            raise ValueError("Padded D0 antibody embeddings must be zero")
        if bool(
            self.antigen_embeddings.masked_select(
                ~self.antigen_mask.unsqueeze(-1)
            ).ne(0).any().item()
        ):
            raise ValueError("Padded D0 antigen embeddings must be zero")
        valid_chain_ids = self.antibody_chain_ids.masked_select(
            self.antibody_mask
        )
        if not bool(
            torch.logical_or(valid_chain_ids == 0, valid_chain_ids == 1)
            .all()
            .item()
        ):
            raise ValueError("Valid D0 antibody chain IDs must be zero or one")
        if bool(
            self.antibody_chain_ids.masked_select(~self.antibody_mask)
            .ne(-1)
            .any()
            .item()
        ):
            raise ValueError("Padded D0 antibody chain IDs must use -1")
        if any(
            not bool(
                (self.antibody_chain_ids[index][self.antibody_mask[index]] == 0)
                .any()
                .item()
            )
            for index in range(batch_size)
        ):
            raise ValueError("Every D0 antibody item must contain a heavy chain")

    @property
    def batch_size(self) -> int:
        """Return the number of local exact-pair slots."""
        return len(self.instance_ids)

    def to(self, device: Any, non_blocking: bool = False) -> "D0FrontierBatch":
        """Move every compact D0 tensor to one device."""
        tensor_fields = {
            name: getattr(self, name).to(
                device=device,
                non_blocking=non_blocking,
            )
            for name in (
                "antibody_embeddings",
                "antigen_embeddings",
                "antibody_chain_ids",
                "antibody_lengths",
                "antigen_lengths",
                "antibody_mask",
                "antigen_mask",
            )
        }
        return replace(self, **tensor_fields)

    def pin_memory(self) -> "D0FrontierBatch":
        """Pin every compact CPU tensor for asynchronous CUDA transfer."""
        tensor_fields = {
            name: getattr(self, name).pin_memory()
            for name in (
                "antibody_embeddings",
                "antigen_embeddings",
                "antibody_chain_ids",
                "antibody_lengths",
                "antigen_lengths",
                "antibody_mask",
                "antigen_mask",
            )
        }
        return replace(self, **tensor_fields)


class D0FrontierBatchCollator:
    """Pad only the residue tensors required by the D0 dual encoder."""

    def __call__(self, examples: Sequence[FrontierExample]) -> D0FrontierBatch:
        """Build a compact batch without allocating residue-pair tensors."""
        torch = _require_torch()
        if not isinstance(examples, Sequence) or not examples:
            raise ValueError("examples must be a nonempty sequence")
        if any(not isinstance(example, FrontierExample) for example in examples):
            raise TypeError("Every D0 collator item must be a FrontierExample")
        instance_ids = tuple(example.instance_id for example in examples)
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("D0 batch instance IDs must be unique")
        devices = {
            example.antibody_embeddings.device for example in examples
        } | {example.antigen_embeddings.device for example in examples}
        if len(devices) != 1:
            raise ValueError("D0 examples must share one device")
        device = next(iter(devices))
        antibody_dims = {
            int(example.antibody_embeddings.shape[1]) for example in examples
        }
        antigen_dims = {
            int(example.antigen_embeddings.shape[1]) for example in examples
        }
        antibody_dtypes = {
            example.antibody_embeddings.dtype for example in examples
        }
        antigen_dtypes = {
            example.antigen_embeddings.dtype for example in examples
        }
        if (
            len(antibody_dims) != 1
            or len(antigen_dims) != 1
            or len(antibody_dtypes) != 1
            or len(antigen_dtypes) != 1
        ):
            raise ValueError("D0 examples have inconsistent embedding contracts")
        batch_size = len(examples)
        maximum_antibody = max(example.antibody_length for example in examples)
        maximum_antigen = max(example.antigen_length for example in examples)
        antibody_embeddings = torch.zeros(
            batch_size,
            maximum_antibody,
            next(iter(antibody_dims)),
            dtype=next(iter(antibody_dtypes)),
            device=device,
        )
        antigen_embeddings = torch.zeros(
            batch_size,
            maximum_antigen,
            next(iter(antigen_dims)),
            dtype=next(iter(antigen_dtypes)),
            device=device,
        )
        antibody_chain_ids = torch.full(
            (batch_size, maximum_antibody),
            -1,
            dtype=torch.long,
            device=device,
        )
        antibody_lengths = torch.tensor(
            [example.antibody_length for example in examples],
            dtype=torch.long,
            device=device,
        )
        antigen_lengths = torch.tensor(
            [example.antigen_length for example in examples],
            dtype=torch.long,
            device=device,
        )
        for index, example in enumerate(examples):
            antibody_embeddings[index, : example.antibody_length].copy_(
                example.antibody_embeddings
            )
            antigen_embeddings[index, : example.antigen_length].copy_(
                example.antigen_embeddings
            )
            antibody_chain_ids[index, : example.antibody_length].copy_(
                example.antibody_chain_ids
            )
        antibody_mask = (
            torch.arange(maximum_antibody, device=device).unsqueeze(0)
            < antibody_lengths.unsqueeze(1)
        )
        antigen_mask = (
            torch.arange(maximum_antigen, device=device).unsqueeze(0)
            < antigen_lengths.unsqueeze(1)
        )
        return D0FrontierBatch(
            instance_ids=instance_ids,
            antibody_ids=tuple(example.antibody_id for example in examples),
            antigen_ids=tuple(example.antigen_id for example in examples),
            antibody_embeddings=antibody_embeddings,
            antigen_embeddings=antigen_embeddings,
            antibody_chain_ids=antibody_chain_ids,
            antibody_lengths=antibody_lengths,
            antigen_lengths=antigen_lengths,
            antibody_mask=antibody_mask,
            antigen_mask=antigen_mask,
        )


class DistributedTrainingDataset:
    """Adapt a strict fold-local dataset to sampler reference objects."""

    def __init__(self, dataset: FoldLocalFrontierDataset) -> None:
        if not isinstance(dataset, FoldLocalFrontierDataset):
            raise TypeError("dataset must be a FoldLocalFrontierDataset")
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the underlying authorized instance count."""
        return len(self.dataset)

    def __getitem__(
        self,
        reference: DistributedSampleReference,
    ) -> SampledFrontierExample:
        """Load a real example and neutralize any deterministic tail filler."""
        if not isinstance(reference, DistributedSampleReference):
            raise TypeError(
                "DistributedTrainingDataset requires DistributedSampleReference indices"
            )
        example = self.dataset[reference.index]
        if not reference.valid:
            suffix = f"e{reference.epoch}:p{reference.global_position}"
            example = replace(
                example,
                instance_id=f"{RESERVED_PADDING_PREFIX}:instance:{suffix}",
                antibody_id=f"{RESERVED_PADDING_PREFIX}:antibody:{suffix}",
                antigen_id=f"{RESERVED_PADDING_PREFIX}:antigen:{suffix}",
                contact_target=None,
                contact_mask=None,
            )
        return SampledFrontierExample(reference=reference, example=example)


@dataclass(frozen=True)
class LoadedTrainingBatch:
    """A local tensor batch and the exact global denominator masks."""

    frontier: FrontierBatch | D0FrontierBatch
    local_valid_mask: Any
    global_positive_mask: Any
    global_exclusion_mask: Any
    global_indices: tuple[int, ...]
    global_valid_mask: tuple[bool, ...]
    epoch: int
    step: int

    def __post_init__(self) -> None:
        """Validate all local and global mask shapes."""
        torch = _require_torch()
        _nonnegative_integer(self.epoch, "batch.epoch")
        _nonnegative_integer(self.step, "batch.step")
        local_batch = self.frontier.batch_size
        global_batch = len(self.global_indices)
        if global_batch == 0 or len(self.global_valid_mask) != global_batch:
            raise ValueError("Global batch metadata is empty or misaligned")
        if (
            not isinstance(self.global_indices, tuple)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.global_indices
            )
            or not isinstance(self.global_valid_mask, tuple)
            or any(not isinstance(value, bool) for value in self.global_valid_mask)
        ):
            raise ValueError("Global batch indices and validity metadata are malformed")
        valid_indices = [
            index
            for index, valid in zip(self.global_indices, self.global_valid_mask)
            if valid
        ]
        if len(set(valid_indices)) != len(valid_indices):
            raise ValueError("Valid global batch indices must be unique")
        if global_batch % local_batch != 0:
            raise ValueError("Global batch width must be divisible by local batch width")
        if (
            not torch.is_tensor(self.local_valid_mask)
            or self.local_valid_mask.dtype != torch.bool
            or tuple(self.local_valid_mask.shape) != (local_batch,)
        ):
            raise ValueError("local_valid_mask must be boolean with shape [local_batch]")
        for name, value in (
            ("global_positive_mask", self.global_positive_mask),
            ("global_exclusion_mask", self.global_exclusion_mask),
        ):
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.bool
                or tuple(value.shape) != (global_batch, global_batch)
            ):
                raise ValueError(f"{name} must be boolean with shape [global, global]")
        if bool(
            (
                self.global_positive_mask
                & self.global_exclusion_mask
            ).any().item()
        ):
            raise ValueError("A global positive cannot be excluded")
        device = self.frontier.antibody_embeddings.device
        if any(
            value.device != device
            for value in (
                self.local_valid_mask,
                self.global_positive_mask,
                self.global_exclusion_mask,
            )
        ):
            raise ValueError("Training masks and frontier tensors must share one device")
        global_valid_tensor = torch.tensor(
            self.global_valid_mask,
            dtype=torch.bool,
            device=device,
        )
        valid_pair_mask = (
            global_valid_tensor.unsqueeze(1) & global_valid_tensor.unsqueeze(0)
        )
        if bool((self.global_positive_mask & ~valid_pair_mask).any().item()):
            raise ValueError("Padded global slots cannot contain positive graph edges")
        if not bool(self.global_exclusion_mask[~valid_pair_mask].all().item()):
            raise ValueError("Every pair touching a padded global slot must be excluded")

    def to(self, device: Any, non_blocking: bool = False) -> "LoadedTrainingBatch":
        """Move the complete training batch to one device."""
        return replace(
            self,
            frontier=self.frontier.to(device, non_blocking=non_blocking),
            local_valid_mask=self.local_valid_mask.to(
                device=device,
                non_blocking=non_blocking,
            ),
            global_positive_mask=self.global_positive_mask.to(
                device=device,
                non_blocking=non_blocking,
            ),
            global_exclusion_mask=self.global_exclusion_mask.to(
                device=device,
                non_blocking=non_blocking,
            ),
        )

    def pin_memory(self) -> "LoadedTrainingBatch":
        """Pin every CPU tensor so host-to-device copies can be asynchronous."""
        if isinstance(self.frontier, D0FrontierBatch):
            pinned_frontier = self.frontier.pin_memory()
        else:
            tensor_fields = {
                name: getattr(self.frontier, name).pin_memory()
                for name in (
                    "antibody_embeddings",
                    "antigen_embeddings",
                    "antibody_chain_ids",
                    "antibody_lengths",
                    "antigen_lengths",
                    "antibody_mask",
                    "antigen_mask",
                    "pair_mask",
                    "contact_target",
                    "contact_mask",
                    "positive_pair_mask",
                )
            }
            pinned_frontier = replace(self.frontier, **tensor_fields)
        return replace(
            self,
            frontier=pinned_frontier,
            local_valid_mask=self.local_valid_mask.pin_memory(),
            global_positive_mask=self.global_positive_mask.pin_memory(),
            global_exclusion_mask=self.global_exclusion_mask.pin_memory(),
        )


class TrainingBatchCollator:
    """Collate local examples and construct authorization-bound global masks."""

    def __init__(
        self,
        dataset: FoldLocalFrontierDataset,
        *,
        materialize_contact_tensors: bool = True,
    ) -> None:
        if not isinstance(dataset, FoldLocalFrontierDataset):
            raise TypeError("dataset must be a FoldLocalFrontierDataset")
        if not isinstance(materialize_contact_tensors, bool):
            raise TypeError("materialize_contact_tensors must be boolean")
        self._identities = tuple(dataset.identity(index) for index in range(len(dataset)))
        self._positive_pairs = frozenset(dataset.contract.positive_pairs)
        self._excluded_pairs = frozenset(dataset.contract.excluded_pairs)
        self._frontier_collator = (
            FrontierBatchCollator()
            if materialize_contact_tensors
            else D0FrontierBatchCollator()
        )

    def __call__(
        self,
        sampled: Sequence[SampledFrontierExample],
    ) -> LoadedTrainingBatch:
        """Build one synchronized local batch without inferring graph labels."""
        torch = _require_torch()
        if not isinstance(sampled, Sequence) or not sampled:
            raise ValueError("sampled must be a nonempty sequence")
        if any(not isinstance(item, SampledFrontierExample) for item in sampled):
            raise TypeError("Every collator item must be a SampledFrontierExample")
        references = [item.reference for item in sampled]
        first = references[0]
        shared_fields = (
            first.epoch,
            first.step,
            first.global_indices,
            first.global_valid_mask,
        )
        if any(
            (
                reference.epoch,
                reference.step,
                reference.global_indices,
                reference.global_valid_mask,
            )
            != shared_fields
            for reference in references
        ):
            raise ValueError("Local samples do not share one exact global batch plan")
        global_size = len(first.global_indices)
        if global_size == 0 or len(first.global_valid_mask) != global_size:
            raise ValueError("Global sampler metadata is misaligned")
        if any(index >= len(self._identities) for index in first.global_indices):
            raise ValueError("Global sampler metadata contains an out-of-range index")
        valid_global_indices = [
            index
            for index, valid in zip(
                first.global_indices,
                first.global_valid_mask,
            )
            if valid
        ]
        if len(set(valid_global_indices)) != len(valid_global_indices):
            raise ValueError("Valid global sampler indices must be unique within a step")
        local_size = len(references)
        block_start = first.step * global_size
        local_offsets = [
            reference.global_position - block_start for reference in references
        ]
        if (
            global_size % local_size != 0
            or local_offsets != list(
                range(local_offsets[0], local_offsets[0] + local_size)
            )
            or local_offsets[0] < 0
            or local_offsets[-1] >= global_size
            or local_offsets[0] % local_size != 0
        ):
            raise ValueError("Local slots are not a rank-major slice of the global batch")
        for reference, offset in zip(references, local_offsets):
            if (
                reference.index != first.global_indices[offset]
                or reference.valid != first.global_valid_mask[offset]
            ):
                raise ValueError(
                    "Local slot metadata differs from the rank-major global plan"
                )
        for item in sampled:
            reference = item.reference
            example = item.example
            if reference.valid:
                expected_identity = self._identities[reference.index]
                if (
                    example.instance_id != expected_identity.instance_id
                    or example.antibody_id != expected_identity.antibody_id
                    or example.antigen_id != expected_identity.antigen_id
                ):
                    raise ValueError(
                        "Loaded valid example identity differs from its sampler index"
                    )
            else:
                suffix = f"e{reference.epoch}:p{reference.global_position}"
                expected_padding_identity = (
                    f"{RESERVED_PADDING_PREFIX}:instance:{suffix}",
                    f"{RESERVED_PADDING_PREFIX}:antibody:{suffix}",
                    f"{RESERVED_PADDING_PREFIX}:antigen:{suffix}",
                )
                if (
                    example.instance_id,
                    example.antibody_id,
                    example.antigen_id,
                ) != expected_padding_identity:
                    raise ValueError(
                        "Invalid sampler slots must use deterministic neutral identities"
                    )
        identities = [self._identities[index] for index in first.global_indices]
        valid = tuple(bool(value) for value in first.global_valid_mask)
        positive = torch.zeros(global_size, global_size, dtype=torch.bool)
        excluded = torch.zeros_like(positive)
        for antibody_index, antibody_identity in enumerate(identities):
            for antigen_index, antigen_identity in enumerate(identities):
                if not valid[antibody_index] or not valid[antigen_index]:
                    excluded[antibody_index, antigen_index] = True
                    continue
                pair = (
                    antibody_identity.antibody_id,
                    antigen_identity.antigen_id,
                )
                positive[antibody_index, antigen_index] = pair in self._positive_pairs
                excluded[antibody_index, antigen_index] = pair in self._excluded_pairs
        valid_indices = [
            index for index, is_valid in enumerate(valid) if is_valid
        ]
        if any(not bool(positive[index, index]) for index in valid_indices):
            raise ValueError("A valid matched batch slot is absent from the positive graph")
        if bool((positive & excluded).any().item()):
            raise ValueError("Authorized positive and excluded graphs overlap")
        frontier = self._frontier_collator([item.example for item in sampled])
        local_valid = torch.tensor(
            [reference.valid for reference in references],
            dtype=torch.bool,
        )
        return LoadedTrainingBatch(
            frontier=frontier,
            local_valid_mask=local_valid,
            global_positive_mask=positive,
            global_exclusion_mask=excluded,
            global_indices=first.global_indices,
            global_valid_mask=valid,
            epoch=first.epoch,
            step=first.step,
        )


def create_training_dataloader(
    dataset: FoldLocalFrontierDataset,
    sampler: ExactDistributedBatchSampler,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    materialize_contact_tensors: bool = True,
) -> Any:
    """Create a deterministic loader whose sampler owns all sharding."""
    torch = _require_torch()
    num_workers = _nonnegative_integer(num_workers, "num_workers")
    prefetch_factor = _positive_integer(prefetch_factor, "prefetch_factor")
    if sampler.dataset_size != len(dataset):
        raise ValueError("sampler dataset_size differs from the strict dataset")
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be boolean")
    if not isinstance(materialize_contact_tensors, bool):
        raise ValueError("materialize_contact_tensors must be boolean")
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if not isinstance(persistent_workers, bool):
        raise ValueError("persistent_workers must be boolean")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires at least one worker")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_epoch_seed(sampler.seed, 0, len(dataset)))
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(
        DistributedTrainingDataset(dataset),
        batch_sampler=sampler,
        collate_fn=TrainingBatchCollator(
            dataset,
            materialize_contact_tensors=materialize_contact_tensors,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
        **kwargs,
    )


def d0_forward_loss(
    model: Any,
    batch: LoadedTrainingBatch,
    *,
    group: Any = None,
    gather_gradient_backend: str = LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
) -> dict[str, Any]:
    """Run D0 and compute a tail-safe global FP32 contrastive objective."""
    output = model(
        batch.frontier.antibody_embeddings,
        batch.frontier.antigen_embeddings,
        batch.frontier.antibody_mask,
        batch.frontier.antigen_mask,
        batch.frontier.antibody_chain_ids,
    )
    required = {"antibody_embeddings", "antigen_embeddings", "temperature"}
    if not isinstance(output, Mapping) or not required.issubset(output):
        raise ValueError("D0 model output is missing required fields")
    details = symmetric_multi_positive_info_nce(
        output["antibody_embeddings"],
        output["antigen_embeddings"],
        batch.global_positive_mask,
        batch.global_exclusion_mask,
        temperature=output["temperature"],
        local_valid_mask=batch.local_valid_mask,
        group=group,
        gather_gradient_backend=gather_gradient_backend,
        return_details=True,
    )
    return {
        **details,
        "temperature": output["temperature"],
    }


@dataclass(frozen=True)
class TrainingCursor:
    """An exact position in the deterministic sampler stream."""

    epoch: int
    next_step: int
    global_step: int
    dataset_manifest_sha256: str
    authorization_manifest_sha256: str
    sampler_contract_sha256: str
    sampler_world_size: int
    sampler_steps_per_epoch: int

    def __post_init__(self) -> None:
        """Validate cursor scalar types and every externally bound digest."""
        _nonnegative_integer(self.epoch, "cursor.epoch")
        _nonnegative_integer(self.next_step, "cursor.next_step")
        _nonnegative_integer(self.global_step, "cursor.global_step")
        _positive_integer(self.sampler_world_size, "cursor.sampler_world_size")
        _positive_integer(
            self.sampler_steps_per_epoch,
            "cursor.sampler_steps_per_epoch",
        )
        for name, value in (
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            (
                "authorization_manifest_sha256",
                self.authorization_manifest_sha256,
            ),
            ("sampler_contract_sha256", self.sampler_contract_sha256),
        ):
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"cursor.{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        """Return the locked checkpoint representation."""
        return {
            "format_version": CURSOR_FORMAT,
            "epoch": self.epoch,
            "next_step": self.next_step,
            "global_step": self.global_step,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "authorization_manifest_sha256": self.authorization_manifest_sha256,
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "sampler_world_size": self.sampler_world_size,
            "sampler_steps_per_epoch": self.sampler_steps_per_epoch,
        }


def make_training_cursor(
    dataset: FoldLocalFrontierDataset,
    sampler: ExactDistributedBatchSampler,
    *,
    epoch: int,
    next_step: int,
    global_step: int,
) -> TrainingCursor:
    """Create a normalized cursor, advancing completed epochs to step zero."""
    epoch = _nonnegative_integer(epoch, "epoch")
    next_step = _nonnegative_integer(next_step, "next_step")
    global_step = _nonnegative_integer(global_step, "global_step")
    if sampler.dataset_size != len(dataset):
        raise ValueError("sampler dataset_size differs from the strict dataset")
    if sampler.epoch != epoch:
        raise ValueError("sampler epoch must match the cursor epoch")
    if next_step > len(sampler):
        raise ValueError("next_step exceeds the number of sampler steps")
    expected_global_step = epoch * len(sampler) + next_step
    if global_step != expected_global_step:
        raise ValueError(
            "global_step must equal completed sampler steps without skips or replays"
        )
    if next_step == len(sampler):
        epoch += 1
        next_step = 0
    return TrainingCursor(
        epoch=epoch,
        next_step=next_step,
        global_step=global_step,
        dataset_manifest_sha256=dataset.contract.manifest_sha256,
        authorization_manifest_sha256=(
            dataset.contract.authorization_manifest_sha256
        ),
        sampler_contract_sha256=sampler.contract_sha256,
        sampler_world_size=sampler.world_size,
        sampler_steps_per_epoch=len(sampler),
    )


def build_training_checkpoint_config(
    base_config: Mapping[str, Any],
    cursor: TrainingCursor,
) -> dict[str, Any]:
    """Bind immutable run configuration to an exact deterministic cursor."""
    if not isinstance(base_config, Mapping):
        raise ValueError("base_config must be a mapping")
    canonical_base = json.loads(_canonical_json_bytes(dict(base_config)))
    if not isinstance(canonical_base, dict):
        raise ValueError("base_config must be a JSON object")
    return {
        "format_version": CHECKPOINT_CONFIG_FORMAT,
        "base_config": canonical_base,
        "base_config_sha256": _json_sha256(canonical_base),
        "training_cursor": cursor.to_dict(),
    }


def training_cursor_from_checkpoint_config(
    checkpoint_config: Mapping[str, Any],
    *,
    expected_base_config: Mapping[str, Any],
    dataset: FoldLocalFrontierDataset,
    sampler: ExactDistributedBatchSampler,
) -> TrainingCursor:
    """Verify a checkpoint training contract and return its resume cursor."""
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint_config must be a mapping")
    expected_fields = {
        "format_version",
        "base_config",
        "base_config_sha256",
        "training_cursor",
    }
    if set(checkpoint_config) != expected_fields:
        raise ValueError("checkpoint training config fields differ from the schema")
    if checkpoint_config["format_version"] != CHECKPOINT_CONFIG_FORMAT:
        raise ValueError("Unsupported checkpoint training config format")
    expected_base_digest = _json_sha256(dict(expected_base_config))
    if (
        checkpoint_config["base_config_sha256"] != expected_base_digest
        or _json_sha256(checkpoint_config["base_config"]) != expected_base_digest
    ):
        raise ValueError("Checkpoint base configuration differs from this run")
    raw_cursor = checkpoint_config["training_cursor"]
    if not isinstance(raw_cursor, Mapping):
        raise ValueError("training_cursor must be a mapping")
    expected_cursor_fields = {
        "format_version",
        "epoch",
        "next_step",
        "global_step",
        "dataset_manifest_sha256",
        "authorization_manifest_sha256",
        "sampler_contract_sha256",
        "sampler_world_size",
        "sampler_steps_per_epoch",
    }
    if set(raw_cursor) != expected_cursor_fields:
        raise ValueError("training_cursor fields differ from the locked schema")
    if raw_cursor["format_version"] != CURSOR_FORMAT:
        raise ValueError("Unsupported training cursor format")
    cursor = TrainingCursor(
        epoch=_nonnegative_integer(raw_cursor["epoch"], "cursor.epoch"),
        next_step=_nonnegative_integer(
            raw_cursor["next_step"],
            "cursor.next_step",
        ),
        global_step=_nonnegative_integer(
            raw_cursor["global_step"],
            "cursor.global_step",
        ),
        dataset_manifest_sha256=str(raw_cursor["dataset_manifest_sha256"]),
        authorization_manifest_sha256=str(
            raw_cursor["authorization_manifest_sha256"]
        ),
        sampler_contract_sha256=str(raw_cursor["sampler_contract_sha256"]),
        sampler_world_size=_positive_integer(
            raw_cursor["sampler_world_size"],
            "cursor.sampler_world_size",
        ),
        sampler_steps_per_epoch=_positive_integer(
            raw_cursor["sampler_steps_per_epoch"],
            "cursor.sampler_steps_per_epoch",
        ),
    )
    expected_hashes = (
        (
            cursor.dataset_manifest_sha256,
            dataset.contract.manifest_sha256,
            "dataset manifest",
        ),
        (
            cursor.authorization_manifest_sha256,
            dataset.contract.authorization_manifest_sha256,
            "authorization manifest",
        ),
        (
            cursor.sampler_contract_sha256,
            sampler.contract_sha256,
            "sampler contract",
        ),
    )
    for observed, expected, name in expected_hashes:
        if observed != expected:
            raise ValueError(f"Checkpoint {name} differs from this run")
    if cursor.next_step >= len(sampler) and cursor.next_step != 0:
        raise ValueError("Checkpoint cursor next_step is outside the sampler epoch")
    if (
        cursor.sampler_world_size != sampler.world_size
        or cursor.sampler_steps_per_epoch != len(sampler)
    ):
        raise ValueError("Checkpoint cursor runtime geometry differs from the sampler")
    expected_global_step = cursor.epoch * len(sampler) + cursor.next_step
    if cursor.global_step != expected_global_step:
        raise ValueError(
            "Checkpoint cursor global_step implies skipped or replayed sampler steps"
        )
    return cursor


def gather_rank_rng_states(group: Any = None) -> list[Mapping[str, Any]]:
    """Capture RNG state on every rank in deterministic rank order."""
    torch = _require_torch()
    local_state = capture_rng_state()
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return [local_state]
    world_size = int(distributed.get_world_size(group=group))
    gathered: list[Any] = [None for _ in range(world_size)]
    distributed.all_gather_object(gathered, local_state, group=group)
    if any(not isinstance(value, Mapping) for value in gathered):
        raise RuntimeError("Distributed RNG gather returned an invalid rank state")
    return gathered


def save_training_checkpoint(
    root: Path,
    generation: str,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    base_config: Mapping[str, Any],
    cursor: TrainingCursor,
    provenance_hashes: Mapping[str, str],
    scaler: Any | None = None,
    group: Any = None,
) -> dict[str, Any] | None:
    """Collect all RNG states and publish one immutable rank-zero generation."""
    torch = _require_torch()
    rng_states = gather_rank_rng_states(group=group)
    distributed = torch.distributed
    initialized = distributed.is_available() and distributed.is_initialized()
    rank = int(distributed.get_rank(group=group)) if initialized else 0
    world_size = (
        int(distributed.get_world_size(group=group)) if initialized else 1
    )
    if world_size != cursor.sampler_world_size:
        raise ValueError(
            "Observed distributed world size differs from the checkpoint cursor"
        )
    if len(rng_states) != world_size:
        raise RuntimeError("RNG state count differs from the observed world size")
    receipt = None
    publication_error: Exception | None = None
    if rank == 0:
        try:
            receipt = create_checkpoint_generation(
                root,
                generation,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=cursor.epoch,
                global_step=cursor.global_step,
                provenance_hashes=provenance_hashes,
                config=build_training_checkpoint_config(base_config, cursor),
                rng_states=rng_states,
            )
        except Exception as error:
            publication_error = error
    if initialized:
        status: list[Any] = [
            (
                {"ok": True}
                if publication_error is None
                else {
                    "ok": False,
                    "error_type": type(publication_error).__name__,
                    "message": str(publication_error),
                }
            )
            if rank == 0
            else None
        ]
        distributed.broadcast_object_list(status, src=0, group=group)
        if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
            raise RuntimeError(
                "Rank-zero checkpoint publication failed collectively: "
                f"{status[0]}"
            )
    elif publication_error is not None:
        raise publication_error
    return receipt


def restore_training_checkpoint(
    root: Path,
    generation: str,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    expected_base_config: Mapping[str, Any],
    dataset: FoldLocalFrontierDataset,
    sampler: ExactDistributedBatchSampler,
    provenance_hashes: Mapping[str, str],
    map_location: Any,
    rank: int,
    scaler: Any | None = None,
    expected_receipt_sha256: str | None = None,
    group: Any = None,
) -> tuple[TrainingCursor, dict[str, Any]]:
    """Restore components, rank RNG state, and the exact sampler cursor."""
    torch = _require_torch()
    distributed = torch.distributed
    initialized = distributed.is_available() and distributed.is_initialized()
    observed_rank = int(distributed.get_rank(group=group)) if initialized else 0
    observed_world_size = (
        int(distributed.get_world_size(group=group)) if initialized else 1
    )
    if rank != observed_rank:
        raise ValueError("Requested checkpoint RNG rank differs from the runtime rank")
    if sampler.rank != observed_rank or sampler.world_size != observed_world_size:
        raise ValueError("Sampler rank/world size differs from the observed runtime")
    preflight = load_checkpoint_generation(
        root,
        generation,
        map_location=map_location,
        expected_provenance_hashes=provenance_hashes,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    cursor = training_cursor_from_checkpoint_config(
        preflight["config"],
        expected_base_config=expected_base_config,
        dataset=dataset,
        sampler=sampler,
    )
    if preflight["global_step"] != cursor.global_step:
        raise ValueError("Checkpoint payload and cursor global_step values differ")
    if preflight["epoch"] != cursor.epoch:
        raise ValueError("Checkpoint payload and cursor epoch values differ")
    if len(preflight["rng_states"]) != observed_world_size:
        raise ValueError("Checkpoint RNG world size differs from the observed runtime")
    metadata = _restore_verified_checkpoint_payload(
        preflight,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_rng_world_size=observed_world_size,
        restore_random_generators=True,
        rng_rank=rank,
    )
    if metadata["rng_world_size"] != cursor.sampler_world_size:
        raise ValueError("Checkpoint RNG world size differs from the sampler cursor")
    sampler.set_epoch(cursor.epoch)
    return cursor, metadata


def parameter_state_sha256(module: Any) -> str:
    """Hash every named model tensor in deterministic order."""
    torch = _require_torch()
    if module is None or not hasattr(module, "state_dict"):
        raise TypeError("module must provide state_dict")
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise ValueError("Model state must contain tensors only")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_parameter_state_agreement(module: Any, group: Any = None) -> str:
    """Fail collectively unless every rank constructed bit-identical state."""
    torch = _require_torch()
    local_digest = parameter_state_sha256(module)
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return local_digest
    world_size = int(distributed.get_world_size(group=group))
    gathered: list[Any] = [None for _ in range(world_size)]
    distributed.all_gather_object(gathered, local_digest, group=group)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Initial model states differ across ranks: {gathered}")
    return local_digest
