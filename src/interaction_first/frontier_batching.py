"""Fail-closed batching primitives for partner-specific frontier models.

The contact axes in this module are always ordered as ``[antigen, antibody]``.
No file or dataset access occurs here; callers must provide already-authorized
examples and, when needed, an authorized positive-pair graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence
import unicodedata

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch() -> None:
    """Raise a focused error when PyTorch is unavailable."""
    if torch is None:
        raise RuntimeError("PyTorch is required for frontier batching")


def canonical_identifier(value: str, field_name: str = "identifier") -> str:
    """Return a stable identifier while rejecting ambiguous input."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain leading or trailing whitespace")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError(f"{field_name} cannot contain control or format characters")
    return normalized


def _validate_embedding(value: Any, field_name: str) -> None:
    """Validate a dense, finite, nonempty residue-embedding matrix."""
    _require_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{field_name} must be a dense strided tensor")
    if value.ndim != 2:
        raise ValueError(f"{field_name} must have rank 2 [residue, feature]")
    if int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0:
        raise ValueError(f"{field_name} must have nonzero residue and feature dimensions")
    if not value.is_floating_point():
        raise TypeError(f"{field_name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{field_name} must contain only finite values")


def _validate_contact_supervision(
    target: Any,
    mask: Any,
    expected_shape: tuple[int, int],
    expected_device: Any,
) -> None:
    """Validate labels before any floating-point conversion occurs."""
    _require_torch()
    if (target is None) != (mask is None):
        raise ValueError("contact_target and contact_mask must be provided together")
    if target is None:
        return
    if not isinstance(target, torch.Tensor):
        raise TypeError("contact_target must be a torch.Tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("contact_mask must be a torch.Tensor")
    if target.layout != torch.strided or mask.layout != torch.strided:
        raise ValueError("Contact supervision must use dense strided tensors")
    if target.ndim != 2 or tuple(target.shape) != expected_shape:
        raise ValueError(
            "contact_target must have shape [antigen_length, antibody_length]"
        )
    if mask.ndim != 2 or tuple(mask.shape) != expected_shape:
        raise ValueError(
            "contact_mask must have shape [antigen_length, antibody_length]"
        )
    if mask.dtype != torch.bool:
        raise TypeError("contact_mask must use torch.bool")
    if target.device != expected_device or mask.device != expected_device:
        raise ValueError("Contact supervision and embeddings must share one device")
    if target.is_complex():
        raise TypeError("contact_target cannot use a complex dtype")
    if target.is_floating_point() and not bool(torch.isfinite(target).all().item()):
        raise ValueError("contact_target must contain only finite values")
    is_binary = torch.logical_or(target == 0, target == 1)
    if not bool(is_binary.all().item()):
        raise ValueError("contact_target must contain exact binary values")
    if not bool(mask.any().item()):
        raise ValueError(
            "A provided contact_target requires at least one resolved contact_mask entry"
        )
    hidden_values = torch.logical_and(~mask, target != 0)
    if bool(hidden_values.any().item()):
        raise ValueError(
            "Unresolved contact entries must use zero placeholders, not hidden labels"
        )


@dataclass(frozen=True)
class FrontierExample:
    """One authorized matched complex with optional native-contact supervision."""

    instance_id: str
    antibody_id: str
    antigen_id: str
    antibody_embeddings: Any
    antigen_embeddings: Any
    antibody_chain_ids: Any
    contact_target: Any | None = None
    contact_mask: Any | None = None

    def __post_init__(self) -> None:
        """Canonicalize identifiers and validate all tensor boundaries."""
        instance_id = canonical_identifier(self.instance_id, "instance_id")
        antibody_id = canonical_identifier(self.antibody_id, "antibody_id")
        antigen_id = canonical_identifier(self.antigen_id, "antigen_id")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "antibody_id", antibody_id)
        object.__setattr__(self, "antigen_id", antigen_id)

        _validate_embedding(self.antibody_embeddings, "antibody_embeddings")
        _validate_embedding(self.antigen_embeddings, "antigen_embeddings")
        if self.antibody_embeddings.device != self.antigen_embeddings.device:
            raise ValueError("Antibody and antigen embeddings must share one device")
        if not isinstance(self.antibody_chain_ids, torch.Tensor):
            raise TypeError("antibody_chain_ids must be a torch.Tensor")
        if self.antibody_chain_ids.layout != torch.strided:
            raise ValueError("antibody_chain_ids must be a dense strided tensor")
        if self.antibody_chain_ids.ndim != 1 or tuple(
            self.antibody_chain_ids.shape
        ) != (self.antibody_length,):
            raise ValueError(
                "antibody_chain_ids must align exactly to the antibody residue axis"
            )
        if self.antibody_chain_ids.dtype != torch.long:
            raise TypeError("antibody_chain_ids must use torch.int64")
        if self.antibody_chain_ids.device != self.antibody_embeddings.device:
            raise ValueError("antibody_chain_ids and embeddings must share one device")
        valid_chain_ids = torch.logical_or(
            self.antibody_chain_ids == 0, self.antibody_chain_ids == 1
        )
        if not bool(valid_chain_ids.all().item()):
            raise ValueError(
                "Every antibody residue must use heavy-chain ID 0 or light-chain ID 1"
            )
        if not bool((self.antibody_chain_ids == 0).any().item()):
            raise ValueError("Every antibody example must contain a heavy chain")
        _validate_contact_supervision(
            self.contact_target,
            self.contact_mask,
            (self.antigen_length, self.antibody_length),
            self.antigen_embeddings.device,
        )

    @property
    def antibody_length(self) -> int:
        """Return the number of antibody residues."""
        return int(self.antibody_embeddings.shape[0])

    @property
    def antigen_length(self) -> int:
        """Return the number of antigen residues."""
        return int(self.antigen_embeddings.shape[0])


@dataclass(frozen=True)
class FrontierBatch:
    """A padded batch whose masks make every valid training cell explicit."""

    instance_ids: tuple[str, ...]
    antibody_ids: tuple[str, ...]
    antigen_ids: tuple[str, ...]
    positive_pairs: tuple[tuple[str, str], ...]
    antibody_embeddings: Any
    antigen_embeddings: Any
    antibody_chain_ids: Any
    antibody_lengths: Any
    antigen_lengths: Any
    antibody_mask: Any
    antigen_mask: Any
    pair_mask: Any
    contact_target: Any
    contact_mask: Any
    positive_pair_mask: Any

    def __post_init__(self) -> None:
        """Verify batch invariants so manually constructed batches also fail closed."""
        _require_torch()
        batch_size = len(self.instance_ids)
        if batch_size == 0:
            raise ValueError("A frontier batch cannot be empty")
        if len(set(self.instance_ids)) != batch_size:
            raise ValueError("instance_ids must be unique within a batch")
        if len(self.antibody_ids) != batch_size or len(self.antigen_ids) != batch_size:
            raise ValueError("Identifier tuples must match the batch dimension")
        canonical_instances = tuple(
            canonical_identifier(value, "instance_id") for value in self.instance_ids
        )
        canonical_antibodies = tuple(
            canonical_identifier(value, "antibody_id") for value in self.antibody_ids
        )
        canonical_antigens = tuple(
            canonical_identifier(value, "antigen_id") for value in self.antigen_ids
        )
        if canonical_instances != self.instance_ids:
            raise ValueError("instance_ids are not in canonical form")
        if canonical_antibodies != self.antibody_ids:
            raise ValueError("antibody_ids are not in canonical form")
        if canonical_antigens != self.antigen_ids:
            raise ValueError("antigen_ids are not in canonical form")

        if self.antibody_embeddings.ndim != 3 or self.antigen_embeddings.ndim != 3:
            raise ValueError("Padded embeddings must have rank 3")
        if int(self.antibody_embeddings.shape[0]) != batch_size:
            raise ValueError("antibody_embeddings has an invalid batch dimension")
        if int(self.antigen_embeddings.shape[0]) != batch_size:
            raise ValueError("antigen_embeddings has an invalid batch dimension")
        if not self.antibody_embeddings.is_floating_point():
            raise TypeError("antibody_embeddings must use a floating-point dtype")
        if not self.antigen_embeddings.is_floating_point():
            raise TypeError("antigen_embeddings must use a floating-point dtype")

        device = self.antigen_embeddings.device
        tensors = (
            self.antibody_embeddings,
            self.antibody_chain_ids,
            self.antibody_lengths,
            self.antigen_lengths,
            self.antibody_mask,
            self.antigen_mask,
            self.pair_mask,
            self.contact_target,
            self.contact_mask,
            self.positive_pair_mask,
        )
        if any(value.device != device for value in tensors):
            raise ValueError("All padded batch tensors must share one device")
        if not bool(torch.isfinite(self.antibody_embeddings).all().item()):
            raise ValueError("antibody_embeddings must contain only finite values")
        if not bool(torch.isfinite(self.antigen_embeddings).all().item()):
            raise ValueError("antigen_embeddings must contain only finite values")

        max_antibody = int(self.antibody_embeddings.shape[1])
        max_antigen = int(self.antigen_embeddings.shape[1])
        if tuple(self.antibody_lengths.shape) != (batch_size,):
            raise ValueError("antibody_lengths must have shape [batch]")
        if tuple(self.antigen_lengths.shape) != (batch_size,):
            raise ValueError("antigen_lengths must have shape [batch]")
        if (
            self.antibody_lengths.dtype != torch.long
            or self.antigen_lengths.dtype != torch.long
        ):
            raise TypeError("Residue lengths must use torch.long")
        if bool(
            torch.logical_or(
                self.antibody_lengths <= 0, self.antibody_lengths > max_antibody
            ).any().item()
        ):
            raise ValueError("antibody_lengths contains an invalid value")
        if bool(
            torch.logical_or(
                self.antigen_lengths <= 0, self.antigen_lengths > max_antigen
            ).any().item()
        ):
            raise ValueError("antigen_lengths contains an invalid value")

        if self.antibody_mask.dtype != torch.bool or tuple(
            self.antibody_mask.shape
        ) != (batch_size, max_antibody):
            raise ValueError("antibody_mask must be boolean with shape [batch, residue]")
        if self.antigen_mask.dtype != torch.bool or tuple(self.antigen_mask.shape) != (
            batch_size,
            max_antigen,
        ):
            raise ValueError("antigen_mask must be boolean with shape [batch, residue]")
        expected_antibody_mask = (
            torch.arange(max_antibody, device=device).unsqueeze(0)
            < self.antibody_lengths.unsqueeze(1)
        )
        expected_antigen_mask = (
            torch.arange(max_antigen, device=device).unsqueeze(0)
            < self.antigen_lengths.unsqueeze(1)
        )
        if not torch.equal(self.antibody_mask, expected_antibody_mask):
            raise ValueError("antibody_mask must be a contiguous prefix mask")
        if not torch.equal(self.antigen_mask, expected_antigen_mask):
            raise ValueError("antigen_mask must be a contiguous prefix mask")
        if bool(
            self.antibody_embeddings.masked_select(
                ~self.antibody_mask.unsqueeze(-1)
            ).ne(0).any().item()
        ):
            raise ValueError("Padded antibody embeddings must be zero")
        if bool(
            self.antigen_embeddings.masked_select(
                ~self.antigen_mask.unsqueeze(-1)
            ).ne(0).any().item()
        ):
            raise ValueError("Padded antigen embeddings must be zero")
        if self.antibody_chain_ids.dtype != torch.long or tuple(
            self.antibody_chain_ids.shape
        ) != (batch_size, max_antibody):
            raise ValueError(
                "antibody_chain_ids must use int64 with shape [batch, antibody]"
            )
        valid_chain_entries = self.antibody_chain_ids.masked_select(self.antibody_mask)
        if not bool(
            torch.logical_or(valid_chain_entries == 0, valid_chain_entries == 1)
            .all()
            .item()
        ):
            raise ValueError(
                "Valid antibody residues must use heavy-chain ID 0 or light-chain ID 1"
            )
        if bool(
            self.antibody_chain_ids.masked_select(~self.antibody_mask).ne(-1).any().item()
        ):
            raise ValueError("Padded antibody_chain_ids must use -1")
        if not bool(
            torch.stack(
                [
                    (self.antibody_chain_ids[index][self.antibody_mask[index]] == 0).any()
                    for index in range(batch_size)
                ]
            )
            .all()
            .item()
        ):
            raise ValueError("Every antibody example must contain a heavy chain")

        pair_shape = (batch_size, max_antigen, max_antibody)
        expected_pair_mask = torch.logical_and(
            self.antigen_mask.unsqueeze(2), self.antibody_mask.unsqueeze(1)
        )
        if self.pair_mask.dtype != torch.bool or tuple(self.pair_mask.shape) != pair_shape:
            raise ValueError("pair_mask must be boolean with shape [batch, antigen, antibody]")
        if not torch.equal(self.pair_mask, expected_pair_mask):
            raise ValueError("pair_mask must be the outer product of the residue masks")
        if self.contact_mask.dtype != torch.bool or tuple(
            self.contact_mask.shape
        ) != pair_shape:
            raise ValueError(
                "contact_mask must be boolean with shape [batch, antigen, antibody]"
            )
        if bool(torch.logical_and(self.contact_mask, ~self.pair_mask).any().item()):
            raise ValueError("contact_mask cannot include padded residue pairs")
        if tuple(self.contact_target.shape) != pair_shape:
            raise ValueError("contact_target must have shape [batch, antigen, antibody]")
        if not self.contact_target.is_floating_point():
            raise TypeError("Padded contact_target must use a floating-point dtype")
        if not bool(torch.isfinite(self.contact_target).all().item()):
            raise ValueError("Padded contact_target must contain only finite values")
        if not bool(
            torch.logical_or(self.contact_target == 0, self.contact_target == 1)
            .all()
            .item()
        ):
            raise ValueError("Padded contact_target must contain exact binary values")
        if bool(
            torch.logical_and(~self.contact_mask, self.contact_target != 0).any().item()
        ):
            raise ValueError("Unresolved or padded contact targets must be zero")

        graph = frozenset(self.positive_pairs)
        if len(graph) != len(self.positive_pairs):
            raise ValueError("positive_pairs cannot contain duplicate edges")
        for antibody_id, antigen_id in self.positive_pairs:
            if (
                canonical_identifier(antibody_id, "positive antibody_id") != antibody_id
                or canonical_identifier(antigen_id, "positive antigen_id") != antigen_id
            ):
                raise ValueError("positive_pairs must use canonical identifiers")
        expected_positive_mask = torch.tensor(
            [
                [
                    (antibody_id, antigen_id) in graph
                    for antigen_id in self.antigen_ids
                ]
                for antibody_id in self.antibody_ids
            ],
            dtype=torch.bool,
            device=device,
        )
        if self.positive_pair_mask.dtype != torch.bool or tuple(
            self.positive_pair_mask.shape
        ) != (batch_size, batch_size):
            raise ValueError("positive_pair_mask must be boolean with shape [batch, batch]")
        if not torch.equal(self.positive_pair_mask, expected_positive_mask):
            raise ValueError("positive_pair_mask does not match the authorized graph")
        if not bool(torch.diagonal(self.positive_pair_mask).all().item()):
            raise ValueError("Every matched batch example must be an authorized positive")

    @property
    def batch_size(self) -> int:
        """Return the number of matched complexes."""
        return len(self.instance_ids)

    def to(self, device: Any, non_blocking: bool = False) -> "FrontierBatch":
        """Move all tensors together while preserving immutable metadata."""
        tensor_fields = {
            field_name: getattr(self, field_name).to(
                device=device, non_blocking=non_blocking
            )
            for field_name in (
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
        return replace(self, **tensor_fields)


class FrontierBatchCollator:
    """Pad authorized examples without inferring missing labels or graph edges."""

    def __init__(
        self,
        authorized_positive_pairs: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        self._explicit_positive_graph = authorized_positive_pairs is not None
        if authorized_positive_pairs is None:
            self._positive_pairs: tuple[tuple[str, str], ...] = ()
            return
        canonical_pairs: list[tuple[str, str]] = []
        for index, pair in enumerate(authorized_positive_pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    f"authorized_positive_pairs[{index}] must contain exactly two IDs"
                )
            canonical_pairs.append(
                (
                    canonical_identifier(pair[0], "positive antibody_id"),
                    canonical_identifier(pair[1], "positive antigen_id"),
                )
            )
        if len(set(canonical_pairs)) != len(canonical_pairs):
            raise ValueError("authorized_positive_pairs cannot contain duplicate edges")
        self._positive_pairs = tuple(sorted(canonical_pairs))

    def __call__(self, examples: Sequence[FrontierExample]) -> FrontierBatch:
        """Validate and pad a nonempty sequence of matched examples."""
        _require_torch()
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            raise TypeError("examples must be a sequence of FrontierExample objects")
        if not examples:
            raise ValueError("Cannot collate an empty example sequence")
        if any(not isinstance(example, FrontierExample) for example in examples):
            raise TypeError("Every item must be a FrontierExample")

        instance_ids = tuple(example.instance_id for example in examples)
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("instance_id values must be unique within a batch")
        devices = {
            example.antigen_embeddings.device for example in examples
        } | {example.antibody_embeddings.device for example in examples}
        if len(devices) != 1:
            raise ValueError("All examples in a batch must share one device")
        device = next(iter(devices))

        antigen_dims = {int(example.antigen_embeddings.shape[1]) for example in examples}
        antibody_dims = {
            int(example.antibody_embeddings.shape[1]) for example in examples
        }
        antigen_dtypes = {example.antigen_embeddings.dtype for example in examples}
        antibody_dtypes = {example.antibody_embeddings.dtype for example in examples}
        if len(antigen_dims) != 1 or len(antibody_dims) != 1:
            raise ValueError("All examples must use consistent per-stream feature dimensions")
        if len(antigen_dtypes) != 1 or len(antibody_dtypes) != 1:
            raise ValueError("All examples must use consistent per-stream embedding dtypes")

        batch_size = len(examples)
        max_antigen = max(example.antigen_length for example in examples)
        max_antibody = max(example.antibody_length for example in examples)
        antigen_dim = next(iter(antigen_dims))
        antibody_dim = next(iter(antibody_dims))
        antigen_dtype = next(iter(antigen_dtypes))
        antibody_dtype = next(iter(antibody_dtypes))

        antigen_embeddings = torch.zeros(
            batch_size,
            max_antigen,
            antigen_dim,
            dtype=antigen_dtype,
            device=device,
        )
        antibody_embeddings = torch.zeros(
            batch_size,
            max_antibody,
            antibody_dim,
            dtype=antibody_dtype,
            device=device,
        )
        antibody_chain_ids = torch.full(
            (batch_size, max_antibody),
            -1,
            dtype=torch.long,
            device=device,
        )
        antigen_lengths = torch.tensor(
            [example.antigen_length for example in examples],
            dtype=torch.long,
            device=device,
        )
        antibody_lengths = torch.tensor(
            [example.antibody_length for example in examples],
            dtype=torch.long,
            device=device,
        )
        antigen_mask = (
            torch.arange(max_antigen, device=device).unsqueeze(0)
            < antigen_lengths.unsqueeze(1)
        )
        antibody_mask = (
            torch.arange(max_antibody, device=device).unsqueeze(0)
            < antibody_lengths.unsqueeze(1)
        )
        pair_mask = torch.logical_and(
            antigen_mask.unsqueeze(2), antibody_mask.unsqueeze(1)
        )
        contact_target = torch.zeros(
            batch_size,
            max_antigen,
            max_antibody,
            dtype=torch.float32,
            device=device,
        )
        contact_mask = torch.zeros_like(contact_target, dtype=torch.bool)

        for index, example in enumerate(examples):
            antigen_embeddings[index, : example.antigen_length].copy_(
                example.antigen_embeddings
            )
            antibody_embeddings[index, : example.antibody_length].copy_(
                example.antibody_embeddings
            )
            antibody_chain_ids[index, : example.antibody_length].copy_(
                example.antibody_chain_ids
            )
            if example.contact_target is not None:
                target_slice = contact_target[
                    index, : example.antigen_length, : example.antibody_length
                ]
                mask_slice = contact_mask[
                    index, : example.antigen_length, : example.antibody_length
                ]
                mask_slice.copy_(example.contact_mask)
                target_slice.copy_(
                    example.contact_target.to(dtype=torch.float32)
                    * example.contact_mask.to(dtype=torch.float32)
                )

        observed_pairs = tuple(
            (example.antibody_id, example.antigen_id) for example in examples
        )
        if self._explicit_positive_graph:
            positive_pairs = self._positive_pairs
            missing = sorted(set(observed_pairs).difference(positive_pairs))
            if missing:
                raise ValueError(
                    "The authorized positive graph omits matched batch edges: "
                    + ", ".join(f"{antibody_id}->{antigen_id}" for antibody_id, antigen_id in missing)
                )
        else:
            positive_pairs = tuple(sorted(set(observed_pairs)))
        positive_graph = frozenset(positive_pairs)
        antibody_ids = tuple(example.antibody_id for example in examples)
        antigen_ids = tuple(example.antigen_id for example in examples)
        positive_pair_mask = torch.tensor(
            [
                [
                    (antibody_id, antigen_id) in positive_graph
                    for antigen_id in antigen_ids
                ]
                for antibody_id in antibody_ids
            ],
            dtype=torch.bool,
            device=device,
        )

        return FrontierBatch(
            instance_ids=instance_ids,
            antibody_ids=antibody_ids,
            antigen_ids=antigen_ids,
            positive_pairs=positive_pairs,
            antibody_embeddings=antibody_embeddings,
            antigen_embeddings=antigen_embeddings,
            antibody_chain_ids=antibody_chain_ids,
            antibody_lengths=antibody_lengths,
            antigen_lengths=antigen_lengths,
            antibody_mask=antibody_mask,
            antigen_mask=antigen_mask,
            pair_mask=pair_mask,
            contact_target=contact_target,
            contact_mask=contact_mask,
            positive_pair_mask=positive_pair_mask,
        )
