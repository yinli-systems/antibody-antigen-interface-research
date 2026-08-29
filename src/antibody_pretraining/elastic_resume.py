"""Topology-aware recovery contracts for deterministic antibody pretraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ELASTIC_STATE_FORMAT = "antibody_elastic_semantic_state_v1"
RNG_MANIFEST_FORMAT = "antibody_rng_checkpoint_manifest_v1"
RNG_STATE_FORMAT = "antibody_rank_rng_state_v1"
TOPOLOGY_ONLY_TRAINER_FIELDS = frozenset(
    {
        "checkpoint_rng_manifest_sha256",
        "world_size",
    }
)


class ElasticResumeError(RuntimeError):
    """Report a fail-closed elastic checkpoint contract violation."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ElasticResumeError(f"{label} must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ElasticResumeError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise ElasticResumeError(f"{label} must be a canonical JSON object")
    return value


def semantic_trainer_state(trainer_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the topology-independent scientific continuation coordinate."""
    required = {
        "consumed_residue_tokens",
        "next_update",
        "run_seed",
        "sample_cursor",
        "target_residue_tokens",
        "training_universe_count",
        "world_size",
    }
    if not isinstance(trainer_state, Mapping) or not required.issubset(trainer_state):
        raise ElasticResumeError("Trainer state lacks an elastic recovery coordinate")
    for name in (
        "consumed_residue_tokens",
        "next_update",
        "run_seed",
        "sample_cursor",
        "target_residue_tokens",
        "training_universe_count",
        "world_size",
    ):
        value = trainer_state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ElasticResumeError(f"Trainer state field {name} is invalid")
    if (
        trainer_state["target_residue_tokens"] <= 0
        or trainer_state["training_universe_count"] <= 0
        or trainer_state["world_size"] <= 0
    ):
        raise ElasticResumeError("Trainer state dimensions must be positive")
    invariant = {
        key: value
        for key, value in trainer_state.items()
        if key not in TOPOLOGY_ONLY_TRAINER_FIELDS
    }
    return {
        "format_version": ELASTIC_STATE_FORMAT,
        "state": invariant,
    }


def semantic_trainer_state_sha256(trainer_state: Mapping[str, Any]) -> str:
    """Hash the global training coordinate without rank-local RNG topology."""
    return hashlib.sha256(
        _canonical_json_bytes(semantic_trainer_state(trainer_state))
    ).hexdigest()


def semantic_optimizer_state(
    module: Any,
    optimizer: Any,
) -> dict[str, Any]:
    """Name optimizer tensors by model parameter instead of process-local IDs."""
    named_parameters = list(module.named_parameters())
    parameter_names = {id(parameter): name for name, parameter in named_parameters}
    if len(parameter_names) != len(named_parameters):
        raise ElasticResumeError("Model parameter identity is not unique")
    groups: list[dict[str, Any]] = []
    optimizer_parameter_ids: list[int] = []
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise ElasticResumeError("Optimizer parameter group is invalid")
        names = []
        for parameter in parameters:
            parameter_id = id(parameter)
            name = parameter_names.get(parameter_id)
            if name is None:
                raise ElasticResumeError(
                    "Optimizer contains a parameter outside the named model"
                )
            names.append(name)
            optimizer_parameter_ids.append(parameter_id)
        groups.append(
            {
                "options": {
                    key: value for key, value in group.items() if key != "params"
                },
                "parameters": names,
            }
        )
    if (
        len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
        or set(optimizer_parameter_ids) != set(parameter_names)
    ):
        raise ElasticResumeError(
            "Optimizer parameter inventory differs from the named model"
        )
    state = {}
    for parameter, value in optimizer.state.items():
        name = parameter_names.get(id(parameter))
        if name is None:
            raise ElasticResumeError(
                "Optimizer state contains a parameter outside the named model"
            )
        state[name] = value
    return {
        "format_version": "antibody_named_optimizer_state_v1",
        "param_groups": groups,
        "state": state,
    }


def validate_randomness_contract(
    model_family: Mapping[str, Any],
    *,
    masking_implementation: str,
) -> None:
    """Require stateless masking and zero model dropout before topology reseeding."""
    architecture = model_family.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ElasticResumeError("Model-family architecture is unavailable")
    if (
        architecture.get("attention_dropout") != 0.0
        or architecture.get("residual_dropout") != 0.0
    ):
        raise ElasticResumeError("Elastic topology reseeding requires zero dropout")
    if masking_implementation != "counter_derived_by_sample_key_update_and_seed":
        raise ElasticResumeError("Elastic recovery requires counter-derived MLM masks")


def authenticate_source_rng_manifest(
    checkpoint_root: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate every source-rank RNG file without restoring rank-local state."""
    manifest_path = checkpoint_root / "rng_manifest.json"
    manifest = _load_canonical_json(manifest_path, "RNG checkpoint manifest")
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifest_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    files = manifest.get("files")
    world_size = manifest.get("world_size")
    if (
        recorded != observed
        or recorded != expected_manifest_sha256
        or manifest.get("format_version") != RNG_MANIFEST_FORMAT
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or not isinstance(files, list)
        or len(files) != world_size
    ):
        raise ElasticResumeError("RNG checkpoint manifest differs")
    state_sha256: list[str] = []
    for rank, entry in enumerate(files):
        expected_path = f"rng/rank-{rank:05d}.json"
        if (
            not isinstance(entry, dict)
            or entry.get("rank") != rank
            or entry.get("path") != expected_path
        ):
            raise ElasticResumeError("RNG checkpoint rank inventory differs")
        state_path = checkpoint_root / expected_path
        if _sha256_file(state_path) != entry.get("file_sha256"):
            raise ElasticResumeError("RNG checkpoint file hash differs")
        state = _load_canonical_json(state_path, f"RNG state for rank {rank}")
        unsigned_state = dict(state)
        recorded_state = unsigned_state.pop("state_sha256", None)
        observed_state = hashlib.sha256(
            _canonical_json_bytes(unsigned_state)
        ).hexdigest()
        if (
            state.get("format_version") != RNG_STATE_FORMAT
            or state.get("rank") != rank
            or recorded_state != observed_state
            or recorded_state != entry.get("state_sha256")
        ):
            raise ElasticResumeError("RNG checkpoint state differs")
        state_sha256.append(str(recorded_state))
    return {
        "manifest_sha256": str(recorded),
        "source_rank_state_sha256": state_sha256,
        "source_world_size": world_size,
    }


def destination_rank_seed(
    *,
    run_seed: int,
    next_update: int,
    rank: int,
    world_size: int,
) -> int:
    """Derive a stable destination-rank seed for non-scientific RNG streams."""
    values = (run_seed, next_update, rank, world_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ElasticResumeError("Elastic destination seed inputs must be integers")
    if run_seed < 0 or next_update < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ElasticResumeError("Elastic destination seed inputs are out of range")
    payload = {
        "format_version": "antibody_elastic_destination_seed_v1",
        "next_update": next_update,
        "rank": rank,
        "run_seed": run_seed,
        "world_size": world_size,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(payload)).digest()[:8],
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)
