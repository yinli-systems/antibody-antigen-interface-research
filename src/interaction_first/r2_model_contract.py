"""Load and construct the unique frozen partner-geometry R2 architecture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .pair_geometry import ProspectivePairGeometryModel
from .pair_geometry_schema import PairGeometrySchema


FORMAT_VERSION = "sabdab2_partner_geometry_r2_model_contract_v1"
EXPECTED_STATUS = (
    "prospective_implementation_frozen_training_blocked_at_gate_0"
)
EXPECTED_ENABLED_HEADS = (
    "contact",
    "distance",
    "orientation",
    "interaction",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class R2ModelContractError(ValueError):
    """Raised when the R2 model contract or its source boundary differs."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite contract metadata deterministically."""
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise R2ModelContractError(
            "R2 model contract must be finite ASCII JSON"
        ) from error


def sha256_file(path: Path) -> str:
    """Hash one regular, non-symbolic file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise R2ModelContractError(f"source is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise R2ModelContractError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R2ModelContractError(f"{name} must be a mapping")
    return value


def _require_exact_json(value: Any, expected: Any, name: str) -> None:
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise R2ModelContractError(f"{name} differs")


def _exact_keys(
    value: Any,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    value = _mapping(value, name)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise R2ModelContractError(
            f"{name} fields differ: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise R2ModelContractError(
                f"R2 model contract contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def load_r2_model_contract(
    path: Path,
    *,
    expected_file_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    """Authenticate the contract, parent protocol, and exact model sources."""
    expected_file_sha256 = _sha256(
        expected_file_sha256,
        "expected_file_sha256",
    )
    raw_path = Path(path)
    raw_repository_root = Path(repository_root)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise R2ModelContractError("R2 model contract must be a regular file")
    if (
        raw_repository_root.is_symlink()
        or not raw_repository_root.is_dir()
    ):
        raise R2ModelContractError("repository_root must be a regular directory")
    path = raw_path.resolve()
    repository_root = raw_repository_root.resolve()
    payload = path.read_bytes()
    observed_file_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_file_sha256 != expected_file_sha256:
        raise R2ModelContractError("R2 model contract file digest differs")
    try:
        contract = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                R2ModelContractError(
                    f"unsupported JSON constant {value!r}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2ModelContractError("R2 model contract is not strict ASCII JSON") from error
    if not isinstance(contract, dict):
        raise R2ModelContractError("R2 model contract must be an object")
    _validate_contract(contract)

    parent = contract["parent_protocol"]
    source_bindings = contract["source_bindings"]
    paths = [parent, *source_bindings.values()]
    for binding in paths:
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise R2ModelContractError("contract source path escapes the repository")
        unresolved_source_path = repository_root / relative
        if unresolved_source_path.is_symlink():
            raise R2ModelContractError(
                f"contract source must not be a symbolic link: "
                f"{relative.as_posix()}"
            )
        source_path = unresolved_source_path.resolve()
        try:
            source_path.relative_to(repository_root)
        except ValueError as error:
            raise R2ModelContractError(
                "contract source resolves outside the repository"
            ) from error
        if sha256_file(source_path) != binding["sha256"]:
            raise R2ModelContractError(
                f"contract source digest differs: {relative.as_posix()}"
            )
    return contract, observed_file_sha256


def _validate_contract(contract: Mapping[str, Any]) -> None:
    top = _exact_keys(
        contract,
        {
            "format_version",
            "status",
            "parent_protocol",
            "source_bindings",
            "model",
            "runtime_schema",
            "authorization",
        },
        "contract",
    )
    if top["format_version"] != FORMAT_VERSION:
        raise R2ModelContractError("unsupported R2 model contract format")
    if top["status"] != EXPECTED_STATUS:
        raise R2ModelContractError("R2 model contract status differs")
    parent = _exact_keys(
        top["parent_protocol"],
        {"path", "sha256"},
        "parent_protocol",
    )
    if parent["path"] != "protocols/sabdab2_partner_geometry_r2_v1.json":
        raise R2ModelContractError("R2 parent protocol path differs")
    _sha256(parent["sha256"], "parent_protocol.sha256")

    sources = _exact_keys(
        top["source_bindings"],
        {
            "pair_geometry",
            "intra_chain_geometry",
            "pair_geometry_schema",
            "r2_model_contract",
        },
        "source_bindings",
    )
    expected_source_paths = {
        "pair_geometry": "src/interaction_first/pair_geometry.py",
        "intra_chain_geometry": (
            "src/interaction_first/intra_chain_geometry.py"
        ),
        "pair_geometry_schema": (
            "src/interaction_first/pair_geometry_schema.py"
        ),
        "r2_model_contract": "src/interaction_first/r2_model_contract.py",
    }
    for name, expected_path in expected_source_paths.items():
        binding = _exact_keys(
            sources[name],
            {"path", "sha256"},
            f"source_bindings.{name}",
        )
        if binding["path"] != expected_path:
            raise R2ModelContractError(f"{name} source path differs")
        _sha256(binding["sha256"], f"{name}.sha256")

    model = _exact_keys(
        top["model"],
        {
            "candidate_id",
            "class_name",
            "antibody_input_dim",
            "antigen_input_dim",
            "d_model",
            "pair_dim",
            "num_heads",
            "num_fusion_layers",
            "ffn_dim",
            "dropout",
            "intra_chain_geometry",
            "pair_representation",
            "prediction",
        },
        "model",
    )
    expected_model_values = {
        "candidate_id": "R2",
        "class_name": "ProspectivePairGeometryModel",
        "antibody_input_dim": 512,
        "antigen_input_dim": 1280,
        "d_model": 256,
        "pair_dim": 128,
        "num_heads": 8,
        "num_fusion_layers": 2,
        "ffn_dim": 512,
        "dropout": 0.1,
    }
    for name, expected in expected_model_values.items():
        _require_exact_json(model[name], expected, f"model field {name}")

    geometry = _mapping(model["intra_chain_geometry"], "intra_chain_geometry")
    expected_geometry = {
        "parameters_shared_between_partners": False,
        "num_layers": 1,
        "feature_mode": "distance_direction",
        "attention_topology": "spatial",
        "num_rbf": 16,
        "distance_max_angstrom": 20.0,
        "neighbor_k": 32,
        "neighbor_radius_angstrom": 16.0,
        "bound_inter_chain_geometry_allowed": False,
    }
    _require_exact_json(
        geometry,
        expected_geometry,
        "intra-chain geometry contract",
    )

    pair = _mapping(model["pair_representation"], "pair_representation")
    expected_pair = {
        "orientation": "antibody_by_antigen",
        "construction": [
            "projected_antibody_residue",
            "projected_antigen_residue",
            "elementwise_gated_interaction",
        ],
        "num_update_layers": 2,
        "update_num_heads": 8,
        "update_ffn_dim": 256,
        "update_order": [
            "antigen_axis_within_antibody_row",
            "antibody_axis_within_antigen_column",
            "feed_forward",
        ],
        "antibody_axis_chunk_size": 64,
    }
    _require_exact_json(
        pair,
        expected_pair,
        "pair representation contract",
    )

    prediction = _mapping(model["prediction"], "prediction")
    expected_prediction = {
        "enabled_heads": list(EXPECTED_ENABLED_HEADS),
        "contact_logit_mode": "unary_plus_interaction_residual",
        "marginal_pooling": "logmeanexp",
        "marginal_top_k": 3,
        "marginal_temperature": 1.0,
        "partner_score_temperature": 1.0,
        "double_centering_iterations": 16,
        "confidence": (
            "one_minus_mean_normalized_distance_orientation_entropy"
        ),
        "uncertainty": (
            "mean_normalized_distance_orientation_entropy"
        ),
    }
    _require_exact_json(
        prediction,
        expected_prediction,
        "R2 prediction contract",
    )

    runtime = _mapping(top["runtime_schema"], "runtime_schema")
    expected_runtime = {
        "pair_geometry_schema_sha256": (
            "externally_frozen_per_admitted_label_contract"
        ),
        "antibody_backbone_required": True,
        "antigen_backbone_required": True,
        "antibody_chain_ids_required": True,
        "antigen_chain_ids_required": True,
        "candidate_pair_mask_required": True,
        "complete_candidate_domain_required_for_evaluation": True,
    }
    _require_exact_json(runtime, expected_runtime, "R2 runtime schema")

    authorization = _mapping(top["authorization"], "authorization")
    _require_exact_json(
        authorization,
        {
            "source_implementation": True,
            "synthetic_forward_and_backward_tests": True,
            "scientific_training": False,
            "model_selection": False,
            "outer_evaluation": False,
            "external_test_access": False,
        },
        "R2 model contract authorization",
    )


def build_frozen_r2_model(
    contract: Mapping[str, Any],
    *,
    geometry_schema: PairGeometrySchema,
    expected_geometry_schema_sha256: str,
) -> ProspectivePairGeometryModel:
    """Build only the exact frozen R2 architecture with an anchored schema."""
    _validate_contract(contract)
    if not isinstance(geometry_schema, PairGeometrySchema):
        raise R2ModelContractError(
            "geometry_schema must be a PairGeometrySchema"
        )
    expected_geometry_schema_sha256 = _sha256(
        expected_geometry_schema_sha256,
        "expected_geometry_schema_sha256",
    )
    if geometry_schema.sha256 != expected_geometry_schema_sha256:
        raise R2ModelContractError("pair-geometry schema digest differs")
    model = contract["model"]
    geometry = model["intra_chain_geometry"]
    pair = model["pair_representation"]
    prediction = model["prediction"]
    return ProspectivePairGeometryModel(
        antibody_input_dim=model["antibody_input_dim"],
        antigen_input_dim=model["antigen_input_dim"],
        d_model=model["d_model"],
        pair_dim=model["pair_dim"],
        num_heads=model["num_heads"],
        num_fusion_layers=model["num_fusion_layers"],
        intra_chain_geometry_layers=geometry["num_layers"],
        intra_chain_geometry_feature_mode=geometry["feature_mode"],
        intra_chain_geometry_num_rbf=geometry["num_rbf"],
        intra_chain_geometry_distance_max=geometry[
            "distance_max_angstrom"
        ],
        intra_chain_geometry_neighbor_k=geometry["neighbor_k"],
        intra_chain_geometry_neighbor_radius=geometry[
            "neighbor_radius_angstrom"
        ],
        ffn_dim=model["ffn_dim"],
        geometry_schema=geometry_schema,
        num_pair_update_layers=pair["num_update_layers"],
        pair_update_num_heads=pair["update_num_heads"],
        pair_update_ffn_dim=pair["update_ffn_dim"],
        pair_chunk_size=pair["antibody_axis_chunk_size"],
        dropout=model["dropout"],
        marginal_pooling=prediction["marginal_pooling"],
        marginal_top_k=prediction["marginal_top_k"],
        marginal_temperature=prediction["marginal_temperature"],
        partner_score_temperature=prediction["partner_score_temperature"],
        centering_iterations=prediction["double_centering_iterations"],
        contact_logit_mode=prediction["contact_logit_mode"],
    )


__all__ = [
    "EXPECTED_ENABLED_HEADS",
    "EXPECTED_STATUS",
    "FORMAT_VERSION",
    "R2ModelContractError",
    "build_frozen_r2_model",
    "canonical_json_bytes",
    "load_r2_model_contract",
    "sha256_file",
]
