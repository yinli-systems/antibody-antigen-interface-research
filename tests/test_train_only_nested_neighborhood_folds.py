#!/usr/bin/env python3
"""Focused tests for nested direct-neighborhood construction and audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

import audit_train_only_nested_neighborhood_folds as auditor
import build_train_only_nested_neighborhood_folds as builder
import pytest
from interaction_first import nested_joint_milp


# Production code executes the externally authenticated joint source bytes
# directly. Tests expose the repository module only as a synthetic-policy helper.
builder.nested_joint_milp = nested_joint_milp


@pytest.mark.parametrize("module", [builder, auditor])
def test_sha256_fields_require_native_json_strings(module: object) -> None:
    """A numeric JSON value must not be coerced into a digest string."""
    numeric_digest = int("1" * 64)
    with pytest.raises(ValueError, match="digest string"):
        module.validate_sha256(numeric_digest, "synthetic digest")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (True, 1),
        (False, 0),
        ({"passed": True}, {"passed": 1}),
        ({"range": 0.0}, {"range": 0}),
    ],
)
def test_exact_json_equality_preserves_scalar_types(
    left: object,
    right: object,
) -> None:
    """Independent receipts must not inherit Python's bool/int aliasing."""
    assert not auditor._exact_json_equal(left, right)
    assert auditor._exact_json_equal(left, json.loads(json.dumps(left)))


class FakeAlignment:
    """Minimal deterministic full-length alignment."""

    def __init__(self, left: str, right: str) -> None:
        width = min(len(left), len(right))
        self.aligned = (((0, width),), ((0, width),))


class FakeAligner:
    """Align equal-length test sequences position by position."""

    def align(self, left: str, right: str) -> list[FakeAlignment]:
        """Return one deterministic synthetic optimum."""
        return [FakeAlignment(left, right)]


class DirectionSensitiveAlignment:
    """Minimal alignment with explicitly selected residue blocks."""

    def __init__(
        self,
        left_blocks: tuple[tuple[int, int], ...],
        right_blocks: tuple[tuple[int, int], ...],
    ) -> None:
        self.aligned = (left_blocks, right_blocks)


class DirectionSensitiveAligner:
    """Reproduce a first-optimum traceback that changes with sequence order."""

    def align(
        self, left: str, right: str
    ) -> list[DirectionSensitiveAlignment]:
        """Qualify only the CAAAA-to-AAAAC orientation at the 80/80 boundary."""
        if (left, right) == ("CAAAA", "AAAAC"):
            return [
                DirectionSensitiveAlignment(
                    ((1, 5),),
                    ((0, 4),),
                )
            ]
        width = min(len(left), len(right))
        return [
            DirectionSensitiveAlignment(
                ((0, width),),
                ((0, width),),
            )
        ]


def _toy_policy(_: int) -> builder.LevelPolicy:
    """Return gates that exercise the full nested machinery on nine pairs."""
    return builder.LevelPolicy(
        minimum_validation_pairs=1,
        maximum_validation_pairs=2,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=1,
        minimum_training_pair_fraction=0.50,
        minimum_training_endpoint_fraction=0.50,
        maximum_quarantine_pair_fraction=0.50,
        maximum_validation_pair_relative_range=1.0,
        maximum_other_count_relative_range=1.0,
    )


def _toy_policy_payload(pair_count: int) -> dict[str, float | int]:
    """Return the synthetic policy in the auditor's independent JSON shape."""
    policy = _toy_policy(pair_count)
    return {
        "minimum_validation_pairs": policy.minimum_validation_pairs,
        "maximum_validation_pairs": policy.maximum_validation_pairs,
        "minimum_validation_endpoints_per_axis": (
            policy.minimum_validation_endpoints_per_axis
        ),
        "minimum_validation_components_per_axis": (
            policy.minimum_validation_components_per_axis
        ),
        "minimum_training_pair_fraction": policy.minimum_training_pair_fraction,
        "minimum_training_endpoint_fraction": (
            policy.minimum_training_endpoint_fraction
        ),
        "maximum_quarantine_pair_fraction": (
            policy.maximum_quarantine_pair_fraction
        ),
        "maximum_validation_pair_relative_range": (
            policy.maximum_validation_pair_relative_range
        ),
        "maximum_other_count_relative_range": (
            policy.maximum_other_count_relative_range
        ),
    }


def _toy_joint_policy(
    validation_upper_numerator: int,
    validation_upper_denominator: int,
) -> builder.nested_joint_milp.JointLevelPolicy:
    """Return an exact policy for the nine-pair joint integration fixture."""
    ratio = builder.nested_joint_milp.Ratio
    return builder.nested_joint_milp.JointLevelPolicy(
        minimum_validation_pairs=1,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=1,
        validation_pair_lower_fraction=ratio(1, 1000),
        validation_pair_upper_fraction=ratio(
            validation_upper_numerator,
            validation_upper_denominator,
        ),
        minimum_training_pair_fraction=ratio(1, 2),
        minimum_training_endpoint_fraction=ratio(1, 2),
        maximum_quarantine_pair_fraction=ratio(0, 1),
        maximum_validation_pair_relative_range=ratio(0, 1),
        maximum_other_count_relative_range=ratio(0, 1),
    )


def _toy_joint_policy_payload(
    pair_count: int,
    *,
    upper_numerator: int,
    upper_denominator: int,
) -> dict[str, float | int]:
    """Serialize one exact toy joint policy after universe realization."""
    policy = builder._level_policy_from_joint_policy(
        _toy_joint_policy(upper_numerator, upper_denominator),
        pair_count,
        joint_module=nested_joint_milp,
    )
    return builder._policy_payload(policy)


def _fixture(root: Path, output_name: str = "assignment.json") -> Namespace:
    """Create nine independent rows in a physical train-only projection."""
    antibody_sequences = [
        "ACDEF",
        "GHIKL",
        "MNPQR",
        "STVWY",
        "AAAAA",
        "CCCCC",
        "DDDDD",
        "EEEEE",
        "FFFFF",
    ]
    antigen_sequences = [
        "YWTSA",
        "RQNPM",
        "LKIHG",
        "FEDCA",
        "GGGGG",
        "HHHHH",
        "IIIII",
        "KKKKK",
        "LLLLL",
    ]
    rows = [
        (f"train_{index}", heavy, "", antigen)
        for index, (heavy, antigen) in enumerate(
            zip(antibody_sequences, antigen_sequences)
        )
    ]
    authority = root / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "format_version": builder.TRAIN_AUTHORITY_FORMAT_VERSION,
                "authorized_split": "train",
                "instance_ids": [row[0] for row in rows],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["INSTANCE", "vh_seq", "vl_seq", "ag_seq"])
        for row in rows:
            writer.writerow(row)
    return Namespace(
        train_authority_allowlist=authority,
        train_authority_allowlist_sha256=builder.sha256_file(authority),
        train_canonical_manifest=manifest,
        train_canonical_manifest_sha256=builder.sha256_file(manifest),
        expected_builder_source_sha256=builder.sha256_file(
            Path(builder.__file__)
        ),
        joint_milp_source=None,
        expected_joint_milp_source_sha256=None,
        expected_joint_policy_sha256=None,
        expected_joint_selected_membership_sha256=None,
        joint_assignment_mode=None,
        provisional_attempt_id=None,
        provisional_ledger_entry=None,
        pinned_replay_id=None,
        source_provisional_attempt_id=None,
        output=root / output_name,
        fold_count=3,
        selection_method="deterministic_greedy",
        selection_seed=2026072401,
        joint_maximum_binary_variables=10_000,
        joint_maximum_linear_constraints=20_000,
        joint_maximum_nonzero_coefficients=100_000,
        joint_time_limit_seconds=10.0,
        workers=1,
        chunk_size=2,
    )


def _build_toy(args: Namespace) -> dict[str, object]:
    """Build one complete nested toy assignment."""
    with (
        mock.patch.object(builder, "build_locked_aligner", return_value=FakeAligner()),
        mock.patch.object(builder, "outer_policy", side_effect=_toy_policy),
        mock.patch.object(builder, "inner_policy", side_effect=_toy_policy),
    ):
        return builder.run(args)


def _configure_joint_toy(
    args: Namespace,
    root: Path,
    *,
    mode: str,
    expected_selected_membership_sha256: str | None = None,
) -> None:
    """Configure one explicit provisional-discovery or pinned-replay attempt."""
    prior_provisional_attempt_id = getattr(
        args,
        "provisional_attempt_id",
        None,
    )
    args.selection_method = "joint_core_milp"
    args.joint_milp_source = Path(nested_joint_milp.__file__)
    args.expected_joint_milp_source_sha256 = builder.sha256_file(
        args.joint_milp_source
    )
    args.expected_joint_policy_sha256 = nested_joint_milp.joint_policy_sha256(
        _toy_joint_policy(1, 9),
        _toy_joint_policy(1, 8),
    )
    args.joint_assignment_mode = mode
    args.expected_joint_selected_membership_sha256 = (
        expected_selected_membership_sha256
    )
    if mode == "provisional_discovery":
        args.provisional_attempt_id = f"toy-{args.output.stem}"
        args.provisional_ledger_entry = root / f"{args.output.stem}.attempt.json"
        args.pinned_replay_id = None
        args.source_provisional_attempt_id = None
    else:
        args.provisional_attempt_id = None
        args.provisional_ledger_entry = None
        args.pinned_replay_id = f"replay-{args.output.stem}"
        args.source_provisional_attempt_id = (
            prior_provisional_attempt_id
            or f"source-attempt-{args.output.stem}"
        )


def _run_joint_toy(args: Namespace) -> dict[str, object]:
    """Run the toy policy while retaining verified-byte production plumbing."""
    original_joint_builder = builder.build_joint_nested_levels

    def solve_toy_joint(
        pairs: tuple[builder.PairGroup, ...],
        antibody_graph: dict[str, set[str]],
        antigen_graph: dict[str, set[str]],
        *,
        resource_limits: nested_joint_milp.ResourceLimits,
        time_limit_seconds: float,
        joint_module: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return original_joint_builder(
            pairs,
            antibody_graph,
            antigen_graph,
            resource_limits=resource_limits,
            time_limit_seconds=time_limit_seconds,
            joint_outer_policy=_toy_joint_policy(1, 9),
            joint_inner_policy=_toy_joint_policy(1, 8),
            joint_module=joint_module,
        )

    with (
        mock.patch.object(
            builder,
            "build_locked_aligner",
            return_value=FakeAligner(),
        ),
        mock.patch.object(
            builder,
            "LOCKED_JOINT_POLICY_SHA256",
            args.expected_joint_policy_sha256,
        ),
        mock.patch.object(
            nested_joint_milp,
            "LOCKED_JOINT_POLICY_SHA256",
            args.expected_joint_policy_sha256,
        ),
        mock.patch.object(
            builder,
            "load_verified_joint_module",
            return_value=nested_joint_milp,
        ),
        mock.patch.object(
            builder,
            "build_joint_nested_levels",
            side_effect=solve_toy_joint,
        ),
    ):
        return builder.run(args)


def _build_pinned_joint_toy(
    root: Path,
    stem: str,
) -> tuple[Namespace, dict[str, object]]:
    """Discover one candidate, preserve it, then replay an external digest."""
    provisional_args = _fixture(root, f"{stem}-provisional.json")
    _configure_joint_toy(
        provisional_args,
        root,
        mode="provisional_discovery",
    )
    provisional = _run_joint_toy(provisional_args)
    pinned_args = Namespace(**vars(provisional_args))
    pinned_args.output = root / f"{stem}-pinned.json"
    _configure_joint_toy(
        pinned_args,
        root,
        mode="externally_pinned_replay",
        expected_selected_membership_sha256=provisional["construction"][
            "selector"
        ]["selected_binary_membership_sha256"],
    )
    return pinned_args, _run_joint_toy(pinned_args)


def _audit_args(
    build_args: Namespace,
    assignment: dict[str, object],
    assignment_path: Path,
    output: Path,
) -> Namespace:
    """Construct an independently pinned synthetic-audit argument set."""
    selector = assignment["construction"]["selector"]
    exact_joint = assignment["construction"]["selection_method"] == "joint_core_milp"
    return Namespace(
        assignment=assignment_path,
        assignment_sha256=auditor.sha256_file(assignment_path),
        assignment_scientific_sha256=assignment["scientific_payload_sha256"],
        train_authority_allowlist=build_args.train_authority_allowlist,
        train_authority_allowlist_sha256=(
            build_args.train_authority_allowlist_sha256
        ),
        train_canonical_manifest=build_args.train_canonical_manifest,
        train_canonical_manifest_sha256=(
            build_args.train_canonical_manifest_sha256
        ),
        expected_auditor_source_sha256=auditor.sha256_file(
            Path(auditor.__file__)
        ),
        builder_source=Path(builder.__file__),
        expected_builder_source_sha256=builder.sha256_file(
            Path(builder.__file__)
        ),
        joint_milp_source=Path(nested_joint_milp.__file__),
        expected_joint_milp_source_sha256=builder.sha256_file(
            Path(nested_joint_milp.__file__)
        ),
        expected_joint_policy_sha256=(
            selector["exact_joint_policy_sha256"]
            if exact_joint
            else nested_joint_milp.LOCKED_JOINT_POLICY_SHA256
        ),
        expected_joint_selected_membership_sha256=(
            build_args.expected_joint_selected_membership_sha256
            if exact_joint
            else "0" * 64
        ),
        expected_pinned_replay_id=(
            build_args.pinned_replay_id
            or "negative-audit-replay-placeholder"
        ),
        expected_source_provisional_attempt_id=(
            build_args.source_provisional_attempt_id
            or "negative-audit-source-attempt-placeholder"
        ),
        output=output,
        workers=1,
        chunk_size=2,
    )


def _write_rehashed_assignment(path: Path, assignment: dict[str, object]) -> None:
    """Rehash and publish a synthetic assignment after deliberate tampering."""
    assignment.pop("scientific_payload_sha256", None)
    assignment["scientific_payload_sha256"] = auditor.canonical_payload_sha256(
        assignment
    )
    path.write_bytes(auditor.canonical_json_bytes(assignment))


def _reanchor_selected_membership(assignment: dict[str, object]) -> str:
    """Reanchor every selected-vector digest after an adversarial ledger edit."""
    construction = assignment["construction"]
    assert isinstance(construction, dict)
    outer = construction["outer_level"]
    assert isinstance(outer, dict)

    def reanchor_level(
        level: dict[str, object],
        prefix: tuple[str, ...],
        label: str,
    ) -> set[tuple[str, ...]]:
        selected: set[tuple[str, ...]] = set()
        folds = level["folds"]
        assert isinstance(folds, list)
        for fold_index, fold in enumerate(folds):
            assert isinstance(fold, dict)
            selected.update(
                auditor._fold_selected_binary_keys(
                    fold,
                    (*prefix, str(fold_index)),
                    f"{label}.fold_{fold_index}",
                )
            )
        selection = level["selection"]
        assert isinstance(selection, dict)
        selection["selected_binary_membership_sha256"] = (
            auditor.membership_sha256("\0".join(key) for key in selected)
        )
        return selected

    selected = reanchor_level(outer, ("outer",), "outer")
    outer_folds = outer["folds"]
    assert isinstance(outer_folds, list)
    for outer_index, fold in enumerate(outer_folds):
        assert isinstance(fold, dict)
        inner = fold["inner_level"]
        assert isinstance(inner, dict)
        selected.update(
            reanchor_level(
                inner,
                ("inner", str(outer_index)),
                f"outer.fold_{outer_index}.inner",
            )
        )
    digest = auditor.membership_sha256(
        "\0".join(key) for key in selected
    )
    selector = construction["selector"]
    assert isinstance(selector, dict)
    selector["selected_binary_membership_sha256"] = digest
    selector["externally_pinned_selected_binary_membership_sha256"] = digest
    return digest


def _assert_reanchored_component_tamper_rejected(
    root: Path,
    label: str,
    build_args: Namespace,
    tampered: dict[str, object],
) -> None:
    """Require independent graph reconstruction to reject a reanchored lie."""
    tampered_digest = _reanchor_selected_membership(tampered)
    tampered_path = root / f"{label}.json"
    _write_rehashed_assignment(tampered_path, tampered)
    audit_args = _audit_args(
        build_args,
        tampered,
        tampered_path,
        root / f"{label}-audit.json",
    )
    audit_args.expected_joint_selected_membership_sha256 = tampered_digest
    with (
        mock.patch.object(
            auditor,
            "build_locked_aligner",
            return_value=FakeAligner(),
        ),
        mock.patch.object(
            auditor,
            "independently_recomputed_outer_policy",
            side_effect=lambda count: _toy_joint_policy_payload(
                count,
                upper_numerator=1,
                upper_denominator=9,
            ),
        ),
        mock.patch.object(
            auditor,
            "independently_recomputed_inner_policy",
            side_effect=lambda count: _toy_joint_policy_payload(
                count,
                upper_numerator=1,
                upper_denominator=8,
            ),
        ),
        mock.patch.object(
            auditor,
            "LOCKED_JOINT_POLICY_SHA256",
            build_args.expected_joint_policy_sha256,
        ),
        mock.patch.object(
            nested_joint_milp,
            "LOCKED_JOINT_POLICY_SHA256",
            build_args.expected_joint_policy_sha256,
        ),
        mock.patch.object(
            auditor,
            "load_verified_joint_module",
            return_value=nested_joint_milp,
        ),
        pytest.raises(
            ValueError,
            match="component inventory differs from the independently",
        ),
    ):
        auditor.run(audit_args)
    assert not audit_args.output.exists()


def test_firewall_accepts_only_physical_train_projection() -> None:
    """Only an exact four-column physical train-only projection is accepted."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        args = _fixture(Path(directory))
        payload = _build_toy(args)
        firewall = payload["data_firewall"]
        assert firewall["authorized_rows_materialized"] == 9
        assert firewall["exact_sequence_only_schema_verified"]
        assert firewall["one_record_per_physical_line_verified"]
        assert firewall["validation_or_test_rows_present"] == 0
        assert firewall["validation_or_test_sequence_fields_parsed"] == 0


@pytest.mark.parametrize(
    "loader",
    (
        builder.load_train_authority_allowlist,
        auditor.read_train_authority_allowlist,
    ),
)
def test_firewall_rejects_full_split_and_non_string_authority_ids(
    loader: object,
) -> None:
    """Old full-split schemas and coerced numeric/bool IDs fail closed."""
    full_split = builder.canonical_json_bytes(
        {
            "splits": {
                "train": ["train_0"],
                "validation": ["sealed_validation"],
                "test": ["sealed_test"],
            }
        }
    )
    with pytest.raises(ValueError, match="full split or sealed"):
        loader(full_split)
    for invalid_identifier in (7, True):
        invalid = builder.canonical_json_bytes(
            {
                "format_version": builder.TRAIN_AUTHORITY_FORMAT_VERSION,
                "authorized_split": "train",
                "instance_ids": [invalid_identifier],
            }
        )
        with pytest.raises(ValueError, match="must be a string"):
            loader(invalid)


@pytest.mark.parametrize(
    "loader",
    (
        builder.load_train_only_records,
        auditor.reconstruct_train_rows,
    ),
)
@pytest.mark.parametrize(
    "manifest",
    (
        (
            b"INSTANCE,vh_seq,vl_seq,ag_seq,score\n"
            b"train_0,ACDEF,,GGGGG,forbidden\n"
        ),
        (
            b"INSTANCE,vh_seq,vh_seq,ag_seq\n"
            b"train_0,ACDEF,,GGGGG\n"
        ),
    ),
)
def test_firewall_rejects_extra_or_duplicate_manifest_columns(
    loader: object,
    manifest: bytes,
) -> None:
    """Only the duplicate-free exact four-column sequence schema is valid."""
    with pytest.raises(ValueError, match="exactly INSTANCE"):
        loader(manifest, {"train_0"})


@pytest.mark.parametrize(
    "loader",
    (
        builder.load_train_only_records,
        auditor.reconstruct_train_rows,
    ),
)
def test_firewall_rejects_multiline_unknown_and_missing_rows(
    loader: object,
) -> None:
    """Quoted continuation tricks and non-allowlisted coverage fail closed."""
    multiline = (
        b"INSTANCE,vh_seq,vl_seq,ag_seq\n"
        b'train_0,"ACD\n'
        b'train_1,EF",,GGGGG\n'
    )
    with pytest.raises(ValueError, match="not one complete CSV record"):
        loader(multiline, {"train_0"})
    unknown = (
        b"INSTANCE,vh_seq,vl_seq,ag_seq\n"
        b"train_0,ACDEF,,GGGGG\n"
        b"outside_train,CCCCC,,HHHHH\n"
    )
    with pytest.raises(ValueError, match="outside the externally pinned"):
        loader(unknown, {"train_0"})
    missing = b"INSTANCE,vh_seq,vl_seq,ag_seq\n"
    with pytest.raises(ValueError, match="does not exactly equal"):
        loader(missing, {"train_0"})


@pytest.mark.parametrize("module", (builder, auditor))
@pytest.mark.parametrize("symlink_parent", (False, True))
def test_verified_acquisition_rejects_final_and_parent_symlinks(
    module: object,
    symlink_parent: bool,
) -> None:
    """Every path component, including the final file, is opened no-follow."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        real_parent = root / "real"
        real_parent.mkdir()
        target = real_parent / "payload.bin"
        target.write_bytes(b"authenticated bytes")
        expected = hashlib.sha256(target.read_bytes()).hexdigest()
        if symlink_parent:
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            candidate = linked_parent / target.name
        else:
            candidate = root / "linked-file"
            candidate.symlink_to(target)
        with pytest.raises(ValueError, match="symlink"):
            module.verify_file(candidate, expected, "adversarial input")


@pytest.mark.parametrize(
    ("module", "loader"),
    (
        (builder, builder.load_train_authority_allowlist),
        (auditor, auditor.read_train_authority_allowlist),
    ),
)
def test_verified_payload_survives_pathname_replacement_without_reopen(
    module: object,
    loader: object,
) -> None:
    """Parsing consumes authenticated bytes even if the pathname is replaced."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        path = root / "authority.json"
        original = builder.canonical_json_bytes(
            {
                "format_version": builder.TRAIN_AUTHORITY_FORMAT_VERSION,
                "authorized_split": "train",
                "instance_ids": ["train_0"],
            }
        )
        path.write_bytes(original)
        verified = module.verify_file(
            path,
            hashlib.sha256(original).hexdigest(),
            "train authority",
        )
        replacement = root / "replacement.json"
        replacement.write_bytes(
            builder.canonical_json_bytes(
                {
                    "splits": {
                        "train": ["attacker"],
                        "validation": [],
                        "test": [],
                    }
                }
            )
        )
        os.replace(replacement, path)
        train_ids, _ = loader(verified.payload)
        assert train_ids == {"train_0"}
        assert (
            os.stat(path, follow_symlinks=False).st_ino
            != verified.reference["stat_fingerprint_after"]["inode"]
        )


@pytest.mark.parametrize("module", (builder, auditor))
def test_verified_acquisition_detects_in_place_growth(module: object) -> None:
    """Before/after fd fingerprints and EOF checks reject mid-read mutation."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        path = Path(directory) / "growing.bin"
        path.write_bytes(b"A" * 1024)
        original_read = os.read
        mutated = False

        def adversarial_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            chunk = original_read(descriptor, count)
            if chunk and not mutated:
                mutated = True
                with path.open("ab") as handle:
                    handle.write(b"B")
                    handle.flush()
                    os.fsync(handle.fileno())
            return chunk

        with (
            mock.patch.object(module.os, "read", side_effect=adversarial_read),
            pytest.raises(ValueError, match="grew|changed"),
        ):
            module.verify_file(
                path,
                hashlib.sha256(b"A" * 1024).hexdigest(),
                "mutating input",
            )


@pytest.mark.parametrize("module", (builder, auditor))
def test_verified_acquisition_fails_without_nofollow_capability(
    module: object,
) -> None:
    """A missing no-follow flag cannot silently degrade the receipt claim."""
    with (
        mock.patch.object(module.os, "O_NOFOLLOW", None),
        pytest.raises(RuntimeError, match="missing_flags=.*O_NOFOLLOW"),
    ):
        module._require_secure_open_capabilities()


def test_framed_antibody_identity_prevents_raw_concatenation_collapse() -> None:
    """Different chain tuples with equal raw concatenation remain distinct."""
    left = builder.antibody_id("AC", "DEF")
    right = builder.antibody_id("ACD", "EF")
    assert left != right
    records = {
        "one": builder.SourceRecord("one", "AC", "DEF", "GGGGG"),
        "two": builder.SourceRecord("two", "ACD", "EF", "HHHHH"),
    }
    antibodies, _, _, summary = builder.canonicalize_inventory(records)
    assert len(antibodies) == 2
    assert summary["raw_concatenation_collision_count"] == 1


def test_closed_neighborhood_is_direct_not_transitive() -> None:
    """A--B--C must exclude B for held-out A without excluding C."""
    graph = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B"},
    }
    observed = builder._closed_neighborhood({"A"}, graph, {"A", "B", "C"})
    assert observed == {"A", "B"}
    assert "C" not in observed


def test_inner_component_gate_keeps_full_graph_bridge_identity() -> None:
    """Removing a bridge from a level must not create new countable components."""
    antibody_graph = {
        "a0": {"a1"},
        "a1": {"a0", "a2"},
        "a2": {"a1"},
    }
    antigen_graph = {
        "g0": {"g1"},
        "g1": {"g0", "g2"},
        "g2": {"g1"},
    }
    antibody_components = builder.canonical_component_ids(antibody_graph)
    antigen_components = builder.canonical_component_ids(antigen_graph)
    assert antibody_components == {"a0": "a0", "a1": "a0", "a2": "a0"}
    assert antigen_components == {"g0": "g0", "g1": "g0", "g2": "g0"}

    pairs = (
        builder.PairGroup("p0", "a0", "g0", ("s0",)),
        builder.PairGroup("p2", "a2", "g2", ("s2",)),
    )
    policy = builder.LevelPolicy(
        minimum_validation_pairs=2,
        maximum_validation_pairs=2,
        minimum_validation_endpoints_per_axis=2,
        minimum_validation_components_per_axis=2,
        minimum_training_pair_fraction=0.0,
        minimum_training_endpoint_fraction=0.0,
        maximum_quarantine_pair_fraction=1.0,
        maximum_validation_pair_relative_range=1.0,
        maximum_other_count_relative_range=1.0,
    )
    fold, _ = builder._derive_fold(
        pairs,
        {"a0", "a2"},
        {"g0", "g2"},
        antibody_graph,
        antigen_graph,
        antibody_components,
        antigen_components,
        {"a0", "a2"},
        {"g0", "g2"},
        policy,
        fold_index=0,
    )
    assert fold["validation_antibody_component_ids"] == ["a0"]
    assert fold["validation_antigen_component_ids"] == ["g0"]
    assert not fold["hard_gates"][
        "validation_antibody_component_count_passed"
    ]
    assert not fold["hard_gates"][
        "validation_antigen_component_count_passed"
    ]


def test_builder_parity_check_binds_both_component_variable_roles() -> None:
    """The builder cannot claim parity after a component-role substitution."""
    pairs = (
        builder.PairGroup("p0", "a0", "g0", ("s0",)),
        builder.PairGroup("p1", "a1", "g1", ("s1",)),
    )
    policy = builder.LevelPolicy(
        minimum_validation_pairs=1,
        maximum_validation_pairs=1,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=1,
        minimum_training_pair_fraction=0.0,
        minimum_training_endpoint_fraction=0.0,
        maximum_quarantine_pair_fraction=1.0,
        maximum_validation_pair_relative_range=1.0,
        maximum_other_count_relative_range=1.0,
    )
    fold, _ = builder._derive_fold(
        pairs,
        {"a0", "a1"},
        {"g0", "g1"},
        {"a0": set(), "a1": set()},
        {"g0": set(), "g1": set()},
        {"a0": "a0", "a1": "a1"},
        {"g0": "g0", "g1": "g1"},
        {"a0"},
        {"g0"},
        policy,
        fold_index=0,
    )
    prefix = ("outer", "0")
    role_members = {
        "holdout_a": fold["validation_antibody_ids"],
        "holdout_g": fold["validation_antigen_ids"],
        "closed_a": fold["closed_direct_neighborhood_antibody_ids"],
        "closed_g": fold["closed_direct_neighborhood_antigen_ids"],
        "train_present_a": fold["model_antibody_allowlist"],
        "train_present_g": fold["model_antigen_allowlist"],
        "validation_component_a": fold[
            "validation_antibody_component_ids"
        ],
        "validation_component_g": fold[
            "validation_antigen_component_ids"
        ],
        "validation_pair": [
            pair["pair_id"] for pair in fold["validation_pairs"]
        ],
        "training_pair": [pair["pair_id"] for pair in fold["training_pairs"]],
        "quarantine_pair": [
            pair["pair_id"] for pair in fold["quarantined_pairs"]
        ],
    }
    assignment = {
        (*prefix, role, member): 1
        for role, members in role_members.items()
        for member in members
    }
    builder._assert_fold_matches_joint_assignment(fold, assignment, prefix)

    for role, observed, replacement in (
        ("validation_component_a", "a0", "a1"),
        ("validation_component_g", "g0", "g1"),
    ):
        tampered = dict(assignment)
        del tampered[(*prefix, role, observed)]
        tampered[(*prefix, role, replacement)] = 1
        with pytest.raises(RuntimeError, match=role):
            builder._assert_fold_matches_joint_assignment(
                fold,
                tampered,
                prefix,
            )


def test_inner_auditor_rejects_reanchored_induced_bridge_component() -> None:
    """An omitted bridge cannot split a full-graph component during audit."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(
            root,
            "bridge-component-audit",
        )
        train_ids, _ = auditor.read_train_authority_allowlist(
            build_args.train_authority_allowlist.read_bytes()
        )
        rows, _ = auditor.reconstruct_train_rows(
            build_args.train_canonical_manifest.read_bytes(),
            train_ids,
        )
        _, _, all_pairs, _ = auditor.reconstruct_entities(rows)
        outer_fold = assignment["construction"]["outer_level"]["folds"][0]
        inner = outer_fold["inner_level"]
        inner_pair_ids = [
            pair["pair_id"] for pair in outer_fold["training_pairs"]
        ]
        inner_pairs = {
            pair_id: all_pairs[pair_id] for pair_id in inner_pair_ids
        }
        all_antibody_ids = sorted(
            {pair.antibody_id for pair in all_pairs.values()}
        )
        all_antigen_ids = sorted(
            {pair.antigen_id for pair in all_pairs.values()}
        )
        removed_bridge = outer_fold["validation_antibody_ids"][0]
        selected_by_fold = [
            fold["validation_antibody_ids"][0] for fold in inner["folds"]
        ]
        selected_endpoint = next(
            entity
            for entity in selected_by_fold
            if entity != all_antibody_ids[0]
        )
        other_endpoint = next(
            entity
            for entity in all_antibody_ids
            if entity not in {removed_bridge, selected_endpoint}
        )
        antibody_graph = {entity: set() for entity in all_antibody_ids}
        antibody_graph[other_endpoint].add(removed_bridge)
        antibody_graph[removed_bridge].update(
            {other_endpoint, selected_endpoint}
        )
        antibody_graph[selected_endpoint].add(removed_bridge)
        antibody_components = (
            auditor.independently_reconstructed_component_ids(antibody_graph)
        )
        assert antibody_components[selected_endpoint] != selected_endpoint
        antigen_components = auditor.independently_reconstructed_component_ids(
            {entity: set() for entity in all_antigen_ids}
        )

        universe = inner["universe"]
        universe["full_graph_antibody_component_count"] = len(
            set(antibody_components.values())
        )
        universe["full_graph_antibody_component_membership_sha256"] = (
            auditor.membership_sha256(
                f"{entity}\0{representative}"
                for entity, representative in antibody_components.items()
            )
        )
        target_fold = None
        for fold in inner["folds"]:
            effective = {
                inner_pairs[pair["pair_id"]].antibody_id
                for pair in fold["validation_pairs"]
            }
            components = sorted(
                {antibody_components[entity] for entity in effective}
            )
            fold["validation_antibody_component_ids"] = components
            fold["counts"]["validation_antibody_component_count"] = len(
                components
            )
            fold["membership_hashes"][
                "validation_antibody_component_sha256"
            ] = auditor.membership_sha256(components)
            fold["hard_gates"][
                "validation_antibody_component_count_passed"
            ] = bool(components)
            if selected_endpoint in effective:
                target_fold = fold
        assert target_fold is not None
        target_fold["validation_antibody_component_ids"] = [selected_endpoint]
        target_fold["membership_hashes"][
            "validation_antibody_component_sha256"
        ] = auditor.membership_sha256([selected_endpoint])
        _reanchor_selected_membership(assignment)

        with (
            mock.patch.object(
                auditor,
                "independently_recomputed_inner_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=8,
                ),
            ),
            pytest.raises(
                ValueError,
                match="component inventory differs from the independently",
            ),
        ):
            auditor.validate_level(
                inner,
                inner_pairs,
                "outer.fold_0.inner",
                level_kind="inner",
                antibody_component_by_entity=antibody_components,
                antigen_component_by_entity=antigen_components,
            )


def test_symmetric_builder_edge_is_fail_closed_under_directional_traceback() -> None:
    """Either qualifying alignment orientation must create one undirected edge."""
    aligner = DirectionSensitiveAligner()
    qualifies, forward, reverse = builder.symmetric_threshold_evaluation(
        "AAAAC", "CAAAA", aligner
    )
    assert qualifies
    assert forward == (3, 5)
    assert reverse == (4, 4)

    with mock.patch.object(
        builder, "build_locked_aligner", return_value=aligner
    ):
        edges, receipt = builder.build_exact_view_graph(
            {"held_out": "AAAAC", "training": "CAAAA"},
            workers=1,
            chunk_size=1,
        )
    assert edges == {("held_out", "training")}
    assert receipt["dynamic_programming_alignment_count"] == 1
    assert (
        receipt["directional_dynamic_programming_alignment_execution_count"]
        == 2
    )
    assert receipt["every_aligned_pair_evaluated_in_both_orientations"]
    graph = builder.adjacency({"held_out", "training"}, edges)
    assert builder._closed_neighborhood(
        {"held_out"}, graph, {"held_out", "training"}
    ) == {"held_out", "training"}


def test_symmetric_audit_is_invariant_to_boundary_role_direction() -> None:
    """Swapping train and validation roles must not change edge existence."""
    a_left = auditor.independent_antibody_id("CAAAA", "")
    a_right = auditor.independent_antibody_id("AAAAC", "")
    g_left = auditor.independent_antigen_id("CCCCC")
    g_right = auditor.independent_antigen_id("DDDDD")
    p_left = auditor.independent_pair_id(a_left, g_left)
    p_right = auditor.independent_pair_id(a_right, g_right)
    pairs = {
        p_left: auditor.IndependentPair(p_left, a_left, g_left, ("left",)),
        p_right: auditor.IndependentPair(
            p_right, a_right, g_right, ("right",)
        ),
    }
    antibodies = {a_left: ("CAAAA", ""), a_right: ("AAAAC", "")}
    antigens = {g_left: "CCCCC", g_right: "DDDDD"}
    aligner = DirectionSensitiveAligner()
    with mock.patch.object(
        auditor, "build_locked_aligner", return_value=aligner
    ):
        forward = auditor.audit_boundary_view(
            auditor.Boundary("forward", (p_left,), (p_right,)),
            "vh",
            pairs,
            antibodies,
            antigens,
            workers=1,
            chunk_size=1,
        )
        reverse = auditor.audit_boundary_view(
            auditor.Boundary("reverse", (p_right,), (p_left,)),
            "vh",
            pairs,
            antibodies,
            antigens,
            workers=1,
            chunk_size=1,
        )
    assert forward["threshold_crossing_edge_count"] == 1
    assert reverse["threshold_crossing_edge_count"] == 1
    assert (
        forward["directional_dynamic_programming_alignment_execution_count"]
        == 2
    )
    assert (
        reverse["directional_dynamic_programming_alignment_execution_count"]
        == 2
    )
    assert forward["crossing_edges"][0][
        "selected_qualifying_orientation"
    ] == "training_to_validation"
    assert reverse["crossing_edges"][0][
        "selected_qualifying_orientation"
    ] == "validation_to_training"
    assert forward["crossing_edges"][0][
        "symmetric_either_orientation_threshold_edge"
    ]
    assert reverse["crossing_edges"][0][
        "symmetric_either_orientation_threshold_edge"
    ]


def test_quarantine_ledger_records_every_incompatible_pair_and_source_row() -> None:
    """Mixed-role pairs must be quarantined with exact source-row grouping."""
    pairs = (
        builder.PairGroup("p00", "a0", "g0", ("s00", "s00_dup")),
        builder.PairGroup("p01", "a0", "g1", ("s01",)),
        builder.PairGroup("p10", "a1", "g0", ("s10",)),
        builder.PairGroup("p11", "a1", "g1", ("s11",)),
    )
    policy = _toy_policy(len(pairs))
    fold, training = builder._derive_fold(
        pairs,
        {"a0", "a1"},
        {"g0", "g1"},
        {"a0": set(), "a1": set()},
        {"g0": set(), "g1": set()},
        {"a0": "a0", "a1": "a1"},
        {"g0": "g0", "g1": "g1"},
        {"a0"},
        {"g0"},
        policy,
        fold_index=0,
    )
    assert {pair.pair_id for pair in training} == {"p11"}
    assert {pair["pair_id"] for pair in fold["validation_pairs"]} == {"p00"}
    quarantined = {
        pair["pair_id"]: pair for pair in fold["quarantined_pairs"]
    }
    assert set(quarantined) == {"p01", "p10"}
    assert quarantined["p01"]["source_instance_ids"] == ["s01"]
    assert set(quarantined["p01"]["reasons"]) == {
        "held_out_antibody",
        "mixed_role_pair",
    }
    assert set(quarantined["p10"]["reasons"]) == {
        "held_out_antigen",
        "mixed_role_pair",
    }
    assert fold["counts"]["training_source_row_count"] == 1
    assert fold["counts"]["validation_source_row_count"] == 2
    assert fold["counts"]["quarantined_source_row_count"] == 2


def test_infeasible_validation_interval_is_rejected_before_selection() -> None:
    """A minimum above the maximum must fail before the selector is entered."""
    pair = builder.PairGroup("p0", "a0", "g0", ("s0",))
    impossible = builder.LevelPolicy(
        minimum_validation_pairs=2,
        maximum_validation_pairs=1,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=1,
        minimum_training_pair_fraction=0.0,
        minimum_training_endpoint_fraction=0.0,
        maximum_quarantine_pair_fraction=1.0,
        maximum_validation_pair_relative_range=1.0,
        maximum_other_count_relative_range=1.0,
    )
    with mock.patch.object(
        builder,
        "_select_held_out_sets",
        side_effect=AssertionError("selector must not run"),
    ):
        try:
            builder.build_level(
                (pair,),
                {"a0": set()},
                {"g0": set()},
                impossible,
                fold_count=3,
                seed_label="impossible",
            )
        except ValueError as error:
            assert "validation-pair interval is infeasible" in str(error)
        else:
            raise AssertionError("An infeasible validation interval was accepted")

    real_inner_policy = builder.inner_policy(847)
    assert real_inner_policy.minimum_validation_pairs == 80
    assert real_inner_policy.maximum_validation_pairs == 76
    try:
        builder.validate_level_policy(
            real_inner_policy,
            847,
            label="real-inner-regression",
        )
    except ValueError as error:
        assert "minimum=80" in str(error)
        assert "maximum=76" in str(error)
    else:
        raise AssertionError("The 847-pair infeasible interval was accepted")


def test_independent_locked_policy_recomputation_preserves_declared_formulas() -> None:
    """The auditor must own exact outer and inner formulas, including infeasibility."""
    assert auditor.independently_recomputed_outer_policy(1830) == {
        "minimum_validation_pairs": 129,
        "maximum_validation_pairs": 164,
        "minimum_validation_endpoints_per_axis": 60,
        "minimum_validation_components_per_axis": 25,
        "minimum_training_pair_fraction": 0.60,
        "minimum_training_endpoint_fraction": 0.60,
        "maximum_quarantine_pair_fraction": 0.33,
        "maximum_validation_pair_relative_range": 0.15,
        "maximum_other_count_relative_range": 0.20,
    }
    inner = auditor.independently_recomputed_inner_policy(847)
    assert inner["minimum_validation_pairs"] == 80
    assert inner["maximum_validation_pairs"] == 76
    try:
        auditor._parse_policy(inner, "locked-inner-regression")
    except ValueError as error:
        assert "validation-pair interval is infeasible" in str(error)
    else:
        raise AssertionError("The auditor relaxed the locked infeasible interval")


def test_balance_gates_do_not_override_failed_absolute_fold_gates() -> None:
    """Balanced fold sizes remain unauthorized when absolute retention fails."""
    pairs = tuple(
        builder.PairGroup(f"p{index}", f"a{index}", f"g{index}", (f"s{index}",))
        for index in range(9)
    )
    antibody_graph = {f"a{index}": set() for index in range(9)}
    antigen_graph = {f"g{index}": set() for index in range(9)}
    policy = builder.LevelPolicy(
        minimum_validation_pairs=1,
        maximum_validation_pairs=1,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=1,
        minimum_training_pair_fraction=1.0,
        minimum_training_endpoint_fraction=0.0,
        maximum_quarantine_pair_fraction=1.0,
        maximum_validation_pair_relative_range=0.0,
        maximum_other_count_relative_range=0.0,
    )
    level, _ = builder.build_level(
        pairs,
        antibody_graph,
        antigen_graph,
        policy,
        fold_count=3,
        seed_label="balanced-but-under-retained",
    )
    relative_range_gates = {
        key: value
        for key, value in level["level_hard_gates"].items()
        if key.endswith("_relative_range_passed")
    }
    assert relative_range_gates
    assert all(relative_range_gates.values())
    assert not level["level_hard_gates"]["every_fold_hard_gate_passed"]
    assert not level["all_level_hard_gates_passed"]
    assert all(
        not fold["hard_gates"]["training_pair_retention_passed"]
        for fold in level["folds"]
    )


def test_complete_three_outer_nine_inner_ledgers_are_deterministic() -> None:
    """Identical inputs must yield byte-identical complete nested assignments."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        first_args = _fixture(root, "first.json")
        second_args = Namespace(**vars(first_args))
        second_args.output = root / "second.json"
        first = _build_toy(first_args)
        second = _build_toy(second_args)
        assert first == second
        assert first_args.output.read_bytes() == second_args.output.read_bytes()
        outer = first["construction"]["outer_level"]
        assert len(outer["folds"]) == 3
        assert all(len(fold["inner_level"]["folds"]) == 3 for fold in outer["folds"])
        assert first["core_feasibility_constraints_passed"]
        assert not first["construction_invariants_passed"]
        assert first["artifact_role"] == "non_authoritative_greedy_construction"
        assert not first["authorized_for_nested_model_selection"]
        for fold in outer["folds"]:
            all_pair_ids = {
                pair["pair_id"]
                for key in (
                    "training_pairs",
                    "validation_pairs",
                    "quarantined_pairs",
                )
                for pair in fold[key]
            }
            assert len(all_pair_ids) == 9
        scientific = dict(first)
        claimed = scientific.pop("scientific_payload_sha256")
        assert claimed == builder.canonical_payload_sha256(scientific)


def test_joint_core_milp_materializes_exact_three_plus_nine_ledgers() -> None:
    """One joint solve must survive exact role parity and payload publication."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        provisional_args = _fixture(root, "joint-provisional.json")
        _configure_joint_toy(
            provisional_args,
            root,
            mode="provisional_discovery",
        )
        provisional = _run_joint_toy(provisional_args)
        selected_anchor = provisional["construction"]["selector"][
            "selected_binary_membership_sha256"
        ]
        assert provisional["format_version"] == builder.PROVISIONAL_FORMAT_VERSION
        assert provisional["workflow_identity"] == {
            "joint_assignment_mode": "provisional_discovery",
            "provisional_attempt_id": provisional_args.provisional_attempt_id,
            "pinned_replay_id": None,
            "source_provisional_attempt_id": None,
        }
        assert not provisional["construction_invariants_passed"]
        assert provisional_args.provisional_ledger_entry.exists()
        assert (
            stat.S_IMODE(provisional_args.output.stat().st_mode)
            == 0o400
        )
        assert (
            stat.S_IMODE(
                provisional_args.provisional_ledger_entry.stat().st_mode
            )
            == 0o400
        )
        started_entry = provisional_args.provisional_ledger_entry.with_name(
            f"{provisional_args.provisional_ledger_entry.name}.started.json"
        )
        assert started_entry.exists()
        assert stat.S_IMODE(started_entry.stat().st_mode) == 0o400
        started = json.loads(started_entry.read_text(encoding="utf-8"))
        assert started["ledger_entry_kind"] == "attempt_started"
        assert not started["externally_pinned_assignment_identity"]
        ledger = json.loads(
            provisional_args.provisional_ledger_entry.read_text(encoding="utf-8")
        )
        assert ledger["ledger_entry_kind"] == "attempt_outcome"
        assert ledger["selected_binary_membership_sha256"] == selected_anchor
        assert not ledger["eligible_for_independent_pinned_replay_audit"]
        provisional_audit_args = _audit_args(
            provisional_args,
            provisional,
            provisional_args.output,
            root / "provisional-audit-must-not-exist.json",
        )
        with pytest.raises(ValueError, match="provisional"):
            auditor.run(provisional_audit_args)
        assert not provisional_audit_args.output.exists()

        wrong_replay_args = Namespace(**vars(provisional_args))
        wrong_replay_args.output = root / "wrong-replay.json"
        _configure_joint_toy(
            wrong_replay_args,
            root,
            mode="externally_pinned_replay",
            expected_selected_membership_sha256="0" * 64,
        )
        with pytest.raises(ValueError, match="identity differs"):
            _run_joint_toy(wrong_replay_args)
        assert not wrong_replay_args.output.exists()

        args = Namespace(**vars(provisional_args))
        args.output = root / "joint.json"
        _configure_joint_toy(
            args,
            root,
            mode="externally_pinned_replay",
            expected_selected_membership_sha256=selected_anchor,
        )
        payload = _run_joint_toy(args)

        selector = payload["construction"]["selector"]
        assert payload["construction"]["selection_method"] == "joint_core_milp"
        assert payload["workflow_identity"] == {
            "joint_assignment_mode": "externally_pinned_replay",
            "provisional_attempt_id": None,
            "pinned_replay_id": args.pinned_replay_id,
            "source_provisional_attempt_id": (
                args.source_provisional_attempt_id
            ),
        }
        assert not payload["construction"]["selection_seed_used"]
        assert selector["exact_nested_milp_used"]
        assert selector["joint_outer_and_all_dependent_inner_folds_in_one_model"]
        assert selector["exact_integer_row_audit_passed"]
        assert selector["mechanical_role_parity_check_passed_for_all_12_folds"]
        assert not selector["lexicographic_optimization_performed"]
        assert "joint_core_milp" in payload["source_files"]

        outer = payload["construction"]["outer_level"]
        assert len(outer["folds"]) == 3
        outer_validation: set[str] = set()
        for outer_fold in outer["folds"]:
            assert len(outer_fold["validation_antibody_component_ids"]) == 1
            assert len(outer_fold["validation_antigen_component_ids"]) == 1
            assert outer_fold["counts"][
                "validation_antibody_component_count"
            ] == 1
            assert outer_fold["counts"][
                "validation_antigen_component_count"
            ] == 1
            outer_validation.update(
                pair["pair_id"] for pair in outer_fold["validation_pairs"]
            )
            outer_training = {
                pair["pair_id"] for pair in outer_fold["training_pairs"]
            }
            inner = outer_fold["inner_level"]
            assert len(inner["folds"]) == 3
            inner_validation: set[str] = set()
            for inner_fold in inner["folds"]:
                assert len(inner_fold["validation_antibody_component_ids"]) == 1
                assert len(inner_fold["validation_antigen_component_ids"]) == 1
                validation = {
                    pair["pair_id"]
                    for pair in inner_fold["validation_pairs"]
                }
                training = {
                    pair["pair_id"] for pair in inner_fold["training_pairs"]
                }
                quarantine = {
                    pair["pair_id"]
                    for pair in inner_fold["quarantined_pairs"]
                }
                assert len(validation) == 1
                assert len(training) == 7
                assert not quarantine
                assert validation | training == outer_training
                inner_validation.update(validation)
            assert len(inner_validation) == 3
        assert len(outer_validation) == 3
        assert payload["construction_invariants_passed"]
        assert not payload["authorized_for_nested_model_selection"]

        audit_args = _audit_args(
            args,
            payload,
            args.output,
            Path(directory) / "joint-audit.json",
        )
        with (
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                return_value=FakeAligner(),
            ),
            mock.patch.object(
                auditor,
                "independently_recomputed_outer_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=9,
                ),
            ),
            mock.patch.object(
                auditor,
                "independently_recomputed_inner_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=8,
                ),
            ),
            mock.patch.object(
                auditor,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                nested_joint_milp,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                auditor,
                "load_verified_joint_module",
                return_value=nested_joint_milp,
            ),
        ):
            audit_receipt = auditor.run(audit_args)
        assert audit_receipt["core_feasibility_precursor_passed"]
        assert not audit_receipt["split_domain_gate0_requirement_satisfied"]
        assert not audit_receipt["authorized_for_training"]
        assert not audit_receipt["authorized_for_model_selection"]
        assert not audit_receipt["authorized_for_nested_model_selection"]
        assert audit_receipt["workflow_identity"] == {
            **payload["workflow_identity"],
            "matches_external_audit_anchors": True,
        }
        assert audit_receipt["exact_joint_selector_audit"][
            "exact_joint_selector_verified"
        ]
        exact_audit = audit_receipt["exact_joint_selector_audit"]
        assert exact_audit[
            "all_four_exact_direct_graphs_reconstructed_and_bound"
        ]
        assert exact_audit[
            "matrix_recomputed_from_independent_inventory_and_exact_graphs"
        ]
        assert exact_audit[
            "shared_externally_pinned_joint_formulation_source_reused"
        ]
        assert not exact_audit[
            "second_independent_joint_formulation_implementation_used"
        ]
        assert exact_audit[
            "selected_membership_recomputed_from_all_twelve_ledgers"
        ]
        assert exact_audit["every_reconstructed_exact_model_row_passed"]
        assert not exact_audit[
            "historical_solver_execution_independently_attested"
        ]
        assert any(
            "no second independent formulation implementation" in limitation
            for limitation in audit_receipt["limitations"]
        )
        original_audit_bytes = audit_args.output.read_bytes()
        with pytest.raises(FileExistsError):
            auditor.run(audit_args)
        assert audit_args.output.read_bytes() == original_audit_bytes
        assert stat.S_IMODE(audit_args.output.stat().st_mode) == 0o400

        wrong_lineage_args = _audit_args(
            args,
            payload,
            args.output,
            Path(directory) / "wrong-lineage-audit.json",
        )
        wrong_lineage_args.expected_pinned_replay_id = "different-replay-id"
        with (
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                side_effect=AssertionError("alignment must not start"),
            ),
            pytest.raises(ValueError, match="external audit anchors"),
        ):
            auditor.run(wrong_lineage_args)
        assert not wrong_lineage_args.output.exists()

        tampered = json.loads(json.dumps(payload))
        tampered["construction"]["selector"]["solver_reported_mip_gap"] = 0.01
        tampered["construction"]["selector"][
            "solver_reported_mip_gap_is_finite_zero"
        ] = False
        tampered_path = Path(directory) / "joint-tampered-gap.json"
        _write_rehashed_assignment(tampered_path, tampered)
        tampered_args = _audit_args(
            args,
            tampered,
            tampered_path,
            Path(directory) / "joint-tampered-gap-audit.json",
        )
        with (
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                side_effect=AssertionError("alignment must not start"),
            ),
            mock.patch.object(
                auditor,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                nested_joint_milp,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                auditor,
                "load_verified_joint_module",
                return_value=nested_joint_milp,
            ),
            pytest.raises(ValueError, match="finite_zero"),
        ):
            auditor.run(tampered_args)
        assert not tampered_args.output.exists()

        wrong_policy_args = _audit_args(
            args,
            payload,
            args.output,
            Path(directory) / "wrong-policy-audit.json",
        )
        wrong_policy_args.expected_joint_policy_sha256 = "0" * 64
        with (
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                side_effect=AssertionError("alignment must not start"),
            ),
            pytest.raises(ValueError, match="independently locked"),
        ):
            auditor.run(wrong_policy_args)
        assert not wrong_policy_args.output.exists()


def test_failed_provisional_attempt_retains_create_only_start_ledger() -> None:
    """An orderly solver failure cannot erase that a provisional attempt began."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        args = _fixture(root, "failed-provisional.json")
        _configure_joint_toy(
            args,
            root,
            mode="provisional_discovery",
        )
        with (
            mock.patch.object(
                builder,
                "build_locked_aligner",
                return_value=FakeAligner(),
            ),
            mock.patch.object(
                builder,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                nested_joint_milp,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                builder,
                "load_verified_joint_module",
                return_value=nested_joint_milp,
            ),
            mock.patch.object(
                builder,
                "build_joint_nested_levels",
                side_effect=RuntimeError("synthetic solver failure"),
            ),
            pytest.raises(RuntimeError, match="synthetic solver failure"),
        ):
            builder.run(args)
        started_entry = args.provisional_ledger_entry.with_name(
            f"{args.provisional_ledger_entry.name}.started.json"
        )
        assert started_entry.exists()
        assert not args.provisional_ledger_entry.exists()
        assert not args.output.exists()


def test_provisional_start_ledger_cannot_alias_assignment_output() -> None:
    """All three create-only provisional publications require distinct paths."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        args = _fixture(root, "path-alias.json")
        _configure_joint_toy(
            args,
            root,
            mode="provisional_discovery",
        )
        args.output = args.provisional_ledger_entry.with_name(
            f"{args.provisional_ledger_entry.name}.started.json"
        )
        with (
            mock.patch.object(
                builder,
                "LOCKED_JOINT_POLICY_SHA256",
                args.expected_joint_policy_sha256,
            ),
            pytest.raises(
                ValueError,
                match="start ledger and assignment output must differ",
            ),
        ):
            builder.run(args)
        assert not args.output.exists()
        assert not args.provisional_ledger_entry.exists()


def test_provisional_outcome_is_published_only_after_candidate() -> None:
    """A failed outcome write cannot claim publication of a missing candidate."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        args = _fixture(root, "candidate-before-outcome.json")
        _configure_joint_toy(
            args,
            root,
            mode="provisional_discovery",
        )
        original_atomic_create_json = builder.atomic_create_json

        def fail_outcome_write(
            path: Path,
            payload: dict[str, object],
        ) -> None:
            if Path(path).resolve() == args.provisional_ledger_entry.resolve():
                raise OSError("synthetic outcome publication failure")
            original_atomic_create_json(path, payload)

        with (
            mock.patch.object(
                builder,
                "atomic_create_json",
                side_effect=fail_outcome_write,
            ),
            pytest.raises(OSError, match="synthetic outcome publication failure"),
        ):
            _run_joint_toy(args)

        started_entry = args.provisional_ledger_entry.with_name(
            f"{args.provisional_ledger_entry.name}.started.json"
        )
        assert started_entry.exists()
        assert args.output.exists()
        assert not args.provisional_ledger_entry.exists()
        candidate = json.loads(args.output.read_text(encoding="utf-8"))
        assert candidate["stage"] == (
            "completed_provisional_unpinned_core_feasibility_discovery"
        )


def test_provisional_candidate_write_failure_leaves_only_start_ledger() -> None:
    """Failure of the second create-only write cannot publish an outcome."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        args = _fixture(root, "candidate-write-failure.json")
        _configure_joint_toy(
            args,
            root,
            mode="provisional_discovery",
        )
        original_atomic_create_json = builder.atomic_create_json

        def fail_candidate_write(
            path: Path,
            payload: dict[str, object],
        ) -> None:
            if Path(path).resolve() == args.output.resolve():
                raise OSError("synthetic candidate publication failure")
            original_atomic_create_json(path, payload)

        with (
            mock.patch.object(
                builder,
                "atomic_create_json",
                side_effect=fail_candidate_write,
            ),
            pytest.raises(OSError, match="synthetic candidate publication failure"),
        ):
            _run_joint_toy(args)

        started_entry = args.provisional_ledger_entry.with_name(
            f"{args.provisional_ledger_entry.name}.started.json"
        )
        assert started_entry.exists()
        assert not args.output.exists()
        assert not args.provisional_ledger_entry.exists()


def test_joint_auditor_recomputes_matrix_and_selected_membership_digests() -> None:
    """Syntactically valid invented MILP digests must fail semantic binding."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        provisional_args = _fixture(root, "joint-binding-provisional.json")
        _configure_joint_toy(
            provisional_args,
            root,
            mode="provisional_discovery",
        )
        provisional = _run_joint_toy(provisional_args)
        args = Namespace(**vars(provisional_args))
        args.output = root / "joint-binding.json"
        _configure_joint_toy(
            args,
            root,
            mode="externally_pinned_replay",
            expected_selected_membership_sha256=provisional["construction"][
                "selector"
            ]["selected_binary_membership_sha256"],
        )
        payload = _run_joint_toy(args)

        attacks = (
            (
                "matrix_sha256",
                "0" * 64,
                "matrix SHA-256 differs",
            ),
            (
                "selected_binary_membership_sha256",
                "1" * 64,
                "assignment identity is not bound",
            ),
        )
        for index, (field, invented_digest, message) in enumerate(attacks):
            tampered = json.loads(json.dumps(payload))
            tampered["construction"]["selector"][field] = invented_digest
            tampered_path = root / f"invented-receipt-{index}.json"
            _write_rehashed_assignment(tampered_path, tampered)
            audit_args = _audit_args(
                args,
                tampered,
                tampered_path,
                root / f"invented-receipt-{index}-audit.json",
            )
            with (
                mock.patch.object(
                    auditor,
                    "build_locked_aligner",
                    return_value=FakeAligner(),
                ),
                mock.patch.object(
                    auditor,
                    "independently_recomputed_outer_policy",
                    side_effect=lambda count: _toy_joint_policy_payload(
                        count,
                        upper_numerator=1,
                        upper_denominator=9,
                    ),
                ),
                mock.patch.object(
                    auditor,
                    "independently_recomputed_inner_policy",
                    side_effect=lambda count: _toy_joint_policy_payload(
                        count,
                        upper_numerator=1,
                        upper_denominator=8,
                    ),
                ),
                mock.patch.object(
                    auditor,
                    "LOCKED_JOINT_POLICY_SHA256",
                    args.expected_joint_policy_sha256,
                ),
                mock.patch.object(
                    nested_joint_milp,
                    "LOCKED_JOINT_POLICY_SHA256",
                    args.expected_joint_policy_sha256,
                ),
                mock.patch.object(
                    auditor,
                    "load_verified_joint_module",
                    return_value=nested_joint_milp,
                ),
                pytest.raises(ValueError, match=message),
            ):
                auditor.run(audit_args)
            assert not audit_args.output.exists()


def test_auditor_rejects_reanchored_full_graph_component_substitution() -> None:
    """A new external vector anchor cannot legitimize an induced-component lie."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(root, "component-tamper")
        tampered = json.loads(json.dumps(assignment))
        outer = tampered["construction"]["outer_level"]
        folds = outer["folds"]
        replacement = folds[1]["validation_antibody_component_ids"][0]
        assert replacement != folds[0]["validation_antibody_component_ids"][0]
        folds[0]["validation_antibody_component_ids"] = [replacement]
        folds[0]["membership_hashes"][
            "validation_antibody_component_sha256"
        ] = auditor.membership_sha256([replacement])

        _assert_reanchored_component_tamper_rejected(
            root,
            "component-tampered",
            build_args,
            tampered,
        )


def test_auditor_rejects_reanchored_unknown_and_axis_swapped_components() -> None:
    """External anchors cannot authorize unknown or cross-axis component IDs."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(
            root,
            "component-domain-tamper",
        )
        source_fold = assignment["construction"]["outer_level"]["folds"][0]
        antibody_components = {
            component
            for fold in assignment["construction"]["outer_level"]["folds"]
            for component in fold["validation_antibody_component_ids"]
        }
        unknown = "f" * 64
        assert unknown not in antibody_components
        axis_swapped = source_fold["validation_antigen_component_ids"][0]
        assert axis_swapped not in antibody_components

        for label, replacement in (
            ("unknown-component", unknown),
            ("axis-swapped-component", axis_swapped),
        ):
            tampered = json.loads(json.dumps(assignment))
            fold = tampered["construction"]["outer_level"]["folds"][0]
            fold["validation_antibody_component_ids"] = [replacement]
            fold["membership_hashes"][
                "validation_antibody_component_sha256"
            ] = auditor.membership_sha256([replacement])
            _assert_reanchored_component_tamper_rejected(
                root,
                label,
                build_args,
                tampered,
            )


def test_auditor_rejects_unknown_assignment_container_fields() -> None:
    """Unknown fields cannot smuggle prohibited data into an audited artifact."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(
            root,
            "unknown-schema",
        )
        attacks: tuple[
            tuple[str, tuple[str | int, ...]],
            ...,
        ] = (
            ("top", ()),
            ("construction", ("construction",)),
            ("selector", ("construction", "selector")),
            ("construction-gates", ("construction_gates",)),
            ("required-next-gate", ("required_next_gate",)),
            ("authority", ("authority",)),
            ("data-firewall", ("data_firewall",)),
            ("alignment-policy", ("alignment_policy",)),
            ("runtime", ("runtime",)),
            ("outer-level", ("construction", "outer_level")),
            ("outer-fold", ("construction", "outer_level", "folds", 0)),
            (
                "inner-level",
                ("construction", "outer_level", "folds", 0, "inner_level"),
            ),
            (
                "inner-fold",
                (
                    "construction",
                    "outer_level",
                    "folds",
                    0,
                    "inner_level",
                    "folds",
                    0,
                ),
            ),
        )
        for label, path in attacks:
            tampered = json.loads(json.dumps(assignment))
            container: object = tampered
            for component in path:
                container = container[component]  # type: ignore[index]
            assert isinstance(container, dict)
            container["unexpected_field"] = {
                "prohibited_data": ["must-not-be-audited"]
            }
            tampered_path = root / f"unknown-{label}.json"
            _write_rehashed_assignment(tampered_path, tampered)
            audit_args = _audit_args(
                build_args,
                tampered,
                tampered_path,
                root / f"unknown-{label}-audit.json",
            )
            with (
                mock.patch.object(
                    auditor,
                    "build_locked_aligner",
                    side_effect=AssertionError("alignment must not start"),
                ),
                mock.patch.object(
                    auditor,
                    "independently_recomputed_outer_policy",
                    side_effect=lambda count: _toy_joint_policy_payload(
                        count,
                        upper_numerator=1,
                        upper_denominator=9,
                    ),
                ),
                mock.patch.object(
                    auditor,
                    "independently_recomputed_inner_policy",
                    side_effect=lambda count: _toy_joint_policy_payload(
                        count,
                        upper_numerator=1,
                        upper_denominator=8,
                    ),
                ),
                mock.patch.object(
                    auditor,
                    "LOCKED_JOINT_POLICY_SHA256",
                    build_args.expected_joint_policy_sha256,
                ),
                mock.patch.object(
                    nested_joint_milp,
                    "LOCKED_JOINT_POLICY_SHA256",
                    build_args.expected_joint_policy_sha256,
                ),
                mock.patch.object(
                    auditor,
                    "load_verified_joint_module",
                    return_value=nested_joint_milp,
                ),
                pytest.raises(
                    ValueError,
                    match="schema mismatch|differs",
                ),
            ):
                auditor.run(audit_args)
            assert not audit_args.output.exists()


def test_auditor_rejects_boolean_integer_alias_in_replay_envelope() -> None:
    """A rehashed artifact cannot replace a required JSON boolean with 1."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        _, assignment = _build_pinned_joint_toy(
            root,
            "typed-envelope",
        )
        tampered = json.loads(json.dumps(assignment))
        tampered["alignment_policy"]["threshold_equality_is_edge"] = 1
        with pytest.raises(ValueError, match="alignment_policy differs"):
            auditor.validate_replay_assignment_envelope(tampered)


def test_auditor_rejects_noncanonical_duplicate_and_reordered_ledgers() -> None:
    """One selected-membership digest cannot authorize malleable ledgers."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(
            root,
            "canonical-ledgers",
        )
        train_ids, _ = auditor.read_train_authority_allowlist(
            build_args.train_authority_allowlist.read_bytes()
        )
        rows, _ = auditor.reconstruct_train_rows(
            build_args.train_canonical_manifest.read_bytes(),
            train_ids,
        )
        _, _, pairs, _ = auditor.reconstruct_entities(rows)
        antibody_components = {
            pair.antibody_id: pair.antibody_id for pair in pairs.values()
        }
        antigen_components = {
            pair.antigen_id: pair.antigen_id for pair in pairs.values()
        }

        attacks: list[tuple[dict[str, object], str]] = []
        duplicate = json.loads(json.dumps(assignment["construction"]["outer_level"]))
        duplicate_ids = duplicate["folds"][0]["validation_antibody_ids"]
        duplicate_ids.append(duplicate_ids[0])
        attacks.append((duplicate, "canonical sorted unique list"))

        reordered = json.loads(json.dumps(assignment["construction"]["outer_level"]))
        training_pairs = reordered["folds"][0]["training_pairs"]
        assert len(training_pairs) > 1
        training_pairs.reverse()
        attacks.append((reordered, "canonical pair order"))

        for tampered_level, message in attacks:
            with (
                mock.patch.object(
                    auditor,
                    "independently_recomputed_outer_policy",
                    side_effect=lambda count: _toy_joint_policy_payload(
                        count,
                        upper_numerator=1,
                        upper_denominator=9,
                    ),
                ),
                mock.patch.object(
                    auditor,
                    "independently_recomputed_inner_policy",
                    side_effect=lambda count: _toy_joint_policy_payload(
                        count,
                        upper_numerator=1,
                        upper_denominator=8,
                    ),
                ),
                pytest.raises(ValueError, match=message),
            ):
                auditor.validate_level(
                    tampered_level,
                    pairs,
                    "outer",
                    level_kind="outer",
                    antibody_component_by_entity=antibody_components,
                    antigen_component_by_entity=antigen_components,
                )


def test_joint_cli_requires_pinned_source_and_caps_reviewed_resources() -> None:
    """Joint mode separates provisional discovery from externally pinned replay."""
    digest = "0" * 64
    required = [
        "--train-authority-allowlist",
        "authority.json",
        "--train-authority-allowlist-sha256",
        digest,
        "--train-canonical-manifest",
        "manifest.csv",
        "--train-canonical-manifest-sha256",
        digest,
        "--expected-builder-source-sha256",
        digest,
        "--output",
        "assignment.json",
        "--selection-method",
        "joint_core_milp",
    ]
    with pytest.raises(SystemExit):
        builder.parse_args(required)

    joint_common = [
        *required,
        "--joint-milp-source",
        "nested_joint_milp.py",
        "--expected-joint-milp-source-sha256",
        digest,
        "--expected-joint-policy-sha256",
        nested_joint_milp.LOCKED_JOINT_POLICY_SHA256,
    ]
    provisional = [
        *joint_common,
        "--joint-assignment-mode",
        "provisional_discovery",
        "--provisional-attempt-id",
        "attempt-001",
        "--provisional-ledger-entry",
        "attempt-001.json",
    ]
    provisional_args = builder.parse_args(provisional)
    assert provisional_args.expected_joint_selected_membership_sha256 is None
    pinned = [
        *joint_common,
        "--joint-assignment-mode",
        "externally_pinned_replay",
        "--expected-joint-selected-membership-sha256",
        digest,
        "--pinned-replay-id",
        "replay-001",
        "--source-provisional-attempt-id",
        "attempt-001",
    ]
    args = builder.parse_args(pinned)
    assert args.selection_method == "joint_core_milp"
    assert args.joint_assignment_mode == "externally_pinned_replay"
    assert args.pinned_replay_id == "replay-001"
    assert args.source_provisional_attempt_id == "attempt-001"
    assert (
        args.joint_maximum_linear_constraints
        == builder.REVIEWED_JOINT_MAXIMUM_LINEAR_CONSTRAINTS
    )
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                *required,
                "--joint-milp-source",
                "nested_joint_milp.py",
                "--expected-joint-milp-source-sha256",
                digest,
                "--expected-joint-policy-sha256",
                digest,
            ]
        )
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                *provisional,
                "--expected-joint-selected-membership-sha256",
                digest,
            ]
        )
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                *pinned,
                "--joint-maximum-binary-variables",
                str(builder.REVIEWED_JOINT_MAXIMUM_BINARY_VARIABLES + 1),
            ]
        )
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                *joint_common,
                "--joint-assignment-mode",
                "externally_pinned_replay",
                "--expected-joint-selected-membership-sha256",
                digest,
            ]
        )
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                *joint_common,
                "--joint-assignment-mode",
                "externally_pinned_replay",
                "--expected-joint-selected-membership-sha256",
                digest,
                "--pinned-replay-id",
                "same-id",
                "--source-provisional-attempt-id",
                "same-id",
            ]
        )


def test_auditor_cli_requires_distinct_external_replay_lineage() -> None:
    """The auditor accepts only a separately numbered externally anchored replay."""
    digest = "0" * 64
    required = [
        "--assignment",
        "assignment.json",
        "--assignment-sha256",
        digest,
        "--assignment-scientific-sha256",
        digest,
        "--train-authority-allowlist",
        "authority.json",
        "--train-authority-allowlist-sha256",
        digest,
        "--train-canonical-manifest",
        "manifest.csv",
        "--train-canonical-manifest-sha256",
        digest,
        "--expected-auditor-source-sha256",
        digest,
        "--builder-source",
        "builder.py",
        "--expected-builder-source-sha256",
        digest,
        "--joint-milp-source",
        "joint.py",
        "--expected-joint-milp-source-sha256",
        digest,
        "--expected-joint-policy-sha256",
        auditor.LOCKED_JOINT_POLICY_SHA256,
        "--expected-joint-selected-membership-sha256",
        digest,
        "--output",
        "audit.json",
    ]
    with pytest.raises(SystemExit):
        auditor.parse_args(required)

    complete = [
        *required,
        "--expected-pinned-replay-id",
        "replay-001",
        "--expected-source-provisional-attempt-id",
        "attempt-001",
    ]
    args = auditor.parse_args(complete)
    assert args.expected_pinned_replay_id == "replay-001"
    assert args.expected_source_provisional_attempt_id == "attempt-001"

    with pytest.raises(SystemExit):
        auditor.parse_args(
            [
                *required,
                "--expected-pinned-replay-id",
                "same-id",
                "--expected-source-provisional-attempt-id",
                "same-id",
            ]
        )


def test_slurm_stages_only_nofollow_verified_train_only_inputs() -> None:
    """Launcher must execute only sealed copies under external anchors."""
    launcher = (
        Path(builder.__file__).resolve().parents[1]
        / "slurm"
        / "frontier_nested_split_build_audit.sbatch"
    ).read_text(encoding="utf-8")
    required_external_variables = {
        "FRONTIER_EXPECTED_SOURCE_MANIFEST_SHA256",
        "FRONTIER_EXPECTED_JOINT_POLICY_SHA256",
        "FRONTIER_EXPECTED_BUILDER_SOURCE_SHA256",
        "FRONTIER_EXPECTED_AUDITOR_SOURCE_SHA256",
        "FRONTIER_EXPECTED_JOINT_MILP_SOURCE_SHA256",
        "FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256",
        "FRONTIER_EXPECTED_TRAIN_CANONICAL_MANIFEST_SHA256",
        "FRONTIER_TRAIN_AUTHORITY_ALLOWLIST",
        "FRONTIER_TRAIN_CANONICAL_MANIFEST",
        "FRONTIER_JOINT_ASSIGNMENT_MODE",
    }
    assert all(
        f"${{{name}:?" in launcher
        for name in required_external_variables
    )
    mode_dependent_external_variables = {
        "FRONTIER_PROVISIONAL_ATTEMPT_ID",
        "FRONTIER_PINNED_REPLAY_ID",
        "FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID",
        "FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256",
    }
    assert all(name in launcher for name in mode_dependent_external_variables)
    expected_argument_bindings = {
        '--expected-builder-source-sha256 "$EXPECTED_BUILDER_SHA256"',
        '--expected-auditor-source-sha256 "$EXPECTED_AUDITOR_SHA256"',
        '--expected-joint-milp-source-sha256 "$EXPECTED_JOINT_MILP_SHA256"',
        '--expected-joint-policy-sha256 "$EXPECTED_JOINT_POLICY_SHA256"',
        '--joint-milp-source "$STAGED_JOINT_MILP"',
        '--builder-source "$STAGED_BUILDER"',
        '--train-authority-allowlist "$STAGED_TRAIN_AUTHORITY"',
        '--train-canonical-manifest "$STAGED_TRAIN_MANIFEST"',
        '--pinned-replay-id "$PINNED_REPLAY_ID"',
        (
            '--source-provisional-attempt-id '
            '"$SOURCE_PROVISIONAL_ATTEMPT_ID"'
        ),
        '--expected-pinned-replay-id "$PINNED_REPLAY_ID"',
        (
            '--expected-source-provisional-attempt-id '
            '"$SOURCE_PROVISIONAL_ATTEMPT_ID"'
        ),
    }
    assert all(binding in launcher for binding in expected_argument_bindings)
    required_hardening_markers = {
        "open_directory_nofollow",
        "O_NOFOLLOW",
        "O_EXCL",
        "single_verified_fd_read",
        "seal_tree",
        "provisional_discovery",
        "externally_pinned_replay",
        "provisional_attempt_ledger_entry.json",
        "artifact_tree_sealed_after_inventory",
        "launcher_success.json",
        "artifact_tree_post_seal_hash_and_mode_verification_passed",
        "artifact_tree_post_seal_fsync_completed",
        '"sealed_file_mode": "0400"',
        '"sealed_directory_mode": "0500"',
        'unset PYTHONPATH PYTHONHOME',
        "PATH=/usr/bin:/bin",
        "builtin unset -f",
        "launcher_preexecution_shell_state_attested",
        "full_runtime_closure_attested",
        "successful_completion_required_for_final_artifact_inventory_and_tree_seal",
        "failed_launch_artifact_tree_inventory_or_seal_guaranteed",
        '"$PYTHON_BIN" -I -S -B -',
    }
    assert all(marker in launcher for marker in required_hardening_markers)
    forbidden_legacy_or_in_band_markers = {
        "FRONTIER_AUTHORITY_SPLIT",
        "FRONTIER_CANONICAL_MANIFEST",
        "--authority-split",
        "--canonical-manifest",
        "double_remote_split.json",
        "sha256sum",
        "| tee",
        "$(hostname)",
        "date --iso-8601",
        '"$PYTHON_BIN" "$SOURCE_ROOT/scripts/',
    }
    assert all(
        marker not in launcher
        for marker in forbidden_legacy_or_in_band_markers
    )
    assert (
        '--expected-joint-selected-membership-sha256 '
        '"$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256"'
        in launcher
    )


def test_slurm_replay_lineage_fails_closed_before_bootstrap() -> None:
    """Missing, reused, or inapplicable replay identifiers stop the launcher."""
    launcher_path = (
        Path(builder.__file__).resolve().parents[1]
        / "slurm"
        / "frontier_nested_split_build_audit.sbatch"
    )
    digest = "0" * 64
    base = os.environ.copy()
    for name in (
        "FRONTIER_PROVISIONAL_ATTEMPT_ID",
        "FRONTIER_PINNED_REPLAY_ID",
        "FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID",
    ):
        base.pop(name, None)
    base.update(
        {
            "FRONTIER_SOURCE_ROOT": "/must-not-be-opened",
            "FRONTIER_OUTPUT_ROOT": "/must-not-be-created",
            "FRONTIER_TRAIN_AUTHORITY_ALLOWLIST": "/must-not-be-opened.json",
            "FRONTIER_TRAIN_CANONICAL_MANIFEST": "/must-not-be-opened.csv",
            "FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256": digest,
            "FRONTIER_EXPECTED_TRAIN_CANONICAL_MANIFEST_SHA256": digest,
            "FRONTIER_EXPECTED_SOURCE_MANIFEST_SHA256": digest,
            "FRONTIER_EXPECTED_JOINT_POLICY_SHA256": digest,
            "FRONTIER_EXPECTED_BUILDER_SOURCE_SHA256": digest,
            "FRONTIER_EXPECTED_AUDITOR_SOURCE_SHA256": digest,
            "FRONTIER_EXPECTED_JOINT_MILP_SOURCE_SHA256": digest,
            "FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256": digest,
            "SLURM_JOB_ID": "lineage-negative-test",
        }
    )

    missing = dict(base)
    missing["FRONTIER_JOINT_ASSIGNMENT_MODE"] = "externally_pinned_replay"
    completed = subprocess.run(
        ["bash", str(launcher_path)],
        env=missing,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "requires FRONTIER_PINNED_REPLAY_ID" in completed.stderr

    reused = dict(missing)
    reused["FRONTIER_PINNED_REPLAY_ID"] = "attempt-001"
    reused["FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID"] = "attempt-001"
    completed = subprocess.run(
        ["bash", str(launcher_path)],
        env=reused,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "Replay ID must differ" in completed.stderr

    provisional_with_replay_ids = dict(base)
    provisional_with_replay_ids.update(
        {
            "FRONTIER_JOINT_ASSIGNMENT_MODE": "provisional_discovery",
            "FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256": "",
            "FRONTIER_PROVISIONAL_ATTEMPT_ID": "attempt-001",
            "FRONTIER_PINNED_REPLAY_ID": "replay-001",
            "FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID": "attempt-001",
        }
    )
    completed = subprocess.run(
        ["bash", str(launcher_path)],
        env=provisional_with_replay_ids,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "forbids pinned replay identifiers" in completed.stderr

    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        marker = Path(directory) / "inherited-function-ran"
        inherited_function_attack = dict(base)
        inherited_function_attack.update(
            {
                "PATH": directory,
                "BASH_FUNC_validate_sha256_identifier%%": (
                    f"() {{ : > {marker}; return 0; }}"
                ),
                "FRONTIER_JOINT_ASSIGNMENT_MODE": "externally_pinned_replay",
                "FRONTIER_PINNED_REPLAY_ID": "replay-001",
                "FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID": "attempt-001",
                "FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256": "invalid",
            }
        )
        completed = subprocess.run(
            ["/bin/bash", str(launcher_path)],
            env=inherited_function_attack,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "must be one lowercase SHA-256 digest" in completed.stderr
        assert not marker.exists()


def test_slurm_bootstrap_rejects_extra_source_manifest_entries_before_read() -> None:
    """The staged application-source closure is exactly three files, never data."""
    launcher_path = (
        Path(builder.__file__).resolve().parents[1]
        / "slurm"
        / "frontier_nested_split_build_audit.sbatch"
    )
    launcher = launcher_path.read_text(encoding="utf-8")
    marker = '"$PYTHON_BIN" -I -S -B - <<\'PY\'\n'
    start = launcher.index(marker) + len(marker)
    end = launcher.index("\nPY\n", start)
    bootstrap = launcher[start : end + 1]
    required_entries = {
        "scripts/build_train_only_nested_neighborhood_folds.py": "1" * 64,
        "scripts/audit_train_only_nested_neighborhood_folds.py": "2" * 64,
        "src/interaction_first/nested_joint_milp.py": "3" * 64,
    }
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        output.mkdir()
        lines = [
            f"{digest}  {relative}\n"
            for relative, digest in required_entries.items()
        ]
        lines.append(f"{'4' * 64}  data/sealed_test.csv\n")
        manifest_payload = "".join(lines).encode("utf-8")
        (source / "SOURCE_MANIFEST.sha256").write_bytes(manifest_payload)
        environment = os.environ.copy()
        environment.update(
            {
                "SOURCE_ROOT": str(source),
                "OUTPUT_ROOT": str(output),
                "RUN_ID": "extra-entry-test",
                "TRAIN_AUTHORITY_ALLOWLIST": str(root / "must-not-open.json"),
                "TRAIN_CANONICAL_MANIFEST": str(root / "must-not-open.csv"),
                "EXPECTED_TRAIN_AUTHORITY_SHA256": "5" * 64,
                "EXPECTED_TRAIN_MANIFEST_SHA256": "6" * 64,
                "EXPECTED_SOURCE_MANIFEST_SHA256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
                "EXPECTED_BUILDER_SHA256": required_entries[
                    "scripts/build_train_only_nested_neighborhood_folds.py"
                ],
                "EXPECTED_AUDITOR_SHA256": required_entries[
                    "scripts/audit_train_only_nested_neighborhood_folds.py"
                ],
                "EXPECTED_JOINT_MILP_SHA256": required_entries[
                    "src/interaction_first/nested_joint_milp.py"
                ],
            }
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", bootstrap],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert (
            "exactly the three-file staged application-source closure"
            in completed.stderr
        )
        assert not (
            output
            / "slurm-extra-entry-test"
            / "stage"
            / "source"
            / "data"
        ).exists()


def test_slurm_bootstrap_stages_and_seals_exact_code_only_closure() -> None:
    """The positive bootstrap path copies the exact closure and train projection."""
    launcher_path = (
        Path(builder.__file__).resolve().parents[1]
        / "slurm"
        / "frontier_nested_split_build_audit.sbatch"
    )
    launcher = launcher_path.read_text(encoding="utf-8")
    marker = '"$PYTHON_BIN" -I -S -B - <<\'PY\'\n'
    start = launcher.index(marker) + len(marker)
    end = launcher.index("\nPY\n", start)
    bootstrap = launcher[start : end + 1]
    source_payloads = {
        "scripts/build_train_only_nested_neighborhood_folds.py": b"builder\n",
        "scripts/audit_train_only_nested_neighborhood_folds.py": b"auditor\n",
        "src/interaction_first/nested_joint_milp.py": b"joint\n",
    }
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        output.mkdir()
        source_digests: dict[str, str] = {}
        for relative, payload in source_payloads.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            source_digests[relative] = hashlib.sha256(payload).hexdigest()
        manifest_payload = "".join(
            f"{source_digests[relative]}  {relative}\n"
            for relative in source_payloads
        ).encode("utf-8")
        (source / "SOURCE_MANIFEST.sha256").write_bytes(manifest_payload)
        authority_payload = builder.canonical_json_bytes(
            {
                "format_version": builder.TRAIN_AUTHORITY_FORMAT_VERSION,
                "authorized_split": "train",
                "instance_ids": ["train_0"],
            }
        )
        manifest_data_payload = (
            b"INSTANCE,vh_seq,vl_seq,ag_seq\n"
            b"train_0,ACDEF,,GGGGG\n"
        )
        authority_path = root / "train-authority.json"
        manifest_data_path = root / "train-manifest.csv"
        authority_path.write_bytes(authority_payload)
        manifest_data_path.write_bytes(manifest_data_payload)
        environment = os.environ.copy()
        environment.update(
            {
                "SOURCE_ROOT": str(source),
                "OUTPUT_ROOT": str(output),
                "RUN_ID": "positive-stage-test",
                "TRAIN_AUTHORITY_ALLOWLIST": str(authority_path),
                "TRAIN_CANONICAL_MANIFEST": str(manifest_data_path),
                "EXPECTED_TRAIN_AUTHORITY_SHA256": hashlib.sha256(
                    authority_payload
                ).hexdigest(),
                "EXPECTED_TRAIN_MANIFEST_SHA256": hashlib.sha256(
                    manifest_data_payload
                ).hexdigest(),
                "EXPECTED_SOURCE_MANIFEST_SHA256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
                "EXPECTED_BUILDER_SHA256": source_digests[
                    "scripts/build_train_only_nested_neighborhood_folds.py"
                ],
                "EXPECTED_AUDITOR_SHA256": source_digests[
                    "scripts/audit_train_only_nested_neighborhood_folds.py"
                ],
                "EXPECTED_JOINT_MILP_SHA256": source_digests[
                    "src/interaction_first/nested_joint_milp.py"
                ],
            }
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", bootstrap],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        run_root = output / "slurm-positive-stage-test"
        stage = run_root / "stage"
        try:
            for relative, payload in source_payloads.items():
                assert (stage / "source" / relative).read_bytes() == payload
            assert (
                stage / "inputs" / "train_authority_allowlist.json"
            ).read_bytes() == authority_payload
            assert (
                stat.S_IMODE(stage.stat().st_mode) == 0o500
            )
            inventory = json.loads(
                (
                    stage / "inventory" / "stage_inventory.json"
                ).read_text(encoding="utf-8")
            )
            assert set(inventory["source_entries"]) == set(source_payloads)
            assert inventory[
                "exact_three_file_staged_application_source_closure"
            ]
            assert not inventory["full_runtime_closure_attested"]
            assert not inventory["full_split_or_sealed_manifest_input_accepted"]
        finally:
            for current, directories, files in os.walk(
                stage,
                topdown=False,
            ):
                current_path = Path(current)
                for filename in files:
                    os.chmod(current_path / filename, 0o600)
                for dirname in directories:
                    os.chmod(current_path / dirname, 0o700)
                os.chmod(current_path, 0o700)


def test_slurm_final_inventory_seals_artifact_tree() -> None:
    """Only the post-seal sibling receipt claims final launcher success."""
    launcher_path = (
        Path(builder.__file__).resolve().parents[1]
        / "slurm"
        / "frontier_nested_split_build_audit.sbatch"
    )
    launcher = launcher_path.read_text(encoding="utf-8")
    marker = (
        '"$PYTHON_BIN" -I -S -B - \\\n'
        '  "$ARTIFACT_ROOT" "$RUN_ROOT/launcher_success.json" <<\'PY\'\n'
    )
    start = launcher.index(marker) + len(marker)
    end = launcher.index("\nPY\n", start)
    inventory_program = launcher[start : end + 1]
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        payload_path = artifacts / "candidate.json"
        payload_path.write_bytes(b'{"candidate":true}\n')
        success_path = root / "launcher_success.json"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    inventory_program,
                    str(artifacts),
                    str(success_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
            inventory_path = artifacts / "artifact_sha256.json"
            inventory = json.loads(
                inventory_path.read_text(encoding="utf-8")
            )
            assert not inventory["successful_launcher_completion"]
            assert not inventory["artifact_tree_sealed_after_inventory"]
            assert inventory[
                "artifact_tree_seal_pending_at_inventory_publication"
            ]
            assert inventory["separate_post_seal_success_receipt_required"]
            assert inventory["artifacts"]["candidate.json"]["sha256"] == (
                hashlib.sha256(payload_path.read_bytes()).hexdigest()
            )
            success = json.loads(success_path.read_text(encoding="utf-8"))
            assert success["successful_launcher_completion"]
            assert success["artifact_tree_sealed_after_inventory"]
            assert success[
                "artifact_tree_post_seal_hash_and_mode_verification_passed"
            ]
            assert success["artifact_tree_post_seal_fsync_completed"]
            assert success["artifact_inventory_sha256"] == hashlib.sha256(
                inventory_path.read_bytes()
            ).hexdigest()
            assert success["success_receipt_is_outside_artifact_inventory"]
            assert not success["full_run_root_tree_sealed"]
            assert stat.S_IMODE(artifacts.stat().st_mode) == 0o500
            assert stat.S_IMODE(payload_path.stat().st_mode) == 0o400
            assert stat.S_IMODE(inventory_path.stat().st_mode) == 0o400
            assert stat.S_IMODE(success_path.stat().st_mode) == 0o400
        finally:
            for current, directories, files in os.walk(
                artifacts,
                topdown=False,
            ):
                current_path = Path(current)
                for filename in files:
                    os.chmod(current_path / filename, 0o600)
                for dirname in directories:
                    os.chmod(current_path / dirname, 0o700)
                os.chmod(current_path, 0o700)
            if success_path.exists():
                os.chmod(success_path, 0o600)


def test_slurm_finalizer_failure_cannot_leave_false_success_receipt() -> None:
    """A fault after inventory publication leaves only conservative evidence."""
    launcher_path = (
        Path(builder.__file__).resolve().parents[1]
        / "slurm"
        / "frontier_nested_split_build_audit.sbatch"
    )
    launcher = launcher_path.read_text(encoding="utf-8")
    marker = (
        '"$PYTHON_BIN" -I -S -B - \\\n'
        '  "$ARTIFACT_ROOT" "$RUN_ROOT/launcher_success.json" <<\'PY\'\n'
    )
    start = launcher.index(marker) + len(marker)
    end = launcher.index("\nPY\n", start)
    inventory_program = launcher[start : end + 1]
    seal_loop = (
        "for current, directories, files in os.walk(\n"
        "    root,\n"
        "    topdown=False,\n"
        "    followlinks=False,\n"
        "):"
    )
    assert inventory_program.count(seal_loop) == 1
    faulted_program = inventory_program.replace(
        seal_loop,
        'raise RuntimeError("injected post-inventory seal failure")\n'
        + seal_loop,
        1,
    )
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        (artifacts / "candidate.json").write_bytes(b'{"candidate":true}\n')
        success_path = root / "launcher_success.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                faulted_program,
                str(artifacts),
                str(success_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        inventory = json.loads(
            (artifacts / "artifact_sha256.json").read_text(encoding="utf-8")
        )
        assert not inventory["successful_launcher_completion"]
        assert not inventory["artifact_tree_sealed_after_inventory"]
        assert inventory["artifact_tree_seal_pending_at_inventory_publication"]
        assert not success_path.exists()


def test_builder_is_create_only() -> None:
    """A builder rerun must preserve already-published bytes."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        args = _fixture(Path(directory))
        _build_toy(args)
        original = args.output.read_bytes()
        try:
            _build_toy(args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Builder unexpectedly overwrote an assignment")
        assert args.output.read_bytes() == original


def test_greedy_assignment_cannot_receive_exact_joint_authorization() -> None:
    """Boundary success alone must not authorize a sequential greedy split."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args = _fixture(root)
        assignment = _build_toy(build_args)
        audit_output = root / "audit.json"
        audit_args = _audit_args(
            build_args,
            assignment,
            build_args.output,
            audit_output,
        )
        with (
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                side_effect=AssertionError("alignment must not start"),
            ),
            mock.patch.object(
                auditor,
                "independently_recomputed_outer_policy",
                side_effect=_toy_policy_payload,
            ),
            mock.patch.object(
                auditor,
                "independently_recomputed_inner_policy",
                side_effect=_toy_policy_payload,
            ),
            pytest.raises(ValueError, match="externally pinned replay"),
        ):
            auditor.run(audit_args)
        assert not audit_output.exists()


def test_auditor_rejects_rehashed_outer_policy_tampering() -> None:
    """A self-consistently rehashed outer policy cannot replace the locked policy."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(root, "outer-policy")
        tampered = json.loads(json.dumps(assignment))
        tampered["construction"]["outer_level"]["policy"][
            "minimum_training_pair_fraction"
        ] = 0.49
        tampered_path = root / "tampered-outer.json"
        _write_rehashed_assignment(tampered_path, tampered)
        audit_args = _audit_args(
            build_args,
            tampered,
            tampered_path,
            root / "tampered-outer-audit.json",
        )
        with (
            mock.patch.object(
                auditor,
                "independently_recomputed_outer_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=9,
                ),
            ),
            mock.patch.object(
                auditor,
                "independently_recomputed_inner_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=8,
                ),
            ),
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                side_effect=AssertionError("alignment must not start"),
            ),
            mock.patch.object(
                auditor,
                "LOCKED_JOINT_POLICY_SHA256",
                build_args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                nested_joint_milp,
                "LOCKED_JOINT_POLICY_SHA256",
                build_args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                auditor,
                "load_verified_joint_module",
                return_value=nested_joint_milp,
            ),
        ):
            try:
                auditor.run(audit_args)
            except ValueError as error:
                assert (
                    "does not match the independently recomputed locked outer policy"
                    in str(error)
                )
            else:
                raise AssertionError("A rehashed outer-policy change was accepted")
        assert not audit_args.output.exists()


def test_auditor_rejects_rehashed_inner_policy_tampering() -> None:
    """A self-consistently rehashed inner policy cannot replace the locked policy."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args, assignment = _build_pinned_joint_toy(root, "inner-policy")
        tampered = json.loads(json.dumps(assignment))
        tampered["construction"]["outer_level"]["folds"][0]["inner_level"][
            "policy"
        ]["maximum_quarantine_pair_fraction"] = 0.51
        tampered_path = root / "tampered-inner.json"
        _write_rehashed_assignment(tampered_path, tampered)
        audit_args = _audit_args(
            build_args,
            tampered,
            tampered_path,
            root / "tampered-inner-audit.json",
        )
        with (
            mock.patch.object(
                auditor,
                "independently_recomputed_outer_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=9,
                ),
            ),
            mock.patch.object(
                auditor,
                "independently_recomputed_inner_policy",
                side_effect=lambda count: _toy_joint_policy_payload(
                    count,
                    upper_numerator=1,
                    upper_denominator=8,
                ),
            ),
            mock.patch.object(
                auditor,
                "build_locked_aligner",
                side_effect=AssertionError("alignment must not start"),
            ),
            mock.patch.object(
                auditor,
                "LOCKED_JOINT_POLICY_SHA256",
                build_args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                nested_joint_milp,
                "LOCKED_JOINT_POLICY_SHA256",
                build_args.expected_joint_policy_sha256,
            ),
            mock.patch.object(
                auditor,
                "load_verified_joint_module",
                return_value=nested_joint_milp,
            ),
        ):
            try:
                auditor.run(audit_args)
            except ValueError as error:
                assert (
                    "does not match the independently recomputed locked inner policy"
                    in str(error)
                )
            else:
                raise AssertionError("A rehashed inner-policy change was accepted")
        assert not audit_args.output.exists()


@pytest.mark.parametrize("legacy_generation", ("v3", "v4"))
def test_v5_auditor_rejects_legacy_assignment_formats(
    legacy_generation: str,
) -> None:
    """The v5 auditor must reject both historical assignment generations."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        build_args = _fixture(root)
        assignment = _build_toy(build_args)
        legacy = json.loads(json.dumps(assignment))
        legacy["format_version"] = (
            "train_only_nested_direct80_four_view_neighborhood_folds_"
            f"{legacy_generation}"
        )
        legacy.pop("scientific_payload_sha256")
        legacy["scientific_payload_sha256"] = auditor.canonical_payload_sha256(
            legacy
        )
        legacy_path = root / f"legacy-{legacy_generation}.json"
        legacy_path.write_bytes(auditor.canonical_json_bytes(legacy))
        audit_output = root / "legacy-audit.json"
        audit_args = _audit_args(
            build_args,
            legacy,
            legacy_path,
            audit_output,
        )
        try:
            auditor.run(audit_args)
        except ValueError as error:
            assert "Unsupported nested assignment format_version" in str(error)
        else:
            raise AssertionError(
                f"The legacy {legacy_generation} assignment was accepted"
            )
        assert not audit_output.exists()


def test_exact_boundary_audit_detects_inclusive_80_percent_crossing() -> None:
    """A four-of-five match must be recorded as a threshold crossing."""
    a_train = auditor.independent_antibody_id("AAAAA", "")
    a_validation = auditor.independent_antibody_id("AAAAT", "")
    g_train = auditor.independent_antigen_id("CCCCC")
    g_validation = auditor.independent_antigen_id("DDDDD")
    p_train = auditor.independent_pair_id(a_train, g_train)
    p_validation = auditor.independent_pair_id(a_validation, g_validation)
    pairs = {
        p_train: auditor.IndependentPair(p_train, a_train, g_train, ("t",)),
        p_validation: auditor.IndependentPair(
            p_validation, a_validation, g_validation, ("v",)
        ),
    }
    boundary = auditor.Boundary("toy", (p_train,), (p_validation,))
    with mock.patch.object(
        auditor, "build_locked_aligner", return_value=FakeAligner()
    ):
        result = auditor.audit_boundary_view(
            boundary,
            "vh",
            pairs,
            {a_train: ("AAAAA", ""), a_validation: ("AAAAT", "")},
            {g_train: "CCCCC", g_validation: "DDDDD"},
            workers=1,
            chunk_size=1,
        )
    assert result["threshold_crossing_edge_count"] == 1
    assert not result["zero_threshold_crossing_edges"]
    assert result["crossing_edges"][0]["aligned_region_identity"] == 0.8
    assert (
        result["directional_dynamic_programming_alignment_execution_count"]
        == 2
    )
    assert result["every_aligned_pair_evaluated_in_both_orientations"]
    assert result["crossing_edges"][0][
        "qualifying_alignment_orientations"
    ] == ["training_to_validation", "validation_to_training"]
    encoded = json.dumps(result["crossing_edges"][0], sort_keys=True)
    assert "AAAAA" not in encoded
    assert "AAAAT" not in encoded


def test_input_hash_tampering_fails_before_publication() -> None:
    """A byte mismatch must fail without creating an output."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        args = _fixture(Path(directory))
        args.train_canonical_manifest_sha256 = hashlib.sha256(b"wrong").hexdigest()
        with (
            mock.patch.object(
                builder, "build_locked_aligner", return_value=FakeAligner()
            ),
            mock.patch.object(builder, "outer_policy", side_effect=_toy_policy),
            mock.patch.object(builder, "inner_policy", side_effect=_toy_policy),
        ):
            try:
                builder.run(args)
            except ValueError as error:
                assert "SHA-256 mismatch" in str(error)
            else:
                raise AssertionError("Tampered input was accepted")
        assert not args.output.exists()


def test_assignment_scientific_digest_detects_post_build_tampering() -> None:
    """Independent scientific digest verification must reject changed content."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        args = _fixture(Path(directory))
        assignment = _build_toy(args)
        assert (
            auditor.verify_scientific_digest(assignment)
            == assignment["scientific_payload_sha256"]
        )
        tampered = json.loads(json.dumps(assignment))
        tampered["construction"]["selection_seed"] += 1
        try:
            auditor.verify_scientific_digest(tampered)
        except ValueError as error:
            assert "mismatch" in str(error)
        else:
            raise AssertionError("Tampered scientific content was accepted")
