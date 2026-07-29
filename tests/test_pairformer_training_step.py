"""Adversarial tests for the train-only PAIRFormer execution boundary."""

from __future__ import annotations

from dataclasses import replace
import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _digest(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _schema():
    from interaction_first.pair_geometry_schema import PairGeometrySchema

    return PairGeometrySchema(
        distance_bin_edges_angstrom=(2.0, 4.5, 8.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=("ab_azimuth", "ab_polar"),
        orientation_axis_is_cyclic=(True, False),
        orientation_bin_edges_radians=((-1.0, 1.0), (1.0, 2.0)),
        label_format_version="pair_geometry_labels_v2",
        orientation_schema_version="test_orientation_schema_v1",
        primary_contact_definition="test_positive_or_complete_contact",
        distance_definition="test_complete_pair_distance",
        orientation_definition="test_right_handed_frames",
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def _model():
    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    return ProspectivePairGeometryModel(
        antibody_input_dim=4,
        antigen_input_dim=5,
        d_model=8,
        pair_dim=6,
        num_heads=2,
        num_fusion_layers=0,
        ffn_dim=16,
        geometry_schema=_schema(),
        pair_chunk_size=2,
        dropout=0.0,
        centering_iterations=8,
    )


def _unsealed_batch(
    *,
    token: str = "batch-0",
    role: str = "train",
    decoys: bool = False,
):
    import torch

    from interaction_first.pairformer_training_step import (
        PairFormerTrainBatch,
        TrainOnlyDecoyProvenance,
        TrainOnlyLabelProvenance,
    )
    from interaction_first.hard_negative_contract import (
        BindingEvidence,
        ControlFamily,
    )

    torch.manual_seed(71)
    antibody_features = torch.randn(2, 2, 4)
    antigen_features = torch.randn(2, 3, 5)
    antibody_mask = torch.ones(2, 2, dtype=torch.bool)
    antigen_mask = torch.ones(2, 3, dtype=torch.bool)
    candidate_pair_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    candidate_pair_mask[0, 1, 2] = False

    distance = torch.tensor(
        [
            [[0, 2, -1], [3, 1, -1]],
            [[2, -1, -1], [0, 3, -1]],
        ],
        dtype=torch.long,
    )
    distance_mask = distance >= 0
    contact = torch.full(distance.shape, -1.0)
    contact[distance_mask] = (distance[distance_mask] < 2).float()
    contact_mask = distance_mask.clone()
    orientation = torch.full((2, 2, 3, 2), -1, dtype=torch.long)
    orientation[0, 0, 0] = torch.tensor([1, 2])
    orientation[0, 1, 1, 0] = 0
    orientation[1, 1, 0] = torch.tensor([2, 1])
    orientation_mask = orientation >= 0

    example_ids = (f"{token}-a", f"{token}-b")
    provenance = tuple(
        TrainOnlyLabelProvenance(
            example_id=example_id,
            source_role="train",
            outer_role="train",
            inner_role="train",
            outer_fold=1,
            inner_fold=2,
            split_manifest_sha256=_digest("split"),
            train_membership_receipt_sha256=_digest("train-membership"),
            source_semantic_type="experimentally_resolved_pair_geometry",
            structure_artifact_sha256=_digest(f"structure-{example_id}"),
            label_artifact_sha256=_digest(f"label-{example_id}"),
            label_builder_sha256=_digest("label-builder"),
            geometry_schema_sha256=_schema().sha256,
            example_train_membership_binding_sha256=_digest(
                f"example-membership-{example_id}"
            ),
            dataset_admission_receipt_sha256=_digest("dataset-admission"),
        )
        for example_id in example_ids
    )
    decoy_scores = None
    decoy_mask = None
    decoy_semantic_types = ((), ())
    decoy_provenance = ((), ())
    decoy_scorer_source_sha256 = None
    if decoys:
        decoy_score_source = torch.tensor(
            [[0.2, -0.1], [0.7, 9.0]],
            requires_grad=True,
        )
        decoy_scores = decoy_score_source * 1.0
        decoy_mask = torch.tensor([[True, True], [True, False]])
        decoy_semantic_types = (
            ("measured_non_binder", "unknown_unmeasured_decoy"),
            ("wrong_pose", "padding"),
        )
        decoy_provenance = (
            (
                TrainOnlyDecoyProvenance(
                    semantic_type=ControlFamily.MEASURED_NON_BINDER,
                    binding_evidence=BindingEvidence.MEASURED_NON_BINDER,
                    source_role="train",
                    record_id=f"{token}-nonbinder",
                    match_group_id=f"{token}-group-a",
                    parent_cognate_id=example_ids[0],
                    outer_fold=1,
                    inner_fold=2,
                    split_manifest_sha256=_digest("split"),
                    train_membership_receipt_sha256=_digest("train-membership"),
                    train_allowlist_sha256=_digest("train-allowlist"),
                    hard_negative_manifest_sha256=_digest("hard-negative-manifest"),
                    evidence_registry_sha256=_digest("evidence-registry"),
                    evidence_authority_receipt_sha256=_digest(
                        "evidence-authority"
                    ),
                    control_record_sha256=_digest("control-record-0"),
                    source_artifact_sha256=_digest("measured-nonbinder-source"),
                    control_support_sha256=_digest("control-support-0"),
                    measured_evidence_binding_sha256=_digest(
                        "measured-evidence-0"
                    ),
                ),
                TrainOnlyDecoyProvenance(
                    semantic_type=ControlFamily.UNKNOWN_UNMEASURED_DECOY,
                    binding_evidence=BindingEvidence.UNKNOWN_UNMEASURED,
                    source_role="train",
                    record_id=f"{token}-unknown-partner",
                    match_group_id=f"{token}-group-a",
                    parent_cognate_id=example_ids[0],
                    outer_fold=1,
                    inner_fold=2,
                    split_manifest_sha256=_digest("split"),
                    train_membership_receipt_sha256=_digest("train-membership"),
                    train_allowlist_sha256=_digest("train-allowlist"),
                    hard_negative_manifest_sha256=_digest("hard-negative-manifest"),
                    evidence_registry_sha256=_digest("evidence-registry"),
                    evidence_authority_receipt_sha256=_digest(
                        "evidence-authority"
                    ),
                    control_record_sha256=_digest("control-record-1"),
                    source_artifact_sha256=_digest("unknown-partner-source"),
                    control_support_sha256=_digest("control-support-1"),
                    measured_evidence_binding_sha256=None,
                ),
            ),
            (
                TrainOnlyDecoyProvenance(
                    semantic_type=ControlFamily.WRONG_POSE,
                    binding_evidence=BindingEvidence.UNKNOWN_UNMEASURED,
                    source_role="train",
                    record_id=f"{token}-wrong-pose",
                    match_group_id=f"{token}-group-b",
                    parent_cognate_id=example_ids[1],
                    outer_fold=1,
                    inner_fold=2,
                    split_manifest_sha256=_digest("split"),
                    train_membership_receipt_sha256=_digest("train-membership"),
                    train_allowlist_sha256=_digest("train-allowlist"),
                    hard_negative_manifest_sha256=_digest("hard-negative-manifest"),
                    evidence_registry_sha256=_digest("evidence-registry"),
                    evidence_authority_receipt_sha256=_digest(
                        "evidence-authority"
                    ),
                    control_record_sha256=_digest("control-record-2"),
                    source_artifact_sha256=_digest("wrong-pose-source"),
                    control_support_sha256=_digest("control-support-2"),
                    measured_evidence_binding_sha256=None,
                ),
                None,
            ),
        )
        decoy_scorer_source_sha256 = _digest("decoy-scorer")

    return PairFormerTrainBatch(
        role=role,
        outer_fold=1,
        inner_fold=2,
        split_manifest_sha256=_digest("split"),
        batch_manifest_sha256=_digest(token),
        feature_provenance_sha256=_digest(f"features-{token}"),
        geometry_schema_sha256=_schema().sha256,
        candidate_mask_sha256="",
        batch_contract_sha256="",
        example_ids=example_ids,
        label_provenance=provenance,
        decoy_semantic_types=decoy_semantic_types,
        decoy_provenance=decoy_provenance,
        decoy_scorer_source_sha256=decoy_scorer_source_sha256,
        antibody_features=antibody_features,
        antigen_features=antigen_features,
        antibody_mask=antibody_mask,
        antigen_mask=antigen_mask,
        candidate_pair_mask=candidate_pair_mask,
        contact_targets=contact,
        contact_mask=contact_mask,
        distance_bin_targets=distance,
        distance_mask=distance_mask,
        orientation_bin_targets=orientation,
        orientation_mask=orientation_mask,
        decoy_scores=decoy_scores,
        decoy_mask=decoy_mask,
    )


def _sealed_batch(**kwargs):
    from interaction_first.pairformer_training_step import seal_train_only_batch

    return seal_train_only_batch(_unsealed_batch(**kwargs), _schema())


def _batch_authority(batch):
    from interaction_first.pairformer_training_step import (
        build_train_only_batch_authority,
    )

    return build_train_only_batch_authority(
        batch,
        _schema(),
        gate0_policy_sha256=_digest("gate0-policy"),
        gate0_decision_sha256=_digest("gate0-decision"),
        launcher_receipt_sha256=_digest("launcher-receipt"),
    )


def _forward(
    *,
    token: str = "batch-0",
    accumulation_index: int = 0,
    autocast_policy=None,
):
    from interaction_first.pairformer_training_step import (
        build_pairformer_microbatch_statistics,
        derive_seed_receipt,
    )

    model = _model()
    batch = _sealed_batch(token=token)
    authority = _batch_authority(batch)
    seed = derive_seed_receipt(
        base_seed=20260725,
        optimizer_step=3,
        accumulation_index=accumulation_index,
        rank=0,
        batch_contract_sha256=batch.batch_contract_sha256,
        run_manifest_sha256=_digest("run-manifest"),
        checkpoint_parent_sha256=_digest("checkpoint-parent"),
    )
    kwargs = {
        "seed_receipt": seed,
        "batch_authority": authority,
        "expected_batch_authority_sha256": authority.sha256,
    }
    if autocast_policy is not None:
        kwargs["autocast_policy"] = autocast_policy
    result = build_pairformer_microbatch_statistics(model, batch, **kwargs)
    return model, batch, result


def _manual_statistics(
    *,
    numerator,
    denominator: int,
    rank: int,
    accumulation_index: int = 0,
    batch_token: str = "manual",
    autocast_policy=None,
):
    import torch

    from interaction_first.pairformer_training_step import (
        AutocastPolicy,
        LOSS_TERM_ORDER,
        PairFormerMicrobatchStatistics,
        SourceCodeReceipt,
        derive_seed_receipt,
    )

    zero = numerator * 0.0
    batch_hash = _digest(f"{batch_token}-{rank}-{accumulation_index}")
    seed = derive_seed_receipt(
        base_seed=1,
        optimizer_step=0,
        accumulation_index=accumulation_index,
        rank=rank,
        batch_contract_sha256=batch_hash,
        run_manifest_sha256=_digest("manual-run"),
        checkpoint_parent_sha256=_digest("manual-checkpoint"),
    )
    result = PairFormerMicrobatchStatistics(
        term_numerators=(numerator,) + tuple(
            zero for _ in range(len(LOSS_TERM_ORDER) - 1)
        ),
        term_denominators=(denominator,) + tuple(
            0 for _ in range(len(LOSS_TERM_ORDER) - 1)
        ),
        term_weights=(1.0,) + tuple(
            0.0 for _ in range(len(LOSS_TERM_ORDER) - 1)
        ),
        schema_sha256=_digest("schema"),
        source_receipt=SourceCodeReceipt.current(),
        model_parameter_ids=(1,),
        model_parameter_manifest_sha256=_digest("model-parameters"),
        model_state_sha256=_digest("manual-model-state"),
        batch_contract_sha256=batch_hash,
        batch_authority_sha256=_digest(
            f"manual-batch-authority-{rank}-{accumulation_index}"
        ),
        seed_receipt=seed,
        autocast_policy=(
            AutocastPolicy() if autocast_policy is None else autocast_policy
        ),
    )
    result.validate()
    return result


def _sum_authority(statistics_by_rank, reduced_packet, *, world_size: int):
    from interaction_first.pairformer_training_step import (
        build_sum_reduction_authority,
    )

    return build_sum_reduction_authority(
        statistics_by_rank,
        reduced_packet,
        world_size=world_size,
        collective_id_sha256=_digest("collective"),
        process_group_manifest_sha256=_digest("process-group"),
        issuer_receipt_sha256=_digest("collective-authority"),
        backend="test",
    )


def _authorized_loss(statistics, reduced_packet):
    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
    )

    authority = _sum_authority(
        {0: statistics},
        reduced_packet,
        world_size=1,
    )
    loss = compose_ddp_accumulation_loss(
        statistics,
        reduced_packet,
        world_size=1,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    return loss, authority


def test_train_forward_exposes_exact_macro_numerators_and_denominators() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        LOSS_TERM_ORDER,
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        recompute_reduced_equal_complex_objective,
    )

    _, _, result = _forward()
    statistics = result.statistics
    assert statistics.term_denominators == (2, 2, 2, 0, 0, 0, 0)
    packet = pack_detached_loss_statistics([statistics], world_size=1)
    global_objective = recompute_reduced_equal_complex_objective(
        packet,
        statistics.term_weights,
        world_size=1,
    )
    authority = _sum_authority({0: [statistics]}, packet, world_size=1)
    ddp_loss = compose_ddp_accumulation_loss(
        [statistics],
        packet,
        world_size=1,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    torch.testing.assert_close(global_objective.float(), result.local_objective)
    torch.testing.assert_close(ddp_loss, result.local_objective)
    assert packet.shape == (3, len(LOSS_TERM_ORDER))
    assert not packet.requires_grad
    assert result.outputs["executed_heads"] == (
        "contact",
        "distance",
        "orientation",
    )
    assert result.outputs["pair_representations"] is None


def test_ddp_like_wrapper_executes_through_the_wrapper_boundary() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        build_pairformer_microbatch_statistics,
        derive_seed_receipt,
    )

    class RecordingWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.called = False

        def forward(self, *args, **kwargs):
            self.called = True
            return self.module(*args, **kwargs)

    wrapped = RecordingWrapper(_model())
    batch = _sealed_batch(token="wrapped")
    authority = _batch_authority(batch)
    seed = derive_seed_receipt(
        base_seed=3,
        optimizer_step=0,
        accumulation_index=0,
        rank=0,
        batch_contract_sha256=batch.batch_contract_sha256,
        run_manifest_sha256=_digest("wrapped-run"),
        checkpoint_parent_sha256=_digest("wrapped-checkpoint"),
    )
    result = build_pairformer_microbatch_statistics(
        wrapped,
        batch,
        seed_receipt=seed,
        batch_authority=authority,
        expected_batch_authority_sha256=authority.sha256,
    )
    assert wrapped.called
    assert result.statistics.model_parameter_ids == tuple(
        sorted(id(parameter) for parameter in wrapped.parameters())
    )


@pytest.mark.parametrize("role", ["validation", "val", "test", "sealed"])
def test_nontraining_roles_are_rejected_before_model_execution(role: str) -> None:
    from interaction_first.pairformer_training_step import seal_train_only_batch

    with pytest.raises(ValueError, match="role='train' only"):
        seal_train_only_batch(_unsealed_batch(role=role), _schema())


def test_predicted_or_nontrain_label_provenance_is_rejected() -> None:
    from interaction_first.pairformer_training_step import seal_train_only_batch

    batch = _unsealed_batch()
    predicted = replace(
        batch.label_provenance[0],
        source_semantic_type="predicted_pair_geometry",
    )
    with pytest.raises(ValueError, match="experimentally resolved"):
        seal_train_only_batch(
            replace(
                batch,
                label_provenance=(predicted, batch.label_provenance[1]),
            ),
            _schema(),
        )
    wrong_role = replace(batch.label_provenance[0], source_role="validation")
    with pytest.raises(ValueError, match="source_role"):
        seal_train_only_batch(
            replace(
                batch,
                label_provenance=(wrong_role, batch.label_provenance[1]),
            ),
            _schema(),
        )


def test_candidate_and_target_tampering_break_the_sealed_contract() -> None:
    from interaction_first.pairformer_training_step import validate_train_only_batch

    batch = _sealed_batch()
    candidate = batch.candidate_pair_mask.clone()
    candidate[0, 0, 0] = False
    with pytest.raises(ValueError, match="candidate_pair_mask"):
        validate_train_only_batch(
            replace(batch, candidate_pair_mask=candidate),
            _schema(),
        )

    target = batch.distance_bin_targets.clone()
    target[0, 0, 0] = 1
    with pytest.raises(ValueError, match="sealed contract"):
        validate_train_only_batch(
            replace(batch, distance_bin_targets=target),
            _schema(),
        )

    features = batch.antibody_features.clone()
    features[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="sealed contract"):
        validate_train_only_batch(
            replace(batch, antibody_features=features),
            _schema(),
        )


def test_feature_dtype_mask_device_and_candidate_support_are_explicit() -> None:
    import torch

    from interaction_first.pairformer_training_step import seal_train_only_batch

    batch = _unsealed_batch()
    with pytest.raises(ValueError, match="floating dtype"):
        seal_train_only_batch(
            replace(batch, antibody_features=batch.antibody_features.long()),
            _schema(),
        )
    with pytest.raises(ValueError, match="boolean residue masks"):
        seal_train_only_batch(
            replace(batch, antibody_mask=batch.antibody_mask.float()),
            _schema(),
        )
    candidate = batch.candidate_pair_mask.clone()
    antigen_mask = batch.antigen_mask.clone()
    antigen_mask[0, 0] = False
    candidate[0, 0, 0] = True
    with pytest.raises(ValueError, match="unsupported residues"):
        seal_train_only_batch(
            replace(
                batch,
                antigen_mask=antigen_mask,
                candidate_pair_mask=candidate,
            ),
            _schema(),
        )
    nonfinite = batch.antibody_features.clone()
    nonfinite[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        seal_train_only_batch(
            replace(batch, antibody_features=nonfinite),
            _schema(),
        )


def test_schema_and_contract_hashes_are_fail_closed() -> None:
    from interaction_first.pairformer_training_step import validate_train_only_batch

    batch = _sealed_batch()
    with pytest.raises(ValueError, match="model schema"):
        validate_train_only_batch(
            replace(batch, geometry_schema_sha256=_digest("wrong-schema")),
            _schema(),
        )
    with pytest.raises(ValueError, match="sealed contract"):
        validate_train_only_batch(
            replace(batch, batch_contract_sha256=_digest("wrong-contract")),
            _schema(),
        )


def test_batch_authority_is_external_exact_and_schema_bound() -> None:
    from interaction_first.pairformer_training_step import (
        build_train_only_batch_authority,
        seal_train_only_batch,
        validate_train_only_batch_authority,
    )

    batch = _sealed_batch()
    authority = _batch_authority(batch)
    validate_train_only_batch_authority(
        batch,
        _schema(),
        authority,
        expected_authority_sha256=authority.sha256,
    )
    with pytest.raises(ValueError, match="external expected digest"):
        validate_train_only_batch_authority(
            batch,
            _schema(),
            authority,
            expected_authority_sha256=_digest("wrong-authority"),
        )

    unsealed = _unsealed_batch()
    labels = list(unsealed.label_provenance)
    labels[0] = replace(
        labels[0],
        geometry_schema_sha256=_digest("wrong-label-schema"),
    )
    with pytest.raises(ValueError, match="exact batch geometry schema"):
        seal_train_only_batch(
            replace(unsealed, label_provenance=tuple(labels)),
            _schema(),
        )

    rebuilt = build_train_only_batch_authority(
        batch,
        _schema(),
        gate0_policy_sha256=_digest("other-policy"),
        gate0_decision_sha256=authority.gate0_decision_sha256,
        launcher_receipt_sha256=authority.launcher_receipt_sha256,
    )
    assert rebuilt.sha256 != authority.sha256


def test_decoy_semantics_distinguish_measured_unknown_and_padding() -> None:
    from interaction_first.pairformer_training_step import (
        seal_train_only_batch,
        validate_train_only_batch,
    )

    batch = _sealed_batch(decoys=True)
    validate_train_only_batch(batch, _schema())
    semantics = list(list(row) for row in batch.decoy_semantic_types)
    semantics[0][0] = "unknown_negative"
    with pytest.raises(ValueError, match="canonical ControlFamily"):
        seal_train_only_batch(
            replace(
                _unsealed_batch(decoys=True),
                decoy_semantic_types=tuple(tuple(row) for row in semantics),
            ),
            _schema(),
        )
    semantics = list(list(row) for row in batch.decoy_semantic_types)
    semantics[1][1] = "measured_non_binder"
    with pytest.raises(ValueError, match="Inactive decoy slots"):
        seal_train_only_batch(
            replace(
                _unsealed_batch(decoys=True),
                decoy_semantic_types=tuple(tuple(row) for row in semantics),
            ),
            _schema(),
        )
    provenance = [list(row) for row in batch.decoy_provenance]
    provenance[0][0] = None
    with pytest.raises(ValueError, match="manifest-bound provenance"):
        seal_train_only_batch(
            replace(
                _unsealed_batch(decoys=True),
                decoy_provenance=tuple(tuple(row) for row in provenance),
            ),
            _schema(),
        )
    provenance = [list(row) for row in batch.decoy_provenance]
    provenance[1][1] = provenance[0][0]
    with pytest.raises(ValueError, match="Padding decoy slots"):
        seal_train_only_batch(
            replace(
                _unsealed_batch(decoys=True),
                decoy_provenance=tuple(tuple(row) for row in provenance),
            ),
            _schema(),
        )

    provenance = [list(row) for row in batch.decoy_provenance]
    provenance[0][0] = replace(
        provenance[0][0],
        measured_evidence_binding_sha256=None,
    )
    with pytest.raises(ValueError, match="Measured decoys"):
        seal_train_only_batch(
            replace(
                _unsealed_batch(decoys=True),
                decoy_provenance=tuple(tuple(row) for row in provenance),
            ),
            _schema(),
        )


def test_exact_split_parent_and_manifest_linkage_is_fail_closed() -> None:
    from interaction_first.pairformer_training_step import seal_train_only_batch

    labels = list(_unsealed_batch().label_provenance)
    labels[0] = replace(labels[0], outer_fold=0)
    with pytest.raises(ValueError, match="exact batch split and folds"):
        seal_train_only_batch(
            replace(_unsealed_batch(), label_provenance=tuple(labels)),
            _schema(),
        )

    base = _unsealed_batch(decoys=True)
    provenance = [list(row) for row in base.decoy_provenance]
    provenance[0][0] = replace(provenance[0][0], parent_cognate_id="other")
    with pytest.raises(ValueError, match="parent cognate"):
        seal_train_only_batch(
            replace(base, decoy_provenance=tuple(tuple(row) for row in provenance)),
            _schema(),
        )

    provenance = [list(row) for row in base.decoy_provenance]
    provenance[0][1] = replace(
        provenance[0][1],
        hard_negative_manifest_sha256=_digest("other-manifest"),
    )
    with pytest.raises(ValueError, match="share one manifest"):
        seal_train_only_batch(
            replace(base, decoy_provenance=tuple(tuple(row) for row in provenance)),
            _schema(),
        )

    provenance = [list(row) for row in base.decoy_provenance]
    provenance[1][0] = replace(
        provenance[1][0],
        record_id=provenance[0][0].record_id,
    )
    with pytest.raises(ValueError, match="cannot be replayed"):
        seal_train_only_batch(
            replace(base, decoy_provenance=tuple(tuple(row) for row in provenance)),
            _schema(),
        )


def test_decoy_semantics_round_trip_the_canonical_control_family_enum() -> None:
    from interaction_first.hard_negative_contract import (
        BindingEvidence,
        ControlFamily,
    )
    from interaction_first.pairformer_training_step import (
        ALLOWED_DECOY_CONTROL_FAMILIES,
        TrainOnlyDecoyProvenance,
    )

    expected = set(ControlFamily) - {ControlFamily.COGNATE}
    assert set(ALLOWED_DECOY_CONTROL_FAMILIES) == expected
    for family in sorted(expected, key=lambda value: value.value):
        receipt = TrainOnlyDecoyProvenance(
            semantic_type=family,
            binding_evidence=(
                BindingEvidence.MEASURED_NON_BINDER
                if family
                in {
                    ControlFamily.MEASURED_NON_BINDER,
                    ControlFamily.MUTATION_FLIP,
                }
                else BindingEvidence.UNKNOWN_UNMEASURED
            ),
            source_role="train",
            record_id=f"record-{family.value}",
            match_group_id=f"group-{family.value}",
            parent_cognate_id=f"cognate-{family.value}",
            outer_fold=1,
            inner_fold=2,
            split_manifest_sha256=_digest("split"),
            train_membership_receipt_sha256=_digest("train-membership"),
            train_allowlist_sha256=_digest("train-allowlist"),
            hard_negative_manifest_sha256=_digest("hard-negative-manifest"),
            evidence_registry_sha256=_digest("evidence-registry"),
            evidence_authority_receipt_sha256=_digest("evidence-authority"),
            control_record_sha256=_digest(f"record-{family.value}"),
            source_artifact_sha256=_digest(f"source-{family.value}"),
            control_support_sha256=_digest(f"support-{family.value}"),
            measured_evidence_binding_sha256=(
                _digest(f"evidence-{family.value}")
                if family.value in {"measured_non_binder", "mutation_flip"}
                else None
            ),
        )
        assert receipt.canonical_metadata()["semantic_type"] == family.value


def test_ranking_rejects_detached_or_constant_decoy_scores() -> None:
    from interaction_first.pair_geometry_losses import PairGeometryLossWeights
    from interaction_first.pairformer_training_step import (
        build_pairformer_microbatch_statistics,
        derive_seed_receipt,
        seal_train_only_batch,
    )

    unsealed = _unsealed_batch(decoys=True)
    batch = seal_train_only_batch(
        replace(unsealed, decoy_scores=unsealed.decoy_scores.detach()),
        _schema(),
    )
    authority = _batch_authority(batch)
    seed = derive_seed_receipt(
        base_seed=1,
        optimizer_step=0,
        accumulation_index=0,
        rank=0,
        batch_contract_sha256=batch.batch_contract_sha256,
        run_manifest_sha256=_digest("ranking-run"),
        checkpoint_parent_sha256=_digest("ranking-checkpoint"),
    )
    weights = PairGeometryLossWeights(
        contact_bce=0.0,
        distance_ce=0.0,
        orientation_ce=0.0,
        matched_decoy_ranking=1.0,
    )
    with pytest.raises(ValueError, match="current autograd graph"):
        build_pairformer_microbatch_statistics(
            _model(),
            batch,
            weights=weights,
            seed_receipt=seed,
            batch_authority=authority,
            expected_batch_authority_sha256=authority.sha256,
        )

    unrelated = _unsealed_batch(decoys=True)
    unrelated_batch = seal_train_only_batch(unrelated, _schema())
    unrelated_authority = _batch_authority(unrelated_batch)
    unrelated_seed = derive_seed_receipt(
        base_seed=1,
        optimizer_step=0,
        accumulation_index=0,
        rank=0,
        batch_contract_sha256=unrelated_batch.batch_contract_sha256,
        run_manifest_sha256=_digest("ranking-run"),
        checkpoint_parent_sha256=_digest("ranking-checkpoint"),
    )
    with pytest.raises(ValueError, match="current model parameters"):
        build_pairformer_microbatch_statistics(
            _model(),
            unrelated_batch,
            weights=weights,
            seed_receipt=unrelated_seed,
            batch_authority=unrelated_authority,
            expected_batch_authority_sha256=unrelated_authority.sha256,
        )


def test_external_sum_reduction_uses_global_counts_not_rank_mean_average() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        pack_detached_loss_statistics,
        recompute_reduced_equal_complex_objective,
    )

    rank_zero = _manual_statistics(
        numerator=torch.tensor(1.0, requires_grad=True),
        denominator=1,
        rank=0,
    )
    rank_one = _manual_statistics(
        numerator=torch.tensor(9.0, requires_grad=True),
        denominator=3,
        rank=1,
    )
    reduced = pack_detached_loss_statistics([rank_zero], world_size=2)
    reduced = reduced + pack_detached_loss_statistics(
        [rank_one],
        world_size=2,
    )
    objective = recompute_reduced_equal_complex_objective(
        reduced,
        rank_zero.term_weights,
        world_size=2,
    )
    assert objective.item() == pytest.approx(2.5)
    rank_mean_average = 0.5 * (1.0 / 1.0 + 9.0 / 3.0)
    assert rank_mean_average == pytest.approx(2.0)
    assert objective.item() != pytest.approx(rank_mean_average)


def test_sum_packet_rejects_mean_missing_duplicate_and_wrong_world_size() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
    )

    rank_zero = _manual_statistics(
        numerator=torch.tensor(1.0, requires_grad=True),
        denominator=1,
        rank=0,
    )
    rank_one = _manual_statistics(
        numerator=torch.tensor(3.0, requires_grad=True),
        denominator=1,
        rank=1,
    )
    packet_zero = pack_detached_loss_statistics([rank_zero], world_size=2)
    packet_one = pack_detached_loss_statistics([rank_one], world_size=2)
    reduced_sum = packet_zero + packet_one
    authority = _sum_authority(
        {0: [rank_zero], 1: [rank_one]},
        reduced_sum,
        world_size=2,
    )
    compose_ddp_accumulation_loss(
        [rank_zero],
        reduced_sum,
        world_size=2,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )

    with pytest.raises(ValueError, match="collective authority"):
        compose_ddp_accumulation_loss(
            [rank_zero],
            reduced_sum / 2.0,
            world_size=2,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )
    with pytest.raises(ValueError, match="collective authority"):
        compose_ddp_accumulation_loss(
            [rank_zero],
            packet_zero,
            world_size=2,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )
    with pytest.raises(ValueError, match="collective authority"):
        compose_ddp_accumulation_loss(
            [rank_zero],
            packet_zero + packet_zero,
            world_size=2,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )
    with pytest.raises(ValueError, match="world_size"):
        compose_ddp_accumulation_loss(
            [rank_zero],
            reduced_sum,
            world_size=8,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )


def test_reduction_authority_binds_local_window_and_ddp_runtime() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
    )

    rank_zero = _manual_statistics(
        numerator=torch.tensor(1.0, requires_grad=True),
        denominator=1,
        rank=0,
    )
    rank_one = _manual_statistics(
        numerator=torch.tensor(2.0, requires_grad=True),
        denominator=2,
        rank=1,
    )
    reduced = (
        pack_detached_loss_statistics([rank_zero], world_size=2)
        + pack_detached_loss_statistics([rank_one], world_size=2)
    )
    authority = _sum_authority(
        {0: [rank_zero], 1: [rank_one]},
        reduced,
        world_size=2,
    )

    with pytest.raises(ValueError, match="external expected digest"):
        compose_ddp_accumulation_loss(
            [rank_zero],
            reduced,
            world_size=2,
            reduction_authority=authority,
            expected_reduction_authority_sha256=_digest("forged-authority"),
        )

    same_packet_different_batch = replace(
        rank_zero,
        batch_authority_sha256=_digest("different-batch-authority"),
    )
    with pytest.raises(ValueError, match="Local accumulation window"):
        compose_ddp_accumulation_loss(
            [same_packet_different_batch],
            reduced,
            world_size=2,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )

    with pytest.raises(ValueError, match="Custom DDP communication hooks"):
        replace(
            authority,
            custom_communication_hook_sha256=_digest("comm-hook"),
        ).canonical_metadata()

    rank_one_wrong_state = replace(
        rank_one,
        model_state_sha256=_digest("different-model-state"),
    )
    with pytest.raises(ValueError, match="Every rank must share"):
        _sum_authority(
            {0: [rank_zero], 1: [rank_one_wrong_state]},
            reduced,
            world_size=2,
        )


def test_unequal_rank_counts_produce_exact_ddp_averaged_gradient() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
    )

    parameter_zero = torch.tensor(2.0, requires_grad=True)
    parameter_one = torch.tensor(2.0, requires_grad=True)
    rank_zero = _manual_statistics(
        numerator=parameter_zero.square(),
        denominator=1,
        rank=0,
    )
    rank_one = _manual_statistics(
        numerator=3.0 * parameter_one.square(),
        denominator=3,
        rank=1,
    )
    reduced = (
        pack_detached_loss_statistics([rank_zero], world_size=2)
        + pack_detached_loss_statistics([rank_one], world_size=2)
    )
    authority = _sum_authority(
        {0: [rank_zero], 1: [rank_one]},
        reduced,
        world_size=2,
    )
    loss_zero = compose_ddp_accumulation_loss(
        [rank_zero],
        reduced,
        world_size=2,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    loss_one = compose_ddp_accumulation_loss(
        [rank_one],
        reduced,
        world_size=2,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    gradient_zero = torch.autograd.grad(loss_zero, parameter_zero)[0]
    gradient_one = torch.autograd.grad(loss_one, parameter_one)[0]
    simulated_ddp_average = 0.5 * (gradient_zero + gradient_one)

    reference_parameter = torch.tensor(2.0, requires_grad=True)
    reference_loss = (
        reference_parameter.square() + 3.0 * reference_parameter.square()
    ) / 4.0
    reference_gradient = torch.autograd.grad(
        reference_loss,
        reference_parameter,
    )[0]
    torch.testing.assert_close(simulated_ddp_average, reference_gradient)
    assert gradient_zero.item() == pytest.approx(2.0)
    assert gradient_one.item() == pytest.approx(6.0)


def test_accumulation_uses_complete_window_denominator() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
    )

    first_value = torch.tensor(1.0, requires_grad=True)
    second_value = torch.tensor(3.0, requires_grad=True)
    first = _manual_statistics(
        numerator=first_value,
        denominator=1,
        rank=0,
        accumulation_index=0,
        batch_token="window-a",
    )
    second = _manual_statistics(
        numerator=second_value,
        denominator=3,
        rank=0,
        accumulation_index=1,
        batch_token="window-b",
    )
    packet = pack_detached_loss_statistics(
        [first, second],
        world_size=1,
    )
    authority = _sum_authority(
        {0: [first, second]},
        packet,
        world_size=1,
    )
    loss = compose_ddp_accumulation_loss(
        [first, second],
        packet,
        world_size=1,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    assert loss.item() == pytest.approx(1.0)
    gradients = torch.autograd.grad(loss, (first_value, second_value))
    assert gradients[0].item() == pytest.approx(0.25)
    assert gradients[1].item() == pytest.approx(0.25)


def test_empty_global_support_fails_before_backward_or_scaler_step() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        LOSS_TERM_ORDER,
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
    )

    value = torch.tensor(0.0, requires_grad=True)
    statistics = _manual_statistics(
        numerator=value * 0.0,
        denominator=0,
        rank=0,
    )
    packet = pack_detached_loss_statistics([statistics], world_size=1)
    authority = _sum_authority({0: [statistics]}, packet, world_size=1)
    assert packet.shape == (3, len(LOSS_TERM_ORDER))
    with pytest.raises(ValueError, match="Global support is empty"):
        compose_ddp_accumulation_loss(
            [statistics],
            packet,
            world_size=1,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )
    assert value.grad is None


def test_empty_local_rank_window_is_explicitly_unsupported() -> None:
    from interaction_first.pairformer_training_step import (
        pack_detached_loss_statistics,
    )

    with pytest.raises(ValueError, match="must contain a microbatch"):
        pack_detached_loss_statistics([], world_size=2)


def test_zero_support_cannot_carry_a_hidden_numerator() -> None:
    import torch

    statistics = _manual_statistics(
        numerator=torch.tensor(1.0, requires_grad=True),
        denominator=1,
        rank=0,
    )
    forged = replace(
        statistics,
        term_denominators=(0,) + statistics.term_denominators[1:],
    )
    with pytest.raises(ValueError, match="zero-support"):
        forged.validate()


def test_replayed_or_noncontiguous_microbatches_are_rejected() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        pack_detached_loss_statistics,
    )

    first = _manual_statistics(
        numerator=torch.tensor(1.0, requires_grad=True),
        denominator=1,
        rank=0,
        accumulation_index=0,
    )
    with pytest.raises(ValueError, match="replayed"):
        pack_detached_loss_statistics([first, first], world_size=1)
    noncontiguous = _manual_statistics(
        numerator=torch.tensor(1.0, requires_grad=True),
        denominator=1,
        rank=0,
        accumulation_index=2,
        batch_token="noncontiguous",
    )
    with pytest.raises(ValueError, match="contiguous"):
        pack_detached_loss_statistics(
            [first, noncontiguous],
            world_size=1,
        )


def test_seed_receipts_are_deterministic_and_batch_bound() -> None:
    from interaction_first.pairformer_training_step import derive_seed_receipt

    kwargs = {
        "base_seed": 9,
        "optimizer_step": 3,
        "accumulation_index": 1,
        "rank": 4,
        "batch_contract_sha256": _digest("batch"),
        "run_manifest_sha256": _digest("run"),
        "checkpoint_parent_sha256": _digest("checkpoint"),
    }
    first = derive_seed_receipt(**kwargs)
    second = derive_seed_receipt(**kwargs)
    assert first == second
    assert first.sha256 == second.sha256
    assert first.derived_seed < 2**63
    changed = derive_seed_receipt(
        **{**kwargs, "batch_contract_sha256": _digest("other-batch")}
    )
    assert changed.derived_seed != first.derived_seed
    assert changed.sha256 != first.sha256
    with pytest.raises(ValueError, match="derived_seed"):
        replace(first, derived_seed=first.derived_seed + 1).canonical_metadata()
    with pytest.raises(ValueError, match="interoperable JSON integer range"):
        derive_seed_receipt(
            **{**kwargs, "base_seed": 2**53}
        )


def test_autocast_policy_is_explicit_and_device_matched() -> None:
    from interaction_first.pairformer_training_step import AutocastPolicy

    AutocastPolicy(enabled=False, device_type="cpu").validate("cpu")
    AutocastPolicy(enabled=True, device_type="cpu", dtype_name="bfloat16").validate(
        "cpu"
    )
    with pytest.raises(ValueError, match="must match"):
        AutocastPolicy(enabled=False, device_type="cuda").validate("cpu")
    AutocastPolicy(enabled=True, device_type="cpu", dtype_name="float16").validate(
        "cpu"
    )


def test_optimizer_boundary_receipt_binds_sources_seeds_and_global_counts() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    model, _, result = _forward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    receipt = perform_optimizer_step(
        loss,
        model,
        optimizer,
        statistics,
        packet,
        world_size=1,
        optimizer_step=3,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
        max_grad_norm=10.0,
    )
    assert receipt.status == "applied"
    assert receipt.optimizer_step_before == 3
    assert receipt.optimizer_step_after == 4
    assert receipt.accumulation_steps == 1
    assert receipt.world_size == 1
    assert receipt.schema_sha256 == _schema().sha256
    assert receipt.batch_contract_sha256 == (
        result.statistics.batch_contract_sha256,
    )
    assert receipt.seed_receipt_sha256 == (
        result.statistics.seed_receipt.sha256,
    )
    assert receipt.global_term_denominators[:3] == (2, 2, 2)
    assert receipt.preclip_gradient_norm is not None
    assert receipt.reduction_operation == "sum"
    assert receipt.participating_ranks == (0,)
    assert receipt.reduced_packet_sha256
    assert receipt.model_parameter_manifest_sha256
    assert receipt.model_state_sha256_before
    assert receipt.model_state_sha256_after
    assert receipt.optimizer_state_sha256_before
    assert receipt.optimizer_state_sha256_after
    assert receipt.rng_state_sha256_before
    assert receipt.rng_state_sha256_after
    assert receipt.checkpoint_parent_sha256 == _digest("checkpoint-parent")
    assert receipt.reduction_authority_sha256 == authority.sha256
    assert receipt.gradients_finite
    assert len(receipt.sha256) == 64
    with pytest.raises(ValueError, match="disagrees with receipt statistics"):
        replace(
            receipt,
            global_equal_complex_objective=(
                receipt.global_equal_complex_objective + 1.0
            ),
        ).canonical_metadata()


def test_optimizer_uses_the_locked_graph_not_an_equal_valued_substitute() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    model, _, result = _forward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    expected, authority = _authorized_loss(statistics, packet)
    first_parameter = next(model.parameters())
    substituted = expected.detach() + first_parameter.sum() * 0.0
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    perform_optimizer_step(
        substituted,
        model,
        optimizer,
        statistics,
        packet,
        world_size=1,
        optimizer_step=3,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    after = tuple(parameter.detach() for parameter in model.parameters())
    assert any(
        not torch.equal(before_value, after_value)
        for before_value, after_value in zip(before, after)
    )


def test_optimizer_must_cover_the_exact_bound_model_parameter_set() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    model, _, result = _forward()
    optimizer = torch.optim.SGD([next(model.parameters())], lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    with pytest.raises(ValueError, match="exactly match"):
        perform_optimizer_step(
            loss,
            model,
            optimizer,
            statistics,
            packet,
            world_size=1,
            optimizer_step=3,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )


def test_optimizer_rejects_model_state_change_after_forward() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    model, _, result = _forward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    with torch.no_grad():
        next(model.parameters()).add_(0.25)
    with pytest.raises(ValueError, match="Model state changed"):
        perform_optimizer_step(
            loss,
            model,
            optimizer,
            statistics,
            packet,
            world_size=1,
            optimizer_step=3,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_multi_rank_optimizer_requires_a_live_ddp_wrapper() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        derive_seed_receipt,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    model, _, result = _forward()
    rank_zero = result.statistics
    rank_one_batch = _digest("rank-one-batch")
    rank_one_seed = derive_seed_receipt(
        base_seed=rank_zero.seed_receipt.base_seed,
        optimizer_step=rank_zero.seed_receipt.optimizer_step,
        accumulation_index=0,
        rank=1,
        batch_contract_sha256=rank_one_batch,
        run_manifest_sha256=rank_zero.seed_receipt.run_manifest_sha256,
        checkpoint_parent_sha256=(
            rank_zero.seed_receipt.checkpoint_parent_sha256
        ),
    )
    rank_one = replace(
        rank_zero,
        batch_contract_sha256=rank_one_batch,
        batch_authority_sha256=_digest("rank-one-authority"),
        seed_receipt=rank_one_seed,
    )
    reduced = (
        pack_detached_loss_statistics([rank_zero], world_size=2)
        + pack_detached_loss_statistics([rank_one], world_size=2)
    )
    authority = _sum_authority(
        {0: [rank_zero], 1: [rank_one]},
        reduced,
        world_size=2,
    )
    loss = compose_ddp_accumulation_loss(
        [rank_zero],
        reduced,
        world_size=2,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="live DistributedDataParallel"):
        perform_optimizer_step(
            loss,
            model,
            optimizer,
            [rank_zero],
            reduced,
            world_size=2,
            optimizer_step=3,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )


def test_checkpoint_receipt_binds_persisted_bytes_and_post_step_state() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        build_checkpoint_continuation_receipt,
        pack_detached_loss_statistics,
        perform_optimizer_step,
        verify_checkpoint_continuation_receipt,
    )

    model, _, result = _forward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    optimizer_receipt = perform_optimizer_step(
        loss,
        model,
        optimizer,
        statistics,
        packet,
        world_size=1,
        optimizer_step=3,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
    )
    checkpoint = build_checkpoint_continuation_receipt(
        optimizer_receipt,
        checkpoint_artifact_sha256=_digest("checkpoint-bytes"),
        checkpoint_writer_source_sha256=_digest("checkpoint-writer"),
        immutable_storage_receipt_sha256=_digest("immutable-storage"),
    )
    verify_checkpoint_continuation_receipt(
        checkpoint,
        expected_receipt_sha256=checkpoint.sha256,
        expected_checkpoint_artifact_sha256=_digest("checkpoint-bytes"),
    )
    assert checkpoint.model_state_sha256 == (
        optimizer_receipt.model_state_sha256_after
    )
    assert checkpoint.optimizer_state_sha256 == (
        optimizer_receipt.optimizer_state_sha256_after
    )
    assert checkpoint.rng_state_sha256 == optimizer_receipt.rng_state_sha256_after
    with pytest.raises(ValueError, match="Checkpoint artifact"):
        verify_checkpoint_continuation_receipt(
            checkpoint,
            expected_receipt_sha256=checkpoint.sha256,
            expected_checkpoint_artifact_sha256=_digest("other-checkpoint"),
        )


def test_optimizer_rejects_a_rank_local_mean_substituted_for_ddp_formula() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    model, _, result = _forward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    _, authority = _authorized_loss(statistics, packet)
    wrong_loss = result.local_objective * 0.5
    with pytest.raises(ValueError, match="locked DDP accumulation formula"):
        perform_optimizer_step(
            wrong_loss,
            model,
            optimizer,
            statistics,
            packet,
            world_size=1,
            optimizer_step=3,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
        )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_grad_scaler_step_and_update_occur_once_at_valid_window_boundary() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        AutocastPolicy,
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    class RecordingScaler:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.scale_value = 16.0

        def get_scale(self):
            return self.scale_value

        def scale(self, loss):
            self.calls.append("scale")
            return loss

        def unscale_(self, optimizer):
            self.calls.append("unscale")

        def step(self, optimizer):
            self.calls.append("step")
            optimizer.step()

        def update(self):
            self.calls.append("update")

        def state_dict(self):
            return {"scale": self.scale_value}

    model, _, result = _forward(
        autocast_policy=AutocastPolicy(
            enabled=True,
            device_type="cpu",
            dtype_name="float16",
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    scaler = RecordingScaler()
    receipt = perform_optimizer_step(
        loss,
        model,
        optimizer,
        statistics,
        packet,
        world_size=1,
        optimizer_step=3,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
        grad_scaler=scaler,
        max_grad_norm=10.0,
    )
    assert scaler.calls == ["scale", "unscale", "step", "update"]
    assert receipt.status == "applied"
    assert receipt.grad_scaler_used
    assert receipt.scaler_scale_before == pytest.approx(16.0)
    assert receipt.scaler_scale_after == pytest.approx(16.0)


def test_grad_scaler_nonfinite_boundary_is_recorded_as_skipped() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        AutocastPolicy,
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    class SkippingScaler:
        def __init__(self) -> None:
            self.scale_value = 16.0
            self.skipped = False

        def get_scale(self):
            return self.scale_value

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            return None

        def step(self, optimizer):
            finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all().item())
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            self.skipped = not finite
            if finite:
                optimizer.step()

        def update(self):
            if self.skipped:
                self.scale_value /= 2.0

        def state_dict(self):
            return {
                "scale": self.scale_value,
                "skipped": self.skipped,
            }

    model, _, result = _forward(
        autocast_policy=AutocastPolicy(
            enabled=True,
            device_type="cpu",
            dtype_name="float16",
        )
    )
    parameter = next(model.parameters())
    hook = parameter.register_hook(lambda gradient: gradient * float("inf"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    scaler = SkippingScaler()
    receipt = perform_optimizer_step(
        loss,
        model,
        optimizer,
        statistics,
        packet,
        world_size=1,
        optimizer_step=3,
        reduction_authority=authority,
        expected_reduction_authority_sha256=authority.sha256,
        grad_scaler=scaler,
        max_grad_norm=10.0,
    )
    hook.remove()
    assert receipt.status == "skipped_by_grad_scaler"
    assert receipt.optimizer_step_after == 3
    assert not receipt.gradients_finite
    assert receipt.preclip_gradient_norm is None
    assert receipt.scaler_scale_before == pytest.approx(16.0)
    assert receipt.scaler_scale_after == pytest.approx(8.0)


def test_grad_scaler_is_rejected_without_enabled_float16_autocast() -> None:
    import torch

    from interaction_first.pairformer_training_step import (
        compose_ddp_accumulation_loss,
        pack_detached_loss_statistics,
        perform_optimizer_step,
    )

    class UnusedScaler:
        def get_scale(self):
            return 1.0

        def scale(self, loss):
            raise AssertionError("scale must not be reached")

        def unscale_(self, optimizer):
            raise AssertionError("unscale_ must not be reached")

        def step(self, optimizer):
            raise AssertionError("step must not be reached")

        def update(self):
            raise AssertionError("update must not be reached")

    model, _, result = _forward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    statistics = [result.statistics]
    packet = pack_detached_loss_statistics(statistics, world_size=1)
    loss, authority = _authorized_loss(statistics, packet)
    with pytest.raises(ValueError, match="enabled float16"):
        perform_optimizer_step(
            loss,
            model,
            optimizer,
            statistics,
            packet,
            world_size=1,
            optimizer_step=3,
            reduction_authority=authority,
            expected_reduction_authority_sha256=authority.sha256,
            grad_scaler=UnusedScaler(),
        )
