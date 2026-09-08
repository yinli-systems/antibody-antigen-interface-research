#!/usr/bin/env python3
"""Verify in code that the R2 model consumes no bound inter-chain geometry.

``protocols/sabdab2_partner_geometry_r2_model_contract_v1.json`` declares
``bound_inter_chain_geometry_allowed: false``, and
``interaction_first.r2_model_contract`` checks that the manifest says so. It
checks the manifest only. This checks the model, by the behaviour the clause
exists to guarantee: moving one chain rigidly must leave every output unchanged.

    python3 scripts/verify_partner_pose_contract.py
    python3 scripts/verify_partner_pose_contract.py --json

Exit status is ``0`` when both chains' poses are proven irrelevant and ``1``
otherwise, including when the probe is inconclusive because the model ignored
the backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.partner_pose_invariance import (  # noqa: E402
    InvarianceVerdict,
    PartnerPoseInvarianceError,
    activate_zero_initialised_parameters,
    probe_partner_pose_invariance,
)

FORMAT_VERSION = "r2_partner_pose_contract_check_v1"
CONTRACT = "protocols/sabdab2_partner_geometry_r2_model_contract_v1.json"


def _schema():
    from interaction_first.pair_geometry_schema import PairGeometrySchema

    return PairGeometrySchema(
        distance_bin_edges_angstrom=(4.5, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0),
        contact_cutoff_angstrom=4.5,
        orientation_axis_names=(
            "antibody_direction_azimuth",
            "antibody_direction_polar",
            "antigen_direction_azimuth",
            "antigen_direction_polar",
            "relative_twist",
        ),
        orientation_axis_is_cyclic=(True, False, True, False, True),
        orientation_bin_edges_radians=tuple(
            (-2.0, -1.0, 0.0, 1.0, 2.0) if cyclic else (0.4, 0.8, 1.2, 1.6, 2.0)
            for cyclic in (True, False, True, False, True)
        ),
        label_format_version="pair_geometry_labels_v2",
        orientation_schema_version="pair_geometry_orientation_v1",
        primary_contact_definition="positive_or_complete_contact",
        distance_definition="complete_pair_minimum_heavy_atom_distance",
        orientation_definition=(
            "right_handed_N_CA_C_backbone_frames_with_CA_to_CA_directions"
        ),
        orientation_bin_edge_convention=(
            "internal_right_closed_thresholds_num_bins_minus_one"
        ),
    )


def _model_and_inputs():
    import torch

    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    torch.manual_seed(20260908)
    model = ProspectivePairGeometryModel(
        antibody_input_dim=7,
        antigen_input_dim=11,
        d_model=16,
        pair_dim=12,
        num_heads=4,
        num_fusion_layers=2,
        ffn_dim=32,
        geometry_schema=_schema(),
        num_distance_bins=8,
        contact_distance_bin_count=1,
        num_orientation_axes=5,
        num_orientation_bins=6,
        num_pair_update_layers=1,
        pair_chunk_size=4,
        dropout=0.0,
        centering_iterations=32,
        intra_chain_geometry_layers=1,
    )
    # float64: a 120 A translation costs about 1e-6 in float32, which cannot be
    # told apart from a small real dependence.
    model = model.eval().to(torch.float64)
    activated = activate_zero_initialised_parameters(model)

    torch.manual_seed(4)
    batch, antibody_length, antigen_length = 2, 5, 6
    antibody_mask = torch.ones(batch, antibody_length, dtype=torch.bool)
    antigen_mask = torch.ones(batch, antigen_length, dtype=torch.bool)
    inputs = {
        "antibody_features": torch.randn(batch, antibody_length, 7, dtype=torch.float64),
        "antigen_features": torch.randn(batch, antigen_length, 11, dtype=torch.float64),
        "antibody_mask": antibody_mask,
        "antigen_mask": antigen_mask,
        "candidate_pair_mask": antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1),
        "antibody_backbone": torch.randn(batch, antibody_length, 3, 3, dtype=torch.float64),
        "antigen_backbone": torch.randn(batch, antigen_length, 3, 3, dtype=torch.float64),
    }
    return model, inputs, activated


def build_report() -> dict:
    """Probe both chains and assemble the receipt."""
    import torch

    model, inputs, activated = _model_and_inputs()
    probes = {}
    for moved in ("antigen_backbone", "antibody_backbone"):

        def run(replacement, moved=moved):
            with torch.no_grad():
                return model(**{**inputs, moved: replacement})

        probes[moved] = probe_partner_pose_invariance(run, inputs[moved]).as_json()

    passed = all(entry["verdict"] == InvarianceVerdict.INVARIANT.value for entry in probes.values())
    return {
        "format_version": FORMAT_VERSION,
        "contract": CONTRACT,
        "clause": "bound_inter_chain_geometry_allowed = false",
        "model": "interaction_first.pair_geometry.ProspectivePairGeometryModel",
        "precision": "float64",
        "zero_initialised_parameters_activated": activated,
        "probes": probes,
        "passed": passed,
        "claim_boundary": (
            "Structural check of one model class on synthetic inputs. It proves that "
            "no output depends on the rigid pose of either chain relative to the other, "
            "which is the property the contract clause protects. It is not a training, "
            "selection, evaluation, or performance result."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the machine-readable receipt")
    arguments = parser.parse_args(argv)

    try:
        report = build_report()
    except PartnerPoseInvarianceError as error:
        print(f"partner-pose contract check could not run: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1

    print(f"contract : {report['contract']}")
    print(f"clause   : {report['clause']}")
    print(f"model    : {report['model']} ({report['precision']})")
    print(f"activated: {report['zero_initialised_parameters_activated']} zero-initialised tensors")
    print()
    for moved, entry in report["probes"].items():
        print(f"  moving {moved}")
        print(f"    sensitivity (non-rigid) {entry['sensitivity']:.6e}")
        for name, value in sorted(entry["deviations"].items()):
            print(f"    {name:26s} {value:.6e}")
        print(f"    verdict                {entry['verdict']}")
    print()
    print("PASS: no output depends on the pose of either chain relative to the other."
          if report["passed"] else
          "FAIL: an output depends on the relative pose of the two chains.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
