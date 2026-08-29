from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_framecontact_validation_interventions.py"
SLURM = Path(__file__).resolve().parents[1] / "scripts" / "slurm_evaluate_framecontact_validation_interventions_v1.sbatch"


def test_validation_intervention_audit_is_test_free_and_write_once() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    ast.parse(text)
    assert 'split_role="validation"' in text
    assert 'split_role="test"' not in text
    assert "Refusing to overwrite validation-intervention output" in text
    assert '"test_loaded": False' in text
    assert '"external_test_loaded": False' in text
    assert "selection_performed_by_this_audit" in text


def test_validation_intervention_audit_covers_required_geometry_controls() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "independent_monomer_rigid_transform",
        "geometry_shuffle",
        "local_frame_corruption",
        "degree_matched_graph_rewiring",
        "coordinate_noise",
    ):
        assert name in text
    assert "validation_calibration_frozen" in text
    assert "delta_vs_clean" in text


def test_validation_intervention_slurm_is_complete_and_test_free() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-14%15" in text
    assert "seeds=(17 42 73 101 137)" in text
    assert "validation-reciprocal-31.json" in text
    assert "--coordinate-noise 0.10 0.25 0.50" in text
    assert "--test" not in text
    assert "--external" not in text
    assert "sha256sum -c SOURCE_SHA256SUMS.txt" in text
