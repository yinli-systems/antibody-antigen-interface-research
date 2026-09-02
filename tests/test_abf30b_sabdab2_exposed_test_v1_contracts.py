from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "remote_tools" / "abf30b_sabdab2_exposed_test_v1"


def read(name: str) -> str:
    return (TOOLS / name).read_text(encoding="utf-8")


def test_python_and_sbatch_files_parse() -> None:
    for path in TOOLS.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert list(TOOLS.glob("*.sbatch"))


def test_preparation_boundary_is_honest_and_fixed() -> None:
    source = read("freeze_label_free_test_plan_v1.py")
    assert "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ" in source
    assert '"test_sequence_metadata_already_audited": True' in source
    assert '"task_labels_read_during_freeze": False' in source
    assert '"fresh_blind_test": False' in source
    assert '"model_unseen": True' in source
    assert '"analyst_exposed": True' in source
    assert '"selection_or_tuning_from_this_measurement": False' in source
    assert '"cumulative_test_access_ordinal": 3' in source
    assert '"sable_repeated": False' in source


def test_fixed_three_seed_task_router() -> None:
    freezer = read("freeze_label_free_test_plan_v1.py")
    analyzer = read("analyze_exposed_test_v1.py")
    assert '"seeds": [17, 29, 43]' in freezer
    assert '"contact": "baseline same seed"' in freezer
    assert '"epitope": "edge_topk2_supervised same seed"' in freezer
    assert '"paratope": "baseline same seed"' in freezer
    assert 'TASK_SOURCE = {"contact": "baseline", "epitope": "edge_topk2_supervised", "paratope": "baseline"}' in analyzer


def test_gpu_and_array_contracts() -> None:
    sbatches = {path.name: path.read_text(encoding="utf-8") for path in TOOLS.glob("*.sbatch")}
    for source in sbatches.values():
        assert "#SBATCH --partition=gpu_4090" in source
        assert "#SBATCH --partition=hp_4090" not in source
        assert "#SBATCH --partition=hp_5090" not in source
    assert "#SBATCH --array=0-2%3" in sbatches["run_test_sdh_af3_array_v1.sbatch"]
    assert "#SBATCH --array=0-5%6" in sbatches["evaluate_exposed_test_v1.sbatch"]
    assert "#SBATCH --array=0-5%6" in sbatches["evaluate_exposed_test_contact_authority_recovery_v1.sbatch"]


def test_no_test_selection_or_model_update() -> None:
    freezer = read("freeze_test_measurement_v1.py")
    evaluator = read("evaluate_exposed_test_component_v1.py")
    analyzer = read("analyze_exposed_test_v1.py")
    assert '"calibration_refit_on_test": False' in freezer
    assert '"model_update_on_test": False' in freezer
    assert '"test_label_artifacts_read_during_freeze": False' in freezer
    assert 'frozen_calibration=component["frozen_calibration"]' in evaluator
    assert 'if state_after != state_before:' in evaluator
    assert '"model_or_checkpoint_selected_on_test": False' in analyzer
    assert '"temporal_holdout_loaded": False' in analyzer


def test_pinned_reused_sources_match_local_sha256_constants() -> None:
    import hashlib

    cases = (
        (
            "materialize_test_structure_cache_v1.py",
            ROOT / "remote_tools" / "sabdab2_predicted_geometry_v1" / "materialize_predicted_structure_cache.py",
        ),
        (
            "build_test_chain_map_v1.py",
            ROOT / "remote_tools" / "abf30b_framecontact_v1" / "build_abf30b_chain_map_v1.py",
        ),
        (
            "materialize_abf30b_test_embeddings_v1.py",
            ROOT / "remote_tools" / "abf30b_framecontact_v1" / "materialize_abf30b_sabdab2_embeddings_v1.py",
        ),
    )
    for wrapper_name, source_path in cases:
        tree = ast.parse(read(wrapper_name))
        expected = next(
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SOURCE_SHA256" for target in node.targets)
            and isinstance(node.value, ast.Constant)
        )
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == expected


def test_same_opening_contact_authority_recovery_is_metadata_only() -> None:
    freezer = read("freeze_contact_authority_adapter_recovery_v1.py")
    materializer = read("materialize_test_contact_authority_adapter_v1.py")
    binder = read("freeze_contact_authority_evaluation_recovery_v1.py")
    evaluator = read("evaluate_exposed_test_contact_authority_recovery_v1.py")
    assert "FROZEN_AFTER_CONTACT_AUTHORITY_FAILURE_FOR_SAME_OPENING" in freezer
    assert '"failure_before_contact_artifact_read": True' in freezer
    assert '"scientific_protocol_changed": False' in freezer
    assert '"labels_regenerated": False' in freezer
    assert "os.link(source_artifact, target_artifact)" in materializer
    assert '"pair_label_artifacts_modified": False' in materializer
    assert '"pair_labels_regenerated": False' in materializer
    assert '"record_count": 549' in materializer
    assert "FROZEN_AFTER_TEST_CONTACT_AUTHORITY_ADAPTER_FOR_SAME_OPENING" in binder
    assert "adapter_contact_manifest_sha256" in evaluator
    assert "test_split_path = args.structure_root" in evaluator
