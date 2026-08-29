#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-prepare-r2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-recovery-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-recovery-v2-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools_root=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly data_root=${root}/data/framecontact_20260730
readonly cache_root=${campaign}/data/structure_cache_v1
readonly original_sdh_root=${campaign}/structure_prediction/sdh_af3
readonly canonical_sdh_root=${campaign}/structure_prediction/sdh_af3_canonical_recovery_v1
readonly paired_root=${campaign}/structure_prediction/paired_af3_case_exact_recovery_v2
readonly paired_manifest=${paired_root}/PAIRED_CASE_EXACT_VIEW_MANIFEST.json
readonly paired_receipt=${paired_root}/PAIRED_CASE_EXACT_VIEW_RECEIPT.json
readonly combined_manifest=${paired_root}/COMBINED_RECOVERY_V2_MANIFEST.json
readonly combined_receipt=${paired_root}/COMBINED_RECOVERY_V2_RECEIPT.json
readonly paired_manifest_sha256=1fbae16037aa27f5e089e0edd2353de74ab9499a779ff537c7dbc116191aa9fd
readonly paired_receipt_sha256=4ae683274a5477854ee2ab186fce65a3b9fc5a9da2a5227e22f9a33c61792b4c
readonly combined_manifest_sha256=41dd4cec92a52bddafedb99745283a35c28b218bb752e0c566bc5b363b2c0681
readonly combined_receipt_sha256=a4f8164913b1a78f36948af2234263cd5bb60485d4d810a9734bcf34a078b305
readonly failed_receipt=${campaign}/FAIL_CLOSED_PREPARATION_1483067.json
readonly baseline_campaign=${root}/results/development/abpairfm_joint_energy_balanced_full_v1
readonly expert_campaign=${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${cache_root}"
test ! -e "${campaign}/authority"
printf '%s  %s\n' \
  "${paired_manifest_sha256}" "${paired_manifest}" \
  "${paired_receipt_sha256}" "${paired_receipt}" \
  "${combined_manifest_sha256}" "${combined_manifest}" \
  "${combined_receipt_sha256}" "${combined_receipt}" \
  fa1dfd7882b40bcf12f25236b981cda79a262a6863f34194ea1340bea686558b "${failed_receipt}" \
  a163cbf66d175add6f52fef00f62c7514fcf1b58812deb0ae9b8e563ab46926a "${tools_root}/build_paired_case_exact_recovery_view_v2.py" \
  f7f9f048fe5a5fc70a6c6f7dd2bda3e0eb616b0920d594860f7a7b86d95721ed "${tools_root}/materialize_predicted_structure_cache_paired_recovery_v2.py" \
  4326a5af92f1ff17102db0f10570d80f9d258bfc9fd1d88425554bd183b0b3fe "${tools_root}/materialize_predicted_structure_cache.py" \
  2ce582226e08626873a5434411c21def321120822bb1dceb0f1e76165d16c3c0 "${tools_root}/freeze_predicted_geometry_protocols_recovery_v1.py" \
  4540780a34d0cd8e32481aa6954b440bdcd05f6fe37e03b82be471b9e5eb6885 "${tools_root}/slurm_predicted_geometry_development_v1.sbatch" \
  e72e6017037fd0d3f83519867b042b5da3431ba1987c9b6a111955725ddc8ec7 "${tools_root}/analyze_predicted_geometry_development_v1.py" \
  | sha256sum --check --strict

"${python_bin}" - \
  "${paired_manifest}" "${paired_receipt}" \
  "${combined_manifest}" "${combined_receipt}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

paired_manifest_path, paired_receipt_path, combined_path, combined_receipt_path = (
    map(Path, sys.argv[1:])
)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
paired = json.loads(paired_manifest_path.read_text(encoding="utf-8"))
receipt = json.loads(paired_receipt_path.read_text(encoding="utf-8"))
combined = json.loads(combined_path.read_text(encoding="utf-8"))
combined_receipt = json.loads(combined_receipt_path.read_text(encoding="utf-8"))
assert paired["status"] == receipt["status"] == "PASS_PAIRED_CASE_EXACT_RECOVERY_VIEW"
assert paired["development_count"] == 3114
assert paired["role_counts"] == {"train": 2565, "validation": 549}
assert paired["paired_instance_count"] == receipt["paired_instance_count"] == 2224
assert paired["sdh_instance_count"] == 890
assert paired["case_collision_group_count"] == receipt["case_collision_group_count"] == 7
assert paired["canonical_source_count"] == 2217
assert paired["timestamped_case_collision_recovery_count"] == 7
assert receipt["manifest_sha256"] == sha(paired_manifest_path)
assert receipt["all_target_directories_unique"] is True
assert receipt["all_source_directories_unique"] is True
assert combined["paired_case_exact_manifest_sha256"] == sha(paired_manifest_path)
assert combined["paired_case_exact_receipt_sha256"] == sha(paired_receipt_path)
assert combined["user_authorized_recovery_after_failure"] is True
assert combined_receipt["manifest_sha256"] == sha(combined_path)
assert all(
    value is False
    for value in (
        paired["source_predictions_modified"],
        paired["task_labels_loaded"],
        paired["test_access"],
        paired["temporal_holdout_access"],
        combined["experimental_bound_complex_coordinates_read"],
        combined["cross_partner_geometry_used"],
        combined["task_labels_loaded"],
        combined["test_access"],
        combined["temporal_holdout_access"],
    )
)
records = paired["records"]
assert len(records) == 2224
assert len({record["instance_id"] for record in records}) == 2224
for record in records:
    directory = Path(record["canonical_directory"])
    assert directory.is_dir() and directory.name == record["target_name"]
    models = sorted(directory.glob("*_model.cif"))
    assert len(models) == 1 and sha(models[0]) == record["model_sha256"]
    ranking = directory / "ranking_scores.csv"
    assert ranking.is_file() and sha(ranking) == record["ranking_sha256"]
    selected = directory / (
        f"seed-{record['selected_seed']}_sample-{record['selected_sample']}"
    )
    selected_models = sorted(selected.glob("*model.cif"))
    assert len(selected_models) == 1
    assert sha(selected_models[0]) == record["selected_model_sha256"]
PY

for batch_index in $(seq 0 16); do
  batch_dir=$(printf '%s/batches/batch-%03d' "${original_sdh_root}" "${batch_index}")
  test -f "${batch_dir}/RESULT_SHA256SUMS.txt"
  (cd "${batch_dir}" && sha256sum --check --strict RESULT_SHA256SUMS.txt)
done

"${python_bin}" "${tools_root}/materialize_predicted_structure_cache_paired_recovery_v2.py" \
  --instance-manifest-csv /ssd/scxi253/yangwuyu/p_e_first_research/datasets/sabdab2/manifest.csv \
  --source-split-csv "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --paired-antibody-root "${paired_root}" \
  --sdh-plan "${original_sdh_root}/AF3_SDH_INPUT_PLAN.json" \
  --sdh-result-root "${canonical_sdh_root}" \
  --antigen-root /data/run01/scxi253/af3test/workdir/sabdab2_antigen_scxi5090_v2_5seed_20260716/output \
  --af3-runner-script /data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh \
  --output-root "${cache_root}"

"${python_bin}" "${tools_root}/freeze_predicted_geometry_protocols_recovery_v1.py" \
  --campaign-root "${campaign}" \
  --structure-cache-root "${cache_root}" \
  --baseline-template "${baseline_campaign}/authority/PROTOCOL_BASELINE_FULL_V2.json" \
  --expert-template "${expert_campaign}/authority/PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1.json" \
  --baseline-source-manifest "${baseline_campaign}/authority/SOURCE_MANIFEST_ENERGY_BALANCED_V3.json" \
  --baseline-source-manifest-sha256 ca31451b4be85ee1e07045609a41ed3c90789971368191383c28cbf4251b581e \
  --expert-source-manifest "${root}/immutable_stage_a1_candidates/edge_topk2_training_source_v1/SOURCE_MANIFEST_EDGE_TOPK2_V1.json" \
  --expert-source-manifest-sha256 f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55 \
  --materializer "${tools_root}/materialize_predicted_structure_cache_paired_recovery_v2.py" \
  --training-launcher "${tools_root}/slurm_predicted_geometry_development_v1.sbatch" \
  --analyzer "${tools_root}/analyze_predicted_geometry_development_v1.py" \
  --recovery-manifest "${combined_manifest}" \
  --recovery-manifest-sha256 "${combined_manifest_sha256}" \
  --recovery-receipt "${combined_receipt}" \
  --recovery-receipt-sha256 "${combined_receipt_sha256}"

cd "${campaign}/authority"
sha256sum --check --strict AUTHORITY_SHA256SUMS.txt
