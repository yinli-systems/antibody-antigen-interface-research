#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-prepare-r1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-recovery-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools_root=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly data_root=${root}/data/framecontact_20260730
readonly cache_root=${campaign}/data/structure_cache_v1
readonly original_sdh_root=${campaign}/structure_prediction/sdh_af3
readonly canonical_sdh_root=${campaign}/structure_prediction/sdh_af3_canonical_recovery_v1
readonly recovery_manifest=${canonical_sdh_root}/CANONICAL_RECOVERY_VIEW_MANIFEST.json
readonly recovery_receipt=${canonical_sdh_root}/CANONICAL_RECOVERY_VIEW_RECEIPT.json
readonly recovery_manifest_sha256=65268479b15c586d1da9e2eabcea283758af94972bc3763fef8304439232ab4e
readonly recovery_receipt_sha256=bbad4a07c4e934a620cda5bc57b34ac76d52ffd6afc3d8a209ca35764644c2c7
readonly baseline_campaign=${root}/results/development/abpairfm_joint_energy_balanced_full_v1
readonly expert_campaign=${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${cache_root}"
test ! -e "${campaign}/authority"
printf '%s  %s\n' \
  "${recovery_manifest_sha256}" "${recovery_manifest}" \
  "${recovery_receipt_sha256}" "${recovery_receipt}" \
  e8c839fcff99c0638d31b6b4be2c4c398d9f1ec30ee0610d2f6c8754200d0a70 "${tools_root}/build_sdh_canonical_recovery_view.py" \
  2ce582226e08626873a5434411c21def321120822bb1dceb0f1e76165d16c3c0 "${tools_root}/freeze_predicted_geometry_protocols_recovery_v1.py" \
  | sha256sum --check --strict

"${python_bin}" - "${recovery_manifest}" "${recovery_receipt}" <<'PY'
import hashlib
import json
import sys
manifest_path, receipt_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
receipt = json.load(open(receipt_path, encoding="utf-8"))
assert manifest["status"] == receipt["status"] == "PASS_CANONICAL_SDH_RECOVERY_VIEW"
assert manifest["record_count"] == receipt["record_count"] == 332
assert manifest["exact_canonical_name_count"] == 331
assert manifest["timestamp_suffixed_recovery_count"] == 1
assert receipt["source_predictions_modified"] is False
assert manifest["test_access"] is receipt["test_access"] is False
assert manifest["temporal_holdout_access"] is receipt["temporal_holdout_access"] is False
payload = open(manifest_path, "rb").read()
assert hashlib.sha256(payload).hexdigest() == receipt["manifest_sha256"]
PY

for batch_index in $(seq 0 16); do
  batch_dir=$(printf '%s/batches/batch-%03d' "${original_sdh_root}" "${batch_index}")
  test -f "${batch_dir}/RESULT_SHA256SUMS.txt"
  (cd "${batch_dir}" && sha256sum --check --strict RESULT_SHA256SUMS.txt)
done

"${python_bin}" "${tools_root}/materialize_predicted_structure_cache.py" \
  --instance-manifest-csv /ssd/scxi253/yangwuyu/p_e_first_research/datasets/sabdab2/manifest.csv \
  --source-split-csv "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --paired-antibody-root /data/run01/scxi253/af3test/workdir/sabdab2_antibody_scxi5090_v2_5seed_20260716/output \
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
  --materializer "${tools_root}/materialize_predicted_structure_cache.py" \
  --training-launcher "${tools_root}/slurm_predicted_geometry_development_v1.sbatch" \
  --analyzer "${tools_root}/analyze_predicted_geometry_development_v1.py" \
  --recovery-manifest "${recovery_manifest}" \
  --recovery-manifest-sha256 "${recovery_manifest_sha256}" \
  --recovery-receipt "${recovery_receipt}" \
  --recovery-receipt-sha256 "${recovery_receipt_sha256}"

cd "${campaign}/authority"
sha256sum --check --strict AUTHORITY_SHA256SUMS.txt
