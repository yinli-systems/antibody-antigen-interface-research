#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-prepare
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools_root=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly data_root=${root}/data/framecontact_20260730
readonly cache_root=${campaign}/data/structure_cache_v1
readonly sdh_root=${campaign}/structure_prediction/sdh_af3
readonly baseline_campaign=${root}/results/development/abpairfm_joint_energy_balanced_full_v1
readonly expert_campaign=${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${cache_root}"
test ! -e "${campaign}/authority"

for batch_index in $(seq 0 16); do
  batch_dir=$(printf '%s/batches/batch-%03d' "${sdh_root}" "${batch_index}")
  test -f "${batch_dir}/RESULT_SHA256SUMS.txt"
  input_count=$(find "${batch_dir}/inputs" -maxdepth 1 -type f -name '*.json' | wc -l)
  model_count=$(find "${batch_dir}/results" -mindepth 2 -maxdepth 2 -type f -name '*_model.cif' | wc -l)
  test "${input_count}" -gt 0
  test "${input_count}" -eq "${model_count}"
  (cd "${batch_dir}" && sha256sum --check --strict RESULT_SHA256SUMS.txt)
done

"${python_bin}" "${tools_root}/materialize_predicted_structure_cache.py" \
  --instance-manifest-csv /ssd/scxi253/yangwuyu/p_e_first_research/datasets/sabdab2/manifest.csv \
  --source-split-csv "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --paired-antibody-root /data/run01/scxi253/af3test/workdir/sabdab2_antibody_scxi5090_v2_5seed_20260716/output \
  --sdh-plan "${sdh_root}/AF3_SDH_INPUT_PLAN.json" \
  --sdh-result-root "${sdh_root}" \
  --antigen-root /data/run01/scxi253/af3test/workdir/sabdab2_antigen_scxi5090_v2_5seed_20260716/output \
  --af3-runner-script /data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh \
  --output-root "${cache_root}"

"${python_bin}" "${tools_root}/freeze_predicted_geometry_protocols.py" \
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
  --analyzer "${tools_root}/analyze_predicted_geometry_development_v1.py"

cd "${campaign}/authority"
sha256sum --check --strict AUTHORITY_SHA256SUMS.txt
