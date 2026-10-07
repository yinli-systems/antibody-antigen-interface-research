#!/usr/bin/env bash
#SBATCH --job-name=sable-predgeom-cache
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/predicted-structure-cache-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/predicted-structure-cache-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly prediction_root=${campaign}/structure_prediction/af3_label_free_v1
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/materialize_sable_v0_2_0_predicted_structure_cache_v1.py
readonly script_sha256=9d5f4f087031fb8e02e5ac1cd3cbcc2c624e5fd8d80c57f7076ed8f02231a498
readonly parser=${root}/tools/sabdab2_predicted_geometry_v1/materialize_predicted_structure_cache.py
readonly runner=/data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh
readonly output_root=${campaign}/structure_cache/predicted_label_free_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output_root}"
echo "${script_sha256}  ${script}" | sha256sum --check --strict -

"${python_bin}" "${script}" \
  --input-plan "${prediction_root}/AF3_LABEL_FREE_INPUT_PLAN.json" \
  --input-plan-receipt "${prediction_root}/AF3_LABEL_FREE_INPUT_PLAN_RECEIPT.json" \
  --prediction-root "${prediction_root}" \
  --prediction-parser "${parser}" \
  --af3-runner "${runner}" \
  --output-root "${output_root}"
