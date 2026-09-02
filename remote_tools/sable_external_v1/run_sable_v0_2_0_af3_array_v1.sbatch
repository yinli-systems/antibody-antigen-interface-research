#!/usr/bin/env bash
#SBATCH --job-name=sable-af3-prediction
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/af3-prediction-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/af3-prediction-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly prediction_root=${campaign}/structure_prediction/af3_label_free_v1
readonly runner=/data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh
readonly runner_sha256=996c1886f7b77454d24e93c402a3e19078e5d8da0191dfdc6b08368206c14aea
readonly batch_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly batch_dir=$(printf '%s/batches/batch-%03d' "${prediction_root}" "${batch_index}")

test "${SLURM_JOB_PARTITION}" = gpu_4090
test -d "${batch_dir}/inputs"
test ! -e "${batch_dir}/results"
test ! -e "${batch_dir}/RESULT_SHA256SUMS.txt"
echo "${runner_sha256}  ${runner}" | sha256sum --check --strict -

mkdir -p "${batch_dir}/results"
cd "${batch_dir}"
export AF3_USE_NODE_SANDBOX=0

echo "SABLE_AF3_BATCH_START batch=${batch_index} job=${SLURM_JOB_ID} host=$(hostname) time=$(date -Is)"
bash "${runner}" inputs results

readonly input_count=$(find inputs -maxdepth 1 -type f -name '*.json' | wc -l)
readonly model_count=$(find results -mindepth 2 -maxdepth 2 -type f -name '*_model.cif' | wc -l)
if [[ "${input_count}" -ne "${model_count}" ]]; then
  echo "SABLE_AF3_OUTPUT_ERROR batch=${batch_index} inputs=${input_count} models=${model_count}" >&2
  exit 3
fi
find results -type f -print0 | sort -z | xargs -0 sha256sum > RESULT_SHA256SUMS.txt
echo "SABLE_AF3_BATCH_PASS batch=${batch_index} inputs=${input_count} models=${model_count} time=$(date -Is)"
