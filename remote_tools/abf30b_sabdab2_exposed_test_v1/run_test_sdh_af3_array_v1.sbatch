#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-sdh-af3
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/sdh-af3-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/sdh-af3-%A_%a.err

set -euo pipefail
readonly campaign=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1
readonly af3_root=${campaign}/label_free_inputs_v1/sdh_af3
readonly batch_dir=$(printf '%s/batches/batch-%03d' "${af3_root}" "${SLURM_ARRAY_TASK_ID}")
readonly runner=/data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh
test "${SLURM_JOB_PARTITION}" = gpu_4090
test -d "${batch_dir}/inputs"
test ! -e "${batch_dir}/results"
mkdir -p "${batch_dir}/results"
cd "${batch_dir}"
export AF3_USE_NODE_SANDBOX=0
bash "${runner}" inputs results
readonly inputs=$(find inputs -maxdepth 1 -type f -name '*.json' | wc -l)
readonly models=$(find results -mindepth 2 -maxdepth 2 -type f -name '*_model.cif' | wc -l)
test "${inputs}" -eq "${models}"
find results -type f -print0 | sort -z | xargs -0 sha256sum > RESULT_SHA256SUMS.txt
printf 'PASS_TEST_SDH_AF3 batch=%s inputs=%s models=%s\n' "${SLURM_ARRAY_TASK_ID}" "${inputs}" "${models}"
