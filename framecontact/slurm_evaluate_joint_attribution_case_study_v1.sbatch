#!/usr/bin/env bash
# Development-only same-antigen counterfactual diagnostic for attribution runs.
#SBATCH --job-name=fc-joint-case-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --array=0-14%15
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-case-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-case-v1-%A_%a.err

set -euo pipefail
data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_REPOSITORY_ROOT:?FRAMECONTACT_REPOSITORY_ROOT must name the immutable attribution source}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
cache_root="${data_root}/asep/development-cache"
run_root_base="${data_root}/asep/development-runs/joint-attribution-dev-v1"
challenge="${repository_root}/protocols/ASEP_SAME_ANTIGEN_CHALLENGE.json"
output_base="${data_root}/asep/development-runs/joint-attribution-case-study-v1"
cache_sha="69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7"
challenge_sha="846d551f736545cfcdf92788d57f94ee4776c138108d5d093c814c51fabd3a96"
seeds=(17 42 73 101 137)
experiments=(framecontact_joint_a1_noisy_or framecontact_joint_attribution_mean_probability framecontact_joint_attribution_length_normalized_noisy_or)
task_index="${SLURM_ARRAY_TASK_ID:-${FC_TASK_INDEX:-}}"
if [[ -z "${task_index}" || "${task_index}" -lt 0 || "${task_index}" -ge 15 ]]; then echo "SLURM_ARRAY_TASK_ID must be in 0..14" >&2; exit 2; fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"
run_root="${run_root_base}/${experiment}/seed-${seed}"
output_root="${output_base}/${experiment}/seed-${seed}"
if [[ -e "${output_root}" ]]; then echo "Refusing to overwrite existing case-study output: ${output_root}" >&2; exit 2; fi
cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/evaluate_asep_same_antigen_challenge.py \
  --run-root "${run_root}" --cache-root "${cache_root}" --cache-manifest-sha256 "${cache_sha}" \
  --challenge "${challenge}" --challenge-sha256 "${challenge_sha}" --output-root "${output_root}" \
  --device cuda --precision bf16 --bootstrap-replicates 10000 --bootstrap-seed 20260816
