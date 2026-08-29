#!/usr/bin/env bash
# Development-only aggregate for the joint attribution same-antigen diagnostic.
#SBATCH --job-name=fc-joint-case-aggregate
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-case-aggregate-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-case-aggregate-%j.err

set -euo pipefail
data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_AGGREGATION_ROOT:?FRAMECONTACT_AGGREGATION_ROOT must name an immutable deployed commit}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
runs_root="${data_root}/asep/development-runs/joint-attribution-case-study-v1"
output="${runs_root}/FRAMECONTACT_ASEP_JOINT_ATTRIBUTION_CASE_STUDY_V1_AGGREGATE.json"
cache_sha="69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7"
challenge_sha="846d551f736545cfcdf92788d57f94ee4776c138108d5d093c814c51fabd3a96"
conditions=(framecontact_joint_a1_noisy_or framecontact_joint_attribution_mean_probability framecontact_joint_attribution_length_normalized_noisy_or)
seeds=(17 42 73 101 137)

for required in "${repository_root}/scripts/aggregate_asep_joint_attribution_case_study_v1.py" "${runs_root}"; do
  [[ -e "${required}" ]] || { echo "Missing case-study aggregation authority: ${required}" >&2; exit 2; }
done
if [[ -e "${output}" || -e "${output}.receipt.json" ]]; then
  echo "Refusing to overwrite an existing case-study aggregate" >&2
  exit 2
fi

cd "${repository_root}"
"${python_bin}" scripts/aggregate_asep_joint_attribution_case_study_v1.py \
  --runs-root "${runs_root}" --conditions "${conditions[@]}" --seeds "${seeds[@]}" \
  --cache-manifest-sha256 "${cache_sha}" --challenge-sha256 "${challenge_sha}" \
  --output "${output}"
sha256sum "${output}" "${output}.receipt.json"
