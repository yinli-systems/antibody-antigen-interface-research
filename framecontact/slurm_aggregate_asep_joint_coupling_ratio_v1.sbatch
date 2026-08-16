#!/usr/bin/env bash
# Development-only aggregate for the six-condition surface-ratio matrix.
#SBATCH --job-name=fc-joint-ratio-aggregate
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-ratio-aggregate-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-ratio-aggregate-%j.err

set -euo pipefail
data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_AGGREGATION_ROOT:?FRAMECONTACT_AGGREGATION_ROOT must name an immutable deployed commit}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
runs_root="${data_root}/asep/development-runs/joint-coupling-ratio-v1"
protocol="${repository_root}/protocols/framecontact_asep_joint_coupling_ratio_v1.json"
protocol_receipt="${repository_root}/protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_RATIO_V1_RECEIPT.json"
output="${runs_root}/FRAMECONTACT_ASEP_JOINT_COUPLING_RATIO_V1_AGGREGATE.json"

for required in \
  "${repository_root}/scripts/aggregate_asep_joint_coupling_dev_v2.py" \
  "${protocol}" "${protocol_receipt}" "${runs_root}"; do
  [[ -e "${required}" ]] || { echo "Missing aggregation authority: ${required}" >&2; exit 2; }
done
if [[ -e "${output}" || -e "${output}.receipt.json" ]]; then
  echo "Refusing to overwrite an existing ratio aggregate" >&2
  exit 2
fi

cd "${repository_root}"
"${python_bin}" scripts/aggregate_asep_joint_coupling_dev_v2.py \
  --runs-root "${runs_root}" --protocol "${protocol}" \
  --protocol-receipt "${protocol_receipt}" --output "${output}" \
  --bootstrap-resamples 50000
sha256sum "${output}" "${output}.receipt.json"
