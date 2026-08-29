#!/usr/bin/env bash
#SBATCH --job-name=fc-joint-v2-aggregate
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=fc-joint-v2-aggregate-%j.out
#SBATCH --error=fc-joint-v2-aggregate-%j.err

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_AGGREGATION_ROOT:?FRAMECONTACT_AGGREGATION_ROOT must name an immutable deployed commit}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
runs_root="${data_root}/asep/development-runs/joint-coupling-dev-v2"
protocol="${repository_root}/protocols/framecontact_asep_joint_coupling_dev_v2.json"
protocol_receipt="${repository_root}/protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V2_RECEIPT.json"
output="${runs_root}/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V2_AGGREGATE.json"

for required in \
  "${repository_root}/scripts/aggregate_asep_joint_coupling_dev_v2.py" \
  "${protocol}" \
  "${protocol_receipt}" \
  "${runs_root}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required development aggregation authority is missing: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${output}" || -e "${output}.receipt.json" ]]; then
  echo "Refusing to overwrite an existing development aggregate" >&2
  exit 2
fi

cd "${repository_root}"
"${python_bin}" scripts/aggregate_asep_joint_coupling_dev_v2.py \
  --runs-root "${runs_root}" \
  --protocol "${protocol}" \
  --protocol-receipt "${protocol_receipt}" \
  --output "${output}" \
  --bootstrap-resamples 50000

sha256sum "${output}" "${output}.receipt.json"
