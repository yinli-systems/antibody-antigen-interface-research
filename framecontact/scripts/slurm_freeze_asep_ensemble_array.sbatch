#!/usr/bin/env bash
#SBATCH --job-name=fc-asep-ensemble
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=fc-asep-ensemble-%A_%a.out
#SBATCH --error=fc-asep-ensemble-%A_%a.err

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${data_root}/source"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
run_root="${data_root}/asep/development-runs/epitope-group-v1"
output_base="${data_root}/asep/development-ensembles/epitope-group-v1"
experiments=(
  sequence_dense
  spatial_topology
  distance
  distance_direction
  full_pose
  frozen_frame_ipa
  framecontact
  framecontact_interleaved
)
experiment="${experiments[${SLURM_ARRAY_TASK_ID}]}"
output_root="${output_base}/${experiment}"
if [[ -f "${output_root}/ENSEMBLE_RECEIPT.json" ]]; then
  exit 0
fi
if [[ -e "${output_root}" ]]; then
  echo "Partial output exists without a complete receipt: ${output_root}" >&2
  exit 2
fi

mkdir -p "${data_root}/asep/results/ensemble-logs"
exec > >(tee -a "${data_root}/asep/results/ensemble-logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") \
  2> >(tee -a "${data_root}/asep/results/ensemble-logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)
cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/freeze_asep_development_ensemble.py \
  --run-roots \
    "${run_root}/${experiment}/seed-17" \
    "${run_root}/${experiment}/seed-42" \
    "${run_root}/${experiment}/seed-73" \
    "${run_root}/${experiment}/seed-101" \
    "${run_root}/${experiment}/seed-137" \
  --cache-root "${data_root}/asep/development-cache" \
  --cache-manifest-sha256 69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7 \
  --protocol "${repository_root}/protocols/framecontact_asep_epitope_group_v1.json" \
  --protocol-receipt "${repository_root}/protocols/FRAMECONTACT_ASEP_EPITOPE_GROUP_V1_RECEIPT.json" \
  --protocol-receipt-sha256 370f36155e54be15c9b5ef4460a768a5c82a36bf561a206424b9d09e0d0b05fb \
  --training-partner-panel "${data_root}/asep/panels/train-reciprocal-31.json" \
  --training-partner-panel-sha256 a0c5f3e8ea1a0edf7cd9a2043b326ee3be7b054c81dff9fbfcb510db6f2969d1 \
  --validation-partner-panel "${data_root}/asep/panels/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 10f50aa122568d4ed1a8261637f2e807f2702997a1aa5d9925983d37cc0d83ec \
  --output-root "${output_root}" \
  --device cuda \
  --precision bf16
