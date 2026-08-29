#!/usr/bin/env bash
#SBATCH --job-name=fc-asep-same-ag
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=fc-asep-same-ag-%A_%a.out
#SBATCH --error=fc-asep-same-ag-%A_%a.err

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${data_root}/source"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
run_root="${data_root}/asep/development-runs/epitope-group-v1"
ensemble_root="${data_root}/asep/development-ensembles/epitope-group-v1"
output_base="${data_root}/asep/same-antigen-challenge/epitope-group-v1"
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
if [[ -f "${output_root}/RUN_RECEIPT.json" ]]; then
  exit 0
fi
if [[ -e "${output_root}" ]]; then
  echo "Partial output exists without a complete receipt: ${output_root}" >&2
  exit 2
fi

mkdir -p "${data_root}/asep/results/same-antigen-logs"
exec > >(tee -a "${data_root}/asep/results/same-antigen-logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") \
  2> >(tee -a "${data_root}/asep/results/same-antigen-logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)
cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/evaluate_asep_same_antigen_challenge.py \
  --ensemble-root "${ensemble_root}/${experiment}" \
  --run-roots \
    "${run_root}/${experiment}/seed-17" \
    "${run_root}/${experiment}/seed-42" \
    "${run_root}/${experiment}/seed-73" \
    "${run_root}/${experiment}/seed-101" \
    "${run_root}/${experiment}/seed-137" \
  --cache-root "${data_root}/asep/development-cache" \
  --cache-manifest-sha256 69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7 \
  --challenge "${data_root}/asep/panels/same-antigen-different-antibody-validation-v1.json" \
  --challenge-sha256 846d551f736545cfcdf92788d57f94ee4776c138108d5d093c814c51fabd3a96 \
  --output-root "${output_root}" \
  --device cuda \
  --precision bf16 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260730
