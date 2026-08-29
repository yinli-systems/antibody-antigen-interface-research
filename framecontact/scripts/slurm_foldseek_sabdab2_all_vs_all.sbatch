#!/usr/bin/env bash
#SBATCH --job-name=fc-foldseek-3663
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/source"
output_root="${FRAMECONTACT_FOLDSEEK_JOB_OUTPUT_ROOT:-${data_root}/foldseek_full_all_vs_all_v1}"
log_root="${data_root}/slurm"

mkdir -p "${log_root}"
exec > >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "hostname=$(hostname)"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
scontrol show job "${SLURM_JOB_ID}" -o

FRAMECONTACT_FOLDSEEK="/ssd/scxi253/model_deployment/tools/foldseek-official-avx2-20260730/foldseek/bin/foldseek" \
FRAMECONTACT_ANTIGEN_EXPORT_ROOT="${data_root}/antigen_structures_full_v1" \
FRAMECONTACT_FOLDSEEK_OUTPUT_ROOT="${output_root}" \
FRAMECONTACT_PYTHON_BIN="python3" \
FRAMECONTACT_FOLDSEEK_THREADS="${SLURM_CPUS_PER_TASK}" \
"${source_root}/scripts/run_foldseek_sabdab2_all_vs_all.sh"

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
du -sh "${output_root}"
cat "${output_root}/SHA256SUMS.txt"
