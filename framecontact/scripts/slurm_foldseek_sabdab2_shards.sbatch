#!/usr/bin/env bash
#SBATCH --job-name=fc-foldseek-shards
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/source"
sharded_root="${data_root}/foldseek_sharded_4_v1"
log_root="${data_root}/slurm"
mkdir -p "${log_root}"
exec > >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") \
  2> >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "hostname=$(hostname)"
scontrol show job "${SLURM_JOB_ID}" -o

FRAMECONTACT_FOLDSEEK="/ssd/scxi253/model_deployment/tools/foldseek-official-avx2-20260730/foldseek/bin/foldseek" \
FRAMECONTACT_ANTIGEN_EXPORT_ROOT="${data_root}/antigen_structures_full_v1" \
FRAMECONTACT_FOLDSEEK_SHARD_PLAN_ROOT="${sharded_root}/query-plan" \
FRAMECONTACT_FOLDSEEK_SHARD_OUTPUT_ROOT="${sharded_root}/shard-outputs" \
FRAMECONTACT_FOLDSEEK_SHARD_INDEX="${SLURM_ARRAY_TASK_ID}" \
FRAMECONTACT_FOLDSEEK_THREADS="${SLURM_CPUS_PER_TASK}" \
FRAMECONTACT_PYTHON_BIN="python3" \
"${source_root}/scripts/run_foldseek_sabdab2_shard.sh"

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
