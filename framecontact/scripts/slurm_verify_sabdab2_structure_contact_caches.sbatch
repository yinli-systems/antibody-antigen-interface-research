#!/usr/bin/env bash
#SBATCH --job-name=fc-cache-audit
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/source"
cache_root="${data_root}/sabdab2"
output_root="${data_root}/cache_verification_v1"
log_root="${data_root}/slurm"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
mkdir -p "${log_root}" "${output_root}"
exec > >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "hostname=$(hostname)"
scontrol show job "${SLURM_JOB_ID}" -o

PYTHONPATH="${source_root}/src" \
"${python_bin}" "${source_root}/scripts/verify_sabdab2_structure_contact_caches.py" \
  --structure-root "${cache_root}/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-root "${cache_root}/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --output "${output_root}/SABDAB2_STRUCTURE_CONTACT_CACHE_VERIFICATION.json"

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sha256sum "${output_root}/SABDAB2_STRUCTURE_CONTACT_CACHE_VERIFICATION.json"
