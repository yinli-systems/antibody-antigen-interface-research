#!/usr/bin/env bash
#SBATCH --job-name=fc-sab-smoke
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

: "${FRAMECONTACT_REPOSITORY_ROOT:?missing FRAMECONTACT_REPOSITORY_ROOT}"
data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
python_bin="${FRAMECONTACT_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/epiformer/bin/python}"
output_root="${FRAMECONTACT_SMOKE_OUTPUT_ROOT:-${data_root}/sabdab2/smoke-v1}"
mkdir -p "${output_root}" "${data_root}/slurm"
exec > >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

cd "${FRAMECONTACT_REPOSITORY_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${FRAMECONTACT_REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" scripts/smoke_sabdab2_framecontact_gpu.py \
  --structure-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --output "${output_root}/SABDAB2_GPU_SMOKE.json" \
  --device cuda \
  --precision bf16
sha256sum "${output_root}/SABDAB2_GPU_SMOKE.json"
