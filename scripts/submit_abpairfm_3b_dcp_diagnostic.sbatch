#!/usr/bin/env bash
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --job-name=abpairfm-3b-dcp-diag
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-dcp-diag-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-dcp-diag-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
PROJECT_ROOT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
RUN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/dcp/job-1458557
DIAGNOSTIC_ROOT="$(dirname "${RUN_ROOT}")/diagnostic-${SLURM_JOB_ID}"
OFFICIAL_CHECKPOINT=/ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
CHECKPOINT_SIZE=5678116398
CHECKPOINT_SHA256=7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500
SOURCE_SHA256=9bab2019d37cc454397d04d5a6ee1bba394d78cd456699be7bc86a3dd5acb766
PROTOCOL="${PROJECT_ROOT}/protocols/abpairfm_3b_dcp_recovery_gate_v1.json"
PROTOCOL_SHA256=3a046c99bb6d501e3d0c7116493f621940e4f43dad0df2860cc6f0517150ed57
SAVE_RESULT="${RUN_ROOT}/save_result.json"
SAVE_RESULT_SHA256="$(${PYTHON} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${SAVE_RESULT}")"

mkdir -p "${DIAGNOSTIC_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
  --project-root "${PROJECT_ROOT}"
  --checkpoint "${OFFICIAL_CHECKPOINT}"
  --expected-checkpoint-size "${CHECKPOINT_SIZE}"
  --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}"
  --expected-source-sha256 "${SOURCE_SHA256}"
  --protocol "${PROTOCOL}"
  --expected-protocol-sha256 "${PROTOCOL_SHA256}"
  --output-root "${DIAGNOSTIC_ROOT}"
  --save-result "${SAVE_RESULT}"
  --expected-save-result-sha256 "${SAVE_RESULT_SHA256}"
)

srun --ntasks=1 --cpus-per-task=24 bash -lc "nvidia-smi --query-gpu=name --format=csv,noheader"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_ROOT}/scripts/diagnose_abpairfm_3b_dcp_mismatch.py" \
  --mode baseline "${COMMON[@]}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_ROOT}/scripts/diagnose_abpairfm_3b_dcp_mismatch.py" \
  --mode restore "${COMMON[@]}"
