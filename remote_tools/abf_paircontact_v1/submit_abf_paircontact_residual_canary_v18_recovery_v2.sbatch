#!/usr/bin/env bash
#SBATCH --job-name=abf-paircontact-v18-r2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_paircontact_residual_canary_v18_recovery_v2/slurm-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_paircontact_residual_canary_v18_recovery_v2/slurm-%j.err

set -euo pipefail
umask 077

readonly python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly project=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/5084815_paircontact_residual_v18
readonly source_sha256=5084815803bead4497c2cc723aa375af0ab25fe3ecaa4cb973d873b57e1a518b
readonly protocol=${project}/protocols/abpairfm_stage_a1_paircontact_residual_canary_v18_recovery_v2.json
readonly protocol_sha256=a91ef7a4b7fe20db4208d12e07debd6eaef168c74e65ba5224c6d5d9dcfced20
readonly audit=${project}/scripts/audit_abf_paircontact_residual_canary.py
readonly audit_sha256=29a6e995230b3229ca8eda70a8b9e6bd3eb365b2d0ead9c3799529ce65027f2f
readonly failure_receipt=/ssd/scxi253/model_deployment/tools/abf_paircontact_v1/PHASE1_MECHANISM_CANARY_FAILURE_V1.json
readonly failure_receipt_sha256=f5fe4e07a04b402550f00e2bcbe88d617c9fade5d861221d28856a026418e3d6
readonly output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_paircontact_residual_canary_v18_recovery_v2

test "${SLURM_JOB_PARTITION}" = gpu_4090
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#gpu_names[@]}" -eq 4
for name in "${gpu_names[@]}"; do
  test "${name}" = "NVIDIA GeForce RTX 4090"
done

printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  "${audit_sha256}" "${audit}" \
  "${failure_receipt_sha256}" "${failure_receipt}" \
  45e0ff7deb6a1e1ef54fb5efba1c82bd8d918e34061ad4c4201e903097bccaed /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST.jsonl.zst \
  2513b8548baca2755ca681e8fc2589ab45922d71491966551aeb82ecd754d74a /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST_RECEIPT.json \
  | sha256sum --check --strict

mkdir -p "${output}"
test ! -e "${output}/RESULT.json"
test ! -e "${output}/AUDIT.json"
test ! -e "${output}/PROGRESS.json"
test ! -e "${output}/checkpoints/pilot-000000"

export PYTHONPATH="${project}/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABPAIRFM_TRACE_LAYERS=0

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${project}/scripts/run_abf_paircontact_residual_canary.py" \
  --project-root "${project}" \
  --protocol "${protocol}" \
  --expected-protocol-sha256 "${protocol_sha256}" \
  --expected-source-sha256 "${source_sha256}" \
  --manifest /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST.jsonl.zst \
  --manifest-receipt /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST_RECEIPT.json \
  --initial-checkpoint /ssd/scxi253/model_deployment/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt \
  --expected-initial-checkpoint-size 5678116398 \
  --expected-initial-checkpoint-sha256 7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500 \
  --input-checkpoint-root /ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_string25m_v1/checkpoints \
  --input-generation segment-000127 \
  --expected-input-content-sha256 3f3e79d4b49fa16c3d837a1417ed3c8b677da11e357e5fb9af133edcfa83bd2e \
  --checkpoint-root "${output}/checkpoints" \
  --result "${output}/RESULT.json" \
  --progress "${output}/PROGRESS.json" \
  --expected-world-size 4 \
  --shard-group-size 4 \
  --maximum-tokens 1024 \
  --efficiency-microbatch-pairs-per-rank 48 \
  --efficiency-activation-checkpoint-mode off \
  --progress-interval 10 \
  --rank-phase-telemetry off

readonly result_sha256=$(sha256sum "${output}/RESULT.json" | cut -d' ' -f1)
"${python_bin}" "${audit}" \
  --result "${output}/RESULT.json" \
  --expected-result-sha256 "${result_sha256}" \
  --protocol "${protocol}" \
  --expected-protocol-sha256 "${protocol_sha256}" \
  --expected-source-sha256 "${source_sha256}" \
  --output "${output}/AUDIT.json"
