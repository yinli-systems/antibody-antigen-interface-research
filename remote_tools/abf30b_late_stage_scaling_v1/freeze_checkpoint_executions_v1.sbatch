#!/usr/bin/env bash
#SBATCH --job-name=abf-late-scale-freeze
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/freeze-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/freeze-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_late_stage_scaling_v1
readonly tools=${root}/tools/abf30b_late_stage_scaling_v1
readonly protocol=${campaign}/authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1.json
readonly protocol_receipt=${campaign}/authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1_RECEIPT.json
readonly base_campaign=${root}/results/development/abf30b_framecontact_v1
readonly reference_execution=${base_campaign}/authority/development_execution_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly label=segment_000058 ;;
  1) readonly label=segment_000063 ;;
  *) exit 64 ;;
esac

readonly checkpoint_root=${campaign}/checkpoints/${label}
readonly canary=${checkpoint_root}/authority/DCP_EMBEDDING_CANARY_V1.json
readonly canary_receipt=${checkpoint_root}/authority/DCP_EMBEDDING_CANARY_V1_RECEIPT.json
readonly embedding_root=${checkpoint_root}/embeddings/sabdab2_abf_antibody_esm2_antigen_v1
readonly execution_root=${checkpoint_root}/authority/development_execution_v1

test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha256=$("${python_bin}" - "${protocol}" "${protocol_receipt}" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); receipt=json.load(open(sys.argv[2]))
digest=hashlib.sha256(raw).hexdigest(); assert receipt['protocol_sha256']==digest
print(digest)
PY
)

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${tools}/freeze_checkpoint_development_execution_v1.py" \
  --scaling-protocol "${protocol}" --scaling-protocol-sha256 "${protocol_sha256}" \
  --checkpoint-label "${label}" --canary "${canary}" \
  --canary-receipt "${canary_receipt}" --embedding-root "${embedding_root}" \
  --reference-execution-root "${reference_execution}" \
  --baseline-source-root "${root}/immutable_stage_a1_candidates/abf30b_embedding_baseline_source_v1" \
  --topk2-source-root "${root}/immutable_stage_a1_candidates/abf30b_embedding_topk2_source_v1" \
  --original-source-root "${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_distance_labels_source_v4" \
  --output-root "${execution_root}"
