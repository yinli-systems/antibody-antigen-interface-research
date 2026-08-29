#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-joint-emb-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/logs/%x-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly PROJECT=${ABPAIRFM_SOURCE_ROOT:?ABPAIRFM_SOURCE_ROOT must name an immutable source tree}
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly CHECKPOINT_PROTOCOL=${PROJECT}/protocols/abpairfm_stage_a1_geometry_development_contact_v2.json
readonly EMBEDDING_PROTOCOL=${PROJECT}/protocols/abpairfm_sabdab2_joint_embedding_v1.json
readonly INITIAL_CHECKPOINT=${ROOT}/models/abpairfm_3b_v1/esm2_t36_3B_UR50D/esm2_t36_3B_UR50D.pt
readonly DCP_ROOT=${ROOT}/results/pretraining/abpairfm_3b_stage_a1_geometry_ranking_canary_v17/checkpoints
readonly PROMOTION_RESULT=${ROOT}/results/pretraining/abpairfm_3b_stage_a1_geometry_ranking_canary_v17/RESULT.json
readonly DATA_ROOT=${ROOT}/data/framecontact_20260730
readonly SEQUENCE_VIEW_ROOT=${DATA_ROOT}/authorities/abpairfm_joint_embedding_v1
readonly OUTPUT_ROOT=${ROOT}/data/abpairfm_sabdab2_joint_embeddings_v1

mkdir -p "${ROOT}/results/logs"
test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${OUTPUT_ROOT}"
test "$(sha256sum "${CHECKPOINT_PROTOCOL}" | cut -d' ' -f1)" = 38cec6cd23a934d1a8ad9be0ecd92c9e4b742dd4b095e994d03aa18e961ca69a
test "$(sha256sum "${EMBEDDING_PROTOCOL}" | cut -d' ' -f1)" = a74bb6890dc35e9090926a8e925c578d880ed2517deec5f47f840993e07eea7c
test "$(sha256sum "${PROJECT}/scripts/materialize_abpairfm_sabdab2_joint_embeddings.py" | cut -d' ' -f1)" = 95c5862d9abc96e7937f2dff7eb8b982269af7a0cbcb3759316b0e87204cd63e

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=${PROJECT}/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6

cd "${PROJECT}"
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  scripts/materialize_abpairfm_sabdab2_joint_embeddings.py \
  --project-root "${PROJECT}" \
  --checkpoint-protocol "${CHECKPOINT_PROTOCOL}" \
  --expected-checkpoint-protocol-sha256 38cec6cd23a934d1a8ad9be0ecd92c9e4b742dd4b095e994d03aa18e961ca69a \
  --embedding-protocol "${EMBEDDING_PROTOCOL}" \
  --expected-embedding-protocol-sha256 a74bb6890dc35e9090926a8e925c578d880ed2517deec5f47f840993e07eea7c \
  --initial-checkpoint "${INITIAL_CHECKPOINT}" \
  --dcp-root "${DCP_ROOT}" \
  --dcp-generation pilot-000000 \
  --expected-dcp-content-sha256 87d1a57148abde8bf47b3aaea4437219ad9c71f0df5d99a53e9de800b48bc479 \
  --promotion-result "${PROMOTION_RESULT}" \
  --sequence-view "${SEQUENCE_VIEW_ROOT}/sabdab2_train_validation_sequences_v1.jsonl" \
  --expected-sequence-view-sha256 27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468 \
  --sequence-view-receipt "${SEQUENCE_VIEW_ROOT}/SABDAB2_TRAIN_VALIDATION_SEQUENCE_VIEW_V1_RECEIPT.json" \
  --expected-sequence-view-receipt-sha256 36a9249d5d1cec4065292e79d1efae8623987a4643675557f32252bcde7752ee \
  --split-manifest "${DATA_ROOT}/split_matrix_v3_1/antigen_fold.csv" \
  --expected-split-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --output-root "${OUTPUT_ROOT}" \
  --maximum-window-residues 510 \
  --maximum-batch-size 7 \
  --maximum-attention-cells 7340032
