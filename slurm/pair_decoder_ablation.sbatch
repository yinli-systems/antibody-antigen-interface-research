#!/bin/bash
# Pair-decoder ablation: bilinear against hadamard_mlp on real contact data.
#
# Development-only. Trains on the authority train split and evaluates on
# validation. The frozen test split is never named and never read.
#
# One array task per decoder. Everything else -- split, seed, embeddings,
# schedule, losses -- is identical, so the decoder is the only difference.
#
#   sbatch --export=ALL,\
#ABLATION_SOURCE_ROOT=...,ABLATION_PREPARE_DIR=...,ABLATION_CONTACT_DIR=...,\
#ABLATION_EMBEDDING_DIR=...,ABLATION_SPLIT_MANIFEST=...,ABLATION_OUTPUT_ROOT=...,\
#ABLATION_PYTHON_BIN=... \
#     slurm/pair_decoder_ablation.sbatch

#SBATCH --job-name=decoder-ablation
#SBATCH --partition=gpu_4090
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# gpu_4090 forbids --mem/--mem-per-cpu/--mem-per-gpu: the site's job_submit_lua
# plugin allocates 60GB per card automatically and rejects any explicit request.
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

SOURCE_ROOT=${ABLATION_SOURCE_ROOT:?ABLATION_SOURCE_ROOT is required}
PREPARE_DIR=${ABLATION_PREPARE_DIR:?ABLATION_PREPARE_DIR is required}
CONTACT_DIR=${ABLATION_CONTACT_DIR:?ABLATION_CONTACT_DIR is required}
EMBEDDING_DIR=${ABLATION_EMBEDDING_DIR:?ABLATION_EMBEDDING_DIR is required}
SPLIT_MANIFEST=${ABLATION_SPLIT_MANIFEST:?ABLATION_SPLIT_MANIFEST is required}
OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT:?ABLATION_OUTPUT_ROOT is required}
PYTHON_BIN=${ABLATION_PYTHON_BIN:?ABLATION_PYTHON_BIN is required}
SEED=${ABLATION_SEED:-42}
MAX_EPOCHS=${ABLATION_MAX_EPOCHS:-30}
INPUT_DIM=${ABLATION_INPUT_DIM:-1280}

DECODERS=(bilinear hadamard_mlp)
DECODER=${DECODERS[${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}]}
RUN_ROOT="${OUTPUT_ROOT}/${DECODER}"

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SOURCE_ROOT}"

# The split file has drifted from the digest the S2 assignment registered: its
# train membership and sealed pool both still reproduce, but one instance moved
# between test and validation. Record what was actually used, so a result is
# attributable to a specific split rather than to whatever was on disk.
echo "[provenance] split manifest sha256: $(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)"
echo "[provenance] decoder: ${DECODER}  seed: ${SEED}"

echo "[train] ${DECODER}"
"${PYTHON_BIN}" scripts/sabdab2_interaction_first.py train \
    --prepare-dir "${PREPARE_DIR}" \
    --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${RUN_ROOT}/train" \
    --input-dim "${INPUT_DIM}" \
    --pair-decoder "${DECODER}" \
    --seed "${SEED}" \
    --max-epochs "${MAX_EPOCHS}" \
    --device cuda

# Validation only. Naming the frozen test split here would be a protocol breach,
# not a convenience.
echo "[evaluate] ${DECODER} on validation"
"${PYTHON_BIN}" scripts/sabdab2_interaction_first.py evaluate \
    --prepare-dir "${PREPARE_DIR}" \
    --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --checkpoint "${RUN_ROOT}/train/best.pt" \
    --output-dir "${RUN_ROOT}/validation" \
    --split validation \
    --device cuda

echo "[done] ${DECODER}"
