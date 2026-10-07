#!/bin/bash
# Powered replication of the margin over the predicted-marginal outer product.
#
# The single-seed ablation put the pair head 32.7% above that baseline in macro
# AUPRC. One observation cannot establish it. This repeats the *same* arm across
# seeds so the margin gets a confidence interval.
#
# What is being replicated is the margin over the baseline, not a comparison
# between decoders: the two decoders differed in the third decimal and disagreed
# in direction across metrics, so that comparison is not what deserves compute.
#
# Development-only. Trains on the authority train split and evaluates on
# validation. The frozen test split is never named and never read.
#
#   sbatch --export=ALL,\
#MARGIN_SOURCE_ROOT=...,MARGIN_PREPARE_DIR=...,MARGIN_CONTACT_DIR=...,\
#MARGIN_EMBEDDING_DIR=...,MARGIN_SPLIT_MANIFEST=...,MARGIN_OUTPUT_ROOT=...,\
#MARGIN_PYTHON_BIN=... \
#     slurm/pair_baseline_margin_replication.sbatch

#SBATCH --job-name=margin-replication
# gpu_5090 has free cards but its RTX 5090s are sm_120, and the torch this
# project runs (2.4.1+cu121) compiles only up to sm_90. Every task there dies
# at .to(device) within a minute. gpu_4090 queues but works.
#SBATCH --partition=gpu_4090
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# gpu_4090 forbids --mem/--mem-per-cpu/--mem-per-gpu: the site's job_submit_lua
# plugin allocates memory per card and rejects any explicit request.
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

SOURCE_ROOT=${MARGIN_SOURCE_ROOT:?MARGIN_SOURCE_ROOT is required}
PREPARE_DIR=${MARGIN_PREPARE_DIR:?MARGIN_PREPARE_DIR is required}
CONTACT_DIR=${MARGIN_CONTACT_DIR:?MARGIN_CONTACT_DIR is required}
EMBEDDING_DIR=${MARGIN_EMBEDDING_DIR:?MARGIN_EMBEDDING_DIR is required}
SPLIT_MANIFEST=${MARGIN_SPLIT_MANIFEST:?MARGIN_SPLIT_MANIFEST is required}
OUTPUT_ROOT=${MARGIN_OUTPUT_ROOT:?MARGIN_OUTPUT_ROOT is required}
PYTHON_BIN=${MARGIN_PYTHON_BIN:?MARGIN_PYTHON_BIN is required}
DECODER=${MARGIN_DECODER:-bilinear}
INPUT_DIM=${MARGIN_INPUT_DIM:-1280}

# The single-seed run peaked at epoch 1 and then overfitted for ten more, so a
# long budget buys nothing here; patience still governs the stop.
MAX_EPOCHS=${MARGIN_MAX_EPOCHS:-12}

SEED=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}
RUN_ROOT="${OUTPUT_ROOT}/seed_${SEED}"

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SOURCE_ROOT}"

echo "[provenance] split manifest sha256: $(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)"
echo "[provenance] decoder=${DECODER} seed=${SEED} max_epochs=${MAX_EPOCHS}"

echo "[train] seed ${SEED}"
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

echo "[evaluate] seed ${SEED} on validation"
"${PYTHON_BIN}" scripts/sabdab2_interaction_first.py evaluate \
    --prepare-dir "${PREPARE_DIR}" \
    --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --checkpoint "${RUN_ROOT}/train/best.pt" \
    --output-dir "${RUN_ROOT}/validation" \
    --split validation \
    --device cuda

echo "[done] seed ${SEED}"
