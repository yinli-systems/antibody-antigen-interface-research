#!/bin/bash
# Powered replication of the intra-chain geometry comparison.
#
# Governed by protocols/SABDAB2_GEOMETRY_POWERED_REPLICATION_PROTOCOL_V1.md.
# Development-only: no checkpoint selection, no outer or frozen test, no contact
# training. One array task per seed; every task runs all four modes in a single
# process so the modes share an initial state and stay paired.
#
# Submit with the roots supplied explicitly, for example:
#
#   sbatch --export=ALL,\
#GEOMETRY_CACHE_ROOT=...,GEOMETRY_EMBEDDING_ROOT=...,GEOMETRY_LABEL_ROOT=...,\
#GEOMETRY_OUTPUT_ROOT=...,GEOMETRY_SOURCE_ROOT=... \
#     slurm/sabdab2_geometry_powered_replication.sbatch

#SBATCH --job-name=geom-powered-repl
#SBATCH --partition=gpu_4090
#SBATCH --array=1-24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

SOURCE_ROOT=${GEOMETRY_SOURCE_ROOT:?GEOMETRY_SOURCE_ROOT is required}
CACHE_ROOT=${GEOMETRY_CACHE_ROOT:?GEOMETRY_CACHE_ROOT is required}
EMBEDDING_ROOT=${GEOMETRY_EMBEDDING_ROOT:?GEOMETRY_EMBEDDING_ROOT is required}
LABEL_ROOT=${GEOMETRY_LABEL_ROOT:?GEOMETRY_LABEL_ROOT is required}
OUTPUT_ROOT=${GEOMETRY_OUTPUT_ROOT:?GEOMETRY_OUTPUT_ROOT is required}
PYTHON_BIN=${GEOMETRY_PYTHON_BIN:?GEOMETRY_PYTHON_BIN is required}

# Frozen by the protocol. A mismatch is a stop, not a warning.
EXPECTED_CACHE_MANIFEST_SHA256=2580e41ced744ade8a16fa94ae4b9b22bda5e85b7477a71ae1344ed732bf3e6d

SEED=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}
RUN_ROOT="${OUTPUT_ROOT}/seed_${SEED}"

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${SOURCE_ROOT}"

# --- preflight ----------------------------------------------------------
# Both gates are cheap and dependency-light. They run in every array task so a
# mid-run source change cannot slip past the one task that happened to start
# first. A failure stops this task under `set -e`; it is never overridden.

echo "[preflight] partner-pose contract"
"${PYTHON_BIN}" scripts/verify_partner_pose_contract.py

echo "[preflight] protocol source bindings"
"${PYTHON_BIN}" scripts/verify_protocol_source_bindings.py

# --- run ----------------------------------------------------------------
# The trainer refuses to overwrite an existing output root, so a resubmitted
# seed fails loudly instead of silently replacing a recorded result.

echo "[run] seed ${SEED} -> ${RUN_ROOT}"
"${PYTHON_BIN}" scripts/train_sabdab2_structure_ablation.py \
    --cache-root "${CACHE_ROOT}" \
    --embedding-root "${EMBEDDING_ROOT}" \
    --label-root "${LABEL_ROOT}" \
    --output-root "${RUN_ROOT}" \
    --expected-cache-manifest-sha256 "${EXPECTED_CACHE_MANIFEST_SHA256}" \
    --modes topology_only distance distance_direction full_pose \
    --seed "${SEED}" \
    --epochs 3 \
    --batch-size 4 \
    --max-antigen-length 384 \
    --d-model 64 \
    --num-heads 4 \
    --num-layers 1 \
    --num-rbf 16 \
    --neighbor-k 32 \
    --neighbor-radius 16.0 \
    --dropout 0.1 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --precision bf16 \
    --device cuda

echo "[done] seed ${SEED}"
