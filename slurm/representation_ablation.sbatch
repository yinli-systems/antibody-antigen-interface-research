#!/bin/bash
# ESM-2 alone against ESM-2 + AntiBERTy + ESM-IF1, paired by seed.
#
# The ten-seed replication established that this model beats the
# predicted-marginal outer product by about 16%, and that it peaks at epoch 1
# and then overfits for eleven more. 5.5M parameters against 3,229 instances is
# data-limited, and the lever for a data-limited model is better frozen
# features, not more capacity. AntiBERTy carries antibody statistics the cohort
# cannot supply; ESM-IF1 carries structure the sequence encoder never sees.
#
# Both arms are run here rather than reusing the existing ten seeds as a
# control. Reusing them would compare across a code change, and the point of a
# paired design is that the only difference between arms is the one under test.
# It also gives a reproducibility check: the control should land on the
# already-recorded per-seed numbers.
#
# Array 1-10 is the control, 11-20 is the treatment at seed (task - 10), so the
# two arms share seeds exactly.
#
# Development-only. Trains on the authority train split and evaluates on
# validation. The frozen test split is never named and never read.
#
#   sbatch --export=ALL,\
#ABLATION_SOURCE_ROOT=...,ABLATION_PREPARE_DIR=...,ABLATION_CONTACT_DIR=...,\
#ABLATION_EMBEDDING_DIR=...,ABLATION_SPLIT_MANIFEST=...,ABLATION_OUTPUT_ROOT=...,\
#ABLATION_PYTHON_BIN=...,ABLATION_MANIFEST=...,ABLATION_ANTIBERTY_ROOT=...,\
#ABLATION_ESM_IF1_ROOT=... \
#     slurm/representation_ablation.sbatch

#SBATCH --job-name=repr-ablation
# gpu_5090 has free cards but its RTX 5090s are sm_120, and the torch this
# project runs (2.4.1+cu121) compiles only up to sm_90.
#SBATCH --partition=gpu_4090
#SBATCH --array=1-20
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

SOURCE_ROOT=${ABLATION_SOURCE_ROOT:?ABLATION_SOURCE_ROOT is required}
PREPARE_DIR=${ABLATION_PREPARE_DIR:?ABLATION_PREPARE_DIR is required}
CONTACT_DIR=${ABLATION_CONTACT_DIR:?ABLATION_CONTACT_DIR is required}
EMBEDDING_DIR=${ABLATION_EMBEDDING_DIR:?ABLATION_EMBEDDING_DIR is required}
SPLIT_MANIFEST=${ABLATION_SPLIT_MANIFEST:?ABLATION_SPLIT_MANIFEST is required}
OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT:?ABLATION_OUTPUT_ROOT is required}
PYTHON_BIN=${ABLATION_PYTHON_BIN:?ABLATION_PYTHON_BIN is required}
MAX_EPOCHS=${ABLATION_MAX_EPOCHS:-12}

TASK=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}
if [ "${TASK}" -le 10 ]; then
    ARM=esm2
    SEED=${TASK}
    EXTRA_ARGS=()
else
    ARM=esm2_antiberty_esmif1
    SEED=$((TASK - 10))
    EXTRA_ARGS=(
        --representation-manifest "${ABLATION_MANIFEST:?treatment arm needs ABLATION_MANIFEST}"
        --antiberty-root "${ABLATION_ANTIBERTY_ROOT:?treatment arm needs ABLATION_ANTIBERTY_ROOT}"
        --esm-if1-root "${ABLATION_ESM_IF1_ROOT:?treatment arm needs ABLATION_ESM_IF1_ROOT}"
    )
fi
RUN_ROOT="${OUTPUT_ROOT}/${ARM}/seed_${SEED}"

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SOURCE_ROOT}"

echo "[provenance] split manifest sha256: $(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)"
echo "[provenance] arm=${ARM} seed=${SEED} max_epochs=${MAX_EPOCHS}"

echo "[train] ${ARM} seed ${SEED}"
"${PYTHON_BIN}" -u scripts/sabdab2_interaction_first.py train \
    --prepare-dir "${PREPARE_DIR}" \
    --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${RUN_ROOT}/train" \
    --seed "${SEED}" \
    --max-epochs "${MAX_EPOCHS}" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "[evaluate] ${ARM} seed ${SEED} on validation"
"${PYTHON_BIN}" -u scripts/sabdab2_interaction_first.py evaluate \
    --prepare-dir "${PREPARE_DIR}" \
    --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --checkpoint "${RUN_ROOT}/train/best.pt" \
    --output-dir "${RUN_ROOT}/validation" \
    --split validation \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "[done] ${ARM} seed ${SEED}"
