#!/bin/bash
# Train and evaluate on AsEP's own split, so this project has a number that can
# be compared to published work.
#
# Nothing here is currently comparable to anything outside it. AsEP is the
# benchmark WALLE and EpiFormer quote, and its epitope_group split is the harder
# of the two it publishes -- WALLE reports MCC 0.077 there.
#
# ASEP_SPLIT decides what is read. The default, validation, is development.
# Reading test is a benchmark measurement and belongs at the end of a
# development cycle, once, with the configuration already fixed; it is not a
# split to iterate against.
#
# AsEP models a different object from the sabdab2 cohort: its antibody side is
# CDR residues only, roughly 60 of them against 220 for a full variable region.
# Numbers are comparable to AsEP's own leaderboard, not to this project's
# sabdab2 figures.
#
#   sbatch --array=1-10 --export=ALL,ASEP_SOURCE_ROOT=...,ASEP_DATA_ROOT=...,\
#ASEP_SPLIT_MANIFEST=...,ASEP_OUTPUT_ROOT=...,ASEP_PYTHON_BIN=... \
#     slurm/asep_benchmark.sbatch

#SBATCH --job-name=asep-bench
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# gpu_4090 forbids --mem/--mem-per-cpu/--mem-per-gpu.
#SBATCH --time=08:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

SOURCE_ROOT=${ASEP_SOURCE_ROOT:?ASEP_SOURCE_ROOT is required}
DATA_ROOT=${ASEP_DATA_ROOT:?ASEP_DATA_ROOT is required}
SPLIT_MANIFEST=${ASEP_SPLIT_MANIFEST:?ASEP_SPLIT_MANIFEST is required}
OUTPUT_ROOT=${ASEP_OUTPUT_ROOT:?ASEP_OUTPUT_ROOT is required}
PYTHON_BIN=${ASEP_PYTHON_BIN:?ASEP_PYTHON_BIN is required}
MAX_EPOCHS=${ASEP_MAX_EPOCHS:-12}
EVAL_SPLIT=${ASEP_SPLIT:-validation}
ARM=${ASEP_ARM:-esm2}
SEED=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}

case "${ARM}" in
  esm2) EXTRA=() ;;
  esm2_esmif1)
    EXTRA=(
      --representation-manifest "${ASEP_MANIFEST:?esm2_esmif1 needs ASEP_MANIFEST}"
      --esm-if1-root "${ASEP_ESM_IF1_ROOT:?esm2_esmif1 needs ASEP_ESM_IF1_ROOT}"
    ) ;;
  *) echo "unknown ASEP_ARM ${ARM}" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${ARM}/seed_${SEED}"
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SOURCE_ROOT}"

echo "[provenance] split manifest sha256: $(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)"
echo "[provenance] arm=${ARM} seed=${SEED} evaluate_on=${EVAL_SPLIT}"

"${PYTHON_BIN}" -u scripts/sabdab2_interaction_first.py train \
    --prepare-dir "${DATA_ROOT}" \
    --contact-dir "${DATA_ROOT}/contacts" \
    --embedding-dir "${DATA_ROOT}/embeddings" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${RUN_ROOT}/train" \
    --seed "${SEED}" --max-epochs "${MAX_EPOCHS}" --device cuda "${EXTRA[@]}"

"${PYTHON_BIN}" -u scripts/sabdab2_interaction_first.py evaluate \
    --prepare-dir "${DATA_ROOT}" \
    --contact-dir "${DATA_ROOT}/contacts" \
    --embedding-dir "${DATA_ROOT}/embeddings" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --checkpoint "${RUN_ROOT}/train/best.pt" \
    --output-dir "${RUN_ROOT}/${EVAL_SPLIT}" \
    --split "${EVAL_SPLIT}" --device cuda "${EXTRA[@]}"

echo "[done] ${ARM} seed ${SEED} on ${EVAL_SPLIT}"
