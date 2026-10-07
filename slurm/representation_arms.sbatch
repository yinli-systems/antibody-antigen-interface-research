#!/bin/bash
# Train one seed of one representation arm.
#
# The first ablation compared ESM-2 against ESM-2 + AntiBERTy + ESM-IF1 and put
# the second 26.7% ahead on macro AUPRC at 10/10 seeds. Two questions it could
# not answer follow, and each needs its own arm:
#
#   random_control   Does a wider input projection explain part of the gain?
#                    Same widths, same parameter count, columns that are a fixed
#                    random projection of the base embedding and so carry
#                    nothing new.
#   antiberty_only   Which encoder is doing the work? AntiBERTy covers the
#   esm_if1_only     antibody side alone, ESM-IF1 covers both.
#
# Arms differ in width, so their parameter counts differ; that is the property
# random_control exists to measure rather than to hide.
#
# Development-only. Trains on the authority train split and evaluates on
# validation. The frozen test split is never named and never read.
#
#   sbatch --array=1-10 --export=ALL,ARM=random_control,... \
#     slurm/representation_arms.sbatch

#SBATCH --job-name=repr-arm
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# gpu_4090 forbids --mem/--mem-per-cpu/--mem-per-gpu.
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
MANIFEST=${ABLATION_MANIFEST:?ABLATION_MANIFEST is required}
ANTIBERTY_ROOT=${ABLATION_ANTIBERTY_ROOT:?ABLATION_ANTIBERTY_ROOT is required}
ESM_IF1_ROOT=${ABLATION_ESM_IF1_ROOT:?ABLATION_ESM_IF1_ROOT is required}
MAX_EPOCHS=${ABLATION_MAX_EPOCHS:-12}
ARM=${ARM:?ARM is required}
SEED=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}

case "${ARM}" in
  esm2)           EXTRA=() ;;
  random_control) EXTRA=(--representation-manifest "${MANIFEST}" --random-projection-control) ;;
  antiberty_only) EXTRA=(--representation-manifest "${MANIFEST}" --antiberty-root "${ANTIBERTY_ROOT}") ;;
  esm_if1_only)   EXTRA=(--representation-manifest "${MANIFEST}" --esm-if1-root "${ESM_IF1_ROOT}") ;;
  esm_if1_shuffled)
                  # The attribution arm: same features, same complex, same
                  # length, same distribution, with the residue correspondence
                  # destroyed. If the gain survives this, it is not positional.
                  EXTRA=(--representation-manifest "${MANIFEST}" --esm-if1-root "${ESM_IF1_ROOT}"
                         --shuffle-representation-residues "${SHUFFLE_SEED:-20260911}") ;;
  esm2_antiberty_esmif1)
                  EXTRA=(--representation-manifest "${MANIFEST}" --antiberty-root "${ANTIBERTY_ROOT}" --esm-if1-root "${ESM_IF1_ROOT}") ;;
  *) echo "unknown ARM ${ARM}" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${ARM}/seed_${SEED}"
export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SOURCE_ROOT}"

echo "[provenance] split manifest sha256: $(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)"
echo "[provenance] arm=${ARM} seed=${SEED} max_epochs=${MAX_EPOCHS}"

"${PYTHON_BIN}" -u scripts/sabdab2_interaction_first.py train \
    --prepare-dir "${PREPARE_DIR}" --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${RUN_ROOT}/train" --seed "${SEED}" \
    --max-epochs "${MAX_EPOCHS}" --device cuda "${EXTRA[@]}"

"${PYTHON_BIN}" -u scripts/sabdab2_interaction_first.py evaluate \
    --prepare-dir "${PREPARE_DIR}" --contact-dir "${CONTACT_DIR}" \
    --embedding-dir "${EMBEDDING_DIR}" --split-manifest "${SPLIT_MANIFEST}" \
    --checkpoint "${RUN_ROOT}/train/best.pt" --output-dir "${RUN_ROOT}/validation" \
    --split validation --device cuda "${EXTRA[@]}"

echo "[done] ${ARM} seed ${SEED}"
