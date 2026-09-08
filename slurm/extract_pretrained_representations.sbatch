#!/bin/bash
# Extract frozen per-residue representations from external pretrained encoders.
#
# One array task per backend. Development-only feature extraction: no label,
# contact, prediction, or score is read, and no training or selection follows
# from it. Train split only unless REPRESENTATION_SPLITS is set deliberately.
#
#   sbatch --export=ALL,\
#REPRESENTATION_SOURCE_ROOT=...,REPRESENTATION_MANIFEST=...,\
#REPRESENTATION_OUTPUT_ROOT=...,REPRESENTATION_PYTHON_BIN=...,\
#REPRESENTATION_BACKBONES=... \
#     slurm/extract_pretrained_representations.sbatch

#SBATCH --job-name=repr-extract
#SBATCH --partition=gpu_4090
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

SOURCE_ROOT=${REPRESENTATION_SOURCE_ROOT:?REPRESENTATION_SOURCE_ROOT is required}
MANIFEST=${REPRESENTATION_MANIFEST:?REPRESENTATION_MANIFEST is required}
OUTPUT_ROOT=${REPRESENTATION_OUTPUT_ROOT:?REPRESENTATION_OUTPUT_ROOT is required}
PYTHON_BIN=${REPRESENTATION_PYTHON_BIN:?REPRESENTATION_PYTHON_BIN is required}
SPLITS=${REPRESENTATION_SPLITS:-train}

# Frozen by the split artifacts; a mismatch stops extraction rather than
# silently encoding a different manifest.
MANIFEST_SHA256=${REPRESENTATION_MANIFEST_SHA256:-548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63}

# AntiBERTy was trained on individual heavy and light chains, so it takes them
# separately. ESM-IF1 needs a backbone, and the store holds one per side, so it
# takes the concatenated antibody and the antigen.
BACKENDS=(antiberty esm2 esm_if1)
CHAINS_FOR=("vh vl" "vh vl ag" "ab ag")
BACKEND=${BACKENDS[${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}]}
CHAINS=${CHAINS_FOR[${SLURM_ARRAY_TASK_ID}]}

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
cd "${SOURCE_ROOT}"

BACKBONE_ARGS=()
if [ "${BACKEND}" = "esm_if1" ]; then
    BACKBONE_ARGS=(--backbones "${REPRESENTATION_BACKBONES:?esm_if1 needs REPRESENTATION_BACKBONES}")
fi

echo "[run] backend=${BACKEND} chains=${CHAINS} splits=${SPLITS}"
"${PYTHON_BIN}" scripts/extract_pretrained_representations.py \
    --manifest "${MANIFEST}" \
    --manifest-sha256 "${MANIFEST_SHA256}" \
    --backend "${BACKEND}" \
    --chains ${CHAINS} \
    --splits ${SPLITS} \
    --output-root "${OUTPUT_ROOT}/${BACKEND}" \
    --receipt "${OUTPUT_ROOT}/${BACKEND}_receipt.json" \
    --device cuda \
    "${BACKBONE_ARGS[@]}"

echo "[done] ${BACKEND}"
