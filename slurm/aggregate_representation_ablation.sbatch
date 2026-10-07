#!/bin/bash
# Aggregate the paired representation arms into one verdict.
#
# Twenty evaluations of roughly 9.6M candidate pairs each, single-threaded, so
# this takes over an hour. It is submitted rather than run interactively for the
# same reason as the margin aggregation: the work should outlive the session
# that starts it.
#
#   sbatch --export=ALL,\
#ABLATION_SOURCE_ROOT=...,ABLATION_OUTPUT_ROOT=...,ABLATION_PYTHON_BIN=... \
#     slurm/aggregate_representation_ablation.sbatch

#SBATCH --job-name=repr-ablation-aggregate
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# gpu_4090 forbids --mem/--mem-per-cpu/--mem-per-gpu: the site's job_submit_lua
# plugin allocates memory per card and rejects any explicit request.
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SOURCE_ROOT=${ABLATION_SOURCE_ROOT:?ABLATION_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT:?ABLATION_OUTPUT_ROOT is required}
PYTHON_BIN=${ABLATION_PYTHON_BIN:?ABLATION_PYTHON_BIN is required}
REPORT=${ABLATION_REPORT:-${OUTPUT_ROOT}/representation_ablation_v1.json}

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
cd "${SOURCE_ROOT}"

"${PYTHON_BIN}" -u scripts/aggregate_representation_ablation.py \
    --run-root "${OUTPUT_ROOT}" \
    --json "${REPORT}"
