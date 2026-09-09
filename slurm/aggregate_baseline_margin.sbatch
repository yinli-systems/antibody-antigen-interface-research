#!/bin/bash
# Aggregate the seed replicates of the baseline margin into one verdict.
#
# This is CPU-only work, but it does not belong on the login node: the
# aggregation reads ten evaluations of ~9.6M candidate pairs each and runs
# single-threaded for tens of minutes, and two attempts to run it interactively
# were killed without a traceback and without writing their output. Submitting
# it gives the work a real allocation and a log that survives a dropped session.
#
#   sbatch --export=ALL,\
#MARGIN_SOURCE_ROOT=...,MARGIN_OUTPUT_ROOT=...,MARGIN_PYTHON_BIN=... \
#     slurm/aggregate_baseline_margin.sbatch

#SBATCH --job-name=margin-aggregate
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# gpu_4090 forbids --mem/--mem-per-cpu/--mem-per-gpu: the site's job_submit_lua
# plugin allocates memory per card and rejects any explicit request.
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SOURCE_ROOT=${MARGIN_SOURCE_ROOT:?MARGIN_SOURCE_ROOT is required}
OUTPUT_ROOT=${MARGIN_OUTPUT_ROOT:?MARGIN_OUTPUT_ROOT is required}
PYTHON_BIN=${MARGIN_PYTHON_BIN:?MARGIN_PYTHON_BIN is required}
REPORT=${MARGIN_REPORT:-${OUTPUT_ROOT}/baseline_margin_replication_v1.json}

export PYTHONPATH="${SOURCE_ROOT}/src:${PYTHONPATH:-}"
cd "${SOURCE_ROOT}"

"${PYTHON_BIN}" -u scripts/aggregate_baseline_margin.py \
    --run-root "${OUTPUT_ROOT}" \
    --json "${REPORT}"
