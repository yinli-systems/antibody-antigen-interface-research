#!/bin/bash
#SBATCH --job-name=oas-dl-audit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/oas-download-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/oas-download-audit-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${OAS_AUDIT_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${OAS_AUDIT_SOURCE_ROOT:?OAS_AUDIT_SOURCE_ROOT is required}
DOWNLOAD_ROOT=${OAS_AUDIT_DOWNLOAD_ROOT:?OAS_AUDIT_DOWNLOAD_ROOT is required}
PLAN=${OAS_AUDIT_PLAN:?OAS_AUDIT_PLAN is required}
PLAN_FILE_SHA256=${OAS_AUDIT_PLAN_FILE_SHA256:?OAS_AUDIT_PLAN_FILE_SHA256 is required}
PLAN_PAYLOAD_SHA256=${OAS_AUDIT_PLAN_PAYLOAD_SHA256:?OAS_AUDIT_PLAN_PAYLOAD_SHA256 is required}
AUDITOR_SOURCE_SHA256=${OAS_AUDIT_CONTROLLER_SHA256:?OAS_AUDIT_CONTROLLER_SHA256 is required}
OUTPUT_ROOT=${OAS_AUDIT_OUTPUT_ROOT:?OAS_AUDIT_OUTPUT_ROOT is required}
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/audit_oas_download_completion.py" \
  --download-root "$DOWNLOAD_ROOT" \
  --plan "$PLAN" \
  --plan-file-sha256 "$PLAN_FILE_SHA256" \
  --plan-payload-sha256 "$PLAN_PAYLOAD_SHA256" \
  --summary "$DOWNLOAD_ROOT/summaries/task_0.json" \
  --summary "$DOWNLOAD_ROOT/summaries/task_1.json" \
  --summary "$DOWNLOAD_ROOT/summaries/task_2.json" \
  --summary "$DOWNLOAD_ROOT/summaries/task_3.json" \
  --task-count 4 \
  --auditor-source-sha256 "$AUDITOR_SOURCE_SHA256" \
  --output "$OUTPUT_ROOT/receipt.json"
