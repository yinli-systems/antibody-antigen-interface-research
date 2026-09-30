#!/bin/bash
set -euo pipefail
exec /ssd/scxi253/repr_extract/venv/bin/python3 /ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1/source_0bd7366/extensions/execution_recovery_v1/af3.py run --plan /ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1/af3_attempt2/PLAN.json --plan-sha256 6db2ca822003647080561af7124795d855117a920fd634ce0392d8e995539df3 --task-id "${SLURM_ARRAY_TASK_ID:?}"
