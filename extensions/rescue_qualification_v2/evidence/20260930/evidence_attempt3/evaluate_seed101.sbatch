#!/bin/bash
set -euo pipefail
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /ssd/scxi253/repr_extract/venv/bin/python3 /ssd/scxi253/frontier_multimodel_v1/frontier_performance_v1/rescue_qualification_v2_20260930/source_attempt3/extensions/rescue_qualification_v2/qualified_evaluate.py run --plan /ssd/scxi253/frontier_multimodel_v1/frontier_performance_v1/rescue_qualification_v2_20260930/evidence_attempt3/EVALUATION_PLAN.json --plan-sha256 9b84a97c3f68dbc0f6ba167b2ee573ee8a6e6f5c7663794b74b9af1a42ba907a --index 0 --output /ssd/scxi253/frontier_multimodel_v1/frontier_performance_v1/rescue_qualification_v2_20260930/evidence_attempt3/evaluation_runs
