#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
exec /ssd/scxi253/repr_extract/venv/bin/python3 /ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1/source_b72e6bf/extensions/execution_recovery_v1/run_scheduled_analysis.py --authority /ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1/layer_analysis_v1/AUTHORITY.json --authority-sha256 146222211e5a88d789ba61174103db3a88e34e6ef944978295d94a149d1ed505
