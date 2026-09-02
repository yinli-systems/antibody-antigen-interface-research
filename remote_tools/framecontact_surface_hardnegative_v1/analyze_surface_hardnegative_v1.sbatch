#!/usr/bin/env bash
#SBATCH --job-name=fc-surface-hard-analyze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_surface_hardnegative_v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_surface_hardnegative_v1/logs/analyze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_surface_hardnegative_v1
readonly reference=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools_root=${root}/tools/framecontact_surface_hardnegative_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

"${python_bin}" "${tools_root}/analyze_surface_hardnegative_v1.py" \
  --campaign-root "${campaign}" --reference-root "${reference}" \
  --base-analyzer "${root}/tools/sabdab2_predicted_geometry_v1/analyze_predicted_geometry_development_v1.py" \
  --output "${campaign}/FRAMECONTACT_SURFACE_HARDNEGATIVE_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/FRAMECONTACT_SURFACE_HARDNEGATIVE_ANALYSIS_V1_RECEIPT.json"
