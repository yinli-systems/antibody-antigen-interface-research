#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-plddt70-prep
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-confidence-geometry-plddt70-v1-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-confidence-geometry-plddt70-v1-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools=${root}/remote_tools/sabdab2_confidence_geometry_v1
readonly source_cache=${campaign}/data/structure_cache_v1
readonly output_cache=${campaign}/data/structure_cache_plddt70_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${campaign}/authority/confidence_geometry_plddt70_v1"
test ! -e "${output_cache}"

"${python_bin}" "${tools}/freeze_confidence_geometry_plan_v1.py" \
  --campaign-root "${campaign}" \
  --source-cache-root "${source_cache}" \
  --materializer "${tools}/materialize_confidence_masked_structure_cache_v1.py" \
  --finalizer "${tools}/finalize_confidence_geometry_protocol_v1.py" \
  --training-launcher "${tools}/slurm_confidence_geometry_development_v1.sbatch" \
  --analyzer "${tools}/analyze_confidence_geometry_development_v1.py"

"${python_bin}" "${tools}/materialize_confidence_masked_structure_cache_v1.py" \
  --source-cache-root "${source_cache}" \
  --threshold 70 \
  --canary-records 3

"${python_bin}" "${tools}/materialize_confidence_masked_structure_cache_v1.py" \
  --source-cache-root "${source_cache}" \
  --output-root "${output_cache}" \
  --threshold 70

"${python_bin}" "${tools}/finalize_confidence_geometry_protocol_v1.py" \
  --campaign-root "${campaign}" \
  --masked-cache-root "${output_cache}"
