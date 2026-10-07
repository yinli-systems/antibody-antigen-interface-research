#!/usr/bin/env bash
#SBATCH --job-name=abf30b-dev-freeze
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/dev-freeze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/dev-freeze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${root}/tools/abf30b_framecontact_v1/freeze_abf30b_development_execution_v1.py" \
  --campaign-root "${campaign}" \
  --upstream-root "${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1" \
  --baseline-source-root "${root}/immutable_stage_a1_candidates/abf30b_embedding_baseline_source_v1" \
  --topk2-source-root "${root}/immutable_stage_a1_candidates/abf30b_embedding_topk2_source_v1" \
  --original-source-root "${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_distance_labels_source_v4" \
  --embedding-root "${campaign}/embeddings/sabdab2_abf30b_antibody_esm2_antigen_v1" \
  --output-root "${campaign}/authority/development_execution_v1"
