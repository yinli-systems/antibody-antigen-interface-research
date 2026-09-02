#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-percomplex-recovery
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/export-per-complex-recovery-v1-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/export-per-complex-recovery-v1-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly plan=${tools}/GEOMETRY_TASK_ASSOCIATION_PLAN_V1.json
readonly structure_root=${campaign}/data/structure_cache_v1
readonly contact_root=${campaign}/data/contact_cache_predicted_geometry_authority_v1
readonly split=${structure_root}/DEVELOPMENT_SPLIT_VIEW.csv
readonly run_root=${campaign}/runs_distance_label_recovery_v2/edge_topk2_supervised/seed-43
readonly output_root=${campaign}/quality_task_metrics_v1/predicted/edge_topk2_supervised/seed-43

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output_root}"
printf '%s  %s\n' \
  86ef7dff66b4f636f929f28728cf300f360041577741c891059ff601f9c5f272 "${plan}" \
  0dff6da931f1651c1bfe94a04e7890bd8f39b5b627d9d5de32a6627437298cc4 "${tools}/export_predicted_geometry_per_complex_metrics_recovery_v1.py" \
  354697863c9428c954c85f7e74bb95fbf9d3bc72cc6100d31b801ac89df57e26 "${tools}/analyze_predicted_geometry_quality_mechanism_v1.py" \
  | sha256sum --check --strict

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tools}/export_predicted_geometry_per_complex_metrics_recovery_v1.py" \
  --repository-root "${root}/immutable_stage_a1_candidates/edge_topk2_distance_labels_source_v2" \
  --run-root "${run_root}" \
  --geometry-source predicted \
  --arm edge_topk2_supervised \
  --structure-cache-root "${structure_root}" \
  --contact-cache-root "${contact_root}" \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --split-manifest "${split}" \
  --split-manifest-sha256 ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700 \
  --plan "${plan}" \
  --output-root "${output_root}" \
  --workers 4
