#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-percomplex
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/export-per-complex-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/export-per-complex-v1-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly plan=${tools}/GEOMETRY_TASK_ASSOCIATION_PLAN_V1.json
readonly local_task=$((SLURM_ARRAY_TASK_ID % 6))
if (( SLURM_ARRAY_TASK_ID < 6 )); then
  readonly geometry_source=predicted
  readonly structure_root=${campaign}/data/structure_cache_v1
  readonly contact_root=${campaign}/data/contact_cache_predicted_geometry_authority_v1
  readonly split=${structure_root}/DEVELOPMENT_SPLIT_VIEW.csv
  readonly split_sha256=ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700
else
  readonly geometry_source=bound
  readonly structure_root=${root}/data/framecontact_20260730/sabdab2/structure_cohort_v2
  readonly contact_root=${root}/data/framecontact_20260730/sabdab2/contact_cohort_v2
  readonly split=${root}/data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv
  readonly split_sha256=fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6
fi

case "${local_task}" in
  0) readonly arm=baseline; readonly seed=17 ;;
  1) readonly arm=baseline; readonly seed=29 ;;
  2) readonly arm=baseline; readonly seed=43 ;;
  3) readonly arm=edge_topk2_supervised; readonly seed=17 ;;
  4) readonly arm=edge_topk2_supervised; readonly seed=29 ;;
  5) readonly arm=edge_topk2_supervised; readonly seed=43 ;;
  *) exit 64 ;;
esac

if test "${arm}" = baseline; then
  readonly source_root=${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_distance_labels_source_v4
else
  readonly source_root=${root}/immutable_stage_a1_candidates/edge_topk2_distance_labels_source_v2
fi
if test "${geometry_source}" = predicted; then
  readonly run_root=${campaign}/runs_distance_label_recovery_v2/${arm}/seed-${seed}
elif test "${arm}" = baseline; then
  readonly run_root=${root}/results/development/abpairfm_joint_energy_balanced_full_v1/baseline/seed-${seed}
else
  readonly run_root=${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1/runs/edge_topk2_supervised/seed-${seed}
fi
readonly output_root=${campaign}/quality_task_metrics_v1/${geometry_source}/${arm}/seed-${seed}

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output_root}"
printf '%s  %s\n' \
  86ef7dff66b4f636f929f28728cf300f360041577741c891059ff601f9c5f272 "${plan}" \
  354697863c9428c954c85f7e74bb95fbf9d3bc72cc6100d31b801ac89df57e26 "${tools}/analyze_predicted_geometry_quality_mechanism_v1.py" \
  | sha256sum --check --strict

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tools}/export_predicted_geometry_per_complex_metrics_v1.py" \
  --repository-root "${source_root}" \
  --run-root "${run_root}" \
  --geometry-source "${geometry_source}" \
  --arm "${arm}" \
  --structure-cache-root "${structure_root}" \
  --contact-cache-root "${contact_root}" \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --split-manifest "${split}" \
  --split-manifest-sha256 "${split_sha256}" \
  --plan "${plan}" \
  --output-root "${output_root}" \
  --workers 4
