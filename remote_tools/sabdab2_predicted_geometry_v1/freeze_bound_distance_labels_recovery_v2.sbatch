#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-freeze-distance-v2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/freeze-distance-label-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/freeze-distance-label-v2-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools_root=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly staging=${campaign}/staging_distance_label_recovery_v2
readonly distance_root=${campaign}/data/bound_distance_labels_v1
readonly failure=${campaign}/FAIL_CLOSED_FREEZER_SCHEMA_DIFFERENCE_1484895.json
readonly failure_receipt=${campaign}/FAIL_CLOSED_FREEZER_SCHEMA_DIFFERENCE_1484895_RECEIPT.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${campaign}/authority/distance_label_recovery_v2"

printf '%s  %s\n' \
  a8c13f0be4a5b6483897142e6dc03ddcb2741ef9d8300c28aeda0c87bdf9ccb3 "${distance_root}/DISTANCE_LABEL_CACHE_MANIFEST.json" \
  a767257fc4862e1b230cd06a2cd6f9a3a95e9ad6c74c58c120519ab16b733a89 "${distance_root}/DISTANCE_LABEL_CACHE_RECEIPT.json" \
  a320a5f60794ea386d355e8ab5ec895130d3b714596bc817e303eee589dbc7e8 "${staging}/SOURCE_MANIFEST_BASELINE_DISTANCE_LABELS_V4.json" \
  885449525b7f41b3b00511245ca6c0b0859336db959becc7680ad9cb6e872332 "${staging}/SOURCE_MANIFEST_EDGE_TOPK2_DISTANCE_LABELS_V2.json" \
  1a50babe958def5a14adbe469df6a3330a2ab01536537a5e4ebc5a1af732ab46 "${failure}" \
  8890843d4199cab4357aa5e91f39d3159cc48a0e80080a07bcd67320b1f677c1 "${failure_receipt}" \
  | sha256sum --check --strict

export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${tools_root}/freeze_distance_label_recovery_v2.py" \
  --campaign-root "${campaign}" \
  --baseline-source-root "${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_distance_labels_source_v4" \
  --topk-source-root "${root}/immutable_stage_a1_candidates/edge_topk2_distance_labels_source_v2" \
  --baseline-source-manifest "${staging}/SOURCE_MANIFEST_BASELINE_DISTANCE_LABELS_V4.json" \
  --topk-source-manifest "${staging}/SOURCE_MANIFEST_EDGE_TOPK2_DISTANCE_LABELS_V2.json" \
  --materializer "${tools_root}/materialize_bound_distance_labels_v1.py" \
  --training-launcher "${tools_root}/slurm_predicted_geometry_distance_label_recovery_v2.sbatch" \
  --analyzer "${tools_root}/analyze_predicted_geometry_distance_label_recovery_v2.py"

sha256sum --check --strict "${campaign}/authority/distance_label_recovery_v2/AUTHORITY_SHA256SUMS.txt"
