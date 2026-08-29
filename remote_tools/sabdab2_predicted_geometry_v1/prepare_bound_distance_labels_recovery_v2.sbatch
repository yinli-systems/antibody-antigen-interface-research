#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-distance-labels-v2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-distance-label-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-distance-label-v2-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools_root=${root}/tools/sabdab2_predicted_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly bound_structure_root=${root}/data/framecontact_20260730/sabdab2/structure_cohort_v2
readonly model_input_structure_root=${campaign}/data/structure_cache_v1
readonly split=${model_input_structure_root}/DEVELOPMENT_SPLIT_VIEW.csv
readonly output_root=${campaign}/data/bound_distance_labels_v1
readonly staging=${campaign}/staging_distance_label_recovery_v2
readonly baseline_source=${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_distance_labels_source_v4
readonly topk_source=${root}/immutable_stage_a1_candidates/edge_topk2_distance_labels_source_v2
readonly materializer=${tools_root}/materialize_bound_distance_labels_v1.py
readonly freezer=${tools_root}/freeze_distance_label_recovery_v2.py
readonly training_launcher=${tools_root}/slurm_predicted_geometry_distance_label_recovery_v2.sbatch
readonly analyzer=${tools_root}/analyze_predicted_geometry_distance_label_recovery_v2.py

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output_root}"
test ! -e "${campaign}/authority/distance_label_recovery_v2"

printf '%s  %s\n' \
  edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 "${bound_structure_root}/STRUCTURE_CACHE_MANIFEST.json" \
  bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614 "${model_input_structure_root}/STRUCTURE_CACHE_MANIFEST.json" \
  ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700 "${split}" \
  a320a5f60794ea386d355e8ab5ec895130d3b714596bc817e303eee589dbc7e8 "${staging}/SOURCE_MANIFEST_BASELINE_DISTANCE_LABELS_V4.json" \
  885449525b7f41b3b00511245ca6c0b0859336db959becc7680ad9cb6e872332 "${staging}/SOURCE_MANIFEST_EDGE_TOPK2_DISTANCE_LABELS_V2.json" \
  2e181e95d80268ee79c734c6377e7f75aa56c987eb56e100bb34b39672278728 "${campaign}/FAIL_CLOSED_INVALID_INDEPENDENT_COORDINATE_DISTANCE_SUPERVISION_1483894.json" \
  c45f01585f7eaa6d5e1220b7221aef4a2bd703ba7d7b3709ca3cd5487c11a1bc "${campaign}/FAIL_CLOSED_INVALID_INDEPENDENT_COORDINATE_DISTANCE_SUPERVISION_1483894_RECEIPT.json" \
  | sha256sum --check --strict

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${materializer}" \
  --bound-structure-root "${bound_structure_root}" \
  --bound-structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --model-input-structure-root "${model_input_structure_root}" \
  --model-input-structure-manifest-sha256 bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614 \
  --development-split "${split}" \
  --development-split-sha256 ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700 \
  --output-root "${output_root}"

"${python_bin}" "${freezer}" \
  --campaign-root "${campaign}" \
  --baseline-source-root "${baseline_source}" \
  --topk-source-root "${topk_source}" \
  --baseline-source-manifest "${staging}/SOURCE_MANIFEST_BASELINE_DISTANCE_LABELS_V4.json" \
  --topk-source-manifest "${staging}/SOURCE_MANIFEST_EDGE_TOPK2_DISTANCE_LABELS_V2.json" \
  --materializer "${materializer}" \
  --training-launcher "${training_launcher}" \
  --analyzer "${analyzer}"

sha256sum --check --strict "${campaign}/authority/distance_label_recovery_v2/AUTHORITY_SHA256SUMS.txt"
