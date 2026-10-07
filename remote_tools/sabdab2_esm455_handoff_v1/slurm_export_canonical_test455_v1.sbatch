#!/usr/bin/env bash
#SBATCH --job-name=esm2-test455-export
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/export-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/logs/export-%j.err

set -euo pipefail
umask 027

readonly root=/ssd/scxi253/model_deployment
readonly data_root=${root}/data/framecontact_20260730
readonly tool_root=${root}/tools/sabdab2_esm455_handoff_v1
readonly output_root=${root}/handoffs/framecontact_case_study_20260904/sabdab2_esm2_frozen_test455_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

mkdir -p "${root}/handoffs/framecontact_case_study_20260904/logs"

exec "${python_bin}" "${tool_root}/export_sabdab2_esm_subset_v1.py" \
  --prepared "${data_root}/authorities/sabdab2_prepared_instances.parquet" \
  --prepared-sha256 a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335 \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --output-root "${output_root}" \
  --expected-instance-count 455 \
  --split-column frozen_split \
  --split-role test \
  --transfer-mode hardlink
