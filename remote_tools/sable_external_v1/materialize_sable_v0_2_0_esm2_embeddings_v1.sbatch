#!/usr/bin/env bash
#SBATCH --job-name=sable-esm2-embed
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/esm2-embedding-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/esm2-embedding-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly prediction_root=${campaign}/structure_prediction/af3_label_free_v1
readonly source_root=${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_distance_labels_source_v4
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/materialize_sable_v0_2_0_esm2_embeddings_v1.py
readonly script_sha256=51251b28f5d1aed961bc03984e6937059b3f8731f035229c5c718f7db63c2e91
readonly esm_module=${source_root}/src/interaction_first/esm_embeddings.py
readonly model_root=${root}/models/sppider_seq/esm2_t33_650M_UR50D
readonly dependency_site_packages=${root}/envs/epiformer/lib/python3.10/site-packages
readonly embedding_root=${campaign}/embeddings/esm2_t33_650m_label_free_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo "${script_sha256}  ${script}" | sha256sum --check --strict -
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TMPDIR=${campaign}/tmp/esm2-label-free-v1
mkdir -p "${TMPDIR}"

"${python_bin}" "${script}" \
  --input-plan "${prediction_root}/AF3_LABEL_FREE_INPUT_PLAN.json" \
  --input-plan-receipt "${prediction_root}/AF3_LABEL_FREE_INPUT_PLAN_RECEIPT.json" \
  --esm-module "${esm_module}" \
  --model-root "${model_root}" \
  --dependency-site-packages "${dependency_site_packages}" \
  --embedding-root "${embedding_root}" \
  --device cuda \
  --resume
