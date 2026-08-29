#!/usr/bin/env bash
#SBATCH --job-name=edge-topk2-routed-fresh156-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_topk2_task_routed_holdout_v1/logs/measurement-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_topk2_task_routed_holdout_v1/logs/measurement-%A_%a.err

set -euo pipefail
: "${EXPECTED_PROTOCOL_SHA256:?submit with the frozen protocol SHA-256}"

root=/ssd/scxi253/model_deployment
campaign_root=${root}/results/development/sabdab2_edge_topk2_task_routed_holdout_v1
python_bin=${root}/envs/contact/bin/python3.10
data_root=${root}/data/framecontact_20260730
case "${SLURM_ARRAY_TASK_ID}" in
  0) arm=baseline_current; seed=17 ;;
  1) arm=baseline_current; seed=29 ;;
  2) arm=baseline_current; seed=43 ;;
  3) arm=edge_topk2_supervised; seed=17 ;;
  4) arm=edge_topk2_supervised; seed=29 ;;
  5) arm=edge_topk2_supervised; seed=43 ;;
  *) exit 64 ;;
esac
case "${arm}" in
  baseline_current)
    source_root=${root}/immutable_stage_a1_candidates/matched_local_training_source_v3
    source_manifest=${root}/results/development/sabdab2_matched_local_screen_v3/authority/SOURCE_MANIFEST_MATCHED_LOCAL_V3.json
    ;;
  edge_topk2_supervised)
    source_root=${root}/immutable_stage_a1_candidates/edge_topk2_training_source_v1
    source_manifest=${source_root}/SOURCE_MANIFEST_EDGE_TOPK2_V1.json
    ;;
esac
protocol=${campaign_root}/authority/PROTOCOL_TASK_ROUTED_FRESH156_V1.json
observed_protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
if [[ "${observed_protocol_sha256}" != "${EXPECTED_PROTOCOL_SHA256}" ]]; then
  echo "task-routing protocol SHA-256 differs" >&2
  exit 65
fi
output_dir=${campaign_root}/component-results/${arm}/seed-${seed}
output=${output_dir}/FRESH156_RESULT.json
receipt=${output_dir}/FRESH156_RECEIPT.json
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "immutable fresh156 output exists: ${output_dir}" >&2
  exit 73
fi
mkdir -p "${output_dir}"

cd "${source_root}"
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"${python_bin}" ${root}/tools/evaluate_edge_topk2_task_routed_holdout_v1.py \
  --repository-root "${source_root}" \
  --source-manifest "${source_manifest}" \
  --campaign-root "${campaign_root}" \
  --protocol-sha256 "${EXPECTED_PROTOCOL_SHA256}" \
  --checkpoint-binding ${root}/results/development/sabdab2_edge_topk2_three_seed_confirmation_v1/authority/CHECKPOINT_BINDING_EDGE_TOPK2_CONFIRMATION_V1.json \
  --arm "${arm}" --seed "${seed}" \
  --structure-cache-root ${data_root}/sabdab2/structure_cohort_v2 \
  --contact-cache-root ${data_root}/sabdab2/contact_cohort_v2 \
  --embedding-root ${root}/data/sppider_seq_sabdab2_embeddings \
  --output "${output}" --receipt "${receipt}" --workers 4
