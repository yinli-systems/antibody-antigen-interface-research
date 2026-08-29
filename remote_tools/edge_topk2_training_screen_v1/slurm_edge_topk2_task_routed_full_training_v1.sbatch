#!/usr/bin/env bash
#SBATCH --job-name=edge-topk2-routed-full-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_topk2_task_routed_full_v1/logs/training-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_topk2_task_routed_full_v1/logs/training-%A_%a.err

set -euo pipefail
: "${EXPECTED_PROTOCOL_SHA256:?submit with the frozen protocol SHA-256}"
: "${EXPECTED_RECEIPT_SHA256:?submit with the frozen receipt SHA-256}"

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/edge_topk2_training_source_v1
readonly source_manifest=${source_root}/SOURCE_MANIFEST_EDGE_TOPK2_V1.json
readonly data_root=${root}/data/framecontact_20260730
readonly campaign=${root}/results/development/sabdab2_edge_topk2_task_routed_full_v1
readonly authority=${campaign}/authority
readonly protocol=${authority}/PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1.json
readonly receipt=${authority}/PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1_RECEIPT.json
readonly full_split=${data_root}/split_matrix_v3_1/antigen_fold.csv
readonly train_panel=${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json
readonly validation_panel=${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json
readonly python_bin=${root}/envs/contact/bin/python3.10
case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly seed=17 ;;
  1) readonly seed=29 ;;
  2) readonly seed=43 ;;
  *) exit 64 ;;
esac
readonly output=${campaign}/runs/edge_topk2_supervised/seed-${seed}

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e "${source_root}/scripts/train_framecontact.py" \
  121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe "${source_root}/src/interaction_first/framecontact.py" \
  f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55 "${source_manifest}" \
  fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 "${full_split}" \
  fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 "${train_panel}" \
  222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 "${validation_panel}" \
  "${EXPECTED_PROTOCOL_SHA256}" "${protocol}" \
  "${EXPECTED_RECEIPT_SHA256}" "${receipt}" \
  | sha256sum --check --strict
if [[ -e "${output}" || -L "${output}" ]]; then
  echo "immutable output exists: ${output}" >&2
  exit 73
fi
mkdir -p "$(dirname "${output}")"

"${python_bin}" - "${protocol}" "${seed}" <<'PY'
import json
import sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
seed = int(sys.argv[2])
assert protocol["protocol_status"] == "frozen_before_full_development_training"
assert protocol["arm"] == "edge_topk2_supervised"
assert seed in protocol["model_selection"]["seeds"]
assert protocol["task_routing"]["routing_is_fixed_per_task"] is True
assert protocol["task_routing"]["mixture_weights_tuned"] is False
assert protocol["claim_boundary"]["test_loaded"] is False
assert protocol["claim_boundary"]["temporal_holdout_loaded"] is False
PY

cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${full_split}" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol "${protocol}" --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${EXPECTED_RECEIPT_SHA256}" \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55 \
  --training-partner-panel "${train_panel}" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output}" --seed "${seed}" \
  --experiment epiformer_style_same_protocol \
  --epochs 20 --early-stopping-patience 5 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16
