#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-predgeom-dev
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/training-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/training-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly authority=${campaign}/authority
readonly data_root=${root}/data/framecontact_20260730
readonly cache_root=${campaign}/data/structure_cache_v1
readonly split=${cache_root}/DEVELOPMENT_SPLIT_VIEW.csv
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly train_panel=${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json
readonly validation_panel=${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=baseline; readonly seed=17 ;;
  1) readonly arm=baseline; readonly seed=29 ;;
  2) readonly arm=baseline; readonly seed=43 ;;
  3) readonly arm=edge_topk2_supervised; readonly seed=17 ;;
  4) readonly arm=edge_topk2_supervised; readonly seed=29 ;;
  5) readonly arm=edge_topk2_supervised; readonly seed=43 ;;
  *) exit 64 ;;
esac

if test "${arm}" = baseline; then
  readonly source_root=${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_source_v3
  readonly source_manifest=${authority}/SOURCE_MANIFEST_BASELINE_V3.json
  readonly source_sha256=ca31451b4be85ee1e07045609a41ed3c90789971368191383c28cbf4251b581e
  readonly protocol=${authority}/PROTOCOL_PREDICTED_GEOMETRY_BASELINE_V1.json
  readonly protocol_receipt=${authority}/PROTOCOL_PREDICTED_GEOMETRY_BASELINE_V1_RECEIPT.json
  readonly trainer_sha256=0e183bfc2f5f572a8f42fa10287c84b43ab1438f0f264c8af2ecb2150a098925
  extra_args=()
else
  readonly source_root=${root}/immutable_stage_a1_candidates/edge_topk2_training_source_v1
  readonly source_manifest=${authority}/SOURCE_MANIFEST_EDGE_TOPK2_V1.json
  readonly source_sha256=f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55
  readonly protocol=${authority}/PROTOCOL_PREDICTED_GEOMETRY_TOPK2_V1.json
  readonly protocol_receipt=${authority}/PROTOCOL_PREDICTED_GEOMETRY_TOPK2_V1_RECEIPT.json
  readonly trainer_sha256=ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e
  extra_args=(--matched-local-loss-weight 0.0 --matched-local-margin 0.05)
fi

readonly protocol_sha256=$(sha256sum "${protocol}" | awk '{print $1}')
readonly protocol_receipt_sha256=$(sha256sum "${protocol_receipt}" | awk '{print $1}')
readonly structure_sha256=$(sha256sum "${cache_root}/STRUCTURE_CACHE_MANIFEST.json" | awk '{print $1}')
readonly split_sha256=$(sha256sum "${split}" | awk '{print $1}')
readonly output=${campaign}/runs/${arm}/seed-${seed}

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
printf '%s  %s\n' \
  "${trainer_sha256}" "${source_root}/scripts/train_framecontact.py" \
  "${source_sha256}" "${source_manifest}" \
  fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 "${train_panel}" \
  222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 "${validation_panel}" \
  | sha256sum --check --strict

"${python_bin}" - "${authority}/CAMPAIGN_PREDICTED_GEOMETRY_DEVELOPMENT_V1.json" \
  "${arm}" "${seed}" "${protocol_sha256}" "${protocol_receipt_sha256}" \
  "${source_sha256}" "${structure_sha256}" "${split_sha256}" <<'PY'
import json
import sys
path, arm, seed, protocol, receipt, source, structure, split = sys.argv[1:]
campaign = json.load(open(path, encoding="utf-8"))
assert campaign["status"] == "FROZEN_BEFORE_TRAINING"
assert int(seed) in campaign["seeds"]
assert campaign["arms"][arm] == {
    "protocol_sha256": protocol,
    "receipt_sha256": receipt,
    "source_manifest_sha256": source,
}
assert campaign["structure_manifest_sha256"] == structure
assert campaign["development_split_view_sha256"] == split
assert campaign["test_loaded"] is False
assert campaign["temporal_holdout_loaded"] is False
assert campaign["selection_or_tuning"] is False
PY

mkdir -p "$(dirname "${output}")"
cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "${cache_root}" \
  --structure-manifest-sha256 "${structure_sha256}" \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${split}" --split-manifest-sha256 "${split_sha256}" \
  --protocol "${protocol}" --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 "${protocol_receipt_sha256}" \
  --source-manifest "${source_manifest}" --source-manifest-sha256 "${source_sha256}" \
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
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16 \
  "${extra_args[@]}"
