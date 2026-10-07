#!/usr/bin/env bash
#SBATCH --job-name=abf-late-scale-dev
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/training-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/training-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_late_stage_scaling_v1
readonly upstream=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly structure_root=${upstream}/data/structure_cache_v1
readonly contact_root=${upstream}/data/contact_cache_predicted_geometry_authority_v1
readonly split=${structure_root}/DEVELOPMENT_SPLIT_VIEW.csv
readonly data_root=${root}/data/framecontact_20260730
readonly train_panel=${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json
readonly validation_panel=${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json
readonly python_bin=${root}/envs/contact/bin/python3.10

readonly checkpoint_index=$((SLURM_ARRAY_TASK_ID / 6))
readonly run_index=$((SLURM_ARRAY_TASK_ID % 6))
case "${checkpoint_index}" in
  0) readonly label=segment_000058 ;;
  1) readonly label=segment_000063 ;;
  *) exit 64 ;;
esac
case "${run_index}" in
  0) readonly arm=baseline; readonly seed=17 ;;
  1) readonly arm=baseline; readonly seed=29 ;;
  2) readonly arm=baseline; readonly seed=43 ;;
  3) readonly arm=edge_topk2_supervised; readonly seed=17 ;;
  4) readonly arm=edge_topk2_supervised; readonly seed=29 ;;
  5) readonly arm=edge_topk2_supervised; readonly seed=43 ;;
  *) exit 64 ;;
esac

readonly checkpoint_root=${campaign}/checkpoints/${label}
readonly execution_root=${checkpoint_root}/authority/development_execution_v1
readonly execution=${execution_root}/CHECKPOINT_DEVELOPMENT_EXECUTION_V1.json
readonly execution_receipt=${execution_root}/CHECKPOINT_DEVELOPMENT_EXECUTION_V1_RECEIPT.json
readonly embedding_root=${checkpoint_root}/embeddings/sabdab2_abf_antibody_esm2_antigen_v1
readonly embedding_receipt=${embedding_root}/SABDAB2_ABF_CHECKPOINT_MIXED_EMBEDDING_RECEIPT.json

if test "${arm}" = baseline; then
  readonly source_root=${root}/immutable_stage_a1_candidates/abf30b_embedding_baseline_source_v1
  readonly source_manifest=${execution_root}/SOURCE_MANIFEST_BASELINE_SCALING_V1.json
  readonly protocol=${execution_root}/PROTOCOL_BASELINE_SCALING_V1.json
  readonly protocol_receipt=${execution_root}/PROTOCOL_BASELINE_SCALING_V1_RECEIPT.json
  readonly trainer_sha256=41ed07dd5fc058cefc74b63e004feb068dbd93b3e90fdf55a95010975c92d3f3
  extra_args=()
else
  readonly source_root=${root}/immutable_stage_a1_candidates/abf30b_embedding_topk2_source_v1
  readonly source_manifest=${execution_root}/SOURCE_MANIFEST_EDGE_TOPK2_SUPERVISED_SCALING_V1.json
  readonly protocol=${execution_root}/PROTOCOL_EDGE_TOPK2_SUPERVISED_SCALING_V1.json
  readonly protocol_receipt=${execution_root}/PROTOCOL_EDGE_TOPK2_SUPERVISED_SCALING_V1_RECEIPT.json
  readonly trainer_sha256=04f23fb8a28cb876b7c83be1f3ce4731a01421dd96427711b4f68c0553417e18
  extra_args=(--matched-local-loss-weight 0.0 --matched-local-margin 0.05)
fi

read -r execution_sha256 protocol_sha256 protocol_receipt_sha256 source_sha256 embedding_sha256 < <(
  "${python_bin}" - "${execution}" "${execution_receipt}" "${arm}" "${seed}" "${embedding_receipt}" "${label}" <<'PY'
import hashlib,json,sys
execution_path, receipt_path, arm, seed, embedding_receipt_path, label = sys.argv[1:]
raw=open(execution_path,'rb').read(); execution=json.loads(raw)
receipt=json.load(open(receipt_path)); embeddings=json.load(open(embedding_receipt_path))
digest=hashlib.sha256(raw).hexdigest()
assert execution['status']=='FROZEN_BEFORE_SIX_RUN_LATE_STAGE_SCALING_DEVELOPMENT'
assert receipt['status']==execution['status']
assert receipt['campaign_sha256']==digest
assert execution['checkpoint_label']==label
assert embeddings['checkpoint_label']==label
assert embeddings['checkpoint_model_sha256']==execution['checkpoint_model_sha256']
assert int(seed) in execution['seeds']
assert execution['distance_loss_weight']==0.0
assert execution['distance_label_cache_supplied'] is False
assert execution['test_loaded'] is False
assert execution['temporal_holdout_loaded'] is False
assert execution['embedding_manifest_sha256']==embeddings['manifest_sha256']
a=execution['arms'][arm]
print(digest,a['protocol_sha256'],a['protocol_receipt_sha256'],a['source_manifest_sha256'],embeddings['manifest_sha256'])
PY
)

readonly structure_sha256=$(sha256sum "${structure_root}/STRUCTURE_CACHE_MANIFEST.json" | awk '{print $1}')
readonly contact_sha256=$(sha256sum "${contact_root}/CONTACT_CACHE_MANIFEST.json" | awk '{print $1}')
readonly split_sha256=$(sha256sum "${split}" | awk '{print $1}')
readonly output=${checkpoint_root}/runs_development_v1/${arm}/seed-${seed}

test "${SLURM_JOB_PARTITION}" = gpu_4090
test -n "${CUDA_VISIBLE_DEVICES:-}"
test ! -e "${output}"
printf '%s  %s\n' \
  "${execution_sha256}" "${execution}" \
  "${protocol_sha256}" "${protocol}" \
  "${protocol_receipt_sha256}" "${protocol_receipt}" \
  "${source_sha256}" "${source_manifest}" \
  "${embedding_sha256}" "${embedding_root}/SABDAB2_ESM_EMBEDDING_MANIFEST.json" \
  "${trainer_sha256}" "${source_root}/scripts/train_framecontact.py" \
  d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88 "${contact_root}/CONTACT_CACHE_MANIFEST.json" \
  426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9 "${contact_root}/CONTACT_CACHE_RECEIPT.json" \
  bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614 "${structure_root}/STRUCTURE_CACHE_MANIFEST.json" \
  ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700 "${split}" \
  fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 "${train_panel}" \
  222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 "${validation_panel}" \
  | sha256sum --check --strict

mkdir -p "$(dirname "${output}")"
cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "${structure_root}" \
  --structure-manifest-sha256 "${structure_sha256}" \
  --contact-cache-root "${contact_root}" \
  --contact-manifest-sha256 "${contact_sha256}" \
  --embedding-root "${embedding_root}" \
  --embedding-manifest-sha256 "${embedding_sha256}" \
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
  --distance-loss-weight 0.0 --contrastive-loss-weight 0.0 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16 \
  "${extra_args[@]}"
