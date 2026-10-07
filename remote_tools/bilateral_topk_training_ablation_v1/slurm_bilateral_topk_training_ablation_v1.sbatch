#!/usr/bin/env bash
#SBATCH --job-name=bilateral-topk-train-v1
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-7%6
#SBATCH --no-requeue
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_training_ablation_v1/logs/train-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_training_ablation_v1/logs/train-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/multitask_topk2_training_source_v1
readonly source_manifest=${source_root}/SOURCE_MANIFEST_MULTITASK_TOPK2_TRAINING_V1.json
readonly source_sha=ddd43e25f29c22f71bb5816b4ac1d2739dad59f345c485b2fc7f51ce9655769e
readonly campaign=${root}/results/development/sabdab2_bilateral_topk_training_ablation_v1
readonly structure_root=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1/data/structure_cache_v1
readonly contact_root=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1/data/contact_cache_predicted_geometry_authority_v1
readonly split=${structure_root}/DEVELOPMENT_SPLIT_VIEW.csv
readonly data_root=${root}/data/framecontact_20260730
readonly train_panel=${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json
readonly validation_panel=${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json
readonly python_bin=${root}/envs/contact/bin/python3.10

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=epitope_k1; readonly epitope_k=1; readonly paratope_pooling=logmeanexp; readonly paratope_k=2 ;;
  1) readonly arm=epitope_k2; readonly epitope_k=2; readonly paratope_pooling=logmeanexp; readonly paratope_k=2 ;;
  2) readonly arm=epitope_k4; readonly epitope_k=4; readonly paratope_pooling=logmeanexp; readonly paratope_k=2 ;;
  3) readonly arm=epitope_k8; readonly epitope_k=8; readonly paratope_pooling=logmeanexp; readonly paratope_k=2 ;;
  4) readonly arm=paratope_k1; readonly epitope_k=2; readonly paratope_pooling=topk_mean; readonly paratope_k=1 ;;
  5) readonly arm=paratope_k2; readonly epitope_k=2; readonly paratope_pooling=topk_mean; readonly paratope_k=2 ;;
  6) readonly arm=paratope_k4; readonly epitope_k=2; readonly paratope_pooling=topk_mean; readonly paratope_k=4 ;;
  7) readonly arm=paratope_k8; readonly epitope_k=2; readonly paratope_pooling=topk_mean; readonly paratope_k=8 ;;
  *) exit 64 ;;
esac

readonly protocol=${campaign}/authority/PROTOCOL_${arm^^}_V1.json
readonly receipt=${campaign}/authority/PROTOCOL_${arm^^}_V1_RECEIPT.json
readonly output=${campaign}/runs/${arm}/seed-17
test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
test "$(sha256sum "${source_manifest}" | cut -d' ' -f1)" = "${source_sha}"
readonly receipt_sha=$(sha256sum "${receipt}" | cut -d' ' -f1)
readonly structure_sha=$(sha256sum "${structure_root}/STRUCTURE_CACHE_MANIFEST.json" | cut -d' ' -f1)
readonly contact_sha=$(sha256sum "${contact_root}/CONTACT_CACHE_MANIFEST.json" | cut -d' ' -f1)
readonly split_sha=$(sha256sum "${split}" | cut -d' ' -f1)

cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "${structure_root}" --structure-manifest-sha256 "${structure_sha}" \
  --contact-cache-root "${contact_root}" --contact-manifest-sha256 "${contact_sha}" \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${split}" --split-manifest-sha256 "${split_sha}" \
  --protocol "${protocol}" --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${receipt_sha}" \
  --source-manifest "${source_manifest}" --source-manifest-sha256 "${source_sha}" \
  --training-partner-panel "${train_panel}" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output}" --seed 17 --experiment epiformer_style_same_protocol \
  --epitope-pooling topk_mean --epitope-pooling-k "${epitope_k}" \
  --paratope-pooling "${paratope_pooling}" --paratope-pooling-k "${paratope_k}" \
  --contact-readout pair \
  --epochs 20 --early-stopping-patience 5 --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 --dropout 0.1 \
  --learning-rate 0.0003 --weight-decay 0.0001 --contact-loss-weight 1.0 \
  --paratope-loss-weight 0.1 --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.0 --contrastive-loss-weight 0.0 --matched-local-loss-weight 0.0 \
  --matched-local-margin 0.05 --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 --epitope-count-weight 0.2 \
  --calibration-max-pairs 500000 --offload-saved-tensors --device cuda --precision bf16
