#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-residual-screen
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abpairfm_joint_residual_screen_v2/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abpairfm_joint_residual_screen_v2/logs/%x-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/ca1f580_joint_residual_source_v2
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/abpairfm_joint_residual_screen_v2
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly joint_root=${root}/data/abpairfm_sabdab2_joint_embeddings_v1
readonly joint_manifest=${joint_root}/SABDAB2_ABPAIRFM_JOINT_EMBEDDING_MANIFEST.json
readonly protocol=${run_root}/authority/PROTOCOL_ABPAIRFM_JOINT_RESIDUAL_SCREEN_V2.json
readonly receipt=${run_root}/authority/PROTOCOL_ABPAIRFM_JOINT_RESIDUAL_SCREEN_V2_RECEIPT.json
readonly source_manifest=${run_root}/authority/SOURCE_MANIFEST_RESIDUAL_V2.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  6976f7cf286614321240f084e7577a109e5283b4f011325c8c0bac7fe333f26d "${source_root}/scripts/train_framecontact.py" \
  bb1e832cd9adadc6d638733f1931720a9072d5f9ae7b6658377802cd2d8922d7 "${source_manifest}" \
  57dde0903f934ade9fb709343291de7df5358b452d942ab2c61ceef87c31e813 "${protocol}" \
  35eb4bcb9cc9c8f2e796e3853876ef6ecb35571537942d9f80e2fb21b73d79e8 "${receipt}" \
  d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323 "${joint_manifest}" \
  4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  | sha256sum --check --strict

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly seed=17 ;;
  1) readonly seed=29 ;;
  2) readonly seed=43 ;;
  *) exit 64 ;;
esac

readonly output=${run_root}/abpairfm_epiformer_residual_fusion_same_protocol/seed-${seed}
test ! -e "${output}"
mkdir -p "$(dirname "${output}")" "${run_root}/logs"

"${python_bin}" - "${protocol}" "${seed}" <<'PY'
import json
import sys

protocol_path, seed = sys.argv[1:]
protocol = json.load(open(protocol_path, encoding="utf-8"))
assert int(seed) in protocol["model_selection"]["seeds"]
assert protocol["claim_boundary"]["test_loaded"] is False
assert protocol["data"]["input_feature_mode"] == "abpairfm_joint_residual_concatenation_v1"
assert protocol["tasks"]["winner"]["experiment"] == "abpairfm_epiformer_residual_fusion_same_protocol"
PY

cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src

"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --joint-embedding-root "${joint_root}" \
  --joint-embedding-manifest-sha256 d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323 \
  --joint-embedding-source-split-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --split-manifest "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  --split-manifest-sha256 4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd \
  --protocol "${protocol}" \
  --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 35eb4bcb9cc9c8f2e796e3853876ef6ecb35571537942d9f80e2fb21b73d79e8 \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 bb1e832cd9adadc6d638733f1931720a9072d5f9ae7b6658377802cd2d8922d7 \
  --training-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output}" \
  --seed "${seed}" --experiment abpairfm_epiformer_residual_fusion_same_protocol \
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
  --offload-saved-tensors --device cuda --precision bf16
