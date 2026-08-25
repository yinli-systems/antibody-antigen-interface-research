#!/usr/bin/env bash
#SBATCH --job-name=joint-reg-obj-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_joint_regularization_objective_screen_v1/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_joint_regularization_objective_screen_v1/logs/%x-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/matched_local_training_source_v3
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/sabdab2_joint_regularization_objective_screen_v1
readonly authority=${run_root}/authority
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly source_manifest=${root}/results/development/sabdab2_matched_local_screen_v3/authority/SOURCE_MANIFEST_MATCHED_LOCAL_V3.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e "${source_root}/scripts/train_framecontact.py" \
  f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81 "${source_manifest}" \
  4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/train-reciprocal-7.json" \
  c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/validation-reciprocal-7.json" \
  64779b5b7cece143359d9d9a95cdb7244db718f69a4cc31cadd72f0793714405 "${authority}/CAMPAIGN_AUTHORITY_V1.json" \
  ca089218e598cd026b2d3cc7ad0362b3b7b0fe50da55599577abf3a3c6b78a72 "${authority}/EARLY_STOPPING_REPLAY_V1.json" \
  | sha256sum --check --strict

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    readonly arm=dropout020_count020
    readonly dropout=0.2
    readonly count_weight=0.2
    readonly protocol_sha256=3fb7d3e5176f657cb96831a2cf925c13eadddea2043f1659ee93da76011cd275
    readonly receipt_sha256=baf00d571ddc6be34fdea6b015cefccccfb63ec5d0dd656c95efb9c86d0c20e8
    ;;
  1)
    readonly arm=dropout010_count000
    readonly dropout=0.1
    readonly count_weight=0.0
    readonly protocol_sha256=b51e51a7ad51f17bdbdafb2f6f5902bb3aa50d8099a8b26c01d4cd4ef4f0536c
    readonly receipt_sha256=36116d2767bc4a2a764a286be317240528b9c33e160ab711538da2b3a1d17530
    ;;
  2)
    readonly arm=dropout020_count000
    readonly dropout=0.2
    readonly count_weight=0.0
    readonly protocol_sha256=1bf5308e7b49882f68231fd897a9df7243fc23bf2c433e8bdeae57260404fc2a
    readonly receipt_sha256=aa5eaf8134cd97538d84fef374e15a479f1e01342b8c9e54ea0f4ca8ff5c0e58
    ;;
  *) exit 64 ;;
esac

readonly upper_arm=${arm^^}
readonly protocol=${authority}/PROTOCOL_${upper_arm}_V1.json
readonly receipt=${authority}/PROTOCOL_${upper_arm}_V1_RECEIPT.json
printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  "${receipt_sha256}" "${receipt}" \
  | sha256sum --check --strict

readonly output=${run_root}/${arm}/seed-17
test ! -e "${output}"
mkdir -p "$(dirname "${output}")"

"${python_bin}" - "${protocol}" "${arm}" "${dropout}" "${count_weight}" <<'PY'
import json
import sys

protocol_path, arm, dropout, count_weight = sys.argv[1:]
protocol = json.load(open(protocol_path, encoding="utf-8"))
objective = protocol["joint_objective"]
regularization = protocol["regularization"]
assert protocol["protocol_status"] == "frozen_before_training"
assert protocol["arm"] == arm
assert protocol["seed"] == 17
assert protocol["model_selection"]["early_stopping_patience"] == 3
assert float(objective["contact_loss_weight"]) == 1.0
assert float(objective["paratope_loss_weight"]) == 0.1
assert float(objective["epitope_loss_weight"]) == 0.5
assert float(objective["epitope_dice_weight"]) == 1.0
assert float(objective["epitope_count_weight"]) == float(count_weight)
assert float(regularization["dropout"]) == float(dropout)
assert float(regularization["weight_decay"]) == 0.0001
assert protocol["claim_boundary"]["test_loaded"] is False
assert protocol["claim_boundary"]["temporal_holdout_loaded"] is False
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
  --split-manifest "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  --split-manifest-sha256 4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd \
  --protocol "${protocol}" \
  --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${receipt_sha256}" \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81 \
  --training-partner-panel "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/train-reciprocal-7.json" \
  --training-partner-panel-sha256 2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24 \
  --validation-partner-panel "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/validation-reciprocal-7.json" \
  --validation-partner-panel-sha256 c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11 \
  --output-root "${output}" \
  --seed 17 --experiment epiformer_style_same_protocol \
  --epochs 12 --early-stopping-patience 3 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout "${dropout}" --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight "${count_weight}" --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16
