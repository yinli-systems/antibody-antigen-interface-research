#!/usr/bin/env bash
#SBATCH --job-name=epi-head-refine-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_head_refinement_v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_head_refinement_v1/logs/%x-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/epitope_head_refinement_source_v1
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/sabdab2_epitope_head_refinement_v1
readonly authority=${run_root}/authority
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly source_manifest=${authority}/SOURCE_MANIFEST_EPITOPE_HEAD_REFINEMENT_V1.json
readonly protocol=${authority}/PROTOCOL_EPITOPE_HEAD_REFINEMENT_V1.json
readonly protocol_receipt=${authority}/PROTOCOL_EPITOPE_HEAD_REFINEMENT_V1_RECEIPT.json
readonly campaign=${authority}/CAMPAIGN_AUTHORITY_V1.json
readonly initial_checkpoint=${root}/results/development/sabdab2_joint_task_competition_screen_v1/joint_current/seed-17/FRAMECONTACT_SELECTED.pt
readonly output=${run_root}/joint_checkpoint_epitope_head_only/seed-17

readonly source_script_sha256=db595f246290e3e8441341f5cd889fa73c0419f8bf2bb0de1e2a36c3e2b3a642
readonly source_manifest_sha256=9a3a77065a9011b7d9cd4b7048c2af330d80df2f7f3cff16e48c995b673ed6c0
readonly protocol_sha256=3e33395ad4e88494db3dcc4749d16e12bd49fdf786cbdf7b475a24b7df4d4320
readonly protocol_receipt_sha256=d6e0a7de68c63fcc641a01e0c1ce57313252ba57da59c5f447ddd4448d8fdaef
readonly campaign_sha256=8fd8cbf5975db4a58407819d30d5d7d69350f5db1f40884aa6ae6da3f35d04fc

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  "${source_script_sha256}" "${source_root}/scripts/train_framecontact.py" \
  "${source_manifest_sha256}" "${source_manifest}" \
  "${protocol_sha256}" "${protocol}" \
  "${protocol_receipt_sha256}" "${protocol_receipt}" \
  "${campaign_sha256}" "${campaign}" \
  5c2448c28a335dac0dcfa03d97a63be7331d664c3181c83a38bb563ca61b9a22 "${initial_checkpoint}" \
  4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/train-reciprocal-7.json" \
  c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/validation-reciprocal-7.json" \
  | sha256sum --check --strict

test ! -e "${output}"
mkdir -p "$(dirname "${output}")" "${run_root}/logs"

"${python_bin}" - "${protocol}" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p["protocol_status"] == "frozen_before_training"
assert p["model_selection"]["seeds"] == [17]
assert p["continuation"]["training_policy"] == "epitope_head_only"
assert p["continuation"]["trainable_parameter_names"] == [
    "epitope_head.weight",
    "epitope_head.bias",
]
assert p["claim_boundary"]["test_loaded"] is False
assert p["claim_boundary"]["temporal_holdout_loaded"] is False
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
  --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 "${protocol_receipt_sha256}" \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 "${source_manifest_sha256}" \
  --initial-checkpoint "${initial_checkpoint}" \
  --initial-checkpoint-sha256 5c2448c28a335dac0dcfa03d97a63be7331d664c3181c83a38bb563ca61b9a22 \
  --training-policy epitope_head_only \
  --training-partner-panel "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/train-reciprocal-7.json" \
  --training-partner-panel-sha256 2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24 \
  --validation-partner-panel "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/validation-reciprocal-7.json" \
  --validation-partner-panel-sha256 c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11 \
  --output-root "${output}" \
  --seed 17 --experiment epiformer_style_same_protocol \
  --epochs 5 --early-stopping-patience 2 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.00006464091462 \
  --weight-decay 0.000098653968 \
  --contact-loss-weight 0.0 --paratope-loss-weight 0.0 \
  --epitope-loss-weight 1.0 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.0 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 \
  --epitope-dice-weight 0.1992184401407531 \
  --epitope-count-weight 0.06957874447576923 \
  --calibration-max-pairs 500000 --device cuda --precision bf16
