#!/usr/bin/env bash
#SBATCH --job-name=epiformer-top2-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_epiformer_top2_screen_v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_epiformer_top2_screen_v1/logs/%x-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/epiformer_top2_training_source_v1
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/sabdab2_epiformer_top2_screen_v1
readonly authority=${run_root}/authority
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly source_manifest=${source_root}/SOURCE_MANIFEST_EPIFORMER_TOP2_V1.json
readonly protocol=${authority}/PROTOCOL_EPIFORMER_TOP2_V1.json
readonly protocol_receipt=${authority}/PROTOCOL_EPIFORMER_TOP2_V1_RECEIPT.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  ccc3ba7f9402289a7e65a358aa88690822fdd22ad69273e722571bf5711a62f5 "${source_root}/src/interaction_first/framecontact.py" \
  ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e "${source_root}/scripts/train_framecontact.py" \
  757373282bec0340f3f0171b7471a881fbd4f854a088bc81b72315e9bba24157 "${source_manifest}" \
  4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/train-reciprocal-7.json" \
  c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/validation-reciprocal-7.json" \
  5796560023097215b9d87e48fafb4e3facccf5ef1c40dbf0ffc0eb29032a03d6 "${protocol}" \
  b4df9dae6ccfd44326e960d806eab8cbd14037efb398441e0bb07f3aa7f5853a "${protocol_receipt}" \
  2e68ab19a83c416c04f5054c7c11366be72e464afc9eaf39c72e8fcf941dfc39 "${authority}/CAMPAIGN_AUTHORITY_V1.json" \
  | sha256sum --check --strict

"${python_bin}" -B - "${protocol}" <<'PY'
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
assert protocol["protocol_status"] == "frozen_before_training"
assert protocol["arm"] == "epiformer_top2"
assert protocol["seed"] == 17
assert protocol["tasks"]["winner"]["experiment"] == "epiformer_style_same_protocol"
assert protocol["implementation"]["contact_marginalization"] == {
    "input": "sigmoid_contact_probability",
    "pooling": "masked_top_k_mean",
    "k": 2,
    "output": "stable_logit_added_to_direct_residue_head",
    "changed_for_this_screen": True,
}
assert protocol["joint_objective"] == {
    "contact_loss_weight": 1.0,
    "distance_loss_weight": 0.1,
    "epitope_count_weight": 0.2,
    "epitope_dice_weight": 1.0,
    "epitope_loss_weight": 0.5,
    "paratope_loss_weight": 0.1,
}
assert protocol["claim_boundary"]["test_loaded"] is False
assert protocol["claim_boundary"]["temporal_holdout_loaded"] is False
PY

readonly output=${run_root}/epiformer_top2/seed-17
test ! -e "${output}"
mkdir -p "$(dirname "${output}")"

cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1

"${python_bin}" -B scripts/train_framecontact.py \
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
  --protocol-receipt-sha256 b4df9dae6ccfd44326e960d806eab8cbd14037efb398441e0bb07f3aa7f5853a \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 757373282bec0340f3f0171b7471a881fbd4f854a088bc81b72315e9bba24157 \
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
  --dropout 0.2 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16
