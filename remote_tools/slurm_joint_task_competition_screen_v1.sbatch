#!/usr/bin/env bash
#SBATCH --job-name=joint-task-2x2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_joint_task_competition_screen_v1/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_joint_task_competition_screen_v1/logs/%x-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/matched_local_training_source_v3
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/sabdab2_joint_task_competition_screen_v1
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
  cc1892e720f98567dc62ebf4e85343cd7deb1d6d862b50d5934dc0acf01d30ab "${authority}/CAMPAIGN_AUTHORITY_V1.json" \
  | sha256sum --check --strict

readonly seed_index=$((SLURM_ARRAY_TASK_ID / 4))
readonly arm_index=$((SLURM_ARRAY_TASK_ID % 4))
case "${seed_index}" in
  0) readonly seed=17 ;;
  1) readonly seed=29 ;;
  2) readonly seed=43 ;;
  *) exit 64 ;;
esac
case "${arm_index}" in
  0)
    readonly arm=epitope_only
    readonly contact_weight=0.0
    readonly paratope_weight=0.0
    readonly protocol_sha256=34cd4150894d7429ff9bd5d11d777c34627106f00d7d2d0545bcdae0e4271012
    readonly receipt_sha256=1a32103081e19bb0f1bcd6a49cce29b8b90ae81ae847f924d05859aa64686900
    ;;
  1)
    readonly arm=epitope_paratope
    readonly contact_weight=0.0
    readonly paratope_weight=0.1
    readonly protocol_sha256=668c5b29ca814e8e00c1aefb4eaa87ee515a3fbe3bfbf6fffc8d3ceeb789c19e
    readonly receipt_sha256=0170a1561b8a9176012f2a2642ab214521fc1a4d4a2327065486791076ea672b
    ;;
  2)
    readonly arm=epitope_contact
    readonly contact_weight=1.0
    readonly paratope_weight=0.0
    readonly protocol_sha256=d8016c6da54116c2b579da21ef1092fbe137911844938721adc75fb43d1a3f8f
    readonly receipt_sha256=c12b13949b50057bbb646a32b089dc5b15575580279c3bd1eaa2a7050aa6bb2d
    ;;
  3)
    readonly arm=joint_current
    readonly contact_weight=1.0
    readonly paratope_weight=0.1
    readonly protocol_sha256=2678f09ff99639862acb75147b100deffb14046bdded4d9c06ca31181bfae335
    readonly receipt_sha256=dd4ea39df0e60f7b538997c2a1177113f831c04a73b74555613860a50f2d8069
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

readonly output=${run_root}/${arm}/seed-${seed}
test ! -e "${output}"
mkdir -p "$(dirname "${output}")" "${run_root}/logs"

"${python_bin}" - "${protocol}" "${arm}" "${seed}" "${contact_weight}" "${paratope_weight}" <<'PY'
import json
import sys
protocol_path, arm, seed, contact, paratope = sys.argv[1:]
protocol = json.load(open(protocol_path, encoding="utf-8"))
weights = protocol["training_objective"]["loss_weights"]
assert protocol["protocol_status"] == "frozen_before_training"
assert protocol["arm"] == arm
assert protocol["model_selection"]["seeds"] == [17, 29, 43]
assert int(seed) in protocol["model_selection"]["seeds"]
assert protocol["model_selection"]["primary_metric"] == "epitope_mean_per_complex_auprc"
assert float(weights["contact"]) == float(contact)
assert float(weights["paratope"]) == float(paratope)
assert float(weights["epitope"]) == 0.5
assert protocol["claim_boundary"]["test_loaded"] is False
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
  --seed "${seed}" --experiment epiformer_style_same_protocol \
  --epochs 12 --early-stopping-patience 3 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight "${contact_weight}" --paratope-loss-weight "${paratope_weight}" \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16
