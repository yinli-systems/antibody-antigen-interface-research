#!/usr/bin/env bash
#SBATCH --job-name=matched-local-screen
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_matched_local_screen_v3/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_matched_local_screen_v3/logs/%x-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/matched_local_training_source_v3
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/sabdab2_matched_local_screen_v3
readonly authority=${run_root}/authority
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly matched_root=${data_root}/sabdab2/matched_local_train512_p95_frozen_v1
readonly matched_manifest=${matched_root}/MATCHED_LOCAL_DECOY_MANIFEST.json
readonly matched_audit=${run_root}/MATCHED_LOCAL_TRAIN512_P95_AUDIT_V1.json
readonly source_manifest=${authority}/SOURCE_MANIFEST_MATCHED_LOCAL_V3.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e "${source_root}/scripts/train_framecontact.py" \
  f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81 "${source_manifest}" \
  58860008dad531d65938b2e24ddfdfaa1360b36caf4c7ce7fe16ca5a88505dc0 "${matched_manifest}" \
  69c2db5b71f763d4e6abf6b6e4fb2643c6d7c45a46441635c03319ce0d90f525 "${matched_audit}" \
  4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/train-reciprocal-7.json" \
  c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11 "${data_root}/partner_metadata_antigen_fold_v3_screen512_v1/validation-reciprocal-7.json" \
  | sha256sum --check --strict

matched_args=()
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    readonly arm=control
    readonly matched_weight=0.0
    readonly protocol=${authority}/PROTOCOL_CONTROL_V2.json
    readonly protocol_sha256=cd7fc6eb4d68054e9d3a4fee5bc10a81b4bf3d645d7760bffca181ed678d7cfe
    readonly receipt=${authority}/PROTOCOL_CONTROL_V2_RECEIPT.json
    readonly receipt_sha256=2c9b012bda1ff3acb6055551bae9dc7f2fcfc23fc7c8d29cb97647f4311a9e17
    ;;
  1)
    readonly arm=matched_local
    readonly matched_weight=0.25
    readonly protocol=${authority}/PROTOCOL_MATCHED_LOCAL_V2.json
    readonly protocol_sha256=1fbde99c69acadad32ecf54e554dd2e4001e1575c0ead5177931f5f7c321e639
    readonly receipt=${authority}/PROTOCOL_MATCHED_LOCAL_V2_RECEIPT.json
    readonly receipt_sha256=8d91f2782915b752e01bea75fce0f2a3411f31bd332b1f71ac9570761728a027
    matched_args=(
      --matched-local-root "${matched_root}"
      --matched-local-manifest-sha256 58860008dad531d65938b2e24ddfdfaa1360b36caf4c7ce7fe16ca5a88505dc0
    )
    ;;
  *) exit 64 ;;
esac

printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  "${receipt_sha256}" "${receipt}" \
  | sha256sum --check --strict

readonly output=${run_root}/${arm}/seed-17
test ! -e "${output}"
mkdir -p "$(dirname "${output}")" "${run_root}/logs"

"${python_bin}" - "${protocol}" "${arm}" "${matched_weight}" <<'PY'
import json
import sys
protocol_path, arm, weight = sys.argv[1:]
protocol = json.load(open(protocol_path, encoding="utf-8"))
assert protocol["protocol_status"] == "frozen_before_training"
assert protocol["arm"] == arm
assert protocol["model_selection"]["seeds"] == [17]
assert protocol["model_selection"]["primary_metric"] == "contact_mean_per_complex_auprc"
assert float(protocol["training_objective"]["loss_weights"]["matched_local"]) == float(weight)
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
  --seed 17 --experiment epiformer_style_same_protocol \
  --epochs 12 --early-stopping-patience 3 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight "${matched_weight}" --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16 \
  "${matched_args[@]}"
