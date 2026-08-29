#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-paired-full
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abpairfm_joint_energy_balanced_full_v1/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abpairfm_joint_energy_balanced_full_v1/logs/%x-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/abpairfm_energy_balanced_residual_source_v3
readonly data_root=${root}/data/framecontact_20260730
readonly run_root=${root}/results/development/abpairfm_joint_energy_balanced_full_v1
readonly authority=${run_root}/authority
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly source_manifest=${authority}/SOURCE_MANIFEST_ENERGY_BALANCED_V3.json
readonly promotion=${authority}/FULL_DEVELOPMENT_PROMOTION_MANIFEST_V1.json
readonly full_split=${data_root}/split_matrix_v3_1/antigen_fold.csv
readonly joint_root=${root}/data/abpairfm_sabdab2_joint_embeddings_v1
readonly joint_manifest=${joint_root}/SABDAB2_ABPAIRFM_JOINT_EMBEDDING_MANIFEST.json
readonly energy_audit=${root}/tools/FEATURE_ENERGY_AUDIT_V1.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' \
  0e183bfc2f5f572a8f42fa10287c84b43ab1438f0f264c8af2ecb2150a098925 "${source_root}/scripts/train_framecontact.py" \
  ca31451b4be85ee1e07045609a41ed3c90789971368191383c28cbf4251b581e "${source_manifest}" \
  370b1f4f9b0aec6ce86dc3664489b63f9b8f44381ea1338e043e80b578a1f292 "${promotion}" \
  fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 "${full_split}" \
  d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323 "${joint_manifest}" \
  aaa674a411d1373ed4e307086f7613cbb086e2ec1455a47e5538598f470c5fed "${energy_audit}" \
  | sha256sum --check --strict

joint_args=()
case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=baseline; readonly seed=17 ;;
  1) readonly arm=baseline; readonly seed=29 ;;
  2) readonly arm=baseline; readonly seed=43 ;;
  3) readonly arm=energy_balanced; readonly seed=17 ;;
  4) readonly arm=energy_balanced; readonly seed=29 ;;
  5) readonly arm=energy_balanced; readonly seed=43 ;;
  *) exit 64 ;;
esac

if test "${arm}" = baseline; then
  readonly experiment=epiformer_style_same_protocol
  readonly protocol=${authority}/PROTOCOL_BASELINE_FULL_V2.json
  readonly protocol_sha256=cb5c6c7569a26ed86f3dbc2b0167ecd0260f8fe6bd7de0ad4bde94bc995d49ef
  readonly receipt=${authority}/PROTOCOL_BASELINE_FULL_V2_RECEIPT.json
  readonly receipt_sha256=e64c57f4181d56fde7dbecf8858e655b57f7ecfdb8764010be618bc576876275
else
  readonly experiment=abpairfm_epiformer_energy_balanced_residual_same_protocol
  readonly protocol=${authority}/PROTOCOL_ENERGY_BALANCED_FULL_V2.json
  readonly protocol_sha256=90d0844b23c268c2b595c873405222503ae63e870bdd4584a80e23001d988944
  readonly receipt=${authority}/PROTOCOL_ENERGY_BALANCED_FULL_V2_RECEIPT.json
  readonly receipt_sha256=e78eaf54b7068ffc3d2757f10aabbd6464346c220285f95d11dbbf18432949b4
  joint_args=(
    --joint-embedding-root "${joint_root}"
    --joint-embedding-manifest-sha256 d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323
    --joint-embedding-source-split-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6
    --joint-residual-scale 0.6800912466498086
  )
fi

printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  "${receipt_sha256}" "${receipt}" \
  | sha256sum --check --strict

readonly output=${run_root}/${arm}/seed-${seed}
test ! -e "${output}"
mkdir -p "$(dirname "${output}")" "${run_root}/logs"

"${python_bin}" - "${protocol}" "${arm}" "${experiment}" "${seed}" <<'PY'
import json
import sys
protocol_path, arm, experiment, seed = sys.argv[1:]
protocol = json.load(open(protocol_path, encoding="utf-8"))
assert protocol["protocol_status"] == "frozen_before_full_development_training"
assert protocol["arm"] == arm
assert protocol["tasks"]["winner"]["experiment"] == experiment
assert int(seed) in protocol["model_selection"]["seeds"]
assert protocol["claim_boundary"]["test_loaded"] is False
assert protocol["claim_boundary"]["future_temporal_holdout_forbidden_before_full_development_freeze"] is True
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
  --split-manifest "${full_split}" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol "${protocol}" \
  --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${receipt_sha256}" \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 ca31451b4be85ee1e07045609a41ed3c90789971368191383c28cbf4251b581e \
  --training-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output}" \
  --seed "${seed}" --experiment "${experiment}" \
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
  "${joint_args[@]}"
