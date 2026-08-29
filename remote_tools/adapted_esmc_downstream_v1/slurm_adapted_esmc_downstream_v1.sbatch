#!/usr/bin/env bash
#SBATCH -J esmc-adapt-down-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-5%4
#SBATCH -o /ssd/scxi253/model_deployment/results/development/sabdab2_adapted_esmc_downstream_v1/logs/training-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/sabdab2_adapted_esmc_downstream_v1/logs/training-%A_%a.err

set -euo pipefail
: "${EXPECTED_PROTOCOL_SHA256:?submit with frozen protocol SHA-256}"
: "${EXPECTED_RECEIPT_SHA256:?submit with frozen receipt SHA-256}"
readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/edge_topk2_training_source_v1
readonly source_manifest=${source_root}/SOURCE_MANIFEST_EDGE_TOPK2_V1.json
readonly data_root=${root}/data/framecontact_20260730
readonly campaign=${root}/results/development/sabdab2_adapted_esmc_downstream_v1
readonly protocol=${campaign}/authority/PROTOCOL_ADAPTED_ESMC_DOWNSTREAM_V1.json
readonly receipt=${campaign}/authority/PROTOCOL_ADAPTED_ESMC_DOWNSTREAM_V1_RECEIPT.json
readonly full_split=${data_root}/split_matrix_v3_1/antigen_fold.csv
readonly train_panel=${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json
readonly validation_panel=${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json
readonly python_bin=${root}/envs/contact/bin/python3.10
case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=whole_chain_adapted seed=17 ;;
  1) readonly arm=whole_chain_adapted seed=29 ;;
  2) readonly arm=whole_chain_adapted seed=43 ;;
  3) readonly arm=cdr_focused_adapted seed=17 ;;
  4) readonly arm=cdr_focused_adapted seed=29 ;;
  5) readonly arm=cdr_focused_adapted seed=43 ;;
  *) exit 64 ;;
esac
case "${arm}" in
  whole_chain_adapted)
    readonly embedding_root=${data_root}/sabdab2_esmc600m_whole_chain_adapted_dev_v1
    readonly embedding_sha=9823e519fda598082cebd784fb4297d8442f447514b2817bd7df26a2023823c9
    ;;
  cdr_focused_adapted)
    readonly embedding_root=${data_root}/sabdab2_esmc600m_cdr_focused_adapted_dev_v1
    readonly embedding_sha=e35531d3c788c43620e6efaa233ec2426aa169eabd825ed1d4741ab9528d3f46
    ;;
esac
readonly output=${campaign}/runs/${arm}/seed-${seed}
test "${SLURM_JOB_PARTITION}" = gpu_4090
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')" = "NVIDIA GeForce RTX 4090"
printf '%s  %s\n' \
  ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e "${source_root}/scripts/train_framecontact.py" \
  121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe "${source_root}/src/interaction_first/framecontact.py" \
  f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55 "${source_manifest}" \
  fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 "${full_split}" \
  fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 "${train_panel}" \
  222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 "${validation_panel}" \
  "${embedding_sha}" "${embedding_root}/SABDAB2_ESM_EMBEDDING_MANIFEST.json" \
  "${EXPECTED_PROTOCOL_SHA256}" "${protocol}" \
  "${EXPECTED_RECEIPT_SHA256}" "${receipt}" | sha256sum --check --strict
if [[ -e "${output}" || -L "${output}" ]]; then
  echo "immutable output exists: ${output}" >&2
  exit 73
fi
mkdir -p "$(dirname "${output}")"
"${python_bin}" - "${protocol}" "${arm}" "${seed}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); arm=sys.argv[2]; seed=int(sys.argv[3])
assert p["protocol_status"] == "frozen_before_same_protocol_development_training"
assert arm in p["representation_arms"]
assert seed in p["model_selection"]["seeds"]
assert p["claim_boundary"]["old_test_loaded"] is False
assert p["claim_boundary"]["temporal_holdout_loaded"] is False
PY
cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root "${embedding_root}" --embedding-manifest-sha256 "${embedding_sha}" \
  --split-manifest "${full_split}" --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol "${protocol}" --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${EXPECTED_RECEIPT_SHA256}" \
  --source-manifest "${source_manifest}" --source-manifest-sha256 f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55 \
  --training-partner-panel "${train_panel}" --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${validation_panel}" --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output}" --seed "${seed}" --experiment epiformer_style_same_protocol \
  --epochs 20 --early-stopping-patience 5 --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 --epitope-loss-weight 0.5 \
  --specificity-loss-weight 0.0 --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 --contrastive-dim 64 \
  --specificity-margin 0.05 --score-fraction 0.01 --contact-focal-gamma 1.0 \
  --epitope-dice-weight 1.0 --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16
