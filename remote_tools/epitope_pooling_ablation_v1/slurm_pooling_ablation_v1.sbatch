#!/usr/bin/env bash
#SBATCH --job-name=epi-pooling-ablation-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-6%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_pooling_ablation_v1/logs/training-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_pooling_ablation_v1/logs/training-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly source_root=${root}/immutable_stage_a1_candidates/epitope_pooling_ablation_source_v1
readonly source_manifest=${source_root}/SOURCE_MANIFEST_EPITOPE_POOLING_ABLATION_V1.json
readonly source_manifest_sha=10536e3a71d982d2cc6d8b82eefb6c7d51059fab0bf7f486fdbd266ada420348
readonly data_root=${root}/data/framecontact_20260730
readonly campaign=${root}/results/development/sabdab2_epitope_pooling_ablation_v1
readonly authority=${campaign}/authority
readonly full_split=${data_root}/split_matrix_v3_1/antigen_fold.csv
readonly train_panel=${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json
readonly validation_panel=${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json
readonly python_bin=${root}/envs/contact/bin/python3.10

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=topk1_max; readonly pooling=topk_mean; readonly pooling_k=1 ;;
  1) readonly arm=topk2; readonly pooling=topk_mean; readonly pooling_k=2 ;;
  2) readonly arm=topk4; readonly pooling=topk_mean; readonly pooling_k=4 ;;
  3) readonly arm=topk8; readonly pooling=topk_mean; readonly pooling_k=8 ;;
  4) readonly arm=noisy_or; readonly pooling=noisy_or; readonly pooling_k=2 ;;
  5) readonly arm=logmeanexp; readonly pooling=logmeanexp; readonly pooling_k=2 ;;
  6) readonly arm=learned_softmax; readonly pooling=learned_softmax; readonly pooling_k=2 ;;
  *) exit 64 ;;
esac

readonly protocol=${authority}/PROTOCOL_${arm^^}_V1.json
readonly receipt=${authority}/PROTOCOL_${arm^^}_V1_RECEIPT.json
readonly output=${campaign}/runs/${arm}/seed-17

test "${SLURM_JOB_PARTITION}" = gpu_4090
for required in "${source_manifest}" "${protocol}" "${receipt}" "${full_split}" "${train_panel}" "${validation_panel}"; do
  test -f "${required}"
done
test "$(sha256sum "${source_manifest}" | cut -d' ' -f1)" = "${source_manifest_sha}"
if [[ -e "${output}" || -L "${output}" ]]; then
  echo "immutable output exists: ${output}" >&2
  exit 73
fi
mkdir -p "$(dirname "${output}")"

readarray -t hashes < <("${python_bin}" - "${protocol}" "${receipt}" "${arm}" "${pooling}" <<'PY'
import hashlib,json,sys
protocol_path,receipt_path,arm,pooling=sys.argv[1:]
raw=open(protocol_path,'rb').read(); protocol=json.loads(raw); receipt=json.load(open(receipt_path))
assert receipt['protocol_file_sha256']==hashlib.sha256(raw).hexdigest()
assert protocol['protocol_status']=='frozen_before_seed17_training'
assert protocol['arm']==arm and protocol['epitope_readout']['pooling']==pooling
assert protocol['model_selection']['seeds']==[17]
assert protocol['claim_boundary']['test_loaded'] is False
assert protocol['claim_boundary']['temporal_holdout_loaded'] is False
print(hashlib.sha256(raw).hexdigest())
print(hashlib.sha256(open(receipt_path,'rb').read()).hexdigest())
PY
)
readonly protocol_sha=${hashes[0]}
readonly receipt_sha=${hashes[1]}

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
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${full_split}" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol "${protocol}" --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${receipt_sha}" \
  --source-manifest "${source_manifest}" --source-manifest-sha256 "${source_manifest_sha}" \
  --training-partner-panel "${train_panel}" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "${output}" --seed 17 --experiment epiformer_style_same_protocol \
  --epitope-pooling "${pooling}" --epitope-pooling-k "${pooling_k}" \
  --epochs 20 --early-stopping-patience 5 --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 --dropout 0.1 \
  --learning-rate 0.0003 --weight-decay 0.0001 --contact-loss-weight 1.0 \
  --paratope-loss-weight 0.1 --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 --matched-local-loss-weight 0.0 \
  --matched-local-margin 0.05 --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 --epitope-count-weight 0.2 \
  --calibration-max-pairs 500000 --offload-saved-tensors --device cuda --precision bf16
