#!/usr/bin/env bash
#SBATCH --job-name=edge-mean-train-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_supervised_readout_screen_v2/logs/%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_supervised_readout_screen_v2/logs/%A_%a.err

set -euo pipefail

root=/ssd/scxi253/model_deployment
output_root=${root}/results/development/sabdab2_edge_supervised_readout_screen_v2
authority=${output_root}/authority
python_bin=${root}/envs/contact/bin/python3.10
data_root=${root}/data/framecontact_20260730
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    arm=baseline_current
    source_root=${root}/immutable_stage_a1_candidates/matched_local_training_source_v3
    source_manifest=${root}/results/development/sabdab2_matched_local_screen_v3/authority/SOURCE_MANIFEST_MATCHED_LOCAL_V3.json
    ;;
  1)
    arm=edge_mean_supervised
    source_root=${root}/immutable_stage_a1_candidates/edge_mean_training_source_v1
    source_manifest=${source_root}/SOURCE_MANIFEST_EDGE_MEAN_V1.json
    ;;
  *)
    exit 64
    ;;
esac
protocol=${authority}/PROTOCOL_${arm^^}_V1.json
receipt=${authority}/PROTOCOL_${arm^^}_V1_RECEIPT.json
split=${authority}/split-train512-selection63.csv
train_panel=${authority}/train512-reciprocal-6.json
validation_panel=${authority}/validation-selection63-reciprocal-6.json
run_root=${output_root}/runs/${arm}/seed-17

test "${SLURM_JOB_PARTITION}" = gpu_4090
for required in \
  "${python_bin}" \
  "${source_root}/scripts/train_framecontact.py" \
  "${source_manifest}" \
  "${protocol}" \
  "${receipt}" \
  "${split}" \
  "${train_panel}" \
  "${validation_panel}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 66
  fi
done
if [[ -e "${run_root}" || -L "${run_root}" ]]; then
  echo "immutable run root already exists: ${run_root}" >&2
  exit 73
fi
protocol_receipt_sha256=$(sha256sum "${receipt}" | awk '{print $1}')
source_manifest_sha256=$(sha256sum "${source_manifest}" | awk '{print $1}')
split_sha256=$(sha256sum "${split}" | awk '{print $1}')
validation_panel_sha256=$(sha256sum "${validation_panel}" | awk '{print $1}')
train_panel_sha256=$(sha256sum "${train_panel}" | awk '{print $1}')

"${python_bin}" - "${protocol}" "${receipt}" "${arm}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

protocol_path, receipt_path, arm = map(Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text())
receipt = json.loads(receipt_path.read_text())
observed = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
assert receipt["protocol_file_sha256"] == observed
assert protocol["protocol_status"] == "frozen_before_training"
assert protocol["arm"] == str(arm)
assert protocol["model_selection"]["seeds"] == [17]
assert protocol["claim_boundary"]["confirmation_panel_loaded"] is False
assert protocol["claim_boundary"]["test_loaded"] is False
assert protocol["claim_boundary"]["temporal_holdout_loaded"] is False
assert receipt["frozen_before_training"] is True
assert receipt["confirmation_panel_loaded"] is False
assert receipt["test_loaded"] is False
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
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${split}" \
  --split-manifest-sha256 "${split_sha256}" \
  --protocol "${protocol}" \
  --protocol-receipt "${receipt}" \
  --protocol-receipt-sha256 "${protocol_receipt_sha256}" \
  --source-manifest "${source_manifest}" \
  --source-manifest-sha256 "${source_manifest_sha256}" \
  --training-partner-panel "${train_panel}" \
  --training-partner-panel-sha256 "${train_panel_sha256}" \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 "${validation_panel_sha256}" \
  --output-root "${run_root}" \
  --seed 17 --experiment epiformer_style_same_protocol \
  --epochs 12 --early-stopping-patience 3 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 --contrastive-loss-weight 0.0 \
  --matched-local-loss-weight 0.0 --matched-local-margin 0.05 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 --calibration-max-pairs 500000 \
  --offload-saved-tensors --device cuda --precision bf16
