#!/usr/bin/env bash
#SBATCH --job-name=fc-full-dev
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/logs/%x-%j.err

set -euo pipefail

root=/ssd/scxi253/model_deployment
source_root=/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab_source_distance_baselines_v3
data_root=${root}/data/framecontact_20260730
python_bin=${root}/envs/contact/bin/python3.10
protocol=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/authority/PROTOCOL_FULL_DEVELOPMENT_V1.json
receipt=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/authority/PROTOCOL_FULL_DEVELOPMENT_V1_RECEIPT.json
promotion=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/authority/FULL_DEVELOPMENT_PROMOTION_MANIFEST_V1.json
source_manifest=/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab_source_distance_baselines_v3_authority/SABDAB2_DISTANCE_BASELINES_SOURCE_MANIFEST_V3.json
full_split=/ssd/scxi253/model_deployment/data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv
output=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/epiformer_style_same_protocol/seed-17

printf '%s  %s
' \
  ce244ab5e677c2819ce4f4e9702568a54e3a094ebd89f1ac144fd73040748780 "${protocol}" \
  e4dd5de3a89fb05e877db760edf0b45095872fe4881a07d643a6372bc9afc937 "${receipt}" \
  e5474fc5d0f2b4a264e10e77c8af1876d7e84700950a1a5c0ac40ce687499d9d "${promotion}" \
  fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 "${full_split}" \
  1f776a12869fcf6a32e145bbcc44c579cdec0fe97460ce74a168a8035a4bff68 "${source_manifest}" \
  07ce142ee37c245f9c0d3c8e6705e350d3b758ceaff7385a967756125bbae2fc "${source_root}/scripts/train_framecontact.py" \
  a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c "${source_root}/src/interaction_first/framecontact.py" \
  380bca9791f022c5f19238db5fba897ad2350706bb1286b3c1cd80fe4a8eba48 "${source_root}/src/interaction_first/framecontact_objective.py" \
  | sha256sum --check --strict

"${python_bin}" - "${protocol}" "${promotion}" <<'PY'
import json
import math
import sys

protocol_path, promotion_path = sys.argv[1:]
with open(protocol_path, encoding="utf-8") as handle:
    protocol = json.load(handle)
with open(promotion_path, encoding="utf-8") as handle:
    promotion = json.load(handle)
task = protocol["tasks"]["winner"]
assert promotion["winner"] == "epiformer_style_same_protocol"
assert task["name"] == "epiformer_style_same_protocol"
assert task["experiment"] == "epiformer_style_same_protocol"
assert math.isclose(float(task["distance_loss_weight"]), 0.1, abs_tol=1e-12)
assert math.isclose(float(task["contrastive_loss_weight"]), 0.0, abs_tol=1e-12)
assert protocol["claim_boundary"]["test_loaded"] is False
assert promotion["test_loaded"] is False
PY

mkdir -p "$(dirname "${output}")" "/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_full_development_v1/logs"
cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src

"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root ${data_root}/sabdab2/structure_cohort_v2 \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root ${data_root}/sabdab2/contact_cohort_v2 \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root ${root}/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest ${full_split} \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol ${protocol} \
  --protocol-receipt ${receipt} \
  --protocol-receipt-sha256 e4dd5de3a89fb05e877db760edf0b45095872fe4881a07d643a6372bc9afc937 \
  --source-manifest ${source_manifest} \
  --source-manifest-sha256 1f776a12869fcf6a32e145bbcc44c579cdec0fe97460ce74a168a8035a4bff68 \
  --training-partner-panel ${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel ${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root ${output} \
  --seed 17 --experiment epiformer_style_same_protocol \
  --epochs 20 --early-stopping-patience 5 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight 0.1 \
  --contrastive-loss-weight 0.0 \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 \
  --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 \
  --calibration-max-pairs 500000 --offload-saved-tensors \
  --device cuda --precision bf16
