#!/usr/bin/env bash
#SBATCH --job-name=fc-control
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_same_protocol_v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_same_protocol_v1/logs/%x-%j.err

set -euo pipefail

root=/ssd/scxi253/model_deployment
source_root=${root}/data/framecontact_20260730/sabdab_source_same_protocol_v1
data_root=${root}/data/framecontact_20260730
python_bin=${root}/envs/contact/bin/python3.10
control_manifest=${root}/results/development/sabdab2_same_protocol_v1/launchers/SCREEN_CONTROL_MANIFEST_V1.json
protocol=${source_root}/protocols/protocol_fusion_only_screen_v1.json
receipt=${source_root}/protocols/PROTOCOL_FUSION_ONLY_SCREEN_V1_RECEIPT.json
output=${root}/results/development/sabdab2_same_protocol_v1/control/seed-17

printf '%s  %s\n' \
  f967709e9b0dfac4f89e78fc3c959ceb8bb914c411f02002fbff398cc2b72ce8 "${control_manifest}" \
  196f895a8ecd3aae66805009b8404a98c4dbcc2e7bcaaf502ce575e55d16d328 "${source_root}/src/interaction_first/framecontact.py" \
  54f9c0005f7433379eeca6c7509d61cf7719d835b2b28f2021031368bf103d5b "${source_root}/src/interaction_first/framecontact_objective.py" \
  7bc229b420717f6d4633c8b324cc4c10fee8bf160170681565662c1d32ec3315 "${source_root}/scripts/train_framecontact.py" \
  | sha256sum --check --strict

mkdir -p "$(dirname "${output}")" "${root}/results/development/sabdab2_same_protocol_v1/logs"
cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${source_root}/src

${python_bin} scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root ${data_root}/sabdab2/structure_cohort_v2 \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root ${data_root}/sabdab2/contact_cohort_v2 \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root ${root}/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest ${source_root}/protocols/screen-antigen-fold-v3.csv \
  --split-manifest-sha256 4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd \
  --protocol ${protocol} \
  --protocol-receipt ${receipt} \
  --protocol-receipt-sha256 3947fa5c37be40c6a718eed41a81814d0d44d757dae2369782f25548f2deef2a \
  --source-manifest ${data_root}/sabdab_source_same_protocol_v1_authority/SABDAB2_SAME_PROTOCOL_SOURCE_MANIFEST_V2.json \
  --source-manifest-sha256 93fb0306894f0ce436429331b8d26c1a6065d8147c066df35c0be5e021e0ff2e \
  --training-partner-panel ${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel ${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root ${output} \
  --seed 17 \
  --experiment framecontact_interleaved \
  --epochs 20 --early-stopping-patience 5 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 0.0 --epitope-dice-weight 0.0 --epitope-count-weight 0.0 \
  --calibration-max-pairs 500000 --offload-saved-tensors \
  --device cuda --precision bf16
