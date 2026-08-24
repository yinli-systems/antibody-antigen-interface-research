#!/usr/bin/env bash
#SBATCH --job-name=fc-dist-base
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_v2/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_distance_baselines_v2/logs/%x-%A_%a.err

set -euo pipefail

root=/ssd/scxi253/model_deployment
source_root=${root}/data/framecontact_20260730/sabdab_source_distance_baselines_v2
authority_root=${root}/data/framecontact_20260730/sabdab_source_distance_baselines_v2_authority
data_root=${root}/data/framecontact_20260730
result_root=${root}/results/development/sabdab2_distance_baselines_v2
python_bin=${root}/envs/contact/bin/python3.10
protocol=${source_root}/protocols/protocol_distance_baseline_screen_v2.json
receipt=${source_root}/protocols/PROTOCOL_DISTANCE_BASELINE_SCREEN_V2_RECEIPT.json
task_manifest=${source_root}/protocols/SCREEN_ADAPTER_MANIFEST_V2.json
source_manifest=${authority_root}/SABDAB2_DISTANCE_BASELINES_SOURCE_MANIFEST_V2.json

printf '%s  %s\n' \
  54eb1f15b96a18f9ef4f0a8a4836bb7a179a123c8ef9baff9a84fb26b7c182af "${task_manifest}" \
  959dc5cc084979595b26b7ba051cae8ae456873fd585d3d9a68223646885a877 "${protocol}" \
  f321f47f90682fccdc7a18849547bbcdaec28ed9c126e93e05e1fc751c4358ac "${receipt}" \
  a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c "${source_root}/src/interaction_first/framecontact.py" \
  380bca9791f022c5f19238db5fba897ad2350706bb1286b3c1cd80fe4a8eba48 "${source_root}/src/interaction_first/framecontact_objective.py" \
  07ce142ee37c245f9c0d3c8e6705e350d3b758ceaff7385a967756125bbae2fc "${source_root}/scripts/train_framecontact.py" \
  7bc6e00e60dc99d447fa63512493fc45295aa43f3b70429a984876584ff6c166 "${source_manifest}" \
  4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd "${source_root}/protocols/screen-antigen-fold-v3.csv" \
  | sha256sum --check --strict

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    name=framecontact_distance_aux
    experiment=framecontact_distance_aux
    distance_weight=0.1
    contrastive_weight=0.0
    ;;
  1)
    name=epiformer_style_same_protocol
    experiment=epiformer_style_same_protocol
    distance_weight=0.1
    contrastive_weight=0.0
    ;;
  2)
    name=mipe_style_same_protocol
    experiment=mipe_style_same_protocol
    distance_weight=0.0
    contrastive_weight=0.05
    ;;
  *)
    exit 64
    ;;
esac

"${python_bin}" - "${task_manifest}" "${SLURM_ARRAY_TASK_ID}" \
  "${name}" "${experiment}" "${distance_weight}" "${contrastive_weight}" <<'PY'
import json
import math
import sys

manifest_path, task_id, name, experiment, distance, contrastive = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    task = json.load(handle)["tasks"][task_id]
assert task["name"] == name
assert task["experiment"] == experiment
assert math.isclose(float(task["distance_loss_weight"]), float(distance), abs_tol=1e-12)
assert math.isclose(float(task["contrastive_loss_weight"]), float(contrastive), abs_tol=1e-12)
PY

output=${result_root}/${name}/seed-17
mkdir -p "$(dirname "${output}")" "${result_root}/logs"
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
  --split-manifest ${source_root}/protocols/screen-antigen-fold-v3.csv \
  --split-manifest-sha256 4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd \
  --protocol ${protocol} \
  --protocol-receipt ${receipt} \
  --protocol-receipt-sha256 f321f47f90682fccdc7a18849547bbcdaec28ed9c126e93e05e1fc751c4358ac \
  --source-manifest ${source_manifest} \
  --source-manifest-sha256 7bc6e00e60dc99d447fa63512493fc45295aa43f3b70429a984876584ff6c166 \
  --training-partner-panel ${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel ${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root ${output} \
  --seed 17 --experiment ${experiment} \
  --epochs 20 --early-stopping-patience 5 \
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \
  --distance-loss-weight ${distance_weight} \
  --contrastive-loss-weight ${contrastive_weight} \
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \
  --contrastive-dim 64 \
  --specificity-margin 0.05 --score-fraction 0.01 \
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \
  --epitope-count-weight 0.2 \
  --calibration-max-pairs 500000 --offload-saved-tensors \
  --device cuda --precision bf16
