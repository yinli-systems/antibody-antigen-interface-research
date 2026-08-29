#!/usr/bin/env bash
#SBATCH --job-name=fc-neg-v2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=5-00:00:00
#SBATCH --array=0-19%20
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-neg-v2-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-neg-v2-%A_%a.err

set -euo pipefail
umask 077

seeds=(17 42 73 101 137)
task=${SLURM_ARRAY_TASK_ID}
seed=${seeds[$((task % 5))]}
dataset_block=$((task / 10))
arm_block=$(((task % 10) / 5))
if (( dataset_block == 0 )); then
  dataset=asep
else
  dataset=sabdab2
fi
if (( arm_block == 0 )); then
  arm=with_negative
  specificity_loss_weight=0.25
else
  arm=no_negative
  specificity_loss_weight=0.0
fi

data_root=/ssd/scxi253/model_deployment/data/framecontact_20260730
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
study_root=${data_root}/negative-sample-ablation-v2
source_root=${study_root}/sources/${dataset}
protocol_root=${study_root}/protocols/${dataset}/${arm}
protocol=${protocol_root}/framecontact_${dataset}_${arm}_v2.json
protocol_receipt=${protocol_root}/PROTOCOL_RECEIPT.json
output_root=${data_root}/${dataset}/development-runs/negative-sample-ablation-v2/${arm}/framecontact/seed-${seed}
freezer=${study_root}/tools/freeze_negative_sample_ablation_v2.py

[[ ${SLURM_JOB_PARTITION:-} == gpu_4090 ]] \
  || { echo "gpu_4090 is required" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "NVIDIA GeForce RTX 4090" ]] \
  || { echo "one RTX 4090 is required" >&2; exit 2; }
[[ ! -e ${output_root} ]] \
  || { echo "refusing to overwrite ${output_root}" >&2; exit 2; }
for path in "${python_bin}" "${protocol}" "${protocol_receipt}" "${freezer}"; do
  [[ -e ${path} ]] || { echo "missing ${path}" >&2; exit 2; }
done

"${python_bin}" "${freezer}" verify \
  --receipt "${protocol_receipt}" \
  --protocol "${protocol}" \
  --source-root "${source_root}" \
  > /dev/null
protocol_receipt_sha256=$("${python_bin}" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "${protocol_receipt}")

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=${source_root}/src

common_args=(
  --dataset-backend "${dataset}"
  --protocol "${protocol}"
  --protocol-receipt "${protocol_receipt}"
  --protocol-receipt-sha256 "${protocol_receipt_sha256}"
  --output-root "${output_root}"
  --seed "${seed}"
  --experiment framecontact
  --workers 4
  --d-model 128
  --num-heads 8
  --num-self-layers 2
  --num-rbf 16
  --neighbor-k 32
  --neighbor-radius 16.0
  --sequence-window 32
  --dropout 0.1
  --learning-rate 0.0003
  --weight-decay 0.0001
  --contact-loss-weight 1.0
  --paratope-loss-weight 0.25
  --epitope-loss-weight 0.25
  --specificity-loss-weight "${specificity_loss_weight}"
  --specificity-margin 0.05
  --score-fraction 0.01
  --calibration-max-pairs 2000000
  --device cuda
  --precision bf16
)

cd "${source_root}"
if [[ ${dataset} == asep ]]; then
  exec "${python_bin}" scripts/train_framecontact.py \
    --asep-cache-root "${data_root}/asep/development-cache" \
    --asep-cache-manifest-sha256 69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7 \
    --asep-feature-mode official_esm2_35m \
    --training-partner-panel "${data_root}/asep/panels/train-reciprocal-31.json" \
    --training-partner-panel-sha256 a0c5f3e8ea1a0edf7cd9a2043b326ee3be7b054c81dff9fbfcb510db6f2969d1 \
    --validation-partner-panel "${data_root}/asep/panels/validation-reciprocal-31.json" \
    --validation-partner-panel-sha256 10f50aa122568d4ed1a8261637f2e807f2702997a1aa5d9925983d37cc0d83ec \
    --epochs 100 \
    --early-stopping-patience 15 \
    --batch-size 2 \
    --evaluation-batch-size 8 \
    "${common_args[@]}"
fi

exec "${python_bin}" scripts/train_framecontact.py \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --source-manifest "${study_root}/authority/sabdab2/SABDAB2_TRAINING_SOURCE_MANIFEST.json" \
  --source-manifest-sha256 d73bb84ba1fd056737e1d9a985b02ebc0c5f3bb5f5d1edbd694d3db9ccf18cdd \
  --training-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "${data_root}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --epochs 200 \
  --early-stopping-patience 20 \
  --batch-size 1 \
  --evaluation-batch-size 1 \
  --offload-saved-tensors \
  "${common_args[@]}"
