#!/usr/bin/env bash
#SBATCH --job-name=fc-val-int-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --array=0-14%15
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-val-int-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-val-int-v1-%A_%a.err

# Submit with --dependency=afterok:1349969.  Every task consumes one immutable,
# validation-selected v3 checkpoint and never opens a test or external role.

set -euo pipefail

seeds=(17 42 73 101 137)
experiments=(
  framecontact_gauge_marginal
  framecontact_gauge_hybrid
  framecontact_gauge_hybrid_contrastive
)
task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index < 0 || task_index >= 15 )); then
  echo "SLURM_ARRAY_TASK_ID is outside the frozen 0-14 matrix" >&2
  exit 2
fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/validation-intervention-source-v1"
development_root="$base/runs/specificity-gauge-exploratory-v3/$experiment/seed-$seed"
output_root="$base/runs/validation-interventions-v1/$experiment/seed-$seed"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3

if [[ -f "$output_root/VALIDATION_INTERVENTION_RECEIPT.json" && -f "$output_root/FRAMECONTACT_VALIDATION_INTERVENTIONS.json" ]]; then
  exit 0
fi
if [[ -e "$output_root" ]]; then
  echo "Partial validation-intervention output exists: $output_root" >&2
  exit 2
fi
for required in \
  "$development_root/FRAMECONTACT_DEVELOPMENT_RESULT.json" \
  "$development_root/FRAMECONTACT_SELECTED.pt" \
  "$development_root/RUN_RECEIPT.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Development dependency is incomplete: $required" >&2
    exit 2
  fi
done

cd "$source_root"
sha256sum -c SOURCE_SHA256SUMS.txt >/dev/null
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$source_root/src:$source_root/scripts"

"$python_bin" scripts/evaluate_framecontact_validation_interventions.py \
  --run-root "$development_root" \
  --structure-cache-root "$base/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "$base/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "$base/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --validation-partner-panel "$base/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "$output_root" \
  --evaluation-batch-size 1 \
  --workers 4 \
  --intervention-seed 20260803 \
  --coordinate-noise 0.10 0.25 0.50 \
  --device cuda \
  --precision bf16

sha256sum \
  "$output_root/FRAMECONTACT_VALIDATION_INTERVENTIONS.json" \
  "$output_root/VALIDATION_INTERVENTION_RECEIPT.json"
