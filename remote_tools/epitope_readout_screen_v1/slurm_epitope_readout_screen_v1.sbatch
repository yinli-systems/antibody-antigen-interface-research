#!/bin/bash
#SBATCH --job-name=epi-readout-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_readout_screen_v2/logs/%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_readout_screen_v2/logs/%A_%a.err

set -euo pipefail

python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
repository_root=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/matched_local_training_source_v3
tool_root=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/epitope_readout_screen_source_v1
output_root=/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_readout_screen_v2
source_runs=/ssd/scxi253/model_deployment/results/development/sabdab2_joint_task_competition_screen_v1/joint_current
structure_root=/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2/structure_cohort_v2
contact_root=/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2/contact_cohort_v2
embedding_root=/ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings
structure_manifest="${structure_root}/STRUCTURE_CACHE_MANIFEST.json"
contact_manifest="${contact_root}/CONTACT_CACHE_MANIFEST.json"
embedding_manifest="${embedding_root}/SABDAB2_ESM_EMBEDDING_MANIFEST.json"
split_manifest="${repository_root}/protocols/screen-antigen-fold-v3.csv"
source_split_manifest="${repository_root}/protocols/screen-antigen-fold-v3.csv"
partition_manifest="${output_root}/authority/PARTITION_EPITOPE_READOUT_V1.json"
phase="${EPITOPE_READOUT_PHASE:-}"
case "${phase}" in
  selection)
    if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
      echo "selection phase only permits array task 0" >&2
      exit 64
    fi
    seed=17
    protocol="${output_root}/authority/PROTOCOL_EPITOPE_READOUT_SELECTION_V1.json"
    protocol_receipt="${output_root}/authority/PROTOCOL_EPITOPE_READOUT_SELECTION_V1_RECEIPT.json"
    split_manifest="${output_root}/authority/split-selection.csv"
    result="${output_root}/seed-results/selection-seed-17.json"
    receipt="${output_root}/seed-results/selection-seed-17.receipt.json"
    ;;
  confirmation)
    seeds=(17 29 43)
    seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
    protocol="${output_root}/authority/PROTOCOL_EPITOPE_READOUT_CONFIRMATION_V1.json"
    protocol_receipt="${output_root}/authority/PROTOCOL_EPITOPE_READOUT_CONFIRMATION_V1_RECEIPT.json"
    split_manifest="${output_root}/authority/split-confirmation.csv"
    result="${output_root}/seed-results/confirmation-seed-${seed}.json"
    receipt="${output_root}/seed-results/confirmation-seed-${seed}.receipt.json"
    ;;
  *)
    echo "EPITOPE_READOUT_PHASE must be selection or confirmation" >&2
    exit 64
    ;;
esac

if [[ "${SLURM_JOB_PARTITION:-}" != "gpu_4090" ]]; then
  echo "gpu_4090 is required" >&2
  exit 64
fi
for required in \
  "${python_bin}" \
  "${tool_root}/evaluate_epitope_readouts_v1.py" \
  "${tool_root}/write_epitope_readout_receipt_v1.py" \
  "${protocol}" \
  "${protocol_receipt}" \
  "${source_runs}/seed-${seed}/FRAMECONTACT_SELECTED.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 66
  fi
done
if [[ -e "${result}" || -e "${receipt}" ]]; then
  echo "immutable output already exists for seed ${seed}" >&2
  exit 73
fi

protocol_sha256="$(sha256sum "${protocol}" | awk '{print $1}')"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

srun "${python_bin}" "${tool_root}/evaluate_epitope_readouts_v1.py" \
  --repository-root "${repository_root}" \
  --run-root "${source_runs}/seed-${seed}" \
  --structure-cache-root "${structure_root}" \
  --structure-manifest "${structure_manifest}" \
  --contact-cache-root "${contact_root}" \
  --contact-manifest "${contact_manifest}" \
  --embedding-root "${embedding_root}" \
  --embedding-manifest "${embedding_manifest}" \
  --source-split-manifest "${source_split_manifest}" \
  --split-manifest "${split_manifest}" \
  --partition-manifest "${partition_manifest}" \
  --protocol "${protocol}" \
  --protocol-sha256 "${protocol_sha256}" \
  --output "${result}" \
  --workers "${SLURM_CPUS_PER_TASK}"

"${python_bin}" "${tool_root}/write_epitope_readout_receipt_v1.py" \
  --result "${result}" \
  --protocol "${protocol}" \
  --protocol-receipt "${protocol_receipt}" \
  --output "${receipt}"
