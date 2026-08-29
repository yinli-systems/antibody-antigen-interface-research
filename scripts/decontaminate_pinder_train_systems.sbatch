#!/usr/bin/env bash
#SBATCH -J abpairfm-pinder-decontam
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-pinder-decontam-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-pinder-decontam-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly PINDER_ROOT=${ROOT}/normalized_candidates/pinder_2024_02
readonly PANEL_ROOT=${ROOT}/development_anchor
readonly MMSEQS=${ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly INPUT=${PINDER_ROOT}/train_systems_v1.tsv.gz
readonly INPUT_RECEIPT=${PINDER_ROOT}/train_system_manifest_audit_v1.json
readonly SEQUENCES=${PINDER_ROOT}/pinder_holo_sequences_v3.fasta
readonly SEQUENCE_MANIFEST=${PINDER_ROOT}/pinder_holo_sequences_v3.tsv.gz
readonly SEQUENCE_RECEIPT=${PINDER_ROOT}/pinder_holo_sequences_v3.receipt.json
readonly OUTPUT=${PINDER_ROOT}/train_systems_devclean_v1.tsv.gz
readonly RECEIPT=${PINDER_ROOT}/train_systems_devclean_v1.receipt.json
readonly HITS_OUTPUT=${PINDER_ROOT}/development_homology_hits_v1.tsv
readonly EXCLUDED_OUTPUT=${PINDER_ROOT}/development_excluded_endpoints_v1.txt
readonly WORK=${PINDER_ROOT}/.decontam-work-${SLURM_JOB_ID}

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden PINDER decontamination allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
export ABPAIRFM_GPU_COUNT="${GPU_COUNT}"
export ABPAIRFM_GPU_NAME="${GPU_NAME}"

for path in "${PYTHON}" "${MMSEQS}" "${INPUT}" "${INPUT_RECEIPT}" "${SEQUENCES}" "${SEQUENCE_MANIFEST}" "${SEQUENCE_RECEIPT}" "${PANEL_ROOT}/development_sequences_v1.fasta" "${PANEL_ROOT}/development_sequences_v1.receipt.json"; do
  test -e "${path}"
done
for path in "${OUTPUT}" "${RECEIPT}" "${HITS_OUTPUT}" "${EXCLUDED_OUTPUT}"; do
  test ! -e "${path}"
done
mkdir -p "${WORK}/tmp"

"${MMSEQS}" createdb "${PANEL_ROOT}/development_sequences_v1.fasta" "${WORK}/development_db"
"${MMSEQS}" createdb "${SEQUENCES}" "${WORK}/pinder_db"
"${MMSEQS}" search "${WORK}/development_db" "${WORK}/pinder_db" "${WORK}/hits_db" "${WORK}/tmp" --min-seq-id 0.5 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 1000000 -s 7.0 --threads 6
"${MMSEQS}" convertalis "${WORK}/development_db" "${WORK}/pinder_db" "${WORK}/hits_db" "${WORK}/hits.tsv" --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits

readonly MMSEQS_VERSION=$("${MMSEQS}" version)
RUN_ARGS=(
  --train-manifest "${INPUT}"
  --train-manifest-receipt "${INPUT_RECEIPT}"
  --sequence-manifest "${SEQUENCE_MANIFEST}"
  --sequence-receipt "${SEQUENCE_RECEIPT}"
  --development-fasta "${PANEL_ROOT}/development_sequences_v1.fasta"
  --development-receipt "${PANEL_ROOT}/development_sequences_v1.receipt.json"
  --hits "${WORK}/hits.tsv"
  --hits-output "${HITS_OUTPUT}"
  --excluded-endpoints-output "${EXCLUDED_OUTPUT}"
  --output "${OUTPUT}"
  --receipt "${RECEIPT}"
  --mmseqs-version "${MMSEQS_VERSION}"
)
"${PYTHON}" "${PROJECT}/scripts/decontaminate_pinder_train_systems.py" "${RUN_ARGS[@]}"
