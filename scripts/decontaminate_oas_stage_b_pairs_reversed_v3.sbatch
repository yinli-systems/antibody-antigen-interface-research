#!/usr/bin/env bash
#SBATCH -J abpairfm-oas-stageb-decontam-v3
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-oas-stageb-decontam-v3-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-oas-stageb-decontam-v3-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_STAGE_B_PROJECT:?missing immutable Stage-B project root}
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly SOURCE=${ROOT}/normalized_candidates/oas_paired_20260821
readonly INPUT=${SOURCE}/oas_stage_b_exact_pairs_v2.tsv.gz
readonly INPUT_RECEIPT=${SOURCE}/oas_stage_b_exact_pairs_v2.receipt.json
readonly PANEL=${ROOT}/development_anchor
readonly MMSEQS=${ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly OUTPUT=${SOURCE}/oas_stage_b_devclean_50id80cov_reversed_v3.tsv.gz
readonly HITS_OUTPUT=${SOURCE}/oas_stage_b_development_homology_hits_reversed_v3.tsv.gz
readonly RECEIPT=${SOURCE}/oas_stage_b_devclean_50id80cov_reversed_v3.receipt.json
readonly SOURCE_SHA256=${ABPAIRFM_STAGE_B_SOURCE_SHA256:?missing frozen Stage-B source SHA-256}
readonly MAXIMUM_TARGET_CANDIDATES=64

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
for path in "${PYTHON}" "${PROJECT}/scripts/decontaminate_oas_stage_b_pairs.py" \
  "${INPUT}" "${INPUT_RECEIPT}" "${PANEL}/development_sequences_v1.fasta" \
  "${PANEL}/development_sequences_v1.receipt.json" "${MMSEQS}"; do
  test -e "${path}"
done
for path in "${OUTPUT}" "${HITS_OUTPUT}" "${RECEIPT}"; do
  test ! -e "${path}"
done

readonly OBSERVED_SOURCE_SHA256=$("${PYTHON}" - "${PROJECT}" <<'PY'
import importlib.util
import sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "scripts/decontaminate_oas_stage_b_pairs.py"
spec = importlib.util.spec_from_file_location("oas_stage_b_decontam", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.sha256_file(path))
PY
)
if [[ "${OBSERVED_SOURCE_SHA256}" != "${SOURCE_SHA256}" ]]; then
  echo "Frozen Stage-B source SHA-256 differs" >&2
  exit 2
fi

WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/abpairfm-oas-stageb-v3.XXXXXX")
readonly WORK
trap 'rm -rf "${WORK}"' EXIT

"${PYTHON}" "${PROJECT}/scripts/decontaminate_oas_stage_b_pairs.py" export-fasta \
  --input "${INPUT}" --source-receipt "${INPUT_RECEIPT}" \
  --output "${WORK}/oas_pairs.fasta"
mkdir -p "${WORK}/tmp"
"${MMSEQS}" createdb "${WORK}/oas_pairs.fasta" "${WORK}/oas_db"
"${MMSEQS}" createdb "${PANEL}/development_sequences_v1.fasta" "${WORK}/development_db"
"${MMSEQS}" search "${WORK}/oas_db" "${WORK}/development_db" \
  "${WORK}/hits_db" "${WORK}/tmp" --min-seq-id 0.5 -c 0.8 --cov-mode 0 \
  --alignment-mode 3 --max-seqs "${MAXIMUM_TARGET_CANDIDATES}" -s 7.0 \
  --mask 0 --threads 6
"${MMSEQS}" convertalis "${WORK}/oas_db" "${WORK}/development_db" \
  "${WORK}/hits_db" "${WORK}/hits.tsv" \
  --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits

"${PYTHON}" "${PROJECT}/scripts/decontaminate_oas_stage_b_pairs.py" filter \
  --input "${INPUT}" --source-receipt "${INPUT_RECEIPT}" \
  --development-fasta "${PANEL}/development_sequences_v1.fasta" \
  --development-receipt "${PANEL}/development_sequences_v1.receipt.json" \
  --hits "${WORK}/hits.tsv" --hits-output "${HITS_OUTPUT}" \
  --output "${OUTPUT}" --receipt "${RECEIPT}" \
  --oas-column query --search-orientation oas_query_development_target \
  --maximum-target-candidates "${MAXIMUM_TARGET_CANDIDATES}" \
  --no-sequence-masking --mmseqs-version "$("${MMSEQS}" version)"
