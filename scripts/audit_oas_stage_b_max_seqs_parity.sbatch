#!/usr/bin/env bash
#SBATCH -J abpairfm-oas-maxseqs-audit
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-oas-maxseqs-audit-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-oas-maxseqs-audit-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=${ABPAIRFM_STAGE_B_PROJECT:?missing immutable Stage-B project root}
readonly SOURCE_SHA256=${ABPAIRFM_STAGE_B_PARITY_SOURCE_SHA256:?missing frozen parity source SHA-256}
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly SOURCE=${ROOT}/normalized_candidates/oas_paired_20260821
readonly INPUT=${SOURCE}/oas_stage_b_exact_pairs_v2.tsv.gz
readonly INPUT_RECEIPT=${SOURCE}/oas_stage_b_exact_pairs_v2.receipt.json
readonly DEVELOPMENT_FASTA=${ROOT}/development_anchor/development_sequences_v1.fasta
readonly DEVELOPMENT_RECEIPT=${ROOT}/development_anchor/development_sequences_v1.receipt.json
readonly MMSEQS=${ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly CAPPED_HITS_OUTPUT=${SOURCE}/oas_stage_b_maxseqs64_canary_hits_v1.tsv.gz
readonly EXHAUSTIVE_HITS_OUTPUT=${SOURCE}/oas_stage_b_maxseqs1000000_canary_hits_v1.tsv.gz
readonly RECEIPT=${SOURCE}/oas_stage_b_maxseqs64_parity_v1.receipt.json
readonly MODULUS=32
readonly REMAINDER=0
readonly MINIMUM_PAIRS=50000
readonly CAPPED_MAXIMUM_TARGET_CANDIDATES=64
readonly EXHAUSTIVE_MAXIMUM_TARGET_CANDIDATES=1000000

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
for path in "${PYTHON}" "${PROJECT}/scripts/audit_oas_stage_b_max_seqs_parity.py" \
  "${PROJECT}/scripts/decontaminate_oas_stage_b_pairs.py" "${INPUT}" \
  "${INPUT_RECEIPT}" "${DEVELOPMENT_FASTA}" "${DEVELOPMENT_RECEIPT}" "${MMSEQS}"; do
  test -e "${path}"
done
for path in "${CAPPED_HITS_OUTPUT}" "${EXHAUSTIVE_HITS_OUTPUT}" "${RECEIPT}"; do
  test ! -e "${path}"
done

readonly OBSERVED_SOURCE_SHA256=$(sha256sum \
  "${PROJECT}/scripts/audit_oas_stage_b_max_seqs_parity.py" | cut -d' ' -f1)
if [[ "${OBSERVED_SOURCE_SHA256}" != "${SOURCE_SHA256}" ]]; then
  echo "Frozen Stage-B parity source SHA-256 differs" >&2
  exit 2
fi

WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/abpairfm-oas-maxseqs-audit.XXXXXX")
readonly WORK
trap 'rm -rf "${WORK}"' EXIT

readonly SAMPLE_JSON=$("${PYTHON}" \
  "${PROJECT}/scripts/audit_oas_stage_b_max_seqs_parity.py" sample-fasta \
  --input "${INPUT}" --source-receipt "${INPUT_RECEIPT}" \
  --output "${WORK}/sample.fasta" --modulus "${MODULUS}" \
  --remainder "${REMAINDER}" --minimum-pairs "${MINIMUM_PAIRS}")
readonly SOURCE_PAIRS=$("${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["source_pairs"])' "${SAMPLE_JSON}")
readonly SAMPLED_PAIRS=$("${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["sampled_pairs"])' "${SAMPLE_JSON}")

"${MMSEQS}" createdb "${WORK}/sample.fasta" "${WORK}/sample_db"
"${MMSEQS}" createdb "${DEVELOPMENT_FASTA}" "${WORK}/development_db"
for label in capped exhaustive; do
  mkdir -p "${WORK}/tmp-${label}"
done
"${MMSEQS}" search "${WORK}/sample_db" "${WORK}/development_db" \
  "${WORK}/capped_db" "${WORK}/tmp-capped" --min-seq-id 0.5 -c 0.8 \
  --cov-mode 0 --alignment-mode 3 --max-seqs "${CAPPED_MAXIMUM_TARGET_CANDIDATES}" \
  -s 7.0 --mask 0 --threads 6
"${MMSEQS}" search "${WORK}/sample_db" "${WORK}/development_db" \
  "${WORK}/exhaustive_db" "${WORK}/tmp-exhaustive" --min-seq-id 0.5 -c 0.8 \
  --cov-mode 0 --alignment-mode 3 --max-seqs "${EXHAUSTIVE_MAXIMUM_TARGET_CANDIDATES}" \
  -s 7.0 --mask 0 --threads 6
for label in capped exhaustive; do
  "${MMSEQS}" convertalis "${WORK}/sample_db" "${WORK}/development_db" \
    "${WORK}/${label}_db" "${WORK}/${label}.tsv" \
    --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits
done

"${PYTHON}" "${PROJECT}/scripts/audit_oas_stage_b_max_seqs_parity.py" compare \
  --input "${INPUT}" --source-receipt "${INPUT_RECEIPT}" \
  --development-fasta "${DEVELOPMENT_FASTA}" \
  --development-receipt "${DEVELOPMENT_RECEIPT}" \
  --sample-fasta "${WORK}/sample.fasta" \
  --capped-hits "${WORK}/capped.tsv" --exhaustive-hits "${WORK}/exhaustive.tsv" \
  --capped-hits-output "${CAPPED_HITS_OUTPUT}" \
  --exhaustive-hits-output "${EXHAUSTIVE_HITS_OUTPUT}" --receipt "${RECEIPT}" \
  --source-pairs "${SOURCE_PAIRS}" --sampled-pairs "${SAMPLED_PAIRS}" \
  --modulus "${MODULUS}" --remainder "${REMAINDER}" \
  --capped-maximum-target-candidates "${CAPPED_MAXIMUM_TARGET_CANDIDATES}" \
  --exhaustive-maximum-target-candidates "${EXHAUSTIVE_MAXIMUM_TARGET_CANDIDATES}" \
  --mmseqs-version "$("${MMSEQS}" version)"
