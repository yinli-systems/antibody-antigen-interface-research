#!/usr/bin/env bash
#SBATCH -J pinder-cdr-ag-s40
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_cdr_antigen_s40_v1/run-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_cdr_antigen_s40_v1/run-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_cdr_antigen_s40_v1
readonly ASSETS=${ROOT}/results/development/pinder_antibody_s2_v1/assets
readonly S40=${ROOT}/results/development/pinder_antibody_s40_v1
readonly WORK=${OUTPUT}/.mmseqs-${SLURM_JOB_ID}
readonly ANARCII_PYTHON=${ROOT}/envs/oas-imgt-anarcii-2.0.8-torch-2.7.1-cpu-py3.12/bin/python
readonly CONTACT_PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly ANARCII_WHEEL=${ROOT}/artifacts/oas-imgt-wheelhouse-anarcii-2.0.8-torch-2.7.1-cpu/anarcii-2.0.8-py3-none-any.whl
readonly MMSEQS=${ROOT}/data/abpairfm_3b_v1/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly MMSEQS_SHA=b0ca32bf507e6f76eab90d5d1f0a802b68d08192000f2aad7d38cffcd4bdd4fb

readonly PROTOCOL=${SOURCE}/PINDER_ANTIBODY_CDR_ANTIGEN_S40_PROTOCOL_V1.json
readonly MATERIALIZER=${SOURCE}/materialize_pinder_antibody_cdr_views_v1.py
readonly FINALIZER=${SOURCE}/finalize_pinder_antibody_cdr_mmseqs_v1.py
readonly BASE_FINALIZER=${SOURCE}/finalize_pinder_antibody_s2_mmseqs_v1.py
readonly DIAGNOSTIC=${SOURCE}/diagnose_pinder_cdr_antigen_s40_v1.py
readonly CANONICAL=${ASSETS}/PINDER_ANTIBODY_S2_CANONICAL_V1.csv
readonly ANTIGEN_RELATION=${S40}/antigen.identity40_coverage80.relation.tsv
readonly S40_RESULT=${S40}/PINDER_ANTIBODY_S40_RELATION_RESULT_V1.json

mkdir -p "${OUTPUT}" "${WORK}"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" = 48f46a64e0eccb5ec6b49798685b98939f45c3b598e99c66d6f2d4f39b79c98f
test "$(sha256sum "${MATERIALIZER}" | cut -d' ' -f1)" = fcfd77894f6c3eb7a8102bf7bd42c4d9a7e198bb71d4b953efd6e6bee0ade939
test "$(sha256sum "${FINALIZER}" | cut -d' ' -f1)" = 7ef2061e51c5168f7367eeee627fbfabffed3ef2e310bd64ef81350401f5ec7e
test "$(sha256sum "${BASE_FINALIZER}" | cut -d' ' -f1)" = 17f9226ddc0103ce4811561938a8b8d75ff9cfcca8c39ce14ac9fb0013bc78b4
test "$(sha256sum "${DIAGNOSTIC}" | cut -d' ' -f1)" = f227944304d85c474b3687f4fd03ec7d6a25d9bb44e9bdc962e36e065edffdb2
test "$(sha256sum "${CANONICAL}" | cut -d' ' -f1)" = 624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c
test "$(sha256sum "${S40_RESULT}" | cut -d' ' -f1)" = aceaf0b53b60f4c124c3c3b26e57082477678ccebf3b0c8cb939b5692b1f6b5d
test "$(sha256sum "${MMSEQS}" | cut -d' ' -f1)" = "${MMSEQS_SHA}"

test ! -e "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEWS_V1.csv"
test ! -e "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEWS_V1.fasta"
test ! -e "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEW_RESULT_V1.json"
test ! -e "${OUTPUT}/antibody_cdr.identity40_coverage80.relation.tsv"
test ! -e "${OUTPUT}/antibody_cdr.identity40_coverage80.receipt.json"
test ! -e "${OUTPUT}/PINDER_ANTIBODY_CDR_ANTIGEN_S40_DIAGNOSTIC_V1.json"

"${ANARCII_PYTHON}" "${MATERIALIZER}" \
  --protocol "${PROTOCOL}" \
  --canonical-manifest "${CANONICAL}" \
  --anarcii-wheel "${ANARCII_WHEEL}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output-csv "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEWS_V1.csv" \
  --output-fasta "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEWS_V1.fasta" \
  --output-result "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEW_RESULT_V1.json"

readonly CDR_FASTA=${OUTPUT}/PINDER_ANTIBODY_CDR_VIEWS_V1.fasta
readonly CDR_FASTA_SHA=$(sha256sum "${CDR_FASTA}" | cut -d' ' -f1)
readonly MMSEQS_VERSION=$("${MMSEQS}" version)
"${MMSEQS}" createdb "${CDR_FASTA}" "${WORK}/cdr_db"
"${MMSEQS}" search "${WORK}/cdr_db" "${WORK}/cdr_db" \
  "${WORK}/cdr_hits" "${WORK}/cdr_tmp" \
  --min-seq-id 0.4 -c 0.8 --cov-mode 0 --alignment-mode 3 \
  --max-seqs 1000000 -s 7.5 --threads "${SLURM_CPUS_PER_TASK}"
"${MMSEQS}" convertalis "${WORK}/cdr_db" "${WORK}/cdr_db" \
  "${WORK}/cdr_hits" "${WORK}/cdr_raw.tsv" \
  --format-output query,target,fident,qcov,tcov
"${CONTACT_PYTHON}" "${FINALIZER}" \
  --fasta "${CDR_FASTA}" --fasta-sha256 "${CDR_FASTA_SHA}" \
  --raw-tsv "${WORK}/cdr_raw.tsv" --mmseqs "${MMSEQS}" \
  --mmseqs-sha256 "${MMSEQS_SHA}" --mmseqs-version "${MMSEQS_VERSION}" \
  --identity-threshold 0.4 --coverage-threshold 0.8 \
  --output-tsv "${OUTPUT}/antibody_cdr.identity40_coverage80.relation.tsv" \
  --receipt "${OUTPUT}/antibody_cdr.identity40_coverage80.receipt.json"

"${CONTACT_PYTHON}" "${DIAGNOSTIC}" \
  --protocol "${PROTOCOL}" \
  --canonical-manifest "${CANONICAL}" \
  --cdr-csv "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEWS_V1.csv" \
  --cdr-result "${OUTPUT}/PINDER_ANTIBODY_CDR_VIEW_RESULT_V1.json" \
  --cdr-relation "${OUTPUT}/antibody_cdr.identity40_coverage80.relation.tsv" \
  --cdr-relation-receipt "${OUTPUT}/antibody_cdr.identity40_coverage80.receipt.json" \
  --antigen-relation "${ANTIGEN_RELATION}" \
  --s40-relation-result "${S40_RESULT}" \
  --output "${OUTPUT}/PINDER_ANTIBODY_CDR_ANTIGEN_S40_DIAGNOSTIC_V1.json"

rm -rf "${WORK}"
