#!/usr/bin/env bash
#SBATCH -J pinder-ab-s2-mmseqs
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/mmseqs-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s2_v1/mmseqs-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abf_paircontact_v1/pinder_scale_v1
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s2_v1
readonly ASSETS=${OUTPUT}/assets
readonly WORK=${OUTPUT}/.mmseqs-${SLURM_JOB_ID}
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly MMSEQS=${ROOT}/data/abpairfm_3b_v1/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly MMSEQS_SHA=b0ca32bf507e6f76eab90d5d1f0a802b68d08192000f2aad7d38cffcd4bdd4fb
readonly AB_FASTA=${ASSETS}/PINDER_ANTIBODY_S2_CONCATENATED_ANTIBODY_V1.fasta
readonly AB_FASTA_SHA=4b2cd17a250a55fc382e677783a54bbf6adbbf1c0f9993c9198831f6a1c45d53
readonly AG_FASTA=${ASSETS}/PINDER_ANTIBODY_S2_ANTIGEN_V1.fasta
readonly AG_FASTA_SHA=010cd3328aae41ebca953bd63dd641128ac20b86508e1665de203a68783fd494

test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${MMSEQS}" | cut -d' ' -f1)" = "${MMSEQS_SHA}"
test "$(sha256sum "${AB_FASTA}" | cut -d' ' -f1)" = "${AB_FASTA_SHA}"
test "$(sha256sum "${AG_FASTA}" | cut -d' ' -f1)" = "${AG_FASTA_SHA}"
mkdir -p "${WORK}"
test ! -e "${OUTPUT}/PINDER_ANTIBODY_S2_CONCATENATED_ANTIBODY_RELATION_V1.tsv"
test ! -e "${OUTPUT}/PINDER_ANTIBODY_S2_ANTIGEN_RELATION_V1.tsv"

readonly MMSEQS_VERSION=$("${MMSEQS}" version)
for role in concatenated_antibody antigen; do
  if [[ "${role}" == "concatenated_antibody" ]]; then
    fasta=${AB_FASTA}
    fasta_sha=${AB_FASTA_SHA}
  else
    fasta=${AG_FASTA}
    fasta_sha=${AG_FASTA_SHA}
  fi
  "${MMSEQS}" createdb "${fasta}" "${WORK}/${role}_db"
  "${MMSEQS}" search "${WORK}/${role}_db" "${WORK}/${role}_db" \
    "${WORK}/${role}_hits" "${WORK}/${role}_tmp" \
    --min-seq-id 0.8 -c 0.8 --cov-mode 0 --alignment-mode 3 \
    --max-seqs 1000000 -s 7.5 --threads "${SLURM_CPUS_PER_TASK}"
  "${MMSEQS}" convertalis "${WORK}/${role}_db" "${WORK}/${role}_db" \
    "${WORK}/${role}_hits" "${WORK}/${role}_raw.tsv" \
    --format-output query,target,fident,qcov,tcov
  "${PYTHON}" "${SOURCE}/finalize_pinder_antibody_s2_mmseqs_v1.py" \
    --role "${role}" --fasta "${fasta}" --fasta-sha256 "${fasta_sha}" \
    --raw-tsv "${WORK}/${role}_raw.tsv" --mmseqs "${MMSEQS}" \
    --mmseqs-sha256 "${MMSEQS_SHA}" --mmseqs-version "${MMSEQS_VERSION}" \
    --identity-threshold 0.8 --coverage-threshold 0.8 \
    --output-tsv "${OUTPUT}/PINDER_ANTIBODY_S2_${role^^}_RELATION_V1.tsv" \
    --receipt "${OUTPUT}/PINDER_ANTIBODY_S2_${role^^}_RELATION_RECEIPT_V1.json"
done
