#!/usr/bin/env bash
#SBATCH -J pinder-ab-s40-rel
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s40_v1/relations-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s40_v1/relations-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/results/development/pinder_antibody_s40_v1/source
readonly ASSETS=${ROOT}/results/development/pinder_antibody_s2_v1/assets
readonly OUTPUT=${ROOT}/results/development/pinder_antibody_s40_v1
readonly WORK=${OUTPUT}/.mmseqs-${SLURM_JOB_ID}
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly MMSEQS=${ROOT}/data/abpairfm_3b_v1/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly MMSEQS_SHA=b0ca32bf507e6f76eab90d5d1f0a802b68d08192000f2aad7d38cffcd4bdd4fb

declare -Ar FASTA_NAME=(
  [vh]=PINDER_ANTIBODY_S2_VH_V1.fasta
  [vl]=PINDER_ANTIBODY_S2_VL_V1.fasta
  [concatenated_antibody]=PINDER_ANTIBODY_S2_CONCATENATED_ANTIBODY_V1.fasta
  [antigen]=PINDER_ANTIBODY_S2_ANTIGEN_V1.fasta
)
declare -Ar FASTA_SHA=(
  [vh]=eff253bba83bdcdaf5270e51e697efa417d6388f0b3ea3651e6461e6f5121310
  [vl]=12c2d4b90d76a2137ca86ea59bf2954a07efe940819a699e7630d176c8cb48c9
  [concatenated_antibody]=4b2cd17a250a55fc382e677783a54bbf6adbbf1c0f9993c9198831f6a1c45d53
  [antigen]=010cd3328aae41ebca953bd63dd641128ac20b86508e1665de203a68783fd494
)

mkdir -p "${OUTPUT}" "${WORK}"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${MMSEQS}" | cut -d' ' -f1)" = "${MMSEQS_SHA}"
readonly MMSEQS_VERSION=$("${MMSEQS}" version)

for role in vh vl concatenated_antibody antigen; do
  fasta=${ASSETS}/${FASTA_NAME[${role}]}
  fasta_sha=${FASTA_SHA[${role}]}
  test "$(sha256sum "${fasta}" | cut -d' ' -f1)" = "${fasta_sha}"
  test ! -e "${OUTPUT}/${role}.identity40_coverage80.relation.tsv"
  test ! -e "${OUTPUT}/${role}.identity40_coverage80.receipt.json"
  "${MMSEQS}" createdb "${fasta}" "${WORK}/${role}_db"
  "${MMSEQS}" search "${WORK}/${role}_db" "${WORK}/${role}_db" \
    "${WORK}/${role}_hits" "${WORK}/${role}_tmp" \
    --min-seq-id 0.4 -c 0.8 --cov-mode 0 --alignment-mode 3 \
    --max-seqs 1000000 -s 7.5 --threads "${SLURM_CPUS_PER_TASK}"
  "${MMSEQS}" convertalis "${WORK}/${role}_db" "${WORK}/${role}_db" \
    "${WORK}/${role}_hits" "${WORK}/${role}_raw.tsv" \
    --format-output query,target,fident,qcov,tcov
  "${PYTHON}" "${SOURCE}/finalize_pinder_antibody_s40_mmseqs_v1.py" \
    --role "${role}" --fasta "${fasta}" --fasta-sha256 "${fasta_sha}" \
    --raw-tsv "${WORK}/${role}_raw.tsv" --mmseqs "${MMSEQS}" \
    --mmseqs-sha256 "${MMSEQS_SHA}" --mmseqs-version "${MMSEQS_VERSION}" \
    --identity-threshold 0.4 --coverage-threshold 0.8 \
    --output-tsv "${OUTPUT}/${role}.identity40_coverage80.relation.tsv" \
    --receipt "${OUTPUT}/${role}.identity40_coverage80.receipt.json"
done

"${PYTHON}" "${SOURCE}/verify_pinder_antibody_s40_relations_v1.py" \
  --protocol "${SOURCE}/PROTOCOL_PINDER_ANTIBODY_S40_RELATION_V1.json" \
  --output-dir "${OUTPUT}"
rm -rf "${WORK}"
