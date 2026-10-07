#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-plddt70-contact-recovery
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-confidence-contact-recovery-v1-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_sequence_predicted_geometry_development_v1/logs/prepare-confidence-contact-recovery-v1-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly tools=${root}/remote_tools/sabdab2_confidence_geometry_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly old_contact=${campaign}/data/contact_cache_predicted_geometry_authority_v1
readonly structure_cache=${campaign}/data/structure_cache_plddt70_v1
readonly contact_cache=${campaign}/data/contact_cache_plddt70_authority_v1
readonly authority=${campaign}/authority/confidence_geometry_plddt70_v1
readonly failed_runs=${campaign}/runs_confidence_geometry_plddt70_v1
readonly failure_root=${campaign}/failures/confidence_geometry_plddt70_training_1495054

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${contact_cache}"
test -d "${authority}"
test -d "${failed_runs}"
test ! -e "${failure_root}/authority"
test ! -e "${failure_root}/runs"
if find "${failed_runs}" -type f -print -quit | grep -q .; then
  echo "failed run roots unexpectedly contain training artifacts" >&2
  exit 2
fi

printf '%s  %s\n' \
  3e599c945b2a96df8f5a017e7d8c85cea7857ada473a2eab2740a2c02e89a7a9 "${tools}/materialize_contact_authority_adapter_v1.py" \
  1562b466bf15377e21f044bc1e02a2ef1296c8bbe9f226dfa88fc9807500f63e "${tools}/finalize_confidence_geometry_protocol_v1.py" \
  8ed99e5c1c6db76e9cce750c6820170d6420acbe20fbf09352250da62373928b "${tools}/slurm_confidence_geometry_development_v1.sbatch" \
  a23a3282788d8699239b0e5a54ed3d19926789587ad5099e65992b1d307a321d "${tools}/analyze_confidence_geometry_development_v1.py" \
  3bb52f61700bb7d41fe094feae5b8525483c061f91da21eb5b26ecb44432d69a "${tools}/FAIL_CLOSED_TRAINING_1495054.json" \
  6d716570d367102205dec2ff288918eefdbdcde97422ffd3e97de01bf35fd750 "${tools}/FAIL_CLOSED_TRAINING_1495054_RECEIPT.json" \
  ebafc1ae97564c4e0a4ff77ffa52522f1e83a0c2e6b34e5f938dc055cdedca59 "${authority}/CAMPAIGN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json" \
  5b14008d8398d24604c4862a9422d52c26b3cbd077d7bdc407cdca86a2574f25 "${authority}/PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json" \
  4f4342b483102521bcbd1247298a0ef43f427c2b5dd3c395a3e3dfc9081415cb "${authority}/PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1_RECEIPT.json" \
  d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88 "${old_contact}/CONTACT_CACHE_MANIFEST.json" \
  426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9 "${old_contact}/CONTACT_CACHE_RECEIPT.json" \
  1c49c5d49742bd7f1a2dba5c52539750cbd90ef03f0b4778492abe0955e496e5 "${structure_cache}/STRUCTURE_CACHE_MANIFEST.json" \
  c2f3129d1b59a96937e7b2094dfde149211451b58da92748e7359d9d5ccf0ab3 "${structure_cache}/STRUCTURE_CACHE_RECEIPT.json" \
  | sha256sum --check --strict

"${python_bin}" "${tools}/materialize_contact_authority_adapter_v1.py" \
  --source-contact-root "${old_contact}" \
  --target-structure-root "${structure_cache}" \
  --output-root "${contact_cache}"

mkdir -p "${failure_root}"
cp "${tools}/FAIL_CLOSED_TRAINING_1495054.json" "${failure_root}/"
cp "${tools}/FAIL_CLOSED_TRAINING_1495054_RECEIPT.json" "${failure_root}/"
mv "${authority}" "${failure_root}/authority"
mv "${failed_runs}" "${failure_root}/runs"
mkdir -p "${authority}"
cp "${failure_root}/authority/PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json" "${authority}/"
cp "${failure_root}/authority/PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1_RECEIPT.json" "${authority}/"

"${python_bin}" "${tools}/finalize_confidence_geometry_protocol_v1.py" \
  --campaign-root "${campaign}" \
  --masked-cache-root "${structure_cache}" \
  --contact-cache-root "${contact_cache}"

test "$(find "${contact_cache}/artifacts" -type f | wc -l)" -eq 3114
test ! -e "${campaign}/runs_confidence_geometry_plddt70_v1"
