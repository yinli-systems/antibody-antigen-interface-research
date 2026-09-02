#!/usr/bin/env bash
#SBATCH --job-name=sable-af3-inputs
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/af3-inputs-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/af3-inputs-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly tools=${root}/tools/sable_external_v1
readonly campaign=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly gate=${campaign}/label_free_gate
readonly output=${campaign}/structure_prediction/af3_label_free_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly source_sha256=51f21f544dd0615943463f149ca9a5ed0be68cd1f531bd9c95a7bbf29a7f38a6

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
echo "${source_sha256}  ${tools}/build_sable_v0_2_0_af3_inputs_v1.py" | sha256sum --check --strict -

"${python_bin}" "${tools}/build_sable_v0_2_0_af3_inputs_v1.py" \
  --screen-result "${gate}/SABLE_V0_2_0_LABEL_FREE_SCREEN_RESULT.json" \
  --screen-receipt "${gate}/SABLE_V0_2_0_LABEL_FREE_SCREEN_RECEIPT.json" \
  --output-root "${output}" \
  --batch-size 10
