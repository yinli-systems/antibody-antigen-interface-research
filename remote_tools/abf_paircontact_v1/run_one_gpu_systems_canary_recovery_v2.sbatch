#!/usr/bin/env bash
#SBATCH --job-name=abf-paircontact-gpu1-r2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf_paircontact_one_gpu_systems_canary_recovery_v2/slurm-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf_paircontact_one_gpu_systems_canary_recovery_v2/slurm-%j.err

set -euo pipefail
umask 077

readonly root=/ssd/scxi253/model_deployment
readonly candidate=${root}/tools/abf_paircontact_v1/phase0_candidate_v1
readonly protocol=${root}/tools/abf_paircontact_v1/ONE_GPU_SYSTEMS_CANARY_RECOVERY_PROTOCOL_V2.json
readonly failure=${root}/tools/abf_paircontact_v1/ONE_GPU_SYSTEMS_CANARY_FAILURE_V1.json
readonly output=${root}/results/pretraining/abf_paircontact_one_gpu_systems_canary_recovery_v2
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}/checkpoint.pt"
test ! -e "${output}/RESULT.json"

printf '%s  %s\n' \
  5e003de23855b0c6b61352b855466c7b6310fe7e6a1fc9a1a10d0504aac359c6 "${candidate}/scripts/run_one_gpu_systems_canary.py" \
  739c93b9af5a189f028831784040d7c1391a58b8af58458a7f0c4e18471cf539 "${candidate}/src/antibody_pretraining/stage_a1_objective.py" \
  c260abd2f993056f3b45b6b62ffcf21ebdb3dd21b05ec03e46702a4163939cea "${protocol}" \
  8792020b3117c2cd98507a745f54217380f41e539b3ab6e9240250cc82145230 "${failure}" \
  | sha256sum --check --strict

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH="${candidate}/src" "${python_bin}" \
  "${candidate}/scripts/run_one_gpu_systems_canary.py" \
  --protocol "${protocol}" \
  --expected-protocol-sha256 c260abd2f993056f3b45b6b62ffcf21ebdb3dd21b05ec03e46702a4163939cea \
  --expected-objective-sha256 739c93b9af5a189f028831784040d7c1391a58b8af58458a7f0c4e18471cf539 \
  --checkpoint "${output}/checkpoint.pt" \
  --result "${output}/RESULT.json"
