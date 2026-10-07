#!/usr/bin/env bash
#SBATCH --job-name=abf30b-beyond-marginals-analysis
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/beyond-marginals-analysis-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/beyond-marginals-analysis-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly tool_root=${root}/remote_tools/abf30b_framecontact_v1
readonly authority=${campaign}/authority/beyond_marginals_v1
readonly result_root=${campaign}/results/beyond_marginals_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

"${python_bin}" "${tool_root}/analyze_abf30b_beyond_marginals_v1.py" \
  --protocol "${authority}/ABF30B_BEYOND_MARGINALS_PROTOCOL_V1.json" \
  --protocol-receipt "${authority}/ABF30B_BEYOND_MARGINALS_PROTOCOL_V1_RECEIPT.json" \
  --component-root "${result_root}" \
  --output "${result_root}/ABF30B_BEYOND_MARGINALS_ANALYSIS_V1.json" \
  --receipt-output "${result_root}/ABF30B_BEYOND_MARGINALS_ANALYSIS_V1_RECEIPT.json"
