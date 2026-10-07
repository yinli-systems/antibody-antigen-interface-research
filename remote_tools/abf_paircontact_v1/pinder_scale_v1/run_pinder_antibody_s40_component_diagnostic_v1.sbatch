#!/usr/bin/env bash
#SBATCH -J pinder-ab-s40-diag
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/development/pinder_antibody_s40_v1/diagnostic-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/development/pinder_antibody_s40_v1/diagnostic-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/results/development/pinder_antibody_s40_v1/diagnostic_source_v1
readonly S40=${ROOT}/results/development/pinder_antibody_s40_v1
readonly S2=${ROOT}/results/development/pinder_antibody_s2_v1/assets
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test ! -e "${S40}/PINDER_ANTIBODY_S40_COMPONENT_DIAGNOSTIC_V1.json"
"${PYTHON}" "${SOURCE}/diagnose_pinder_antibody_s40_components_v1.py" \
  --canonical-manifest "${S2}/PINDER_ANTIBODY_S2_CANONICAL_V1.csv" \
  --canonical-manifest-sha256 624612079838536934c81fa3918486e322c16caed19fbba70b6f8b33adc5fb6c \
  --relation-result "${S40}/PINDER_ANTIBODY_S40_RELATION_RESULT_V1.json" \
  --relation-result-sha256 aceaf0b53b60f4c124c3c3b26e57082477678ccebf3b0c8cb939b5692b1f6b5d \
  --vh-relation "${S40}/vh.identity40_coverage80.relation.tsv" \
  --vl-relation "${S40}/vl.identity40_coverage80.relation.tsv" \
  --concatenated-antibody-relation "${S40}/concatenated_antibody.identity40_coverage80.relation.tsv" \
  --antigen-relation "${S40}/antigen.identity40_coverage80.relation.tsv" \
  --output "${S40}/PINDER_ANTIBODY_S40_COMPONENT_DIAGNOSTIC_V1.json"
