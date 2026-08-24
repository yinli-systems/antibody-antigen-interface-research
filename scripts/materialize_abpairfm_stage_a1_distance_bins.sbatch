#!/usr/bin/env bash
#SBATCH -J abpairfm-a1-distance-bins
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-a1-distance-bins-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-a1-distance-bins-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=${ABPAIRFM_STAGE_A1_DISTANCE_SOURCE:?missing immutable source root}
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_distance_bins_v1

mkdir -p "${OUTPUT_ROOT}"
test ! -e "${OUTPUT_ROOT}/STAGE_A1_DISTANCE_BIN_MANIFEST.jsonl.zst"
test ! -e "${OUTPUT_ROOT}/STAGE_A1_DISTANCE_BIN_MANIFEST_RECEIPT.json"

"${PYTHON}" "${SOURCE}/materialize_abpairfm_stage_a1_distance_bins.py" \
  --pilot-manifest /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a1_pilot_v4/STAGE_A1_PILOT_MANIFEST.jsonl.zst \
  --expected-pilot-manifest-sha256 45e0ff7deb6a1e1ef54fb5efba1c82bd8d918e34061ad4c4201e903097bccaed \
  --ppiref-manifest /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/ppiref50k_2024_01/interfaces_devclean_pindernovel_v1.jsonl.zst \
  --expected-ppiref-manifest-sha256 db231ea9a668e5d216db7137dc8224333f58eb040a4ff6270b806e37afc1ac89 \
  --archive /ssd/scxi253/model_deployment/data/pairscale_registered_raw_b6b86f1/ppiref_zenodo_14845086/ppi_6A.zip \
  --expected-archive-sha256 dece33d55f1098c699fdddeac3e1cb38fa23c65f9eac36fec7d339190ddc8cd3 \
  --output "${OUTPUT_ROOT}/STAGE_A1_DISTANCE_BIN_MANIFEST.jsonl.zst" \
  --receipt "${OUTPUT_ROOT}/STAGE_A1_DISTANCE_BIN_MANIFEST_RECEIPT.json"
