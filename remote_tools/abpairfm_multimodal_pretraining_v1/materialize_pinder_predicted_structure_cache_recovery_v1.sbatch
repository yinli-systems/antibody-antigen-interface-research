#!/usr/bin/env bash
#SBATCH -J abpairfm-mm-structure-cache-recovery
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/cache-recovery-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/cache-recovery-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abpairfm_multimodal_pretraining_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_multimodal_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly MATERIALIZER=${SOURCE}/materialize_pinder_predicted_structure_cache_v1.py
readonly MATERIALIZER_SHA=12f8b8c1d15ee04c7e1965f9c489787012c6c3aa5c3bc8b5346ce313bbe1b57e
readonly INVENTORY=${OUTPUT}/PINDER_MULTIMODAL_BFULL_INVENTORY_V1.tsv.gz
readonly INVENTORY_SHA=914fee292921061b68822f6bde7cd49c0e13802b99c05e0b73558a04277e9a11
readonly ARCHIVE=${ROOT}/data/abpairfm_3b_v1/raw/pinder_2024_02/pdbs.zip
readonly ARCHIVE_SHA=125bc49d515803576c19123afdc72f4f59e84d64522866f99d089c53b7d27912
readonly ARCHIVE_SIZE=168884420877

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${MATERIALIZER}" | cut -d' ' -f1)" = "${MATERIALIZER_SHA}"
test "$(sha256sum "${INVENTORY}" | cut -d' ' -f1)" = "${INVENTORY_SHA}"
test ! -e "${OUTPUT}/predicted_structure_cache_v1"
test ! -e "${OUTPUT}/PINDER_PREDICTED_STRUCTURE_CACHE_RESULT_V1.json"

"${PYTHON}" "${MATERIALIZER}" \
  --inventory "${INVENTORY}" \
  --inventory-sha256 "${INVENTORY_SHA}" \
  --structure-archive "${ARCHIVE}" \
  --structure-archive-sha256 "${ARCHIVE_SHA}" \
  --structure-archive-size "${ARCHIVE_SIZE}" \
  --expected-structure-members 19923 \
  --output-root "${OUTPUT}/predicted_structure_cache_v1" \
  --output-result "${OUTPUT}/PINDER_PREDICTED_STRUCTURE_CACHE_RESULT_V1.json"
