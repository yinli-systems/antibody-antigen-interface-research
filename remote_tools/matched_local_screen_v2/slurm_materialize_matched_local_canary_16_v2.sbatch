#!/usr/bin/env bash
#SBATCH --job-name=fc-ml-v2c16
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_matched_local_canary_16_v2/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_matched_local_canary_16_v2/logs/%x-%j.err

set -euo pipefail

root=/ssd/scxi253/model_deployment
source_root=${root}/data/framecontact_20260730/sabdab_source_matched_local_v2
data_root=${root}/data/framecontact_20260730
python_bin=${root}/envs/contact/bin/python3.10

cd ${source_root}
export PYTHONPATH=${source_root}/src
${python_bin} scripts/materialize_sabdab2_matched_local_decoys.py \
  --raw-archive ${root}/data/pairscale_registered_raw_b6b86f1/sabdab2_ml_zenodo_20083995/splits.tar.gz \
  --raw-archive-sha256 54e3b9cdae5f5ff57a4210866f87e86686203556eea6a033200ff55c0f81e2b7 \
  --metadata-sha256 01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb \
  --canonical-archive ${root}/data/sabdab2-baseline-outsourcing/sabdab2-canonical-20260721.tar.gz \
  --canonical-archive-sha256 9184da63526662066243114ab5597588497df460b805214d1dd3b35f8bf056dd \
  --split ${source_root}/protocols/matched-local-canary-16-v2.csv \
  --split-sha256 d800f896440b51d9b32b1c57ea6bd39efd4a0ad0a2e8f97d5ccdfe858218cc38 \
  --contact-root ${data_root}/sabdab2/contact_cohort_v2 \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --structure-root ${data_root}/sabdab2/structure_cohort_v2 \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --output-root ${data_root}/sabdab2/matched_local_canary_16_v2_dirichlet05 \
  --workers 6 \
  --composition-pseudocount 0.5 \
  --minimum-strict-pair-coverage 0.95
