#!/usr/bin/env bash
#SBATCH -J abpairfm-string-high700-collapse
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-collapse-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-collapse-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly ASSETS=${ROOT}/cluster_assets_high700_v1
readonly EDGES=${ROOT}/physical_combined700_all_channels_v2.tsv.zst
readonly EDGE_RECEIPT=${ROOT}/physical_combined700_all_channels_v2.receipt.json
readonly CLUSTERS=${ASSETS}/clusters.tsv.zst
readonly HITS=${ASSETS}/development_hits.tsv
readonly CLUSTER_RECEIPT=${ASSETS}/cluster_assets.receipt.json
readonly PROTOCOL=${PROJECT}/protocols/abpairfm_3b_scale_expansion_v2.json
readonly OUTPUT=${ROOT}/physical_combined700_cluster50_devclean_scale_v1.tsv.zst
readonly RECEIPT=${ROOT}/physical_combined700_cluster50_devclean_scale_v1.receipt.json
readonly WORK=$(mktemp -d "${ROOT}/.high700-collapse-v2-${SLURM_JOB_ID}.XXXXXX")
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden STRING collapse allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
for path in "${EDGES}" "${EDGE_RECEIPT}" "${CLUSTERS}" "${HITS}" "${CLUSTER_RECEIPT}" "${PROTOCOL}"; do
  test -f "${path}"
done
for path in "${OUTPUT}" "${RECEIPT}"; do
  test ! -e "${path}"
done
cleanup_collapse() {
  [[ ! -d "${WORK}" ]] || rm -r -- "${WORK}"
}
trap cleanup_collapse EXIT

export LC_ALL=C
"${PYTHON}" "${PROJECT}/scripts/collapse_string_v12_high_confidence_scale_v2.py" decorate \
  --edges "${EDGES}" \
  --clusters "${CLUSTERS}" \
  --development-hits "${HITS}" \
  --stats "${WORK}/decorate.stats" | \
  sort --parallel=6 -S 32G -T "${WORK}" -t $'\t' \
    -k1,1 -k2,2 -k3,3nr -k4,4nr -k5,5 -k6,6 -k7,14 | \
  awk -F $'\t' -v OFS=$'\t' -v stats="${WORK}/collapse.stats" '
    BEGIN {
      print "cluster1", "cluster2", "protein1", "protein2", "homology", "experiments", "experiments_transferred", "database", "database_transferred", "textmining", "textmining_transferred", "combined_score"
    }
    $1 != previous1 || $2 != previous2 {
      print $1, $2, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
      output_rows++
      previous1 = $1
      previous2 = $2
    }
    END { print "output_cluster_pairs=" output_rows + 0 > stats }
  ' | zstd -q -T6 -3 --check -o "${WORK}/output.tsv.zst"

grep -E '^(development_excluded_edges|eligible_nonself_edges|missing_endpoint_memberships|self_cluster_edges_removed|source_edges)=' \
  "${WORK}/decorate.stats" > "${WORK}/final.stats"
cat "${WORK}/collapse.stats" >> "${WORK}/final.stats"
zstd --test "${WORK}/output.tsv.zst"
chmod 0444 "${WORK}/output.tsv.zst"
"${PYTHON}" "${PROJECT}/scripts/collapse_string_v12_high_confidence_scale_v2.py" finalize \
  --edges "${EDGES}" \
  --edge-receipt "${EDGE_RECEIPT}" \
  --clusters "${CLUSTERS}" \
  --development-hits "${HITS}" \
  --cluster-receipt "${CLUSTER_RECEIPT}" \
  --protocol "${PROTOCOL}" \
  --decorate-stats "${WORK}/decorate.stats" \
  --stats "${WORK}/final.stats" \
  --output "${WORK}/output.tsv.zst" \
  --output-name "$(basename "${OUTPUT}")" \
  --receipt "${WORK}/receipt.json"
chmod 0444 "${WORK}/receipt.json"
ln "${WORK}/output.tsv.zst" "${OUTPUT}"
ln "${WORK}/receipt.json" "${RECEIPT}"
