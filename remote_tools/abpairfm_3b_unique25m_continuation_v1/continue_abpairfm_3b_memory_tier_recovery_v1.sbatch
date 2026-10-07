#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-a0-successor-mem
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-successor-mem-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-successor-mem-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment/abpairfm_3b_unique25m_continuation_v1
readonly BASE=${ROOT}/continue_abpairfm_3b_unique25m_v1.sbatch
readonly EXPECTED_BASE_SHA256=29a187a93fc78d78f13883cd863a704b821a5a587a21e03ce96decf5a2eb30d8
readonly EXPECTED_GENERATED_SHA256=9d72116253d6bfbd8977da6c86e49d0805a77dd63ce081de4539f0a635200dfe
readonly GENERATED=$(mktemp /tmp/abpairfm-3b-successor-memory-tier.XXXXXX.sbatch)
trap 'rm -f "${GENERATED}"' EXIT

if [[ "$(sha256sum "${BASE}" | awk '{print $1}')" != "${EXPECTED_BASE_SHA256}" ]]; then
  echo "Frozen base continuation source differs" >&2
  exit 2
fi

sed \
  -e 's/readonly EXCLUDED_TRAINING_NODE=wqd10naf05g3$/readonly EXCLUDED_TRAINING_NODE=wqd10naf05g3,wqd10naf05g5/' \
  -e 's/continue_abpairfm_3b_unique25m_v1\.sbatch/continue_abpairfm_3b_memory_tier_recovery_v1.sbatch/g' \
  "${BASE}" > "${GENERATED}"

if [[ "$(sha256sum "${GENERATED}" | awk '{print $1}')" != "${EXPECTED_GENERATED_SHA256}" ]]; then
  echo "Generated memory-tier continuation source differs" >&2
  exit 2
fi

bash "${GENERATED}"
