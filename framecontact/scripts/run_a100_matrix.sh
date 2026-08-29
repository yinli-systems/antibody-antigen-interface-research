#!/usr/bin/env bash
set -euo pipefail

: "${GEOM_REPOSITORY_ROOT:?Set GEOM_REPOSITORY_ROOT to this repository on AB03}"
: "${GEOM_CACHE_ROOT:?Set GEOM_CACHE_ROOT to the authenticated structure cache}"
: "${GEOM_EMBEDDING_ROOT:?Set GEOM_EMBEDDING_ROOT to frozen ESM embeddings}"
: "${GEOM_LABEL_ROOT:?Set GEOM_LABEL_ROOT to development labels}"
: "${GEOM_RUN_ROOT:?Set GEOM_RUN_ROOT to a new durable output directory}"

python_bin=${GEOM_PYTHON_BIN:-python3}
epochs=${GEOM_EPOCHS:-20}
seeds=${GEOM_SEEDS:-"17 42 73"}
experiments=${GEOM_EXPERIMENTS:-"sequence_dense sequence_local spatial_topology distance distance_direction full_pose"}
gpu_name_regex=${GEOM_GPU_NAME_REGEX:-A100}
validate=${GEOM_VALIDATE:-1}
aggregate=${GEOM_AGGREGATE:-1}
worker_tag=${GEOM_WORKER_TAG:-worker}

export PYTHONPATH="${GEOM_REPOSITORY_ROOT}:${GEOM_REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

mkdir -p "${GEOM_RUN_ROOT}"
manifest_sha256=$(sha256sum "${GEOM_CACHE_ROOT}/STRUCTURE_CACHE_MANIFEST.json" | awk '{print $1}')

if [[ "${validate}" == "1" ]]; then
  "${python_bin}" -m pytest -q "${GEOM_REPOSITORY_ROOT}/tests"
  smoke_path="${GEOM_RUN_ROOT}/A100_SMOKE.${worker_tag}.json"
  if [[ -e "${smoke_path}" ]]; then
    echo "refusing to overwrite smoke receipt: ${smoke_path}" >&2
    exit 1
  fi
  "${python_bin}" "${GEOM_REPOSITORY_ROOT}/scripts/smoke_a100.py" \
    --gpu-name-regex "${gpu_name_regex}" \
    | tee "${smoke_path}"
elif [[ "${validate}" != "0" ]]; then
  echo "GEOM_VALIDATE must be 0 or 1" >&2
  exit 1
fi

for seed in ${seeds}; do
  for experiment in ${experiments}; do
    output_root="${GEOM_RUN_ROOT}/seed_${seed}/${experiment}"
    if [[ -f "${output_root}/RUN_RECEIPT.json" && -f "${output_root}/SABDAB2_SELF_ATTENTION_ABLATION.json" ]]; then
      echo "skip complete seed=${seed} experiment=${experiment}"
      continue
    fi
    if [[ -e "${output_root}" ]]; then
      echo "refusing incomplete existing output: ${output_root}" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${output_root}")"
    "${python_bin}" "${GEOM_REPOSITORY_ROOT}/scripts/train_sabdab2_structure_ablation.py" \
      --cache-root "${GEOM_CACHE_ROOT}" \
      --embedding-root "${GEOM_EMBEDDING_ROOT}" \
      --label-root "${GEOM_LABEL_ROOT}" \
      --output-root "${output_root}" \
      --expected-cache-manifest-sha256 "${manifest_sha256}" \
      --experiments "${experiment}" \
      --seed "${seed}" \
      --epochs "${epochs}" \
      --batch-size 4 \
      --workers 4 \
      --max-antigen-length 384 \
      --d-model 64 \
      --num-heads 4 \
      --num-layers 1 \
      --num-rbf 16 \
      --neighbor-k 32 \
      --neighbor-radius 16 \
      --sequence-window 32 \
      --dropout 0.1 \
      --learning-rate 0.0003 \
      --weight-decay 0.0001 \
      --precision bf16 \
      --device cuda
  done
done

if [[ "${aggregate}" == "1" ]]; then
  summary_path="${GEOM_RUN_ROOT}/MULTISEED_ABLATION_SUMMARY.json"
  if [[ -e "${summary_path}" ]]; then
    echo "refusing to overwrite summary: ${summary_path}" >&2
    exit 1
  fi
  "${python_bin}" "${GEOM_REPOSITORY_ROOT}/scripts/aggregate_multiseed_ablation.py" \
    --run-root "${GEOM_RUN_ROOT}" \
    --output "${summary_path}" \
    --seeds ${seeds} \
    --experiments ${experiments} \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 20260729
elif [[ "${aggregate}" != "0" ]]; then
  echo "GEOM_AGGREGATE must be 0 or 1" >&2
  exit 1
fi
