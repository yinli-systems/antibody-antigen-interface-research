#!/bin/bash
#SBATCH --job-name=frontier-real-gate
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/tmp/frontier-real-gate-%j.out
#SBATCH --error=/tmp/frontier-real-gate-%j.err

set -euo pipefail

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify sealed source}
CACHE_ROOT=${FRONTIER_CACHE_ROOT:?FRONTIER_CACHE_ROOT must identify the isolated cache}
AUTHORIZATION=${FRONTIER_AUTHORIZATION:?FRONTIER_AUTHORIZATION is required}
EXPECTED_AUTHORIZATION_SHA256=${FRONTIER_EXPECTED_AUTHORIZATION_MANIFEST_SHA256:?FRONTIER_EXPECTED_AUTHORIZATION_MANIFEST_SHA256 is required}
EXPECTED_CACHE_SHA256=${FRONTIER_EXPECTED_CACHE_MANIFEST_SHA256:?FRONTIER_EXPECTED_CACHE_MANIFEST_SHA256 is required}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must be create-only}
MODE=${FRONTIER_GATE_MODE:-train}
RUN_PROFILE=${FRONTIER_RUN_PROFILE:-engineering_diagnostic}
NPROC=${FRONTIER_NPROC:-8}
SEED=${FRONTIER_SEED:-17}
DIAGNOSTIC_SYNC_BOUNDARIES=${FRONTIER_DIAGNOSTIC_SYNC_BOUNDARIES:-false}
RESUME_CHECKPOINT_ROOT=${FRONTIER_RESUME_CHECKPOINT_ROOT:-}
RESUME_GENERATION=${FRONTIER_RESUME_GENERATION:-}
RESUME_RECEIPT_SHA256=${FRONTIER_EXPECTED_RESUME_RECEIPT_SHA256:-}
RUN_ID=${FRONTIER_RUN_ID:-model}

if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIER_NPROC must be a positive integer" >&2
  exit 2
fi

if [[ "$RUN_PROFILE" == "contract_d0" ]]; then
  SCHEDULER=${FRONTIER_SCHEDULER:-}
  if [[ -z "$SCHEDULER" ]]; then
    echo "FRONTIER_SCHEDULER must be explicitly set for contract_d0" >&2
    exit 2
  fi
  if (( 1024 % NPROC != 0 )); then
    echo "contract_d0 requires a rank count that divides global batch 1024" >&2
    exit 2
  fi
  EPOCHS=${FRONTIER_EPOCHS:-58}
  LOCAL_BATCH_SIZE=${FRONTIER_LOCAL_BATCH_SIZE:-$((1024 / NPROC))}
  MAX_STEPS=${FRONTIER_MAX_STEPS:-}
  CHECKPOINT_EVERY=${FRONTIER_CHECKPOINT_EVERY:-0}
  PRECISION=${FRONTIER_PRECISION:-bf16}
  GATHER_GRADIENT_BACKEND=${FRONTIER_GATHER_GRADIENT_BACKEND:-local_slice_ddp_exact}
  WARMUP_FRACTION=${FRONTIER_WARMUP_FRACTION:-0.0}
  RESTART_PERIOD_EPOCHS=${FRONTIER_RESTART_PERIOD_EPOCHS:-10}
  RESTART_PERIOD_MULTIPLIER=${FRONTIER_RESTART_PERIOD_MULTIPLIER:-2}
  DDP_INIT_SYNC=${FRONTIER_DDP_INIT_SYNC:-true}
  DDP_STATIC_GRAPH=${FRONTIER_DDP_STATIC_GRAPH:-true}
  DDP_GRADIENT_AS_BUCKET_VIEW=${FRONTIER_DDP_GRADIENT_AS_BUCKET_VIEW:-true}
  NUM_WORKERS=${FRONTIER_NUM_WORKERS:-2}
  MEMORY_CACHE_SIZE=${FRONTIER_MEMORY_CACHE_SIZE:-1830}
  PREFETCH_FACTOR=${FRONTIER_PREFETCH_FACTOR:-1}
  STAGE_CACHE=${FRONTIER_STAGE_CACHE:-true}
elif [[ "$RUN_PROFILE" == "engineering_diagnostic" ]]; then
  SCHEDULER=${FRONTIER_SCHEDULER:-linear_warmup_cosine}
  EPOCHS=${FRONTIER_EPOCHS:-1}
  LOCAL_BATCH_SIZE=${FRONTIER_LOCAL_BATCH_SIZE:-8}
  MAX_STEPS=${FRONTIER_MAX_STEPS:-2}
  CHECKPOINT_EVERY=${FRONTIER_CHECKPOINT_EVERY:-1}
  PRECISION=${FRONTIER_PRECISION:-bf16}
  GATHER_GRADIENT_BACKEND=${FRONTIER_GATHER_GRADIENT_BACKEND:-local_slice_ddp_exact}
  WARMUP_FRACTION=${FRONTIER_WARMUP_FRACTION:-0.05}
  RESTART_PERIOD_EPOCHS=${FRONTIER_RESTART_PERIOD_EPOCHS:-10}
  RESTART_PERIOD_MULTIPLIER=${FRONTIER_RESTART_PERIOD_MULTIPLIER:-2}
  DDP_INIT_SYNC=${FRONTIER_DDP_INIT_SYNC:-true}
  DDP_STATIC_GRAPH=${FRONTIER_DDP_STATIC_GRAPH:-false}
  DDP_GRADIENT_AS_BUCKET_VIEW=${FRONTIER_DDP_GRADIENT_AS_BUCKET_VIEW:-false}
  NUM_WORKERS=${FRONTIER_NUM_WORKERS:-2}
  MEMORY_CACHE_SIZE=${FRONTIER_MEMORY_CACHE_SIZE:-0}
  PREFETCH_FACTOR=${FRONTIER_PREFETCH_FACTOR:-2}
  STAGE_CACHE=${FRONTIER_STAGE_CACHE:-false}
else
  echo "FRONTIER_RUN_PROFILE is unsupported" >&2
  exit 2
fi

RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID-$MODE-nproc$NPROC
MODEL_RUN_ROOT=$RUN_ROOT/$RUN_ID
LOCAL_BASE=${SLURM_TMPDIR:-/tmp}

if [[ "$MODE" != "verify" && "$MODE" != "train" \
  && "$MODE" != "contract_preflight" ]]; then
  echo "FRONTIER_GATE_MODE must be verify, train, or contract_preflight" >&2
  exit 2
fi
if [[ "$SCHEDULER" != "linear_warmup_cosine" \
  && "$SCHEDULER" != "cosine_annealing_warm_restarts" ]]; then
  echo "FRONTIER_SCHEDULER is unsupported" >&2
  exit 2
fi
if [[ "$RUN_PROFILE" == "contract_d0" ]]; then
  if [[ "$MODE" != "train" && "$MODE" != "contract_preflight" ]]; then
    echo "contract_d0 supports only train or contract_preflight mode" >&2
    exit 2
  fi
  if [[ "$SCHEDULER" != "cosine_annealing_warm_restarts" ]]; then
    echo "contract_d0 requires cosine_annealing_warm_restarts" >&2
    exit 2
  fi
  if [[ -n "$MAX_STEPS" ]]; then
    echo "contract_d0 forbids FRONTIER_MAX_STEPS truncation" >&2
    exit 2
  fi
  if [[ ! "$LOCAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || (( LOCAL_BATCH_SIZE * NPROC != 1024 )); then
    echo "contract_d0 requires local batch times ranks to equal 1024" >&2
    exit 2
  fi
fi
if [[ "$GATHER_GRADIENT_BACKEND" != "local_slice_ddp_exact" \
  && "$GATHER_GRADIENT_BACKEND" != "exact_all_reduce" \
  && "$GATHER_GRADIENT_BACKEND" != "legacy_autograd" ]]; then
  echo "FRONTIER_GATHER_GRADIENT_BACKEND is unsupported" >&2
  exit 2
fi
if [[ -n "$RESUME_CHECKPOINT_ROOT" && -z "$RESUME_GENERATION" ]] \
  || [[ -z "$RESUME_CHECKPOINT_ROOT" && -n "$RESUME_GENERATION" ]]; then
  echo "Resume checkpoint root and generation must be supplied together" >&2
  exit 2
fi
if [[ -n "$RESUME_GENERATION" && ! "$RESUME_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Resume requires FRONTIER_EXPECTED_RESUME_RECEIPT_SHA256" >&2
  exit 2
fi
if [[ -z "$RESUME_GENERATION" && -n "$RESUME_RECEIPT_SHA256" ]]; then
  echo "Resume receipt SHA-256 is invalid without a resume generation" >&2
  exit 2
fi
if [[ ! "$EXPECTED_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$EXPECTED_CACHE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "External data anchors must be lowercase SHA-256 digests" >&2
  exit 2
fi
if [[ "$DDP_INIT_SYNC" != "true" && "$DDP_INIT_SYNC" != "false" ]]; then
  echo "FRONTIER_DDP_INIT_SYNC must be true or false" >&2
  exit 2
fi
for value in "$DDP_STATIC_GRAPH" "$DDP_GRADIENT_AS_BUCKET_VIEW" \
  "$DIAGNOSTIC_SYNC_BOUNDARIES"; do
  if [[ "$value" != "true" && "$value" != "false" ]]; then
    echo "DDP optimization and diagnostic flags must be true or false" >&2
    exit 2
  fi
done
if [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ ]] \
  || [[ ! "$MEMORY_CACHE_SIZE" =~ ^[0-9]+$ ]] \
  || [[ ! "$PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ ]]; then
  echo "Worker, memory-cache, and prefetch settings must be non-negative integers" >&2
  exit 2
fi
if [[ "$STAGE_CACHE" != "true" && "$STAGE_CACHE" != "false" ]]; then
  echo "FRONTIER_STAGE_CACHE must be true or false" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" || -e "$MODEL_RUN_ROOT" ]]; then
  echo "Refusing to overwrite an existing run root: $RUN_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_ROOT/scripts/frontier_real_data_gate.py" ]]; then
  echo "Sealed source is missing frontier_real_data_gate.py" >&2
  exit 2
fi
if [[ ! -d "$CACHE_ROOT" || ! -f "$AUTHORIZATION" ]]; then
  echo "Real-data cache or authorization is missing" >&2
  exit 2
fi

VISIBLE_GPU_COUNT=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
if [[ ! "$VISIBLE_GPU_COUNT" =~ ^[0-9]+$ ]] || (( VISIBLE_GPU_COUNT > 8 )); then
  echo "Visible CUDA device count is invalid for one reviewed node" >&2
  exit 2
fi
if (( NPROC > VISIBLE_GPU_COUNT )); then
  echo "Requested $NPROC ranks but only $VISIBLE_GPU_COUNT GPUs are visible" >&2
  exit 2
fi

LOCAL_ROOT=$(mktemp -d "$LOCAL_BASE/frontier-real-gate-$SLURM_JOB_ID.XXXXXX")
MONITOR_PID=
stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=
  fi
}
cleanup() {
  local status=$?
  stop_monitor
  chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT"
  return "$status"
}
trap cleanup EXIT
mkdir -p "$OUTPUT_ROOT" "$LOCAL_ROOT/tmp"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src
export TMPDIR=$LOCAL_ROOT/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

mkdir "$RUN_ROOT"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 2 > "$RUN_ROOT/gpu.csv" &
MONITOR_PID=$!

ORIGIN_CACHE_ROOT=$CACHE_ROOT
ORIGIN_AUTHORIZATION=$AUTHORIZATION
STAGE_COPY_SECONDS=0
STAGE_REQUIRED_KIB=0
STAGE_AVAILABLE_KIB=0
{
  if [[ "$STAGE_CACHE" == "true" ]]; then
    STAGED_DATA_ROOT=$LOCAL_ROOT/staged-data
    STAGED_CACHE_ROOT=$STAGED_DATA_ROOT/fold_cache
    STAGED_AUTHORIZATION=$STAGED_DATA_ROOT/AUTHORIZED_FOLD_INVENTORY.json
    mkdir -p "$STAGED_CACHE_ROOT"
    STAGE_REQUIRED_KIB=$(
      du -sk "$ORIGIN_CACHE_ROOT" "$ORIGIN_AUTHORIZATION" \
        | awk '{total += $1} END {print total + 0}'
    )
    STAGE_AVAILABLE_KIB=$(
      df -Pk "$STAGED_DATA_ROOT" | awk 'NR == 2 {print $4}'
    )
    if (( STAGE_AVAILABLE_KIB < STAGE_REQUIRED_KIB + 1048576 )); then
      echo "Insufficient node-local capacity for fail-closed cache staging" >&2
      exit 2
    fi
    stage_started=$SECONDS
    cp -a --reflink=auto "$ORIGIN_CACHE_ROOT"/. "$STAGED_CACHE_ROOT"/
    cp -p "$ORIGIN_AUTHORIZATION" "$STAGED_AUTHORIZATION"
    STAGE_COPY_SECONDS=$((SECONDS - stage_started))
    if [[ "$(
      sha256sum "$ORIGIN_CACHE_ROOT/FOLD_LOCAL_CACHE_MANIFEST.json" \
        | awk '{print $1}'
    )" != "$(
      sha256sum "$STAGED_CACHE_ROOT/FOLD_LOCAL_CACHE_MANIFEST.json" \
        | awk '{print $1}'
    )" ]]; then
      echo "Staged cache-manifest digest differs from its source" >&2
      exit 2
    fi
    if [[ "$(
      sha256sum "$ORIGIN_AUTHORIZATION" | awk '{print $1}'
    )" != "$(
      sha256sum "$STAGED_AUTHORIZATION" | awk '{print $1}'
    )" ]]; then
      echo "Staged authorization digest differs from its source" >&2
      exit 2
    fi
    CACHE_ROOT=$STAGED_CACHE_ROOT
    AUTHORIZATION=$STAGED_AUTHORIZATION
    echo "Cache staging completed with manifest and authorization equality"
  else
    echo "Cache staging disabled by the selected run profile"
  fi
} > "$RUN_ROOT/staging.log" 2>&1

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "source_root=$SOURCE_ROOT"
  echo "origin_cache_root=$ORIGIN_CACHE_ROOT"
  echo "origin_authorization=$ORIGIN_AUTHORIZATION"
  echo "effective_cache_root=$CACHE_ROOT"
  echo "effective_authorization=$AUTHORIZATION"
  echo "local_base=$LOCAL_BASE"
  echo "local_root=$LOCAL_ROOT"
  echo "stage_cache=$STAGE_CACHE"
  echo "stage_copy_seconds=$STAGE_COPY_SECONDS"
  echo "stage_required_kib=$STAGE_REQUIRED_KIB"
  echo "stage_available_kib=$STAGE_AVAILABLE_KIB"
  echo "run_root=$RUN_ROOT"
  echo "model_run_root=$MODEL_RUN_ROOT"
  echo "mode=$MODE"
  echo "run_profile=$RUN_PROFILE"
  echo "nproc=$NPROC"
  echo "epochs=$EPOCHS"
  echo "local_batch_size=$LOCAL_BATCH_SIZE"
  echo "scheduler=$SCHEDULER"
  echo "warmup_fraction=$WARMUP_FRACTION"
  echo "restart_period_epochs=$RESTART_PERIOD_EPOCHS"
  echo "restart_period_multiplier=$RESTART_PERIOD_MULTIPLIER"
  echo "max_steps=${MAX_STEPS:-<none>}"
  echo "num_workers=$NUM_WORKERS"
  echo "memory_cache_size=$MEMORY_CACHE_SIZE"
  echo "prefetch_factor=$PREFETCH_FACTOR"
  echo "gather_gradient_backend=$GATHER_GRADIENT_BACKEND"
  echo "ddp_init_sync=$DDP_INIT_SYNC"
  echo "ddp_static_graph=$DDP_STATIC_GRAPH"
  echo "ddp_gradient_as_bucket_view=$DDP_GRADIENT_AS_BUCKET_VIEW"
  echo "diagnostic_sync_boundaries=$DIAGNOSTIC_SYNC_BOUNDARIES"
  echo "nccl_cumem_host_enable=${NCCL_CUMEM_HOST_ENABLE:-<runtime-default>}"
  echo "visible_gpu_count=$VISIBLE_GPU_COUNT"
  date --iso-8601=seconds
  "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + torch.version.cuda); print("nccl=" + str(torch.cuda.nccl.version()))'
  nvidia-smi
} > "$RUN_ROOT/environment.txt"

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  "--nproc-per-node=$NPROC"
  "$SOURCE_ROOT/scripts/frontier_real_data_gate.py"
  --cache-root "$CACHE_ROOT"
  --authorization-manifest "$AUTHORIZATION"
  --expected-authorization-manifest-sha256 "$EXPECTED_AUTHORIZATION_SHA256"
  --expected-cache-manifest-sha256 "$EXPECTED_CACHE_SHA256"
  --output-root "$RUN_ROOT"
  --run-id "$RUN_ID"
  --run-profile "$RUN_PROFILE"
  --epochs "$EPOCHS"
  --local-batch-size "$LOCAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --memory-cache-size "$MEMORY_CACHE_SIZE"
  --prefetch-factor "$PREFETCH_FACTOR"
  --seed "$SEED"
  --precision "$PRECISION"
  --scheduler "$SCHEDULER"
  --warmup-fraction "$WARMUP_FRACTION"
  --restart-period-epochs "$RESTART_PERIOD_EPOCHS"
  --restart-period-multiplier "$RESTART_PERIOD_MULTIPLIER"
  --gather-gradient-backend "$GATHER_GRADIENT_BACKEND"
  --checkpoint-every-steps "$CHECKPOINT_EVERY"
  --device cuda
)
if [[ -n "$MAX_STEPS" ]]; then
  COMMAND+=(--max-steps "$MAX_STEPS")
fi
if [[ "$MODE" == "verify" ]]; then
  COMMAND+=(--verify-only)
fi
if [[ "$MODE" == "contract_preflight" ]]; then
  COMMAND+=(--contract-preflight-only)
fi
if [[ "$DDP_INIT_SYNC" == "false" ]]; then
  COMMAND+=(--no-ddp-init-sync)
fi
if [[ "$DDP_STATIC_GRAPH" == "true" ]]; then
  COMMAND+=(--ddp-static-graph)
else
  COMMAND+=(--no-ddp-static-graph)
fi
if [[ "$DDP_GRADIENT_AS_BUCKET_VIEW" == "true" ]]; then
  COMMAND+=(--ddp-gradient-as-bucket-view)
else
  COMMAND+=(--no-ddp-gradient-as-bucket-view)
fi
if [[ "$DIAGNOSTIC_SYNC_BOUNDARIES" == "true" ]]; then
  COMMAND+=(--diagnostic-sync-boundaries)
fi
if [[ -n "$RESUME_CHECKPOINT_ROOT" ]]; then
  COMMAND+=(
    --resume-checkpoint-root "$RESUME_CHECKPOINT_ROOT"
    --resume-generation "$RESUME_GENERATION"
    --expected-resume-receipt-sha256 "$RESUME_RECEIPT_SHA256"
  )
fi

"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/driver.log"

date --iso-8601=seconds >> "$RUN_ROOT/environment.txt"
stop_monitor
find "$RUN_ROOT" -type f ! -name artifact_sha256.txt -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > "$RUN_ROOT/artifact_sha256.txt"
chmod -R a-w "$RUN_ROOT"
