#!/bin/bash
#SBATCH --job-name=score-dissoc
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
set -euo pipefail
: "${PROGRAM_ROOT:?}" "${ATTEMPT_ROOT:?}" "${PLAN_SHA256:?}"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/ssd/scxi253/repr_extract/venv/bin/python3
PLAN="$ATTEMPT_ROOT/PLAN.json"
cd "$PROGRAM_ROOT/source"
readarray -t TASK < <("$PY" - "$PLAN" "$PLAN_SHA256" "${SLURM_ARRAY_TASK_ID:?}" <<'PYCODE'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);expected=sys.argv[2];idx=int(sys.argv[3]);raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=expected:raise SystemExit('plan digest mismatch')
d=json.loads(raw);task=d['tasks'][idx]
for path,sha in d['input_sha256'].items():
    h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h!=sha:raise SystemExit(f'input changed: {path}')
print(task['training_seed']);print(task['condition']);print(task['checkpoint']);print(task['checkpoint_sha256'])
PYCODE
)
SEED=${TASK[0]}; CONDITION=${TASK[1]}; CHECKPOINT=${TASK[2]}; CHECKPOINT_SHA=${TASK[3]}
[[ $(sha256sum "$CHECKPOINT" | cut -d' ' -f1) == "$CHECKPOINT_SHA" ]]
OUT="$ATTEMPT_ROOT/results/seed_${SEED}/${CONDITION/:/_}"
test ! -e "$OUT"
"$PY" -u extensions/frontier_multimodel_v1/score_dissociation.py \
 --prepare-dir /ssd/scxi253/model_deployment/data/interaction_first_sabdab2 \
 --split-manifest /ssd/scxi253/model_deployment/data/interaction_first/double_remote_split.json \
 --canonical-manifest /ssd/scxi253/model_deployment/data/interaction_first/canonical_manifest.csv \
 --embedding-dir /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
 --esm-if1-root /ssd/scxi253/repr_extract/out/esm_if1 \
 --structure-manifest /ssd/scxi253/frontier_multimodel_v1/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json \
 --checkpoint "$CHECKPOINT" --training-seed "$SEED" --condition "$CONDITION" --output "$OUT"
