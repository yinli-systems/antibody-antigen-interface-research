#!/usr/bin/env bash
#SBATCH --job-name=epic-rv-abl
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
set -euo pipefail
root=${1:?campaign root required}
protocol=${2:?campaign protocol required}
protocol_sha=${3:?campaign protocol sha required}
authority_receipt=${4:?authority receipt required}
authority_sha=${5:?authority receipt sha required}
smoke_manifest=${6:?smoke manifest required}
smoke_manifest_sha=${7:?smoke manifest sha required}
index=${SLURM_ARRAY_TASK_ID:?array task required}
read -r condition seed smoke_path smoke_sha < <(python3 - "$protocol" "$protocol_sha" "$smoke_manifest" "$smoke_manifest_sha" "$index" <<'PY'
import hashlib,json,sys

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
protocol,protocol_sha,manifest,manifest_sha,index=sys.argv[1:]
if sha(protocol)!=protocol_sha: raise SystemExit("protocol hash differs")
if sha(manifest)!=manifest_sha: raise SystemExit("smoke manifest hash differs")
p=json.load(open(protocol)); m=json.load(open(manifest)); i=int(index)
conditions=[x["name"] for x in p["conditions"]]; seeds=p["seeds"]
condition=conditions[i//len(seeds)]; seed=seeds[i%len(seeds)]
r=m["receipts"][condition]
print(condition,seed,r["result_path"],r["result_sha256"])
PY
)
output="$root/runs/$condition/seed-$seed"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
exec /ssd/scxi253/model_deployment/envs/contact/bin/python3.10 "$root/source/train_condition.py" \
  --campaign-protocol "$protocol" \
  --campaign-protocol-sha256 "$protocol_sha" \
  --authority-receipt "$authority_receipt" \
  --authority-receipt-sha256 "$authority_sha" \
  --smoke-receipt "$smoke_path" --smoke-receipt-sha256 "$smoke_sha" \
  --condition "$condition" --seed "$seed" --output "$output"
