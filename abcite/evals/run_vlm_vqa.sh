#!/usr/bin/env bash
# Cite-or-abstain VLM VQA on the 5090 AFTER ColQwen2 retrieval embeddings exist.
# Do not start this while another job holds the GPU.
#
# Retrieval stays vidore/colqwen2-base + LoRA vidore/colqwen2-v1.0 (MaxSim).
# VQA generate is Qwen/Qwen2-VL-2B-Instruct — not ColQwen2, not MaxSim, not ViDoRe.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ABCITE_VLM=1
export ABCITE_VLM_MODEL="${ABCITE_VLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
export ABCITE_COLPALI=1
export ABCITE_COLPALI_ENCODE=1
echo "VLM VQA: model=${ABCITE_VLM_MODEL} HF_HOME=${HF_HOME}"
echo "Need Hub snapshot models--Qwen--Qwen2-VL-2B-Instruct (~4.4 GB)."
echo "Do not point ABCITE_VLM_MODEL at vidore/colqwen2-base (custom_text_proj / retrieval)."
python evals/run_visual_qa.py --try-colpali --colpali-only --vlm
echo "wrote evals/visual_qa_latest.json — check vqa.path for qwen2-vl generate; do not invent numbers"
