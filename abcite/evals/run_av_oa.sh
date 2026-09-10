#!/usr/bin/env bash
# Run OA audio/video RAG on a GPU box after visual QA is done.
# Does not replace synthetic fixture receipts.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export ABCITE_CLAP="${ABCITE_CLAP:-1}"
export ABCITE_WHISPER="${ABCITE_WHISPER:-1}"
python evals/download_av.py --max 16
python evals/run_audio_qa.py --oa
python evals/run_video_qa.py --oa
echo "wrote evals/audio_qa_oa_latest.json evals/video_qa_oa_latest.json"
