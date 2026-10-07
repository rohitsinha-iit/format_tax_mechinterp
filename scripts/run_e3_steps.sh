#!/usr/bin/env bash
# E3 steps 1–3 (docs/E3_plan.md): steering vectors, HF manipulation check, steered vLLM-vs-HF parity.
#   bash scripts/run_e3_steps.sh qwen3-4b
set -euo pipefail
M=${1:-qwen3-4b}
PY=${PY:-$HOME/ftmi-venv/bin/python}
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
$PY -m src.steer --model "$M" --stage vectors
$PY -m src.steer --model "$M" --stage manip
$PY -m src.steer --model "$M" --stage parity-vllm --gpu-mem 0.80
$PY -m src.steer --model "$M" --stage parity-hf
