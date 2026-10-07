#!/usr/bin/env bash
# E3 dev calibration (docs/E3_plan.md): 100 reserved dev problems, vLLM + steering hook, then math-verify scoring.
#   bash scripts/run_e3_dev.sh qwen3-4b
set -euo pipefail
M=${1:-qwen3-4b}
PY=${PY:-$HOME/ftmi-venv/bin/python}
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
S="JS:none:0,FF:none:0,JS:minus_b:1.0,FF:plus_b:1.0"
S="$S,JS:minus_r0:1.0,FF:plus_r0:1.0,JS:minus_f:1.0"
S="$S,JS:minus_b:0.5,JS:minus_b:1.5,JS:minus_b:2.0,FF:plus_b:0.5,FF:plus_b:1.5,FF:plus_b:2.0"
S="$S,JS:minus_r1:1.0,JS:minus_r2:1.0,FF:plus_r1:1.0,FF:plus_r2:1.0"
$PY -m src.gen_steer --model "$M" --set dev --settings "$S" --gpu-mem 0.80
$PY -m src.score --model "$M" --stage mv-e3
