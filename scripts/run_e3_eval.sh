#!/usr/bin/env bash
# E3 sufficiency arm (user decision 2026-10-07): finish FF-side dev calibration, freeze α = 1, run E3-S/E3-Sr on
# MATH-500, score, judge E3-S only, report.   bash scripts/run_e3_eval.sh qwen3-4b
set -euo pipefail
M=${1:-qwen3-4b}
PY=${PY:-$HOME/ftmi-venv/bin/python}
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# 1. remaining dev settings for the sufficiency arm (dose–response + extra random controls)
$PY -m src.gen_steer --model "$M" --set dev --gpu-mem 0.80 \
    --settings "FF:plus_b:0.5,FF:plus_b:1.5,FF:plus_b:2.0,FF:plus_r1:1.0,FF:plus_r2:1.0"
$PY -m src.score --model "$M" --stage mv-e3
$PY -m src.analysis_e3 --model "$M" --stage dev
# 2. freeze (refuses if the sufficiency arm failed the dev safety rule)
$PY -m src.analysis_e3 --model "$M" --stage freeze
# 3. eval generation, scoring, judge (E3-S only), final labels, report
$PY -m src.gen_steer --model "$M" --set eval --gpu-mem 0.80 --settings "FF:plus_b:1.0,FF:plus_r0:1.0"
$PY -m src.score --model "$M" --stage mv-e3 --names eval_FF_plus_b_a1,eval_FF_plus_r0_a1
$PY -m src.judge --model "$M" --e3-names eval_FF_plus_b_a1
$PY -m src.score --model "$M" --stage final-e3 --names eval_FF_plus_b_a1
$PY -m src.analysis_e3 --model "$M" --stage eval
