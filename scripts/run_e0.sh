#!/usr/bin/env bash
# E0 pipeline for one generator model. Every step is idempotent/resumable.
#   bash scripts/run_e0.sh qwen3-4b phase1   # FF/JS/JC -> mv scoring -> judge -> final labels -> self-consistency -> audit CSV (STOP)
#   bash scripts/run_e0.sh qwen3-4b phase2   # after audit acceptance: E0 analysis + gate
#   bash scripts/run_e0.sh qwen3-4b twoturn  # 2T generation, scoring, judging, re-analysis
# Extra engine flags (OOM tuning) can be passed via ENGINE_ARGS="--max-num-seqs 2".
set -euo pipefail
M=$1; PHASE=${2:-phase1}
PY=${PY:-$HOME/ftmi-venv/bin/python}
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ENGINE_ARGS=${ENGINE_ARGS:-}
JUDGE_ARGS=${JUDGE_ARGS:-}

case $PHASE in
  phase1)
    $PY -m src.gen_vllm --model "$M" --task e0 --conds FF,JS,JC $ENGINE_ARGS
    $PY -m src.score --model "$M" --stage mv --conds FF,JS,JC
    $PY -m src.judge --model "$M" --conds FF,JS,JC $JUDGE_ARGS
    $PY -m src.score --model "$M" --stage final --conds FF,JS,JC
    $PY -m src.judge --model "$M" --conds FF,JS,JC --selfcons $JUDGE_ARGS
    $PY -m src.audit --model "$M" --stage export --conds FF,JS,JC
    echo "STOP: human labels needed in reports/audit_*.csv for $M"
    ;;
  phase2)
    $PY -m src.audit --model "$M" --stage agree
    $PY -m src.analysis_e0 --model "$M"
    ;;
  twoturn)
    $PY -m src.gen_vllm --model "$M" --task e0 --conds 2T $ENGINE_ARGS
    $PY -m src.score --model "$M" --stage mv --conds 2T
    $PY -m src.judge --model "$M" --conds 2T $JUDGE_ARGS
    $PY -m src.score --model "$M" --stage final --conds 2T
    $PY -m src.analysis_e0 --model "$M"
    ;;
  *) echo "unknown phase $PHASE"; exit 1;;
esac
