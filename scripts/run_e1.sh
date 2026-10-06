#!/usr/bin/env bash
# E1 pipeline for one model (run after the parity check passed). Idempotent/resumable.
#   bash scripts/run_e1.sh qwen3-4b
set -euo pipefail
M=$1
PY=${PY:-$HOME/ftmi-venv/bin/python}
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ENGINE_ARGS=${ENGINE_ARGS:-}

# 1. brevity behavioural check (STOP on failure: exit code 2)
$PY -m src.gen_vllm --model "$M" --task brevity $ENGINE_ARGS
$PY -m src.analysis_e1 --model "$M" --stage brevity || { echo "STOP: brevity check failed for $M"; exit 2; }
# 2. extraction-set generations for P_gen16
$PY -m src.gen_vllm --model "$M" --task ext80 $ENGINE_ARGS
# 3. activations on extract_set (HF, separate process from vLLM)
$PY -m src.acts_hf --model "$M" --set extract --what prompt
$PY -m src.acts_hf --model "$M" --set extract --what gen16
# 4. directions, nulls, reliability, freeze L* (before any eval-set activation)
$PY -m src.directions --model "$M"
# 5. per-question test (only if the E0 gate passed)
if $PY -c "import json,sys; sys.exit(0 if json.load(open('results/$($PY -c "from src.common import model_info; print(model_info('$M')['tag'])")/e0_summary.json'))['gate']['PASS'] else 1)"; then
  $PY -m src.acts_hf --model "$M" --set eval
  $PY -m src.analysis_e1 --model "$M" --stage perq
else
  echo "gate not passed for $M: skipping §8.5"
fi
$PY -m src.analysis_e1 --model "$M" --stage report
