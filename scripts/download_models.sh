#!/usr/bin/env bash
# Download the two generator checkpoints (bf16 safetensors) into the HF cache.
set -euo pipefail
HF=${HF_BIN:-hf}
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  echo "[$(date -Is)] downloading $m"
  "$HF" download "$m"
done
echo "[$(date -Is)] done"
