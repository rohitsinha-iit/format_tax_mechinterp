#!/usr/bin/env bash
# Quick status of background jobs (setup, downloads, pipeline logs) and the GPU.
for f in ~/ftmi_logs/*.log; do
  echo "== $(basename "$f") ($(stat -c %y "$f" | cut -c1-19))"
  tr '\r' '\n' < "$f" | grep -v '^\s*$' | tail -n "${N:-3}"
done
echo "== 8B blobs"
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/blobs/ 2>/dev/null | awk '$5 > 100000000 {print $5, $6, $7, $8, substr($9,1,20)}'
echo "== running"
for p in ~/ftmi_logs/*.pid; do pid=$(cat "$p"); if kill -0 "$pid" 2>/dev/null; then echo "$(basename "$p" .pid) pid $pid RUNNING"; else echo "$(basename "$p" .pid) pid $pid exited"; fi; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
