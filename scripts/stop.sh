#!/usr/bin/env bash
# Stop a background job started by bg.sh, including vLLM's EngineCore child (whole process group).
#   bash scripts/stop.sh <name>
pid=$(cat ~/ftmi_logs/"$1".pid)
kill -TERM -- -"$pid" 2>/dev/null
sleep 10
kill -KILL -- -"$pid" 2>/dev/null
echo "stopped $1 (pgid $pid)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
