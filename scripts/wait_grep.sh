#!/usr/bin/env bash
# Wait (max $3 seconds, default 600) until log <name> matches regex $2 or the job exits; print matching lines.
#   bash scripts/wait_grep.sh <name> <regex> [max_s]
name=$1; re=$2; max=${3:-600}
f=~/ftmi_logs/"$name".log; pid=$(cat ~/ftmi_logs/"$name".pid)
t=0
while [ "$t" -lt "$max" ]; do
  grep -qE "$re" "$f" && break
  kill -0 "$pid" 2>/dev/null || { echo "job exited"; break; }
  sleep 5; t=$((t + 5))
done
tr '\r' '\n' < "$f" | grep -E "$re" | tail -n 10
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
