#!/usr/bin/env bash
# Block until the background job <name> (scripts/bg.sh) exits; then print the tail of its log.
#   bash scripts/wait_for.sh <name> [tail_lines]
name=$1; n=${2:-15}
pid=$(cat ~/ftmi_logs/"$name".pid)
while kill -0 "$pid" 2>/dev/null; do sleep 20; done
echo "== $name (pid $pid) exited"
tr '\r' '\n' < ~/ftmi_logs/"$name".log | grep -v '^\s*$' | tail -n "$n"
