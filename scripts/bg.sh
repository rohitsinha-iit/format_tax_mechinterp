#!/usr/bin/env bash
# Launch a long-running command detached from the wsl.exe session.
#   bash scripts/bg.sh <name> <command...>   -> log in ~/ftmi_logs/<name>.log, pid in ~/ftmi_logs/<name>.pid
# (WSL kills children if wsl.exe returns before the detached process has started, hence the sleep.)
set -euo pipefail
name=$1; shift
mkdir -p ~/ftmi_logs
cd "$(dirname "$0")/.."
setsid nohup "$@" > ~/ftmi_logs/"$name".log 2>&1 < /dev/null &
echo $! > ~/ftmi_logs/"$name".pid
sleep 5
echo "started $name pid $(cat ~/ftmi_logs/"$name".pid)"
