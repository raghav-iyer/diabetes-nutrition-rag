#!/usr/bin/env bash
# Phase 7 live pipeline: waits for the running extraction, then rebuilds scores/index and runs the first eval.
# Each step logs to data/runs/phase7_live.log; the script stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=data/runs/phase7_live.log
mkdir -p data/runs
step() { echo "=== $(TZ=Asia/Kolkata date '+%H:%M IST') $*" | tee -a "$LOG"; }
while pgrep -f "nrag extract --sample all" >/dev/null; do sleep 60; done
step "extraction finished"
step "1/7 score";               .venv/bin/nrag score                        >>"$LOG" 2>&1
step "2/7 summarize-tables";    .venv/bin/nrag summarize-tables --concurrency 2 >>"$LOG" 2>&1
step "3/7 index";               .venv/bin/nrag index                        >>"$LOG" 2>&1
step "4/7 llamaindex-build";    .venv/bin/nrag llamaindex-build             >>"$LOG" 2>&1
step "5/7 eval-pool";           .venv/bin/nrag eval-pool --workers 2        >>"$LOG" 2>&1
step "6/7 eval-import-labels";  .venv/bin/nrag eval-import-labels           >>"$LOG" 2>&1
step "7/7 eval --ragas";        .venv/bin/nrag eval --ragas --label base    >>"$LOG" 2>&1
step "done"
