#!/usr/bin/env bash
# Phase 7 ablations: component ablations now, then index the other chunking strategies and ablate them.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=data/runs/phase7_ablations.log
step() { echo "=== $(TZ=Asia/Kolkata date '+%H:%M IST %d %b') $*" | tee -a "$LOG"; }
step "1/3 component ablations";  .venv/bin/nrag ablate >>"$LOG" 2>&1
step "2/3 index other strategies"
for s in recursive sliding_window semantic hierarchical; do
  step "    index $s"; .venv/bin/nrag index --strategy "$s" >>"$LOG" 2>&1
done
step "3/3 ablations incl. strategies"; .venv/bin/nrag ablate >>"$LOG" 2>&1
step "done"
