#!/usr/bin/env bash
# Run every experiment that has no result yet, and keep going after a restart.
#
# Safe to interrupt and safe to run twice: scripts/experiment.py skips anything
# already in reports/experiments.csv, so the work already paid for is never
# repeated. Safe to run while nothing else is happening, too — it is the whole
# of the compute, and none of it needs an agent attached.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p reports
LOG="reports/run.log"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') resuming ==="
  .venv/bin/python scripts/experiment.py --status
  .venv/bin/python scripts/experiment.py
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') queue drained ==="
} >> "$LOG" 2>&1

tail -40 "$LOG"
