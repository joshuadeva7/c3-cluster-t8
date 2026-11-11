#!/usr/bin/env bash
# Live demo on the Mac (one window; docker-selection panel with the rate sliders). Extra args pass through, e.g.  bash mac/run_demo.sh --no_cutscene --feat xfeat-torch-mps
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; LD="$(dirname "$HERE")"
VENV="${VENV:-$HOME/swarm-live/.venv}"; source "$HERE/venv.sh" || exit 1   # python venv on the system python3 3.11
source "$HERE/ensure_src.sh" || exit 1; export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$LD" && python demo.py --metrics "$HERE/demo_metrics.jsonl" "$@"
