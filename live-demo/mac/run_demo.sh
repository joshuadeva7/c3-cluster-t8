#!/usr/bin/env bash
# Live demo on the Mac (window with sliders). Extra args pass through, e.g.  bash mac/run_demo.sh --no_cutscene --feat xfeat-torch-mps
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; LD="$(dirname "$HERE")"
CONDA="${CONDA:-/opt/miniconda3/bin/conda}"; source "$("$CONDA" info --base)/etc/profile.d/conda.sh"; conda activate swarm-live
source "$HERE/ensure_src.sh" || exit 1; export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$LD" && python demo.py --metrics "$HERE/demo_metrics.jsonl" "$@"
