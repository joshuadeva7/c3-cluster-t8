#!/usr/bin/env bash
# Live-demo diagnostic (GL readback + feature backends). Output: mac/results/diag_<time>/ (diag.json, cam*_color.png, cam*_depth.npy/png).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/swarm-live/.venv}"; source "$HERE/venv.sh" || exit 1   # python venv on the system python3 3.11
source "$HERE/ensure_src.sh" || exit 1; export PYTORCH_ENABLE_MPS_FALLBACK=1 OPENBLAS_NUM_THREADS=1
OUT="$HERE/results/diag_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
python "$HERE/diag.py" "$OUT" 2>&1 | tee "$OUT/diag.log"
echo "send back: $OUT (tar czf diag.tgz -C $HERE/results $(basename "$OUT"))"
