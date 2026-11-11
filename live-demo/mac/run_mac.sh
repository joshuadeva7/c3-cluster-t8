#!/usr/bin/env bash
# Runs the SAME benchmark harness used on the box, on the Mac's CPU / GPU (MPS) / Neural Engine (CoreML).
# Each backend runs in its own process so one backend's thread pool can't slow the next.
# Output: mac/results/<timestamp>/results.jsonl + summary.md (+ e2e loop, if pyrender works on this Mac).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; LD="$(dirname "$HERE")"
CONDA="${CONDA:-/opt/miniconda3/bin/conda}"; source "$("$CONDA" info --base)/etc/profile.d/conda.sh"; conda activate swarm-live
source "$HERE/ensure_src.sh" || exit 1; export PYTORCH_ENABLE_MPS_FALLBACK=1   # any op MPS lacks falls back to CPU (counted in the timing)
OUT="$HERE/results/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"; cd "$LD"
{ sysctl -n machdep.cpu.brand_string; sysctl -n hw.memsize; sw_vers; python -c "import torch,onnxruntime as o,coremltools as c;print(torch.__version__,o.__version__,o.get_available_providers(),c.__version__)"; } > "$OUT/system.txt" 2>&1
BACKENDS="orb xfeat-orig xfeat-torch-cpu xfeat-torch-mps xfeat-torch-mps-fp16 xfeat-onnx-cpu xfeat-onnx-coreml xfeat-coreml-all xfeat-coreml-ne xfeat-coreml-gpu xfeat-coreml-cpu splg-cpu splg-mps"
for b in $BACKENDS; do
  echo "== $b"; python pose_bench.py --backends "$b" --crops 256 192 128 --prior 5 15 --reps 3 --host mac --out "$OUT/results.jsonl" 2>>"$OUT/stderr.log" | grep '^{' | cut -c1-200
done
for b in orb xfeat-onnx-coreml xfeat-coreml-all splg-mps; do   # lighting-mismatch robustness
  python pose_bench.py --backends "$b" --crops 256 --prior 5 15 --mismatch --reps 1 --host mac --out "$OUT/results.jsonl" 2>>"$OUT/stderr.log" | grep '^{' | cut -c1-200
done
python test_mekf_synth.py > "$OUT/mekf.txt" 2>&1; tail -1 "$OUT/mekf.txt"
# end-to-end loop (needs pyrender offscreen via pyglet on macOS; skipped gracefully if it can't open a GL context)
for f in orb xfeat-coreml-all xfeat-onnx-coreml xfeat-torch-mps; do
  python run_loop.py --feat "$f" --T 30 --log "$OUT/loop_$f.csv" --json "$OUT/loop.jsonl" 2>>"$OUT/stderr.log" | cut -c1-300 || echo "e2e $f failed (see stderr.log)"
done
python mac/summarize.py "$OUT" > "$OUT/summary.md"; cat "$OUT/summary.md"
echo; echo "RESULTS: $OUT  (send summary.md + results.jsonl + loop.jsonl + stderr.log back)"
