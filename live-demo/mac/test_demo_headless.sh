#!/usr/bin/env bash
# Headless self-test of the demo on the Mac (no window; standalone CGL context). Writes mac/results/demo_test.mp4 + metrics.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$HERE/results"
bash "$HERE/run_demo.sh" --headless --no_cutscene --fixed_dt 0.1 --T 50 --script "0:w=0.02,0.66,0.02;1:go" --record "$HERE/results/demo_test.mp4" "$@"
