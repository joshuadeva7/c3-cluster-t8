#!/usr/bin/env bash
# Build + run the sim only when cluster_traj.cpp is newer than the cached outputs.
set -e
cd "$(dirname "$0")"; mkdir -p out
CXX=${CXX:-g++}; command -v "$CXX" >/dev/null || CXX="python3 -m ziglang c++"
if [ ! -x out/ct ] || [ cluster_traj.cpp -nt out/ct ]; then
  $CXX -O2 -std=c++17 -Wno-format-security cluster_traj.cpp -o out/ct
fi
if [ "$1" = "-f" ] || [ ! -f out/summary.txt ] || [ out/ct -nt out/summary.txt ]; then
  (cd out && ./ct)        # ~3 min
else
  echo "sim outputs cached in out/ (use -f to force)"
fi
