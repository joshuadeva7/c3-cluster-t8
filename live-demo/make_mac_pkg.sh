#!/usr/bin/env bash
# Box-side: build /workspace/live-demo-mac.tgz = live-demo/ (no videos/run logs) + mac/_src_bundle (XFeat modules + xfeat.pt, LightGlue code).
set -euo pipefail
LD="$(cd "$(dirname "$0")" && pwd)"; SRC="${LIVE_DEMO_SRC:-/workspace/live-demo-env/src}"; ST="$(mktemp -d)"
tar cf - -C "$(dirname "$LD")" --exclude '__pycache__' --exclude 'live-demo/demo_capture.mp4' --exclude 'live-demo/runs' --exclude 'live-demo/mac/_src' --exclude 'live-demo/mac/results' --exclude 'live-demo/mac_results_*.tgz' live-demo | tar xf - -C "$ST"
B="$ST/live-demo/mac/_src_bundle"; mkdir -p "$B/accelerated_features/weights" "$B/LightGlue"
cp -R "$SRC/accelerated_features/modules" "$SRC/accelerated_features/LICENSE" "$SRC/accelerated_features/README.md" "$B/accelerated_features/"
cp "$SRC/accelerated_features/weights/xfeat.pt" "$B/accelerated_features/weights/"
cp -R "$SRC/LightGlue/lightglue" "$SRC/LightGlue/pyproject.toml" "$SRC/LightGlue/LICENSE" "$SRC/LightGlue/README.md" "$B/LightGlue/" 2>/dev/null || true
find "$B" -name '__pycache__' -prune -exec rm -rf {} +
tar czf /workspace/live-demo-mac.tgz -C "$ST" live-demo; rm -rf "$ST"; ls -la /workspace/live-demo-mac.tgz
