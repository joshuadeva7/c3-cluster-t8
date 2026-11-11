#!/usr/bin/env bash
# One-time setup on Apple Silicon (tested target: M1 Pro). A plain venv on the system Python 3.11
# (default VENV=$HOME/swarm-live/.venv, created here if missing; PYTHON=/usr/local/bin/python3 by default).
# Installs torch (MPS), onnxruntime (CoreML EP), coremltools, moderngl + pyglet; puts XFeat + LightGlue sources in mac/_src
# (bundled copy, or git clone at the benchmarked commits) and pre-fetches weights. LightGlue is NOT pip-installed:
# ensure_src.sh puts mac/_src/* on PYTHONPATH, so nothing points into a build folder that a package update deletes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/swarm-live/.venv}"; PYTHON="${PYTHON:-/usr/local/bin/python3}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[setup] creating venv $VENV with $PYTHON ($("$PYTHON" --version 2>&1))"; mkdir -p "$(dirname "$VENV")"; "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -c "import sys; assert sys.version_info >= (3, 10), sys.version; print('[setup] python', sys.version.split()[0], sys.executable)"
python -m pip install -q --upgrade pip
python -m pip install -q "torch==2.7.1" "torchvision==0.22.1" onnx onnxruntime coremltools opencv-python-headless numpy scipy trimesh kornia moderngl "pyglet>=2.0,<2.2"
python -m pip uninstall -y -q lightglue pyrender opencv-python 2>/dev/null || true   # no editable LightGlue install; no OpenCV GUI, no pyrender
python -m pip install -q --force-reinstall --no-deps opencv-python-headless
source "$HERE/ensure_src.sh"
python -c "import moderngl, pyglet, cv2; ctx = moderngl.create_standalone_context(); print('GL', ctx.info['GL_RENDERER'], ctx.version_code, '| pyglet', pyglet.version, '| cv2', cv2.__version__)"
python - <<'PY'
from lightglue import SuperPoint, LightGlue
SuperPoint(max_num_keypoints=512); LightGlue(features="superpoint"); print("LightGlue/SuperPoint weights cached (from mac/_src via PYTHONPATH)")
import torch, onnxruntime as ort, coremltools as ct
print("torch", torch.__version__, "MPS available:", torch.backends.mps.is_available())
print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
print("coremltools", ct.__version__)
PY
echo "[setup] done. Run: bash mac/test_demo_headless.sh  then  bash mac/run_demo.sh"
