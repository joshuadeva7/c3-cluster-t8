#!/usr/bin/env bash
# One-time setup on Apple Silicon (tested target: M1 Pro, macOS 27, conda at /opt/miniconda3).
# Creates conda env "swarm-live" (Python 3.12), installs torch (MPS), onnxruntime (CoreML EP), coremltools,
# clones XFeat + LightGlue at the exact commits benchmarked on the box, and pre-fetches weights.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; SRC="$HERE/_src"
CONDA="${CONDA:-/opt/miniconda3/bin/conda}"
source "$("$CONDA" info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^swarm-live " || conda create -y -n swarm-live -c conda-forge --override-channels python=3.12
conda activate swarm-live
python -m pip install -q --upgrade pip
python -m pip install -q "torch==2.7.1" "torchvision==0.22.1" onnx onnxruntime coremltools opencv-python-headless numpy scipy trimesh kornia moderngl "pyglet>=2.0,<2.2"
mkdir -p "$SRC"
[ -d "$SRC/accelerated_features" ] || git clone -q https://github.com/verlab/accelerated_features.git "$SRC/accelerated_features"
git -C "$SRC/accelerated_features" checkout -q e92685f57f8318b18725c5c8c0bd28c7fe188d9a
[ -d "$SRC/LightGlue" ] || git clone -q https://github.com/cvg/LightGlue.git "$SRC/LightGlue"
git -C "$SRC/LightGlue" checkout -q 5a650c444718caffc2669dc0094cde908f42dd73
python -m pip install -q -e "$SRC/LightGlue"
python -m pip uninstall -y -q pyrender opencv-python 2>/dev/null || true   # demo renders with moderngl (CGL) + one pyglet window; no OpenCV GUI, no pyrender
python -m pip install -q --force-reinstall --no-deps opencv-python-headless
python -c "import moderngl, pyglet, cv2; ctx = moderngl.create_standalone_context(); print('GL', ctx.info['GL_RENDERER'], ctx.version_code, '| pyglet', pyglet.version, '| cv2', cv2.__version__)"
python - <<'PY'
from lightglue import SuperPoint, LightGlue
SuperPoint(max_num_keypoints=512); LightGlue(features="superpoint"); print("LightGlue/SuperPoint weights cached")
import torch, onnxruntime as ort, coremltools as ct
print("torch", torch.__version__, "MPS available:", torch.backends.mps.is_available())
print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
print("coremltools", ct.__version__)
PY
echo "setup OK"
