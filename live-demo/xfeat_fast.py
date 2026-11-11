"""XFeat with detection + top-k + descriptor sampling folded INTO one fixed-shape graph, so the whole
extractor runs as a single call on any backend: torch (cpu/mps), ONNX Runtime (CPU / CoreML EP), CoreML (ANE/GPU).
Output per image: kpts (K,2) in input pixels, scores (K,), desc (K,64) L2-normalised.
Differences vs upstream XFeat.detectAndCompute: bilinear (not bicubic) descriptor sampling, fixed K (invalid -> score 0)."""
import os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SRC = os.environ.get("LIVE_DEMO_SRC", "/workspace/live-demo-env/src")
sys.path.insert(0, os.path.join(SRC, "accelerated_features"))
from modules.model import XFeatModel
_orig_unfold = XFeatModel._unfold2d
XFeatModel._unfold2d = lambda self, x, ws=2: F.pixel_unshuffle(x, ws)   # identical result, exportable to CoreML
WEIGHTS = os.path.join(SRC, "accelerated_features", "weights", "xfeat.pt")

# NOTE (Mac diag 2026-11-11): the CoreML ALL/NE build gives wrong keypoints (0.52 agreement with onnx-cpu) while the
# GPU/CPU builds agree (0.996/0.994). torch fp16 on CPU agrees 0.996 with no overflow, so it is not fp16 math as such;
# the likely culprit is the ANE evaluating the integer index arithmetic (idx % W, idx // W, idx up to 65535) in fp16,
# which cannot represent integers above 2048 exactly. Use coreml-gpu (5.4 ms) instead of ALL/NE.
GRAPH_VER = "v1"

class XFeatGraph(nn.Module):
    def __init__(self, top_k=512, thr=0.05, B=6, H=256, W=256):
        super().__init__(); self.net = XFeatModel(); self.net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")); self.k = top_k; self.thr = thr
        self.B, self.H, self.W = B, H, W   # static shapes (CoreML/ONNX friendly)
    def forward(self, x):                      # x: (B,1,H,W) float in [0,1], H,W multiples of 32
        B, H, W = self.B, self.H, self.W; h, w = H // 8, W // 8
        M, Kl, Hr = self.net(x)
        M = F.normalize(M, dim=1)
        s = F.softmax(Kl, 1)[:, :64]
        heat = s.permute(0, 2, 3, 1).reshape(B, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 1, h * 8, w * 8)
        nms = (heat == F.max_pool2d(heat, 5, 1, 2)) & (heat > self.thr)
        rel = F.interpolate(Hr, size=(H, W), mode="bilinear", align_corners=False)
        score = (heat * rel * nms.to(heat.dtype)).reshape(B, -1)
        sc, idx = torch.topk(score, self.k, dim=1)
        xs = (idx % W).to(x.dtype); ys = torch.div(idx, W, rounding_mode="floor").to(x.dtype)
        kp = torch.stack([xs, ys], -1)
        grid = torch.stack([2 * xs / (W - 1) - 1, 2 * ys / (H - 1) - 1], -1).unsqueeze(2)
        d = F.grid_sample(M, grid, mode="bilinear", align_corners=False).squeeze(-1).permute(0, 2, 1)
        return kp, sc, F.normalize(d, dim=-1)

def mnn(da, db, sa=None, sb=None, min_cos=0.82):
    """mutual nearest neighbours on L2-normalised descriptors (numpy)."""
    if sa is not None: va, vb = np.nonzero(sa > 0)[0], np.nonzero(sb > 0)[0]; da, db = da[va], db[vb]
    else: va, vb = np.arange(len(da)), np.arange(len(db))
    if len(da) < 4 or len(db) < 4: return np.zeros(0, int), np.zeros(0, int)
    S = da @ db.T; j = S.argmax(1); i2 = S.argmax(0); i = np.arange(len(da))
    ok = (i2[j] == i) & (S[i, j] > min_cos); return va[i[ok]], vb[j[ok]]

class Runner:
    """backend in: torch-cpu, torch-mps, torch-cpu-bf16, onnx-cpu, onnx-cpu-int8, onnx-coreml, coreml-all, coreml-ne, coreml-gpu, coreml-cpu,
    coreml-all32 (fp32 compute: no ANE, GPU/CPU)"""
    def __init__(self, backend, res=256, top_k=512, batch=6, cache="/tmp/xfeat_exports"):
        self.b, self.res, self.k, self.B = backend, res, top_k, batch; os.makedirs(cache, exist_ok=True)
        g = XFeatGraph(top_k, B=batch, H=res, W=res).eval(); ex = torch.zeros(batch, 1, res, res)
        if backend.startswith("torch"):
            dev = "mps" if "mps" in backend else "cpu"; self.dev = dev; self.g = g.to(dev)
            self.dtype = torch.bfloat16 if "bf16" in backend else (torch.float16 if "fp16" in backend else torch.float32)
            if self.dtype != torch.float32: self.g = self.g.to(self.dtype)
            self.run = self._torch
        elif backend.startswith("onnx"):
            import onnxruntime as ort
            path = os.path.join(cache, f"xfeat_{res}_{top_k}_b{batch}.onnx")
            if not os.path.exists(path):
                torch.onnx.export(g, ex, path, input_names=["x"], output_names=["kp", "sc", "d"], opset_version=17, dynamo=False)
            if "int8" in backend:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                q = path.replace(".onnx", "_int8.onnx")
                if not os.path.exists(q): quantize_dynamic(path, q, weight_type=QuantType.QUInt8)
                path = q
            so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            prov = [("CoreMLExecutionProvider", {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}), "CPUExecutionProvider"] if "coreml" in backend else ["CPUExecutionProvider"]
            self.s = ort.InferenceSession(path, so, providers=prov); self.used = self.s.get_providers(); self.run = self._onnx
        elif backend.startswith("coreml"):
            import coremltools as ct
            fp32 = backend.endswith("32")
            path = os.path.join(cache, f"xfeat_{res}_{top_k}_b{batch}{'_fp32' if fp32 else ''}.mlpackage")
            if not os.path.exists(path):
                tr = torch.jit.trace(g, ex)
                m = ct.convert(tr, inputs=[ct.TensorType(name="x", shape=ex.shape)], convert_to="mlprogram",
                               compute_precision=ct.precision.FLOAT32 if fp32 else ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS13)
                m.save(path)
            cu = {"coreml-all": ct.ComputeUnit.ALL, "coreml-ne": ct.ComputeUnit.CPU_AND_NE, "coreml-gpu": ct.ComputeUnit.CPU_AND_GPU, "coreml-cpu": ct.ComputeUnit.CPU_ONLY,
                  "coreml-all32": ct.ComputeUnit.ALL}[backend]
            self.m = ct.models.MLModel(path, compute_units=cu); self.outnames = [o.name for o in self.m.get_spec().description.output]; self.run = self._coreml
        else: raise ValueError(backend)

    def _torch(self, x):
        with torch.inference_mode():
            t = torch.from_numpy(x).to(self.dev, self.dtype); kp, sc, d = self.g(t)
            return kp.float().cpu().numpy(), sc.float().cpu().numpy(), d.float().cpu().numpy()
    def _onnx(self, x): return tuple(self.s.run(None, {"x": x}))
    def _coreml(self, x):
        o = self.m.predict({"x": x}); vals = [np.asarray(o[n]) for n in self.outnames]
        vals.sort(key=lambda v: (v.ndim, v.shape[-1] if v.ndim else 0))   # sc (B,K) , kp (B,K,2), d (B,K,64)
        sc, kp, d = vals; return kp, sc, d

    def extract(self, ims):
        """ims: list of uint8 (res,res) crops; pads to the compiled batch."""
        n = len(ims); x = np.zeros((self.B, 1, self.res, self.res), np.float32)
        for i, im in enumerate(ims): x[i, 0] = im / 255.0
        kp, sc, d = self.run(x)
        return [(kp[i], (d[i], sc[i])) for i in range(n)]

class XFeatFastBackend:
    """drop-in for features.py backends (used by tracker/run_loop)."""
    def __init__(self, backend="onnx-cpu", res=256, top_k=512, batch=6):
        self.name = f"xfeat-{backend}"; self.r = Runner(backend, res, top_k, batch)
    def extract(self, ims): return self.r.extract(ims)
    def match(self, a, b): return mnn(a[1][0], b[1][0], a[1][1], b[1][1])
