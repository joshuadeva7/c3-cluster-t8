"""Pretrained local-feature backends used by the CAD render-and-match pose tracker.
Each backend: extract(list_of_gray_uint8) -> list of (kpts Nx2 float32, desc), match(fa, fb) -> (ia, ib)."""
import os, sys, numpy as np, torch, cv2
SRC = os.environ.get("LIVE_DEMO_SRC", "/workspace/live-demo-env/src")
sys.path.insert(0, os.path.join(SRC, "accelerated_features"))

class ORB:
    name = "orb"
    def __init__(self, n=1000): self.o = cv2.ORB_create(n, fastThreshold=5); self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    def extract(self, ims):
        out = []
        for im in ims:
            k, d = self.o.detectAndCompute(im, None)
            out.append((np.float32([p.pt for p in k]).reshape(-1, 2), d))
        return out
    def match(self, a, b):
        if a[1] is None or b[1] is None: return np.zeros(0, int), np.zeros(0, int)
        m = self.bf.match(a[1], b[1]); return np.array([x.queryIdx for x in m], int), np.array([x.trainIdx for x in m], int)

class XFeatMNN:
    name = "xfeat"
    def __init__(self, top_k=1024, min_cossim=0.82):
        from modules.xfeat import XFeat
        self.x = XFeat(top_k=top_k); self.top_k = top_k; self.mc = min_cossim
    @torch.inference_mode()
    def extract(self, ims):
        t = torch.from_numpy(np.stack(ims)).float()[:, None] / 255.0
        fs = self.x.detectAndCompute(t, top_k=self.top_k)
        return [(f["keypoints"].numpy(), f["descriptors"]) for f in fs]
    @torch.inference_mode()
    def match(self, a, b):
        if len(a[0]) < 4 or len(b[0]) < 4: return np.zeros(0, int), np.zeros(0, int)
        i, j = self.x.match(a[1], b[1], self.mc); return i.numpy(), j.numpy()

class SuperPointLG:
    name = "sp_lg"
    def __init__(self, max_kp=512, flash=False):
        from lightglue import SuperPoint, LightGlue
        self.e = SuperPoint(max_num_keypoints=max_kp).eval(); self.m = LightGlue(features="superpoint", depth_confidence=0.9, width_confidence=0.95).eval()
    @torch.inference_mode()
    def extract(self, ims):
        t = torch.from_numpy(np.stack(ims)).float()[:, None] / 255.0
        out = []
        for k in range(len(ims)):   # SuperPoint batches fine but keypoint counts differ; keep per-image dicts
            f = self.e({"image": t[k:k + 1]}); out.append((f["keypoints"][0].numpy(), f))
        return out
    @torch.inference_mode()
    def match(self, a, b):
        fa, fb = a[1], b[1]
        r = self.m({"image0": fa, "image1": fb}); mm = r["matches"][0].numpy()
        return mm[:, 0], mm[:, 1]

def make(name, **kw):
    if name.startswith("xfeat-"):   # fused single-graph XFeat on a chosen backend (onnx-cpu, onnx-coreml, coreml-all, torch-mps, ...)
        import xfeat_fast; return xfeat_fast.XFeatFastBackend(name[6:], res=kw.get("res", 256), top_k=kw.get("top_k", 512), batch=6)
    return {"orb": ORB, "xfeat": XFeatMNN, "sp_lg": SuperPointLG}[name](**kw)
