"""Portable pose-model benchmark (box CPU and Apple Silicon). Render-free: uses bench_data_256.npz.
Pipeline per camera = real feature model on (live crop, reference-render crop) -> matches -> lift via ref depth
-> PnP RANSAC + LM. A '3-camera cycle' = 6 crops through the model (batched where the backend allows) + 3 PnP solves.
Usage: python pose_bench.py --backends onnx-cpu torch-mps coreml-all ... --crops 256 192 128 --out results_<host>.jsonl"""
import os as _os; _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # OpenBLAS thread spinning slowed torch CPU 5x when interleaved
import argparse, json, os, platform, time, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.join(HERE, "bench_data_256.npz"))
p.add_argument("--backends", nargs="+", default=["orb", "xfeat-torch-cpu", "xfeat-onnx-cpu"])
p.add_argument("--crops", nargs="+", type=int, default=[256, 192, 128]); p.add_argument("--topk", type=int, default=512)
p.add_argument("--prior", nargs="+", type=int, default=[5, 15]); p.add_argument("--mismatch", action="store_true")
p.add_argument("--reps", type=int, default=2); p.add_argument("--threads", type=int, default=0); p.add_argument("--out", default="")
p.add_argument("--host", default=platform.node())
a = p.parse_args()
import torch
if a.threads: torch.set_num_threads(a.threads)
D = dict(np.load(a.data)); K = D["K"].astype(np.float64); Kinv = np.linalg.inv(K)

def rerr(Ra, Rb): return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))

def solve(kl, kr, il, ir, dep, A, R0, t0, cr):
    if len(il) < 8: return None
    s = 256.0 / cr
    pl = (kl[il] * s) @ A[:, :2].T + A[:, 2]; prc = kr[ir] * s
    u = np.clip(np.round(prc[:, 0]).astype(int), 0, 255); v = np.clip(np.round(prc[:, 1]).astype(int), 0, 255)
    z = dep[v, u].astype(np.float64); ok = z > 0
    if ok.sum() < 8: return None
    pr = prc[ok] @ A[:, :2].T + A[:, 2]
    Xb = ((np.c_[pr, np.ones(ok.sum())] @ Kinv.T) * z[ok, None] - t0) @ R0
    rv0, _ = cv2.Rodrigues(R0)
    okp, rv, tv, inl = cv2.solvePnPRansac(Xb, pl[ok].astype(np.float64), K, None, rv0.copy(), t0.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                         iterationsCount=100, reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)
    if not okp or inl is None or len(inl) < 8: return None
    inl = inl[:, 0]; rv, tv = cv2.solvePnPRefineLM(Xb[inl], pl[ok][inl].astype(np.float64), K, None, rv, tv)
    return cv2.Rodrigues(rv)[0], tv[:, 0]

def make(name, cr):
    if name == "orb":
        import features; return features.ORB(n=a.topk), False
    if name.startswith("splg"):
        from lightglue import SuperPoint, LightGlue
        dev = "mps" if "mps" in name else "cpu"
        e = SuperPoint(max_num_keypoints=a.topk).eval().to(dev); m = LightGlue(features="superpoint", depth_confidence=0.9, width_confidence=0.95).eval().to(dev)
        class SPLG:
            def extract(self, ims):
                with torch.inference_mode():
                    t = torch.from_numpy(np.stack(ims)).float()[:, None].div(255).to(dev); out = []
                    for k in range(len(ims)):
                        f = e({"image": t[k:k + 1]}); out.append((f["keypoints"][0].cpu().numpy(), f))
                    return out
            def match(self, fa, fb):
                with torch.inference_mode():
                    mm = m({"image0": fa[1], "image1": fb[1]})["matches"][0].cpu().numpy(); return mm[:, 0], mm[:, 1]
        return SPLG(), False
    if name.startswith("xfeat-orig"):
        import features; return features.XFeatMNN(top_k=a.topk), False
    if name.startswith("xfeat-"):
        import xfeat_fast; return xfeat_fast.XFeatFastBackend(name[6:], res=cr, top_k=a.topk, batch=6), True
    raise ValueError(name)

N = len(D["live"]); results = []
def run_one(name, cr, F):
    rs = lambda im: im if cr == 256 else cv2.resize(im, (cr, cr), interpolation=cv2.INTER_AREA)
    for lvl in a.prior:
        idx = [i for i in range(N) if D["prior_deg"][i] == lvl]; refk = "ref_mis" if a.mismatch else "ref"
        idx = idx[: (len(idx) // 3) * 3]
        # warm-up (compile / CoreML load)
        F.extract([rs(D["live"][idx[0]]), rs(D[refk][idx[0]])] * 3)
        tm, tt, re, te, ok = [], [], [], [], 0
        for rep in range(a.reps):
            for c in range(0, len(idx), 3):
                cams = idx[c:c + 3]; ims = []
                for i in cams: ims += [rs(D["live"][i]), rs(D[refk][i])]
                t0 = time.perf_counter(); feats = F.extract(ims); t1 = time.perf_counter()
                for j, i in enumerate(cams):
                    fl, fr = feats[2 * j], feats[2 * j + 1]; il, ir = F.match(fl, fr)
                    sol = solve(fl[0], fr[0], il, ir, D["dep"][i], D["A"][i], D["R0"][i], D["t0"][i], cr)
                    if rep == 0 and sol is not None:
                        ok += 1; re.append(rerr(sol[0], D["R"][i])); te.append(float(np.linalg.norm(sol[1] - D["t"][i])))
                t2 = time.perf_counter(); tm.append(t1 - t0); tt.append(t2 - t0)
        nf = len(idx)
        r = dict(host=a.host, backend=name, used=getattr(getattr(F, "r", None), "used", None), crop=cr, topk=a.topk, prior_deg=lvl, mismatch=a.mismatch,
                 model_ms_3cam=1000 * float(np.median(tm)), cycle_ms_3cam=1000 * float(np.median(tt)), hz_3cam=1 / float(np.median(tt)),
                 success=ok / nf, rot_err_med=float(np.median(re)) if re else None, rot_err_p90=float(np.percentile(re, 90)) if re else None,
                 trans_err_med_m=float(np.median(te)) if te else None, n=nf, threads=torch.get_num_threads())
        print(json.dumps(r)); results.append(r)
        if a.out: open(a.out, "a").write(json.dumps(r) + "\n")

for name in a.backends:
    for cr in a.crops:
        try:
            F, batched = make(name, cr)
            run_one(name, cr, F)
        except Exception as ex:   # one broken backend must not stop the sweep; record why
            r = dict(host=a.host, backend=name, crop=cr, error=repr(ex)[:300]); print(json.dumps(r)); results.append(r)
            if a.out: open(a.out, "a").write(json.dumps(r) + "\n")
