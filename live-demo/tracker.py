"""Render-and-match CAD pose tracker.
Given a prior pose (from the filter), render TOPEX at the prior (colour + depth), crop both the live frame and
the render to the prior's ROI, match pretrained local features, lift render keypoints to TOPEX body 3-D points
via the depth buffer, solve PnP (RANSAC + LM). Output: measured (R, t) body->camera."""
import numpy as np, cv2

def roi_from_mask(mask, pad=0.15, min_side=24):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2]); s = max(x1 - x0, y1 - y0, min_side) * (1 + 2 * pad)
    return c, s

def crop(img, c, s, out):
    """square crop centred c, side s (pixels), resampled to out x out. Returns crop + affine (out->full)."""
    sc = s / out; A = np.array([[sc, 0, c[0] - s / 2], [0, sc, c[1] - s / 2]])
    M = np.array([[1 / sc, 0, -(c[0] - s / 2) / sc], [0, 1 / sc, -(c[1] - s / 2) / sc]])
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if sc > 1 else cv2.INTER_LINEAR), A

class Tracker:
    def __init__(self, renderer, feat, crop_res=256, iters=1, min_inl=8):
        self.r = renderer; self.f = feat; self.cr = crop_res; self.iters = iters; self.min_inl = min_inl
        self.K = renderer.K; self.Kinv = np.linalg.inv(self.K)

    def prepare(self, live_gray_list, priors, suns=None):
        """render refs + crops for a batch of cameras; returns per-camera dicts and the image list to extract."""
        jobs, ims = [], []
        for k, (g, (R0, t0)) in enumerate(zip(live_gray_list, priors)):
            col, dep = self.r.render(R0, t0) if suns is None else self.r.render(R0, t0, sun_cam=suns[k])
            roi = roi_from_mask(dep > 0)
            if roi is None: jobs.append(None); continue
            c, s = roi
            ref = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)
            lc, A = crop(g, c, s, self.cr); rc, _ = crop(ref, c, s, self.cr)
            jobs.append(dict(R0=R0, t0=t0, A=A, dep=dep)); ims += [lc, rc]
        return jobs, ims

    def solve(self, job, fl, fr):
        il, ir = self.f.match(fl, fr)
        if len(il) < self.min_inl: return None
        A = job["A"]
        pl = fl[0][il] @ A[:, :2].T + A[:, 2]; pr = fr[0][ir] @ A[:, :2].T + A[:, 2]
        dep = job["dep"]; H, W = dep.shape
        u = np.clip(np.round(pr[:, 0]).astype(int), 0, W - 1); v = np.clip(np.round(pr[:, 1]).astype(int), 0, H - 1)
        z = dep[v, u]; ok = z > 0
        if ok.sum() < self.min_inl: return None
        Xc = (np.c_[pr[ok], np.ones(ok.sum())] @ self.Kinv.T) * z[ok, None]
        Xb = (Xc - job["t0"]) @ job["R0"]       # body = R0^T (Xc - t0)
        rv0, _ = cv2.Rodrigues(job["R0"])
        okp, rv, tv, inl = cv2.solvePnPRansac(Xb.astype(np.float64), pl[ok].astype(np.float64), self.K, None, rv0.copy(), job["t0"].reshape(3, 1).copy(),
                                             useExtrinsicGuess=True, iterationsCount=100, reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)
        if not okp or inl is None or len(inl) < self.min_inl: return None
        inl = inl[:, 0]
        rv, tv = cv2.solvePnPRefineLM(Xb[inl], pl[ok][inl].astype(np.float64), self.K, None, rv, tv)
        return cv2.Rodrigues(rv)[0], tv[:, 0], len(inl)

    def track_begin(self, live_gray_list, priors, suns=None):
        """GL half of track() (iters == 1): reference renders + crops. Must run on the GL thread."""
        assert self.iters == 1
        jobs, ims = self.prepare(live_gray_list, list(priors), suns); return jobs, ims, len(priors)

    def track_end(self, st):
        """CPU half of track() (feature model + PnP); safe on a worker thread. Same output as track()."""
        jobs, ims, n = st; out = [None] * n; n_in = [0] * n
        feats = self.f.extract(ims) if ims else []
        k = 0
        for ci, job in enumerate(jobs):
            if job is None: continue
            res = self.solve(job, feats[k], feats[k + 1]); k += 2
            if res is not None: out[ci] = (res[0], res[1]); n_in[ci] = res[2]
        return out, n_in

    def track(self, live_gray_list, priors, suns=None):
        out = list(priors); cur = list(priors); n_in = [0] * len(priors)
        for _ in range(self.iters):
            jobs, ims = self.prepare(live_gray_list, cur, suns)
            feats = self.f.extract(ims) if ims else []
            k = 0
            for ci, job in enumerate(jobs):
                if job is None: out[ci] = None; continue
                res = self.solve(job, feats[k], feats[k + 1]); k += 2
                if res is None: out[ci] = None
                else: out[ci] = (res[0], res[1]); cur[ci] = out[ci]; n_in[ci] = res[2]
        return out, n_in
