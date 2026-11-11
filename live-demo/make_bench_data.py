"""Precompute a render-free, portable benchmark file (no OpenGL needed on the Mac):
for each live frame and prior level: live crop, reference-render crop at the prior (same sun / mismatched sun),
reference depth crop, crop->full affine, prior pose, truth pose, K."""
import numpy as np, cv2, render, tracker
S = np.load("/workspace/live-demo-env/set_512.npz"); K = S["K"]; CR = 256
r = render.SpaceRenderer(512); rm = render.SpaceRenderer(512, share=r, sun_dir_cam=(-0.2, -0.6, 0.7))
rng = np.random.default_rng(11)
out = {k: [] for k in ["live", "ref", "ref_mis", "dep", "dep_mis", "A", "R0", "t0", "R", "t", "prior_deg"]}
for i in range(len(S["ims"])):
    g = cv2.GaussianBlur(cv2.cvtColor(S["ims"][i], cv2.COLOR_RGB2GRAY), (0, 0), 0.6)
    for lvl in (5, 15):
        pr = S[f"prior{lvl}"][i]; R0, t0 = pr[:9].reshape(3, 3), pr[9:]
        col, dep = r.render(R0, t0); colm, depm = rm.render(R0, t0)
        c, s = tracker.roi_from_mask(dep > 0)
        lc, A = tracker.crop(g, c, s, CR)
        sc = s / CR; M = np.array([[1 / sc, 0, -(c[0] - s / 2) / sc], [0, 1 / sc, -(c[1] - s / 2) / sc]])
        rc = cv2.warpAffine(cv2.cvtColor(col, cv2.COLOR_RGB2GRAY), M, (CR, CR), flags=cv2.INTER_AREA)
        rcm = cv2.warpAffine(cv2.cvtColor(colm, cv2.COLOR_RGB2GRAY), M, (CR, CR), flags=cv2.INTER_AREA)
        dc = cv2.warpAffine(dep, M, (CR, CR), flags=cv2.INTER_NEAREST)
        for k, v in zip(out, [lc, rc, rcm, dc.astype(np.float16), dc.astype(np.float16), A, R0, t0, S["R"][i], S["t"][i], lvl]): out[k].append(v)
np.savez_compressed("bench_data_256.npz", K=K, **{k: np.stack(v) for k, v in out.items()})
print({k: np.stack(v).shape for k, v in out.items()})
