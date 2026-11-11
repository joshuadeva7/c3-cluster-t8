"""End-to-end closed loop on the box: truth sim -> 3 camera renders -> real feature model + PnP tracker -> MEKF
-> continuous TOPEX state (p, v, q, w) + docker-relative rate. Sim time advances by REAL wall time per cycle
(so the measurement rate shown is what the hardware actually sustains).
Cameras: imager A, imager B, docker, ~30 m from TOPEX, known poses (from cluster nav). Docker can spin to match."""
import os as _os; _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # OpenBLAS thread spinning slowed torch CPU 5x when interleaved
import argparse, time, json, numpy as np, cv2, render, features, tracker, mekf
from scipy.spatial.transform import Rotation as Rot
p = argparse.ArgumentParser()
p.add_argument("--feat", default="orb"); p.add_argument("--res", type=int, default=512); p.add_argument("--crop", type=int, default=256)
p.add_argument("--w", type=float, nargs=3, default=[0.02, 0.06, -0.03], help="TOPEX initial body rates rad/s (the sliders)")
p.add_argument("--T", type=float, default=40.0); p.add_argument("--iters", type=int, default=1); p.add_argument("--topk", type=int, default=512)
p.add_argument("--init_err", type=float, default=15.0, help="deg error of the initial hypothesis if --acquire is off")
p.add_argument("--acquire", action="store_true", help="global acquisition by multi-hypothesis tracking (no truth used)")
p.add_argument("--acq_group", default="I", help="O = 24 hypotheses, I = 60")
p.add_argument("--spinup", type=float, default=20.0, help="docker starts matching spin at this sim time (s)")
p.add_argument("--fixed_dt", type=float, default=0.0, help="if >0, advance sim by this instead of wall time (accuracy-only runs)")
p.add_argument("--mismatch", action="store_true", help="tracker reference renders use a different sun direction than the live cameras")
p.add_argument("--log", default="loop.csv"); p.add_argument("--json", default="")
a = p.parse_args()
rng = np.random.default_rng(3)
J = np.diag([2.6e4, 1.1e4, 2.9e4])   # placeholder inertia shape; replace with CAD mass properties
r = render.SpaceRenderer(a.res)
rref = render.SpaceRenderer(a.res, share=r, sun_dir_cam=(-0.2, -0.6, 0.7)) if a.mismatch else r
kw = {"top_k": a.topk, "res": a.crop} if a.feat.startswith("xfeat-") else {"top_k": a.topk} if a.feat == "xfeat" else ({"max_kp": a.topk} if a.feat == "sp_lg" else {"n": a.topk})
F = features.make(a.feat, **kw); T = tracker.Tracker(rref, F, crop_res=a.crop, iters=a.iters)

def look_at(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = target - pos; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    return np.c_[x, y, z]   # camera->room (OpenCV axes)
cams = {"imagerA": np.array([30.0, 0, 3]), "imagerB": np.array([-12, 27, -4.0]), "docker": np.array([3.0, -32, 1])}
camR = {k: look_at(v) for k, v in cams.items()}
R_IB = Rot.random(random_state=rng).as_matrix(); w = np.array(a.w, float); pB = np.zeros(3)
filt = mekf.TargetMEKF(J=J, sig_rot_meas=np.radians(1.5), sig_pos_meas=0.2)

def measure(R_IB, priors):
    ims = []
    for k in cams:
        Rc, pc = camR[k], cams[k]
        col, _ = r.render(Rc.T @ R_IB, Rc.T @ (pB - pc)); ims.append(render.degrade(col, rng))
    t0 = time.perf_counter(); out, nin = T.track(ims, priors); return out, nin, time.perf_counter() - t0, ims

# ---- acquisition ----
t_acq = time.perf_counter()
if a.acquire:
    # hypotheses: 24-cell rotation grid in imager A's frame, range from silhouette size; score = PnP inliers
    k0 = "imagerA"; Rc, pc = camR[k0], cams[k0]
    col, _ = r.render(Rc.T @ R_IB, Rc.T @ (pB - pc)); g = render.degrade(col, rng)
    m = g > 12; ys, xs = np.nonzero(m); cx, cy = xs.mean(), ys.mean(); ext = max(np.ptp(xs), np.ptp(ys))
    z = r.K[0, 0] * 9.0 / max(ext, 1)          # ~9 m characteristic size (CAD extent) -> range guess
    t_guess = z * np.linalg.inv(r.K) @ np.array([cx, cy, 1])
    best = (0, None)
    grid = Rot.create_group(a.acq_group).as_matrix()
    for Rg in grid:   # rotation-group hypotheses (no truth used)
        out, nin = T.track([g], [(Rg, t_guess)])
        for _ in range(2):
            if out[0] is None: break
            out, nin2 = T.track([g], [out[0]]); nin = nin2
        if out[0] is not None and nin[0] > best[0]: best = (nin[0], out[0])
    if best[1] is None: raise SystemExit(json.dumps(dict(feat=a.feat, acquire=True, acq_group=a.acq_group, acq_failed=True, acq_s=time.perf_counter() - t_acq)))
    Rcb, tcb = best[1]; filt.update(Rc @ Rcb, Rc @ tcb + pc)
else:
    Rp = Rot.from_rotvec(rng.normal(size=3) / np.sqrt(3) * np.radians(a.init_err)).as_matrix() @ R_IB
    filt.update(Rp, pB + rng.normal(size=3) * 0.5)
acq_s = time.perf_counter() - t_acq
acq_err = render.rot_err_deg(mekf.q2R(filt.q), R_IB)

# ---- loop ----
tsim, rows, wd_D, R_ID = 0.0, [], np.zeros(3), camR["docker"].copy()
cyc_lat, mod_lat, rej = [], [], 0
while tsim < a.T:
    c0 = time.perf_counter()
    priors = [filt.pose_in_camera(camR[k], cams[k]) for k in cams]
    out, nin, mlat, _ = measure(R_IB, priors)
    for k, o in zip(cams, out):
        if o is None: rej += 1; continue
        Rc, pc = camR[k], cams[k]
        if not filt.update(Rc @ o[0], Rc @ o[1] + pc): rej += 1
    dt = a.fixed_dt if a.fixed_dt > 0 else time.perf_counter() - c0
    cyc_lat.append(time.perf_counter() - c0); mod_lat.append(mlat)
    # advance truth + filter by dt (filter predicts at 240 Hz internally to show continuous output)
    n = max(1, int(np.ceil(dt * 240)))
    for _ in range(n): filt.predict(dt / n)
    R_IB, w = mekf.torque_free_step(R_IB, w, J, dt, n=max(2, n // 4)); tsim += dt
    # docker spin-matching (uses ESTIMATED state only): docker attitude rate = estimated target inertial rate
    wI_est = mekf.q2R(filt.q) @ filt.w
    if tsim > a.spinup: wd_D = R_ID.T @ wI_est
    th = R_ID @ wd_D * dt; R_ID = Rot.from_rotvec(th).as_matrix() @ R_ID
    w_rel_est = mekf.relative_rate_body(filt.w, mekf.q2R(filt.q), wd_D, R_ID)
    w_rel_true = mekf.relative_rate_body(w, R_IB, wd_D, R_ID)
    rows.append([tsim, render.rot_err_deg(mekf.q2R(filt.q), R_IB), np.degrees(np.linalg.norm(filt.w - w)), np.linalg.norm(filt.p - pB),
                 np.degrees(np.linalg.norm(w_rel_est)), np.degrees(np.linalg.norm(w_rel_true)), dt])
h = np.array(rows); np.savetxt(a.log, h, delimiter=",", header="t,att_err_deg,rate_err_deg_s,pos_err_m,rel_rate_est_deg_s,rel_rate_true_deg_s,cycle_s")
last = h[h[:, 0] > a.T - 10]
res = dict(feat=a.feat, acq_group=a.acq_group, mismatch=a.mismatch, res=a.res, crop=a.crop, iters=a.iters, w0=a.w, w0_deg_s=float(np.degrees(np.linalg.norm(a.w))), acquire=a.acquire, acq_s=acq_s, acq_att_err_deg=acq_err,
           cycles=len(h), meas_hz_3cam=1 / float(np.median(cyc_lat)), model_ms_3cam=1000 * float(np.median(mod_lat)), cycle_ms=1000 * float(np.median(cyc_lat)),
           rejected_frac=rej / (3 * len(h)), att_err_deg_last10s=float(last[:, 1].mean()), rate_err_deg_s_last10s=float(last[:, 2].mean()),
           pos_err_m_last10s=float(last[:, 3].mean()), rel_rate_est_deg_s_last10s=float(last[:, 4].mean()), rel_rate_true_deg_s_last10s=float(last[:, 5].mean()),
           fixed_dt=a.fixed_dt)
print(json.dumps(res))
if a.json: open(a.json, "a").write(json.dumps(res) + "\n")
