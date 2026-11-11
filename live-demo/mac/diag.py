"""Mac diagnostic for the live demo (also runs on Linux). Writes OUT/diag.json + PNG/NPY dumps and prints a summary.
 1. GL: renderer/version; one colour + depth render per camera-like pose; depth stats; colour-vs-depth mask agreement
    (catches flipped/misaligned readback) and depth vs the analytic range of the mesh centre.
 2. Each feature backend: input shape, compute units, per-call timing (10 calls), keypoints, output agreement with
    onnx-cpu on identical input, and known-pose tracking (render a pose, track from a 4 deg prior): matches/inliers/error."""
import os, sys, time, json, traceback, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); LD = os.path.dirname(HERE); sys.path.insert(0, LD); os.chdir(LD)
os.environ.setdefault("LIVE_DEMO_RENDER", "gl")
import render as R_, gl_render, features, tracker as T_
from scipy.spatial.transform import Rotation as Rot
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "diag_" + time.strftime("%Y%m%d-%H%M%S")); os.makedirs(OUT, exist_ok=True)
CROP = int(os.environ.get("DIAG_CROP", "256")); D = {"platform": sys.platform, "crop": CROP}
def log(*a): print(*a, flush=True)
# ---- 1. GL
sc = gl_render.GLScene(); D["gl"] = {k: sc.ctx.info.get(k) for k in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")}; D["gl"]["version_code"] = sc.ctx.version_code
m = R_.topex_mesh(); sc.add(m); K = R_.K_for(512, 30.0)
class Ref:   # same interface as demo.RefRenderer
    K = K; res = 512
    def render(self, Rm, t, sun_cam=None):
        s = np.array([-0.5, -0.3, -0.8]) if sun_cam is None else sun_cam
        return sc.render(K, 512, 512, Rm.T, -Rm.T @ t, Rm.T @ s, znear=0.3, zfar=2000)
ref = Ref(); rs = np.random.default_rng(7); D["renders"] = []
c_mesh = m.bounds.mean(0)
for i in range(3):
    Rt = Rot.random(random_state=rs).as_matrix(); tt = np.array([0.5 * (i - 1), 0.3, 26.0 + 4 * i])
    t0 = time.perf_counter(); col, dep = ref.render(Rt, tt); ms = (time.perf_counter() - t0) * 1000
    cv2.imwrite(f"{OUT}/cam{i}_color.png", col[:, :, ::-1]); np.save(f"{OUT}/cam{i}_depth.npy", dep)
    dv = dep.copy(); dv[dep > 0] = 255 - 200 * (dep[dep > 0] - dep[dep > 0].min()) / max(np.ptp(dep[dep > 0]), 1e-6); cv2.imwrite(f"{OUT}/cam{i}_depth.png", dv.astype(np.uint8))
    cm = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY) > 3; dm = dep > 0
    iou = float((cm & dm).sum() / max((cm | dm).sum(), 1)); iou_flip = float((cm[::-1] & dm).sum() / max((cm[::-1] | dm).sum(), 1))
    pc = Rt @ c_mesh + tt; uv = K @ pc; uv = uv[:2] / uv[2]
    D["renders"].append(dict(ms=round(ms, 2), color_px=int(cm.sum()), depth_px=int(dm.sum()), depth_min=float(dep[dm].min()) if dm.any() else None,
                             depth_max=float(dep.max()), mask_iou=round(iou, 4), mask_iou_if_flipped=round(iou_flip, 4), bbox_centre_range_m=float(pc[2]),
                             bbox_centre_px=[float(uv[0]), float(uv[1])], depth_mask_centroid_px=[float(np.nonzero(dm)[1].mean()), float(np.nonzero(dm)[0].mean())] if dm.any() else None))
    log("render", i, D["renders"][-1])
# ---- 2. backends
pose_set = []
for i in range(5):
    Rt = Rot.random(random_state=rs).as_matrix(); tt = np.array([0.3, -0.2, 30.0]); s_c = np.array([-0.5, -0.3, -0.8])
    col, dep = ref.render(Rt, tt, s_c); g = R_.degrade(col, rs)
    Rp = Rot.from_rotvec(rs.normal(size=3) / np.sqrt(3) * np.radians(4)).as_matrix() @ Rt; tp = tt + np.array([0.3, -0.3, 0.3])
    pose_set.append((Rt, tt, s_c, g, Rp, tp))
rc_, _ = T_.crop(cv2.cvtColor(ref.render(*pose_set[0][:2])[0], cv2.COLOR_RGB2GRAY), np.array([256, 256.]), 300, CROP)
cv2.imwrite(f"{OUT}/crop_ref0.png", rc_)
probe = [rc_] * 6
base = None; D["backends"] = []
for name in ("xfeat-onnx-cpu", "xfeat-coreml-all", "xfeat-coreml-ne", "xfeat-coreml-cpu", "xfeat-coreml-gpu", "xfeat-onnx-coreml", "xfeat-torch-mps", "orb"):
    r = dict(backend=name)
    try:
        t0 = time.perf_counter(); f = features.make(name, res=CROP, top_k=512) if name.startswith("xfeat-") else features.make(name); r["load_s"] = round(time.perf_counter() - t0, 2)
        if hasattr(f, "r"):
            r["input_shape"] = [f.r.B, 1, f.r.res, f.r.res]; r["runner"] = f.r.b
            if hasattr(f.r, "m"): r["compute_units"] = str(getattr(f.r.m, "compute_unit", None))
            if hasattr(f.r, "s"): r["ort_providers"] = f.r.s.get_providers()
        ts = []
        for k in range(11):
            t0 = time.perf_counter(); o = f.extract(probe); ts.append((time.perf_counter() - t0) * 1000)
        r["first_call_ms"] = round(ts[0], 2); r["call_ms_10"] = [round(x, 2) for x in ts[1:]]; r["call_ms_median"] = round(float(np.median(ts[1:])), 2)
        r["kp_per_image"] = int(len(o[0][0])); r["kp_xy_range"] = [float(np.min(o[0][0])), float(np.max(o[0][0]))] if len(o[0][0]) else None
        if name == "xfeat-onnx-cpu": base = o
        elif base is not None and name.startswith("xfeat-"):
            a, b = base[0][0], o[0][0]; d = np.linalg.norm(a[:, None] - b[None], axis=2).min(1)
            r["kp_agree_frac_within_1px_vs_onnx_cpu"] = round(float((d < 1.0).mean()), 3)
            il, ir = f.match(base[0], o[0]); r["self_matches_vs_onnx_cpu"] = int(len(il))
        il, ir = f.match(o[0], o[1]); r["self_matches_identical_crop"] = int(len(il))
        trk = T_.Tracker(ref, f, crop_res=CROP, iters=1); res = []
        for Rt, tt, s_c, g, Rp, tp in pose_set:
            out, nin = trk.track([g], [(Rp, tp)], [s_c])
            if out[0] is not None: out, nin = trk.track([g], [out[0]], [s_c])
            jobs, ims = trk.prepare([g], [(Rp, tp)], [s_c]); nm = len(f.match(*f.extract(ims)[:2])[0]) if ims else 0
            res.append(dict(matches=nm, inliers=int(nin[0]), rot_err_deg=round(R_.rot_err_deg(out[0][0], Rt), 3) if out[0] is not None else None))
        r["known_pose"] = res; r["known_pose_success"] = float(np.mean([x["rot_err_deg"] is not None and x["rot_err_deg"] < 5 for x in res]))
    except Exception as ex: r["error"] = repr(ex)[:400]; r["trace"] = traceback.format_exc()[-1500:]
    D["backends"].append(r); log(json.dumps({k: v for k, v in r.items() if k not in ("trace", "call_ms_10")})[:900])
json.dump(D, open(f"{OUT}/diag.json", "w"), indent=1, default=str); log("wrote", OUT)
