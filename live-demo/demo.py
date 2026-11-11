"""SWARM x TOPEX live demo.
  cutscene (v2 video up to prox-ops) -> live scene: TOPEX tumbles at the slider rates (true torque-free rigid body,
  CAD-derived inertia); imager A, imager B and the docker each render a camera view of the real CAD; the real XFeat model
  + PnP tracks TOPEX in each view; a MEKF fuses all fixes into a continuous TOPEX state; the docker's autonomy uses ONLY that
  estimate: hold 30 m -> spin-match -> final approach -> grip clear of the dish -> detumble -> stable -> back off.
Controls (window mode): 3 sliders = TOPEX body rates (rad/s, team design max 1 rad/s), G = go, X = inject tracking loss,
  L = rotate the sun, R = reset, SPACE = skip cutscene, Q/ESC = quit.
Headless: --headless --record out.mp4 --script "0:w=0.05,0.66,0.03;3:go;45:light;60:loss" (sim advances fixed dt per frame)."""
import os as _os; _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, json, math, argparse, collections
import numpy as np, cv2
import render as R_, tracker as T_, features, mekf
import trimesh, gl_render, ui_window
from scipy.spatial.transform import Rotation as Rot

HERE = _os.path.dirname(_os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--feat", default="xfeat-onnx-cpu" if sys.platform.startswith("linux") else "xfeat-coreml-gpu")
p.add_argument("--crop", type=int, default=192 if sys.platform.startswith("linux") else 256); p.add_argument("--camres", type=int, default=512)
p.add_argument("--headless", action="store_true"); p.add_argument("--record", default="")
p.add_argument("--script", default=""); p.add_argument("--T", type=float, default=120.0)
p.add_argument("--fixed_dt", type=float, default=0.0, help="sim seconds per frame (0 = real time from the wall clock)")
p.add_argument("--no_cutscene", action="store_true"); p.add_argument("--wmax", type=float, default=1.0)
p.add_argument("--approach_speed", type=float, default=1.0, help="m/s final approach in the demo (mission: 0.03 m/s)")
p.add_argument("--detumble_warp", type=float, default=600.0); p.add_argument("--fps_out", type=int, default=10)
p.add_argument("--metrics", default=""); p.add_argument("--seed", type=int, default=0)
p.add_argument("--sun_unknown", action="store_true", help="tracker references use a fixed default sun (lighting-mismatch stress)")
a = p.parse_args()
rng = np.random.default_rng(a.seed)

# ------------------------------------------------------------------ assets
cad = np.load(_os.path.join(HERE, "assets", "cad_render.npz"))
INR = json.load(open(_os.path.join(HERE, "assets", "topex_inertia.json")))
COM = np.array(INR["com_body_m"]); J = np.array(INR["J_used"])        # CAD tensor scaled to team Iz (see derive_inertia.py)
tm = R_.topex_mesh(); tm.vertices -= COM                               # body origin = centre of mass
DOCK_PB = np.array([0.0, -3.877, 0.0]) - COM                           # v2 docked position (prox.csv) in TOPEX body
DOCK_RB = R_.Rot.from_euler("y", -90, degrees=True).as_matrix()        # v2 DOCK_CLOCK: claw rolled clear of the dish
HOLD_D = 30.0; TOPEX_MASS = INR["mass_kg"]
def swarm_mesh(accent, keys=("shell", "inner", "arr_dep", "arm_ext", "arm_end")):
    parts = []
    for key, col in (("shell", (0.92, 0.92, 0.94)), ("inner", (0.80, 0.62, 0.30)), ("arr_dep", (0.10, 0.14, 0.30)), ("arm_ext", accent), ("arm_end", (0.85, 0.85, 0.85))):
        if key not in keys: continue
        m = trimesh.Trimesh(cad[key + "_v"], cad[key + "_f"], process=False)
        m.visual.face_colors = np.c_[np.tile((np.array(col) * 255).astype(np.uint8), (len(m.faces), 1)), np.full(len(m.faces), 255, np.uint8)]
        parts.append(m)
    return trimesh.util.concatenate(parts)
ACC = {"docker": (0.95, 0.55, 0.20), "imagerA": (0.30, 0.80, 0.85), "imagerB": (0.30, 0.65, 0.95)}
SW = {k: swarm_mesh(v, ("shell", "inner", "arr_dep") if k == "docker" else ("shell", "inner", "arr_dep", "arm_ext", "arm_end")) for k, v in ACC.items()}
DOCK_ARM = swarm_mesh(ACC["docker"], ("arm_ext", "arm_end"))
ARM_CLOSED = 1.211 / 2.611   # TRR: grip span 2611 mm open -> 1211 mm closed (arm telescopes along docker x)
J_DOCK = np.diag([36.68, 58.88, 41.62])   # trr/inertia.py, arm open, mapped team (X arm-normal, Y wing, Z arm) -> GLB (y, z, x)
M_DOCK = 166.036
PADS = cad["arm_end_v"]
# dish-clearance check geometry: TOPEX parts that are neither the bus nor the solar array (antenna dish + booms)
# TOPEX dish = the largest appendage outside the bus box that is not the solar array (1,008-triangle dish on its boom, -x side)
_m0 = R_.topex_mesh(); _c = _m0.triangles_center
_out = ((np.abs(_c[:, 0]) > 1.7) | (np.abs(_c[:, 2]) > 1.75) | (_c[:, 1] < -3.4) | (_c[:, 1] > 2.9)) & ~(_c[:, 2] > 2.5)
_dish = max(_m0.submesh([np.nonzero(_out)[0]], append=True).split(only_watertight=False), key=lambda c: c.area)
DISH_PTS = np.asarray(_dish.vertices) - COM
TOPEX_PTS = np.asarray(tm.vertices)
from scipy.spatial import cKDTree
TOPEX_KD = cKDTree(trimesh.sample.sample_surface(tm, 150000, seed=0)[0])
FINGERS = PADS[np.abs(PADS[:, 0]) > 1.30]          # claw finger tips (docker frame); they close along docker x
J_TRUE = J.copy()                                   # truth inertia: TOPEX alone, or the gripped stack

# ------------------------------------------------------------------ rendering (one GL context: the window's, or standalone when headless)
def T44(Rm, t): T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = t; return T
WIN_W, WIN_H = 1280, 800
DISP = ui_window.Display(WIN_W, WIN_H, a.headless, wmax=a.wmax)
RES = a.camres; K = R_.K_for(RES, 30.0)
MAINRES = 512; Km = R_.K_for(MAINRES, 45.0)
world = gl_render.GLScene(DISP.ctx)
n_top = world.add(tm)
n_sc = {k: world.add(m) for k, m in SW.items()}
n_arm = world.add(DOCK_ARM)
class RefRenderer:   # TOPEX-only reference renders in a camera frame, lit by the camera's known sun vector
    def __init__(self):
        self.K = K; self.res = RES; self.scene = gl_render.GLScene(DISP.ctx); self.scene.add(tm)
    def render(self, Rm, t, sun_cam=None):   # (Rm, t) = TOPEX body -> camera
        s = sun_cam if sun_cam is not None else np.array([-0.5, -0.3, -0.8])
        return self.scene.render(K, RES, RES, Rm.T, -Rm.T @ t, Rm.T @ s, znear=0.3, zfar=2000)
ref = RefRenderer()
Kn = R_.K_for(RES, 62.0)     # docker near-field camera (range < 11 m): wide FOV, close near plane, illuminator on
ARMRES = 256; Ka = R_.K_for(ARMRES, 75.0)
def render_view(R_IC, p_IC, main=False, own=None, Kc=None, head=0.0, znear=0.3):
    for k, n in n_sc.items(): world.set_visible(n, k != own)   # a camera never sees its own bus
    world.set_visible(n_arm, own != "docker")
    if main: return world.render(Km, MAINRES, MAINRES, R_IC, p_IC, W.sun, znear=0.3, zfar=4000)
    Kc = K if Kc is None else Kc; n_ = ARMRES if Kc is Ka else RES
    return world.render(Kc, n_, n_, R_IC, p_IC, W.sun, znear=znear, zfar=2000, headlight=head)
def auto_expose(g, target=200.0):   # display auto-exposure: 97th percentile of lit pixels -> target
    m = g > 6
    if m.sum() < 30: return g
    gain = float(np.clip(target / max(np.percentile(g[m], 97), 1.0), 0.3, 5.0))
    return np.clip(g.astype(np.float32) * gain, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ truth state
def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos; z /= np.linalg.norm(z); x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6: x = np.cross(z, [1, 0, 0])
    x /= np.linalg.norm(x); return np.c_[x, np.cross(z, x), z]
SWARM_CAM_IN_BODY = np.c_[[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # camera z along SWARM +y (gripper side), image-down = body -z
class World:
    def __init__(self):
        self.t = 0.0; self.R_IB = Rot.from_euler("xyz", [8, 0, 5], degrees=True).as_matrix(); self.w = np.zeros(3)
        self.sun = np.array([0.55, -0.35, 0.75]); self.sun /= np.linalg.norm(self.sun)
        self.sc = {"imagerA": np.array([26.0, 8.0, 6.0]), "imagerB": np.array([-20.0, 12.0, -14.0]), "docker": np.array([0.0, -HOLD_D, 0.0])}
        self.R_sc = {k: look_at(v, np.zeros(3)) @ SWARM_CAM_IN_BODY.T for k, v in self.sc.items()}
        self.w_dock = np.zeros(3); self.attached = False
    def cam_pose(self, k):   # camera->world rotation and position for spacecraft k
        return self.R_sc[k] @ SWARM_CAM_IN_BODY, self.sc[k]
W = World()
ARM_S = 1.0   # arm span scale (1 = open)
def set_scene_poses():
    world.set_pose(n_top, T44(W.R_IB, np.zeros(3)))
    for _k in W.sc: world.set_pose(n_sc[_k], T44(W.R_sc[_k], W.sc[_k]))
    world.set_pose(n_arm, T44(W.R_sc["docker"], W.sc["docker"]) @ np.diag([ARM_S, 1, 1, 1]))
set_scene_poses()
filt = mekf.TargetMEKF(J=J, sig_rot_meas=np.radians(1.2), sig_pos_meas=0.2, gate_deg=20.0)
kw = {"top_k": 512, "res": a.crop} if a.feat.startswith("xfeat-") else {}
def selftest(trk, n=5):
    """known pose -> render -> track from a 4 deg / 0.5 m perturbed prior. Returns (ok, inliers, rot err deg, model ms/call)."""
    rs = np.random.default_rng(123); res = []
    for i in range(n):
        Rt = Rot.random(random_state=rs).as_matrix(); tt = np.array([0.3, -0.2, 30.0]); s_c = np.array([-0.5, -0.3, -0.8])
        col, dep = ref.render(Rt, tt, s_c); g = R_.degrade(col, rs)
        Rp = Rot.from_rotvec(rs.normal(size=3) / np.sqrt(3) * np.radians(4)).as_matrix() @ Rt; tp = tt + np.array([0.3, -0.3, 0.3])
        out, nin = trk.track([g], [(Rp, tp)], [s_c]); out, nin = (trk.track([g], [out[0]], [s_c]) if out[0] is not None else (out, nin))
        res.append((nin[0], R_.rot_err_deg(out[0][0], Rt) if out[0] is not None else 99.0, int((dep > 0).sum())))
    t0 = time.perf_counter()
    for _ in range(5): trk.f.extract([np.zeros((a.crop, a.crop), np.uint8)] * 6)
    ms = (time.perf_counter() - t0) / 5 * 1000
    inl = int(np.median([r[0] for r in res])); err = float(np.median([r[1] for r in res])); px = int(np.median([r[2] for r in res]))
    good = sum(1 for r in res if r[0] >= 15 and r[1] < 3.0)            # healthy on >= 3 of 5 poses (no one-pose fluke)
    return good >= 3 and px > 100, inl, err, ms, px
FEAT = None; TRK = None; SELFTEST = []
for cand in [a.feat] + [b for b in ("xfeat-coreml-gpu", "xfeat-onnx-coreml", "xfeat-coreml-cpu", "xfeat-coreml-all32", "xfeat-torch-mps", "xfeat-onnx-cpu", "orb") if b != a.feat]:
    try:   # fall back through backends if one can't load OR fails the known-pose self-test on this machine (logged)
        f_ = features.make(cand, **(kw if cand.startswith("xfeat-") else {})); f_.extract([np.zeros((a.crop, a.crop), np.uint8)] * 2)
        trk_ = T_.Tracker(ref, f_, crop_res=a.crop, iters=1); ok_, inl_, err_, ms_, px_ = selftest(trk_)
        SELFTEST.append(dict(backend=cand, ok=ok_, inliers=inl_, rot_err_deg=round(err_, 3), extract_ms_6img=round(ms_, 2), ref_depth_px=px_))
        print(f"[selftest {cand}: ok={ok_} inliers={inl_} rot_err={err_:.2f} deg extract(6 crops)={ms_:.1f} ms ref_depth_px={px_}]", flush=True)
        if ok_: FEAT, TRK = f_, trk_; a.feat = cand; break
    except Exception as ex: print(f"[backend {cand} unavailable: {ex!r}]"[:300]); SELFTEST.append(dict(backend=cand, error=repr(ex)[:200]))
if TRK is None: raise SystemExit("no feature backend passed the self-test (see [selftest] lines); run mac/diag.sh")
CAMS = ["imagerA", "imagerB", "docker"]

# ------------------------------------------------------------------ autonomy (estimate only)
class Autonomy:
    PH = ["ACQUIRE", "CONVERGE", "HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH", "GRIP", "DETUMBLE", "STABLE", "RELEASE", "BACK OFF", "DONE"]
    def __init__(self): self.ph = "WAIT FOR GO"; self.t_ph = 0.0; self.go = False; self.conv_t = 0.0; self.d = HOLD_D; self.blend = 0.0; self.log = []
    def set(self, ph, t): self.ph = ph; self.t_ph = t; self.log.append((round(t, 2), ph)); print(f"[t={t:7.2f}] {ph}", flush=True)
A = Autonomy()

def acquire(cam="imagerA", group="I"):
    Rc, pc = W.cam_pose(cam); col, _ = render_view(Rc, pc, own=cam); g = R_.degrade(col, rng)
    m = g > 12
    if m.sum() < 50: return False
    ys, xs = np.nonzero(m); cx, cy = xs.mean(), ys.mean(); ext = max(np.ptp(xs), np.ptp(ys))
    z = K[0, 0] * 9.0 / max(ext, 1); tg = z * np.linalg.inv(K) @ np.array([cx, cy, 1]); sun_c = Rc.T @ W.sun
    best = (0, None); sc_ = None if a.sun_unknown else sun_c
    hyps = [(Rg, tg) for Rg in Rot.create_group(group).as_matrix()]; nin_h = [0] * len(hyps)
    for it in range(3):   # batched: 3 hypotheses per model call (the compiled batch is 6 crops = 3 live/ref pairs)
        keep = [i for i, h in enumerate(hyps) if h is not None]
        for c0 in range(0, len(keep), 3):
            ch = keep[c0:c0 + 3]; out, nin = TRK.track([g] * len(ch), [hyps[i] for i in ch], [sc_] * len(ch))
            for j, i in enumerate(ch): hyps[i] = out[j]; nin_h[i] = nin[j] if out[j] is not None else 0
    for h, n_ in zip(hyps, nin_h):
        if h is not None and n_ > best[0]: best = (n_, h)
    if best[1] is None or best[0] < 25: return False
    filt.q = None; filt.w = np.zeros(3); filt.P = np.diag([0.3] * 3 + [0.5] * 3) ** 2; filt.Pp = np.diag([5.0] * 3 + [1.0] * 3) ** 2; filt.v = np.zeros(3)
    filt.update(Rc @ best[1][0], Rc @ best[1][1] + pc); return True

# ------------------------------------------------------------------ dashboard
F = cv2.FONT_HERSHEY_SIMPLEX
def txt(img, s, x, y, sc=0.45, col=(230, 230, 230), th=1): cv2.putText(img, s, (int(x), int(y)), F, sc, col, th, cv2.LINE_AA)
HIST = collections.deque(maxlen=600); FULL = []
def draw_axes_overlay(img, Rcb, tcb, col, scale, Kc=None):
    P = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3.0]])
    X = P @ Rcb.T + tcb
    if (X[:, 2] < 0.5).any(): return
    u = X @ (K if Kc is None else Kc).T; u = u[:, :2] / u[:, 2:] * scale
    if not np.isfinite(u).all() or np.abs(u).max() > 5000: return
    for j, c in zip((1, 2, 3), ((60, 60, 255), (60, 255, 60), (255, 120, 60))):
        cv2.line(img, tuple(u[0].astype(int)), tuple(u[j].astype(int)), c if col is None else col, 2 if col is None else 1, cv2.LINE_AA)
def plot(img, x0, y0, w, h, series, cols, lo, hi, label):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1); txt(img, label, x0 + 4, y0 + 14, 0.4, (200, 200, 200))
    if len(HIST) < 2: return
    for s, c in zip(series, cols):
        v = np.array([h_[s] for h_ in HIST], float); n = len(v)
        xs = x0 + np.linspace(0, w, HIST.maxlen)[-n:]; ys = y0 + h - (np.clip(v, lo, hi) - lo) / (hi - lo) * h
        ok = np.isfinite(ys)
        if ok.sum() < 2: continue
        pts = np.c_[xs[ok], ys[ok]].astype(np.int32)
        cv2.polylines(img, [pts], False, c[:3], 1, cv2.LINE_AA)

# ------------------------------------------------------------------ script / UI
def parse_script(s):
    ev = []
    for it in s.split(";"):
        if not it.strip(): continue
        t, cmd = it.split(":", 1); ev.append((float(t), cmd.strip()))
    return sorted(ev)
EV = parse_script(a.script)
sliders = {"wx": 0.0, "wy": 0.0, "wz": 0.0}
def apply_cmd(cmd):
    if cmd.startswith("w="):
        v = [float(x) for x in cmd[2:].split(",")]; v = np.clip(v, -a.wmax, a.wmax)
        if A.ph in ("WAIT FOR GO",): W.w = np.array(v, float)
        sliders.update(wx=v[0], wy=v[1], wz=v[2])
    elif cmd == "go": A.go = True
    elif cmd == "loss":   # corrupt the estimate by 60 deg: the pipeline must notice and re-acquire
        if filt.q is not None: filt.q = mekf.qmul(filt.q, mekf.qexp(np.radians(60) * np.array([0.6, 0.0, 0.8]))); STATS["loss_t"].append(round(W.t, 2))
    elif cmd == "light": W.sun = Rot.from_euler("z", 75, degrees=True).as_matrix() @ W.sun; STATS["light_t"].append(round(W.t, 2))
STATS = {"dock_v": [], "dock_w": [], "loss_t": [], "light_t": [], "reacq_t": [], "fix_hz": [], "cycle_ms": [], "model_ms": []}
UI_SLIDERS = {sl.name: sl for sl in DISP.sliders}
writer = None
if a.record:
    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), a.fps_out, (WIN_W, WIN_H))

def play_cutscene():
    cap = cv2.VideoCapture(_os.path.join(HERE, "assets", "cutscene.mp4")); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr, (WIN_W, int(WIN_W * fr.shape[0] / fr.shape[1]))); canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
        y0 = (WIN_H - fr.shape[0]) // 2; canvas[y0:y0 + fr.shape[0]] = fr; txt(canvas, "CUTSCENE (v2 mission video)  -  SPACE to skip", 20, WIN_H - 20, 0.5, (160, 160, 160))
        if writer is not None and n % max(1, round(fps / a.fps_out)) == 0: writer.write(canvas)
        n += 1
        if not a.headless:
            DISP.draw_controls(canvas, True); t_show = time.perf_counter()
            if any(e[0] == "key" and e[1] in (" ", "q") for e in DISP.show(canvas)): break
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t_show)))
    cap.release()
if not a.no_cutscene: play_cutscene()

# ------------------------------------------------------------------ main loop
lost_count = 0; last_wall = time.perf_counter(); fix_times = collections.deque(maxlen=30); ev_i = 0
grip_report = None; frame = 0; MAIN_T = np.array([0, -13.0, 0]); MAIN_P = np.array([30.0, -38.0, 14.0])
t_demo = 0.0
while t_demo < a.T and A.ph != "DONE" or (A.ph == "DONE" and W.t < A.t_ph + 3):
    c0 = time.perf_counter()
    while ev_i < len(EV) and EV[ev_i][0] <= W.t: apply_cmd(EV[ev_i][1]); ev_i += 1
    if not a.headless and A.ph == "WAIT FOR GO":
        v = [UI_SLIDERS[nm].val for nm in ("wx", "wy", "wz")]; W.w = np.array(v); sliders.update(wx=v[0], wy=v[1], wz=v[2])
    # ---- cameras (truth render) + estimation
    live, Rcs, pcs, use = [], [], [], []
    Kcs = []
    for k in CAMS:
        Rc, pc = W.cam_pose(k)
        near = k == "docker" and np.linalg.norm(W.sc["docker"] - (filt.p if filt.q is not None else np.zeros(3))) < 11.0
        Kc = Kn if near else K; Kcs.append(Kc)
        col, _ = render_view(Rc, pc, own=k, Kc=Kc, head=0.45 if near else 0.0, znear=0.05 if near else 0.3)
        live.append(col); Rcs.append(Rc); pcs.append(pc)
    grays = [R_.degrade(c, rng) for c in live]
    fixes = [None] * 3; nin = [0] * 3; t_model = 0.0
    if A.ph == "WAIT FOR GO" and not A.go:
        pass
    elif W.attached:
        pass   # gripped: TOPEX state comes from the docker's IMU through the rigid grip
    elif filt.q is None or A.ph == "ACQUIRE":
        if A.ph != "ACQUIRE": A.set("ACQUIRE", W.t)
        t0 = time.perf_counter(); ok = acquire(); t_model = time.perf_counter() - t0
        if ok:
            if STATS["loss_t"] and (not STATS["reacq_t"] or STATS["reacq_t"][-1] < STATS["loss_t"][-1]): STATS["reacq_t"].append(round(W.t, 2))
            A.set(A.resume if hasattr(A, "resume") and A.resume else "CONVERGE", W.t); A.resume = None
    else:
        idx = [i for i, k in enumerate(CAMS) if not (k == "docker" and (W.attached or np.linalg.norm(filt.p - W.sc["docker"]) < 11.0))]
        pri = [filt.pose_in_camera(Rcs[i], pcs[i]) for i in idx]
        suns = [None if a.sun_unknown else Rcs[i].T @ W.sun for i in idx]
        t0 = time.perf_counter(); out, nn = TRK.track([grays[i] for i in idx], pri, suns); t_model = time.perf_counter() - t0
        accepted = 0
        for j, i in enumerate(idx):
            if out[j] is None: continue
            fixes[i] = out[j]; nin[i] = nn[j]
            if filt.update(Rcs[i] @ out[j][0], Rcs[i] @ out[j][1] + pcs[i]): accepted += 1
        if accepted: fix_times.append(W.t); lost_count = 0
        else: lost_count += 1
        if lost_count >= 4:   # estimate no longer explains any camera -> re-acquire (autonomy pauses where it was)
            A.resume = A.ph if A.ph not in ("ACQUIRE", "CONVERGE") else None; A.set("ACQUIRE", W.t); filt.q = None; lost_count = 0
    # ---- time step
    wall = time.perf_counter(); dt = a.fixed_dt if a.fixed_dt > 0 else min(wall - last_wall, 0.25); last_wall = wall
    warp = a.detumble_warp if A.ph == "DETUMBLE" else (10.0 if A.ph == "BACK OFF" else 1.0); dts = dt * warp
    # truth: torque-free (+ detumble torque once gripped)
    if A.ph == "DETUMBLE":
        # TIME WARP applies to the detumble torque only: the stack's attitude advances by the real frame time at its true
        # instantaneous rate (smooth, no aliasing), while one 1 N thruster x 0.5 m (v2 / TRR) fired against H removes
        # 0.5 N*m*s of |H| per (warped) second. Euler dynamics of the stack by RK4.
        W.R_IB, W.w = mekf.torque_free_step(W.R_IB, W.w, J_TRUE, dt, n=max(4, int(dt * 120)))
        HI = W.R_IB @ (J_TRUE @ W.w); hn = np.linalg.norm(HI)
        HI = HI * max(0.0, hn - 0.5 * dts) / hn if hn > 1e-12 else HI * 0
        W.w = np.linalg.solve(J_TRUE, W.R_IB.T @ HI)
    else:
        W.R_IB, W.w = mekf.torque_free_step(W.R_IB, W.w, J_TRUE, dts, n=max(2, int(dts * 120)))
    if filt.q is not None:
        if W.attached:   # gripped: docker IMU + arm geometry give TOPEX attitude/rate directly (no vision needed)
            filt.q = mekf.R2q(W.R_IB); filt.w = W.w.copy()
        else:
            nn_ = max(1, int(dts * 240))
            for _ in range(nn_): filt.predict(dts / nn_)
    W.t += dts; t_demo += dt
    # ---- autonomy (estimate only)
    est_ok = filt.q is not None
    if est_ok: R_est, w_est, p_est = mekf.q2R(filt.q), filt.w.copy(), filt.p.copy()
    sig_w = np.degrees(np.sqrt(np.diag(filt.P)[3:].max())) if est_ok else 99
    if A.ph == "CONVERGE":
        A.conv_t = A.conv_t + dts if sig_w < 0.5 else 0.0
        if A.conv_t > 3.0: A.set("HOLD 30 m", W.t)
    elif A.ph == "HOLD 30 m" and W.t - A.t_ph > 2.0: A.set("SPIN-MATCH", W.t); A.blend = 0.0; A.start = (W.sc["docker"].copy(), W.R_sc["docker"].copy())
    # docker desired pose from the ESTIMATE
    if est_ok and A.ph in ("SPIN-MATCH", "FINAL APPROACH", "GRIP"):
        off = DOCK_PB * (A.d / np.linalg.norm(DOCK_PB))
        p_des = p_est + R_est @ off; R_des = R_est @ DOCK_RB
        u = 1.0
        if A.ph == "SPIN-MATCH":
            A.blend = min(1.0, A.blend + dts / 8.0); u = A.blend * A.blend * (3 - 2 * A.blend)
            p_des = (1 - u) * A.start[0] + u * p_des
            R_des = (Rot.from_matrix(A.start[1]) * Rot.from_rotvec(u * (Rot.from_matrix(A.start[1]).inv() * Rot.from_matrix(R_des)).as_rotvec())).as_matrix()
            w_rel_est = np.degrees(np.linalg.norm(mekf.relative_rate_body(w_est, R_est, W.w_dock, W.R_sc["docker"])))
            if A.blend >= 1.0 and w_rel_est < 0.5: A.set("FINAL APPROACH", W.t)
        elif A.ph == "FINAL APPROACH":
            A.d = max(np.linalg.norm(DOCK_PB), A.d - a.approach_speed * dts)
            if A.d <= np.linalg.norm(DOCK_PB) + 1e-6: A.set("GRIP", W.t)
        elif A.ph == "GRIP" and not getattr(A, "closed", False):
            # claw closes (2611 -> 1211 mm span over ~3 s) until a finger tip is 2 cm from the TOPEX surface
            ARM_S = max(ARM_CLOSED, ARM_S - dt * (1 - ARM_CLOSED) / 3.0)
            fb = (W.sc["docker"] + (FINGERS * [ARM_S, 1, 1]) @ W.R_sc["docker"].T) @ W.R_IB
            if TOPEX_KD.query(fb)[0].min() < 0.02: A.claw_contact = True; A.closed = True
            if ARM_S <= ARM_CLOSED + 1e-6: A.closed = True
        if A.ph == "GRIP" and getattr(A, "closed", False):
            # contact assessment against TRUTH (what physically happened), decided by the estimate
            pT = W.R_IB.T @ (W.sc["docker"] - 0.0)   # docker position in true TOPEX body
            pads_world = W.sc["docker"] + (PADS * [ARM_S, 1, 1]) @ W.R_sc["docker"].T
            pads_body = (pads_world) @ W.R_IB
            dmin_dish = float(np.min(np.linalg.norm(DISH_PTS[None, :, :] - pads_body[::4, None, :], axis=2)))
            dmin_topex = float(np.min(np.linalg.norm(TOPEX_PTS[None, ::3, :] - pads_body[::8, None, :], axis=2)))
            grip_report = dict(t=round(W.t, 2), grip_pos_err_m=float(np.linalg.norm(pT - DOCK_PB)),
                               grip_att_err_deg=R_.rot_err_deg(W.R_IB.T @ W.R_sc["docker"], DOCK_RB),
                               rel_rate_true_deg_s=float(np.degrees(np.linalg.norm(mekf.relative_rate_body(W.w, W.R_IB, W.w_dock, W.R_sc["docker"])))),
                               min_pad_to_dish_m=dmin_dish, min_pad_to_topex_m=dmin_topex, topex_rate_deg_s=float(np.degrees(np.linalg.norm(W.w))),
                               claw_span_m=round(2.611 * ARM_S, 3), claw_contact=bool(getattr(A, "claw_contact", False)))
            # rigid attach at the ACTUAL relative pose; truth becomes the stack (docker inertia + parallel axis; COM shift neglected)
            A.rel = (W.R_IB.T @ W.R_sc["docker"], W.R_IB.T @ W.sc["docker"]); r_ = A.rel[1]
            Jst = J + A.rel[0] @ J_DOCK @ A.rel[0].T + M_DOCK * (r_ @ r_ * np.eye(3) - np.outer(r_, r_))
            H0 = J @ W.w + A.rel[0] @ J_DOCK @ (W.w_dock) + M_DOCK * np.cross(r_, np.cross(W.w, r_))
            J_TRUE = Jst; W.w = np.linalg.solve(Jst, H0)            # angular momentum conserved through the grip
            grip_report["stack_rate_after_grip_deg_s"] = float(np.degrees(np.linalg.norm(W.w)))
            grip_report["stack_H_Nms"] = float(np.linalg.norm(Jst @ W.w)); grip_report["detumble_est_s"] = grip_report["stack_H_Nms"] / 0.5
            W.attached = True; A.set("DETUMBLE", W.t)
        # docker control: feed-forward of the ESTIMATED TOPEX motion + feedback on pose error (0.4 s time constant)
        k_ = min(1.0, dts / 0.4)
        wI_ff = R_est @ w_est if A.ph != "SPIN-MATCH" else (R_est @ w_est) * u
        vI_ff = filt.v + np.cross(R_est @ w_est, p_des - p_est) * (1.0 if A.ph != "SPIN-MATCH" else u)
        if A.ph == "FINAL APPROACH": vI_ff = vI_ff - (R_est @ (DOCK_PB / np.linalg.norm(DOCK_PB))) * a.approach_speed
        err = (Rot.from_matrix(R_des) * Rot.from_matrix(W.R_sc["docker"]).inv()).as_rotvec()   # world-frame attitude error
        w_world = wI_ff + err * k_ / max(dts, 1e-6)
        W.R_sc["docker"] = Rot.from_rotvec(w_world * dts).as_matrix() @ W.R_sc["docker"]; W.w_dock = W.R_sc["docker"].T @ w_world
        W.sc["docker"] = W.sc["docker"] + vI_ff * dts + (p_des - W.sc["docker"] - vI_ff * dts) * k_
        STATS["dock_v"].append(float(np.linalg.norm(vI_ff))); STATS["dock_w"].append(float(np.degrees(np.linalg.norm(w_world))))
    elif W.attached:
        W.R_sc["docker"] = W.R_IB @ A.rel[0]; W.sc["docker"] = W.R_IB @ A.rel[1]; W.w_dock = A.rel[0].T @ W.w
        if A.ph == "DETUMBLE" and np.linalg.norm(W.w) < 0.00175: A.set("STABLE", W.t)
        elif A.ph == "STABLE" and W.t - A.t_ph > 5: A.set("RELEASE", W.t)
        elif A.ph == "RELEASE":   # claw opens (2 s), then the docker lets go
            ARM_S = min(1.0, ARM_S + dt * (1 - ARM_CLOSED) / 2.0)
            if ARM_S >= 1.0: W.attached = False; J_TRUE = J.copy(); A.set("BACK OFF", W.t); A.bo_v = 0.0
    elif A.ph == "BACK OFF":   # straight back along the arm axis: 0.1 m/s (TRR prox-ops +-0.1 m/s), gentle accel; shown at x10
        A.bo_v = min(0.10, A.bo_v + 0.005 * dts)
        W.sc["docker"] = W.sc["docker"] + W.R_sc["docker"] @ np.array([0, -1.0, 0]) * A.bo_v * dts; W.w_dock = np.zeros(3)
        if np.linalg.norm(W.sc["docker"]) > 20: A.set("DONE", W.t)
    elif A.go and A.ph == "WAIT FOR GO": A.set("ACQUIRE", W.t)
    # imagers keep pointing at the estimated TOPEX position
    for k in ("imagerA", "imagerB"):
        tgt = filt.p if est_ok else np.zeros(3); W.R_sc[k] = look_at(W.sc[k], tgt) @ SWARM_CAM_IN_BODY.T
    if not est_ok or A.ph in ("WAIT FOR GO", "ACQUIRE", "CONVERGE", "HOLD 30 m"):
        W.R_sc["docker"] = look_at(W.sc["docker"], filt.p if est_ok else np.zeros(3)) @ SWARM_CAM_IN_BODY.T
    # ---- scene poses for rendering
    set_scene_poses()
    # ---- metrics
    att_err = R_.rot_err_deg(R_est, W.R_IB) if est_ok else float("nan")
    rate_err = float(np.degrees(np.linalg.norm(w_est - W.w))) if est_ok else float("nan")
    wre = float(np.degrees(np.linalg.norm(mekf.relative_rate_body(w_est, R_est, W.w_dock, W.R_sc["docker"])))) if est_ok else float("nan")
    wrt = float(np.degrees(np.linalg.norm(mekf.relative_rate_body(W.w, W.R_IB, W.w_dock, W.R_sc["docker"]))))
    HIST.append(dict(t=W.t, wx=math.degrees(W.w[0]), wy=math.degrees(W.w[1]), wz=math.degrees(W.w[2]),
                     ex=math.degrees(w_est[0]) if est_ok else np.nan, ey=math.degrees(w_est[1]) if est_ok else np.nan, ez=math.degrees(w_est[2]) if est_ok else np.nan,
                     att=att_err, wre=wre, wrt=wrt, ph=A.ph)); FULL.append(HIST[-1])
    cyc = time.perf_counter() - c0
    fhz = (len(fix_times) - 1) / max(fix_times[-1] - fix_times[0], 1e-6) if len(fix_times) > 2 else 0.0
    if A.ph not in ("WAIT FOR GO", "ACQUIRE"): STATS["cycle_ms"].append(cyc * 1000); STATS["model_ms"].append(t_model * 1000)
    # ---- dashboard
    close = A.ph in ("FINAL APPROACH", "GRIP", "DETUMBLE", "STABLE", "RELEASE", "BACK OFF", "DONE")
    tgt_c = 0.5 * W.sc["docker"] if close else np.array([0, -13.0, 0])          # close: frame TOPEX + docker together
    MAIN_T = 0.95 * MAIN_T + 0.05 * tgt_c
    MAIN_P = 0.95 * MAIN_P + 0.05 * ((tgt_c + np.array([0.55, -0.70, 0.40]) / np.linalg.norm([0.55, -0.70, 0.40]) * 21.0) if close else np.array([30.0, -38.0, 14.0]))
    Rm = look_at(MAIN_P, MAIN_T); mcol, _ = render_view(Rm, MAIN_P, main=True)
    canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
    canvas[16:16 + 768, 0:768] = cv2.resize(mcol[:, :, ::-1], (768, 768), interpolation=cv2.INTER_LINEAR)
    txt(canvas, "MAIN VIEW (truth)", 8, 34, 0.5, (180, 180, 180))
    if A.ph in ("FINAL APPROACH", "GRIP", "DETUMBLE", "STABLE", "RELEASE", "BACK OFF") and np.linalg.norm(W.sc["docker"]) < 14:
        # docking/arm cam: on the docker above the arm, looking past the claw at the grip point (docker visible, illuminator on)
        Rd_, pd_ = W.R_sc["docker"], W.sc["docker"]
        pa = pd_ + Rd_ @ np.array([0.55, 0.50, 0.55]); ta = pd_ + Rd_ @ np.array([1.30 * ARM_S, 1.45, 0.05])   # over the +x finger
        acol, _ = render_view(look_at(pa, ta, up=Rd_[:, 2]), pa, own=None, Kc=Ka, head=0.4, znear=0.05)
        ag = auto_expose(cv2.cvtColor(acol, cv2.COLOR_RGB2GRAY), 175)
        x0a, y0a = 768 - 262, 22
        canvas[y0a:y0a + 256, x0a:x0a + 256] = cv2.cvtColor(ag, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x0a, y0a), (x0a + 255, y0a + 255), (60, 140, 240), 1)
        txt(canvas, f"ARM CAM  claw span {2.611 * ARM_S:.2f} m", x0a + 6, y0a + 16, 0.42, (60, 170, 255))
    if warp > 1.5:
        cv2.rectangle(canvas, (14, 650), (470, 704), (20, 20, 60), -1); cv2.rectangle(canvas, (14, 650), (470, 704), (80, 80, 255), 1)
        txt(canvas, f"TIME WARP x{warp:.0f}", 26, 684, 0.9, (120, 160, 255), 2)
        txt(canvas, "detumble torque warped;" if A.ph == "DETUMBLE" else "translation warped", 250, 672, 0.4, (200, 200, 255))
        txt(canvas, "spin drawn at its true rate" if A.ph == "DETUMBLE" else "(0.1 m/s back-off)", 250, 690, 0.4, (200, 200, 255))
    for i, k in enumerate(CAMS):
        x0, y0 = 768 + (i % 2) * 256, (i // 2) * 256
        gt = grays[i] if k != "docker" else auto_expose(grays[i], 170)
        tile = cv2.cvtColor(cv2.resize(gt, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        if est_ok and filt.q is not None and not (k == "docker" and Kcs[i] is Kn):
            draw_axes_overlay(tile, Rcs[i].T @ W.R_IB, Rcs[i].T @ (np.zeros(3) - pcs[i]), (120, 120, 120), 256 / RES, Kcs[i])
            draw_axes_overlay(tile, Rcs[i].T @ R_est, Rcs[i].T @ (p_est - pcs[i]), None, 256 / RES, Kcs[i])
        if k == "docker" and Kcs[i] is Kn: txt(tile, "near-field: 62 deg FOV, illuminator, auto-exp", 6, 34, 0.36, (150, 220, 255))
        canvas[y0:y0 + 256, x0:x0 + 256] = tile
        cv2.rectangle(canvas, (x0, y0), (x0 + 255, y0 + 255), tuple(int(c * 255) for c in ACC[k][::-1]), 1)
        txt(canvas, f"{k} cam", x0 + 6, y0 + 16, 0.45, tuple(int(c * 255) for c in ACC[k][::-1]))
        txt(canvas, f"fix: {'yes' if fixes[i] is not None else '-'}  inl {nin[i]}", x0 + 6, y0 + 248, 0.4, (200, 200, 200))
    x0, y0 = 768 + 256, 256
    lines = [f"PHASE: {A.ph}", f"t = {W.t:7.1f} s   warp x{warp:.0f}", "",
             "TOPEX rate deg/s   truth / estimate",
             f"  x {math.degrees(W.w[0]):7.2f} / {(math.degrees(w_est[0]) if est_ok else float('nan')):7.2f}",
             f"  y {math.degrees(W.w[1]):7.2f} / {(math.degrees(w_est[1]) if est_ok else float('nan')):7.2f}",
             f"  z {math.degrees(W.w[2]):7.2f} / {(math.degrees(w_est[2]) if est_ok else float('nan')):7.2f}",
             f"att err  {att_err:6.2f} deg", f"rate err {rate_err:6.3f} deg/s",
             f"docker rel rate est {wre:6.2f} deg/s", f"            true {wrt:6.2f} deg/s",
             f"fixes {fhz:4.1f} Hz (sim)  cycle {cyc*1000:5.0f} ms", f"{'acquire' if A.ph == 'ACQUIRE' else 'model'} {t_model*1000:5.0f} ms ({a.feat})"]
    for j, s in enumerate(lines): txt(canvas, s, x0 + 6, y0 + 18 + j * 18, 0.42, (120, 230, 255) if j == 0 else (225, 225, 225))
    if A.ph in ("DETUMBLE", "STABLE", "RELEASE", "BACK OFF", "DONE") and grip_report:
        x0p, y0p, wp, hp = 772, 520, 504, 130; cv2.rectangle(canvas, (x0p, y0p), (x0p + wp, y0p + hp), (60, 60, 60), 1)
        pts = [(h_["t"], math.sqrt(h_["wx"] ** 2 + h_["wy"] ** 2 + h_["wz"] ** 2)) for h_ in FULL if h_["t"] >= grip_report["t"]]
        tmax = max(grip_report.get("detumble_est_s", 1.0) * 1.05, pts[-1][0] - grip_report["t"] + 1); wmax0 = max(grip_report["stack_rate_after_grip_deg_s"] * 1.1, 1e-3)
        P_ = np.array([[x0p + (t_ - grip_report["t"]) / tmax * wp, y0p + hp - w_ / wmax0 * hp] for t_, w_ in pts], np.int32)
        if len(P_) > 1: cv2.polylines(canvas, [P_], False, (120, 255, 160), 2, cv2.LINE_AA)
        txt(canvas, f"STACK RATE |w| deg/s vs sim time (0 - {tmax / 3600:.1f} h)   now {pts[-1][1]:.3f} deg/s, {(W.t - grip_report['t']) / 3600:.2f} h", x0p + 4, y0p + 14, 0.4, (200, 255, 210))
        cv2.line(canvas, (x0p, y0p + hp - int(math.degrees(0.00175) / wmax0 * hp)), (x0p + wp, y0p + hp - int(math.degrees(0.00175) / wmax0 * hp)), (90, 90, 90), 1)
    else: plot(canvas, 772, 520, 504, 130, ["wx", "wy", "wz", "ex", "ey", "ez"], [(90, 90, 255), (90, 255, 90), (255, 140, 90), (180, 180, 255), (180, 255, 180), (255, 200, 170)],
         -60 * a.wmax, 60 * a.wmax, "TOPEX body rates deg/s (dark=truth, light=estimate)")
    plot(canvas, 772, 660, 504, 130, ["wrt", "wre", "att"], [(255, 255, 255), (100, 200, 255), (100, 100, 255)], 0, 10, "docker rel. rate true (white) / est (blue), att err deg (red), 0-10")
    txt(canvas, f"sliders (rad/s): wx {sliders['wx']:+.2f}  wy {sliders['wy']:+.2f}  wz {sliders['wz']:+.2f}   [G]o [X] loss [L] sun [Q]uit", 8, 798, 0.42, (170, 170, 170))
    if grip_report: txt(canvas, f"GRIP: pos err {grip_report['grip_pos_err_m']:.2f} m, rel rate {grip_report['rel_rate_true_deg_s']:.2f} deg/s, pads-to-dish {grip_report['min_pad_to_dish_m']:.2f} m", 8, 60, 0.5, (120, 255, 160))
    DISP.draw_controls(canvas, A.ph == "WAIT FOR GO")
    if writer is not None: writer.write(canvas)
    if not a.headless:
        quit_ = False
        for e in DISP.show(canvas):
            if e[0] != "key": continue
            if e[1] == "q": quit_ = True
            elif e[1] == "g": A.go = True
            elif e[1] == "x": apply_cmd("loss")
            elif e[1] == "l": apply_cmd("light")
        if quit_: break
    frame += 1
if writer is not None: writer.release()
summary = dict(selftest=SELFTEST, feat=a.feat, crop=a.crop, script=a.script, fixed_dt=a.fixed_dt, sun_unknown=a.sun_unknown, phases=A.log, grip=grip_report,
               median_cycle_ms=float(np.median(STATS["cycle_ms"])) if STATS["cycle_ms"] else None, median_model_ms=float(np.median(STATS["model_ms"])) if STATS["model_ms"] else None,
               loss_t=STATS["loss_t"], reacq_t=STATS["reacq_t"], light_t=STATS["light_t"],
               att_err_med_tracking=float(np.nanmedian([h["att"] for h in FULL if h["ph"] in ("CONVERGE", "HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH")])) if FULL else None,
               rate_err_med_tracking=float(np.nanmedian([abs(np.linalg.norm([h["wx"]-h["ex"], h["wy"]-h["ey"], h["wz"]-h["ez"]])) for h in FULL if h["ph"] in ("HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH")])) if FULL else None,
               max_docker_speed_m_s=max(STATS['dock_v']) if STATS['dock_v'] else None, max_docker_rate_deg_s=max(STATS['dock_w']) if STATS['dock_w'] else None, t_demo=t_demo, t_sim=W.t,
               rel_rate_true_final_approach=float(np.nanmedian([h["wrt"] for h in FULL if h["ph"] == "FINAL APPROACH"])) if any(h["ph"] == "FINAL APPROACH" for h in FULL) else None)
print(json.dumps(summary))
if a.metrics: open(a.metrics, "a").write(json.dumps(summary) + "\n")
