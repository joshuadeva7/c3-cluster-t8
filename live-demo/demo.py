"""SWARM x TOPEX live demo.
  title screen -> cutscene (v2 mission video re-rendered without text, v = 0..27: arrival to the 1.0/1.3/1.6 km V-bar holds)
  -> the SAME scene continues live (same LVLH camera path, FOV, colours, stars, trails; every position straight from
  v2/out/prox.csv): the imagers transfer into their mapping orbit (NMC, b = 23 m) while SC1 holds 1 km out
  -> DOCKER SELECTION: the sim freezes, a panel sets TOPEX's body rates, GO (greyed out until the rates are valid)
  -> TOPEX's spin starts; SC1 (docker) flies its prox.csv transfer to the 200 m hold and the glideslope to 30 m while the imagers'
  real XFeat model + PnP track TOPEX from the NMC and a MEKF fuses all fixes; the docker's autonomy uses ONLY that estimate:
  hold 30 m -> spin-match (hydrazine) -> final approach (3 cm/s, shown x10) -> grip clear of the dish -> detumble (hydrazine,
  time-warped) -> stable -> release -> back off.
  TOPEX: torque-free rigid body, inertia from assets/topex_inertia.json (DEMO ASSUMPTION roll = major axis, spin about body y).
Keys (window mode): Enter/S start, SPACE skip the cutscene, G = GO, P = pause/resume, Esc/Q = quit (or Ctrl-C in the terminal).
  Hidden test keys: X = inject a tracking loss, L = rotate the sun.
Headless: --headless --record out.mp4 --script "0:w=0.008,0.661,0.008;0:go;45:light;60:loss" (w=/go are applied at the docker-selection
  freeze; other events at sim seconds after GO; the sim advances fixed dt per frame)."""
import os as _os; _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, json, math, argparse, collections
import numpy as np, cv2
import render as R_, tracker as T_, features, mekf
import trimesh, gl_render, ui_window
from scipy.spatial.transform import Rotation as Rot

HERE = _os.path.dirname(_os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--feat", default="xfeat-onnx-cpu" if sys.platform.startswith("linux") else "xfeat-coreml-all32")   # Mac: fp32 CoreML (ALL/NE fp16 build is wrong on the ANE)
p.add_argument("--crop", type=int, default=192 if sys.platform.startswith("linux") else 256); p.add_argument("--camres", type=int, default=512)
p.add_argument("--headless", action="store_true"); p.add_argument("--record", default="")
p.add_argument("--script", default=""); p.add_argument("--T", type=float, default=0.0, help="headless run length (s); 0 = until DONE. Window mode never times out")
p.add_argument("--spinmatch_warp", type=float, default=8.0, help="time warp while COASTING through SPIN-MATCH (docker spin-up on the 1 N*m couple takes >= ~27 s)")
p.add_argument("--fixed_dt", type=float, default=0.0, help="sim seconds per frame (0 = real time from the wall clock)")
p.add_argument("--no_cutscene", action="store_true"); p.add_argument("--no_prologue", action="store_true", help="jump straight to the docker-selection freeze")
p.add_argument("--prologue_warp", type=float, default=400.0, help="mission-time warp while the imagers transfer into the NMC")
p.add_argument("--detumble_warp", type=float, default=600.0); p.add_argument("--fps_out", type=int, default=10)
p.add_argument("--until_grip", action="store_true", help="stop 2 s after the grip (verification runs)"); p.add_argument("--metrics", default=""); p.add_argument("--seed", type=int, default=0)
p.add_argument("--sun_unknown", action="store_true", help="tracker references use a fixed default sun (lighting-mismatch stress)")
a = p.parse_args()
rng = np.random.default_rng(a.seed)

# ------------------------------------------------------------------ assets
cad = np.load(_os.path.join(HERE, "assets", "cad_render.npz"))
INR = json.load(open(_os.path.join(HERE, "assets", "topex_inertia.json")))
COM = np.array(INR["com_body_m"]); J = np.array(INR["J_used"])        # ASSUMED: diag(yaw 9371, roll 11665, pitch 4213), COM on mesh y (derive_inertia.py)
W0_DEFAULT = float(INR["spin_rate_default_rad_s"])                     # 2 pi / 9.5 s about roll (body +y)
tm = R_.topex_mesh(); tm.vertices -= COM                               # body origin = centre of mass
# v2 grip point: docker on the -y spin axis (prox.csv DOCK_R 3.877 m, mesh frame), clock Ry(-90). The live demo goes DOCK_DEEPER further in
# along that same axis so the claw's crossbar stops 3 cm off the -y face and the pads (0.56 m beyond the crossbar) wrap past the bus end.
DOCK_DEEPER = 0.52
# Docker COM: Daniel 2026-11-11, assumed AT the mesh origin, i.e. on TOPEX's spin axis at the grip (no off-centre modelling).
# (The capture-envelope audit's CAD estimate is (-0.001, +0.072, +0.028) m; set C_DOCK to that to model it.)
C_DOCK = np.zeros(3)
_PB0 = np.array([0.0, -3.877 + DOCK_DEEPER, 0.0]) - COM                   # v2 grip point (mesh origin on the axis)
DOCK_RB = R_.Rot.from_euler("y", -90, degrees=True).as_matrix()        # v2 DOCK_CLOCK: claw rolled clear of the dish
_cb = DOCK_RB @ C_DOCK
DOCK_PB = _PB0 - np.array([_cb[0], 0.0, _cb[2]])                         # mesh origin target, shifted so the COM is on the axis
DOCK_COM_B = DOCK_PB + _cb                                               # = (0, -3.077, 0): the docker COM's grip position in TOPEX body
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
J_DOCK = np.diag([36.68, 58.88, 41.62])   # trr/inertia.py, arm open, mapped team (X arm-normal, Y wing, Z arm) -> GLB (y, z, x); 58.88 = gripper axis
M_DOCK = 198.535                          # TRR mass table, total with margin
# ---- TOPEX body-rate slider config (rad/s), from the capture-envelope study (anim-work/NOTES-capture-envelope.md):
#      wx, wz: +-cap; wy (roll spin): |wy| in [min, max], either sign. GO is greyed out until |wy| is inside that band.
# Daniel 2026-11-11: wx = wz = 0 (conservative: no wobble, the grip point sits still on the spin axis). wy_max 0.789 rad/s = the
# 8.58 kg despin hydrazine allocation at the 0.5 m lever (capture-envelope audit, stack_and_budget.json); wy_min 0.10 (a visible spin).
RATE_LIMITS = dict(wx=0.0, wz=0.0, wy_min=0.10, wy_max=0.789, wy_default=0.661)
# ---- docker actuators (midterm main.tex T:366-388, TRR mass table):
#      12x ArianeGroup 1 N hydrazine thrusters, Isp 220 s, 0.5 m lever: 2 N force and one 1 N*m couple per axis.
#      3x Blue Canyon RW8 (8 N*m*s, 0.25 N*m each), one per docker axis: they follow TOPEX's wobble during the hold.
#      The spin-up to TOPEX's 0.66 rad/s needs ~38.9 N*m*s >> 8 N*m*s, so it is done on the thrusters (~36 g of hydrazine).
G0 = 9.80665; ISP = 220.0; LEVER = 0.5
DOCK_A_MAX = 2.0 / M_DOCK                  # 2 N per translation axis -> 0.0101 m/s^2
DOCK_V_MAX = 0.10                          # TRR prox-ops speed limit, m/s
DOCK_V_FINAL = 0.03                        # contact speed limit 3 cm/s (v2: "moves in at 3 cm/s")
DOCK_TAU = 1.0                             # one thruster couple per axis: 2 x 1 N x 0.5 m = 1 N*m (spin axis)
DOCK_ALPHA = DOCK_TAU / 58.88              # spin-up about the docker's largest axis -> 0.017 rad/s^2 (0.66 rad/s in ~39 s)
RW_TAU, RW_H = 0.25, 8.0                   # per wheel
RW_ALPHA = RW_TAU / 41.62                  # wobble/swing tracking on the wheels (smallest authority of the transverse axes)
KG_PER_NMS = 1.0 / (LEVER * ISP * G0)      # hydrazine per N*m*s of angular impulse at a 0.5 m lever: 9.27e-4 kg
DETUMBLE_TAU = 0.5                         # TRR ConOps: one 1 N thruster x 0.5 m against H

APPROACH_WARP = 10.0                       # final-approach TRANSLATION shown x10 (spin always drawn at its true rate)
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
WIN_W, WIN_H = 1280, 720                       # = the cutscene video's 16:9 frame: the picture fills the window (no band, no gap)
WMAX3 = np.array([RATE_LIMITS["wx"], RATE_LIMITS["wy_max"], RATE_LIMITS["wz"]])
def rates_valid(w): return abs(w[0]) <= RATE_LIMITS["wx"] + 1e-9 and abs(w[2]) <= RATE_LIMITS["wz"] + 1e-9 and RATE_LIMITS["wy_min"] - 1e-9 <= abs(w[1]) <= RATE_LIMITS["wy_max"] + 1e-9
DISP = ui_window.Display(WIN_W, WIN_H, a.headless, wmax=tuple(WMAX3), w0=(0.0, RATE_LIMITS["wy_default"], 0.0))
RES = a.camres; K = R_.K_for(RES, 30.0)
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
ARMRES = 200; Ka = R_.K_for(ARMRES, 88.0)
# Chase cam + thruster plumes (Daniel 2026-11-11, "make the thrusters fire"). Thruster LAYOUT is an ASSUMPTION for the visual only:
# the physics uses 1 N thrusters at a 0.5 m lever (TRR / v2). Couple nozzles sit at +-0.5 m on the docker's body axes and fire
# tangentially; translation nozzles fire opposite the commanded delta-v. Plume length scales with the commanded thrust.
CHRES = 300; Kch = R_.K_for(CHRES, 60.0); CHASE_OFF = np.array([4.5, -6.5, 2.5])
LEV = 0.5
def _nozzles(FIRE):
    out = []   # (position in docker body, exhaust direction in docker body, throttle 0..1)
    t = FIRE["tau"] if FIRE["det"] is None else FIRE["det"] * 0.5
    caps = (DOCK_TAU, DOCK_TAU, DOCK_TAU)
    for ax in range(3):
        if ax != 1 and not FIRE["big"] and FIRE["det"] is None: continue   # x/z on the wheels unless a big swing
        u = float(np.clip(t[ax] / caps[ax], -1, 1))
        if abs(u) < 0.05: continue
        e = np.eye(3); a1, a2 = e[(ax + 1) % 3], e[(ax + 2) % 3]
        # couple about axis ax: thrust F = +-a2 at r = +-LEV*a1 gives r x F = +-LEV a1 x a2 = +-LEV e_ax; exhaust opposite the thrust
        for sgn in (1, -1):
            r = sgn * LEV * a1; F = sgn * np.sign(u) * a2
            out.append((r, -F, abs(u)))
    dv = FIRE["dv"]; dn = np.linalg.norm(dv)
    if dn > 1e-7:
        out.append((-0.6 * dv / dn, -dv / dn, 1.0)); out.append((-0.6 * dv / dn + 0.15 * np.cross(dv / dn, [0, 0, 1.0 if abs(dv[2]) < 0.9 * dn else 0]), -dv / dn, 0.8))
    return out
def draw_plumes(img, Rc, pc, Rd, pd, FIRE, frame):
    over = np.zeros_like(img, np.float32)
    def proj(pw):
        x = Rc.T @ (pw - pc)
        if x[2] < 0.05: return None
        u = Kch @ x; return (float(u[0] / u[2]), float(u[1] / u[2]))
    rng_ = np.random.default_rng(frame)
    for r, ex, thr in _nozzles(FIRE):   # short flaring exhaust cone per firing nozzle (BGR, like the canvas)
        p0 = pd + Rd @ r; L = (0.35 + 0.65 * thr) * (0.8 + 0.4 * rng_.random())
        a = proj(p0); b = proj(p0 + Rd @ ex * L)
        if a is None or b is None: continue
        a = np.array(a); b = np.array(b); dvec = b - a; ln = np.linalg.norm(dvec)
        if ln < 1.0: cv2.circle(over, tuple(int(v) for v in a), 5, tuple(255.0 * thr for _ in range(3)), -1, cv2.LINE_AA); continue
        nrm = np.array([-dvec[1], dvec[0]]) / ln
        for frac, wmul, col in ((1.0, 0.45, (30, 110, 255)), (0.75, 0.28, (90, 180, 255)), (0.45, 0.12, (225, 245, 255))):   # orange glow -> white-hot core (BGR)
            e = a + dvec * frac; w0 = 1.5; w1 = max(2.5, wmul * ln)
            poly = np.array([a + nrm * w0, e + nrm * w1, e - nrm * w1, a - nrm * w0], np.int32)
            cv2.fillPoly(over, [poly], tuple(c * (0.35 + 0.65 * thr) for c in col), cv2.LINE_AA)
        cv2.circle(over, tuple(int(v) for v in a), 3, (255, 255, 255), -1, cv2.LINE_AA)
    over = cv2.GaussianBlur(over, (0, 0), 3.0) * 1.3 + cv2.GaussianBlur(over, (0, 0), 0.8) * 0.5
    return np.clip(img.astype(np.float32) + over, 0, 255).astype(np.uint8)
def render_view(R_IC, p_IC, own=None, Kc=None, head=0.0, znear=0.3):
    for k, n in n_sc.items(): world.set_visible(n, k != own)   # a camera never sees its own bus
    world.set_visible(n_arm, own != "docker")
    Kc = K if Kc is None else Kc; n_ = ARMRES if Kc is Ka else (CHRES if Kc is Kch else RES)
    return world.render(Kc, n_, n_, R_IC, p_IC, W.sun, znear=znear, zfar=2000, headlight=head)
def auto_expose(g, target=200.0):   # display auto-exposure: 97th percentile of lit pixels -> target
    m = g > 6
    if m.sum() < 30: return g
    gain = float(np.clip(target / max(np.percentile(g[m], 97), 1.0), 0.3, 5.0))
    return np.clip(g.astype(np.float32) * gain, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ main view = the v2 video, continued (one camera, no cuts)
# v2look.py: the v2 LVLH camera path (v2/mission_anim.py LK keys, elevation levelled to CUT_EL as in the cutscene), FOV 34 deg,
# background, stars, V-bar, hold markers, keep-out / 200 m rings, trails, far dots, burn rings, vignette; the models in the v2
# colours with the v2 flat shading. Room frame = v2 LVLH, TOPEX at the origin.
import v2look as V2
LOOK = V2.Look(WIN_W, WIN_H)
def v2_sc_mesh(accent, keys):
    cols = {"shell": V2_WHITE * 0.92 + accent * 0.08, "inner": V2_FOIL * 0.9, "arr_dep": V2_CELL, "arm_ext": accent * 0.85, "arm_end": V2_WHITE * 0.75}
    parts = []
    for key in keys:
        m = trimesh.Trimesh(cad[key + "_v"], cad[key + "_f"], process=False)
        m.visual.face_colors = np.c_[np.tile((np.clip(cols[key], 0, 1) * 255).astype(np.uint8), (len(m.faces), 1)), np.full(len(m.faces), 255, np.uint8)]
        parts.append(m)
    return trimesh.util.concatenate(parts)
V2_FOIL, V2_CELL, V2_WHITE = np.array([0.74, 0.60, 0.33]), np.array([0.12, 0.19, 0.36]), np.array([0.80, 0.82, 0.85])   # v2 mission_anim.py:472
mainsc = gl_render.GLScene(DISP.ctx)
m_top = mainsc.add(tm)
BODY_KEYS = {"docker": ("shell", "inner", "arr_dep"), "imagerA": ("shell", "inner", "arr_dep", "arm_ext", "arm_end"), "imagerB": ("shell", "inner", "arr_dep", "arm_ext", "arm_end")}
ROLE_ACC = {"docker": V2.C_DOCK, "imagerA": V2.C_IMG, "imagerB": V2.C_IMG}
m_sc = {k: (mainsc.add(v2_sc_mesh(V2.C_NEU, BODY_KEYS[k])), mainsc.add(v2_sc_mesh(ROLE_ACC[k], BODY_KEYS[k]))) for k in BODY_KEYS}
m_arm = (mainsc.add(v2_sc_mesh(V2.C_NEU, ("arm_ext", "arm_end"))), mainsc.add(v2_sc_mesh(V2.C_DOCK, ("arm_ext", "arm_end"))))
def lvlh_C(t):   # LVLH attitude (ECI -> LVLH) on the cutscene's visual clock, continued from its last frame
    return np.array([np.interp(t, V2.H5["CV_T"], V2.H5["CV"][:, i, j]) for i in range(3) for j in range(3)]).reshape(3, 3)
def render_main(Rc, Pc, d, v, width, cx, roles_s, flashes):
    """the main view: v2 background + vector layers, then the GL models (v2 shading) on top, then the vignette."""
    C = lvlh_C(T_VIS); V2.STARS = V2.H5["STARS_ECI"] @ C.T
    img = LOOK.background(Rc, Pc, cx, width)
    LOOK.overlays(img, Rc, Pc, d, cx, v, (TM_D, TM_I, TM_I), roles_s, burn_flash=flashes)
    sel = roles_s > 0.5
    for k, (n0, n1) in m_sc.items():
        mainsc.set_visible(n0, not sel); mainsc.set_visible(n1, sel); T_ = T44(W.R_sc[k], W.sc[k])
        mainsc.set_pose(n0, T_); mainsc.set_pose(n1, T_)
    Ta = T44(W.R_sc["docker"], W.sc["docker"]) @ np.diag([ARM_S, 1, 1, 1])
    mainsc.set_visible(m_arm[0], not sel); mainsc.set_visible(m_arm[1], sel); mainsc.set_pose(m_arm[0], Ta); mainsc.set_pose(m_arm[1], Ta)
    mainsc.set_pose(m_top, T44(W.R_IB, np.zeros(3)))
    zn = float(np.clip(d * 0.01, 0.05, 5.0))
    col, dep = mainsc.render(LOOK.K(cx), width, WIN_H, Rc, Pc, W.sun, znear=zn, zfar=max(4000.0, 4 * d), v2look=True)
    m = dep > 0; img[m] = col[:, :, ::-1][m]
    return LOOK.vignette(img, 0 if width == WIN_W else (WIN_W - width) // 2)

# ------------------------------------------------------------------ truth state
def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos; z /= np.linalg.norm(z); x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6: x = np.cross(z, [1, 0, 0])
    x /= np.linalg.norm(x); return np.c_[x, np.cross(z, x), z]
SWARM_CAM_IN_BODY = np.c_[[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # camera z along SWARM +y (gripper side), image-down = body -z
def face_R(pos, target):   # v2 attitude of a free-flying SWARM: body +y (gripper / camera side) at the target, body x toward LVLH +x
    by = target - pos; by = by / np.linalg.norm(by) if np.linalg.norm(by) > 1e-9 else np.array([0, 1.0, 0])
    bx = np.array([1.0, 0, 0]) - by[0] * by; bx /= np.linalg.norm(bx); return np.c_[bx, by, np.cross(bx, by)]
P_HOLD, P_BID, P_NMC, P_H30 = (float(V2.H5[k]) for k in ("P_HOLD", "P_BID", "P_HOLD", "P_H30"))
P_NMC = float(V2.H5["ev_t0"][list(V2.H5["ev_names"]).index("NMC inspection (b=23 m)", list(V2.H5["ev_craft"]).index(2))])   # SC3 enters the NMC (10902.8 s)
TM_I = TM_D = P_HOLD; T_VIS = V2.TREL0     # imager / docker mission clocks (s after TOPEX arrival, prox.csv); visual clock (sun, stars)
class World:
    def __init__(self):
        self.t = 0.0; self.R_IB = Rot.from_euler("y", float(V2.H5["PHI_end"])).as_matrix()   # v2's TOPEX attitude at the hand-off; roll axis (body y) along LVLH y
        self.w = np.zeros(3)                                                                    # TOPEX's spin starts on GO
        self.sun = V2.SUN.copy()                                                                 # the cutscene's sun at its last frame (LVLH)
        self.sc = {"imagerA": V2.rel(1, P_HOLD), "imagerB": V2.rel(2, P_HOLD), "docker": V2.rel(0, P_HOLD)}   # prox.csv at the hand-off
        self.R_sc = {k: face_R(v, np.zeros(3)) for k, v in self.sc.items()}
        self.w_dock = np.zeros(3); self.attached = False
        self.v_dock = np.zeros(3); self.wI_dock = np.zeros(3); self.t_mission = 0.0   # docker inertial velocity / rate; mission clock (incl. warped approach)
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
FALLBACK = ("xfeat-onnx-cpu", "orb") if sys.platform.startswith("linux") else \
           ("xfeat-coreml-all32", "xfeat-coreml-gpu", "xfeat-onnx-coreml", "xfeat-onnx-cpu", "orb")   # never coreml-all / coreml-ne (fp16 ANE build is wrong)
for cand in [a.feat] + [b for b in FALLBACK if b != a.feat]:
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
    PH = ["PROLOGUE", "DOCKER SELECTION", "ACQUIRE", "CONVERGE", "HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH", "GRIP", "DETUMBLE", "STABLE", "RELEASE", "BACK OFF", "DONE"]
    def __init__(self): self.ph = "PROLOGUE"; self.arrived = False; self.t_ph = 0.0; self.go = False; self.conv_t = 0.0; self.d = HOLD_D; self.blend = 0.0; self.log = []
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
    # re-acquire after a detected loss: the 60 deg fault is in the attitude only, so the converged body rates are kept (resetting them
    # to zero made the guidance despin the docker and re-converge at 1x for minutes)
    w_keep = filt.w.copy() if getattr(A, "resume", None) else None
    filt.q = None; filt.w = np.zeros(3); filt.P = np.diag([0.3] * 3 + [0.5] * 3) ** 2; filt.Pp = np.diag([5.0] * 3 + [1.0] * 3) ** 2; filt.v = np.zeros(3)
    if w_keep is not None: filt.w = w_keep   # covariance stays loose: the kept rates may predate full convergence
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
    return sorted(ev, key=lambda e: e[0])   # stable: same-time events keep their written order (w= before go)
EV = parse_script(a.script)
sliders = {"wx": 0.0, "wy": RATE_LIMITS["wy_default"], "wz": 0.0}
def corrupt_estimate():   # the hidden X key: rotate the estimate 60 deg off the truth
    filt.q = mekf.qmul(filt.q, mekf.qexp(np.radians(60) * np.array([0.6, 0.0, 0.8]))); STATS["loss_t"].append(round(W.t, 2))
def apply_cmd(cmd):
    if cmd.startswith("w="):
        v = [float(x) for x in cmd[2:].split(",")]; v = np.clip(v, -WMAX3, WMAX3)
        sliders.update(wx=v[0], wy=v[1], wz=v[2])
        for nm, x in zip(("wx", "wy", "wz"), v):
            if nm in UI_SLIDERS: UI_SLIDERS[nm].val = float(x)
    elif cmd == "go": A.go = True
    elif cmd == "loss":   # corrupt the estimate by 60 deg: the pipeline must notice and re-acquire
        if filt.q is not None and A.ph == "SPIN-MATCH" and not getattr(A, "relock", False):
            A.loss_pending = True   # blind coast: nothing could notice it now -> applied the moment vision re-locks (caught there)
        elif filt.q is not None and A.ph not in ("FINAL APPROACH", "GRIP", "DETUMBLE", "STABLE", "RELEASE", "BACK OFF", "DONE"): corrupt_estimate()
    elif cmd == "ir":     # imagerA's visible camera (MSSS LCAM) fails: ~2.5 s dark, then its MSSS ECAM-IR3A LWIR camera takes over
        if IR["cam"] is None: IR.update(cam="imagerA", t_fail=W.t, last=-1e9, img=None); STATS["ir_t"].append(round(W.t, 2))
    elif cmd == "light": W.sun = Rot.from_euler("z", 75, degrees=True).as_matrix() @ W.sun; STATS["light_t"].append(round(W.t, 2))
STATS = {"pos_err": {}, "filt_p": [], "fuel_spin_kg": 0.0, "fuel_trans_kg": 0.0, "fuel_detumble_kg": 0.0, "fuel_swing_kg": 0.0, "rw_h_max": 0.0, "rw_tau_max": 0.0, "dock_v": [], "dock_w": [], "loss_t": [], "light_t": [], "ir_t": [], "reacq_t": [], "fix_hz": [], "cycle_ms": [], "model_ms": []}
UI_SLIDERS = {sl.name: sl for sl in DISP.sliders}
# Camera-fail test (key I): midterm main.tex:401-402 / :331 "ECAM-IR3A ... 640x480 LWIR, 12 fps ... complementary LWIR observations during
# poor visible illumination". Thermal emission does not depend on the sun, so the IR view is lit from the camera (uniform emitter look),
# downsampled to the IR3A's lower resolution, blurred and noisy, and only refreshed at 12 Hz (a fix only on a fresh IR frame).
IR = dict(cam=None, t_fail=None, last=-1e9, img=None); IR_DARK, IR_HZ, IR_SCALE = 2.5, 12.0, 480 / 1024
def ir_state(k):
    if IR["cam"] != k: return None
    return "dark" if W.t - IR["t_fail"] < IR_DARK else "ir"
writer = None
if a.record:
    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), a.fps_out, (WIN_W, WIN_H))

def keys_of(ev): return [e[1] for e in ev if e[0] == "key"]
def badge(img, text, x, y, col=(80, 200, 255)):
    (tw, th), _ = cv2.getTextSize(text, F, 0.6, 2); cv2.rectangle(img, (x, y), (x + tw + 20, y + th + 16), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + tw + 20, y + th + 16), col, 1); txt(img, text, x + 10, y + th + 8, 0.6, col, 2)
def pause_loop(canvas, main_w):
    """P toggles: everything (sim time, dynamics, estimation, UI) is frozen; the last frame stays up with a PAUSED badge."""
    c = canvas.copy(); badge(c, "PAUSED", main_w - 120, 14)
    while True:
        k = keys_of(DISP.show(c)); time.sleep(1 / 30)
        if "p" in k: return False
        if "q" in k: return True
def emit(canvas, main_w, allow_pause=True):
    """write / show one frame; returns 'quit' or the list of keys."""
    if writer is not None: writer.write(canvas)
    if a.headless: return []
    k = keys_of(DISP.show(canvas))
    if "q" in k: return "quit"
    if "p" in k and allow_pause:
        if pause_loop(canvas, main_w): return "quit"
        global last_wall; last_wall = time.perf_counter()
    return k
def title_screen():
    """ready screen: nothing starts until Enter (or S), so the demo can be queued before a talk."""
    canvas = np.empty((WIN_H, WIN_W, 3), np.uint8); canvas[:] = V2.BG
    txt(canvas, "SWARM  x  TOPEX/Poseidon", 120, 280, 1.6, (235, 235, 235), 2)
    txt(canvas, "Live capture demo  -  AE 4322 Team 8", 124, 330, 0.75, (180, 200, 220), 1)
    txt(canvas, "Press ENTER to begin", 124, 420, 0.75, (90, 175, 245), 2)
    txt(canvas, f"feature backend: {a.feat} (self-test passed)", 124, 640, 0.45, (110, 150, 120), 1)
    if writer is not None:
        for _ in range(max(1, int(1 * a.fps_out))): writer.write(canvas)
    if a.headless: return True
    while True:
        k = keys_of(DISP.show(canvas)); time.sleep(1 / 30)
        if "\r" in k: return True
        if "q" in k: return False
def play_cutscene():
    """the re-rendered v2 video (no text at all), full window. SPACE skips silently. Returns its last frame."""
    cap = cv2.VideoCapture(_os.path.join(HERE, "assets", "cutscene.mp4")); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = 0; last = None; t_next = time.perf_counter()
    while True:
        ok, fr = cap.read()
        if not ok: break
        if fr.shape[:2] != (WIN_H, WIN_W): fr = cv2.resize(fr, (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)
        last = fr
        if writer is not None and n % max(1, round(fps / a.fps_out)) == 0: writer.write(fr)
        n += 1
        if not a.headless:
            k = keys_of(DISP.show(fr))
            if "q" in k: cap.release(); return "quit"
            if " " in k: break
            t_next += 1.0 / fps; time.sleep(max(0.0, t_next - time.perf_counter()))
    cap.release(); return last
if not title_screen(): raise SystemExit(0)
CUT_LAST = None
if not a.no_cutscene:
    CUT_LAST = play_cutscene()
    if isinstance(CUT_LAST, str): raise SystemExit(0)
HANDOFF_FADE = 0.6     # s: the video's last frame dissolves into the live render of the same instant (same camera, same state)

# ------------------------------------------------------------------ main loop
# STAGE: PROLOGUE (imagers transfer into the NMC, x prologue_warp) -> SELECT (frozen, rate panel) -> OPS (GO .. DONE)
lost_count = 0; last_wall = time.perf_counter(); fix_times = collections.deque(maxlen=30); ev_i = 0
grip_report = None; frame = 0; DASH_ERR = None; A.aH = None; A.v_contact = float("nan")
t_demo = 0.0; t_live = 0.0; t_go = 0.0; FLASH = []; V_CAM = 27.0
STAGE = "PROLOGUE"
if a.no_prologue: TM_I, TM_D = P_NMC, P_BID
def clock_burns(k, t_old, t_new):
    for tb, kb in zip(V2.BURN_T, V2.BURN_K):
        if kb == k and t_old < tb <= t_new: FLASH.append([k, 0.0])
def place_free_flyers():
    for k, kk in (("imagerA", 1), ("imagerB", 2)): W.sc[k] = V2.rel(kk, TM_I)
def flashes(dt_):
    for f_ in FLASH: f_[1] += dt_ / 0.55 * (1.0 if a.fixed_dt <= 0 else 1.0)
    FLASH[:] = [f_ for f_ in FLASH if f_[1] < 1.0]; return [(k, u) for k, u in FLASH]
def warp_tag(img, x, y, s1, s2=None, alpha=1.0):
    if alpha <= 0.01: return
    ov = img.copy(); txt(ov, s1, x, y, 0.5, (175, 190, 215), 1)
    if s2: txt(ov, s2, x, y + 20, 0.42, (130, 140, 160), 1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
def hms(t): t = max(0.0, t); return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"
quit_all = False
while True:
    c0 = time.perf_counter()
    wall = time.perf_counter(); dt = a.fixed_dt if a.fixed_dt > 0 else min(wall - last_wall, 0.25); last_wall = wall
    if STAGE == "PROLOGUE" and a.no_prologue: STAGE = "SELECT"; A.set("DOCKER SELECTION", 0.0); sel_t = 0.0; place_free_flyers()
    if STAGE == "PROLOGUE":
        t_old = TM_I; TM_I = min(P_NMC, TM_I + dt * a.prologue_warp); T_VIS += dt; t_live += dt
        TM_D = P_HOLD + (P_BID - P_HOLD) * (TM_I - P_HOLD) / (P_NMC - P_HOLD)   # SC1 holds 1 km out (its prox.csv hold, 3034 -> 4834 s)
        for kk in (1, 2): clock_burns(kk, t_old, TM_I)
        place_free_flyers(); W.sc["docker"] = V2.rel(0, TM_D)
        for k in W.sc: W.R_sc[k] = face_R(W.sc[k], np.zeros(3))
        prog = (TM_I - P_HOLD) / (P_NMC - P_HOLD); V_CAM = 27.0 + 12.0 * float(V2.smoother(prog))
        Rm, Pm, dm = V2.cam(V_CAM)
        img = render_main(Rm, Pm, dm, V_CAM, WIN_W, 640.0, 0.0, flashes(dt))
        warp_tag(img, 22, 32, f"T+{hms(TM_I)}   TIME x{a.prologue_warp:.0f}", "imagers SC2, SC3 transfer into the mapping orbit (NMC, b = 23 m)", alpha=float(np.clip((t_live - 1.0) / 1.5, 0, 1)) if CUT_LAST is not None else 1.0)
        if CUT_LAST is not None and t_live < HANDOFF_FADE:
            u = t_live / HANDOFF_FADE; img = (CUT_LAST * (1 - u) + img * u).astype(np.uint8)
        if frame == 0 or (CUT_LAST is not None and t_live <= dt + 1e-9): cv2.imwrite(_os.path.join(HERE, "runs", "frame_handoff_live.png"), img) if _os.path.isdir(_os.path.join(HERE, "runs")) else None
        if TM_I >= P_NMC - 1e-6: STAGE = "SELECT"; A.set("DOCKER SELECTION", 0.0); sel_t = 0.0; SEL_IMG = None
        r = emit(img, WIN_W); frame += 1
        if r == "quit": break
        continue
    if STAGE == "SELECT":
        # FROZEN: no clock advances. Panel = TOPEX body rates + GO (greyed out until valid).
        if not a.headless:
            sliders.update(wx=0.0, wy=UI_SLIDERS["wy"].val, wz=0.0)
        else:   # script: apply w= / go at the freeze (hold the panel on screen ~2 s of video first)
            while ev_i < len(EV) and EV[ev_i][1] != "go": apply_cmd(EV[ev_i][1]); ev_i += 1
        if "SEL_IMG" not in globals() or SEL_IMG is None:
            V_CAM = 39.0; Rm, Pm, dm = V2.cam(V_CAM); SEL_IMG = render_main(Rm, Pm, dm, V_CAM, WIN_W, 640.0, 0.0, [])
        img = SEL_IMG.copy(); sel_t += dt
        wv = np.array([sliders["wx"], sliders["wy"], sliders["wz"]]); valid = rates_valid(wv); go_ok = valid and sel_t > 1.0
        d1 = np.linalg.norm(V2.rel(0, TM_D)) / 1000
        lines = [f"SC1 is closest ({d1:.2f} km on the V-bar): it is the docker.", "SC2 and SC3 keep imaging TOPEX from the NMC.",
                 "" if valid else f"|wy| must be {RATE_LIMITS['wy_min']:.2f}-{RATE_LIMITS['wy_max']:.3f} rad/s (wx = wz = 0)"]
        DISP.draw_select(img, go_ok, lines)
        if frame % 1 == 0 and sel_t <= dt + 1e-9 and _os.path.isdir(_os.path.join(HERE, "runs")): cv2.imwrite(_os.path.join(HERE, "runs", "frame_select.png"), img)
        r = emit(img, WIN_W, allow_pause=False); frame += 1
        if r == "quit": break
        go_now = ("g" in r or "\r" in r) and go_ok
        if a.headless and go_ok and sel_t >= 2.0 and ev_i < len(EV) and EV[ev_i][1] == "go": ev_i += 1; go_now = True
        if a.headless and go_ok and sel_t >= 2.0 and not EV: go_now = True
        if go_now:
            W.w = wv.copy(); W.t = 0.0; TM_D = P_BID; STAGE = "OPS"; t_go = 0.0; A.go = True
            A.set("ACQUIRE", 0.0); print(f"[GO] TOPEX body rates {np.round(W.w, 4).tolist()} rad/s", flush=True)
        continue
    # ================================================================ OPS
    while ev_i < len(EV) and EV[ev_i][0] <= W.t: apply_cmd(EV[ev_i][1]); ev_i += 1
    # ---- cameras (truth render) + estimation
    live, Rcs, pcs, use = [], [], [], []
    Kcs = []
    for k in CAMS:
        Rc, pc = W.cam_pose(k)
        near = k == "docker" and np.linalg.norm(W.sc["docker"] - (filt.p if filt.q is not None else np.zeros(3))) < 11.0
        Kc = Kn if near else K; Kcs.append(Kc)
        rng_ = float(np.linalg.norm(W.sc["docker"])) if near else 0.0
        st = ir_state(k)
        if st == "dark": col = np.zeros((RES, RES, 3), np.uint8); IR["fresh"] = False
        elif st == "ir":
            if W.t - IR["last"] >= 1.0 / IR_HZ - 1e-9 or IR["img"] is None:
                for kk, n in n_sc.items(): world.set_visible(n, kk != k)
                world.set_visible(n_arm, True)
                c_, _ = world.render(K, RES, RES, Rc, pc, Rc @ np.array([0.0, 0.0, -1.0]), znear=0.3, zfar=2000)
                lo = max(8, int(RES * IR_SCALE)); c_ = cv2.resize(cv2.resize(c_, (lo, lo), interpolation=cv2.INTER_AREA), (RES, RES), interpolation=cv2.INTER_LINEAR)
                c_ = cv2.GaussianBlur(c_, (0, 0), 1.2).astype(np.float32) * 0.85 + 18 + rng.normal(0, 4.0, c_.shape)
                IR["img"] = np.clip(c_, 0, 255).astype(np.uint8); IR["last"] = W.t; IR["fresh"] = True
            else: IR["fresh"] = False
            col = IR["img"]
        else:
            col, _ = render_view(Rc, pc, own=k, Kc=Kc, head=(1.4 * float(np.clip(rng_ / 9.0, 0.12, 1.0))) if near else 0.0, znear=0.05 if near else 0.3)
        live.append(col); Rcs.append(Rc); pcs.append(pc)
    grays = [R_.degrade(c, rng) if ir_state(CAMS[i]) is None else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for i, c in enumerate(live)]
    fixes = [None] * 3; nin = [0] * 3; t_model = 0.0
    # COAST (Daniel 2026-11-11): wx = wz = 0 -> TOPEX spins steadily about one fixed axis, fully predictable once converged. Through
    # SPIN-MATCH the cameras are paused and the MEKF propagates on its rate estimate (any warp); when spin-match is reached it drops to
    # 1x and RE-LOCKS on vision (>= 10 accepted fixes) before FINAL APPROACH.
    # ... and only on a settled rate estimate: >= 20 s of tracking since the last (re-)acquire (a normal run has ~21 s); before that
    # SPIN-MATCH runs at 1x on vision (an X press during CONVERGE left 7 s, and the coast then chased a nutating rate estimate)
    COAST = A.ph == "SPIN-MATCH" and not getattr(A, "relock", False) and filt.q is not None and W.t - getattr(A, "t_acq", 0.0) >= getattr(A, "settle", 20.0)
    if COAST and getattr(A, "t_coast0", None) is None: A.t_coast0 = W.t
    if not COAST: A.t_coast0 = None
    if COAST and W.t - A.t_coast0 > 60.0:   # watchdog: a coast this long means the coasted rates are off -> back to 1x on vision, re-settle, replan
        A.t_acq = W.t; A.settle = 10.0; A.delta_goal = None; A.t_coast0 = None; COAST = False
        if _os.environ.get("DEMO_DEBUG"): print(f"[coast watchdog t={W.t:.1f}]", flush=True)
    FIRE = dict(tau=np.zeros(3), big=False, dv=np.zeros(3), det=None)   # what the docker's thrusters did this frame (for the plumes)
    if W.attached:
        pass   # gripped: TOPEX state comes from the docker's IMU through the rigid grip
    elif COAST:
        pass   # coasting: no vision this frame
    elif filt.q is None or A.ph == "ACQUIRE":
        if A.ph != "ACQUIRE": A.set("ACQUIRE", W.t)
        t0 = time.perf_counter(); ok = acquire(); t_model = time.perf_counter() - t0
        if ok:
            if STATS["loss_t"] and (not STATS["reacq_t"] or STATS["reacq_t"][-1] < STATS["loss_t"][-1]): STATS["reacq_t"].append(round(W.t, 2))
            A.settle = 10.0 if A.resume else 20.0; A.t_acq = W.t   # fresh vision before any coast: 20 s from scratch, 10 s on kept rates
            if getattr(A, "resume", None) == "SPIN-MATCH": A.relock = False; A.delta_goal = None   # replan the clocking and coast again
            A.set(A.resume if hasattr(A, "resume") and A.resume else "CONVERGE", W.t); A.resume = None
    else:
        idx = [i for i, k in enumerate(CAMS) if not (k == "docker" and (W.attached or not 11.0 <= np.linalg.norm(filt.p - W.sc["docker"]) <= 60.0))
               and not (ir_state(k) is not None and not IR.get("fresh", False))]
        pri = [filt.pose_in_camera(Rcs[i], pcs[i]) for i in idx]
        suns = [np.array([0.0, 0.0, -1.0]) if ir_state(CAMS[i]) == "ir" else (None if a.sun_unknown else Rcs[i].T @ W.sun) for i in idx]
        t0 = time.perf_counter(); out, nn = TRK.track([grays[i] for i in idx], pri, suns); t_model = time.perf_counter() - t0
        accepted = 0
        for j, i in enumerate(idx):
            if out[j] is None: continue
            fixes[i] = out[j]; nin[i] = nn[j]
            STATS["pos_err"].setdefault(CAMS[i], []).append((Rcs[i] @ out[j][1] + pcs[i]).tolist())   # truth TOPEX COM = origin
            if filt.update(Rcs[i] @ out[j][0], Rcs[i] @ out[j][1] + pcs[i]): accepted += 1
        if accepted: fix_times.append(W.t); lost_count = 0
        else: lost_count += 1
        if lost_count >= 4 and A.ph not in ("FINAL APPROACH", "GRIP"):   # estimate no longer explains any camera -> re-acquire (autonomy pauses where it was)
            A.resume = A.ph if A.ph not in ("ACQUIRE", "CONVERGE") else None; A.set("ACQUIRE", W.t); filt.q = None; lost_count = 0
            if getattr(A, "pf_hist", None):   # drop the COM average's last ~20 frames: they came from the estimate that just failed
                A.pf, A.npf = A.pf_hist[0]; A.pf_hist.clear()
    # ---- time step
    warp = a.detumble_warp if A.ph == "DETUMBLE" else (10.0 if A.ph == "BACK OFF" else (a.spinmatch_warp if COAST else 1.0)); dts = dt * warp
    if A.ph == "DETUMBLE":
        # TIME WARP applies to the detumble torque only: the stack's attitude advances by the real frame time at its true
        # instantaneous rate (smooth, no aliasing), while one 1 N thruster x 0.5 m (v2 / TRR) fired against H removes
        # 0.5 N*m*s of |H| per (warped) second, burning hydrazine at 0.5 N*m / (0.5 m * Isp * g0).
        W.R_IB, W.w = mekf.torque_free_step(W.R_IB, W.w, J_TRUE, dt, n=max(4, int(dt * 120)))
        HI = W.R_IB @ (J_TRUE @ W.w); hn = np.linalg.norm(HI); dH = min(hn, DETUMBLE_TAU * dts)
        if hn > 1e-9 and dH > 0: FIRE["det"] = -(W.R_sc["docker"].T @ HI) / hn   # torque direction (docker body) fighting H
        HI = HI * (hn - dH) / hn if hn > 1e-12 else HI * 0; STATS["fuel_detumble_kg"] += dH * KG_PER_NMS
        W.w = np.linalg.solve(J_TRUE, W.R_IB.T @ HI)
    else:
        W.R_IB, W.w = mekf.torque_free_step(W.R_IB, W.w, J_TRUE, dts, n=max(2, int(dts * 120)))
    if filt.q is not None:
        if W.attached:   # gripped: docker IMU + arm geometry give TOPEX attitude/rate directly (no vision needed)
            filt.q = mekf.R2q(W.R_IB); filt.w = W.w.copy()
        else:
            nn_ = max(1, int(dts * 240))
            for _ in range(nn_): filt.predict(dts / nn_)
    W.t += dts; t_demo += dt; t_go += dt; T_VIS += dt
    dtr = dt * APPROACH_WARP if A.ph == "FINAL APPROACH" else dts   # docker translation step
    # ---- mission clocks: SC1 flies its prox.csv transfer + glideslope to the 30 m hold (x300; the 200 m hold x3000), then the autonomy
    if not A.arrived:
        rate = 3000.0 if 7700.0 < TM_D < 14100.0 else 300.0; t_old_d, t_old_i = TM_D, TM_I
        TM_D = min(P_H30, TM_D + dt * rate); TM_I += TM_D - t_old_d
        clock_burns(0, t_old_d, TM_D); [clock_burns(kk, t_old_i, TM_I) for kk in (1, 2)]
        p_new = V2.rel(0, TM_D); W.v_dock = (p_new - V2.rel(0, TM_D - 1.0)); W.sc["docker"] = p_new
        if TM_D >= P_H30 - 1e-6:
            A.arrived = True; W.v_dock = np.zeros(3); A.d = float(np.linalg.norm(p_new)); print(f"[t={W.t:7.2f}] docker at the 30 m hold ({A.d:.1f} m)", flush=True)
    else:
        t_old_i = TM_I; TM_I = min(33500.0, TM_I + dts * (APPROACH_WARP if A.ph == "FINAL APPROACH" else 1.0)); [clock_burns(kk, t_old_i, TM_I) for kk in (1, 2)]
    place_free_flyers(); W.t_mission = TM_I
    # ---- autonomy (estimate only)
    est_ok = filt.q is not None
    if est_ok: R_est, w_est, p_est = mekf.q2R(filt.q), filt.w.copy(), filt.p.copy()
    sig_w = np.degrees(np.sqrt(np.diag(filt.P)[3:].max())) if est_ok else 99
    if A.ph == "CONVERGE":
        A.conv_t = A.conv_t + dts if sig_w < 0.5 else 0.0
        if A.conv_t > 3.0 and A.arrived: A.set("HOLD 30 m", W.t)
    elif A.ph == "HOLD 30 m" and W.t - A.t_ph > 2.0: A.set("SPIN-MATCH", W.t); A.delta_goal = None
    # docker guidance from the ESTIMATE. Hold on TOPEX's (inertially fixed) H axis and let the gripper absorb the wobble (team ConOps S2):
    # spin-up / twist on the hydrazine thruster couple (1 N*m), wobble (swing) tracking on the RW8 wheels (0.25 N*m), translation 2 N.
    # TOPEX COM for guidance: mean of the vision-updated estimate (fixed point: COM unforced), from 5 s into CONVERGE on, skipping
    # COAST frames (no new fixes there: averaging the propagated value would crowd out the real data); window capped at 300 frames
    if est_ok and not COAST and A.ph in ("CONVERGE", "HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH", "GRIP") and not (A.ph == "CONVERGE" and W.t - A.t_ph < 5.0):
        if not hasattr(A, "pf_hist"): A.pf_hist = collections.deque(maxlen=20)
        A.pf_hist.append((None if getattr(A, "pf", None) is None else A.pf.copy(), getattr(A, "npf", 0)))
        A.npf = getattr(A, "npf", 0) + 1
        A.pf = p_est.copy() if getattr(A, "pf", None) is None else A.pf + (p_est - A.pf) / min(A.npf, 300)
    if est_ok and A.ph in ("SPIN-MATCH", "FINAL APPROACH", "GRIP"):
        wI_est = R_est @ w_est; hI = R_est @ (J @ w_est); u_grip = R_est @ (DOCK_PB / np.linalg.norm(DOCK_PB))
        a_m = hI / np.linalg.norm(hI) * np.sign(hI @ u_grip) if np.linalg.norm(hI) > 1.0 else u_grip
        a_m = a_m if a_m @ u_grip > 0 else u_grip
        A.aH = a_m if A.aH is None else A.aH + (a_m - A.aH) * min(1.0, dts / 4.0); A.aH = A.aH / np.linalg.norm(A.aH)   # H is fixed: average the noisy estimate
        a_h = A.aH
        # TOPEX's COM is unforced (and CW drift over ~2 min at 30 m is negligible): the guidance uses a 30 s running average of the
        # estimate. The per-fix PnP position noise is 6-9 cm with ~1 cm mean bias per camera (pos_fix_bias_m), so a 3 s filter left
        # 5-9 cm of wander at the grip; the long average takes it to ~1-2 cm.
        if getattr(A, "pf", None) is None: A.pf = p_est.copy()
        p_est_g = A.pf
        R_des = R_est @ DOCK_RB
        # propagate the docker attitude over this step with the rates commanded last step (same instant as the truth/estimate)
        com = W.sc["docker"] + W.R_sc["docker"] @ C_DOCK      # W.v_dock is the COM velocity; the mesh origin follows the COM
        W.R_sc["docker"] = Rot.from_rotvec(W.wI_dock * dts).as_matrix() @ W.R_sc["docker"]
        Rd0 = W.R_sc["docker"]
        # attitude: split the error into twist about the spin axis n and swing (docking-axis pointing)
        n = a_h
        q = Rot.from_matrix(R_des @ Rd0.T).as_quat(); delta = 2 * math.atan2(q[:3] @ n, q[3]); delta = (delta + math.pi) % (2 * math.pi) - math.pi
        swing = (Rot.from_rotvec(-n * delta) * Rot.from_matrix(R_des @ Rd0.T)).as_rotvec()   # R_err = twist(n) after swing
        w_t = float(wI_est @ n); w_rel = w_t - float(W.wI_dock @ n)
        if A.ph == "SPIN-MATCH" and A.delta_goal is None:
            # plan once: brake the relative spin late at 0.9 alpha so the docker arrives phase-locked on the v2 clocking
            A.delta_u = delta; A.delta_prev = delta
            # nearest clocking the docker can reach WITHOUT overshooting (overshoot = leading TOPEX, capped at +0.06 rad/s: ~100 s at
            # wy 0.5). Brake budget 0.9 alpha.
            a9 = 0.9 * DOCK_ALPHA; s9 = (delta + w_rel * abs(w_rel) / (2 * a9)) / (2 * math.pi)
            if abs(w_rel) < 0.05: g = round(delta / (2 * math.pi))   # nearly matched (re-plan after a re-acquire): nearest clocking
            else: g = math.ceil(s9 - 1e-9) if w_rel > 0 else math.floor(s9 + 1e-9)
            A.delta_goal = 2 * math.pi * g; dist = abs(A.delta_goal - delta)
            A.a_prof = a9; A.w_cruise = max(abs(w_rel), 0.15)   # cruise at the current relative spin, brake late at 0.9 alpha (time-optimal)
            if _os.environ.get("DEMO_DEBUG"): print(f"[plan delta={math.degrees(delta):.1f} w_rel={w_rel:.3f} s9={s9:.3f} g={g} a_prof/alpha={A.a_prof / DOCK_ALPHA:.2f}]", flush=True)
        if A.delta_goal is not None:
            A.delta_u += (delta - A.delta_prev + math.pi) % (2 * math.pi) - math.pi; A.delta_prev = delta; e_tw = A.delta_goal - A.delta_u
        else: e_tw = delta
        dd_des = math.copysign(min(math.sqrt(2 * getattr(A, 'a_prof', 0.7 * DOCK_ALPHA) * abs(e_tw)), 0.6 * abs(e_tw), getattr(A, 'w_cruise', 9.9)), e_tw)
        sn = np.linalg.norm(swing); a_sw = RW_TAU / 41.62 if sn < math.radians(3.0) else DOCK_TAU / 41.62   # rate-profile shaping only; the Euler step below enforces the per-wheel 0.25 N*m
        sw_rate = swing / sn * min(0.8 * sn, math.sqrt(2 * 0.8 * a_sw * sn)) if sn > 1e-9 else np.zeros(3)
        w_cmd = (wI_est - (wI_est @ n) * n) + n * (w_t - dd_des) + sw_rate
        # docker rotational dynamics (Euler, body frame): tau = J dw/dt + w x J w, per-axis limited. Body y (gripper / spin axis)
        # torque = the hydrazine couple (1 N*m); body x, z = the RW8 wheels (0.25 N*m, 8 N*m*s), thrusters only for >3 deg swings.
        wB = Rd0.T @ W.wI_dock; gyro = np.cross(wB, J_DOCK @ wB); big = sn > math.radians(3.0)
        tau_req = J_DOCK @ (Rd0.T @ w_cmd - wB) / max(dts, 0.4) + gyro
        cap = np.array([DOCK_TAU if big else RW_TAU, DOCK_TAU, DOCK_TAU if big else RW_TAU]); tau = np.clip(tau_req, -cap, cap)
        wB = wB + np.linalg.solve(J_DOCK, tau - gyro) * dts; W.wI_dock = Rd0 @ wB
        STATS["fuel_spin_kg"] += abs(tau[1]) * dts * KG_PER_NMS
        FIRE["tau"] = tau.copy(); FIRE["big"] = big
        if big: STATS["fuel_swing_kg"] += (abs(tau[0]) + abs(tau[2])) * dts * KG_PER_NMS
        else:   # wheels: momentum exchange only (no propellant)
            A.hw = getattr(A, "hw", np.zeros(3)) - np.array([tau[0], 0.0, tau[2]]) * dts
            STATS["rw_h_max"] = max(STATS["rw_h_max"], float(np.abs(A.hw).max())); STATS["rw_tau_max"] = max(STATS["rw_tau_max"], float(max(abs(tau[0]), abs(tau[2]))))
        wcap = np.linalg.norm(wI_est) + 0.06; wn = np.linalg.norm(W.wI_dock)
        if wn > wcap: W.wI_dock *= wcap / wn
        W.w_dock = W.R_sc["docker"].T @ W.wI_dock
        # translation of the docker COM along the (inertially fixed) approach line, in <= 0.1 s substeps (the x10 approach warp
        # must not coast past the stop point), 2 N per axis, 1 cm deadband; the hold point is fixed in inertial space (on H)
        d_dock = float(np.linalg.norm(DOCK_COM_B)); nsub = max(1, int(math.ceil(dtr / 0.1 - 1e-9))); h_ = dtr / nsub
        for _ in range(nsub):
            com = com + W.v_dock * h_
            if A.ph == "FINAL APPROACH":
                rem = A.d - d_dock
                v_ref = min(DOCK_V_MAX if rem > 1.5 else DOCK_V_FINAL, math.sqrt(2 * 0.8 * DOCK_A_MAX * max(rem, 0.0)) + (0.003 if rem > 0.03 else 0.0))
                A.v_app = float(np.clip(v_ref, A.v_app - DOCK_A_MAX * h_, A.v_app + DOCK_A_MAX * h_))
                d_act = float((com - p_est_g) @ a_h)
                A.d = max(d_dock, A.d - A.v_app * h_, d_act - 0.15)
                if A.d <= d_dock + 1e-6: A.v_app = 0.0
            p_des = p_est_g + a_h * A.d
            vff = -(a_h * A.v_app) if A.ph == "FINAL APPROACH" else np.zeros(3)   # hold point fixed in inertial space
            pe = p_des - com; pn = np.linalg.norm(pe)
            v_cmd = vff + (pe / pn * min(0.15 * max(pn - 0.01, 0.0), math.sqrt(2 * 0.8 * DOCK_A_MAX * max(pn - 0.01, 0.0))) if pn > 1e-9 else 0)
            dv = v_cmd - W.v_dock; dn = np.linalg.norm(dv); dv_ = dv * min(1.0, DOCK_A_MAX * h_ / dn) if dn > 1e-12 else np.zeros(3)
            W.v_dock = W.v_dock + dv_; FIRE["dv"] = FIRE["dv"] + Rd0.T @ dv_; STATS["fuel_trans_kg"] += np.linalg.norm(dv_) * M_DOCK / (ISP * G0)
            vn = np.linalg.norm(W.v_dock)
            if vn > DOCK_V_MAX: W.v_dock *= DOCK_V_MAX / vn
        W.sc["docker"] = com - Rd0 @ C_DOCK
        if A.ph == "FINAL APPROACH":
            rl = (com - p_est_g) - ((com - p_est_g) @ a_h) * a_h; A.lat_est = float(np.linalg.norm(rl))   # COM off the estimated axis
            if A.d <= d_dock + 1e-6 and abs(d_act - d_dock) <= 0.01 and A.lat_est <= 0.02 and np.linalg.norm(W.v_dock) < 0.01:
                A.set("GRIP", W.t); A.t_grip_mission = W.t_mission
                if _os.environ.get("DEMO_DEBUG"): print("[grip] com_body", np.round(W.R_IB.T @ com, 3), "DOCK_COM_B", np.round(DOCK_COM_B, 3), "lat_est", round(A.lat_est, 3), "p_est", np.round(p_est_g, 3), "v", np.round(W.v_dock, 3))
        if A.ph == "SPIN-MATCH":
            w_rel_est = np.degrees(np.linalg.norm(mekf.relative_rate_body(w_est, R_est, W.w_dock, W.R_sc["docker"])))
            att_e = np.degrees(np.linalg.norm(Rot.from_matrix(R_des @ Rd0.T).as_rotvec()))   # before this step's rotation
            if _os.environ.get("DEMO_DEBUG") and int(W.t * 10) % 20 == 0: print(f"[sm t={W.t:.1f} e_tw={math.degrees(e_tw):.1f} swing={math.degrees(sn):.2f} wrel={w_rel_est:.2f} att={att_e:.2f} pn={pn:.2f} wd={math.degrees(np.linalg.norm(W.wI_dock)):.1f}]")
            if w_rel_est < 0.5 and att_e < 1.0 and pn < 0.05:
                if not getattr(A, "relock", False):
                    A.relock = True; A.relock_t0 = W.t; print(f"[t={W.t:7.2f}] spin matched on the coasted estimate -> 1x, re-locking on vision", flush=True)
                    if getattr(A, "loss_pending", False): A.loss_pending = False; corrupt_estimate()
                elif sum(1 for tf in fix_times if tf > A.relock_t0) >= 10:
                    A.set("FINAL APPROACH", W.t); A.v_app = 0.0; A.t_fa_mission = W.t_mission; A.d_fa0 = A.d
        if A.ph == "GRIP" and not getattr(A, "closed", False):
            # claw closes (2611 -> 1211 mm span over ~3 s) until a finger tip is 2 cm from the TOPEX surface
            ARM_S = max(ARM_CLOSED, ARM_S - dt * (1 - ARM_CLOSED) / 3.0)
            fb = (W.sc["docker"] + (FINGERS * [ARM_S, 1, 1]) @ W.R_sc["docker"].T) @ W.R_IB
            if TOPEX_KD.query(fb)[0].min() < 0.02:   # first contact: pad speed relative to the TOPEX surface it touches (truth)
                wI_t = W.R_IB @ W.w; fw = W.sc["docker"] + (FINGERS * [ARM_S, 1, 1]) @ W.R_sc["docker"].T
                vt = np.cross(wI_t, fw); vd = W.v_dock + np.cross(W.wI_dock, fw - (W.sc["docker"] + W.R_sc["docker"] @ C_DOCK))
                A.v_contact = float(np.linalg.norm(vt - vd, axis=1).max()); A.claw_contact = True; A.closed = True
                if _os.environ.get("DEMO_DEBUG"): print("[contact] v_dock", np.round(W.v_dock, 4), "wI_t", np.round(wI_t, 4), "wI_d", np.round(W.wI_dock, 4), "w x x", np.round(np.cross(wI_t, W.sc["docker"]), 4), "vrel", np.round(np.linalg.norm(vt - vd, axis=1), 3), "dpos_body", np.round(W.R_IB.T @ W.sc["docker"] - DOCK_PB, 3))
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
                               claw_span_m=round(2.611 * ARM_S, 3), claw_contact=bool(getattr(A, "claw_contact", False)), contact_speed_m_s=A.v_contact,
                               spinup_hydrazine_g=round(STATS["fuel_spin_kg"] * 1000, 1), rw_h_max_Nms=round(STATS["rw_h_max"], 3), rw_tau_max_Nm=round(STATS["rw_tau_max"], 3))
            e_ = pT - DOCK_PB; yb = DOCK_PB / np.linalg.norm(DOCK_PB)   # lateral = across the pad (the 75 mm tolerance); axial is taken up by the claw
            grip_report["grip_lateral_err_m"] = float(np.linalg.norm(e_ - (e_ @ yb) * yb)); grip_report["grip_axial_err_m"] = float(e_ @ yb)
            # rigid attach at the ACTUAL relative pose; truth becomes the stack (docker inertia + parallel axis; COM shift neglected)
            A.rel = (W.R_IB.T @ W.R_sc["docker"], W.R_IB.T @ W.sc["docker"]); r_ = A.rel[1] + A.rel[0] @ C_DOCK   # docker COM in TOPEX body
            Jst = J + A.rel[0] @ J_DOCK @ A.rel[0].T + M_DOCK * (r_ @ r_ * np.eye(3) - np.outer(r_, r_))
            H0 = J @ W.w + A.rel[0] @ J_DOCK @ (W.w_dock) + M_DOCK * np.cross(r_, np.cross(W.w, r_))
            J_TRUE = Jst; W.w = np.linalg.solve(Jst, H0)            # angular momentum conserved through the grip
            grip_report["stack_rate_after_grip_deg_s"] = float(np.degrees(np.linalg.norm(W.w)))
            grip_report["stack_H_Nms"] = float(np.linalg.norm(Jst @ W.w)); grip_report["detumble_est_s"] = grip_report["stack_H_Nms"] / DETUMBLE_TAU
            grip_report["detumble_hydrazine_est_kg"] = grip_report["stack_H_Nms"] * KG_PER_NMS; A.t_detumble = W.t
            W.attached = True; A.set("DETUMBLE", W.t)
        STATS["dock_v"].append(float(np.linalg.norm(W.v_dock))); STATS["dock_w"].append(float(np.degrees(np.linalg.norm(W.wI_dock))))
    elif W.attached:
        W.R_sc["docker"] = W.R_IB @ A.rel[0]; W.sc["docker"] = W.R_IB @ A.rel[1]; W.w_dock = A.rel[0].T @ W.w
        if A.ph == "DETUMBLE" and np.linalg.norm(W.w) < 0.00175: A.set("STABLE", W.t); grip_report["detumble_hydrazine_kg"] = round(STATS["fuel_detumble_kg"], 3); grip_report["detumble_h"] = round((W.t - A.t_detumble) / 3600, 2)
        elif A.ph == "STABLE" and W.t - A.t_ph > 5: A.set("RELEASE", W.t)
        elif A.ph == "RELEASE":   # claw opens (2 s), then the docker lets go
            ARM_S = min(1.0, ARM_S + dt * (1 - ARM_CLOSED) / 2.0)
            if ARM_S >= 1.0: W.attached = False; J_TRUE = J.copy(); A.set("BACK OFF", W.t); A.bo_v = 0.0
    elif A.ph == "BACK OFF":   # straight back along the arm axis: 0.1 m/s (TRR prox-ops +-0.1 m/s), gentle accel; shown at x10
        A.bo_v = min(0.10, A.bo_v + 0.005 * dts)
        W.sc["docker"] = W.sc["docker"] + W.R_sc["docker"] @ np.array([0, -1.0, 0]) * A.bo_v * dts; W.w_dock = np.zeros(3)
        if np.linalg.norm(W.sc["docker"]) > 20: A.set("DONE", W.t)
    # imagers keep pointing at the estimated TOPEX position; the docker faces it until the spin-match takes over
    for k in ("imagerA", "imagerB"): W.R_sc[k] = face_R(W.sc[k], filt.p if est_ok else np.zeros(3))
    if not est_ok or A.ph in ("ACQUIRE", "CONVERGE", "HOLD 30 m"): W.R_sc["docker"] = face_R(W.sc["docker"], filt.p if est_ok else np.zeros(3))
    if est_ok and not W.attached: STATS["filt_p"].append(filt.p.tolist() + [float(np.linalg.norm(W.sc["docker"]))]); STATS.setdefault("guid_p", []).append((getattr(A, "pf", None) if getattr(A, "pf", None) is not None else filt.p).tolist())
    # ---- scene poses for rendering
    set_scene_poses()
    # ---- metrics
    _H = W.R_IB @ (J_TRUE @ W.w); AXDEV = float(np.degrees(np.arccos(min(1.0, abs(W.R_IB[:, 1] @ _H) / max(np.linalg.norm(_H), 1e-12)))))   # roll axis vs H
    att_err = R_.rot_err_deg(R_est, W.R_IB) if est_ok else float("nan")
    rate_err = float(np.degrees(np.linalg.norm(w_est - W.w))) if est_ok else float("nan")
    wre = float(np.degrees(np.linalg.norm(mekf.relative_rate_body(w_est, R_est, W.w_dock, W.R_sc["docker"])))) if est_ok else float("nan")
    wrt = float(np.degrees(np.linalg.norm(mekf.relative_rate_body(W.w, W.R_IB, W.w_dock, W.R_sc["docker"]))))
    HIST.append(dict(t=W.t, wx=math.degrees(W.w[0]), wy=math.degrees(W.w[1]), wz=math.degrees(W.w[2]),
                     ex=math.degrees(w_est[0]) if est_ok else np.nan, ey=math.degrees(w_est[1]) if est_ok else np.nan, ez=math.degrees(w_est[2]) if est_ok else np.nan,
                     att=att_err, wre=wre, wrt=wrt, ph=A.ph, axdev=AXDEV, dpos=float(np.linalg.norm(W.R_IB.T @ W.sc["docker"] - DOCK_PB)))); FULL.append(HIST[-1])
    cyc = time.perf_counter() - c0
    fhz = (len(fix_times) - 1) / max(fix_times[-1] - fix_times[0], 1e-6) if len(fix_times) > 2 else 0.0
    if A.ph not in ("ACQUIRE",): STATS["cycle_ms"].append(cyc * 1000); STATS["model_ms"].append(t_model * 1000)
    # ---- main view: one continuous camera along the v2 path, advanced by ConOps progress (smoothed), never a cut
    if not A.arrived: v_tgt = 39.0 + 10.0 * (TM_D - P_BID) / (P_H30 - P_BID)
    elif A.ph == "SPIN-MATCH": v_tgt = 49.0 + 3.0 * min(1.0, (W.t - A.t_ph) / 45.0)
    elif A.ph == "FINAL APPROACH": v_tgt = 52.0 + 5.5 * float(np.clip((A.d_fa0 - A.d) / max(A.d_fa0 - np.linalg.norm(DOCK_PB), 1e-6), 0, 1)) if hasattr(A, "d_fa0") else 52.0
    elif A.ph == "GRIP": v_tgt = 57.5
    elif A.ph == "DETUMBLE": v_tgt = 58.0 + 8.0 * (1.0 - min(1.0, np.linalg.norm(W.w) / max(math.radians(grip_report["stack_rate_after_grip_deg_s"]) if grip_report else 1.0, 1e-9)))
    elif A.ph in ("STABLE", "RELEASE"): v_tgt = 66.0
    elif A.ph in ("BACK OFF", "DONE"): v_tgt = 66.0 + 12.0 * float(np.clip((np.linalg.norm(W.sc["docker"]) - np.linalg.norm(DOCK_PB)) / (20.0 - np.linalg.norm(DOCK_PB)), 0, 1))
    else: v_tgt = 49.0
    V_CAM = v_tgt if not A.arrived else V_CAM + (v_tgt - V_CAM) * min(1.0, dt / 1.5)
    Rm, Pm, dm = V2.cam(V_CAM)
    a_pan = float(np.clip(t_go / 1.5, 0, 1)); a_pan = a_pan * a_pan * (3 - 2 * a_pan); roles_s = float(np.clip(t_go / 1.0, 0, 1))
    cx_main = 640.0 * (1 - a_pan) + 384.0 * a_pan
    main_img = render_main(Rm, Pm, dm, V_CAM, WIN_W if a_pan < 1.0 else 768, cx_main, roles_s, flashes(dt))
    canvas = np.zeros((WIN_H, WIN_W, 3), np.uint8)
    canvas[:, 0:768] = main_img[:, :768]
    if A.ph in ("FINAL APPROACH", "GRIP", "DETUMBLE", "STABLE", "RELEASE", "BACK OFF") and np.linalg.norm(W.sc["docker"]) < 14:
        # overhead view of the claw: above the arm, behind the pad plane, looking down at both fingers and the grip face
        Rd_, pd_ = W.R_sc["docker"], W.sc["docker"]
        pa = pd_ + Rd_ @ np.array([0.0, 0.55, 1.75]); ta = pd_ + Rd_ @ np.array([0.0, 1.20, 0.10])
        acol, _ = render_view(look_at(pa, ta, up=Rd_[:, 1]), pa, own=None, Kc=Ka, head=0.35, znear=0.05)
        ARM_IMG = auto_expose(cv2.cvtColor(acol, cv2.COLOR_RGB2GRAY), 160)
    else: ARM_IMG = None
    CHASE_IMG = None
    if A.ph in ("SPIN-MATCH", "FINAL APPROACH", "GRIP", "DETUMBLE") and A.arrived:
        Rd_, pd_ = W.R_sc["docker"], W.sc["docker"]
        pc_ = pd_ + Rd_ @ CHASE_OFF; Rc_ = look_at(pc_, pd_ + Rd_ @ np.array([0.0, 2.5, 0.0]), up=Rd_[:, 2])
        ccol, _ = render_view(Rc_, pc_, own=None, Kc=Kch, head=0.6, znear=0.05)
        CHASE_IMG = draw_plumes(cv2.cvtColor(ccol, cv2.COLOR_RGB2BGR), Rc_, pc_, Rd_, pd_, FIRE, frame)
    warp_disp = APPROACH_WARP if A.ph == "FINAL APPROACH" else warp
    if not A.arrived:
        warp_tag(canvas, 22, 32, f"T+{hms(TM_D)}   TIME x{3000 if 7700 < TM_D < 14100 else 300}", "SC1 (docker): transfer to the 200 m hold, glideslope to 30 m (prox.csv)")
    elif warp_disp > 1.5:
        cv2.rectangle(canvas, (14, 642), (600, 704), (20, 20, 60), -1); cv2.rectangle(canvas, (14, 642), (600, 704), (80, 80, 255), 1)
        txt(canvas, f"TIME WARP x{warp_disp:.0f}", 26, 680, 0.9, (120, 160, 255), 2)
        if A.ph == "DETUMBLE":
            l1 = "detumble torque warped; spin drawn at its true rate"
            l2 = f"hydrazine {STATS['fuel_detumble_kg']:.2f} / ~{grip_report['detumble_hydrazine_est_kg']:.2f} kg   {(W.t - A.t_detumble) / 3600:.2f} / {grip_report['detumble_est_s'] / 3600:.1f} h"
        elif COAST: l1, l2 = "docker spin-up on the 1 N*m thruster couple", "spin drawn at its warped rate"
        elif A.ph == "FINAL APPROACH": l1, l2 = f"approach translation warped ({A.v_app:.3f} m/s real);", "spin drawn at its true rate"
        else: l1, l2 = "translation warped", "(0.1 m/s back-off)"
        txt(canvas, l1, 250, 664, 0.42, (200, 200, 255)); txt(canvas, l2, 250, 686, 0.45, (150, 255, 190) if A.ph == "DETUMBLE" else (200, 200, 255), 1)
    for i, k in enumerate(CAMS):
        x0, y0 = 768 + (i % 2) * 256, (i // 2) * 256
        gt = grays[i] if k != "docker" else auto_expose(grays[i], 150)
        tile = cv2.cvtColor(cv2.resize(gt, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        if est_ok and filt.q is not None and not (k == "docker" and Kcs[i] is Kn):
            draw_axes_overlay(tile, Rcs[i].T @ W.R_IB, Rcs[i].T @ (np.zeros(3) - pcs[i]), (120, 120, 120), 256 / RES, Kcs[i])
            draw_axes_overlay(tile, Rcs[i].T @ R_est, Rcs[i].T @ (p_est - pcs[i]), None, 256 / RES, Kcs[i])
        if k == "docker" and Kcs[i] is Kn: txt(tile, "near-field: 62 deg FOV, illuminator, auto-exp", 6, 34, 0.36, (150, 220, 255))
        if ir_state(k) == "dark": txt(tile, "VISIBLE CAMERA FAILED", 30, 128, 0.55, (60, 60, 255), 2)
        elif ir_state(k) == "ir": txt(tile, "IR: ECAM-IR3A LWIR 640x480 12 fps", 6, 34, 0.36, (80, 170, 255))
        canvas[y0:y0 + 256, x0:x0 + 256] = tile
        cv2.rectangle(canvas, (x0, y0), (x0 + 255, y0 + 255), tuple(int(c * 255) for c in ACC[k][::-1]), 1)
        txt(canvas, f"{k} cam  {np.linalg.norm(W.sc[k]):.0f} m", x0 + 6, y0 + 16, 0.45, tuple(int(c * 255) for c in ACC[k][::-1]))
        txt(canvas, f"fix: {'yes' if fixes[i] is not None else '-'}  inl {nin[i]}", x0 + 6, y0 + 248, 0.4, (200, 200, 200))
    x0, y0 = 768 + 256, 256
    fuel_g = (STATS["fuel_spin_kg"] + STATS["fuel_swing_kg"] + STATS["fuel_trans_kg"]) * 1000
    lines = [f"PHASE: {A.ph if A.arrived or A.ph not in ('HOLD 30 m',) else A.ph}" + ("" if A.arrived else "  (docker en route)"), f"t {W.t:6.1f} s  mission T+{hms(W.t_mission)}  x{warp_disp:.0f}",
             "TOPEX rate deg/s   truth / estimate",
             f"  x {math.degrees(W.w[0]):7.2f} / {(math.degrees(w_est[0]) if est_ok else float('nan')):7.2f}",
             f"  y {math.degrees(W.w[1]):7.2f} / {(math.degrees(w_est[1]) if est_ok else float('nan')):7.2f}",
             f"  z {math.degrees(W.w[2]):7.2f} / {(math.degrees(w_est[2]) if est_ok else float('nan')):7.2f}",
             f"att err {att_err:5.2f} deg  rate err {rate_err:5.3f}",
             f"docker rel rate est {wre:5.2f} true {wrt:5.2f} deg/s",
             f"N2H4: spin/approach {fuel_g:5.1f} g  detumble {STATS['fuel_detumble_kg']:4.2f} kg",
             f"RW8 |h| {float(np.abs(getattr(A, 'hw', np.zeros(3))).max()):4.2f}/8 N*m*s  max {STATS['rw_tau_max']:4.3f} N*m",
             f"fixes {fhz:4.1f} Hz (sim)  cycle {cyc*1000:5.0f} ms", f"{'acquire' if A.ph == 'ACQUIRE' else 'model'} {t_model*1000:5.0f} ms ({a.feat})"]
    for j, s_ in enumerate(lines): txt(canvas, s_, x0 + 6, y0 + 18 + j * 19, 0.4, (120, 230, 255) if j == 0 else (225, 225, 225))
    PW = 504 if ARM_IMG is None else 300
    if A.ph in ("DETUMBLE", "STABLE", "RELEASE", "BACK OFF", "DONE") and grip_report:
        x0p, y0p, wp, hp = 772, 518, PW, 96; cv2.rectangle(canvas, (x0p, y0p), (x0p + wp, y0p + hp), (60, 60, 60), 1)
        pts = [(h_["t"], math.sqrt(h_["wx"] ** 2 + h_["wy"] ** 2 + h_["wz"] ** 2)) for h_ in FULL if h_["t"] >= grip_report["t"] - 0.01]
        if pts:
            tmax = max(grip_report.get("detumble_est_s", 1.0) * 1.05, pts[-1][0] - grip_report["t"] + 1); wmax0 = max(grip_report["stack_rate_after_grip_deg_s"] * 1.1, 1e-3)
            P_ = np.array([[x0p + (t_ - grip_report["t"]) / tmax * wp, y0p + hp - w_ / wmax0 * hp] for t_, w_ in pts], np.int32)
            if len(P_) > 1: cv2.polylines(canvas, [P_], False, (120, 255, 160), 2, cv2.LINE_AA)
            txt(canvas, f"STACK |w| {pts[-1][1]:.3f} deg/s  {(W.t - grip_report['t']) / 3600:.2f} / {tmax / 3600:.1f} h", x0p + 4, y0p + 14, 0.4, (200, 255, 210))
            yl = y0p + hp - int(math.degrees(0.00175) / wmax0 * hp); cv2.line(canvas, (x0p, yl), (x0p + wp, yl), (90, 90, 90), 1)
    else: plot(canvas, 772, 518, PW, 96, ["wx", "wy", "wz", "ex", "ey", "ez"], [(90, 90, 255), (90, 255, 90), (255, 140, 90), (180, 180, 255), (180, 255, 180), (255, 200, 170)],
               -50, 50, "TOPEX body rates deg/s (dark truth, light est)" if PW > 400 else "TOPEX rates deg/s")
    plot(canvas, 772, 620, PW, 96, ["wrt", "wre", "att"], [(255, 255, 255), (100, 200, 255), (100, 100, 255)], 0, 10,
         "docker rel. rate true (white) / est (blue), att err deg (red), 0-10" if PW > 400 else "rel rate true/est, att err, 0-10")
    if CHASE_IMG is not None:
        xc0, yc0 = 12, 300; canvas[yc0:yc0 + CHRES, xc0:xc0 + CHRES] = CHASE_IMG
        cv2.rectangle(canvas, (xc0, yc0), (xc0 + CHRES - 1, yc0 + CHRES - 1), (60, 140, 240), 1)
        txt(canvas, "CHASE CAM  (thrusters)", xc0 + 6, yc0 + 16, 0.42, (60, 170, 255))
    if ARM_IMG is not None:
        x0a, y0a = 1078, 518
        canvas[y0a:y0a + ARMRES, x0a:x0a + ARMRES] = cv2.cvtColor(ARM_IMG, cv2.COLOR_GRAY2BGR)[:WIN_H - y0a]
        cv2.rectangle(canvas, (x0a, y0a), (x0a + ARMRES - 1, y0a + ARMRES - 1), (60, 140, 240), 1)
        txt(canvas, f"ARM CAM  span {2.611 * ARM_S:.2f} m", x0a + 6, y0a + 16, 0.4, (60, 170, 255))
    if grip_report: txt(canvas, f"GRIP: lateral err {grip_report['grip_lateral_err_m'] * 1000:.0f} mm, contact {grip_report['contact_speed_m_s'] * 100:.1f} cm/s, pads-to-dish {grip_report['min_pad_to_dish_m']:.2f} m", 8, 22, 0.45, (120, 255, 160))
    if a_pan < 1.0:   # GO: the dashboard slides in from the right over the full-width picture
        canvas[:, 768:] = (main_img[:, 768:WIN_W] * (1 - a_pan) + canvas[:, 768:] * a_pan).astype(np.uint8)
    r = emit(canvas, 768); frame += 1
    if r == "quit": break
    for kk in r:
        if kk == "x": apply_cmd("loss")
        elif kk == "i": apply_cmd("ir")
        elif kk == "l": apply_cmd("light")
    if (a.headless and a.T > 0 and t_demo >= a.T) or (a.headless and A.ph == "DONE" and W.t > A.t_ph + 3) or (a.until_grip and grip_report and A.ph == "DETUMBLE" and W.t > A.t_ph + 1500): break
if writer is not None: writer.release()
summary = dict(selftest=SELFTEST, feat=a.feat, crop=a.crop, script=a.script, fixed_dt=a.fixed_dt, sun_unknown=a.sun_unknown, w0=W.w.tolist() if not grip_report else None, phases=A.log, grip=grip_report,
               median_cycle_ms=float(np.median(STATS["cycle_ms"])) if STATS["cycle_ms"] else None, median_model_ms=float(np.median(STATS["model_ms"])) if STATS["model_ms"] else None,
               loss_t=STATS["loss_t"], reacq_t=STATS["reacq_t"], light_t=STATS["light_t"], ir_t=STATS["ir_t"],
               att_err_med_tracking=float(np.nanmedian([h["att"] for h in FULL if h["ph"] in ("CONVERGE", "HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH")])) if FULL else None,
               rate_err_med_tracking=float(np.nanmedian([abs(np.linalg.norm([h["wx"]-h["ex"], h["wy"]-h["ey"], h["wz"]-h["ez"]])) for h in FULL if h["ph"] in ("HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH")])) if FULL else None,
               grip_t_mission_s=getattr(A, "t_grip_mission", None), final_approach_mission_s=(getattr(A, "t_grip_mission", 0) - getattr(A, "t_fa_mission", 0)) if hasattr(A, "t_grip_mission") else None,
               axis_dev_max_deg=float(np.nanmax([h["axdev"] for h in FULL if h["ph"] in ("ACQUIRE", "CONVERGE", "HOLD 30 m", "SPIN-MATCH", "FINAL APPROACH", "GRIP")])) if FULL else None,
               fuel_spinup_g=STATS["fuel_spin_kg"] * 1000, fuel_swing_g=STATS["fuel_swing_kg"] * 1000, fuel_translation_g=STATS["fuel_trans_kg"] * 1000, fuel_detumble_kg=STATS["fuel_detumble_kg"],
               rw_h_max_Nms=STATS["rw_h_max"],
               pos_fix_bias_m={k: dict(mean=np.round(np.mean(v, 0), 3).tolist(), med_norm=round(float(np.median(np.linalg.norm(v, axis=1))), 3), n=len(v)) for k, v in STATS["pos_err"].items()},
               filt_p_bias_m=dict(mean=np.round(np.mean(np.array(STATS["filt_p"])[:, :3], 0), 3).tolist(), last50_mean=np.round(np.mean(np.array(STATS["filt_p"])[-50:, :3], 0), 3).tolist(), guidance_last=np.round(STATS["guid_p"][-1], 3).tolist()) if STATS["filt_p"] else None, rw_tau_max_Nm=STATS["rw_tau_max"],
               max_docker_speed_m_s=max(STATS['dock_v']) if STATS['dock_v'] else None, max_docker_rate_deg_s=max(STATS['dock_w']) if STATS['dock_w'] else None, t_demo=t_demo, t_sim=W.t,
               rel_rate_true_final_approach=float(np.nanmedian([h["wrt"] for h in FULL if h["ph"] == "FINAL APPROACH"])) if any(h["ph"] == "FINAL APPROACH" for h in FULL) else None)
print(json.dumps(summary))
if a.metrics: open(a.metrics, "a").write(json.dumps(summary) + "\n")
