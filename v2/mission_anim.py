"""SWARM settling TOPEX/Poseidon -- transfer + proximity-ops animation (v2).

Single continuous shot driven by the C++ trajectory sim (cluster_traj.cpp):
  Earth view -> phasing -> BHT-600 low-thrust spiral -> zoom into TOPEX LVLH ->
  rendezvous -> role bid -> NMC inspection -> glideslope -> dock ->
  arm open + wheel spin-up -> capture -> despin -> stable hold -> release -> depart.

    python3 mission_anim.py preview 12 40 63 [--scale 0.5]   # stills -> out/preview/
    python3 mission_anim.py clip despin --fps 10 --scale 0.5  # one phase, low-res -> out/clip_despin.mp4
    python3 mission_anim.py render                            # full 1080p30 -> topex_mission.mp4 (slow)

Notes
  * Translational motion of all bodies comes from transfer.csv / prox.csv.
  * TOPEX spin is drawn at its true rate in video time (translation is time-warped).
  * Docker gripper: 4 actuations only (open, close on capture, open to release, close), 60 s each.
  * Detumble: constant 0.5 N.m (one 1 N hydrazine thruster on a 0.5 m arm) on TOPEX Iz = 11,665
    kg m^2, 0.661 rad/s -> MR-4 band |w| <= 0.00175 rad/s (time/propellant read from summary.txt).
  * Spacecraft bus sized to midterm Section 4 (600 x 1021 x 1508 mm, 4.34 m array span).
  * Earth rotation is rate-capped in video so it does not strobe during the ~68-day spiral.
"""
import os, sys, re, glob, math, subprocess
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline
import matplotlib
matplotlib.use("Agg")
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection, PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.environ.get("SIM_DIR", os.path.join(HERE, "out"))
for f in glob.glob(os.path.join(HERE, "assets", "*.ttf")):
    fm.fontManager.addfont(f)

# ------------------------------------------------------------------ look
W, H, FPS = 1920, 1080, 30
FOV = math.radians(34.0)
FPX = (W / 2) / math.tan(FOV / 2)
BG = "#04070b"
INK, MUTED, DIM = "#e8edf2", "#7d8a98", "#2f3a46"
C_NEU, C_DOCK, C_IMG = "#d2dae3", "#f0a640", "#6cc7da"
C_RED, C_TGT = "#e2645c", "#f4f6f8"
SANS_L = ["Barlow Light", "DejaVu Sans"]
SANS = ["Barlow", "DejaVu Sans"]
SANS_M = ["Barlow Medium", "DejaVu Sans"]
MONO = ["IBM Plex Mono", "DejaVu Sans Mono"]

RE_ = 6378.137e3
MU = 398600.4418e9
WE = 7.2921159e-5

# ------------------------------------------------------------------ sim data
tr = pd.read_csv(os.path.join(SIM, "transfer.csv"))
px = pd.read_csv(os.path.join(SIM, "prox.csv"))
bu = pd.read_csv(os.path.join(SIM, "burns.csv"))
ev = pd.read_csv(os.path.join(SIM, "events.csv"))
summ = open(os.path.join(SIM, "summary.txt")).read()
m = re.search(r"Sim \(J2\) spiral\s+: dv = ([\d.]+) m/s, spiral time = ([\d.]+) days", summ)
SP_DAYS = float(m.group(2))
# reported spiral: Edelbaum x 1.02 steering factor (trr/SPIRAL-POWER.md), xenon from the rocket equation
m = re.search(r"Reported spiral\s+: dv = ([\d.]+) m/s .*xenon = ([\d.]+) kg", summ)
DV_SP, XE_USED = float(m.group(1)), float(m.group(2))
MP_K = [float(x) for x in re.findall(r"SC\d hydrazine: ([\d.]+) of", summ)]
m = re.search(r"Detumble time = ([\d.]+) s = [\d.]+ h; hydrazine = ([\d.]+) kg", summ)
T_DET, MP_DT = float(m.group(1)), float(m.group(2))
m = re.search(r"Hall duty\s+: last ([\d.]+)% \(500 km\) to ([\d.]+)%", summ)
DUTY0, DUTY1 = float(m.group(1)), float(m.group(2))
m = re.search(r"hydrazine top-up ([\d.]+) kg", summ)
MP_SU = float(m.group(1))
m = re.search(r"Docker hydrazine: prox ([\d.]+) \+ spin-up ([\d.]+) \+ detumble ([\d.]+) = ([\d.]+) kg", summ)
MP_DOCK = float(m.group(4))
DVP = [float(x) for x in re.findall(r"prox-ops dv = ([\d.]+)", summ)]
MINSEP = min(float(x) for x in re.findall(r"SC\d-SC\d ([\d.]+) m", summ))

tr = tr.drop_duplicates("t").reset_index(drop=True)
tt = tr.t.values
T_ARR = tt[-1]
T_B1 = tt[np.argmax(tr.phase.values >= 1)]
spT = CubicSpline(tt, tr[["tx", "ty", "tz"]].values)
spC = [CubicSpline(tt, tr[[f"c{k}x", f"c{k}y", f"c{k}z"]].values) for k in range(3)]
CTRK = tr[["c0x", "c0y", "c0z"]].values          # raw logged SC1 track (spiral coils)
DV0_T = tr.dv0.values
rA, vA = spT(T_ARR), spT(T_ARR, 1)
hA = np.cross(rA, vA); hA /= np.linalg.norm(hA)
nA = np.linalg.norm(vA) / np.linalg.norm(rA)


def rodrigues(v, k, a):
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))


def topex_rv(t):
    if t <= T_ARR:
        return spT(t), spT(t, 1)
    a = nA * (t - T_ARR)
    return rodrigues(rA, hA, a), rodrigues(vA, hA, a)


def lvlh(t):
    r, v = topex_rv(t)
    x = r / np.linalg.norm(r); z = np.cross(r, v); z /= np.linalg.norm(z); y = np.cross(z, x)
    return np.array([x, y, z]), r


# relative trajectories (LVLH) incl. the end of the transfer arc
tpre = np.arange(T_ARR - 2400, T_ARR, 20.0)
REL_T = np.r_[tpre - T_ARR, px.t.values]
REL = []
for k in range(3):
    pre = []
    for t in tpre:
        C, r = lvlh(t)
        pre.append(C @ (spC[k](t) - r))
    REL.append(np.vstack([np.array(pre), px[[f"x{k}", f"y{k}", f"z{k}"]].values]))
DVK = [np.interp(np.r_[px.t.values], px.t.values, px[f"dv{k}"].values) for k in range(3)]


DEP_DIR = np.array([-0.25, -1.0, 0.0]) / np.hypot(0.25, 1.0)   # leaving: back along -V-bar, slightly low
DEP_A = 2e-4   # visual only: Hall-thruster-scale acceleration for the departure shot (no number is shown)


def relv(k, ts):
    ts = np.atleast_1d(ts).astype(float)
    P = np.array([np.interp(ts, REL_T, REL[k][:, j]) for j in range(3)]).T
    tau = np.maximum(ts - P_END, 0.0)
    return P + (0.5 * DEP_A * tau ** 2)[:, None] * DEP_DIR


def rel(k, trel):
    return relv(k, trel)[0]


def evt(craft, prefix, which="t0"):
    e = ev[(ev.craft == craft) & ev.name.str.startswith(prefix)]
    return float(e[which].iloc[0])


P_HOLD = evt(0, "Hold: detect")
P_BID = evt(0, "Hold: detect", "t1")
P_NMC2 = evt(2, "NMC inspection")
P_GO = evt(0, "Glideslope")
P_H30 = evt(0, "Hold 30")
DOCK_R = float(-px.y0[(px.t >= evt(0, "Docked")) & (px.t < evt(0, "Docked", "t1"))].iloc[5])
P_FA = evt(0, "Final approach")
P_CON = evt(0, "Docked")
# detumble model: constant 0.5 N.m on Iz = 11,665 kg m^2, target 0, MR-4 band +-0.00175 rad/s
W0, W_BAND, TAU_DT, IZ = 2 * math.pi / 9.5, 0.00175, 0.5, 11665.0
ALPHA = TAU_DT / IZ                   # 4.29e-5 rad/s^2
P_DT = P_CON + T_DET
P_SVC = P_DT + 3600.0
P_I2OUT = evt(2, "Depart")
P_UND = evt(0, "Undock")
P_BO = evt(0, "Depart to regroup")
ARM_T = 60.0   # actuation time (TRR budgets 'Energy Balance' AO13)
P_END = float(ev.t1.max())

# ------------------------------------------------------------------ video timeline
KN = [(0, 0), (4, 600), (8.6, T_B1 - 300), (10.2, T_B1 + 2400), (15.6, T_ARR - 4 * 3600), (18.6, T_ARR - 900), (21, T_ARR),
      (27, T_ARR + P_HOLD), (31, T_ARR + P_BID), (39, T_ARR + P_NMC2), (43, T_ARR + P_GO),
      (49, T_ARR + P_H30), (52, T_ARR + P_FA), (57, T_ARR + P_CON), (66, T_ARR + P_DT),
      (71, T_ARR + P_SVC), (74, T_ARR + P_I2OUT), (77, T_ARR + P_UND), (84, T_ARR + P_END),
      (90, T_ARR + P_END + 1500), (96, T_ARR + P_END + 3600), (102, T_ARR + P_END + 5400)]
DUR = KN[-1][0]
V_SW = 21.0
vmap = PchipInterpolator([k[0] for k in KN], [k[1] for k in KN])
NF = int(DUR * FPS)
VV = np.arange(NF) / FPS
TG = vmap(VV)
RATE = vmap.derivative()(VV)
# visual clock for Earth spin / LVLH sun: capped at 1500x so the long spiral doesn't strobe
_re = np.minimum(np.where(TG <= T_ARR, RATE, 0.06 * RATE), 1500.0)
TVIS = np.cumsum(_re) / FPS
TVIS += T_ARR - TVIS[int(V_SW * FPS)]

# phases follow the autonomy flowchart (midterm Fig. flowchart, updated in v2/flowchart)
PHASES = [("Climb to TOPEX's orbit", 0, 21), ("Searching for TOPEX", 21, 27), ("Picking the docker", 27, 31),
          ("Imagers survey TOPEX", 31, 39), ("Docker waits at 200 m", 39, 43), ("Docker closes to 30 m", 43, 49),
          ("Docker matches TOPEX's spin", 49, 52), ("Capture", 52, 57), ("Slowing TOPEX down", 57, 66),
          ("TOPEX is stable", 66, 74), ("Letting go", 74, 84), ("Heading to the next target", 84, DUR)]
CAPS = [
    (4.6, 8.6, "All three wait in a 500 km orbit until TOPEX is in the right spot."),
    (9.0, 15.4, f"The Hall thruster runs about {DUTY0:.0f} to {DUTY1:.0f}% of each orbit. The climb takes {SP_DAYS:.0f} days and {XE_USED:.1f} kg of xenon."),
    (15.8, 21.0, "They stop 5 to 7 km behind TOPEX and about 1 km below it."),
    (21.4, 27.0, "Small hydrazine burns bring them to 1 to 1.6 km behind TOPEX, then the cameras look for it."),
    (27.4, 31.0, "The closest one becomes the docker. The other two image."),
    (31.4, 39.0, "The imagers fly around TOPEX at 22 m and measure how it is spinning."),
    (39.4, 43.0, "The docker waits until the spin axis is known."),
    (43.4, 49.0, "The docker moves in along TOPEX's spin axis."),
    (49.4, 52.0, f"The docker spins up to 0.66 rad/s, mostly on its wheel ({MP_SU * 1000:.0f} g of hydrazine)."),
    (52.4, 57.0, f"The docker moves in at 3 cm/s and stops {DOCK_R:.1f} m from TOPEX's center."),
    (57.4, 65.7, f"Hydrazine thrusters slow TOPEX from 0.66 to 0.00175 rad/s over {T_DET / 3600:.1f} h, using {MP_DT:.1f} kg."),
    (66.0, 74.0, "TOPEX is below 0.00175 rad/s. The imagers check it."),
    (74.4, 84.0, "The imagers leave, then the docker lets go and joins them about 1 km behind TOPEX."),
    (84.4, 95.0, "Hall thrusters back on, heading to the next target."),
]

# ------------------------------------------------------------------ attitude / spin
def omega(trel):
    if trel < P_CON: return W0
    return max(W0 - ALPHA * (trel - P_CON), 0.0)


TREL = TG - T_ARR
PHI = np.zeros(NF); ROLL = np.zeros(NF)
for i in range(1, NF):
    w = omega(TREL[i])
    PHI[i] = PHI[i - 1] + w / FPS
    if TREL[i] >= P_CON:
        ROLL[i] = PHI[i]
    else:
        s = np.clip((TREL[i] - P_H30) / 450.0, 0, 1); s = s * s * (3 - 2 * s)
        err = (PHI[i] - ROLL[i - 1] + np.pi) % (2 * np.pi) - np.pi
        ROLL[i] = ROLL[i - 1] + s * (w + 1.2 * err) / FPS

# ------------------------------------------------------------------ geometry helpers
def unit(v): return v / np.linalg.norm(v)


def slerp(a, b, u):
    a, b = unit(a), unit(b)
    d = np.clip(np.dot(a, b), -1, 1); th = math.acos(d)
    if th < 1e-6: return unit(a + (b - a) * u)
    return unit((math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th))


def smoother(u):
    u = np.clip(u, 0, 1); return u * u * u * (u * (u * 6 - 15) + 10)


def ss(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u)


def window(v, a, b, fade=0.45):
    return float(ss(a, a + fade, v) * (1 - ss(b - fade, b, v)))


DOCK_CLOCK = -math.pi / 2   # docker rolled 90 deg about its docking axis so the open claw clears TOPEX's dish


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ------------------------------------------------------------------ camera keyframes
CA, rAa = lvlh(T_ARR)
e1 = unit(spT(0.0)); nrm = unit(np.cross(spT(0.0), spT(0.0, 1))); e2 = np.cross(nrm, e1)


def dir_lvlh(az, el):
    az, el = math.radians(az), math.radians(el)
    return np.array([math.sin(el), -math.cos(el) * math.cos(az), math.cos(el) * math.sin(az)])


XH = np.array([1.0, 0, 0])
SUN_L = unit(np.array([0.50, -0.15, -0.85]))
D_ZOOM = dir_lvlh(-35, 30)
D1_eci = CA.T @ D_ZOOM
D0_eci = unit(rodrigues(unit(CA.T @ SUN_L), nrm, math.radians(-48)) + 0.30 * nrm)
_r0 = unit(np.cross(-D0_eci, nrm))
T0_eci = -_r0 * 7200e3
KEYS = [  # (v, frame, target, dir, dist, up)
    (0.0, "eci", T0_eci, D0_eci, 46000e3, nrm),
    (15.0, "eci", 0.72 * rAa, D1_eci, 36000e3, unit(nrm + 1.6 * (CA.T @ XH))),
    (21.0, "lvlh", np.array([-1600.0, -3600.0, 0]), D_ZOOM, 15000.0, XH),
]


def to_frame(key, frame, C, r):
    _, kf, T, D, d, U = key
    if kf == frame: return T, D, d, U
    if kf == "lvlh":  # -> eci
        return r + C.T @ T, C.T @ D, d, C.T @ U
    return C @ (T - r), C @ D, d, C @ U


# continuous LVLH camera path (no stop-and-go at keyframes)
LK = [  # v, target y, target z, az, el, dist
    (21.0, -3600, 0, -35, 30, 15000), (27.0, -1900, 0, -40, 30, 8200), (31.0, -1250, 0, -45, 30, 4600),
    (35.5, -100, 0, -51, 31, 560), (43.0, -90, 0, -58, 31, 480), (49.0, -38, 0, -63, 29, 300),
    (55.0, -9, 0, -67, 26, 68), (58.0, -1.5, 1.2, -70, 24, 40), (66.0, -1.0, 1.2, -82, 26, 44),
    (72.0, -8, 0, -100, 29, 210), (78.0, -20, 0, -112, 29, 320), (84.0, -1300, 0, -128, 30, 2600), (88.0, -1500, 0, -134, 30, 1800), (91.0, -1800, 0, -138, 31, 2000), (96.0, -2600, 0, -144, 35, 12000), (DUR, -3500, 0, -150, 42, 40000)]
LK = np.array(LK, float)
LKI = [PchipInterpolator(LK[:, 0], LK[:, j]) for j in (1, 2, 3, 4)]
LKD = PchipInterpolator(LK[:, 0], np.log(LK[:, 5]))
LKX = PchipInterpolator([0, 21.0, 24.0, 27.5, 31.0, 200], [-1600, -1600, -1100, -250, 0, 0])   # radial: arrival is ~3 km low


def camera(i):
    v = VV[i]; frame = "eci" if v < V_SW else "lvlh"
    C, r = lvlh(TG[i])
    if frame == "lvlh":
        T = np.array([float(LKX(v)), float(LKI[0](v)), float(LKI[1](v))])
        D = dir_lvlh(float(LKI[2](v)), float(LKI[3](v))); d = float(np.exp(LKD(v)))
        if v > 77.0:   # frame the cluster (and TOPEX until it is left behind) as they leave
            pc = [relv(k, TG[i] - T_ARR)[0] for k in range(3)]
            lo, hi = np.min(pc + [np.zeros(3)], axis=0), np.max(pc + [np.zeros(3)], axis=0)
            lo2, hi2 = np.min(pc, axis=0), np.max(pc, axis=0)
            ct = ss(84.5, 88.0, v) * (1 - ss(92.0, 97.0, v))    # close on the cluster while the plumes light, then let the wide shot take over
            c0 = (lo + hi) / 2 * (1 - ct) + (lo2 + hi2) / 2 * ct
            e0 = np.linalg.norm(hi - lo) * (1 - ct) + max(1.4 * np.linalg.norm(hi2 - lo2), 300.0) * ct
            wf = ss(77.0, 78.5, v)
            T = T + (c0 - T) * wf
            d = max(d * (1 - ct) + d * 0.0 * ct, d + (2.4 * e0 - d) * wf) if ct < 1 else 2.4 * e0
        Pc = T + D * d; f = -D; rr = unit(np.cross(f, XH)); uu = np.cross(rr, f)
        return dict(P=Pc, f=f, r=rr, u=uu, T=T, d=d, frame=frame, C=C, rT=r)
    j = max(k for k in range(len(KEYS)) if KEYS[k][0] <= v)
    j = min(j, len(KEYS) - 2)
    k0, k1 = KEYS[j], KEYS[j + 1]
    u = smoother((v - k0[0]) / (k1[0] - k0[0]))
    T0, D0, d0, U0 = to_frame(k0, frame, C, r)
    T1, D1, d1, U1 = to_frame(k1, frame, C, r)
    d = d0 ** (1 - u) * d1 ** u
    w = (d - d1) / (d0 - d1) if abs(d0 - d1) > 1e-6 * d0 else 1 - u
    T = T1 + (T0 - T1) * w
    D = slerp(D0, D1, u); U = slerp(U0, U1, u)
    Pc = T + D * d
    f = -D; rr = unit(np.cross(f, U)); uu = np.cross(rr, f)
    return dict(P=Pc, f=f, r=rr, u=uu, T=T, d=d, frame=frame, C=C, rT=r)


def project(cam, P):
    P = np.atleast_2d(P) - cam["P"]
    x, y, z = P @ cam["r"], P @ cam["u"], P @ cam["f"]
    zc = np.where(z > 1e-9, z, np.nan)
    return W / 2 + FPX * x / zc, H / 2 - FPX * y / zc, z


def behind_earth(cam, P, E, R):
    d = np.atleast_2d(P) - cam["P"]
    L = np.linalg.norm(d, axis=1); dd = d / L[:, None]
    oc = cam["P"] - E
    b = dd @ oc; c = oc @ oc - R * R
    disc = b * b - c
    th = -b - np.sqrt(np.maximum(disc, 0))
    return (disc > 0) & (th > 0) & (th < L)


# ------------------------------------------------------------------ Earth
LAND = np.load(os.path.join(HERE, "assets", "landmask.npy")).astype(np.float32) / 255.0
from scipy.ndimage import gaussian_filter
LAND = gaussian_filter(LAND, 1.2)
LH, LW = LAND.shape
OCEAN = np.array([0.050, 0.118, 0.200]); LANDC = np.array([0.200, 0.238, 0.196]); ICE = np.array([0.40, 0.44, 0.48])
ATM = np.array([0.36, 0.58, 0.92])
S_ECI = CA.T @ SUN_L
sub = CA[0]
THETA0 = math.atan2(sub[1], sub[0]) - math.radians(-38) - WE * T_ARR
EH, EW = H // 2, W // 2


_DITHER = np.random.default_rng(3).random((H, W)).astype(np.float32)


def earth_image(cam, E, sun, to_ecef):
    ys, xs = np.mgrid[0:EH, 0:EW].astype(np.float32)
    sx = (xs + 0.5) * (W / EW); sy = (ys + 0.5) * (H / EH)
    dirs = (cam["r"][None, None, :] * ((sx - W / 2) / FPX)[..., None]
            + cam["u"][None, None, :] * ((H / 2 - sy) / FPX)[..., None] + cam["f"][None, None, :])
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    oc = cam["P"] - E
    b = dirs @ oc; c = oc @ oc - RE_ * RE_
    disc = b * b - c
    hdist = np.sqrt(np.maximum(oc @ oc - b * b, 0))
    img = np.zeros((EH, EW, 4), np.float32)
    ahead = b < 0
    pix = (-b) * (W / EW) / FPX
    alpha = np.clip((RE_ - hdist) / np.maximum(pix, 1) + 0.5, 0, 1) * ahead
    hit = (alpha > 0)
    if hit.any():
        tq = -b[hit] - np.sqrt(np.maximum(disc[hit], 0))
        p = cam["P"] + dirs[hit] * tq[:, None] - E
        n = p / RE_
        ndl = n @ sun
        ne = to_ecef(n)
        lon = np.arctan2(ne[:, 1], ne[:, 0]); lat = np.arcsin(np.clip(ne[:, 2], -1, 1))
        # bilinear land mask; when a mask texel covers many screen pixels (camera close to Earth),
        # fade the coastlines out instead of showing blocky texels
        uf = (lon + np.pi) / (2 * np.pi) * LW - 0.5; vf = np.clip((np.pi / 2 - lat) / np.pi * LH - 0.5, 0, LH - 1.001)
        u0 = np.floor(uf).astype(int); v0 = np.floor(vf).astype(int); du = uf - u0; dv = vf - v0
        L = LAND.astype(np.float32)
        land = ((L[v0, u0 % LW] * (1 - du) + L[v0, (u0 + 1) % LW] * du) * (1 - dv)
                + (L[v0 + 1, u0 % LW] * (1 - du) + L[v0 + 1, (u0 + 1) % LW] * du) * dv)
        texel_px = (2 * np.pi * RE_ / LW) / np.maximum(tq, 1.0) * FPX
        fade = 1 - ss(3.0, 12.0, texel_px)
        land = (land * fade + 0.3 * (1 - fade))[:, None]
        ice = ss(1.08, 1.22, np.abs(lat))[:, None]
        base = OCEAN * (1 - land) + (LANDC * (1 - ice) + ICE * ice) * land
        lit = 0.035 + 1.05 * np.clip(ndl, 0, 1)[:, None] ** 0.85
        term = ss(-0.12, 0.10, ndl)[:, None]
        col = base * lit + np.array([0.9, 0.45, 0.2]) * 0.05 * (term * (1 - term)) * 4
        mu = np.clip(-(dirs[hit] @ n.T).diagonal() if False else -np.einsum("ij,ij->i", dirs[hit], n), 0, 1)
        rim = (1 - mu) ** 3.2
        col += ATM * rim[:, None] * (0.08 + 0.75 * ss(-0.25, 0.5, ndl))[:, None]
        col += (_DITHER[:EH, :EW][hit][:, None] - 0.5) * (2.0 / 255)   # dither: no 8-bit banding on the dark night side
        img[hit, :3] = np.clip(col, 0, 1)
        img[hit, 3] = alpha[hit]
    # atmospheric halo outside the disk
    Hs = 85e3
    halo = ahead & (hdist > RE_ * 0.995) & (hdist < RE_ + 6 * Hs)
    if halo.any():
        cp = cam["P"] + dirs[halo] * (-b[halo])[:, None] - E
        ln = ss(-0.3, 0.5, (cp / np.linalg.norm(cp, axis=1)[:, None]) @ sun)
        g = np.exp(-np.maximum(hdist[halo] - RE_, 0) / Hs) * (0.05 + 0.85 * ln)
        g = np.clip(g, 0, 1)
        a0 = img[halo, 3]
        img[halo, :3] = img[halo, :3] * a0[:, None] + ATM * g[:, None] * (1 - a0[:, None])
        img[halo, 3] = np.clip(a0 + g * (1 - a0), 0, 1)
    return img


rng = np.random.default_rng(7)
STARS = rng.normal(size=(2600, 3)); STARS /= np.linalg.norm(STARS, axis=1)[:, None]
SMAG = rng.random(2600) ** 3

# ------------------------------------------------------------------ spacecraft models
def box(c, h, R, col, tag):
    faces = []
    for ax in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3); n[ax] = sgn
            a, b = [i for i in range(3) if i != ax]
            q = []
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3); p[ax] = sgn * h[ax]; p[a] = sa * h[a]; p[b] = sb * h[b]
                q.append(p)
            faces.append((np.array(q), n, col, "solid", tag))
    return [(c + q @ R.T, R @ n, col, kind, tag) for q, n, col, kind, tag in faces]


def panel(c, ha, hb, axes, R, col, tag, na=1, nb=1):
    if na > 1 or nb > 1:
        out = []
        a, b = axes
        for ia in range(na):
            for ib in range(nb):
                off = np.zeros(3)
                off[a] = -ha + ha / na * (2 * ia + 1); off[b] = -hb + hb / nb * (2 * ib + 1)
                out += panel(c + R @ off, ha / na, hb / nb, axes, R, col, tag)
        return out
    a, b = axes
    ea = np.zeros(3); ea[a] = ha; eb = np.zeros(3); eb[b] = hb
    q = np.array([-ea - eb, ea - eb, ea + eb, -ea + eb])
    n = np.cross(ea, eb); n /= np.linalg.norm(n)
    return [(c + q @ R.T, R @ n, col, "panel", tag)]


def disk(c, rad, nax, R, col, tag, k=12):
    a, b = [i for i in range(3) if i != nax]
    q = []
    for j in range(k):
        p = np.zeros(3); p[a] = rad * math.cos(2 * math.pi * j / k); p[b] = rad * math.sin(2 * math.pi * j / k)
        q.append(p)
    n = np.zeros(3); n[nax] = 1
    return [(c + np.array(q) @ R.T, R @ n, col, "panel", tag)]


FOIL, CELL, WHITE = np.array([0.74, 0.60, 0.33]), np.array([0.12, 0.19, 0.36]), np.array([0.80, 0.82, 0.85])


_CM = np.load(os.path.join(HERE, "assets", "cad_render.npz"))   # real CAD, cached by build_cad_meshes.py


def _prep(v, f):
    v = v.astype(float); e1 = v[f[:, 1]] - v[f[:, 0]]; e2 = v[f[:, 2]] - v[f[:, 0]]
    n = np.cross(e1, e2); l = np.linalg.norm(n, axis=1); ok = l > 1e-12
    return v, f[ok], n[ok] / l[ok][:, None], ok


TM_V, TM_F, TM_N, _ok = _prep(_CM["topex_v"], _CM["topex_f"]); TM_C = _CM["topex_c"].astype(float)[_ok]
SW = {k: _prep(_CM[k + "_v"], _CM[k + "_f"])[:3] for k in ("shell", "inner", "arr_dep", "arr_sto", "arm_ext", "arm_end")}


def topex_model(R, origin=np.zeros(3)):
    V = origin + TM_V @ R.T; N = TM_N @ R.T
    return [(V[f], N[j], TM_C[j], "panel", "tgt") for j, f in enumerate(TM_F)]


def _mesh(key, R, o, col, tag="sc"):
    v, f, n = SW[key]; V = o + v @ R.T; N = n @ R.T
    return [(V[t], N[j], col, "panel", tag) for j, t in enumerate(f)]


def sc_model(R, o, accent, deployed=True):
    # SWARM Assembly.SLDASM, 'Extended' gripper (always open): body +y = gripper side (faces TOPEX)
    f = _mesh("shell", R, o, WHITE * 0.92 + accent * 0.08)
    f += _mesh("inner", R, o, FOIL * 0.9)
    f += _mesh("arr_dep" if deployed else "arr_sto", R, o, CELL)
    f += _mesh("arm_ext", R, o, accent * 0.85)
    f += _mesh("arm_end", R, o, WHITE * 0.75)   # gripperarm rotor + pads at the arm tips
    return f


def hexc(c): c = c.lstrip("#"); return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ------------------------------------------------------------------ text helpers
def spaced(s): return " ".join(s)


def nice(x):
    e = 10 ** math.floor(math.log10(x))
    for m_ in (1, 2, 5, 10):
        if m_ * e >= x * 0.999: return m_ * e
    return 10 * e


def fmt_len(m_):
    if m_ >= 1000: return f"{m_ / 1000:g} km"
    return f"{m_:g} m"


VIG = None


def vignette():
    y, x = np.mgrid[0:H // 4, 0:W // 4]
    r = np.sqrt(((x - W / 8) / (W / 8)) ** 2 + ((y - H / 8) / (H / 8)) ** 2)
    a = np.clip((r - 0.75) / 0.7, 0, 1) ** 1.6 * 0.55
    img = np.zeros((H // 4, W // 4, 4), np.float32); img[..., 3] = a
    return img


# ------------------------------------------------------------------ frame
def role_color(k, v):
    s = ss(29.4, 30.4, v)
    tgt = hexc(C_DOCK) if k == 0 else hexc(C_IMG)
    return hexc(C_NEU) * (1 - s) + tgt * s


def draw_frame(ax, i):
    global VIG
    if VIG is None: VIG = vignette()
    v = VV[i]; tg = TG[i]; trel = tg - T_ARR
    cam = camera(i)
    C, rT = cam["C"], cam["rT"]
    Cv, _ = lvlh(TVIS[i])
    th = THETA0 + WE * TVIS[i]
    Rz = np.array([[math.cos(-th), -math.sin(-th), 0], [math.sin(-th), math.cos(-th), 0], [0, 0, 1]])
    if cam["frame"] == "eci":
        E = np.zeros(3); sun = S_ECI
        to_ecef = lambda n: n @ Rz.T
        star_dirs = STARS
    else:
        E = np.array([-np.linalg.norm(rT), 0, 0]); sun = Cv @ S_ECI
        to_ecef = lambda n: (n @ Cv) @ Rz.T
        star_dirs = STARS @ Cv.T
    ax.cla(); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, H, color=BG, zorder=0))

    # stars
    z = star_dirs @ cam["f"]
    ok = z > 0.2
    sx = W / 2 + FPX * (star_dirs[ok] @ cam["r"]) / z[ok]; sy = H / 2 - FPX * (star_dirs[ok] @ cam["u"]) / z[ok]
    inb = (sx > 0) & (sx < W) & (sy > 0) & (sy < H)
    ax.scatter(sx[inb], sy[inb], s=0.4 + 5 * SMAG[ok][inb], c="#cfd8e6", alpha=0.75, linewidths=0, zorder=1)

    # earth
    img = earth_image(cam, E, sun, to_ecef)
    ax.imshow(img, extent=(0, W, H, 0), interpolation="bilinear", zorder=2)

    labels = []   # (x, y, text, color, alpha)

    if cam["frame"] == "eci":
        draw_eci(ax, cam, i, v, tg, E, labels)
    else:
        draw_lvlh(ax, cam, i, v, trel, sun, labels)

    for (x, y, t, col, a, dx, dy) in labels:
        if a <= 0.01 or not np.isfinite(x): continue
        if (x + dx > W - 620 and y + dy < 300) or (y + dy < 175): dy = abs(dy)
        if x + dx > W - 260: dx = -abs(dx)
        if x + dx > W - 620 and y + dy < 300: continue
        ax.plot([x + 4 * np.sign(dx), x + dx - 4 * np.sign(dx)], [y + 4 * np.sign(dy), y + dy], color=col, lw=0.8, alpha=0.55 * a, zorder=20)
        ax.text(x + dx + (4 if dx > 0 else -4), y + dy, t, color=col, alpha=a, fontsize=12.5, family=SANS,
                ha="left" if dx > 0 else "right", va="center", zorder=20)

    ax.imshow(VIG, extent=(0, W, H, 0), interpolation="bilinear", zorder=25)
    draw_hud(ax, cam, i, v, tg, trel)


def glow_line(ax, xs, ys, col, a, lw=1.4, z=6, ls="-"):
    ax.plot(xs, ys, color=col, lw=lw, alpha=0.9 * a, zorder=z + 0.1, solid_capstyle="round", ls=ls)


def dot(ax, x, y, col, a, s=26, z=12):
    if a <= 0 or not np.isfinite(x): return
    ax.scatter([x], [y], s=s, c=[col], alpha=a, linewidths=0, zorder=z + 0.1)


def masked_xy(cam, P, E):
    sx, sy, z = project(cam, P)
    hid = behind_earth(cam, P, E, RE_) | (z <= 0)
    sx = sx.copy(); sy = sy.copy(); sx[hid] = np.nan; sy[hid] = np.nan
    return sx, sy


def ring(r, vv, rad):
    n = unit(np.cross(r, vv)); a_ = unit(r); b_ = np.cross(n, a_)
    ang = np.linspace(0, 2 * np.pi, 720)
    return rad * (np.outer(np.cos(ang), a_) + np.outer(np.sin(ang), b_))


C0r, C0v = spC[0](0.0), spC[0](0.0, 1)


def draw_eci(ax, cam, i, v, tg, E, labels):
    rT, vT = topex_rv(tg)
    aT = np.linalg.norm(rT); aP = np.linalg.norm(C0r)
    vb1 = float(np.interp(T_B1, TG, VV))
    fast = float(ss(2500, 15000, RATE[i]))          # orbit motion too fast to read -> hide dots
    fz = 1 - ss(17.5, 20.0, v)
    tt_ = 1 - 0.55 * window(v, 0.0, 4.6, 0.7)
    sx, sy = masked_xy(cam, ring(rT, vT, aT), E)
    glow_line(ax, sx, sy, C_TGT, (0.45 * fz + 0.2 * (1 - fz)) * tt_, lw=1.0, z=4)
    ap_ = 0.55 * fz * (1 - ss(vb1 + 0.5, vb1 + 2.5, v))
    if ap_ > 0.01:
        sx, sy = masked_xy(cam, ring(C0r, C0v, aP), E); ax.plot(sx, sy, color=MUTED, lw=0.9, ls=(0, (4, 4)), alpha=ap_, zorder=4)
    a_tr = 1 - ss(19.0, 20.6, v)
    # parking-orbit track (pre-thrust)
    t0 = max(0.0, min(tg, T_B1) - 2.2 * 3600)
    if tg > 0:
        ts = np.linspace(t0, min(tg, T_B1), 200)
        sx, sy = masked_xy(cam, spC[0](ts), E)
        ax.plot(sx, sy, color=C_NEU, lw=1.2, alpha=0.5 * a_tr * (1 - ss(vb1 + 1, vb1 + 3, v)), zorder=5)
    # spiral coils so far (raw sim log) + bright last revolution
    if tg > T_B1:
        j = np.searchsorted(tt, tg)
        msk = (tt >= T_B1) & (np.arange(len(tt)) < j)
        P = CTRK[msk]
        if len(P) > 1:
            sx, sy = masked_xy(cam, P, E)
            ax.plot(sx, sy, color=C_DOCK, lw=0.45, alpha=0.30 * a_tr * (1 - ss(14.6, 16.4, v)), zorder=5)
        ts = np.linspace(max(T_B1, tg - 6200), tg, 400)
        sx, sy = masked_xy(cam, spC[0](ts), E)
        glow_line(ax, sx, sy, C_DOCK, a_tr, lw=1.8, z=6)
    # bodies
    zoomed = cam["d"] < 200e3
    for k in range(3 if zoomed else 1):
        p = spC[k](tg)
        x, y, _ = project(cam, p); x, y = x[0], y[0]
        hid = behind_earth(cam, p, E, RE_)[0]
        dot(ax, x, y, hexc(C_NEU), (0 if hid else 1) * (1 - fast), s=14 if zoomed else 22)
    x, y, _ = project(cam, rT); x, y = x[0], y[0]
    hid = behind_earth(cam, rT, E, RE_)[0]
    dot(ax, x, y, hexc(C_TGT), (0 if hid else 1) * (1 - fast), s=24)
    la = window(v, 2.5, 20.2) * (0 if hid else 1) * (1 - fast)
    labels.append((x, y, "TOPEX/Poseidon", INK, la, 26, -26))
    pc = spC[0](tg); xc, yc, _ = project(cam, pc)
    hidc = behind_earth(cam, pc, E, RE_)[0]
    if not zoomed:
        labels.append((xc[0], yc[0], "SWARM (3 spacecraft)", C_NEU, window(v, 3.5, 9.0) * (0 if hidc else 1) * (1 - fast), -26, 26))
    # thrust-on marker
    if vb1 <= v < vb1 + 2.4:
        pb = spC[0](T_B1); xb, yb, _ = project(cam, pb)
        u = (v - vb1) / 0.6
        if u < 1:
            ax.scatter([xb[0]], [yb[0]], s=(10 + 60 * u) ** 2 / 4, facecolors="none", edgecolors=C_DOCK,
                       linewidths=1.2, alpha=(1 - u), zorder=13)
        labels.append((xb[0], yb[0], "Hall thruster on", C_DOCK, window(v, vb1, vb1 + 2.4, 0.3), 30, 22))


def draw_lvlh(ax, cam, i, v, trel, sun, labels):
    d = cam["d"]
    E = np.array([-1e9, 0, 0])  # objects never behind Earth here
    # V-bar axis
    yl = np.linspace(-30000, 8000, 400)
    P = np.c_[np.zeros_like(yl), yl, np.zeros_like(yl)]
    sx, sy, z = project(cam, P)
    a_vb = 0.22 * (0.25 + 0.75 * ss(90, 250, d))
    ax.plot(sx, sy, color=C_TGT, lw=1.0, alpha=a_vb + 0.25 * (1 - ss(21.0, 23.0, v)), zorder=4)
    # hold points
    pts = []
    if v < 32: pts += [(-1000, 0), (-1300, 1), (-1600, 2)]
    if 40 < v < 53: pts += [(y, 0) for y in (-200, -120, -70, -45, -30)]
    for yh, k in pts:
        x, y, _ = project(cam, np.array([0, yh, 0]))
        ax.scatter(x, y, s=22, facecolors="none", edgecolors=MUTED, linewidths=0.9, alpha=0.8, zorder=5)
    # spheres (KOZ 25 m, approach 200 m)
    for R, col, a0, ls, lab in ((22.0, C_RED, 0.55, "-", "22 m keep-out"), (200.0, MUTED, 0.45, (0, (3, 4)), "200 m")):
        vis = a0 * (1 - ss(R * 40, R * 80, d)) * (1 - ss(57.5, 59, v) * (R == 200)) * (1 - 0.5 * ss(57.5, 59, v) * (R == 25))
        if vis < 0.01: continue
        cx, cy, cz = project(cam, np.zeros(3))
        rp = FPX * R / max(cz[0], 1e-6)
        circ = plt.Circle((cx[0], cy[0]), rp, fill=False, ec=col, lw=1.0, alpha=vis, ls=ls, zorder=5)
        ax.add_patch(circ)
        if R == 22:
            pass
        labels.append((cx[0] + rp * 0.71, cy[0] - rp * 0.71, lab, col, vis / a0 * 0.9, 16, -16))
    # trails
    wins = 6800.0
    for k in range(3):
        col = role_color(k, v)
        t0 = max(0.0, trel - wins)
        ts = np.linspace(t0, trel, 420)
        Pk = relv(k, ts)
        far = np.linalg.norm(Pk, axis=1) > 8.5e3
        sx, sy, z = project(cam, Pk)
        sx[far] = np.nan
        seg = np.stack([np.c_[sx[:-1], sy[:-1]], np.c_[sx[1:], sy[1:]]], axis=1)
        good = np.isfinite(seg).all(axis=(1, 2))
        al = np.linspace(0.0, 1.0, len(seg)) ** 1.6
        if k == 0 and P_CON <= trel < P_UND: al = al * (1 - ss(P_CON, P_CON + 400, trel))
        if k > 0: al = al * (0.3 + 0.7 * ss(60, 160, d))
        cols = np.c_[np.tile(col, (len(seg), 1)), 0.85 * al]
        ax.add_collection(LineCollection(seg[good], colors=cols[good], linewidths=1.6, zorder=6))
        ax.add_collection(LineCollection(seg[good], colors=np.c_[np.tile(col, (good.sum(), 1)), 0.10 * al[good]],
                                         linewidths=5, zorder=5.9))
    # models
    faces = []
    Rt = Ry(PHI[i])
    faces += topex_model(Rt)
    sc_pos = []
    for k in range(3):
        p = rel(k, trel)
        if k == 0 and P_CON <= trel < P_UND:
            p = np.array([0, -DOCK_R, 0])
        if k == 0:
            Rk = Ry(ROLL[i]) @ Ry(DOCK_CLOCK)
        else:
            by = unit(-p) if np.linalg.norm(p) > 1 else np.array([0, 1.0, 0])
            bx = unit(XH - np.dot(XH, by) * by); bz = np.cross(bx, by)
            Rk = np.c_[bx, by, bz]
        sc_pos.append(p)
        faces += sc_model(Rk, p, role_color(k, v), deployed=trel >= 0)
    draw_faces(ax, cam, faces, sun)
    # Hall thruster plumes on the way out
    pa = ss(P_END, P_END + 300, trel)
    if pa > 0.01:
        for k in range(3):
            p = sc_pos[k]
            x0, y0, z0 = project(cam, p); x1, y1, _ = project(cam, p - DEP_DIR * 50.0)
            dx, dy = x1[0] - x0[0], y1[0] - y0[0]; L = math.hypot(dx, dy)
            if not np.isfinite(L) or L < 1e-6: continue
            Lp = float(np.clip(FPX * 2.5 / max(z0[0], 1e-6), 26, 160)); dx, dy = dx / L * Lp, dy / L * Lp
            for frac, lw, al in ((1.0, 7, 0.10), (0.75, 3.5, 0.22), (0.45, 1.6, 0.55)):
                ax.plot([x0[0], x0[0] + dx * frac], [y0[0], y0[0] + dy * frac], color="#8fb4ff", lw=lw,
                        alpha=al * pa, solid_capstyle="round", zorder=11)
    # dots + labels when small
    x, y, z = project(cam, np.zeros(3))
    pxs = FPX * 10.0 / max(z[0], 1e-6)
    dot(ax, x[0], y[0], hexc(C_TGT), float(np.clip((14 - pxs) / 8, 0, 1)), s=22)
    lt = "TOPEX/POSEIDON"
    off = min(max(26, pxs * 0.9), 70)
    labels.append((x[0], y[0], "TOPEX/Poseidon", INK, window(v, 21.3, 89.0) * float(np.clip((150 - pxs) / 70, 0, 1)), off, -off))
    for k in range(3):
        p = sc_pos[k]
        x, y, z = project(cam, p)
        pxs = FPX * 2.0 / max(z[0], 1e-6)
        col = role_color(k, v)
        dot(ax, x[0], y[0], col, float(np.clip((12 - pxs) / 7, 0, 1)), s=14)
        role = "" if v < 29.6 else (" docker" if k == 0 else " imager")
        txt = f"SC{k + 1}" + role
        a = window(v, 21.6, 85.0)
        if 55 < v < 72 and k > 0: a *= 0.0 if d < 120 else 1
        if 27.4 < v < 31.2:
            cost = np.linalg.norm(rel(k, T_ARR * 0 + P_HOLD)) / 1000.0
            txt = f"SC{k + 1}, {cost:.2f} km away" if v < 29.6 else txt
        o = min(max(30, pxs * 0.75), 110)
        dx = -o if (k == 2 and v < 35) or (k == 0 and v > 50) else o
        dy = o if k == 1 else -o
        labels.append((x[0], y[0], txt, "#%02x%02x%02x" % tuple((col * 255).astype(int)), a, dx, dy))
    # arrival burn
    if 21.0 <= v < 23.6:
        u = (v - 21.0) / 0.6
        for k in range(3):
            x, y, _ = project(cam, rel(k, max(trel, 0)))
            if u < 1:
                ax.scatter(x, y, s=(10 + 60 * u) ** 2 / 4, facecolors="none", edgecolors=C_NEU, linewidths=1.2,
                           alpha=(1 - u), zorder=13)
        x, y, _ = project(cam, rel(0, max(trel, 0)))
        labels.append((x[0], y[0], "Hall thruster off", C_NEU, window(v, 21.0, 23.6, 0.3), -34, 30))
    # burns (prox)
    if v < 84:
        for _, b in bu[(bu.dv > 0.02)].iterrows():
            vb = float(np.interp(b.t + T_ARR, TG, VV))
            if vb <= v < vb + 0.55:
                u = (v - vb) / 0.55
                p = rel(int(b.craft), b.t)
                x, y, _ = project(cam, p)
                col = role_color(int(b.craft), v)
                ax.scatter(x, y, s=(8 + 34 * u) ** 2 / 4, facecolors="none", edgecolors=[col], linewidths=1.1,
                           alpha=(1 - u) * 0.9, zorder=13)


def draw_faces(ax, cam, faces, sun):
    # all models are triangle meshes: shade and depth-sort them in one batch (painter's order, far to near)
    faces = [x for x in faces if len(x[0]) == 3]
    if not faces: return
    Q = np.array([x[0] for x in faces]); N = np.array([x[1] for x in faces]); C = np.array([x[2] for x in faces])
    tgt = np.array([x[4] == "tgt" for x in faces])
    c = Q.mean(axis=1); view = cam["P"] - c
    sx, sy, z = project(cam, Q.reshape(-1, 3)); sx, sy, z = sx.reshape(-1, 3), sy.reshape(-1, 3), z.reshape(-1, 3)
    ok = (z > 0.05).all(axis=1)
    nv = np.where((np.einsum("ij,ij->i", N, view) > 0)[:, None], N, -N)      # double-sided: normal toward camera
    ndl = nv @ sun
    es = np.maximum(-nv[:, 0], 0) * 0.22 if cam["frame"] == "lvlh" else 0
    sh = 0.32 + 0.78 * np.maximum(ndl, 0) + es
    fc = np.clip(C * sh[:, None], 0, 1)
    ec = np.where(tgt[:, None], fc, np.clip(fc * 1.12 + 0.03, 0, 1))
    dep = (c - cam["P"]) @ cam["f"]
    o = np.argsort(dep[ok])[::-1]
    P = np.stack([sx, sy], axis=2)[ok][o]
    ax.add_collection(PolyCollection(P, facecolors=fc[ok][o], edgecolors=ec[ok][o], linewidths=0.3, zorder=10))


# ------------------------------------------------------------------ HUD
def fmt_w(w):
    if w >= 0.1: return f"{w:.2f} rad/s"
    if w >= 1e-4: return f"{w:.4f} rad/s"
    return "< 0.0001 rad/s"


def phase_at(v):
    for name, a, b in PHASES:
        if a <= v < b: return name
    return PHASES[-1][0]


def draw_hud(ax, cam, i, v, tg, trel):
    Z = 30
    hud_a = ss(3.6, 4.6, v) * (1 - ss(DUR - 5.2, DUR - 4.4, v))
    t_title = window(v, 0.4, 4.6, 0.7)
    if t_title > 0:
        ax.text(80, H / 2, "Stopping TOPEX/Poseidon's spin", color=INK, alpha=t_title, fontsize=34, family=SANS_L, va="center", zorder=Z)
    for name, a, b in PHASES:
        al = (window(v, a, b, 0.35) if a > 0 else window(v, -1, b, 0.35)) * hud_a
        if al > 0.01:
            ax.text(80, 70, name, color=INK, alpha=al, fontsize=20, family=SANS, va="center", zorder=Z)
    for a, b, txt in CAPS:
        al = window(v, a, b, 0.4)
        if al > 0:
            ax.text(80, H - 70, txt, color=INK, alpha=al * 0.9, fontsize=17, family=SANS_L, va="center", zorder=Z)
    # one telemetry line
    met = int(tg); dd, hh, mm = met // 86400, (met % 86400) // 3600, (met % 3600) // 60
    if tg < T_ARR:
        tel = f"T+{dd} d {hh:02d}:{mm:02d}   altitude {(np.linalg.norm(spC[0](tg)) - RE_) / 1000:,.0f} km"
    else:
        r0 = DOCK_R if P_CON <= trel < P_UND else np.linalg.norm(rel(0, trel))
        rs = f"{r0 / 1000:.2f} km" if r0 >= 1000 else f"{r0:.1f} m"
        tel = f"T+{dd} d {hh:02d}:{mm:02d}   docker {rs}" if v < 49 else f"T+{dd} d {hh:02d}:{mm:02d}   TOPEX spin {fmt_w(omega(trel))}"
    ax.text(W - 80, 70, tel, color=MUTED, alpha=hud_a, fontsize=13, family=MONO, ha="right", va="center", zorder=Z)
    # scale bar
    mpp = cam["d"] / FPX
    L = nice(150 * mpp); Lp = L / mpp
    xs1 = W - 80; xs0 = xs1 - Lp; ys = H - 70
    sa = hud_a * (1 - window(v, DUR - 5.0, DUR + 5, 0.5))
    ax.plot([xs0, xs1], [ys, ys], color=MUTED, lw=1.0, alpha=sa, zorder=Z)
    ax.text(xs0 - 10, ys, fmt_len(L), color=MUTED, alpha=sa, fontsize=12, family=MONO, ha="right", va="center", zorder=Z)
    # spin rate vs time, plain log plot
    sa2 = window(v, 55.5, 74.5, 0.6) * hud_a
    if sa2 > 0.01:
        gx0, gx1, gy0, gy1 = W - 420, W - 80, 140, 300
        ts = np.linspace(P_CON - 900, P_CON + T_DET + 1500, 400)
        w = np.array([omega(t) for t in ts])
        X2 = lambda t: gx0 + (gx1 - gx0) * (t - ts[0]) / (ts[-1] - ts[0])
        LO, HI = math.log10(3e-4), math.log10(1.0)
        Y2 = lambda o: gy1 - (gy1 - gy0) * (math.log10(max(o, 3e-4)) - LO) / (HI - LO)
        ax.plot([gx0, gx0, gx1], [gy0, gy1, gy1], color=MUTED, lw=0.8, alpha=sa2, zorder=Z)
        ax.plot([gx0, gx1], [Y2(W_BAND)] * 2, color=MUTED, lw=0.8, ls=(0, (3, 3)), alpha=sa2, zorder=Z)
        ax.text(gx1, Y2(W_BAND) - 9, "target 0.00175", color=MUTED, alpha=sa2, fontsize=10, family=MONO, ha="right", va="center", zorder=Z)
        msk = ts <= trel
        if msk.any():
            ax.plot([X2(t) for t in ts[msk]], [Y2(o) for o in w[msk]], color=INK, lw=1.3, alpha=sa2, zorder=Z)
        for o_, lab_ in ((W0, "0.66"), (0.01, "0.01"), (1e-3, "0.001")):
            ax.text(gx0 - 8, Y2(o_), lab_, color=MUTED, alpha=sa2, fontsize=10, family=MONO, ha="right", va="center", zorder=Z)
        ax.text(gx0, gy0 - 14, "TOPEX spin [rad/s]", color=MUTED, alpha=sa2, fontsize=10, family=MONO, va="center", zorder=Z)
        ax.text(gx1, gy1 + 14, f"{(ts[-1] - ts[0]) / 3600:.1f} h", color=MUTED, alpha=sa2, fontsize=10, family=MONO, ha="right", va="center", zorder=Z)
    # end card: plain numbers
    ea = ss(DUR - 5.0, DUR - 4.0, v)
    if ea > 0:
        ax.add_patch(plt.Rectangle((0, 0), W, H, color=BG, alpha=0.75 * ea, zorder=Z - 1))
        lines = [("Climb, 500 to 1336 km", f"{SP_DAYS:.0f} days, {DV_SP:.0f} m/s, {XE_USED:.1f} of 16 kg xenon"),
                 ("Docker hydrazine", f"{MP_K[0]:.1f} of 18 kg"),
                 ("Imager hydrazine", f"{MP_K[1]:.2f} and {MP_K[2]:.2f} of 18 kg"),
                 ("Despin", f"0.66 to 0.00175 rad/s in {T_DET / 3600:.1f} h")]
        for j, (k_, val) in enumerate(lines):
            y = 400 + j * 44
            ax.text(120, y, k_, color=MUTED, alpha=ea, fontsize=17, family=SANS_L, va="center", zorder=Z)
            ax.text(560, y, val, color=INK, alpha=ea, fontsize=17, family=MONO, va="center", zorder=Z)


# ------------------------------------------------------------------ driver
def make_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    return fig, ax


def render_chunk(args):
    a, b, out = args
    fig, ax = make_fig()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(a, b):
        draw_frame(ax, i)
        fig.canvas.draw()
        p.stdin.write(fig.canvas.buffer_rgba().tobytes())
        if (i - a) % 150 == 0: print(f"[{a}-{b}] {i}", flush=True)
    p.stdin.close(); p.wait()
    plt.close(fig)
    return out


def _opt(name, default):
    a = sys.argv
    return float(a[a.index(name) + 1]) if name in a else default


def _phase(q):
    for name, a, b in PHASES:
        if name.lower().startswith(q.lower()): return name, a, b
    raise SystemExit("phases: " + ", ".join(n for n, _, _ in PHASES))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "render"
    scale = _opt("--scale", 1.0)
    out = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
    if mode == "preview":   # preview T [T ...] [--scale s]  -> out/preview/vTTT.T.png
        os.makedirs(os.path.join(out, "preview"), exist_ok=True)
        fig, ax = make_fig()
        for a_ in sys.argv[2:]:
            if a_.startswith("--") or a_ == str(scale): continue
            j = min(int(float(a_) * FPS), NF - 1)
            draw_frame(ax, j)
            fig.savefig(os.path.join(out, "preview", f"v{float(a_):05.1f}{os.environ.get('TAG', '')}.png"), dpi=100 * scale)
            print("preview", a_)
    elif mode == "clip":    # clip PHASE [--fps 10] [--scale 0.5]  -> out/clip_<phase>.mp4
        name, a, b = _phase(sys.argv[2])
        fps = _opt("--fps", 10)
        fig, ax = make_fig(); fig.set_dpi(100 * scale)
        w_, h_ = fig.canvas.get_width_height()
        fn = os.path.join(out, "clip_" + name.split()[0].lower() + ".mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w_}x{h_}",
                              "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", fn],
                             stdin=subprocess.PIPE)
        for j in range(int(a * FPS), min(int(b * FPS), NF), max(1, int(FPS / fps))):
            draw_frame(ax, j); fig.canvas.draw()
            p.stdin.write(fig.canvas.buffer_rgba().tobytes())
        p.stdin.close(); p.wait()
        print("wrote", fn)
    else:
        from multiprocessing import Pool
        nw = int(os.environ.get("NW", os.cpu_count() or 2))
        edges = np.linspace(0, NF, nw * 2 + 1).astype(int)
        jobs = [(edges[j], edges[j + 1], os.path.join(out, f"chunk_{j:02d}.mp4")) for j in range(len(edges) - 1)]
        with Pool(nw) as pool:
            outs = pool.map(render_chunk, jobs)
        with open(os.path.join(out, "chunks.txt"), "w") as f:
            for o in outs: f.write(f"file '{o}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(out, "chunks.txt"),
                        "-c", "copy", "-movflags", "+faststart", os.path.join(out, "SWARM_mission_v2_full.mp4")], check=True)
        for o in outs: os.remove(o)
        print("wrote SWARM_mission_v2_full.mp4", NF, "frames")
