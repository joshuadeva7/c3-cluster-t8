"""The v2 mission video's look, continued live: the same LVLH camera path, FOV, background, stars, V-bar line, hold
markers, keep-out / 200 m rings, trails, far-range dots, burn rings and vignette (all from assets/v2_handoff.npz, which
v2/render_cutscene.py writes from the exact frame the cutscene ends on). Room frame = v2 LVLH (x radial, y along-track,
z cross-track), metres, TOPEX at the origin. Canvas images are BGR uint8."""
import math, os, numpy as np, cv2
from scipy.interpolate import PchipInterpolator
HERE = os.path.dirname(os.path.abspath(__file__))
H5 = np.load(os.path.join(HERE, "assets", "v2_handoff.npz"))
LK = H5["LK"]; CUT_EL = float(H5["CUT_EL"]); V0 = float(H5["v"]); TREL0 = float(H5["trel"])
LKI = [PchipInterpolator(LK[:, 0], LK[:, j]) for j in (1, 2, 3, 4)]; LKD = PchipInterpolator(LK[:, 0], np.log(LK[:, 5]))
LKX = PchipInterpolator([0, 21.0, 24.0, 27.5, 31.0, 200], [-1600, -1600, -1100, -250, 0, 0])
REL_T = H5["REL_T"]; REL = [H5["REL0"], H5["REL1"], H5["REL2"]]
STARS, SMAG = H5["stars"], H5["smag"]; SUN = H5["sun"] / np.linalg.norm(H5["sun"])
def bgr(c): c = np.asarray(c, float); return tuple(int(round(x * 255)) for x in c[::-1])
BG = bgr(H5["BG"]); C_NEU, C_DOCK, C_IMG = H5["C_NEU"], H5["C_DOCK"], H5["C_IMG"]; C_TGT, C_RED, MUTED = H5["C_TGT"], H5["C_RED"], H5["MUTED"]
BURN_T, BURN_K = H5["burn_t"], H5["burn_k"]

def ss(a, b, x): u = np.clip((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u)
def smoother(u): u = np.clip(u, 0, 1); return u * u * u * (u * (u * 6 - 15) + 10)
def relv(k, ts):
    ts = np.atleast_1d(np.asarray(ts, float)); return np.array([np.interp(ts, REL_T, REL[k][:, j]) for j in range(3)]).T
def rel(k, t): return relv(k, t)[0]

def cam(v, el_override=True):
    """v2 camera at story time v -> (R camera->world in OpenCV axes, position, distance d)."""
    v = float(np.clip(v, 21.0, LK[-1, 0])); el = float(LKI[3](v))
    if el_override: el = el * (1 - smoother((v - 22.5) / (27.0 - 22.5))) + CUT_EL * smoother((v - 22.5) / (27.0 - 22.5))
    az, el = math.radians(float(LKI[2](v))), math.radians(el)
    D = np.array([math.sin(el), -math.cos(el) * math.cos(az), math.cos(el) * math.sin(az)])
    T = np.array([float(LKX(v)), float(LKI[0](v)), float(LKI[1](v))]); d = float(np.exp(LKD(v)))
    f = -D; r = np.cross(f, [1.0, 0, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.c_[r, -u, f], T + D * d, d

class Look:
    def __init__(self, W, H):
        self.W, self.H = W, H; self.F = (W / 2) / math.tan(math.radians(34.0) / 2); self.lw = (100 * W / 1920) / 72.0   # matplotlib pt -> px
        y, x = np.mgrid[0:H, 0:W]; r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        self.vig = (1 - np.clip((r - 0.75) / 0.7, 0, 1) ** 1.6 * 0.55).astype(np.float32)[..., None]
    def K(self, cx, cy=None): return np.array([[self.F, 0, cx], [0, self.F, self.H / 2 if cy is None else cy], [0, 0, 1.0]])
    def project(self, R, P, X, cx):
        Xc = (np.atleast_2d(X) - P) @ R; z = Xc[:, 2]; zc = np.where(z > 1e-6, z, np.nan)
        return np.c_[cx + self.F * Xc[:, 0] / zc, self.H / 2 + self.F * Xc[:, 1] / zc], z
    def _blend_poly(self, img, pts, col, a, lw):
        pts = np.clip(pts[np.isfinite(pts).all(1)], -2e4, 2e4)
        if a <= 0.005 or len(pts) < 2: return
        ov = img.copy(); cv2.polylines(ov, [np.round(pts * 16).astype(np.int32)], False, col, max(1, int(round(lw))), cv2.LINE_AA, shift=4)
        cv2.addWeighted(ov, a, img, 1 - a, 0, img)
    def _dot(self, img, xy, col, a, s):
        if a <= 0.01 or not np.isfinite(xy).all(): return
        rad = max(1.0, 0.5 * math.sqrt(s) * self.lw); ov = img.copy()
        cv2.circle(ov, (int(round(xy[0] * 16)), int(round(xy[1] * 16))), int(round(rad * 16)), col, -1, cv2.LINE_AA, shift=4); cv2.addWeighted(ov, a, img, 1 - a, 0, img)
    def background(self, R, P, cx, width):
        img = np.empty((self.H, width, 3), np.uint8); img[:] = BG
        z = STARS @ R[:, 2]; ok = z > 0.2; S = STARS[ok] @ R
        uv = np.c_[cx + self.F * S[:, 0] / S[:, 2], self.H / 2 + self.F * S[:, 1] / S[:, 2]]; m = SMAG[ok]
        inb = (uv[:, 0] > 0) & (uv[:, 0] < width - 1) & (uv[:, 1] > 0) & (uv[:, 1] < self.H - 1)
        for (x, y), mg in zip(uv[inb], m[inb]):
            rad = 0.5 * math.sqrt(0.4 + 5 * mg) * self.lw; a = 0.75 * min(1.0, rad * rad * 3.14 / 1.0) if rad < 0.56 else 0.75
            xi, yi = int(x), int(y); img[yi, xi] = np.clip(img[yi, xi] * (1 - a) + np.array([230, 216, 207]) * a, 0, 255)
            if rad > 0.9: cv2.circle(img, (xi, yi), int(round(rad)), (230, 216, 207), -1, cv2.LINE_AA)
        return img
    def overlays(self, img, R, P, d, cx, v, t_k, roles_s, trail_end_alpha=1.0, burn_flash=()):
        """vector layers in v2's order. t_k: per-craft mission clocks (s, relative to TOPEX arrival); roles_s: 0 neutral -> 1 docker/imager colours."""
        # V-bar axis
        yl = np.linspace(-30000, 8000, 400); pts, z = self.project(R, P, np.c_[np.zeros_like(yl), yl, np.zeros_like(yl)], cx)
        okp = np.isfinite(pts).all(1) & (z > 0)
        self._blend_poly(img, pts[okp], bgr(C_TGT), 0.22 * (0.25 + 0.75 * ss(90, 250, d)), 1.0 * self.lw)
        # hold points (formation holds 1.0 / 1.3 / 1.6 km, then the glideslope holds)
        hold = [-1000, -1300, -1600] if v < 32 else ([-200, -120, -70, -45, -30] if 40 < v < 53 else [])
        for yh in hold:
            xy, zz = self.project(R, P, np.array([0, yh, 0.0]), cx)
            if zz[0] > 0 and np.isfinite(xy).all():
                ov = img.copy(); cv2.circle(ov, (int(xy[0, 0]), int(xy[0, 1])), max(2, int(0.5 * math.sqrt(22) * self.lw)), bgr(MUTED), 1, cv2.LINE_AA); cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
        # keep-out (22 m) and 200 m spheres
        cxy, cz = self.project(R, P, np.zeros(3), cx)
        for Rr, col, a0, dash in ((22.0, C_RED, 0.55, False), (200.0, MUTED, 0.45, True)):
            vis = a0 * (1 - ss(Rr * 40, Rr * 80, d)) * (1 - ss(57.5, 59, v) * (Rr == 200))
            if vis < 0.01 or cz[0] <= 0: continue
            rp = self.F * Rr / cz[0]; th = np.linspace(0, 2 * np.pi, 361); ring = np.c_[cxy[0, 0] + rp * np.cos(th), cxy[0, 1] + rp * np.sin(th)]
            if dash:
                seg = int(max(2, rp * 2 * np.pi / (7 * self.lw * 1.0) ))
                for j in range(0, seg, 2):
                    a_, b_ = int(j * 360 / seg), int((j + 1) * 360 / seg) + 1; self._blend_poly(img, ring[a_:b_], bgr(col), vis, self.lw)
            else: self._blend_poly(img, ring, bgr(col), vis, self.lw)
        # trails (last 6800 s of each craft), glow + line, alpha ramps up toward the craft
        for k in range(3):
            col = role_col(k, roles_s); t1 = t_k[k]; ts = np.linspace(max(0.0, t1 - 6800.0), t1, 420)   # v2: trails start at TOPEX arrival
            Pk = relv(k, ts); far = np.linalg.norm(Pk, axis=1) > 8.5e3; xy, z = self.project(R, P, Pk, cx); xy[far | (z <= 0)] = np.nan
            al = np.linspace(0, 1, len(xy)) ** 1.6 * trail_end_alpha
            if k > 0: al = al * (0.3 + 0.7 * ss(60, 160, d))
            for j0 in range(0, len(xy) - 1, 20):   # piecewise alpha (21-point pieces)
                seg = xy[j0:j0 + 21]; okk = np.isfinite(seg).all(1)
                if okk.sum() < 2: continue
                aa = float(al[min(j0 + 10, len(al) - 1)])
                self._blend_poly(img, seg[okk], bgr(col), 0.10 * aa, 5 * self.lw); self._blend_poly(img, seg[okk], bgr(col), 0.85 * aa, 1.6 * self.lw)
        # dots when small
        pxs = self.F * 10.0 / max(cz[0], 1e-6); self._dot(img, cxy[0], bgr(C_TGT), float(np.clip((14 - pxs) / 8, 0, 1)), 22)
        for k in range(3):
            xy, z = self.project(R, P, rel(k, t_k[k]), cx); pxs = self.F * 2.0 / max(z[0], 1e-6)
            if z[0] > 0: self._dot(img, xy[0], bgr(role_col(k, roles_s)), float(np.clip((12 - pxs) / 7, 0, 1)), 14)
        for (k, u) in burn_flash:   # expanding burn rings
            xy, z = self.project(R, P, rel(k, t_k[k]), cx)
            if z[0] > 0 and np.isfinite(xy).all():
                rr = 0.5 * (8 + 34 * u) * self.lw; ov = img.copy(); cv2.circle(ov, (int(xy[0, 0]), int(xy[0, 1])), max(1, int(rr)), bgr(role_col(k, roles_s)), 1, cv2.LINE_AA)
                cv2.addWeighted(ov, (1 - u) * 0.9, img, 1 - (1 - u) * 0.9, 0, img)
        return img
    def vignette(self, img, x0=0): w = img.shape[1]; return (img * self.vig[:, x0:x0 + w]).astype(np.uint8)

def role_col(k, s):
    tgt = C_DOCK if k == 0 else C_IMG; return C_NEU * (1 - s) + tgt * s
