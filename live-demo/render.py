"""Offscreen space renderer for the live-demo benchmarks.

TOPEX mesh = the exact mesh the v2 animation uses (v2/assets/cad_render.npz: topex_v/f/c, metres,
TOPEX body frame: +y spin axis, +z solar-array side, origin = bus box centre).
Camera convention: OpenCV (x right, y down, z forward). Pose = (R, t) mapping TOPEX body -> camera.
Lighting: one directional "sun", tiny ambient, black background.
"""
import os
import sys as _sys
if _sys.platform.startswith("linux"): os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # macOS: pyglet offscreen
import numpy as np, trimesh
BACKEND = os.environ.get("LIVE_DEMO_RENDER", "pyrender" if _sys.platform.startswith("linux") else "gl")   # macOS: moderngl/CGL (pyrender needs a pyglet screen)
if BACKEND == "pyrender": import pyrender
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "assets", "cad_render.npz")   # copy of v2/assets/cad_render.npz so live-demo/ is self-contained

def topex_mesh():
    d = np.load(NPZ)
    m = trimesh.Trimesh(d["topex_v"], d["topex_f"], process=False)
    m.visual.face_colors = np.c_[(d["topex_c"] * 255).astype(np.uint8), np.full(len(d["topex_f"]), 255, np.uint8)]
    return m

def K_for(res, hfov_deg=30.0):
    f = 0.5 * res / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])

CV2GL = np.diag([1, -1, -1, 1.0])

class SpaceRenderer:
    def __new__(cls, *args, **kw):
        if BACKEND == "gl" and cls is SpaceRenderer: return object.__new__(GLSpaceRenderer)
        return object.__new__(cls)
    def __init__(self, res=512, hfov_deg=30.0, mesh=None, sun_dir_cam=(-0.5, -0.3, 0.8), share=None, smooth=False):
        self.res = res; self.K = K_for(res, hfov_deg)
        self.mesh = mesh if mesh is not None else topex_mesh()
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.03, 0.03, 0.03])
        self.node = self.scene.add(pyrender.Mesh.from_trimesh(self.mesh, smooth=smooth))
        cam = pyrender.IntrinsicsCamera(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], znear=0.5, zfar=500)
        self.scene.add(cam, pose=np.eye(4))
        # sun fixed in camera frame for the bench (direction light travels, OpenGL frame)
        d = np.asarray(sun_dir_cam, float); d /= np.linalg.norm(d)
        z = -d  # pyrender lights shine along -z of their node
        x = np.cross([0, 1, 0], z); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1, 0, 0]); y = np.cross(z, x)
        L = np.eye(4); L[:3, :3] = np.c_[x, y, z]
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=6.0), pose=L)
        self.r = share.r if share is not None else pyrender.OffscreenRenderer(res, res)

    def render(self, R, t, depth=True):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        self.scene.set_pose(self.node, CV2GL @ T)   # camera at origin looking -z (GL) == +z (CV)
        col, dep = self.r.render(self.scene)
        return col, dep

class GLSpaceRenderer(SpaceRenderer):
    """same interface on the moderngl renderer (standalone EGL/CGL context, shared via share=)."""
    def __init__(self, res=512, hfov_deg=30.0, mesh=None, sun_dir_cam=(-0.5, -0.3, 0.8), share=None, smooth=False):
        import gl_render
        self.res = res; self.K = K_for(res, hfov_deg); self.mesh = mesh if mesh is not None else topex_mesh()
        self.sc = gl_render.GLScene(share.sc.ctx if share is not None else None); self.sc.add(self.mesh)
        d = np.asarray(sun_dir_cam, float); self.sun_cam = -d / np.linalg.norm(d)   # toward the sun, camera frame
        self.r = self
    def render(self, R, t, depth=True, sun_cam=None):
        s = self.sun_cam if sun_cam is None else np.asarray(sun_cam, float)
        return self.sc.render(self.K, self.res, self.res, R.T, -R.T @ t, R.T @ s, znear=0.5, zfar=500)

def random_pose(rng, rmin=25.0, rmax=45.0, hfov_deg=30.0, margin=0.6):
    R = Rot.random(random_state=rng).as_matrix()
    z = rng.uniform(rmin, rmax); lim = np.tan(np.radians(hfov_deg) / 2) * z * margin * 0.5
    t = np.array([rng.uniform(-lim, lim), rng.uniform(-lim, lim), z])
    return R, t

def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def degrade(col, rng, noise=3.0, blur=0.6):
    """sensor model for 'live' frames: grey, slight optical blur, read noise, 8-bit."""
    import cv2
    g = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if blur > 0: g = cv2.GaussianBlur(g, (0, 0), blur)
    g += rng.normal(0, noise, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)

def perturb(R, t, rng, rot_deg, trans_frac):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    Rp = Rot.from_rotvec(ax * np.radians(rot_deg)).as_matrix() @ R
    tp = t + rng.normal(size=3) * trans_frac * t[2] * np.array([0.5, 0.5, 1.0])
    return Rp, tp
