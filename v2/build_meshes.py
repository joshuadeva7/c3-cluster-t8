"""Build the render meshes once -> assets/*.npz (rerun only if the CAD changes).

TOPEX: team8-onedrive/Vehicle and System Design Review/CAD/TOPEX-Poseidon.glb (Draco, needs DracoPy).
  Scaled so the body length along the spin axis is 5.5 m (TOPEX/Poseidon press kit, Aug 1992:
  "Length: 18 feet (5.5m)"). Axes remapped to the animation body frame:
  anim +y = spin axis (docker approaches from -y), anim +z = solar array direction.
SWARM arrays: Sparkwing 36V 750x820mm 2-panel deployed STEP (vendor CAD), via cascadio/OpenCASCADE.
"""
import os, numpy as np, trimesh, fast_simplification, cascadio

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.environ.get("CAD_DIR", "/workspace/courses/ae-4322/team8-onedrive/Vehicle and System Design Review/CAD")
TOPEX_LEN = 5.5   # m


def decimate(v, f, target):
    if len(f) <= target: return v, f
    v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - target / len(f))
    return v2, f2


def topex(target=1800):
    s = trimesh.load(os.path.join(CAD, "TOPEX-Poseidon.glb"))
    V, F, C = [], [], []
    n0 = 0
    for node in s.graph.nodes_geometry:
        T, g = s.graph[node]
        m = s.geometry[g].copy(); m.apply_transform(T)
        col = np.array(m.visual.material.baseColorFactor[:3], float) / 255.0
        share = max(40, int(target * len(m.faces) / 20926))
        v, f = decimate(m.vertices, m.faces, share)
        V.append(v); F.append(f + n0); C.append(np.tile(col, (len(f), 1))); n0 += len(v)
    V = np.vstack(V); F = np.vstack(F); C = np.vstack(C)
    lo, hi = V.min(0), V.max(0)
    s_ = TOPEX_LEN / (hi[2] - lo[2])
    ctr = np.array([0.5 * (lo[0] + hi[0]) * 0 + 3.0, 1.0, 0.5 * (lo[2] + hi[2])])   # bus centre in x/y (bus part bounds), mid-length in z
    g = (V - ctr) * s_
    A = np.c_[-g[:, 1], g[:, 2], -g[:, 0]]          # anim = (-gy, gz, -gx), right-handed
    return A, F, C, s_


def sparkwing(target=300):
    fn = os.path.join(CAD, "Sparkwing 36V 750x820mm 2Panel SA Deployed STEP.stp")
    out = "/tmp/sparkwing_dep.glb"
    if not os.path.exists(out): cascadio.step_to_glb(fn, out, tol_linear=0.5, tol_angular=0.5)
    m = trimesh.load(out).to_geometry()
    v, f = decimate(m.vertices, m.faces, target)
    return v, f


if __name__ == "__main__":
    A, F, C, s_ = topex()
    np.savez_compressed(os.path.join(HERE, "assets", "topex_mesh.npz"), v=A.astype(np.float32), f=F.astype(np.int32), c=C.astype(np.float32))
    print("topex", len(F), "tris, scale", round(s_, 5), "m/unit, extents", (A.max(0) - A.min(0)).round(2), "ymin", A[:, 1].min().round(2))
    v, f = sparkwing()
    np.savez_compressed(os.path.join(HERE, "assets", "sparkwing_deployed.npz"), v=v.astype(np.float32), f=f.astype(np.int32))
    print("sparkwing", len(f), "tris, extents", (v.max(0) - v.min(0)).round(3), "min", v.min(0).round(3))
