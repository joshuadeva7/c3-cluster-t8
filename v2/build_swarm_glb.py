"""Replace only the SWARM groups in assets/cad_render.npz with Daniel's 2026-11-04 SolidWorks glTF export
(SWARMAssembly_for_sim_2026-11-04.glb: 2 Sparkwing 3-panel deployed wings, fixed-length gripper arm).
GLB units = meters. Axis map identical to build_cad_meshes.py: anim = (glb y, glb x, -glb z) about the
Primary Structure box centre, so +y = gripper side (faces TOPEX), wings along z, arm span along x.
The export carries only the deployed wings, so arr_sto = arr_dep. TOPEX arrays are left untouched."""
import os, sys, numpy as np, trimesh, fast_simplification
HERE = os.path.dirname(os.path.abspath(__file__))
GLB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "assets", "cad", "SWARMAssembly_for_sim_2026-11-04.glb")
s = trimesh.load(GLB)
parts = []
for n in s.graph.nodes_geometry:
    T, g = s.graph[n]; m = s.geometry[g].copy(); m.apply_transform(T); parts.append((n, m))
prim = [m for n, m in parts if n.startswith("Primary Structure")]
lo = np.min([m.bounds[0] for m in prim], 0); hi = np.max([m.bounds[1] for m in prim], 0); ctr = (lo + hi) / 2
M = lambda v: np.c_[v[:, 1] - ctr[1], v[:, 0] - ctr[0], -(v[:, 2] - ctr[2])]
def dec(m, t):
    v, f = m.vertices, m.faces
    if len(f) > t: v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - t / len(f))
    return np.asarray(v, float), np.asarray(f)
def group(ps, target):
    tot = sum(len(p.faces) for _, p in ps); V, F, n0 = [], [], 0
    for _, p in ps:
        v, f = dec(p, max(12, int(target * len(p.faces) / tot)))
        V.append(v); F.append(f + n0); n0 += len(v)
    return np.vstack(V), np.vstack(F)
isarr = lambda n: n.startswith("Sparkwing")
isarm = lambda n: n.startswith(("TelescopeArm", "Pin"))
isend = lambda n: n.startswith(("pads", "rotor"))
isshell = lambda n: n.startswith(("Primary", "Door", "centralmotorbox"))
lo2, hi2 = lo + 0.01, hi - 0.01
inner = [(n, p) for n, p in parts if not (isarr(n) or isarm(n) or isend(n) or isshell(n))
         and ((p.bounds[0] < lo2).any() or (p.bounds[1] > hi2).any())]
print("visible outside bus:", sorted({n.split("_")[0] for n, _ in inner}))
out = dict(np.load(os.path.join(HERE, "assets", "cad_render.npz")))
for key, sel, tgt in (("shell", [x for x in parts if isshell(x[0])], 400), ("inner", inner, 1400),
                      ("arr_dep", [x for x in parts if isarr(x[0])], 700),
                      ("arm_ext", [x for x in parts if isarm(x[0])], 300), ("arm_end", [x for x in parts if isend(x[0])], 800)):
    v, f = group(sel, tgt)
    out[key + "_v"] = M(v).astype(np.float32); out[key + "_f"] = f.astype(np.int32)
    print(key, len(sel), "parts", len(f), "tris", out[key + "_v"].min(0).round(3), out[key + "_v"].max(0).round(3))
out["arr_sto_v"], out["arr_sto_f"] = out["arr_dep_v"], out["arr_dep_f"]
np.savez_compressed(os.path.join(HERE, "assets", "cad_render.npz"), **out)
