"""Display-quality SWARM + TOPEX meshes -> assets/cad_hq.npz (Daniel 2026-11-16: same view, higher mesh quality).
SWARM: from the full SolidWorks glTF export (SWARMAssembly_for_sim_2026-11-04.glb, 3.4 M tris), same part groups, same axis map and
centre as v2/build_swarm_glb.py (which cut the groups to 400-1400 tris for the cutscene), decimated far less: shell/arm/pads at full
resolution, arrays and the externally visible components to the targets below. TOPEX: the same 20,926-tri mesh as cad_render.npz.
Every mesh also gets per-corner normals smoothed across edges flatter than CREASE deg (curved surfaces shade smoothly, real edges stay
sharp). Used only by the display renders; the simulated cameras keep cad_render.npz, so the vision/physics results do not change.
usage: python tools/build_hq_meshes.py [path/to/SWARMAssembly_for_sim_2026-11-04.glb]"""
import os, sys, numpy as np, trimesh, fast_simplification
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB = sys.argv[1] if len(sys.argv) > 1 else "/workspace/courses/ae-4322/anim-work/cad/SWARMAssembly_for_sim_2026-11-04.glb"
CREASE = 35.0
TGT = dict(shell=None, inner=90000, arr_dep=24000, arm_ext=None, arm_end=None)   # None = full resolution

def corner_normals(v, f, crease=CREASE):
    """(F,3,3) normals: per corner, area-weighted mean of the face normals around that vertex within `crease` deg of this face."""
    m = trimesh.Trimesh(v, f, process=False); m.merge_vertices()
    keep = (m.area_faces > 1e-14) & (m.faces[:, 0] != m.faces[:, 1]) & (m.faces[:, 1] != m.faces[:, 2]) & (m.faces[:, 0] != m.faces[:, 2])
    m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)   # degenerate (zero-area) triangles draw nothing; dropping them
    F = m.faces; fn = m.face_normals; ar = m.area_faces; vf = m.vertex_faces; cosc = np.cos(np.radians(crease))
    out = np.empty((len(F), 3, 3), np.float32); fc = np.repeat(np.arange(len(F)), 3); vc = F.reshape(-1)
    res = np.empty((len(vc), 3))
    for s in range(0, len(vc), 20000):
        adj = vf[vc[s:s + 20000]]; ok = adj >= 0; a_ = np.where(ok, adj, 0)
        w = (np.einsum("cdk,ck->cd", fn[a_], fn[fc[s:s + 20000]]) > cosc) & ok
        n = np.einsum("cd,cdk->ck", w * ar[a_], fn[a_]); res[s:s + 20000] = n
    bad = ~(np.linalg.norm(res, axis=1) > 1e-12); res[bad] = fn[fc[bad]]
    res /= np.linalg.norm(res, axis=1, keepdims=True)
    bad = ~np.isfinite(res).all(1); res[bad] = np.nan_to_num(fn[fc[bad]]); res[bad & ~(np.abs(res).sum(1) > 0)] = (0.0, 0.0, 1.0)   # sliver faces
    # the merged mesh has the same faces in the same order (merge only re-indexes), so corners line up with f
    return m.vertices.astype(np.float32), F.astype(np.int32), res.reshape(-1, 3, 3).astype(np.float32), keep

s = trimesh.load(GLB, process=False)
parts = []
for n in s.graph.nodes_geometry:
    T, g = s.graph[n]; m = trimesh.Trimesh(s.geometry[g].vertices, s.geometry[g].faces, process=False); m.apply_transform(T); parts.append((n, m))
prim = [m for n, m in parts if n.startswith("Primary Structure")]
lo = np.min([m.bounds[0] for m in prim], 0); hi = np.max([m.bounds[1] for m in prim], 0); ctr = (lo + hi) / 2
M = lambda v: np.c_[v[:, 1] - ctr[1], v[:, 0] - ctr[0], -(v[:, 2] - ctr[2])]
isarr = lambda n: n.startswith("Sparkwing"); isarm = lambda n: n.startswith(("TelescopeArm", "Pin"))
isend = lambda n: n.startswith(("pads", "rotor")); isshell = lambda n: n.startswith(("Primary", "Door", "centralmotorbox"))
lo2, hi2 = lo + 0.01, hi - 0.01
inner = [(n, p) for n, p in parts if not (isarr(n) or isarm(n) or isend(n) or isshell(n)) and ((p.bounds[0] < lo2).any() or (p.bounds[1] > hi2).any())]
def group(ps, target):
    tot = sum(len(p.faces) for _, p in ps); V, F, n0 = [], [], 0
    for _, p in ps:
        p = p.copy(); p.merge_vertices(); v, f = p.vertices, p.faces
        t = None if target is None else max(12, int(target * len(f) / tot))
        if t is not None and len(f) > t: v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - t / len(f))
        V.append(np.asarray(v, float)); F.append(np.asarray(f) + n0); n0 += len(v)
    return np.vstack(V), np.vstack(F)
out = {}
for key, sel in (("shell", [x for x in parts if isshell(x[0])]), ("inner", inner), ("arr_dep", [x for x in parts if isarr(x[0])]),
                 ("arm_ext", [x for x in parts if isarm(x[0])]), ("arm_end", [x for x in parts if isend(x[0])])):
    v, f = group(sel, TGT[key]); v, f, cn, _ = corner_normals(M(v), f)
    # M mirrors (z flip): keep the winding consistent with the normals (shader is two-sided; normals are what matter)
    out[key + "_v"], out[key + "_f"], out[key + "_n"] = v, f, cn
    print(key, len(sel), "parts", len(f), "tris", v.min(0).round(3), v.max(0).round(3))
cad = np.load(os.path.join(HERE, "assets", "cad_render.npz"))
v, f, cn, kt = corner_normals(cad["topex_v"].astype(float), cad["topex_f"])
out.update(topex_v=v, topex_f=f, topex_n=cn, topex_c=cad["topex_c"][kt])   # face colours follow the kept faces
print("topex", len(f), "tris")
np.savez_compressed(os.path.join(HERE, "assets", "cad_hq.npz"), **out)
print("wrote", os.path.getsize(os.path.join(HERE, "assets", "cad_hq.npz")) / 1e6, "MB")
