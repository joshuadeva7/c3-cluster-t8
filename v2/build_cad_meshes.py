"""Cache low-poly render meshes from the real SolidWorks CAD -> assets/cad_render.npz.

Source: tessellation stored inside the team's .SLDASM/.SLDPRT files (decoded by cad_extract/),
TOPEX from v2/assets/cad/topex.stl (TOPEX-Poseidon_scaled.SLDPRT, true scale, meters).
SWARM groups: body (always), arrays stowed/deployed, arm retracted/extended, each placed by the
assembly's own component transforms (configs 'Retracted' and 'Extended').
Animation body frame: +y toward TOPEX (assembly +x, the gripper side), x = assembly y, z = -assembly z,
origin = Primary Structure box centre.  Rerun only if the CAD changes (needs /workspace/swx unpacked streams).
"""
import os, sys, numpy as np, trimesh, fast_simplification
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cad_extract")); os.chdir("/workspace/swx")
from build import build

def dec(m, target):
    m = m.copy(); m.merge_vertices(); v, f = m.vertices, m.faces
    if len(f) > target:
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - target / len(f))
    return np.asarray(v, float), np.asarray(f)

def cluster(v, f, cell):
    # vertex-clustering decimation: snap to a grid, merge, drop collapsed triangles
    k = np.round(v / cell).astype(np.int64); _, inv = np.unique(k, axis=0, return_inverse=True); inv = inv.ravel()
    nv = np.zeros((inv.max() + 1, 3)); np.add.at(nv, inv, v); nv /= np.bincount(inv)[:, None]
    f2 = inv[f]; ok = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = np.unique(np.sort(f2[ok], 1), axis=0, return_index=True)[1]; f2 = inv[f][ok][np.sort(f2)]
    return nv, f2

def group(parts, target):
    if not parts: return np.zeros((0, 3)), np.zeros((0, 3), int)
    tot = sum(len(p.faces) for _, p in parts); V, F, n0 = [], [], 0
    for _, p in parts:
        v, f = dec(p, max(12, int(target * len(p.faces) / tot)) if len(p.faces) > 12 else len(p.faces))
        V.append(v); F.append(f + n0); n0 += len(v)
    return np.vstack(V), np.vstack(F)

ext, ret = build("2", False), build("138", False)
prim = dict(ext)["Primary Structure-1"]; ctr = prim.bounds.mean(0)
M = lambda v: np.c_[v[:, 1] - ctr[1], v[:, 0] - ctr[0], -(v[:, 2] - ctr[2])]
isarr = lambda n: "Sparkwing" in n
isarm = lambda n: n.startswith("gripperarm/TelescopeArm")
isend = lambda n: n.startswith(("gripperarm/pads", "gripperarm/rotor"))
out = {}
body = [(n, p) for n, p in ext if not isarr(n) and not isarm(n) and not isend(n)]
shell = [(n, p) for n, p in body if n.startswith(("Primary", "Door", "gripperarm/central"))]
lo, hi = prim.bounds[0] + 0.01, prim.bounds[1] - 0.01
inner = [(n, p) for n, p in body if not any(n == s for s, _ in shell)
         and ((p.bounds[0] < lo).any() or (p.bounds[1] > hi).any())]   # parts fully inside the closed bus never show
print("visible outside the bus:", [n for n, _ in inner])
for key, parts, tgt in (("shell", shell, 200), ("inner", inner, 900),
                        ("arr_dep", [x for x in ext if isarr(x[0])], 260), ("arr_sto", [x for x in ret if isarr(x[0])], 200),
                        ("arm_ext", [x for x in ext if isarm(x[0])], 300), ("arm_end", [x for x in ext if isend(x[0])], 600)):
    v, f = group(parts, tgt); cell = 0.004
    while len(f) > tgt * 1.6 and cell < 0.2: v, f = cluster(v, f, cell); cell *= 1.4
    out[key + "_v"] = M(v).astype(np.float32); out[key + "_f"] = f.astype(np.int32)
    print(key, len(parts), "parts", len(f), "tris", (out[key + "_v"].min(0)).round(3), (out[key + "_v"].max(0)).round(3))
# TOPEX: anim = (-sy, sz, -sx); bus box centre at origin
t = trimesh.load(os.path.join(HERE, "assets", "cad", "topex.stl"))
bus = t.vertices[(t.vertices[:, 0] > -1.3) & (t.vertices[:, 0] < 1.3)]
c = np.array([0.0, 0.5 * (-1.32 + 1.01), 0.5 * (-2.76 + 2.74)])   # bus box centre (main bus solid bounds)
t.merge_vertices(); t.fix_normals(); v, f = t.vertices, t.faces; g = v - c   # full 20,926-tri mesh, no decimation
A = np.c_[-g[:, 1], g[:, 2], -g[:, 0]]   # same body axes as the old glb: +y spin axis (aft end at -y), +z array
cen = A[f].mean(1)
col = np.where((cen[:, 2] > 2.5)[:, None], [0.12, 0.19, 0.36], np.where((np.abs(cen[:, 0]) < 1.3)[:, None] & (np.abs(cen[:, 2]) < 1.3)[:, None], [0.74, 0.60, 0.33], [0.80, 0.82, 0.85]))
out.update(topex_v=A.astype(np.float32), topex_f=f.astype(np.int32), topex_c=col.astype(np.float32))
print("topex", len(f), "tris", A.min(0).round(3), A.max(0).round(3))
np.savez_compressed(os.path.join(HERE, "assets", "cad_render.npz"), **out)
