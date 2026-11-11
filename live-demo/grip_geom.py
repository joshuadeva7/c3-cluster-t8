"""Grip-point candidates on TOPEX for a spin-matched capture.
A spin-matched docker can only sit still (relative to an inertially fixed point) on the angular-momentum axis. For a
steady spin about a principal axis that axis is a body principal axis, so the candidates are the ends of the major (e1)
and minor (e3) principal axes, plus the team's v2 dock point on -y (close to the intermediate axis e2).
For each candidate and claw roll angle, the docker (its +y = arm/camera axis pointing at TOPEX) is moved in along the
axis until the closest pad point is 2 cm from the TOPEX surface. Clearances are measured to the real dish component and
to the solar array. Writes assets/grip_candidates.json."""
import json, os, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import render as R_
HERE = os.path.dirname(os.path.abspath(__file__))
INR = json.load(open(os.path.join(HERE, "assets", "topex_inertia.json")))
COM = np.array(INR["com_body_m"]); J = np.array(INR["J_used"])
cad = np.load(os.path.join(HERE, "assets", "cad_render.npz"))
m0 = R_.topex_mesh(); c = m0.triangles_center
out_ = ((np.abs(c[:, 0]) > 1.7) | (np.abs(c[:, 2]) > 1.75) | (c[:, 1] < -3.4) | (c[:, 1] > 2.9))
arr = c[:, 2] > 2.5
dish = max(m0.submesh([np.nonzero(out_ & ~arr)[0]], append=True).split(only_watertight=False), key=lambda k: k.area)
tm = m0.copy(); tm.vertices = tm.vertices - COM
S_all = cKDTree(trimesh.sample.sample_surface(tm, 300000, seed=0)[0])
S_dish = cKDTree(trimesh.sample.sample_surface(dish, 20000, seed=0)[0] - COM)
S_arr = cKDTree(trimesh.sample.sample_surface_even(m0.submesh([np.nonzero(arr)[0]], append=True), 60000, seed=0)[0] - COM)
pads_m = trimesh.Trimesh(cad["arm_end_v"], cad["arm_end_f"], process=False)
PADS = np.r_[cad["arm_end_v"], trimesh.sample.sample_surface(pads_m, 6000, seed=0)[0]]
BODY = np.r_[cad["shell_v"], cad["arm_ext_v"], cad["arr_dep_v"]]
w, V = np.linalg.eigh(J)   # ascending: e3 minor? -> V[:,0] is minimum inertia
e_min, e_int, e_max = V[:, 0], V[:, 1], V[:, 2]
def frame(u, psi):
    y = -u; x = np.cross(y, [0.3, 0.1, 1.0]); x /= np.linalg.norm(x); z = np.cross(x, y)
    return Rot.from_rotvec(y * np.radians(psi)).as_matrix() @ np.c_[x, y, z]
def place(u, Rd):
    lo, hi = 0.5, 12.0
    for _ in range(22):
        d = 0.5 * (lo + hi); P = (PADS @ Rd.T) + u * d; dm = S_all.query(P)[0].min()
        if dm < 0.02: lo = d
        else: hi = d
    d = hi; P = PADS @ Rd.T + u * d; B = BODY @ Rd.T + u * d
    return d, dict(pad_topex=float(S_all.query(P)[0].min()), body_topex=float(S_all.query(B)[0].min()),
                   pad_dish=float(S_dish.query(P)[0].min()), pad_array=float(S_arr.query(P)[0].min()),
                   contact_pts_10cm=int((S_all.query(P)[0] < 0.10).sum()))
res = []
for name, u in (("+e_max", e_max), ("-e_max", -e_max), ("+e_min", e_min), ("-e_min", -e_min), ("+e_int", e_int), ("-e_int", -e_int)):
    u = u / np.linalg.norm(u); best = None
    for psi in range(0, 180, 15):
        Rd = frame(u, psi); d, m = place(u, Rd)
        ok = m["pad_dish"] > 0.5 and m["pad_array"] > 0.5 and m["body_topex"] >= 0.02
        score = (ok, m["contact_pts_10cm"], m["pad_dish"])
        if best is None or score > best[0]: best = (score, psi, d, m, Rd)
    _, psi, d, m, Rd = best
    res.append(dict(name=name, axis_body=u.tolist(), roll_deg=psi, standoff_m=d, R_dock=Rd.tolist(), ok=bool(best[0][0]), **m))
    print(name, np.round(u, 3), "roll", psi, "standoff %.3f" % d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, "OK" if best[0][0] else "REJECT")
# team v2 dock point (prox.csv: 3.877 m along -y, DOCK_CLOCK Ry(-90))
DOCK_PB = np.array([0, -3.877, 0]) - COM; Rd = Rot.from_euler("y", -90, degrees=True).as_matrix(); P = PADS @ Rd.T + DOCK_PB
mt = dict(pad_topex=float(S_all.query(P)[0].min()), pad_dish=float(S_dish.query(P)[0].min()), pad_array=float(S_arr.query(P)[0].min()), contact_pts_10cm=int((S_all.query(P)[0] < 0.10).sum()))
u = DOCK_PB / np.linalg.norm(DOCK_PB)
res.append(dict(name="team -y (v2)", axis_body=u.tolist(), roll_deg=None, standoff_m=float(np.linalg.norm(DOCK_PB)), R_dock=Rd.tolist(), ok=True, **mt))
print("team -y", np.round(u, 3), "angle to e_int %.2f deg" % np.degrees(np.arccos(abs(u @ e_int))), mt)
print("principal", np.round(w, 1)); print("e_max", np.round(e_max, 4), "e_int", np.round(e_int, 4), "e_min", np.round(e_min, 4))
json.dump(dict(principal=w.tolist(), e_min=e_min.tolist(), e_int=e_int.tolist(), e_max=e_max.tolist(), candidates=res),
          open(os.path.join(HERE, "assets", "grip_candidates.json"), "w"), indent=1)
