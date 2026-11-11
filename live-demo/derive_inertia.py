"""TOPEX inertia from the CAD mesh at the team's 2,500 kg (STM-AUDIT PP2 [107]); cross-check vs team spin-axis Iz = 11,665 kg m^2.
Mesh voxelised (0.08 m, filled). Two mass models:
  uniform : all 2,500 kg spread uniformly over the voxel solid (over-weights the thin solar array)
  two_comp: bus box 2,100 kg uniform, solar array (z > 2.5 m) 300 kg, other appendages (antenna, booms) 100 kg  <- USED
The array/appendage split is an assumption (no team source); it puts Iyy (spin axis) within ~4% of the team's 11,665.
Body frame = TOPEX mesh frame (+y = v2 spin axis).
DEMO ASSUMPTION (Daniel, 2026-11-11): J_used is NOT the CAD tensor. Roll (mesh y) is taken as the major axis: diag(yaw x 9,371, roll y 11,665,
pitch z 4,213) kg m^2 (Dennehy 1988 moments re-assigned), COM on the mesh y axis. The CAD tensor is kept as J_cad."""
import json, numpy as np, render
m = render.topex_mesh(); P = m.voxelized(0.08).fill().points; M = 2500.0; IZ = 11665.0
bus = (np.abs(P[:, 0]) < 1.3) & (np.abs(P[:, 2]) < 1.4) & (P[:, 1] > -3.5) & (P[:, 1] < 3.0); arr = P[:, 2] > 2.5
def tensor(w):
    c = (w[:, None] * P).sum(0) / w.sum(); X = P - c
    return c, (w[:, None, None] * (np.eye(3)[None] * np.sum(X ** 2, 1)[:, None, None] - X[:, :, None] * X[:, None, :])).sum(0)
cu, Iu = tensor(np.full(len(P), M / len(P)))
ma, mo = 300.0, 100.0; mb = M - ma - mo
w = np.where(bus, mb / bus.sum(), np.where(arr, ma / arr.sum(), mo / max(1, (~bus & ~arr).sum())))
c2, I2 = tensor(w)
e, V = np.linalg.eigh(I2)
out = dict(assumption="DEMO ASSUMPTION (Daniel, 2026-11-11): roll (mesh y, the long bus axis) is TOPEX's MAJOR principal axis, so the spin about roll is stable. "
           "Moments are the Dennehy 1988 values (via Benoit 2023) re-assigned: roll (mesh y) 11,665, yaw (mesh x, altimeter +x / HGA -x) 9,371, pitch (mesh z, solar array) 4,213 kg m^2, products of inertia 0. "
           "The centre of mass is placed on the mesh y axis (x = z = 0, CAD y kept) so the spin axis is the axis the v2 videos spin about and the v2 grip point (0, -3.877, 0) lies on it. "
           "Real TOPEX spins about yaw; the CAD-derived tensor (J_cad, com_cad_m) is kept below for reference.",
           mass_kg=M, J_used=np.diag([9371.0, IZ, 4213.0]).tolist(), J_used_principal_xyz=[9371.0, IZ, 4213.0], com_body_m=[0.0, float(c2[1]), 0.0],
           spin_axis_body=[0.0, 1.0, 0.0], spin_rate_default_rad_s=2 * np.pi / 9.5, H_default_Nms=IZ * 2 * np.pi / 9.5,
           voxel_pitch_m=0.08, model_used="two_comp (bus 2100 / array 300 / appendages 100 kg)",
           J_cad=I2.tolist(), J_cad_principal=e.tolist(), com_cad_m=c2.tolist(), cad_principal_axis_y_alignment=np.abs(V[1]).tolist(),
           Iyy_cad=float(I2[1, 1]), team_Iz_spin=IZ, Iyy_cad_vs_team=float(I2[1, 1] / IZ),
           uniform_com=cu.tolist(), uniform_I=Iu.tolist(), uniform_Iyy_vs_team=float(Iu[1, 1] / IZ))
json.dump(out, open("assets/topex_inertia.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("com_body_m", "J_used_principal_xyz", "J_cad_principal", "Iyy_cad_vs_team", "cad_principal_axis_y_alignment")}))
