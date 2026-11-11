"""TOPEX inertia from the CAD mesh at the team's 2,500 kg (STM-AUDIT PP2 [107]); cross-check vs team spin-axis Iz = 11,665 kg m^2.
Mesh voxelised (0.08 m, filled). Two mass models:
  uniform : all 2,500 kg spread uniformly over the voxel solid (over-weights the thin solar array)
  two_comp: bus box 2,100 kg uniform, solar array (z > 2.5 m) 300 kg, other appendages (antenna, booms) 100 kg  <- USED
The array/appendage split is an assumption (no team source); it puts Iyy (spin axis) within ~4% of the team's 11,665.
Body frame = TOPEX mesh frame (+y = v2 spin axis)."""
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
out = dict(mass_kg=M, voxel_pitch_m=0.08, model_used="two_comp (bus 2100 / array 300 / appendages 100 kg)",
           com_body_m=c2.tolist(), J_used=I2.tolist(), J_used_principal=e.tolist(), principal_axis_y_alignment=np.abs(V[1]).tolist(),
           Iyy_used=float(I2[1, 1]), team_Iz_spin=IZ, Iyy_vs_team=float(I2[1, 1] / IZ),
           uniform_com=cu.tolist(), uniform_I=Iu.tolist(), uniform_Iyy_vs_team=float(Iu[1, 1] / IZ),
           note="In both models the +y spin axis is the INTERMEDIATE principal axis -> torque-free spin about y is unstable.")
json.dump(out, open("assets/topex_inertia.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("com_body_m", "J_used_principal", "Iyy_used", "Iyy_vs_team", "uniform_Iyy_vs_team", "principal_axis_y_alignment")}))
