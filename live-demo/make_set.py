"""Fixed benchmark set: N noisy RGB live frames of TOPEX with truth pose + perturbed priors (5/15/30 deg)."""
import sys, numpy as np, render, trimesh
res = int(sys.argv[1]) if len(sys.argv) > 1 else 512; N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rng = np.random.default_rng(7); r = render.SpaceRenderer(res)
ims, Rs, ts, pri = [], [], [], {5: [], 15: [], 30: []}
for i in range(N):
    R, t = render.random_pose(rng); col, _ = r.render(R, t)
    noisy = np.clip(col.astype(np.float32) + rng.normal(0, 3, col.shape), 0, 255).astype(np.uint8)
    ims.append(noisy); Rs.append(R); ts.append(t)
    for k in pri: pri[k].append(np.r_[render.perturb(R, t, rng, k, 0.02)[0].ravel(), render.perturb(R, t, rng, 0, 0.02)[1]])
np.savez_compressed(f"/workspace/live-demo-env/set_{res}.npz", ims=np.stack(ims), R=np.stack(Rs), t=np.stack(ts), K=r.K,
                    **{f"prior{k}": np.stack(v) for k, v in pri.items()})
m = render.topex_mesh(); m2 = m.copy(); m2.unmerge_vertices()
fc = m.visual.face_colors; vc = np.repeat(fc, 3, axis=0)
m3 = trimesh.Trimesh(m2.vertices * 1000.0, np.arange(len(m2.vertices)).reshape(-1, 3), vertex_colors=vc, process=False)  # mm
import os; os.makedirs("/workspace/live-demo-env/topex_mesh", exist_ok=True); m3.export("/workspace/live-demo-env/topex_mesh/topex.ply")
print("ok", res, N)
