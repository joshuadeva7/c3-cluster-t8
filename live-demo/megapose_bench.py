"""MegaPose (happypose) on the fixed TOPEX benchmark set. Runs in the happypose venv.
Modes: 'full' = oracle bbox -> coarse (SO(3) grid) -> refiner -> scoring (absolute pose, no prior);
       'track' = prior pose -> refiner only, n iterations (MegaPose's tracking use)."""
import os, sys, time, json, argparse, numpy as np, torch, pandas as pd
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--set", default="/workspace/live-demo-env/set_512.npz"); p.add_argument("--mode", default="track")
p.add_argument("--iters", type=int, default=1); p.add_argument("--prior", type=int, default=5); p.add_argument("--n", type=int, default=10)
p.add_argument("--grid_step", type=int, default=8); p.add_argument("--batch", type=int, default=1); p.add_argument("--json", default=""); p.add_argument("--threads", type=int, default=0)
a = p.parse_args()
if a.threads: torch.set_num_threads(a.threads)
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.tensor_collection import PandasTensorCollection
ds = RigidObjectDataset([RigidObject(label="topex", mesh_path=Path("/workspace/live-demo-env/topex_mesh/topex.ply"), mesh_units="mm")])
name = "megapose-1.0-RGB"; t0 = time.time(); pe = load_named_model(name, ds).to("cpu"); load_s = time.time() - t0
pe._SO3_grid = pe._SO3_grid[::a.grid_step]
S = np.load(a.set); K = S["K"].astype(np.float32)
def rerr(Ra, Rb): return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
def bbox(R, t, verts):
    X = verts @ R.T + t; u = X @ K.T; u = u[:, :2] / u[:, 2:]
    return np.r_[u.min(0), u.max(0)]
import trimesh; V = trimesh.load("/workspace/live-demo-env/topex_mesh/topex.ply").vertices / 1000.0
lat, re, te = [], [], []
idxs = list(range(a.n))
for start in range(-1, a.n, a.batch):   # -1 = warm-up
    ii = [max(0, start + k) % a.n for k in range(a.batch)]
    rgb = np.stack([S["ims"][i] for i in ii])
    obs = ObservationTensor.from_torch_batched(torch.from_numpy(rgb).permute(0, 3, 1, 2), None, torch.from_numpy(np.repeat(K[None], len(ii), 0))) if hasattr(ObservationTensor, "from_torch_batched") else None
    if obs is None:
        imgs = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        obs = ObservationTensor(images=imgs, K=torch.from_numpy(np.repeat(K[None], len(ii), 0)))
    infos = pd.DataFrame({"label": ["topex"] * len(ii), "batch_im_id": np.arange(len(ii)), "instance_id": np.zeros(len(ii), int)})
    t0 = time.perf_counter()
    if a.mode == "track":
        TCO = []
        for i in ii:
            pr = S[f"prior{a.prior}"][i]; T = np.eye(4, dtype=np.float32); T[:3, :3] = pr[:9].reshape(3, 3); T[:3, 3] = pr[9:]; TCO.append(T)
        est = PandasTensorCollection(infos=infos, poses=torch.from_numpy(np.stack(TCO)))
        preds, _ = pe.forward_refiner(obs, est, n_iterations=a.iters)
        out = preds[f"iteration={a.iters}"]
    else:
        bb = np.stack([bbox(S["R"][i], S["t"][i], V) for i in ii]).astype(np.float32)
        det = PandasTensorCollection(infos=infos, bboxes=torch.from_numpy(bb))
        out, _ = pe.run_inference_pipeline(obs, detections=det, **NAMED_MODELS[name]["inference_parameters"])
    dt = time.perf_counter() - t0
    if start < 0: continue
    lat.append(dt)
    P = out.poses.numpy(); bi = out.infos["batch_im_id"].values
    for k, i in enumerate(ii):
        T = P[list(bi).index(k)]; re.append(rerr(T[:3, :3], S["R"][i])); te.append(float(np.linalg.norm(T[:3, 3] - S["t"][i])))
res = dict(model=name, load_s=load_s, mode=a.mode, iters=a.iters if a.mode == "track" else 5, prior_deg=a.prior if a.mode == "track" else None, grid_step=a.grid_step if a.mode != "track" else None,
           res=int(S["ims"].shape[1]), batch=a.batch, threads=torch.get_num_threads(), n=len(re), rot_err_med=float(np.median(re)), rot_err_p90=float(np.percentile(re, 90)),
           trans_err_med_m=float(np.median(te)), s_per_call=float(np.median(lat)), hz_per_camera=a.batch / float(np.median(lat)))
print(json.dumps(res))
if a.json: open(a.json, "a").write(json.dumps(res) + "\n")
