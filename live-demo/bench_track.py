"""Benchmark the render-and-match tracker: accuracy vs truth and latency, single camera and 3-camera batch."""
import argparse, time, json, numpy as np, torch, render, features, tracker
p = argparse.ArgumentParser()
p.add_argument("--feat", default="xfeat"); p.add_argument("--res", type=int, default=512); p.add_argument("--crop", type=int, default=256)
p.add_argument("--pert", type=float, default=5.0); p.add_argument("--tpert", type=float, default=0.02); p.add_argument("--n", type=int, default=30)
p.add_argument("--iters", type=int, default=1); p.add_argument("--mismatch", action="store_true", help="reference renders use a different sun direction than the live frame")
p.add_argument("--threads", type=int, default=0); p.add_argument("--topk", type=int, default=512); p.add_argument("--json", default="")
a = p.parse_args()
if a.threads: torch.set_num_threads(a.threads)
rng = np.random.default_rng(1)
live = render.SpaceRenderer(a.res)
ref = render.SpaceRenderer(a.res, share=live, sun_dir_cam=(-0.2, -0.6, 0.7) if a.mismatch else (-0.5, -0.3, 0.8))
kw = {"top_k": a.topk} if a.feat == "xfeat" else ({"max_kp": a.topk} if a.feat == "sp_lg" else {"n": a.topk})
F = features.make(a.feat, **kw); T = tracker.Tracker(ref, F, crop_res=a.crop, iters=a.iters)
# warm-up
R, t = render.random_pose(rng); g = render.degrade(live.render(R, t)[0], rng); T.track([g], [(R, t)])
re, te, ok, lat1 = [], [], 0, []
frames = []
for i in range(a.n):
    R, t = render.random_pose(rng); g = render.degrade(live.render(R, t)[0], rng)
    pr = render.perturb(R, t, rng, a.pert, a.tpert); frames.append((g, R, t, pr))
    t0 = time.perf_counter(); out, nin = T.track([g], [pr]); lat1.append(time.perf_counter() - t0)
    if out[0] is None: continue
    ok += 1; re.append(render.rot_err_deg(out[0][0], R)); te.append(np.linalg.norm(out[0][1] - t))
# 3-camera batch: one extract() call over 6 images (3 live crops + 3 refs)
lat3 = []
for i in range(0, len(frames) - 2, 3):
    fr = frames[i:i + 3]; t0 = time.perf_counter(); T.track([f[0] for f in fr], [f[3] for f in fr]); lat3.append(time.perf_counter() - t0)
pe = [render.rot_err_deg(f[3][0], f[1]) for f in frames]
res = dict(feat=a.feat, res=a.res, crop=a.crop, topk=a.topk, iters=a.iters, mismatch=a.mismatch, pert_deg=a.pert, threads=torch.get_num_threads(),
           prior_rot_err_med=float(np.median(pe)), success=ok / a.n,
           rot_err_med=float(np.median(re)) if re else None, rot_err_p90=float(np.percentile(re, 90)) if re else None,
           trans_err_med_m=float(np.median(te)) if te else None, range_m="25-45",
           ms_1cam=1000 * float(np.median(lat1)), hz_1cam=1 / float(np.median(lat1)), ms_3cam=1000 * float(np.median(lat3)), hz_3cam_each=1 / float(np.median(lat3)))
print(json.dumps(res))
if a.json: open(a.json, "a").write(json.dumps(res) + "\n")
