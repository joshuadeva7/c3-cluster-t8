# Live-demo pose pipeline: benchmark (box CPU, Nov 11 2026)

Every number below was measured and is generated from the JSON files in this folder by `gen_tables.py`. None are typed in by hand.
**Box hardware:** no GPU (`nvidia-smi` not found). 8-core Intel Xeon (Sapphire Rapids: AVX-512, AMX), 15 GB RAM, Mesa llvmpipe software OpenGL.
Treat the box numbers as relative comparisons. Real-time speed has to be proven on Daniel's M1 Pro, using `mac/` (see the end of this file).

## What runs (the real thing, not stand-ins)
`render.py` renders the exact TOPEX mesh from the v2 animation (`v2/assets/cad_render.npz`, 20,926 triangles, in metres). The renders use one harsh sun, almost no ambient light and a black background. Live frames add sensor blur and noise. Camera: 512×512 px, 30° HFOV, 25–45 m range.

Pose fix per camera (`tracker.py`):
1. Render TOPEX at the filter's predicted pose (colour + depth).
2. Crop both the live frame and the render to the predicted region of interest, resized to a fixed crop (256/192/128 px).
3. Run a **pretrained open feature model** on both crops: XFeat (CVPR 2024) or SuperPoint+LightGlue. ORB is included as a classical baseline.
4. Match the features. Lift the matched render keypoints to TOPEX body-frame 3-D points through the depth buffer.
5. Solve PnP (EPnP RANSAC, then LM refinement) to get R and t.

This is CAD-model render-and-match tracking, the same idea as the MegaPose/CosyPose refiners in our RR/midterm STM, but with features that run on a CPU budget.

`mekf.py` is a multiplicative EKF on the TOPEX attitude quaternion plus body rate. It optionally includes torque-free Euler dynamics from the CAD inertia, plus a constant-velocity position filter. It fuses any number of camera fixes and predicts at any rate, giving a continuous `[p, v, q, w]`. `relative_rate_body()` gives the docker-relative rate.

`run_loop.py` closes the whole loop live: truth torque-free tumble (slider rates) → render 3 cameras (imager A, imager B, docker) → model + PnP → MEKF → the docker spin-matches **using only the estimate**. Simulation time advances by measured wall time, so the rates shown are what the hardware sustains.

**Global acquisition (no truth used):** 60 icosahedral rotation hypotheses at a range taken from the silhouette size, each tracked for 3 iterations. The hypothesis with the most PnP inliers wins.

## 1. Model × optimization × crop (render-free fixed set, 30 camera-frames per row, `results_box.jsonl`)
"Hz, all 3 cams" means every one of the 3 cameras gets a new pose fix at that rate. Prior error is the error of the predicted pose fed in. The mismatch rows render the reference with a different sun direction than the live frame.
| backend | crop px | prior err | lighting | model ms (6 crops = 3 cams) | full cycle ms (model+match+PnP, 3 cams) | Hz, all 3 cams | success | rot err med / p90 (deg) | trans err med (m) |
|---|---|---|---|---|---|---|---|---|---|
| orb | 256 | 5° | same sun | 9.0 | 16.4 | 60.9 | 1.00 | 0.71 / 1.78 | 0.09 |
| orb | 256 | 15° | same sun | 9.0 | 23.4 | 42.7 | 0.93 | 1.86 / 6.77 | 0.27 |
| orb | 192 | 5° | same sun | 5.6 | 12.3 | 81.2 | 1.00 | 1.16 / 2.03 | 0.19 |
| orb | 192 | 15° | same sun | 5.4 | 19.0 | 52.5 | 0.93 | 4.87 / 15.70 | 0.89 |
| orb | 128 | 5° | same sun | 3.2 | 10.5 | 94.8 | 0.97 | 2.14 / 3.92 | 0.29 |
| orb | 128 | 15° | same sun | 2.8 | 13.3 | 75.0 | 0.67 | 6.19 / 13.30 | 0.58 |
| xfeat-onnx-cpu | 256 | 5° | same sun | 18.5 | 27.5 | 36.4 | 1.00 | 1.99 / 5.81 | 0.24 |
| xfeat-onnx-cpu | 256 | 15° | same sun | 18.9 | 31.1 | 32.2 | 0.93 | 5.80 / 13.96 | 0.63 |
| xfeat-onnx-cpu | 192 | 5° | same sun | 12.1 | 22.2 | 45.1 | 1.00 | 2.06 / 5.11 | 0.22 |
| xfeat-onnx-cpu | 192 | 15° | same sun | 11.9 | 23.7 | 42.2 | 0.63 | 6.52 / 23.53 | 1.59 |
| xfeat-onnx-cpu | 128 | 5° | same sun | 5.7 | 16.2 | 61.9 | 0.63 | 3.87 / 11.89 | 0.51 |
| xfeat-onnx-cpu | 128 | 15° | same sun | 5.8 | 17.1 | 58.4 | 0.20 | 8.65 / 27.61 | 0.28 |
| xfeat-onnx-cpu-int8 | 256 | 5° | same sun | 35.7 | 46.8 | 21.4 | 1.00 | 2.81 / 4.93 | 0.32 |
| xfeat-onnx-cpu-int8 | 256 | 15° | same sun | 41.5 | 57.2 | 17.5 | 1.00 | 6.34 / 11.01 | 0.43 |
| xfeat-onnx-cpu-int8 | 192 | 5° | same sun | 25.4 | 38.6 | 25.9 | 1.00 | 2.54 / 7.23 | 0.44 |
| xfeat-onnx-cpu-int8 | 192 | 15° | same sun | 25.8 | 40.8 | 24.5 | 0.63 | 9.02 / 16.34 | 1.04 |
| xfeat-onnx-cpu-int8 | 128 | 5° | same sun | 14.0 | 25.9 | 38.5 | 0.77 | 2.72 / 5.94 | 0.65 |
| xfeat-onnx-cpu-int8 | 128 | 15° | same sun | 14.0 | 26.1 | 38.4 | 0.30 | 7.73 / 15.99 | 0.84 |
| orb | 256 | 5° | mismatch | 8.0 | 16.5 | 60.5 | 1.00 | 0.91 / 1.97 | 0.10 |
| orb | 256 | 15° | mismatch | 7.8 | 21.4 | 46.7 | 0.93 | 2.78 / 13.70 | 0.47 |
| xfeat-onnx-cpu | 256 | 5° | mismatch | 22.1 | 32.5 | 30.8 | 1.00 | 1.84 / 4.50 | 0.31 |
| xfeat-onnx-cpu | 256 | 15° | mismatch | 20.6 | 33.4 | 29.9 | 1.00 | 4.23 / 9.49 | 0.60 |
| xfeat-torch-cpu | 256 | 5° | same sun | 25.1 | 36.3 | 27.6 | 1.00 | 1.99 / 5.81 | 0.25 |
| xfeat-torch-cpu | 256 | 15° | same sun | 21.5 | 34.1 | 29.4 | 0.93 | 5.80 / 13.96 | 0.63 |
| xfeat-torch-cpu | 192 | 5° | same sun | 13.3 | 23.1 | 43.3 | 1.00 | 2.06 / 5.11 | 0.22 |
| xfeat-torch-cpu | 192 | 15° | same sun | 11.5 | 23.5 | 42.5 | 0.63 | 6.52 / 23.53 | 1.59 |
| xfeat-torch-cpu | 128 | 5° | same sun | 8.6 | 20.7 | 48.3 | 0.63 | 3.87 / 11.89 | 0.51 |
| xfeat-torch-cpu | 128 | 15° | same sun | 10.8 | 23.7 | 42.2 | 0.20 | 8.65 / 27.61 | 0.28 |
| xfeat-torch-cpu-bf16 | 256 | 5° | same sun | 25.6 | 35.7 | 28.0 | 1.00 | 2.28 / 4.80 | 0.28 |
| xfeat-torch-cpu-bf16 | 256 | 15° | same sun | 25.6 | 38.4 | 26.0 | 0.97 | 5.35 / 13.98 | 0.63 |
| xfeat-torch-cpu-bf16 | 192 | 5° | same sun | 13.7 | 25.0 | 40.0 | 1.00 | 2.07 / 6.01 | 0.32 |
| xfeat-torch-cpu-bf16 | 192 | 15° | same sun | 13.3 | 26.8 | 37.3 | 0.63 | 9.71 / 17.11 | 1.28 |
| xfeat-torch-cpu-bf16 | 128 | 5° | same sun | 11.1 | 23.2 | 43.1 | 0.53 | 3.55 / 11.11 | 0.38 |
| xfeat-torch-cpu-bf16 | 128 | 15° | same sun | 9.5 | 22.2 | 45.1 | 0.20 | 13.58 / 20.54 | 0.87 |
| xfeat-orig | 256 | 5° | same sun | 31.7 | 42.7 | 23.4 | 1.00 | 1.89 / 3.94 | 0.28 |
| xfeat-orig | 256 | 15° | same sun | 28.9 | 45.7 | 21.9 | 1.00 | 4.07 / 13.04 | 0.40 |
| xfeat-orig | 192 | 5° | same sun | 261.5 | 333.5 | 3.0 | 1.00 | 2.18 / 4.15 | 0.31 |
| xfeat-orig | 192 | 15° | same sun | 13.3 | 26.0 | 38.4 | 0.63 | 6.32 / 15.67 | 1.27 |
| xfeat-orig | 128 | 5° | same sun | 7.8 | 19.0 | 52.6 | 0.67 | 3.32 / 12.35 | 0.61 |
| xfeat-orig | 128 | 15° | same sun | 8.2 | 19.9 | 50.3 | 0.13 | 12.53 / 26.44 | 0.35 |
| splg-cpu | 256 | 5° | same sun | 330.8 | 393.8 | 2.5 | 1.00 | 0.72 / 2.11 | 0.11 |
| splg-cpu | 256 | 15° | same sun | 344.9 | 413.1 | 2.4 | 0.90 | 1.66 / 4.37 | 0.21 |
| splg-cpu | 192 | 5° | same sun | 211.9 | 258.2 | 3.9 | 1.00 | 0.99 / 4.02 | 0.23 |
| splg-cpu | 192 | 15° | same sun | 209.1 | 254.7 | 3.9 | 0.67 | 1.48 / 2.99 | 0.19 |
| splg-cpu | 128 | 5° | same sun | 86.4 | 124.1 | 8.1 | 0.87 | 1.68 / 4.47 | 0.23 |
| splg-cpu | 128 | 15° | same sun | 83.1 | 119.2 | 8.4 | 0.27 | 2.69 / 7.43 | 0.52 |
| splg-cpu | 256 | 5° | mismatch | 351.3 | 417.7 | 2.4 | 1.00 | 0.72 / 1.76 | 0.13 |
| splg-cpu | 256 | 15° | mismatch | 368.4 | 423.4 | 2.4 | 0.83 | 1.60 / 4.85 | 0.26 |


**Speedups tried:**
- **Fused single-graph XFeat (`xfeat_fast.py`).** Backbone, softmax, NMS, top-k and descriptor sampling are all in one fixed-shape graph, so one call does everything. It exports to ONNX and converts to CoreML (the conversion was verified on the box; prediction needs macOS). Its accuracy matches upstream XFeat at 5° priors and is a little worse at the tail, because it uses bilinear instead of bicubic descriptor sampling.
- **Batching:** 6 crops (3 cameras × live/reference) go through one forward pass.
- **ROI crop:** the model only sees a 128–256 px crop around the predicted pose. Smaller crops are faster but lose accuracy and success at 15° priors; 192 px is the knee.
- **ONNX Runtime CPU** ≈ torch eager after the fix below. **int8 dynamic quantization was slower** on this CPU (35.7 vs 18.5 ms at 256 px), so it's dropped. **bf16** gave no gain.
- **Thread-contention fix:** OpenBLAS thread spinning between torch calls made torch CPU inference 5× slower inside the pipeline (129 → 25 ms for the same graph). `OPENBLAS_NUM_THREADS=1` is now set in the pipeline. ORT spinning is off too.
- **Not worth it:** PnP (EPnP RANSAC + LM) is about 0.15–0.3 ms per solve, and matching plus PnP is about 3 ms per camera.
- **Noisy measurement:** xfeat-orig 192 px / 5° measured 261 ms in this rerun, a system-noise outlier. The previous run in the same session measured 13.1 ms model / 24.0 ms cycle.

## 2. End-to-end closed loop on the box (`loop.jsonl`, 40 s, TOPEX |w0| = 4.0 deg/s, docker spin-match from t = 20 s)
The box renders on its CPU (llvmpipe, about 9 ms per 512² frame, 6 renders per cycle), so loop Hz here is bounded by rendering, not the model. On the Mac, Godot/Metal rendering will be about 1 ms.
| loop backend | lighting | acquisition s | acq. att err deg | meas Hz (3 cams, incl. CPU rendering) | track() ms (3 ref renders + model + PnP) | full cycle ms | att err deg (last 10 s) | rate err deg/s | pos err m | TRUE docker-relative rate after spin-match, deg/s |
|---|---|---|---|---|---|---|---|---|---|---|
| orb | same sun | 1.12 | 0.47 | 13.1 | 39.0 | 76.1 | 0.055 | 0.016 | 0.006 | 0.015 |
| orb | mismatch | 1.22 | 0.27 | 9.6 | 55.2 | 104.7 | 4.078 | 0.067 | 4.559 | 0.140 |
| xfeat-onnx-cpu | same sun | 2.34 | 0.81 | 10.4 | 56.2 | 96.3 | 0.250 | 0.077 | 0.034 | 0.073 |
| xfeat-onnx-cpu | mismatch | 2.11 | 0.60 | 9.3 | 63.9 | 107.1 | 0.522 | 0.163 | 0.100 | 0.161 |
| xfeat-torch-cpu | same sun | 2.27 | 0.81 | 11.0 | 53.9 | 90.7 | 0.225 | 0.065 | 0.027 | 0.061 |
| xfeat-torch-cpu | mismatch | 2.43 | 0.60 | 9.9 | 61.8 | 101.0 | 0.763 | 0.283 | 0.115 | 0.283 |


**Reading it:**
- With the real XFeat model in the loop (ONNX or torch), the continuous state holds 0.23–0.25° attitude error and 0.07–0.08 deg/s rate error with matched lighting, and 0.52–0.76° and 0.16–0.28 deg/s with mismatched lighting. The **true** docker-relative rate after spin-matching is 0.06–0.28 deg/s, so the display would read about 0. (`rel_rate_est` is about 0 by construction, so the true value is the honest metric.)
- The midterm STM rate requirement is ≤ 0.01 rad/s = 0.57 deg/s.
- **ORB is fast but breaks when the reference lighting differs from the live frame.** In the loop it drifted to 4.1° attitude and 4.6 m position error. XFeat stayed at 0.5°. That's the argument for the learned model.
- Acquisition took 1.1–2.4 s and landed within 0.3–0.8°. It was tested on one tumble seed only.

## 3. MegaPose (happypose, `megapose-1.0-RGB`, zero-shot, box CPU) — `megapose.jsonl`
| MegaPose mode | refiner iters | prior err | threads | n frames | s per camera-frame | Hz per camera | rot err med / p90 (deg) | trans err med (m) |
|---|---|---|---|---|---|---|---|---|
| track | 1 | 5° | 8 | 3 | 0.224 | 4.46 | 4.72 / 4.93 | 1.34 |
| track | 3 | 5° | 4 | 5 | 0.298 | 3.36 | 5.37 / 8.73 | 1.50 |
| track | 3 | 15° | 4 | 5 | 0.341 | 2.93 | 14.34 / 17.36 | 1.23 |


MegaPose runs on TOPEX, but in this integration the refiner did **not** reduce the prior error (5° → 4.7–5.4°, 15° → 14.3°) and costs 0.22–0.34 s per camera-frame on 4–8 CPU threads. Most likely this is an integration issue: vertex colours not loaded by panda3d, a 10 m object at 25–45 m being out of distribution, or the crop/K convention. It isn't a verdict on the model yet. Full coarse+refine mode was not run. On the box it isn't a real-time candidate: about 3–4.5 Hz per camera, single-camera only. FoundationPose needs CUDA (nvdiffrast) and is untested.

## 4. Rate filter (MEKF) on synthetic tumbles (`mekf_box.txt`, filter only, noisy pose fixes)
| MEKF case w0 (rad/s) | Euler model | abs w0 deg/s | fix rate Hz | fix noise deg | rate err deg/s (mean, last 10 s of 60) | att err deg | time to stay < 0.5 deg/s (s) |
|---|---|---|---|---|---|---|---|
| [0.0, 0.035, 0.0] | False | 2.01 | 5 | 1.5 | 0.188 | 0.78 | 3.2 |
| [0.02, 0.035, 0.01] | False | 2.38 | 5 | 1.5 | 0.199 | 0.63 | 4.2 |
| [0.02, 0.035, 0.01] | True | 2.38 | 5 | 1.5 | 0.132 | 0.61 | 4.4 |
| [0.05, 0.1, -0.04] | False | 6.80 | 10 | 2.0 | 0.728 | 1.61 | not within 60 s |
| [0.05, 0.1, -0.04] | True | 6.80 | 10 | 2.0 | 0.163 | 0.66 | 3.9 |
| [0.05, 0.1, -0.04] | True | 6.80 | 2 | 2.0 | 0.180 | 0.97 | 5.0 |

MEKF cost (numpy, box CPU): predict 72 µs, update 49 µs.

Without the Euler model, a 6.8 deg/s tumble leaves 0.73 deg/s of rate error. The torque-free Euler model brings it to 0.16 deg/s. Even at 2 Hz fixes the rate error is 0.18 deg/s. The inertia used is a placeholder shape, not TOPEX's (TODO: CAD mass properties).

## Recommendation (best pipeline so far)
**Real model:** fused XFeat (pretrained, no training needed) in render-and-match CAD tracking on a 192–256 px ROI crop, all 3 cameras in one batched call, then PnP, then the MEKF with the Euler model, predicting at 240 Hz for the continuous state. Global acquisition is a 60-hypothesis sweep at start and on loss.
- **Box CPU (ONNX Runtime):** model plus PnP for all 3 cameras takes 22–28 ms at 192–256 px crops, which is 36–45 Hz per camera with real model inference.
- **Mac:** the remaining question is the CoreML/ANE or MPS speed. That's what `mac/run_mac.sh` measures.
- **SuperPoint+LightGlue:** the most accurate learned option (0.72° median), but 2.4 Hz on the box CPU. It's worth checking on the Mac GPU as a slower high-accuracy channel.

**Untested / next:**
- Mac GPU/ANE numbers.
- Rendering in Godot/Metal instead of llvmpipe.
- Acquisition over many seeds.
- Real TOPEX inertia from CAD.
- MegaPose integration debugging, and MegaPose on MPS.
- LightGlue ONNX/CoreML export.
- Fine-tuning a keypoint net on TOPEX renders as a second independent channel. Renders take 9 ms each on the box, so a 20k-image set is about 3 CPU-minutes, but training wasn't attempted.
- Fast tumbles above 7 deg/s.
- Godot domain gap (different materials than these renders).

## Mac (Apple M1 Pro) — not run yet
```
cd live-demo && bash mac/setup_mac.sh && bash mac/run_mac.sh
```
It writes `mac/results/<timestamp>/summary.md` + `results.jsonl` + `loop.jsonl`. Backends: ORB, XFeat (torch CPU / MPS / MPS-fp16, ONNX CPU / CoreML EP, CoreML ALL / NE / GPU / CPU), SuperPoint+LightGlue (CPU / MPS), a lighting-mismatch set, MEKF timing, and the end-to-end loop if pyrender can open an offscreen GL context on macOS.

## Live demo app (`demo.py`), box headless runs, 2026-11-11

Inputs: three TOPEX body-rate sliders (±1.0 rad/s per axis) and a GO trigger. Truth is torque-free rigid-body motion with the CAD inertia. Two imagers and the docker render live camera views. Fused XFeat (ONNX CPU, 192 px crop) render-and-match feeds PnP fixes from all cameras into one MEKF. The docker is guided only by the MEKF estimate. Grip contact is scored against truth. Every run below used `--fixed_dt 0.1` (sim time), 3 cameras and a 512 px camera render. The recording is `demo_capture.mp4` (cutscene, then the live run).

| run (script) | phase times (s, sim) | att err med (deg) | rate err med (deg/s) | grip pos err (m) | grip att err (deg) | true rel rate at grip (deg/s) | pads→dish (m) | pads→TOPEX (m) |
|---|---|---|---|---|---|---|---|---|
| nominal w=(0.02,0.66,0.02) | ACQ 1.1, HOLD 5.6, SPIN 7.7, FINAL 16.8, GRIP 43.1, STABLE 14863, DONE 14883 | 0.354 | 0.135 | 0.196 | 0.564 | 0.400 | 3.89 | 0.020 |
| 1.0 rad/s about y | GRIP 43.2 | 0.249 | 0.135 | 0.348 | 0.444 | 0.840 | 3.65 | 0.333 |
| 1.0 rad/s, (0.577,0.577,0.577) | GRIP 53.5 | 0.369 | 0.205 | 0.460 | 0.691 | 1.168 | 3.96 | 0.208 |
| loss at 12 s, light +75° at 30 s | loss 12.1, re-acq done 14.7 (2.6 s), GRIP 45.5 | 0.356 | 0.204 | 0.145 | 0.226 | 0.178 | 3.85 | 0.079 |
| `--sun_unknown`, light +75° at 30 s | GRIP 44.5 | 1.167 | 0.567 | 0.151 | 0.296 | 2.106 | 3.79 | 0.057 |

- Detumble after grip: one 1 N thruster on a 0.5 m arm (0.5 N·m) fired against the angular momentum, time-warped x300. It took 43.2 → 14,863 s sim = 4.12 h to reach the MR-4 band of 0.00175 rad/s. The team figure is 4.27 h, based on Iz = 11,665 kg·m².
- Nominal run timing, box CPU, single process: median cycle 95 ms, of which 33–48 ms is model time. Fixes run at 10 Hz sim (one per `fixed_dt`).
- Docker kinematics are honest and unconstrained. Max commanded docker speed was 18.3 m/s (nominal), 24.1 m/s (1 rad/s about y) and 30.8 m/s (1 rad/s diagonal). The 30 m hold point sweeps at ω·30 m. Peak commanded rates were 49 deg/s (nominal) and 311–442 deg/s at 1 rad/s; those peaks are the feedback transient when spin-match starts. For the demo the docker is assumed to be able to fly this. A real docker would hold closer in, or match spin first.
- Dish clearance is measured against the real HGA dish mesh component (1,008 triangles on the −x boom). The pads grip the aft bus face along −y, 3.6–4.0 m from the dish in every run.

### Inertia used
`derive_inertia.py` voxelizes the TOPEX CAD at 0.08 m, at a mass of 2,500 kg (STM-AUDIT PP2).
- Two-component model (assumed split: bus 2,100 kg, array 300 kg, appendages 100 kg): principal moments 6,992 / 11,170 / 15,245 kg·m², Iyy = 11,174 kg·m², which is 0.958 of the team value Iz = 11,665 kg·m². **This model is used** (`assets/topex_inertia.json`).
- Uniform-density model: 0.858 of the team value.
- Finding: in the CAD model the y axis (the team's spin axis) is the **intermediate** principal axis. Pure spin about y is therefore unstable, so the 0.66 rad/s tumble wanders across axes. This is visible in the rate plots.

### Slider max
±1.0 rad/s per axis. This matches the team's ~1 rad/s design rate ("inter-frame rotation unambiguous up to ~1 rad/s"); the nominal tumble is 0.661 rad/s. Both 1.0 rad/s cases above tracked and gripped.

## Apple M1 Pro results (Daniel's Mac, 2026-11-11, `mac/run_mac.sh`)
macOS 27.2, torch 2.7.1, coremltools 9.0, ORT 1.31.1 with the CoreML EP. Same 30-frame set as the box: 3 cameras, 512 px renders. Raw output is in `results_mac/` (summary.md, results.jsonl, mekf.txt, system.txt).

| backend | crop | prior | model ms (3 cams) | cycle ms | Hz | success | rot err med/p90 (deg) |
|---|---|---|---|---|---|---|---|
| **xfeat-coreml-all** (demo default on Mac) | 256 | 5° | 7.7 | 20.8 | 48.0 | 1.00 | 2.26/4.17 |
| xfeat-coreml-all | 256 | 15° | 7.8 | 20.9 | 47.9 | 1.00 | 3.76/10.66 |
| xfeat-coreml-all, lighting mismatch | 256 | 5° | 7.2 | 19.5 | 51.2 | 1.00 | 2.03/4.34 |
| xfeat-coreml-ne | 256 | 5° | 7.7 | 20.3 | 49.2 | 1.00 | 2.26/4.17 |
| xfeat-coreml-gpu | 256 | 5° | 6.5 | 13.9 | 72.1 | **0.70** | 2.46/7.90 |
| xfeat-coreml-cpu | 256 | 5° | 13.5 | 21.0 | 47.5 | 1.00 | 1.67/4.76 |
| xfeat-onnx-coreml | 256 | 5° | 15.0 | 23.9 | 41.9 | 1.00 | 2.20/5.59 |
| xfeat-torch-mps | 256 | 5° | 21.1 | 29.4 | 34.0 | 1.00 | 2.20/5.59 |
| xfeat-onnx-cpu | 256 | 5° | 27.6 | 36.4 | 27.5 | 1.00 | 2.20/5.59 |
| xfeat-orig (upstream) | 256 | 5° | 43.6 | 52.3 | 19.1 | 1.00 | 2.06/3.94 |
| orb | 256 | 5° | 8.0 | 13.6 | 73.4 | 1.00 | 0.73/1.90 |
| splg-cpu | 256 | 5° | 886.6 | 1030.0 | 1.0 | 1.00 | 0.72/2.11 |
| splg-mps | 256 | 5° | 208.1 | 1217.4 | 0.8 | 1.00 | 0.72/2.11 |

- The fused XFeat graph on CoreML (all compute units) runs 3 cameras in 7.7 ms of model time, 5.7× faster than upstream XFeat on the same Mac (43.6 ms), with the same 100% success rate.
- The GPU-only CoreML build is faster (6.5 ms) but succeeds on only 70% of frames, against 100% for the other CoreML units on identical inputs. Treat it as numerically off (probably reduced-precision drift in the softmax/NMS head) and don't use it.
- At 192 px CoreML loses accuracy (3.97° median against 2.26° at 256 px) and saves only 1.7 ms, so the Mac demo defaults to 256 px. The box default stays at 192 px.
- SuperPoint+LightGlue on MPS (0.8 Hz; 0.2 Hz with lighting mismatch) is slower than on CPU and unusable live.
- MEKF on the Mac: predict 65 µs, update 44 µs.
- The Mac closed-loop runs (`run_loop.py`) failed: pyrender's macOS path opens a pyglet window, and pyglet found no screen (`IndexError` in `get_default_screen`, see `stderr.log`). Fix: on macOS the renderer now defaults to moderngl with a standalone CGL context (`gl_render.py`, `LIVE_DEMO_RENDER=gl`). The demo UI is a single pyglet window that owns the only GL context, so the OpenCV GUI is no longer used.
