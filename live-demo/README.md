# SWARM live demo (AE 4322 Team 8)

This is a real-time version of the SWARM mission video. It plays the opening cutscene, then hands over to a live simulation. The imagers fly out to their mapping orbits, you pick how fast TOPEX is spinning, and the docker uses its cameras to estimate TOPEX's pose. It then spin-matches, grips, detumbles TOPEX, releases it, and the swarm flies away.

## What you need
- A Mac with Apple Silicon (M1 or newer). This is the only setup tested with a window.
- Python 3.10 or newer (`python3 --version`). If you don't have it, install it from [python.org](https://www.python.org/downloads/macos/).
- Git, plus access to this repo.
- Internet for the one-time setup, which downloads PyTorch and a few other large Python packages.

## Setup (once)
Open Terminal and run:

```
git clone https://github.com/joshuadeva7/c3-cluster-t8.git
cd c3-cluster-t8/live-demo
bash mac/setup_mac.sh
```

The setup creates a self-contained Python environment at `~/swarm-live/.venv`. It doesn't use conda and doesn't touch your system Python. To remove everything later, delete `~/swarm-live` and the cloned folder. To put the environment somewhere else, run `VENV=/path/to/venv bash mac/setup_mac.sh` and use the same `VENV=` when you run the demo.

Optional check, with no window: `bash mac/test_demo_headless.sh` records a short test video to `mac/results/demo_test.mp4`.

## Run it
From the `live-demo` folder:

```
bash mac/run_demo.sh
```

It takes about 20 seconds to start while the vision model loads and self-tests.

## How it plays
1. **Title card:** press Enter.
2. **Cutscene:** press Space to skip it.
3. **Live:** the imagers fly out to their mapping orbits around TOPEX.
4. **Freeze, TOPEX rate pop-up:** drag the slider to set TOPEX's roll spin rate (0.10 to 0.789 rad/s, default 0.661), then press G, press Enter, or click GO. GO stays grayed out until the value is valid.
5. **Prox ops:** the docker estimates TOPEX's pose, approaches, spin-matches, makes its final approach and grips. TOPEX is then detumbled (time-warped), held stable, and released. The swarm backs off, the view zooms out, the Hall thrusters light, and the craft fly away to the mission patch.
6. At the end, press Enter to restart or Esc to quit.

## Keys
- **P:** pause and resume.
- **Esc:** quit (Ctrl-C in Terminal also works).
- **Fault tests:** L changes the lighting, X corrupts the pose estimate so you can watch it recover, and I fails the visible camera so the docker switches to infrared.

## Screen layout
The main view stays centered. The left column has the docker cam above a spare view, and the right column has imager A and imager B. Each cam tile has a small plot of TOPEX's spin rate relative to that craft in deg/s. Bright lines are the estimate and dim dashed lines are the truth, with the plot's max and min values in the small left column. The bottom line shows the attitude and rate estimate errors.

## Options
Extra arguments pass straight through, for example `bash mac/run_demo.sh --no_cutscene` (start live) or `--no_prologue` (jump to the rate pop-up). Run `python demo.py --help` inside the environment for the full list.

## Troubleshooting
- **"no venv at ...":** run `bash mac/setup_mac.sh` first.
- **Setup fails on Python:** check that `python3 --version` is 3.10 or newer, or point it at one with `PYTHON=/path/to/python3 bash mac/setup_mac.sh`.
- **Black or frozen window:** quit with Esc and relaunch. `bash mac/diag.sh` prints a diagnostic report.

---
## Developer notes
- `render.py`: offscreen space renderer for the v2 TOPEX mesh.
- `tracker.py`: CAD render-and-match tracker.
- `features.py`, `xfeat_fast.py`: model backends.
- `mekf.py`: continuous state filter.
- `run_loop.py`: closed loop.
- `pose_bench.py` with `bench_data_256.npz`: portable model benchmark.
- `megapose_bench.py`: MegaPose, runs in the happypose venv.
- `mac/`: Apple Silicon setup and run scripts.

Box environment: `source /workspace/live-demo-env/bin/activate`. MegaPose: `/workspace/live-demo-env/src/happypose/.venv`, with `HAPPYPOSE_DATA_DIR=/workspace/live-demo-env/hp_data P3D_DISPLAY=p3headlessgl`.

## Live demo
- Mac headless self-test (no window): `bash mac/test_demo_headless.sh` writes `mac/results/demo_test.mp4`.
- Headless capture (box): `python demo.py --headless --fixed_dt 0.1 --T 160 --record demo_capture.mp4 --script "0:w=0.02,0.661,0.02;1:go"`
