# live-demo/ — real-time SWARM prox-ops demo (work in progress)
This is new work on branch `live-demo`. Joshua's files are untouched. Start with BENCHMARK.md.
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
- Mac (no conda; a plain venv on the system Python 3.11, default `VENV=$HOME/swarm-live/.venv`): `bash mac/setup_mac.sh` once (creates the venv if it is missing, installs the wheels, puts XFeat/LightGlue in `mac/_src`, which `mac/ensure_src.sh` adds to PYTHONPATH; no pip -e install), then `bash mac/test_demo_headless.sh` and `bash mac/run_demo.sh`. Flow: title screen (Enter) -> cutscene (Space skips) -> the same scene continues live while the imagers enter the NMC -> DOCKER SELECTION freeze: set TOPEX's body rates on the panel (wx, wz within +-0.008 rad/s, |wy| 0.55-0.75, default 0.661; GO is greyed out until valid) -> GO (G or click) -> prox ops to the grip and the time-warped detumble. P pauses/resumes, Esc quits (or Ctrl-C). Hidden test keys: X tracking loss, L rotate the sun. `--no_cutscene` / `--no_prologue` skip ahead.
- Mac headless self-test (no window): `bash mac/test_demo_headless.sh` writes `mac/results/demo_test.mp4`.
- Headless capture (box): `python demo.py --headless --fixed_dt 0.1 --T 160 --record demo_capture.mp4 --script "0:w=0.02,0.661,0.02;1:go"`
