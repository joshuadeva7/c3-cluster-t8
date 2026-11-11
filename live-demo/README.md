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
- Mac (window, sliders, G = go, X = inject loss, L = light change, Q = quit): `bash mac/setup_mac.sh` once, then `bash mac/run_demo.sh`. Add `--no_cutscene` to skip the cutscene.
- Headless capture (box): `python demo.py --headless --fixed_dt 0.1 --T 160 --record demo_capture.mp4 --script "0:w=0.02,0.66,0.02;1:go"`
