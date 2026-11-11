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
- Mac window mode (one pyglet window; drag the wx/wy/wz sliders, then GO; buttons/keys: G go, X inject loss, L sun change, SPACE skip cutscene, Q quit): `bash mac/setup_mac.sh` once (re-run it: it now installs moderngl and pyglet 2 and drops pyrender), then `bash mac/run_demo.sh`. Add `--no_cutscene` to skip the cutscene.
- Mac headless self-test (no window): `bash mac/test_demo_headless.sh` writes `mac/results/demo_test.mp4`.
- Headless capture (box): `python demo.py --headless --fixed_dt 0.1 --T 160 --record demo_capture.mp4 --script "0:w=0.02,0.661,0.02;1:go"`
