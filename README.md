# C3-Cluster: TOPEX/Poseidon trajectory sim + mission animation

## Layout
- `traj/cluster_traj.cpp`: C++ sim (two-body + J2, RK4). Phase 1 is the BHT-600 low-thrust spiral, 500 to 1336 km, solved by shooting. Phase 2 is CW prox ops on hydrazine, docking, and the detumble budget.
- `traj/plot_traj.py`: figures 1-6, written to `traj/figs/`.
- `traj/summary.txt`: results from the last run.
- `anim/mission_anim.py`: 90 s video driven by the sim CSVs. `anim/assets` holds the fonts and land mask.
- `topex_mission.mp4`: the latest render.

## Run
```
cd traj
g++ -O2 -std=c++17 cluster_traj.cpp -o cluster_traj
./cluster_traj        # writes transfer.csv, prox.csv, burns.csv, events.csv, summary.txt
python3 plot_traj.py  # writes figs/
cd ../anim
python3 mission_anim.py preview 12 40 63   # stills, written to preview/
python3 mission_anim.py render             # writes topex_mission.mp4 (needs ffmpeg)
```
The animation reads the sim outputs from `../traj` by default. Set `SIM_DIR` to read them from somewhere else.
Python dependencies: numpy, scipy, pandas, matplotlib.

## Key inputs (midterm report Section 4 / STM)
- Wet mass: 166.036 kg
- BHT-600: 39 mN thrust, Isp 1500 s, 16 kg xenon
- Hydrazine: ArianeGroup 1 N thrusters, Isp 220 s, 18 kg loaded
- Δv budget: 1500 m/s per craft
- Keep-out zone and imager standoff: 22 m. Imager loops use b = 23 m to keep margin.
- Capture: 2.6 m, within gripper reach
- TOPEX: Iz = 11,665 kg·m², spin period 9.5 s (0.661 rad/s)
- Detumble torque: 0.5 N·m (1 N on a 0.5 m arm)
- MR-4 band: ±0.00175 rad/s

## Results (summary.txt)
- Spiral: 417 m/s over 20.3 days, using 4.6 kg of xenon
- Phasing wait: 6.85 h
- RAAN pre-offset: +10.0°
- Prox ops: 8.8 to 10.9 m/s on hydrazine
- Detumble: 4.27 h, using 7.13 kg of hydrazine
- Docker hydrazine total: 7.8 of 18 kg
