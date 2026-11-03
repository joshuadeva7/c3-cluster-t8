# v2: mission sim + animation (TRR budgets)

New work lives here. Nothing outside `v2/` is touched. Merge later.

## Files
- `cluster_traj.cpp`: RK4 two-body + J2. Duty-cycled BHT-600 spiral 500 -> 1336 km, then CW prox ops for SC1 (docker) and SC2/SC3 (imagers). Writes CSVs + `summary.txt`.
- `mission_anim.py`: matplotlib animation driven by the CSVs. Phases follow `flowchart/`.
- `plot_traj.py`: static figures -> `out/figs/`.
- `flowchart/flowchart.py`: editable autonomy flowchart -> `flowchart.png` + `flowchart.mmd` (mermaid).
- `assets/landmask.npy`: Natural Earth 110 m land raster.

## Iterate fast
    ./sim.sh                                    # rebuild/rerun only if the .cpp changed (~3 min run)
    python3 mission_anim.py preview 50 54 62    # stills at those video seconds -> out/preview/ (~2 s first, ~0.6 s each after that)
    python3 mission_anim.py preview 62 --scale 0.5
    python3 mission_anim.py clip despin --fps 10 --scale 0.5   # one phase -> out/clip_despin.mp4 (~20 s)
    python3 mission_anim.py render              # full video (slow; do this last)
    python3 plot_traj.py

## Budgets and sources
- Hall/xenon: 1500 m/s dv capability ("Required Δv 1500 m/s", midterm STM: 3 targets + 3 extra orbit moves). 16 kg Xe (TRR mass budget D29, David's sizing).
  The spiral uses 433 m/s (Edelbaum 424.3 x 1.02 steering) and 4.8 kg Xe.
- Hydrazine: 18 kg N2H4 / 20 L per craft (STM; TRR mass budget D30). No source gives a hydrazine dv budget, so it is reported in kg.
  SC1: 7.7 kg (prox, spin-up, despin). SC2/SC3: 0.55 / 0.58 kg.
- MR-4: |ω| ≤ 0.00175 rad/s.
