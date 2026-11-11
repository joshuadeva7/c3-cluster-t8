"""Markdown tables for BENCHMARK.md straight from the measured JSON (no hand-typed numbers)."""
import json, sys
f = lambda v, d=1: '-' if v is None else f"{v:.{d}f}"
src = sys.argv[1] if len(sys.argv) > 1 else 'results_box.jsonl'
print("| backend | crop px | prior err | lighting | model ms (6 crops = 3 cams) | full cycle ms (model+match+PnP, 3 cams) | Hz, all 3 cams | success | rot err med / p90 (deg) | trans err med (m) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in map(json.loads, open(src)):
    if 'error' in r: print(f"| {r['backend']} | {r['crop']} | | | ERROR {r['error'][:60]} | | | | | |"); continue
    print(f"| {r['backend']} | {r['crop']} | {r['prior_deg']}° | {'mismatch' if r['mismatch'] else 'same sun'} | {f(r['model_ms_3cam'])} | {f(r['cycle_ms_3cam'])} | {f(r['hz_3cam'])} | {r['success']:.2f} | {f(r['rot_err_med'],2)} / {f(r['rot_err_p90'],2)} | {f(r['trans_err_med_m'],2)} |")
if len(sys.argv) > 1: sys.exit()
print("\n| loop backend | lighting | acquisition s | acq. att err deg | meas Hz (3 cams, incl. CPU rendering) | track() ms (3 ref renders + model + PnP) | full cycle ms | att err deg (last 10 s) | rate err deg/s | pos err m | TRUE docker-relative rate after spin-match, deg/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for d in map(json.loads, open('loop.jsonl')):
    print(f"| {d['feat']} | {'mismatch' if d['mismatch'] else 'same sun'} | {d['acq_s']:.2f} | {d['acq_att_err_deg']:.2f} | {d['meas_hz_3cam']:.1f} | {d['model_ms_3cam']:.1f} | {d['cycle_ms']:.1f} | {d['att_err_deg_last10s']:.3f} | {d['rate_err_deg_s_last10s']:.3f} | {d['pos_err_m_last10s']:.3f} | {d['rel_rate_true_deg_s_last10s']:.3f} |")
print("\n| MegaPose mode | refiner iters | prior err | threads | n frames | s per camera-frame | Hz per camera | rot err med / p90 (deg) | trans err med (m) |")
print("|---|---|---|---|---|---|---|---|---|")
for d in map(json.loads, open('megapose.jsonl')):
    print(f"| {d['mode']} | {d['iters']} | {d['prior_deg']}° | {d['threads']} | {d['n']} | {d['s_per_call']:.3f} | {d['hz_per_camera']:.2f} | {d['rot_err_med']:.2f} / {d['rot_err_p90']:.2f} | {d['trans_err_med_m']:.2f} |")
print("\n| MEKF case w0 (rad/s) | Euler model | abs w0 deg/s | fix rate Hz | fix noise deg | rate err deg/s (mean, last 10 s of 60) | att err deg | time to stay < 0.5 deg/s (s) |")
print("|---|---|---|---|---|---|---|---|")
for d in map(json.loads, open('mekf_box.txt')):
    if 'predict_us' in d: tim = d; continue
    s = d['t_settle_rate_err_lt_0p5_deg_s']
    print(f"| {d['w0_rad_s']} | {d['euler_model']} | {d['w0_deg_s']:.2f} | {d['meas_hz']} | {d['meas_sig_deg']} | {d['rate_err_deg_s_final_mean_last10s']:.3f} | {d['att_err_deg_final_mean_last10s']:.2f} | {s if s is not None else 'not within 60 s'} |")
print(f"\nMEKF cost (numpy, box CPU): predict {tim['predict_us']:.0f} µs, update {tim['update_us']:.0f} µs.")
