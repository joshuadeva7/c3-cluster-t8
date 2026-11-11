import json, sys, os
out = sys.argv[1] if len(sys.argv) > 1 else "."
rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl"))] if os.path.exists(os.path.join(out, "results.jsonl")) else []
print("| host | backend | crop | prior deg | mismatch | model ms (3 cams) | cycle ms (3 cams) | Hz (3 cams) | success | rot err med/p90 deg | trans err med m |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if "error" in r: print(f"| {r['host']} | {r['backend']} | {r['crop']} | | | ERROR: {r['error'][:80]} | | | | | |"); continue
    f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
    print(f"| {r['host']} | {r['backend']} | {r['crop']} | {r['prior_deg']} | {r['mismatch']} | {f(r['model_ms_3cam'])} | {f(r['cycle_ms_3cam'])} | {f(r['hz_3cam'])} | {r['success']:.2f} | {f(r['rot_err_med'],2)}/{f(r['rot_err_p90'],2)} | {f(r['trans_err_med_m'],2)} |")
lp = os.path.join(out, "loop.jsonl")
if os.path.exists(lp):
    print("\n| e2e backend | meas Hz (3 cams, incl. render) | model ms | att err deg (last 10 s) | rate err deg/s | true rel rate deg/s after spin-match |\n|---|---|---|---|---|---|")
    for l in open(lp):
        r = json.loads(l); print(f"| {r['feat']} | {r['meas_hz_3cam']:.1f} | {r['model_ms_3cam']:.1f} | {r['att_err_deg_last10s']:.3f} | {r['rate_err_deg_s_last10s']:.3f} | {r['rel_rate_true_deg_s_last10s']:.3f} |")
