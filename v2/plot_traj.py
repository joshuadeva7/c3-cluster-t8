"""Presentation plots for cluster_traj.cpp output.

    ./cluster_traj && python3 plot_traj.py      -> figs/*.png
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

RE = 6378.137
KOZ, APPR = 22.0, 200.0

BG, PANEL, INK, MUTED, GRID = "#0b1118", "#111a24", "#e6ecf2", "#7f8fa0", "#22303e"
SC = ["#f2a33a", "#6fa0ff", "#c792ea"]          # SC1 docker, SC2/SC3 imagers
TGT, RED, GOOD = "#e6ecf2", "#ff6b6b", "#5fd07c"
NAMES = ["SC1 (docker)", "SC2 (imager)", "SC3 (imager)"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": "#2a3746",
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 13, "font.family": "DejaVu Sans", "axes.titleweight": "bold",
    "legend.facecolor": PANEL, "legend.edgecolor": "#2a3746",
})

os.chdir(os.environ.get("SIM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")))
os.makedirs("figs", exist_ok=True)
tr = pd.read_csv("transfer.csv")
px = pd.read_csv("prox.csv")
bu = pd.read_csv("burns.csv")
ev = pd.read_csv("events.csv")
th = px.t / 3600


def save(fig, name):
    fig.savefig(f"figs/{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ 1. transfer
fig = plt.figure(figsize=(16, 7.5))
ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
ax.set_facecolor(BG)
u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
ax.plot_surface(RE * np.cos(u) * np.sin(v), RE * np.sin(u) * np.sin(v), RE * np.cos(v),
                color="#1d4f7a", alpha=0.55, linewidth=0, zorder=0)
T = tr[["tx", "ty", "tz"]].values / 1e3
C = tr[["c0x", "c0y", "c0z"]].values / 1e3
park = tr.phase.values == 0
xfer = tr.t.values >= tr.t.values[np.argmax(tr.phase.values >= 1)]   # from first Hall arc on (coasts included)
# full TOPEX orbit (one period from start)
a = np.linalg.norm(T[-1])
n = np.cross(T[-2], T[-1]); n /= np.linalg.norm(n)
e1 = T[-1] / a; e2 = np.cross(n, e1)
n0 = np.cross(C[0], C[1]); n0 /= np.linalg.norm(n0); f1 = C[0] / np.linalg.norm(C[0]); f2 = np.cross(n0, f1)
ang = np.linspace(0, 2 * np.pi, 300)
orb = a * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
ax.plot(*orb.T, color=TGT, lw=1.2, alpha=0.5, label="TOPEX orbit at arrival (1336 km)", zorder=2)
ap = np.linalg.norm(C[0])
orbp = ap * (np.outer(np.cos(ang), f1) + np.outer(np.sin(ang), f2))
ax.plot(*orbp.T, color=MUTED, lw=1.0, ls="--", alpha=0.6, label="Parking orbit (500 km)", zorder=2)
ax.plot(*C[xfer][::3].T, color=SC[0], lw=0.12, alpha=0.30, label=f"BHT-600 low-thrust spiral ({(tr.t.values[-1] - tr.t.values[np.argmax(xfer)]) / 86400:.1f} days)", zorder=3)
ax.scatter(*C[xfer][0], color=SC[0], s=50, zorder=4)
raan_off = float(re.search(r"RAAN / inc pre-offset\s+: ([\d.-]+)", open("summary.txt").read()).group(1))
ax.text2D(0.02, 0.97, f"spiral plane precesses under J2 over {(tr.t.values[-1] - tr.t.values[np.argmax(xfer)]) / 86400:.0f} days;\nparking RAAN pre-offset {raan_off:.0f}° so planes meet at arrival", transform=ax.transAxes, color=MUTED, fontsize=10, va="top")
ax.scatter(*C[-1], color=GOOD, s=70, zorder=4, label="Arrival: SC1 5 km behind TOPEX")
lim = 8000
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
ax.view_init(elev=22, azim=-60)
leg = ax.legend(loc="lower left", fontsize=11)
leg.get_lines()[2].set_linewidth(2.5); leg.get_lines()[2].set_alpha(1)
ax.set_title("Phase 1: transfer to TOPEX orbit (i = 66°)", color=INK)

t_h = tr.t.values / 86400
alt = np.linalg.norm(C, axis=1) - RE
lead = []
for i in range(len(tr)):
    c, tt = C[i], T[i]
    lead.append(np.degrees(np.arctan2(np.dot(np.cross(c, tt), n), np.dot(c, tt))))
lead = np.array(lead)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_h, alt, color=SC[0], lw=2)
ax2.axhline(1336, color=TGT, ls="--", lw=1, alpha=0.6)
ax2.axhline(500, color=MUTED, ls="--", lw=1, alpha=0.6)
tb = t_h[np.argmax(xfer)]
ax2.axvspan(0, tb, color=MUTED, alpha=0.12)
ax2.text(tb + 0.3, 640, f"phasing wait {float(re.search(r'Phasing wait in parking: ([\d.]+) h', open('summary.txt').read()).group(1)):.1f} h, then 39 mN over the last ~28–32% of each orbit", ha="left", color=MUTED, fontsize=11)
ax2.set_ylabel("SC altitude [km]"); ax2.grid(alpha=0.5)
ax2.set_title("Altitude and phase angle (low-thrust spiral)", color=INK)
ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)
ax3.plot(t_h, lead, color=SC[1], lw=2)
ax3.axvline(tb, color=SC[0], lw=1, ls=":")
ax3.set_ylabel("TOPEX lead angle [deg]"); ax3.set_xlabel("time from injection [days]"); ax3.grid(alpha=0.5)
save(fig, "1_transfer")

# ------------------------------------------------------------ 2. far-field rendezvous (in-plane)
fig, ax = plt.subplots(figsize=(15, 6.2))
tbid = ev[(ev.craft == 0) & ev.name.str.startswith("Hold: detect")].t1.iloc[0] / 3600
mask = th <= tbid
for k in range(3):
    ax.plot(px[f"y{k}"][mask] / 1e3, px[f"x{k}"][mask] / 1e3, color=SC[k], lw=2, label=NAMES[k])
    ax.scatter(px[f"y{k}"].iloc[0] / 1e3, px[f"x{k}"].iloc[0] / 1e3, color=SC[k], s=60, marker="s")
for y, k in zip([-1.0, -1.3, -1.6], range(3)):
    ax.scatter(y, 0, color=SC[k], s=90, marker="o", edgecolor=INK, zorder=5)
ax.scatter(0, 0, color=TGT, s=140, marker="*", zorder=6, label="TOPEX")
y0s = [px[f"y{k}"].iloc[0] / 1e3 for k in range(3)]; x0 = px["x0"].iloc[0] / 1e3
ax.text(np.mean(y0s), x0 - 0.35, f"thrust cutoff {min(y0s):+.1f} to {max(y0s):+.1f} km along-track, {-x0:.1f} km low (drift-free)", color=MUTED, ha="center", fontsize=11)
ax.text(-1.3, 0.12, "V-bar holds: detect / fuse / role bid", color=MUTED, ha="center", fontsize=11)
ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)
ax.set_xlabel("V-bar (along-track) [km]"); ax.set_ylabel("R-bar (radial, up) [km]")
ax.set_title("Phase 2a: CW-guided rendezvous to V-bar holds (LVLH, in-plane)", color=INK)
ax.set_ylim(min(-3.6, x0 - 0.6), max(0.5, max(px[f"x{k}"][mask].max() for k in range(3)) / 1e3 + 0.2)); ax.set_xlim(min(-6.5, min(y0s) - 1), max(0.6, max(y0s) + 1)); ax.set_aspect("equal"); ax.grid(alpha=0.5); ax.legend(loc="lower right", fontsize=11)
save(fig, "2_rendezvous_farfield")

# ------------------------------------------------------------ 3. inspection + docking 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection="3d", computed_zorder=False)
ax.set_facecolor(BG)
u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
ax.plot_wireframe(KOZ * np.cos(u) * np.sin(v), KOZ * np.sin(u) * np.sin(v), KOZ * np.cos(v),
                  color=RED, lw=0.4, alpha=0.35)
lo, hi = 2.0, 9.6
m = (th > lo) & (th < hi)
for k in range(3):
    xs, ys, zs = px[f"x{k}"][m], px[f"y{k}"][m], px[f"z{k}"][m]
    inside = np.sqrt(xs**2 + ys**2 + zs**2) < 260
    ax.plot(np.where(inside, ys, np.nan), np.where(inside, zs, np.nan), np.where(inside, xs, np.nan),
            color=SC[k], lw=1.8, label=NAMES[k])
ax.scatter(0, 0, 0, color=TGT, s=120, marker="*")
bk = bu[(bu.craft == 0) & bu.leg.str.contains("Glideslope|Hold 30|Final")]
ax.scatter(bk.t * 0 + [px.loc[(px.t - t).abs().idxmin(), "y0"] for t in bk.t], 0 * bk.t, 0 * bk.t,
           color=SC[0], s=30, edgecolor=INK, label="docker burns")
ax.set_xlim(-210, 120); ax.set_ylim(-120, 120); ax.set_zlim(-100, 100)
ax.set_box_aspect((330, 240, 200))
ax.set_xlabel("V-bar [m]"); ax.set_ylabel("H-bar (cross-track) [m]"); ax.set_zlabel("R-bar [m]")
ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
ax.view_init(elev=24, azim=-58)
ax.legend(loc="upper left", fontsize=11)
ax.set_title("Phase 2b: inspection NMCs + V-bar glideslope (red = 22 m keep-out)", color=INK)
save(fig, "3_inspection_docking_3d")

# ------------------------------------------------------------ 4. close-in V-bar / R-bar
fig, axs = plt.subplots(1, 2, figsize=(16, 5.4))
for ax, (cx, cy, xl, yl) in zip(axs, [("y", "x", "V-bar [m]", "R-bar (up) [m]"),
                                      ("y", "z", "V-bar [m]", "H-bar (cross-track) [m]")]):
    ax.add_patch(Circle((0, 0), KOZ, fill=True, color=RED, alpha=0.15))
    ax.add_patch(Circle((0, 0), KOZ, fill=False, color=RED, lw=1.2))
    ax.add_patch(Circle((0, 0), APPR, fill=False, color=MUTED, lw=1, ls="--"))
    for k in range(3):
        a_, b_ = px[f"{cx}{k}"][m], px[f"{cy}{k}"][m]
        ins = np.sqrt(px[f"x{k}"][m]**2 + px[f"y{k}"][m]**2 + px[f"z{k}"][m]**2) < 260
        ax.plot(np.where(ins, a_, np.nan), np.where(ins, b_, np.nan), color=SC[k], lw=1.6, label=NAMES[k])
    for yv in [-200, -120, -70, -45, -30]:
        ax.scatter(yv, 0, color=SC[0], s=35, edgecolor=INK, zorder=5)
    ax.scatter(0, 0, color=TGT, s=140, marker="*", zorder=6)
    ax.set_xlim(-230, 130); ax.set_ylim(-110, 110); ax.set_aspect("equal"); ax.grid(alpha=0.5)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
axs[0].text(-205, 14, "200 m hold\n(wait for GO)", color=SC[0], fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec="none", alpha=0.85))
axs[0].annotate("30 m hold → forced\napproach 3 cm/s", xy=(-30, 0), xytext=(-60, -78), color=SC[0], fontsize=10, ha="center", arrowprops=dict(arrowstyle="->", color=SC[0]), bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec="none", alpha=0.85))
axs[0].text(-225, -100, "approach sphere 200 m", color=MUTED, fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec="none", alpha=0.85))
axs[0].set_title("In-plane view", color=INK); axs[1].set_title("Top view", color=INK)
axs[0].legend(loc="upper right", fontsize=10)
fig.suptitle("Close-in proximity ops: imager NMCs b = 23 m (22 m standoff), capture at 2.6 m", color=INK,
             fontweight="bold", fontsize=15)
save(fig, "4_closein_views")

# ------------------------------------------------------------ 5. range timeline
fig, ax = plt.subplots(figsize=(15, 6))
for k in range(3):
    r = np.sqrt(px[f"x{k}"]**2 + px[f"y{k}"]**2 + px[f"z{k}"]**2)
    ax.semilogy(th, r, color=SC[k], lw=2, label=NAMES[k])
ax.axhline(KOZ, color=RED, ls="--", lw=1.2); ax.text(0.05, KOZ * 1.12, "keep-out 22 m", color=RED, fontsize=11)
ax.axhline(APPR, color=MUTED, ls="--", lw=1); ax.text(0.05, APPR * 1.12, "approach sphere 200 m", color=MUTED, fontsize=11)
e0 = ev[ev.craft == 0]
marks = [("V-bar holds", e0.t0[e0.name.str.startswith("Hold: detect")].iloc[0]),
         ("GO", e0.t0[e0.name.str.startswith("Glideslope")].iloc[0]),
         ("contact", e0.t0[e0.name.str.startswith("Docked")].iloc[0]),
         ("undock", e0.t0[e0.name.str.startswith("Undock")].iloc[0])]
for lab, tm in marks:
    ax.axvline(tm / 3600, color=MUTED, lw=0.8, ls=":")
    ax.text(tm / 3600 + 0.04, 4000, lab, color=INK, fontsize=11, rotation=90, va="top")
dk = e0[e0.name.str.startswith("Docked")].iloc[0]
ax.axvspan(dk.t0 / 3600, dk.t1 / 3600, color=GOOD, alpha=0.10)
ax.text((dk.t0 + dk.t1) / 7200, 6, "docked:\ndetumble + service", color=GOOD, ha="center", fontsize=11)
ax.set_ylim(2, 9000); ax.set_xlim(0, th.iloc[-1])
ax.set_xlabel("time from arrival [h]"); ax.set_ylabel("range to TOPEX [m]")
ax.set_title("Range to TOPEX through proximity operations", color=INK)
ax.grid(alpha=0.5, which="both"); ax.legend(loc="lower left", fontsize=11)
save(fig, "5_range_timeline")

# ------------------------------------------------------------ 6. dv budget
summ = open("summary.txt").read()
import re
cats = ["rendezvous", "inspection", "docking", "departure"]
vals = []
for k in range(3):
    line = [l for l in summ.splitlines() if l.startswith(f"SC{k+1}")][0]
    vals.append([float(re.search(rf"{c} ([\d.]+)", line).group(1)) for c in cats])
vals = np.array(vals)
sp = re.search(r"Sim \(J2\) spiral\s+: dv = ([\d.]+) m/s, spiral time = ([\d.]+) days, xenon used = ([\d.]+)", summ)
dv_tr = float(sp.group(1))
fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.6]})
ccol = ["#6fa0ff", "#c792ea", "#f2a33a", "#5fd07c"]
bottom = np.zeros(3)
for i, c in enumerate(cats):
    axs[1].bar(NAMES, vals[:, i], bottom=bottom, color=ccol[i], label=c, width=0.55)
    bottom += vals[:, i]
for i, tot in enumerate(bottom):
    axs[1].text(i, tot + 0.15, f"{tot:.2f} m/s", ha="center", color=INK, fontsize=12)
axs[1].set_ylabel("Δv [m/s]"); axs[1].set_ylim(0, bottom.max() * 1.4)
axs[1].set_title("Prox-ops Δv per spacecraft", color=INK); axs[1].legend(fontsize=11, ncol=4, loc="upper center"); axs[1].grid(alpha=0.4, axis="y")
axs[0].axis("off")
rp = re.search(r"Reported spiral\s+: dv = ([\d.]+) m/s .*xenon = ([\d.]+) kg", summ)
hyd = re.findall(r"SC\d hydrazine: ([\d.]+) of", summ)
rows = [["Spiral dv (Edelbaum x 1.02)", f"{float(rp.group(1)):.0f} m/s"],
        ["Hall dv capability (STM)", "1500 m/s"],
        ["Xenon used", f"{float(rp.group(2)):.2f} of 16 kg"],
        ["SC1 hydrazine, all uses", f"{float(hyd[0]):.2f} of 18 kg"],
        ["SC2 / SC3 hydrazine", f"{float(hyd[1]):.2f} / {float(hyd[2]):.2f} of 18 kg"]]
tb = axs[0].table(cellText=rows, colLabels=["Item", "Value"], loc="center", cellLoc="left", colWidths=[0.66, 0.44])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 2.1)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#2a3746"); cell.set_facecolor(PANEL if r else "#1a2633")
    cell.get_text().set_color(INK)
    if r in (3, 4): cell.get_text().set_fontweight("bold")
axs[0].set_title("Propellant per spacecraft", color=INK)
save(fig, "6_dv_budget")
print("figs written")
