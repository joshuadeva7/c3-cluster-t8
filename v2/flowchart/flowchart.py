"""SWARM autonomy flowchart (v2). Edit ROWS / SIDE, then: python3 flowchart.py -> flowchart.png, flowchart.mmd"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# main path, one list per row; '?' marks a decision (yes = continue along the row)
ROWS = [
    ["Start", "Healthy?", "Transfer to TOPEX vicinity", "Search predicted position (visual + thermal IR)",
     "Detected?", "Sensor OK?", "Cluster agrees within tol?", "Target located"],
    ["Assign roles: 1 docker, 2 imagers", "Imagers: stereo + IR survey at 22 m", "Estimate spin state",
     "Spin axis (max-inertia, H) within tol?", "Docker hold at 200 m", "Glideslope to 30 m hold on spin axis",
     "Arm open (1 of 4)"],
    ["Wheel spin-up to TOPEX rate, hydrazine tops up", "Synced within tol?", "Final approach ~3 cm/s along axis",
     "Arm close (2 of 4): captured?", "Hydrazine despin, 0.5 N·m", "Loads and fuel OK?",
     "|ω| ≤ 0.00175 rad/s verified (MR-4)?"],
    ["Stable hold", "Arm open (3 of 4), release", "Back out to 30 m", "Arm close (4 of 4)", "Regroup, next target"],
]
# 'no' branches: decision -> (action, returns to)
SIDE = {
    "Healthy?": ("Safe mode", None),
    "Detected?": ("Widen search / request new ephemeris", "Search predicted position (visual + thermal IR)"),
    "Sensor OK?": ("Switch to backup sensor or teammate data", "Search predicted position (visual + thermal IR)"),
    "Cluster agrees within tol?": ("More measurements", "Cluster agrees within tol?"),
    "Spin axis (max-inertia, H) within tol?": ("More imaging", "Estimate spin state"),
    "Synced within tol?": ("Keep syncing", "Synced within tol?"),
    "Arm close (2 of 4): captured?": ("Arm open, back off to 30 m, retry", "Wheel spin-up to TOPEX rate, hydrazine tops up"),
    "Loads and fuel OK?": ("Lower torque / call 2nd craft", "Hydrazine despin, 0.5 N·m"),
    "|ω| ≤ 0.00175 rad/s verified (MR-4)?": ("Keep despinning", "Hydrazine despin, 0.5 N·m"),
}

DX, DY, BW, BH = 2.3, 3.0, 1.9, 0.9
pos = {}
for r, row in enumerate(ROWS):
    for c, n in enumerate(row):
        pos[n] = (c * DX, -r * DY)

fig, ax = plt.subplots(figsize=(19, 11.5))
ax.set_aspect("equal"); ax.axis("off")
FS = 8.5


def node(n, x, y, kind):
    t = "\n".join(textwrap.wrap(n, 20))
    if kind == "dec":
        ax.fill([x - BW / 2, x, x + BW / 2, x], [y, y + BH * 0.62, y, y - BH * 0.62], fc="white", ec="black", lw=0.9)
    else:
        ax.add_patch(plt.Rectangle((x - BW / 2, y - BH / 2), BW, BH, fc="white", ec="black", lw=0.9 if kind == "act" else 1.6))
    ax.text(x, y, t, ha="center", va="center", fontsize=FS)


def poly(pts):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color="black", lw=0.8)
    arrow(pts[-2], pts[-1])


def arrow(p, q, lab=None, rad=0.0):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", shrinkA=0, shrinkB=0,
                                          connectionstyle=f"arc3,rad={rad}"))
    if lab: ax.text((p[0] + q[0]) / 2 + 0.08, (p[1] + q[1]) / 2 + 0.08, lab, fontsize=7.5)


ends = {"Start", "Target located", "Stable hold", "Regroup, next target"}
for n, (x, y) in pos.items():
    node(n, x, y, "dec" if n.endswith("?") else ("end" if n in ends else "act"))
seq = [n for row in ROWS for n in row]
for a, b in zip(seq, seq[1:]):
    (xa, ya), (xb, yb) = pos[a], pos[b]
    if ya == yb:
        arrow((xa + BW / 2, ya), (xb - BW / 2, yb), "yes" if a.endswith("?") else None)
    else:   # wrap to next row: right, down, left, down
        yr = ya - DY + BH / 2 + 0.3
        poly([(xa + BW / 2, ya), (xa + BW / 2 + 0.3, ya), (xa + BW / 2 + 0.3, yr), (xb, yr), (xb, yb + BH / 2)])
for d, (act, back) in SIDE.items():
    x, y = pos[d]
    sx, sy = x, y - 1.45
    node(act, sx, sy, "act")
    arrow((x, y - BH * 0.62), (sx, sy + BH / 2), "no")
    if back:
        bx, by = pos[back]
        if back == d:
            arrow((sx + BW / 2, sy), (x + BW / 2 - 0.2, y - 0.15), None, rad=0.5)
        else:   # under the side box, back to the target's bottom
            yu = sy - BH / 2 - 0.25
            poly([(sx, sy - BH / 2), (sx, yu), (bx, yu), (bx, by - BH / 2)])
ax.set_xlim(-1.3, 7 * DX + 1.4); ax.set_ylim(-3 * DY - 1.2, 1.0)
ax.set_title("SWARM autonomy flow", fontsize=12, loc="left")
fig.savefig("flowchart.png", dpi=160, bbox_inches="tight", facecolor="white")

# mermaid source of the same graph
ids = {n: f"n{i}" for i, n in enumerate(seq + [v[0] for v in SIDE.values()])}
q = lambda n: f'{ids[n]}{{"{n}"}}' if n.endswith("?") else f'{ids[n]}["{n}"]'
L = ["flowchart LR"]
for a, b in zip(seq, seq[1:]):
    L.append(f"  {q(a)} -->{'|yes|' if a.endswith('?') else ''} {q(b)}")
for d, (act, back) in SIDE.items():
    L.append(f"  {q(d)} -->|no| {q(act)}")
    if back: L.append(f"  {q(act)} --> {q(back)}")
open("flowchart.mmd", "w").write("\n".join(L) + "\n")
print("wrote flowchart.png flowchart.mmd")
