"""Render the live-demo cutscene from the v2 animation (V2_CUTSCENE mode: no text/HUD, camera levels out at the end) and
export the exact end state the live sim starts from.
  python3 render_cutscene.py [--scale 0.6667] [--nw 7]
-> ../live-demo/assets/cutscene.mp4 (1280x720, 30 fps, v = 0 .. CUT_V) and ../live-demo/assets/v2_handoff.npz"""
import os, sys, subprocess, numpy as np
os.environ["V2_CUTSCENE"] = "1"; sys.argv = [sys.argv[0], "none"] + sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mission_anim as M
from multiprocessing import Pool
SCALE = M._opt("--scale", 2 / 3); NW = int(M._opt("--nw", 7))
LD = os.path.join(os.path.dirname(HERE), "live-demo", "assets")
N_END = int(round(M.CUT_V * M.FPS)) + 1          # frames 0 .. CUT_V inclusive

def chunk(args):
    a, b, out = args
    fig, ax = M.make_fig(); fig.set_dpi(100 * SCALE); w, h = fig.canvas.get_width_height()
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(M.FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(a, b):
        M.draw_frame(ax, i); fig.canvas.draw(); p.stdin.write(fig.canvas.buffer_rgba().tobytes())
    p.stdin.close(); p.wait(); return out

def export_handoff():
    i = N_END - 1; v = M.VV[i]; trel = M.TG[i] - M.T_ARR; cam = M.camera(i)
    Cv, _ = M.lvlh(M.TVIS[i]); sun = Cv @ M.S_ECI; stars = M.STARS @ Cv.T
    bu = M.bu[M.bu.dv > 0.02]; CVT = np.arange(-2400.0, 47801.0, 30.0)   # LVLH attitude table vs mission time, continued from the frame the cutscene ends on (visual clock TVIS there)
    np.savez_compressed(os.path.join(LD, "v2_handoff.npz"), v=v, trel=trel, cam_P=cam["P"], cam_f=cam["f"], cam_r=cam["r"], cam_u=cam["u"], cam_T=cam["T"], cam_d=cam["d"],
                        W=M.W, H=M.H, FPX=M.FPX, sun=sun, stars=stars, smag=M.SMAG, BG=M.hexc(M.BG), C_NEU=M.hexc(M.C_NEU), C_DOCK=M.hexc(M.C_DOCK), C_IMG=M.hexc(M.C_IMG),
                        C_TGT=M.hexc(M.C_TGT), C_RED=M.hexc(M.C_RED), MUTED=M.hexc(M.MUTED), REL_T=M.REL_T, REL0=M.REL[0], REL1=M.REL[1], REL2=M.REL[2],
                        burn_t=bu.t.values, burn_k=bu.craft.values.astype(int), burn_dv=bu.dv.values,
                        LK=M.LK, CUT_EL=M.CUT_EL, PHI_end=M.PHI[i], ROLL_end=M.ROLL[i], DOCK_CLOCK=M.DOCK_CLOCK,
                        ev_names=np.array(M.ev.name.values, dtype=str), ev_craft=M.ev.craft.values, ev_t0=M.ev.t0.values, ev_t1=M.ev.t1.values,
                        CV_T=CVT, CV=np.array([M.lvlh(M.TVIS[i] + t - trel)[0] for t in CVT]), S_ECI=M.S_ECI, STARS_ECI=M.STARS,
                        P_HOLD=M.P_HOLD, P_BID=M.P_BID, P_GO=M.P_GO, P_H30=M.P_H30, P_FA=M.P_FA, P_CON=M.P_CON, DOCK_R=M.DOCK_R)
    print("handoff v=%.3f trel=%.1f cam d=%.1f" % (v, trel, cam["d"]), [np.round(M.rel(k, trel), 1).tolist() for k in range(3)])

if __name__ == "__main__":
    export_handoff()
    if "--export-only" in sys.argv: sys.exit(0)
    edges = np.linspace(0, N_END, NW * 2 + 1).astype(int); tmp = "/tmp/cutscene_chunks"; os.makedirs(tmp, exist_ok=True)
    jobs = [(edges[j], edges[j + 1], os.path.join(tmp, f"c{j:02d}.mp4")) for j in range(len(edges) - 1)]
    with Pool(NW) as pool: outs = pool.map(chunk, jobs)
    lst = os.path.join(tmp, "list.txt"); open(lst, "w").write("".join(f"file '{o}'\n" for o in outs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart",
                    os.path.join(LD, "cutscene.mp4")], check=True)
    print("wrote", os.path.join(LD, "cutscene.mp4"), N_END, "frames")
