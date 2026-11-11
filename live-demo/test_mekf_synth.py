"""Rate-estimate convergence on synthetic torque-free tumbles with noisy pose fixes (no vision): filter only.
Also times predict/update."""
import time, numpy as np, mekf, json
from scipy.spatial.transform import Rotation as Rot
J = np.diag([2.6e4, 1.1e4, 2.9e4])   # placeholder TOPEX-like inertia (kg m^2) -- shape of tumble only, not a TOPEX value
rng = np.random.default_rng(0)
out = []
for case, (w0, hz, sig, useJ) in enumerate([((0.0, 0.035, 0.0), 5, 1.5, False), ((0.02, 0.035, 0.01), 5, 1.5, False), ((0.02, 0.035, 0.01), 5, 1.5, True), ((0.05, 0.10, -0.04), 10, 2.0, False), ((0.05, 0.10, -0.04), 10, 2.0, True), ((0.05, 0.10, -0.04), 2, 2.0, True)]):
    R = np.eye(3); w = np.array(w0, float); f = mekf.TargetMEKF(sig_rot_meas=np.radians(sig), J=J if useJ else None)
    dt = 1 / 120; T = 60.0; k_meas = int(round(120 / hz)); hist = []
    for k in range(int(T / dt)):
        R, w = mekf.torque_free_step(R, w, J, dt); f.predict(dt)
        if k % k_meas == 0:
            Rm = Rot.from_rotvec(rng.normal(size=3) * np.radians(sig)).as_matrix() @ R
            f.update(Rm, np.array([0, 0, 35.0]) + rng.normal(size=3) * 0.15)
        if k % 12 == 0:
            hist.append((k * dt, np.degrees(np.linalg.norm(f.w - w)), np.degrees(np.arccos(np.clip((np.trace(mekf.q2R(f.q).T @ R) - 1) / 2, -1, 1)))))
    h = np.array(hist)
    def first_below(th):
        idx = np.where(h[:, 1] < th)[0]
        for i in idx:
            if (h[i:, 1] < th).all(): return float(h[i, 0])
        return None
    out.append(dict(euler_model=useJ, w0_rad_s=w0, w0_deg_s=float(np.degrees(np.linalg.norm(w0))), meas_hz=hz, meas_sig_deg=sig,
                    rate_err_deg_s_final_mean_last10s=float(h[h[:, 0] > 50, 1].mean()), att_err_deg_final_mean_last10s=float(h[h[:, 0] > 50, 2].mean()),
                    t_settle_rate_err_lt_0p5_deg_s=first_below(0.5), t_settle_lt_0p2=first_below(0.2)))
    np.savetxt(f"/tmp/mekf_case{case}.csv", h, delimiter=",", header="t,rate_err_deg_s,att_err_deg")
f = mekf.TargetMEKF(J=J); f.update(np.eye(3), np.zeros(3)); n = 20000
t0 = time.perf_counter(); [f.predict(1/120) for _ in range(n)]; tp = (time.perf_counter() - t0) / n
t0 = time.perf_counter()
for _ in range(n // 10): f.update(np.eye(3), np.zeros(3))
tu = (time.perf_counter() - t0) / (n // 10)
for o in out: print(json.dumps(o))
print(json.dumps(dict(predict_us=tp * 1e6, update_us=tu * 1e6)))
