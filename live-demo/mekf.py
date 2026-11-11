"""Continuous TOPEX state from intermittent pose fixes.
Attitude: multiplicative EKF (MEKF) on q_IB (TOPEX body -> inertial/room frame) with body rate w in the state,
constant-rate kinematic model + rate random walk (handles torque-free nutation through process noise).
Translation: constant-velocity KF on TOPEX position in the room frame (CW terms are negligible over a
demo-length prox-ops window at 25-45 m; can be added in F).
Each camera's (R_cb, t_cb) fix is mapped to the room frame with that camera's known pose before update,
so any number of cameras fuse by sequential updates. predict(dt) can be called at any rate (e.g. 60-240 Hz)."""
import numpy as np

def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def qexp(th):
    a = np.linalg.norm(th)
    if a < 1e-12: return np.array([1.0, *(0.5 * th)])
    return np.r_[np.cos(a / 2), np.sin(a / 2) * th / a]
def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1); return np.array([0.25 * s, (R[2,1]-R[1,2]) / s, (R[0,2]-R[2,0]) / s, (R[1,0]-R[0,1]) / s])
    i = np.argmax(np.diag(R))
    if i == 0: s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); q = [(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
    elif i == 1: s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); q = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]
    else: s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); q = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]
    q = np.array(q); return q / np.linalg.norm(q)

class TargetMEKF:
    def __init__(self, sig_w_rw=0.002, sig_a_rw=0.01, sig_rot_meas=np.radians(1.5), sig_pos_meas=0.15, gate_deg=25.0, J=None):
        self.J = J; self.Ji = np.linalg.inv(J) if J is not None else None
        self.q = None; self.w = np.zeros(3); self.P = np.diag([0.3]*3 + [0.3]*3) ** 2
        self.p = np.zeros(3); self.v = np.zeros(3); self.Pp = np.diag([5.0]*3 + [1.0]*3) ** 2
        self.qw, self.qa = sig_w_rw**2, sig_a_rw**2; self.Rr = sig_rot_meas**2; self.Rp = sig_pos_meas**2; self.gate = np.radians(gate_deg)

    def predict(self, dt):
        if self.q is None: return
        w = self.w; F = np.eye(6); F[:3, :3] = np.eye(3) - skew(w) * dt; F[:3, 3:] = np.eye(3) * dt
        if self.J is not None:   # torque-free Euler dynamics from CAD mass properties
            Jw = self.J @ w; wd = self.Ji @ (-np.cross(w, Jw))
            F[3:, 3:] = np.eye(3) + self.Ji @ (-skew(w) @ self.J + skew(Jw)) * dt
            w_mid = w + 0.5 * wd * dt; self.w = w + self.Ji @ (-np.cross(w_mid, self.J @ w_mid)) * dt
        self.q = qmul(self.q, qexp(0.5 * (w + self.w) * dt)); self.q /= np.linalg.norm(self.q)
        Q = np.zeros((6, 6)); Q[3:, 3:] = np.eye(3) * self.qw * dt; Q[:3, :3] = np.eye(3) * self.qw * dt**3 / 3
        self.P = F @ self.P @ F.T + Q
        self.p = self.p + self.v * dt
        Fp = np.eye(6); Fp[:3, 3:] = np.eye(3) * dt
        Qp = np.zeros((6, 6)); Qp[3:, 3:] = np.eye(3) * self.qa * dt; Qp[:3, :3] = np.eye(3) * self.qa * dt**3 / 3
        self.Pp = Fp @ self.Pp @ Fp.T + Qp

    def update(self, R_IB_meas, p_I_meas, rot_sig=None, pos_sig=None):
        """returns False if the fix was rejected by the innovation gate."""
        qm = R2q(R_IB_meas)
        if self.q is None:
            self.q = qm; self.p = p_I_meas.copy(); return True
        dq = qmul(np.r_[self.q[0], -self.q[1:]], qm)
        if dq[0] < 0: dq = -dq
        r = 2 * dq[1:]
        if np.linalg.norm(r) > self.gate: return False
        Rr = (rot_sig**2 if rot_sig else self.Rr) * np.eye(3)
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        S = H @ self.P @ H.T + Rr; Kg = self.P @ H.T @ np.linalg.inv(S); dx = Kg @ r
        self.q = qmul(self.q, qexp(dx[:3])); self.q /= np.linalg.norm(self.q); self.w = self.w + dx[3:]
        self.P = (np.eye(6) - Kg @ H) @ self.P
        Rp = (pos_sig**2 if pos_sig else self.Rp) * np.eye(3); Hp = np.zeros((3, 6)); Hp[:, :3] = np.eye(3)
        S = Hp @ self.Pp @ Hp.T + Rp; Kp = self.Pp @ Hp.T @ np.linalg.inv(S)
        y = p_I_meas - self.p; d = Kp @ y; self.p += d[:3]; self.v += d[3:]; self.Pp = (np.eye(6) - Kp @ Hp) @ self.Pp
        return True

    def state(self):
        """continuous state vector: [p_I(3), v_I(3), q_IB(4), w_B(3)]"""
        return np.r_[self.p, self.v, self.q if self.q is not None else np.r_[1, 0, 0, 0], self.w]

    def pose_in_camera(self, R_IC, p_IC):
        """prior for the tracker: TOPEX body->camera given camera pose (camera->room R_IC, position p_IC)."""
        R_IB = q2R(self.q); return R_IC.T @ R_IB, R_IC.T @ (self.p - p_IC)

def relative_rate_body(w_target_B, R_IB, w_docker_D, R_ID):
    """target rate relative to the docker, in target body axes; ~0 once the docker matches the spin."""
    return w_target_B - R_IB.T @ R_ID @ w_docker_D

def torque_free_step(R, w, J, dt, n=4):
    """truth propagator: Euler's equations + attitude, RK4 substeps."""
    h = dt / n; Ji = np.linalg.inv(J)
    f = lambda w: Ji @ (-np.cross(w, J @ w))
    for _ in range(n):
        k1 = f(w); k2 = f(w + h/2*k1); k3 = f(w + h/2*k2); k4 = f(w + h*k3)
        wm = w + h/6*(k1 + 2*k2 + 2*k3 + k4)
        th = 0.5 * (w + wm) * h; a = np.linalg.norm(th)
        if a > 1e-12:
            k = th / a; Kx = skew(k); R = R @ (np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx)
        w = wm
    return R, w
