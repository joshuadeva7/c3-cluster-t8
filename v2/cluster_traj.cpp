// cluster_traj.cpp -- C3-Cluster trajectory sim: 3 homogeneous spacecraft
// rendezvous with TOPEX/Poseidon and run proximity operations.
//
// Truth model : two-body + J2, RK4, all bodies propagated in ECI.
// Guidance    : Phase 1  Low-thrust tangential spiral 500 km -> TOPEX altitude on the
//                         Busek BHT-600 Hall thruster (39 mN, Isp 1500 s, 600 W, 166.407 kg wet).
//                         Power-limited: fires only over the last x of each orbit, x = orbit energy
//                         balance at beta = 0 (27.5% at 500 km, TRR budgets 'Spiral Orbit Cycle' B9),
//                         arc alternated 180 deg each rev so eccentricity does not build up.
//                         Phasing wait, burn duration and parking-orbit RAAN/inc offsets are
//                         found by Newton shooting on the J2 truth so the cluster cuts off
//                         5 km behind TOPEX on V-bar with zero relative drift and no
//                         cross-track error (pre-biases the differential J2 drift over the spiral).
//                         Hall thruster is off for all of Phase 2 (STM row 10).
//               Phase 2  Proximity ops (hydrazine, 12x ArianeGroup 1 N, Isp 220 s) in TOPEX LVLH (x radial, y along-track,
//                         z cross-track). Clohessy-Wiltshire two-impulse targeting,
//                         re-targeted from the true state (midcourse corrections),
//                         natural motion circumnavigation (NMC) for imagers,
//                         V-bar glideslope + forced-motion final approach for docker.
// Navigation  : perfect relative state knowledge (nav errors not modelled here).
//
// build: g++ -O2 -std=c++17 cluster_traj.cpp -o cluster_traj
// run  : ./cluster_traj            -> transfer.csv prox.csv burns.csv events.csv summary.txt

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ------------------------------------------------------------------ constants
const double MU = 398600.4418e9;      // m^3/s^2
const double RE = 6378.137e3;         // m
const double J2 = 1.08262668e-3;
const double PI = 3.14159265358979323846;
const double D2R = PI / 180.0;

// mission parameters
const double H_PARK   = 500e3;        // parking orbit altitude [m]
const double H_TOPEX  = 1336e3;       // TOPEX altitude [m]
const double INC      = 66.04 * D2R;  // TOPEX inclination
const double LEAD0    = 100.0 * D2R;  // TOPEX leads cluster by this at t=0 (arbitrary injection)
const double SEP0     = 1000.0;       // initial along-track spacing guess of the 3 craft in parking orbit [m]
const double AIM_YK[3] = {-5000.0, -6000.0, -7000.0};  // arrival V-bar slots: all behind TOPEX, in hold order [m]
double LEADK[3] = {0, 0, 0};          // extra injection lag per craft [rad], shot so each lands on its slot
const double AIM_Y    = -5000.0;      // aim point after transfer: 5 km behind on V-bar [m]
const double KOZ      = 22.0;         // TOPEX keep-out / precision standoff (STM rows 1, 10) [m]
const double APPROACH_SPHERE = 200.0; // approach ellipsoid / sphere [m]
const double DOCK_R   = 3.9;          // docker CoM to TOPEX CoM at capture: TOPEX aft hardware reaches 3.32 m (topex.stl) + SWARM gripper base 0.50 m ahead of its bus centre (SWARM Assembly.SLDASM) [m]
// vehicle / propulsion (midterm report Section 4, STM)
const double M_WET    = 166.407;      // wet mass per spacecraft [kg] (TRR budgets 'Mass Budget' E3+E7+E12+E16+E21+E31+E35+E38)
const double F_HALL   = 0.039;        // BHT-600 thrust [N]
const double ISP_HALL = 1500.0;       // [s]
// orbit energy balance (TRR budgets 'Spiral Orbit Cycle' B6:B9)
const double P_GEN    = 405.042;      // sunlit array output [W]
const double P_BUS    = 87.07;        // bus draw excl. Hall [W]
const double P_HALL   = 600.0;        // BHT-600 [W]
const double ISP_HYD  = 220.0;        // ArianeGroup 1 N hydrazine [s]
const double M_HYD    = 18.0;         // hydrazine load [kg]
const double M_XE     = 16.0;         // xenon load [kg]
const double DV_HALL  = 1500.0;      // Hall (xenon) dv capability, 3 targets + 3 orbit moves (midterm STM, BHT-600 rows)
const double STEER    = 1.02;         // duty-arc / steering loss on Edelbaum (trr/SPIRAL-POWER.md s1)
const double G0       = 9.80665;
// detumble (STM rows 8, 9; Section 4)
const double IZ_TOPEX = 11665.0;      // spin-axis MOI [kg m^2]
const double W0_SPIN  = 2 * 3.14159265358979323846 / 9.5;  // 9.5 s period -> 0.661 rad/s
const double W_BAND   = 0.00175;      // MR-4
const double F_RCS = 1.0, ARM_RCS = 0.5;   // one 1 N thruster on a 0.5 m arm -> 0.5 N.m
const double TAU_DT   = F_RCS * ARM_RCS;
const double T_DETUMBLE = IZ_TOPEX * (W0_SPIN - W_BAND) / TAU_DT;  // constant torque into the band
const double T_HOLD     = 3600.0;     // MR-4 stable hold after despin [s]
// docker spin-up to TOPEX rate before capture: RW8 wheel first, hydrazine tops up the rest
const double IX_SWARM = 58.88;        // arm open, about the arm-face axis (TRR budgets 'Pointing Budget' H11)
const double H_RW     = 8.0;          // RW8 momentum storage (TRR budgets 'Pointing Budget' H9)
const int NC = 3;

// ------------------------------------------------------------------ vectors
struct V3 { double x, y, z; };
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
V3 operator*(double s, V3 a) { return a * s; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double norm(V3 a) { return std::sqrt(dot(a, a)); }
V3 unit(V3 a) { return a * (1.0 / norm(a)); }

struct M3 { double m[3][3]; };
V3 mul(const M3& A, V3 v) {
  return {A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
          A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
          A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z};
}
M3 inv(const M3& A) {
  const double (*a)[3] = A.m;
  double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
             - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
             + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  M3 B;
  B.m[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  B.m[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
  B.m[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  B.m[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  B.m[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
  B.m[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  B.m[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  B.m[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
  B.m[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  return B;
}

// ------------------------------------------------------------------ dynamics
struct State { V3 r, v; };

V3 accel(V3 r) {
  double r2 = dot(r, r), rn = std::sqrt(r2);
  double z2 = r.z * r.z / r2;
  double k = -MU / (r2 * rn);
  double f = 1.5 * J2 * MU * RE * RE / (r2 * r2 * rn);
  return {k * r.x + f * r.x * (5 * z2 - 1),
          k * r.y + f * r.y * (5 * z2 - 1),
          k * r.z + f * r.z * (5 * z2 - 3)};
}

State rk4(State s, double dt, V3 aext) {
  auto d = [&](State q) { return State{q.v, accel(q.r) + aext}; };
  State k1 = d(s);
  State k2 = d({s.r + k1.r * (dt / 2), s.v + k1.v * (dt / 2)});
  State k3 = d({s.r + k2.r * (dt / 2), s.v + k2.v * (dt / 2)});
  State k4 = d({s.r + k3.r * dt, s.v + k3.v * dt});
  return {s.r + (k1.r + 2 * k2.r + 2 * k3.r + k4.r) * (dt / 6),
          s.v + (k1.v + 2 * k2.v + 2 * k3.v + k4.v) * (dt / 6)};
}

State circular(double a, double raan, double inc, double u) {
  V3 N{std::cos(raan), std::sin(raan), 0};
  V3 M{-std::sin(raan) * std::cos(inc), std::cos(raan) * std::cos(inc), std::sin(inc)};
  // speed balancing the orbit-averaged J2 radial pull, so the orbit starts near-circular under J2
  double vc = std::sqrt(MU / a * (1 + 1.5 * J2 * (RE / a) * (RE / a) * (1 - 1.5 * std::sin(inc) * std::sin(inc))));
  return {(N * std::cos(u) + M * std::sin(u)) * a, (N * (-std::sin(u)) + M * std::cos(u)) * vc};
}

double raan_of(const State& s) { V3 h = cross(s.r, s.v); return std::atan2(h.x, -h.y); }
double wrap(double a) { while (a > PI) a -= 2 * PI; while (a < -PI) a += 2 * PI; return a; }

// ------------------------------------------------------------------ LVLH
struct Frame { V3 xh, yh, zh; double w; };
Frame lvlh(const State& t) {
  Frame F;
  V3 h = cross(t.r, t.v);
  F.xh = unit(t.r); F.zh = unit(h); F.yh = cross(F.zh, F.xh);
  F.w = norm(h) / dot(t.r, t.r);
  return F;
}
V3 toL(const Frame& F, V3 a) { return {dot(F.xh, a), dot(F.yh, a), dot(F.zh, a)}; }
V3 fromL(const Frame& F, V3 l) { return F.xh * l.x + F.yh * l.y + F.zh * l.z; }
void rel(const State& t, const State& c, V3& rho, V3& rhod) {
  Frame F = lvlh(t);
  rho = toL(F, c.r - t.r);
  V3 dv = toL(F, c.v - t.v);
  rhod = {dv.x + F.w * rho.y, dv.y - F.w * rho.x, dv.z};
}
State from_rel(const State& t, V3 rho, V3 rhod) {
  Frame F = lvlh(t);
  V3 dvl{rhod.x - F.w * rho.y, rhod.y + F.w * rho.x, rhod.z};
  return {t.r + fromL(F, rho), t.v + fromL(F, dvl)};
}

// ------------------------------------------------------------------ CW
void cw_blocks(double n, double T, M3& RR, M3& RV) {
  double s = std::sin(n * T), c = std::cos(n * T), nt = n * T;
  RR = {{{4 - 3 * c, 0, 0}, {6 * (s - nt), 1, 0}, {0, 0, c}}};
  RV = {{{s / n, 2 * (1 - c) / n, 0}, {2 * (c - 1) / n, (4 * s - 3 * nt) / n, 0}, {0, 0, s / n}}};
}
V3 cw_v0(V3 r0, V3 rf, double n, double T) {
  M3 RR, RV; cw_blocks(n, T, RR, RV);
  return mul(inv(RV), rf - mul(RR, r0));
}

// ------------------------------------------------------------------ logging
struct Burn { double t; int craft; V3 dvL; double dv; std::string leg, cat; double range, min_range_missed; };
std::vector<Burn> burns;

// =================================================================== PHASE 1
struct TransferResult {
  State tgt; State c[NC]; double t_arrive, t_on, dv_spiral, m_end;
  std::vector<std::vector<double>> log;
};

// Hall on-fraction of an orbit at radius r: (P_GEN (1 - f_ecl) - P_BUS) / P_HALL, cylindrical shadow, beta = 0
double duty(double r) {
  double h = r - RE, fe = std::acos(std::sqrt(h * h + 2 * RE * h) / r) / PI;
  return (P_GEN * (1 - fe) - P_BUS) / P_HALL;
}

// fraction of the swept angle [a, b] that lies on a thrust arc: the last x of even revs, shifted by pi on odd revs
double on_frac(double a, double b, double x) {
  double in = 0;
  for (int j = (int)(a / (2 * PI)); j <= (int)(b / (2 * PI)); ++j) {
    double lo = 2 * PI * j + ((j & 1) ? PI * (1 - 2 * x) : 2 * PI * (1 - x));
    double hi = 2 * PI * j + ((j & 1) ? PI : 2 * PI);
    in += std::max(0.0, std::min(b, hi) - std::max(a, lo));
  }
  return b > a ? in / (b - a) : 0.0;
}

// coast in the parking orbit for `wait`, then spiral: thrust tangentially over the last DUTY of each
// orbit until SC1 has accumulated `tburn` of Hall on-time; arrival = cutoff
TransferResult sim_transfer(double raan_off, double wait, double tburn, double dinc, bool dolog, int ncraft = NC) {
  TransferResult R;
  double at = RE + H_TOPEX, ac = RE + H_PARK;
  State T = circular(at, 0.0, INC, 0.0);
  State C[NC]; double m[NC];
  for (int k = 0; k < NC; ++k) { C[k] = circular(ac, raan_off, INC + dinc, -LEAD0 - LEADK[k]); m[k] = M_WET; }
  const double mdot = F_HALL / (ISP_HALL * G0);
  double t = 0.0, dvs = 0.0, last_log = -1e18, ton = 0.0;
  double th = 0;   // angle SC1 has travelled since the spiral started
  while (ton < tburn - 1e-9) {
    bool spiral = t >= wait - 1e-9;
    double dt = 10.0, fire[NC];
    if (!spiral && t + dt > wait) dt = wait - t;
    {   // the cluster fires on SC1's arc schedule
      double rn = norm(C[0].r), w = norm(cross(C[0].r, C[0].v)) / (rn * rn);
      for (int k = 0; k < NC; ++k) fire[k] = spiral ? on_frac(th, th + w * dt, duty(rn)) : 0.0;
    }
    bool on = fire[0] > 0;
    if (on && ton + fire[0] * dt > tburn) dt = (tburn - ton) / fire[0];
    if (dolog) {
      double every = (ton > tburn - 1800) ? 10.0 : 60.0;
      if (t - last_log >= every - 1e-9) {
        std::vector<double> row{t, T.r.x, T.r.y, T.r.z};
        for (int k = 0; k < NC; ++k) { row.push_back(C[k].r.x); row.push_back(C[k].r.y); row.push_back(C[k].r.z); }
        row.push_back(on ? 1.0 : 0.0); row.push_back(dvs); row.push_back(m[0]);
        R.log.push_back(row); last_log = t;
      }
    }
    T = rk4(T, dt, {0, 0, 0});
    for (int k = 0; k < ncraft; ++k) {
      V3 a{0, 0, 0};
      if (fire[k] > 0) a = unit(C[k].v) * (fire[k] * F_HALL / m[k]);
      V3 r0 = C[k].r;
      C[k] = rk4(C[k], dt, a);
      if (spiral && k == 0) th += std::acos(std::min(1.0, dot(unit(r0), unit(C[k].r))));
      if (fire[k] > 0) { if (k == 0) dvs += fire[k] * F_HALL / m[k] * dt; m[k] -= fire[k] * mdot * dt; }
    }
    ton += fire[0] * dt;
    t += dt;
  }
  R.tgt = T; for (int k = 0; k < NC; ++k) R.c[k] = C[k];
  R.t_arrive = t; R.t_on = wait; R.dv_spiral = dvs; R.m_end = m[0];
  if (dolog) {
    std::vector<double> row{t, T.r.x, T.r.y, T.r.z};
    for (int k = 0; k < NC; ++k) { row.push_back(C[k].r.x); row.push_back(C[k].r.y); row.push_back(C[k].r.z); }
    row.push_back(2.0); row.push_back(dvs); row.push_back(m[0]);
    R.log.push_back(row);
  }
  return R;
}

double arrival_y(const TransferResult& R) {
  V3 rho, rhod; rel(R.tgt, R.c[0], rho, rhod); return rho.y;
}

// =================================================================== PHASE 2
enum LegType { XFER, HOLD, NMC, FORCED, DOCKED };
struct Leg {
  LegType type; double t0, t1; std::string name, cat;
  V3 p{0, 0, 0};                         // XFER/HOLD aim point
  double b = 0, c = 0, alpha = 0, tep = 0; // NMC: x=b cos th, y=-2b sin th, z=c sin th, th=n(t-tep)+alpha
  bool aim_nmc = false;                  // XFER whose aim point is on an NMC
  V3 pa{0, 0, 0}, pb{0, 0, 0}; double speed = 0; // FORCED straight line
};

struct Craft {
  State s; std::vector<Leg> legs; int li = 0;
  double next_rt = -1, dv_imp = 0, dv_cont = 0, dv_cat[4] = {0, 0, 0, 0};
  bool entered = false; V3 lock{0, 0, 0};
};

double N_T;   // TOPEX mean motion (set at runtime)

void nmc_point(const Leg& L, double t, V3& p, V3& v) {
  double th = N_T * (t - L.tep) + L.alpha;
  p = {L.b * std::cos(th), -2 * L.b * std::sin(th), L.c * std::sin(th)};
  v = {-L.b * N_T * std::sin(th), -2 * L.b * N_T * std::cos(th), L.c * N_T * std::cos(th)};
}

int catidx(const std::string& c) {
  if (c == "rendezvous") return 0; if (c == "inspection") return 1;
  if (c == "docking") return 2; return 3;
}

double passive_min_range(State tgt, State c, double horizon) {
  double mn = 1e18, dt = 5.0;
  for (double t = 0; t < horizon; t += dt) {
    mn = std::min(mn, norm(c.r - tgt.r));
    tgt = rk4(tgt, dt, {0, 0, 0}); c = rk4(c, dt, {0, 0, 0});
  }
  return mn;
}

void apply_burn(Craft& C, int k, const State& tgt, double t, V3 dvL, const Leg& L) {
  double m = norm(dvL);
  if (m < 1e-5) return;
  Frame F = lvlh(tgt);
  C.s.v = C.s.v + fromL(F, dvL);
  C.dv_imp += m; C.dv_cat[catidx(L.cat)] += m;
  double range = norm(C.s.r - tgt.r);
  double mr = range < 3000 ? passive_min_range(tgt, C.s, 2 * 2 * PI / N_T) : -1;
  burns.push_back({t, k, dvL, m, L.name, L.cat, range, mr});
}

int main() {
  // ---------------- Phase 1: shoot the low-thrust spiral
  double at = RE + H_TOPEX, ac = RE + H_PARK;
  double nt = std::sqrt(MU / (at * at * at)), nc = std::sqrt(MU / (ac * ac * ac));
  N_T = nt;
  const double mdot = F_HALL / (ISP_HALL * G0);
  double dv_edel = std::sqrt(MU / ac) - std::sqrt(MU / at);           // Edelbaum, coplanar circle-to-circle
  double tb0 = M_WET * (1 - std::exp(-dv_edel / (ISP_HALL * G0))) / mdot;
  auto plane_err = [&](const TransferResult& R, double& dO, double& di) {
    V3 ht = unit(cross(R.tgt.r, R.tgt.v)), hc = unit(cross(R.c[0].r, R.c[0].v));
    dO = wrap(std::atan2(hc.x, -hc.y) - std::atan2(ht.x, -ht.y));
    di = std::acos(hc.z) - std::acos(ht.z);
  };
  auto phase_err = [&](const TransferResult& R) {
    V3 ht = unit(cross(R.tgt.r, R.tgt.v));
    return std::atan2(dot(cross(R.tgt.r, R.c[0].r), ht), dot(R.tgt.r, R.c[0].r));   // + = cluster ahead
  };
  // initial guesses: burn length from Edelbaum, plane offsets by cancelling the measured drift,
  // wait from the phase error (parking orbit gains (nc - nt) rad/s on TOPEX)
  double raan_off = 0, dinc = 0, tburn = tb0, wait = 0;
  for (int it = 0; it < 3; ++it) {
    TransferResult R = sim_transfer(raan_off, 0.0, tburn, dinc, false, 1);
    double dO, di; plane_err(R, dO, di); raan_off -= dO; dinc -= di;
  }
  {
    TransferResult R = sim_transfer(raan_off, 0.0, tburn, dinc, false, 1);
    double need = AIM_Y / at - phase_err(R);
    while (need < 0) need += 2 * PI;
    wait = need / (nc - nt);
  }
  // Shooting on the J2 truth, repeated until all errors < 1 m:
  //   (a) burn length + wait -> zero relative drift and along-track aim point (2x2 Newton; under duty
  //       cycling the cutoff phase moves with both, so they are solved together),
  //   (b) plane offsets -> cancel z / zdot.
  auto errs = [&](double* e) {
    TransferResult R = sim_transfer(raan_off, wait, tburn, dinc, false, 1);
    V3 rho, rhod; rel(R.tgt, R.c[0], rho, rhod);
    e[0] = rho.y - AIM_Y; e[1] = (rhod.y + 2 * nt * rho.x) / nt; e[2] = rho.z; e[3] = rhod.z / nt;
    return R;
  };
  double e[4];
  for (int it = 0; it < 40; ++it) {
    // (a) burn length + wait: Newton with finite-difference Jacobian, step-limited
    for (int s = 0; s < 4; ++s) {
      double e0[4], eb[4], ew[4];
      errs(e0);
      tburn += 20.0; errs(eb); tburn -= 20.0;
      wait += 5.0; errs(ew); wait -= 5.0;
      double a11 = (eb[0] - e0[0]) / 20, a21 = (eb[1] - e0[1]) / 20, a12 = (ew[0] - e0[0]) / 5, a22 = (ew[1] - e0[1]) / 5;
      double det = a11 * a22 - a12 * a21;
      double dtb = -(a22 * e0[0] - a12 * e0[1]) / det, dw = -(-a21 * e0[0] + a11 * e0[1]) / det;
      double n0 = std::hypot(e0[0], e0[1]), tb0_ = tburn, w0_ = wait;
      for (double k = 1.0; k > 1e-3; k *= 0.5) {          // backtrack: wait is a very sensitive lever
        tburn = tb0_ + k * dtb; wait = w0_ + k * dw;
        double e1[4]; errs(e1);
        if (wait >= 0 && std::hypot(e1[0], e1[1]) < n0) break;
      }
    }
    // (b) plane
    { TransferResult R = sim_transfer(raan_off, wait, tburn, dinc, false, 1);
      double dO, di; plane_err(R, dO, di); raan_off -= dO; dinc -= di; }
    errs(e);
    double en = std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]);
    std::printf("shoot it %d: |e| = %.2f m  (y %.2f drift %.2f z %.2f zd %.2f)  on-time %.1f s wait %.1f s\n", it, en, e[0], e[1], e[2], e[3], tburn, wait);
    if (en < 1.0) break;
  }
  // SC2/SC3 injection lag: secant so each arrives on its V-bar slot behind TOPEX (no loop-over)
  for (int k = 1; k < NC; ++k) {
    LEADK[k] = k * SEP0 / ac;
    auto yk = [&]() { TransferResult R = sim_transfer(raan_off, wait, tburn, dinc, false);
                      V3 rho, rhod; rel(R.tgt, R.c[k], rho, rhod); return rho.y - AIM_YK[k]; };
    for (int s = 0; s < 8; ++s) {
      double f0 = yk(); if (std::fabs(f0) < 50) break;
      LEADK[k] += 1e-5; double f1 = yk(); LEADK[k] -= 1e-5;
      LEADK[k] -= 1e-5 * f0 / (f1 - f0);
    }
  }
  TransferResult TR = sim_transfer(raan_off, wait, tburn, dinc, true);
  double m_xe_used = M_WET - TR.m_end;

  FILE* ft = std::fopen("transfer.csv", "w");
  std::fprintf(ft, "t,tx,ty,tz,c0x,c0y,c0z,c1x,c1y,c1z,c2x,c2y,c2z,phase,dv0,m0\n");
  for (auto& r : TR.log) {
    for (size_t i = 0; i < r.size(); ++i) std::fprintf(ft, i ? ",%.3f" : "%.3f", r[i]);
    std::fprintf(ft, "\n");
  }
  std::fclose(ft);

  // ---------------- Phase 2: proximity ops
  const double P = 2 * PI / nt;
  State tgt = TR.tgt;
  Craft C[NC];
  for (int k = 0; k < NC; ++k) C[k].s = TR.c[k];

  V3 arr_rho[NC];
  for (int k = 0; k < NC; ++k) { V3 rd; rel(tgt, C[k].s, arr_rho[k], rd); }

  // ---- timeline (t = 0 at circularisation)
  double tA  = 0.45 * P;                // arrive at V-bar holds
  double tB  = tA + 1800;               // detect / fuse / role bid complete
  double tN1 = tB + 0.40 * P;           // imager 1 enters NMC
  double tN2 = tN1 + 0.5 * P;           // imager 2 enters NMC (phase point on approach side)
  double tGO = tN1 + P;                 // spin estimate converged -> GO
  double hop = 600;
  std::vector<double> gs = {-200, -120, -70, -45, -30};
  double tH30 = tGO + hop * (gs.size() - 1);  // at 30 m hold (sync)
  double tFA = tH30 + 600;                    // final approach start
  double vfa = 0.03;
  double tC = tFA + (30 - DOCK_R) / vfa;      // contact
  double tDT = tC + T_DETUMBLE;               // detumble done
  double tS = tDT + T_HOLD;                   // MR-4 stable hold complete
  // imagers leave the NMC where it crosses V-bar on the approach side (theta = pi/2)
  double tD[NC] = {0, 0, 0};
  for (int k = 1; k < NC; ++k) {
    double alpha = (k == 1) ? PI / 2 : -PI / 2;
    double t0 = tN1 + (PI / 2 - alpha) / nt;
    while (t0 < tS) t0 += P;
    tD[k] = t0;
  }
  double tU = std::max(tD[1], tD[2]) + 1200;  // undock once both imagers are clear
  double vbo = 0.05;
  double tBO = tU + (30 - DOCK_R) / vbo;      // back-out complete at 30 m
  double tR0 = tBO + 0.45 * P;                // docker back at -1 km
  double tEnd = std::max(tR0, std::max(tD[1], tD[2]) + 0.45 * P) + 1800;

  double nmc_b = 23, nmc_c = 19, svc_b = 23, svc_c = 19;   // LCAM precision standoff ~22 m (STM row 1)
  double y_hold[NC] = {-1000, -1300, -1600};

  // docker = craft 0 (lowest bid: closest to target, ties -> lowest ID)
  {
    auto& L = C[0].legs;
    L.push_back({XFER, 0, tA, "Rendezvous to V-bar hold", "rendezvous", {0, y_hold[0], 0}});
    L.push_back({HOLD, tA, tB, "Hold: detect / fuse / bid", "rendezvous", {0, y_hold[0], 0}});
    L.push_back({XFER, tB, tN1, "Transfer to 200 m hold", "docking", {0, -200, 0}});
    L.push_back({HOLD, tN1, tGO, "Hold 200 m: wait for GO", "docking", {0, -200, 0}});
    for (size_t i = 1; i < gs.size(); ++i) {
      double a = tGO + hop * (i - 1);
      L.push_back({XFER, a, a + hop, "Glideslope hop", "docking", {0, gs[i], 0}});
    }
    L.push_back({HOLD, tH30, tFA, "Hold 30 m: sync to spin axis", "docking", {0, -30, 0}});
    Leg fa{FORCED, tFA, tC, "Final approach (forced motion)", "docking"};
    fa.pa = {0, -30, 0}; fa.pb = {0, -DOCK_R, 0}; fa.speed = vfa; L.push_back(fa);
    L.push_back({DOCKED, tC, tU, "Docked: despin + stable hold", "docking"});
    Leg bo{FORCED, tU, tBO, "Undock + back-out", "departure"};
    bo.pa = {0, -DOCK_R, 0}; bo.pb = {0, -30, 0}; bo.speed = vbo; L.push_back(bo);
    L.push_back({XFER, tBO, tR0, "Depart to regroup point", "departure", {0, y_hold[0], 0}});
    L.push_back({HOLD, tR0, tEnd, "Regroup hold", "departure", {0, y_hold[0], 0}});
  }
  for (int k = 1; k < NC; ++k) {
    auto& L = C[k].legs;
    double tin = (k == 1) ? tN1 : tN2;
    double alpha = (k == 1) ? PI / 2 : -PI / 2;   // 180 deg apart on the same NMC
    double tleave = tin - 0.40 * P;
    L.push_back({XFER, 0, tA, "Rendezvous to V-bar hold", "rendezvous", {0, y_hold[k], 0}});
    L.push_back({HOLD, tA, tleave, "Hold: detect / fuse / bid", "rendezvous", {0, y_hold[k], 0}});
    Leg x{XFER, tleave, tin, "Transfer into NMC", "inspection"};
    x.aim_nmc = true; x.b = nmc_b; x.c = nmc_c; x.alpha = alpha; x.tep = tN1; L.push_back(x);
    Leg n1{NMC, tin, tDT, "NMC inspection (b=23 m)", "inspection"};
    n1.b = nmc_b; n1.c = nmc_c; n1.alpha = alpha; n1.tep = tN1; L.push_back(n1);
    Leg n2{NMC, tDT, tD[k], "NMC stable-hold watch (b=23 m)", "inspection"};
    n2.b = svc_b; n2.c = svc_c; n2.alpha = alpha; n2.tep = tN1; L.push_back(n2);
    L.push_back({XFER, tD[k], tD[k] + 0.45 * P, "Depart to regroup point", "departure", {0, y_hold[k], 0}});
    L.push_back({HOLD, tD[k] + 0.45 * P, tEnd, "Regroup hold", "departure", {0, y_hold[k], 0}});
  }

  // events file
  FILE* fe = std::fopen("events.csv", "w");
  std::fprintf(fe, "craft,t0,t1,type,name,cat\n");
  for (int k = 0; k < NC; ++k)
    for (auto& L : C[k].legs) std::fprintf(fe, "%d,%.1f,%.1f,%d,%s,%s\n", k, L.t0, L.t1, (int)L.type, L.name.c_str(), L.cat.c_str());
  std::fclose(fe);

  FILE* fp = std::fopen("prox.csv", "w");
  std::fprintf(fp, "t");
  for (int k = 0; k < NC; ++k) std::fprintf(fp, ",x%d,y%d,z%d,vx%d,vy%d,vz%d,leg%d,dv%d", k, k, k, k, k, k, k, k);
  std::fprintf(fp, "\n");

  double dt = 1.0, t = 0.0;
  double min_pair[3] = {1e18, 1e18, 1e18};
  double min_tgt_free[NC] = {1e18, 1e18, 1e18};   // min range to TOPEX outside docking/forced legs
  int step = 0;
  while (t <= tEnd + 1e-6) {
    Frame F = lvlh(tgt);
    V3 aE[NC];
    for (int k = 0; k < NC; ++k) {
      Craft& c = C[k];
      aE[k] = {0, 0, 0};
      while (c.li + 1 < (int)c.legs.size() && t >= c.legs[c.li].t1 - 1e-9) { c.li++; c.entered = false; }
      Leg& L = c.legs[c.li];
      V3 rho, rhod; rel(tgt, c.s, rho, rhod);
      if (!c.entered) { c.entered = true; c.next_rt = t; if (L.type == DOCKED) c.lock = rho; }
      if (L.type == XFER || L.type == HOLD || L.type == NMC) {
        if (t >= c.next_rt - 1e-9) {
          double ta;
          if (L.type == XFER) {
            double T = L.t1 - L.t0;
            ta = L.t1;
            c.next_rt = (T > 1200 && t < L.t0 + T / 2 - 1) ? L.t0 + T / 2 : 1e18;
          } else {
            double seg = (L.type == NMC) ? 0.45 * P : 0.30 * P;
            ta = std::min(t + seg, L.t1);
            if (ta - t < 120) ta = t + 120;     // never fire very short arcs
            c.next_rt = ta;
          }
          double Tgo = ta - t;
          if (Tgo > 30) {
            V3 aim;
            if (L.type == NMC || L.aim_nmc) { V3 v; nmc_point(L, ta, aim, v); }
            else aim = L.p;
            V3 v0 = cw_v0(rho, aim, F.w, Tgo);
            apply_burn(c, k, tgt, t, v0 - rhod, L);
          }
        }
      } else if (L.type == FORCED) {
        double len = norm(L.pb - L.pa);
        V3 u = (L.pb - L.pa) * (1.0 / len);
        double s = std::min(len, (t - L.t0) * L.speed);
        V3 pref = L.pa + u * s;
        V3 vref = (s < len) ? u * L.speed : V3{0, 0, 0};
        if (t - L.t0 < 0.5) {  // entry: impulsive change to reference velocity
          apply_burn(c, k, tgt, t, vref - rhod, L);
          rel(tgt, c.s, rho, rhod);
        }
        double n = F.w, Kp = 0.0025, Kd = 0.1;
        V3 cwdyn{3 * n * n * rho.x + 2 * n * rhod.y, -2 * n * rhod.x, -n * n * rho.z};
        V3 aL = (pref - rho) * Kp + (vref - rhod) * Kd - cwdyn;
        aE[k] = fromL(F, aL);
        double m = norm(aL) * dt;
        c.dv_cont += m; c.dv_cat[catidx(L.cat)] += m;
      } else if (L.type == DOCKED) {
        c.s = from_rel(tgt, c.lock, {0, 0, 0});
      }
    }
    // checks
    for (int k = 0; k < NC; ++k) {
      LegType ty = C[k].legs[C[k].li].type;
      if (ty != FORCED && ty != DOCKED) min_tgt_free[k] = std::min(min_tgt_free[k], norm(C[k].s.r - tgt.r));
    }
    int pi = 0;
    for (int a = 0; a < NC; ++a) for (int b = a + 1; b < NC; ++b)
      min_pair[pi] = std::min(min_pair[pi], norm(C[a].s.r - C[b].s.r)), ++pi;

    if (step % 5 == 0) {
      std::fprintf(fp, "%.1f", t);
      for (int k = 0; k < NC; ++k) {
        V3 rho, rhod; rel(tgt, C[k].s, rho, rhod);
        std::fprintf(fp, ",%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%d,%.5f", rho.x, rho.y, rho.z, rhod.x, rhod.y, rhod.z,
                     C[k].li, C[k].dv_imp + C[k].dv_cont);
      }
      std::fprintf(fp, "\n");
    }
    // propagate
    State tg2 = rk4(tgt, dt, {0, 0, 0});
    for (int k = 0; k < NC; ++k)
      if (C[k].legs[C[k].li].type != DOCKED) C[k].s = rk4(C[k].s, dt, aE[k]);
    tgt = tg2;
    for (int k = 0; k < NC; ++k)
      if (C[k].legs[C[k].li].type == DOCKED) C[k].s = from_rel(tgt, C[k].lock, {0, 0, 0});
    t += dt; ++step;
  }
  std::fclose(fp);

  FILE* fb = std::fopen("burns.csv", "w");
  std::fprintf(fb, "t,craft,dvx,dvy,dvz,dv,range,min_range_if_missed,leg,cat\n");
  for (auto& b : burns)
    std::fprintf(fb, "%.1f,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%s,%s\n", b.t, b.craft, b.dvL.x, b.dvL.y, b.dvL.z, b.dv,
                 b.range, b.min_range_missed, b.leg.c_str(), b.cat.c_str());
  std::fclose(fb);

  // ---------------- summary
  FILE* fs = std::fopen("summary.txt", "w");
  auto both = [&](const char* fmt, auto... a) { std::printf(fmt, a...); std::fprintf(fs, fmt, a...); };
  double m_arr = TR.m_end;
  both("=== PHASE 1: low-thrust spiral %.0f km -> %.0f km, i = %.2f deg (BHT-600, %.0f mN, Isp %.0f s) ===\n",
       H_PARK / 1e3, H_TOPEX / 1e3, INC / D2R, F_HALL * 1e3, ISP_HALL);
  both("Edelbaum estimate      : dv = %.2f m/s (v_circ,500 - v_circ,1336)\n", dv_edel);
  both("Sim (J2) spiral        : dv = %.2f m/s, spiral time = %.2f days, xenon used = %.3f kg of %.0f kg\n",
       TR.dv_spiral, (TR.t_arrive - wait) / 86400, m_xe_used, M_XE);
  double dv_rep = dv_edel * STEER, xe_rep = M_WET * (1 - std::exp(-dv_rep / (ISP_HALL * G0)));
  both("Reported spiral        : dv = %.1f m/s (Edelbaum x %.2f steering), xenon = %.2f kg of %.0f kg\n", dv_rep, STEER, xe_rep, M_XE);
  both("Hall duty              : last %.1f%% (500 km) to %.1f%% (1336 km) of each orbit at %.0f W, on-time %.0f h\n",
       duty(ac) * 100, duty(at) * 100, P_HALL, tburn / 3600);
  both("Initial accel          : %.3e m/s^2 at %.3f kg wet\n", F_HALL / M_WET, M_WET);
  both("Phasing wait in parking: %.2f h  (synodic period %.2f h)\n", wait / 3600, 2 * PI / (nc - nt) / 3600);
  both("RAAN / inc pre-offset  : %.4f / %.4f deg (cancel differential J2 drift over the spiral)\n", raan_off / D2R, dinc / D2R);
  both("Arrival time (t0)      : %.2f days after injection (thrust cutoff)\n", TR.t_arrive / 86400);
  for (int k = 0; k < NC; ++k)
    both("  SC%d arrival LVLH [m] : R %.0f  V %.0f  H %.0f\n", k + 1, arr_rho[k].x, arr_rho[k].y, arr_rho[k].z);
  both("\n=== PHASE 2: proximity ops (TOPEX orbit period %.1f min) ===\n", P / 60);
  both("Timeline (h from arrival): holds %.2f | bid done %.2f | NMC in %.2f/%.2f | GO %.2f | contact %.2f | detumbled %.2f | stable hold done %.2f | imagers out %.2f/%.2f | undock %.2f | end %.2f\n",
       tA / 3600, tB / 3600, tN1 / 3600, tN2 / 3600, tGO / 3600, tC / 3600, tDT / 3600, tS / 3600, tD[1] / 3600, tD[2] / 3600, tU / 3600, tEnd / 3600);
  const char* cats[4] = {"rendezvous", "inspection", "docking", "departure"};
  for (int k = 0; k < NC; ++k) {
    double dvk = C[k].dv_imp + C[k].dv_cont;
    both("SC%d (%s): prox-ops dv = %.3f m/s (hydrazine %.3f kg)  [", k + 1, k == 0 ? "docker" : "imager", dvk, m_arr * (1 - std::exp(-dvk / (ISP_HYD * G0))));
    for (int i = 0; i < 4; ++i) both("%s %.3f%s", cats[i], C[k].dv_cat[i], i < 3 ? ", " : "]\n");
    both("     min range to TOPEX outside docking corridor: %.1f m (KOZ %.0f m)\n", min_tgt_free[k], KOZ);
  }
  both("Min inter-craft distance: SC1-SC2 %.1f m, SC1-SC3 %.1f m, SC2-SC3 %.1f m\n", min_pair[0], min_pair[1], min_pair[2]);
  int n_unsafe = 0;
  for (auto& b : burns) if (b.range > KOZ && b.min_range_missed >= 0 && b.min_range_missed < KOZ) n_unsafe++;
  both("Burns: %zu total; %d outside the KOZ would enter the %.0f m KOZ if the NEXT burn were missed\n", burns.size(), n_unsafe, KOZ);
  double J_dt = IZ_TOPEX * (W0_SPIN - W_BAND) / ARM_RCS;
  double H_su = IX_SWARM * W0_SPIN, mp_su = std::max(0.0, H_su - H_RW) / ARM_RCS / (ISP_HYD * G0);
  double mp_dt = J_dt / (ISP_HYD * G0);
  double dv0 = C[0].dv_imp + C[0].dv_cont;
  double mp0 = m_arr * (1 - std::exp(-dv0 / (ISP_HYD * G0)));
  both("\n=== DETUMBLE (docker) ===\n");
  both("TOPEX Iz %.0f kg m^2, w0 %.4f rad/s (T = 9.5 s), H = %.0f N m s\n", IZ_TOPEX, W0_SPIN, IZ_TOPEX * W0_SPIN);
  both("Torque %.2f N m (one %.0f N thruster x %.2f m arm), into MR-4 band +-%.5f rad/s\n", TAU_DT, F_RCS, ARM_RCS, W_BAND);
  both("Detumble time = %.0f s = %.2f h; hydrazine = %.2f kg (total impulse %.0f N s)\n", T_DETUMBLE, T_DETUMBLE / 3600, mp_dt, J_dt);
  both("\n=== SPIN-UP (docker, before capture) ===\n");
  both("SWARM Ix %.2f kg m^2 (arm open) x %.4f rad/s = %.1f N m s; RW8 %.0f N m s, hydrazine top-up %.3f kg (%.0f s on one 1 N)\n",
       IX_SWARM, W0_SPIN, H_su, H_RW, mp_su, std::max(0.0, H_su - H_RW) / ARM_RCS);
  both("\n=== BUDGETS (per craft) ===\n");
  both("Hall / xenon : spiral %.1f of %.0f m/s Hall dv capability; %.2f of %.0f kg xenon\n", dv_rep, DV_HALL, xe_rep, M_XE);
  for (int k = 0; k < NC; ++k) {
    double dvk = C[k].dv_imp + C[k].dv_cont, mpk = m_arr * (1 - std::exp(-dvk / (ISP_HYD * G0)));
    if (k == 0) mpk += mp_su + mp_dt;
    both("SC%d hydrazine: %.2f of %.0f kg (prox-ops dv %.2f m/s%s)\n", k + 1, mpk, M_HYD, dvk, k == 0 ? " + spin-up + despin" : "");
  }
  both("Docker hydrazine: prox %.2f + spin-up %.3f + detumble %.2f = %.2f kg of %.0f kg\n", mp0, mp_su, mp_dt, mp0 + mp_su + mp_dt, M_HYD);
  std::fclose(fs);
  return 0;
}
