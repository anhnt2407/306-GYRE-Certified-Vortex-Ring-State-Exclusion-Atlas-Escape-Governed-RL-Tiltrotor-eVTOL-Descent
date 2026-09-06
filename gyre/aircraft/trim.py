"""Quasi-static descent trim of the VRS-augmented plant over the cube
(Vx, Vz_descent, sigma).

Unknowns z = [theta_deg, duty_f, duty_r, delta_rv, om_f, om_r, s_f, s_r]:
attitude is the free variable (not alpha — in rotor-borne near-vertical
descent the flight-path angle is large and wing-alpha bounds are
meaningless), and the rotor-speed and ring-intensity equilibria are part of
the residual vector rather than nested fixed points, so one
``vehicle.derivs`` call evaluates everything:

    r = [udot, wdot, 10*qdot, W_SEL*(duty_r - rear_share*duty_f),
         C_OM*omdot_f, C_OM*omdot_r, C_S*sdot_f, C_S*sdot_r]

The intensity equation s = s_qs(x, s) is bistable inside the hysteresis band
(B_x increases with s).  Continuation selects the branch: marching descent
rate upward from level flight tracks the **entry branch**; marching downward
from deep descent tracks the **exit branch**.  The atlas records both; the
gap is the certified hysteresis band.  Certificates never consume trim
severities — they use the exact box sup (``ring.s_qs_sup``), which dominates
every branch.

Stall margin is gated below V = ALPHA_GATE_V (family convention: wing alpha
is meaningless at hover airspeeds).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..config import RHO0, TASK
from ..vrs import ring
from . import vehicle

W_SEL = 0.05
C_OM = 0.02      # scales omdot (rad/s^2) to ~residual units
C_S = 2.0        # scales sdot (1/s)
RES_TOL = 0.06
ALPHA_GATE_V = 5.0
THETA_BOUNDS = (-25.0, 20.0)


def rear_share(sig_deg):
    """Rear-pair duty share collapses as the wing takes over (296 lineage)."""
    return np.clip((sig_deg - 15.0) / 45.0, 0.0, 1.0)


def _state_from_z(z, vx, vzd, sig_deg, soc=0.9):
    theta_deg, duty_f, duty_r, delta_rv, om_f, om_r, s_f, s_r = z
    v = np.hypot(vx, vzd)
    gamma = np.arctan2(-vzd, np.maximum(vx, 1e-6))
    theta = np.radians(theta_deg)
    alpha = theta - gamma
    u = v * np.cos(alpha)
    w = v * np.sin(alpha)
    x = np.zeros((1, 13))
    x[0, 0], x[0, 1], x[0, 3] = u, w, theta
    x[0, 4] = 60.0
    x[0, 5], x[0, 6] = om_f, om_r
    x[0, 7], x[0, 8] = np.radians(sig_deg), soc
    x[0, 9], x[0, 10] = s_f, s_r
    u_ctrl = np.array([[duty_f, duty_r, delta_rv, 0.0]])
    return x, u_ctrl, np.degrees(alpha)


def trim_residual(z, vx, vzd, sig_deg, soc=0.9, rho=RHO0):
    x, u_ctrl, _ = _state_from_z(z, vx, vzd, sig_deg, soc)
    env = vehicle.make_env(1)
    dx = vehicle.derivs(x, u_ctrl, env, rho)
    tie = W_SEL * (z[2] - rear_share(sig_deg) * z[1])
    return np.array([dx[0, 0], dx[0, 1], 10.0 * dx[0, 2], tie,
                     C_OM * dx[0, 5], C_OM * dx[0, 6],
                     C_S * dx[0, 9], C_S * dx[0, 10]])


LB = [THETA_BOUNDS[0], 0.02, 0.0, -23.0, 50.0, 50.0, 0.0, 0.0]
UB = [THETA_BOUNDS[1], 0.98, 0.98, 23.0, 1300.0, 1300.0, 1.0, 1.0]


def default_guess(vx, vzd, sig_deg):
    sh = rear_share(sig_deg)
    return np.array([2.0 - 0.15 * vx, 0.60, 0.60 * max(sh, 0.02), 0.0,
                     700.0, 700.0 * max(sh, 0.1), 0.0, 0.0])


def solve_point(vx, vzd, sig_deg, z0=None, soc=0.9, rho=RHO0):
    """Trim one cube point.  Returns a record dict or None if infeasible."""
    if z0 is None:
        z0 = default_guess(vx, vzd, sig_deg)
    z0 = np.clip(z0, LB, UB)
    try:
        sol = least_squares(trim_residual, z0, bounds=(LB, UB),
                            args=(vx, vzd, sig_deg, soc, rho),
                            xtol=1e-12, ftol=1e-14, max_nfev=400)
    except Exception:
        return None
    res = trim_residual(sol.x, vx, vzd, sig_deg, soc, rho)
    if np.max(np.abs(res)) > RES_TOL:
        return None
    x, u_ctrl, alpha_deg = _state_from_z(sol.x, vx, vzd, sig_deg, soc)
    env = vehicle.make_env(1)
    prop = vehicle.propulsion_wrench(x, u_ctrl[:, 0], u_ctrl[:, 1], env, rho)
    sev = prop["sev"]
    i_ch = prop["i_ch"][0]
    rec = {
        "vx": float(vx), "vzd": float(vzd), "sig_deg": float(sig_deg),
        "theta_deg": float(sol.x[0]), "alpha_deg": float(alpha_deg),
        "duty_f": float(sol.x[1]), "duty_r": float(sol.x[2]),
        "delta_rv": float(sol.x[3]),
        "om_f": float(sol.x[4]), "om_r": float(sol.x[5]),
        "s_f": float(sol.x[6]), "s_r": float(sol.x[7]),
        "zb_f": float(sev["zb_f"][0]), "xb_f": float(sev["xb_f"][0]),
        "zb_r": float(sev["zb_r"][0]), "xb_r": float(sev["xb_r"][0]),
        "vh_f": float(np.atleast_1d(sev["vh_f"])[0]),
        "vh_r": float(np.atleast_1d(sev["vh_r"])[0]),
        "i_f": float(i_ch[0]), "i_r": float(i_ch[1]),
        "p_bus": float(prop["bus"]["V_bus"][0] * prop["bus"]["I_pack"][0]),
        "res": float(np.max(np.abs(res))),
        "z": [float(v) for v in sol.x],
    }
    rec["mu"] = float(margin_min(rec))
    return rec


def margin_vector(rec):
    """Normalized actuator/aero headrooms (296 four-margin convention);
    wing-alpha margin gated below ALPHA_GATE_V."""
    v = np.hypot(rec["vx"], rec["vzd"])
    if v >= ALPHA_GATE_V:
        m_alpha = (TASK.alpha_stall_deg - rec["alpha_deg"]) / 8.0
    else:
        m_alpha = 1.0
    m_uc = (TASK.uc_max - max(rec["duty_f"], rec["duty_r"])) / 0.25
    m_cur = (TASK.i_mot_cert - max(rec["i_f"], rec["i_r"])) / 30.0
    m_rv = (1.0 - abs(rec["delta_rv"]) / 23.0) / 0.6
    return np.array([m_alpha, m_uc, m_cur, m_rv])


def margin_min(rec):
    return float(np.min(margin_vector(rec)))


def solve_column(vx, sig_deg, vzd_vals, soc=0.9, z0=None):
    """March a descent column seeding each solve from the previous
    (continuation; ordering selects the severity branch — SLEET lesson:
    ordering is load-bearing)."""
    out = []
    for vzd in vzd_vals:
        rec = solve_point(vx, vzd, sig_deg, z0, soc)
        out.append(rec)
        if rec is not None:
            z0 = np.array(rec["z"])
    return out
