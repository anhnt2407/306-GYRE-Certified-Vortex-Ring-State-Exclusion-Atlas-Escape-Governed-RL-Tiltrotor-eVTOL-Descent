"""Batched longitudinal KP-2C descent plant with the ring-state layer.

Symmetric (wings-level) steep descent: lateral-directional axes excluded by
symmetry; the front (tilting) pair shares one duty and one rotor-speed state,
likewise the rear (fixed, vertical) pair — 296 conventions.  GYRE adds per-
pair ring-intensity states and buffet carriers, and removes the family's
``v_ax >= -5 m/s`` inflow clamp: descending-disk flow is the subject here.

State vector x (N, 13):
  0 u      body forward velocity [m/s]     7 sigma  tilt [rad] (pi/2 hover, 0 cruise)
  1 w      body downward velocity          8 soc    battery state of charge
  2 q      pitch rate [rad/s]              9 s_f    front-pair ring intensity [0,1]
  3 theta  pitch attitude [rad]           10 s_r    rear-pair ring intensity
  4 h      altitude AGL [m]               11 xi_f   front buffet OU carrier
  5 om_f   front-pair rotor speed [rad/s] 12 xi_r   rear buffet OU carrier
  6 om_r   rear-pair rotor speed

Control u (N, 4): duty_f, duty_r in [0,1]; delta_rv [deg]; sigma_dot [rad/s].
Environment env: wind_x (N,) headwind [m/s]; optional VRS overrides
  k_t, k_p, c_fl, tau_form, tau_coll (scalars or (N,)) for the OOD ladder.

Plant rate 100 Hz (RK4; the OU carriers are stepped exactly once per substep,
frozen inside the RK4 stages).  Forces in body axes, z down; aero applied at
the aero reference point and transferred to the CG.
"""
from __future__ import annotations

import numpy as np

from ..config import G0, RHO0, SIM, VRS
from ..tables import tables
from ..vrs import ring
from .powertrain import bus_solve, rotor_wrench

IDX_S = 9
IDX_XI = 11
CH_FRONT = (0, 2)   # FR, FL
CH_REAR = (1, 3)    # RL, RR


def aero_forces(ua, w, q, delta_rv, rho=RHO0):
    """Longitudinal aero wrench about the CG.  Returns Fx, Fz, My."""
    t = tables()
    vt = np.sqrt(ua * ua + w * w)
    vt_s = np.maximum(vt, 0.5)
    alpha = np.degrees(np.arctan2(w, np.maximum(ua, 1e-9)))
    qbar = 0.5 * rho * vt * vt
    ci2vel = t.cbar / (2.0 * vt_s)

    cz = t.CZ(alpha) + t.CZ_dlv(delta_rv) + t.CZ_drv(delta_rv) \
        + t.CZq * ci2vel * q
    cx = t.CX(alpha) + t.CX_dlv(delta_rv) + t.CX_drv(delta_rv) \
        + t.CXq * ci2vel * q
    cmy = t.CMY(alpha) + t.CMY_dlv(delta_rv) + t.CMY_drv(delta_rv) \
        + t.CMYq * ci2vel * q

    fx = qbar * t.S * cx
    fz = qbar * t.S * cz
    my = qbar * t.S * t.cbar * cmy
    rx, rz = t.r_rp_body[0], t.r_rp_body[2]
    my = my + rz * fx - rx * fz
    return fx, fz, my


def severity_inputs(x, wind_x, rho=RHO0):
    """Per-pair normalized-plane coordinates and quasi-steady severities from
    the current state.  Normalization by the weight-based hover induced
    velocity (chart convention — see ``ring.v_hover0``); severity is then a
    pure function of kinematics and the hysteresis state."""
    u, w, sigma = x[:, 0], x[:, 1], x[:, 7]
    s_f, s_r = x[:, IDX_S], x[:, IDX_S + 1]
    ua = u + wind_x
    v_ax_f, v_pp_f, v_ax_r, v_pp_r = ring.disk_velocities(ua, w, sigma)
    vh0 = ring.v_hover0(rho)
    zb_f, xb_f = ring.normalized_plane(v_ax_f, v_pp_f, vh0)
    zb_r, xb_r = ring.normalized_plane(v_ax_r, v_pp_r, vh0)
    sq_f = ring.s_qs(zb_f, xb_f, s_f)
    sq_r = ring.s_qs(zb_r, xb_r, s_r)
    return {"v_ax_f": v_ax_f, "v_ax_r": v_ax_r, "v_pp_f": v_pp_f,
            "v_pp_r": v_pp_r, "vh_f": vh0, "vh_r": vh0,
            "zb_f": zb_f, "xb_f": xb_f, "zb_r": zb_r, "xb_r": xb_r,
            "sq_f": sq_f, "sq_r": sq_r}


def propulsion_wrench(x, duty_f, duty_r, env, rho=RHO0):
    """Pair thrust/torque with ring effects, per-channel currents, severity
    diagnostics."""
    om_f, om_r, soc = x[:, 5], x[:, 6], x[:, 8]
    s_f, s_r = x[:, IDX_S], x[:, IDX_S + 1]
    xi_f, xi_r = x[:, IDX_XI], x[:, IDX_XI + 1]
    k_t = env.get("k_t", VRS.k_t)
    k_p = env.get("k_p", VRS.k_p)
    c_fl = env.get("c_fl", VRS.c_fl)

    sev = severity_inputs(x, env["wind_x"], rho)
    duty = np.stack([duty_f, duty_r, duty_f, duty_r], axis=1)
    omega = np.stack([om_f, om_r, om_f, om_r], axis=1)
    bus = bus_solve(duty, omega, soc, rho)

    th_f, qa_f = rotor_wrench(om_f, sev["v_ax_f"], s_f, xi_f, rho,
                              k_t=k_t, k_p=k_p, c_fl=c_fl)
    th_r, qa_r = rotor_wrench(om_r, sev["v_ax_r"], s_r, xi_r, rho,
                              k_t=k_t, k_p=k_p, c_fl=c_fl)
    return {"bus": bus, "i_ch": bus["I"], "thrust_f": th_f, "thrust_r": th_r,
            "q_aero_f": qa_f, "q_aero_r": qa_r, "omega": omega, "sev": sev}


def derivs(x, u_ctrl, env, rho=RHO0):
    """Full state derivative, batched.  env: dict(wind_x (N,), optional VRS
    overrides).  The buffet carriers have zero drift here (stepped exactly in
    ``rk4_step``)."""
    t = tables()
    u, w, q, th = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    sigma = x[:, 7]
    duty_f = np.clip(u_ctrl[:, 0], 0.0, 1.0)
    duty_r = np.clip(u_ctrl[:, 1], 0.0, 1.0)
    delta_rv = np.clip(u_ctrl[:, 2], -23.0, 23.0)
    sig_dot = u_ctrl[:, 3]

    prop = propulsion_wrench(x, duty_f, duty_r, env, rho)
    ua = u + env["wind_x"]
    fx_a, fz_a, my_a = aero_forces(ua, w, q, delta_rv, rho)

    tf, tr = prop["thrust_f"], prop["thrust_r"]
    fx = fx_a + 2.0 * tf * np.cos(sigma)
    fz = fz_a - 2.0 * tf * np.sin(sigma) - 2.0 * tr
    rf = t.r_rotor_body[0]
    rr = t.r_rotor_body[1]
    my_f = 2.0 * (rf[2] * tf * np.cos(sigma) - rf[0] * (-tf * np.sin(sigma)))
    my_r = 2.0 * (-rr[0] * (-tr))
    my = my_a + my_f + my_r

    m = t.mass
    du = fx / m - G0 * np.sin(th) - q * w
    dw = fz / m + G0 * np.cos(th) + q * u
    dq = my / t.Iyy
    dth = q
    dh = u * np.sin(th) - w * np.cos(th)

    i_f = prop["i_ch"][:, CH_FRONT[0]]
    i_r = prop["i_ch"][:, CH_REAR[0]]
    qm_f = t.Kt * np.maximum(i_f - t.I0, 0.0)
    qm_r = t.Kt * np.maximum(i_r - t.I0, 0.0)
    dom_f = (qm_f - prop["q_aero_f"]) / t.I_rotor
    dom_r = (qm_r - prop["q_aero_r"]) / t.I_rotor

    dsig = sig_dot
    dsoc = -prop["bus"]["I_pack"] / t.cap_As

    tau_form = env.get("tau_form", VRS.tau_form)
    tau_coll = env.get("tau_coll", VRS.tau_coll)
    sev = prop["sev"]
    s_f, s_r = x[:, IDX_S], x[:, IDX_S + 1]
    ds_f = ring.s_dot(s_f, sev["sq_f"], tau_form, tau_coll)
    ds_r = ring.s_dot(s_r, sev["sq_r"], tau_form, tau_coll)

    dx = np.zeros_like(x)
    dx[:, 0], dx[:, 1], dx[:, 2], dx[:, 3], dx[:, 4] = du, dw, dq, dth, dh
    dx[:, 5], dx[:, 6], dx[:, 7], dx[:, 8] = dom_f, dom_r, dsig, dsoc
    dx[:, IDX_S], dx[:, IDX_S + 1] = ds_f, ds_r
    return dx


def rk4_step(x, u_ctrl, env, dt=SIM.fdm_dt, rng=None):
    """One 100 Hz step: RK4 on the deterministic states, exact OU update of
    the buffet carriers (rng=None freezes them — used by trim/DP audits)."""
    k1 = derivs(x, u_ctrl, env)
    k2 = derivs(x + 0.5 * dt * k1, u_ctrl, env)
    k3 = derivs(x + 0.5 * dt * k2, u_ctrl, env)
    k4 = derivs(x + dt * k3, u_ctrl, env)
    xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[:, 5] = np.clip(xn[:, 5], 0.0, 1300.0)
    xn[:, 6] = np.clip(xn[:, 6], 0.0, 1300.0)
    xn[:, 7] = np.clip(xn[:, 7], 0.0, np.pi / 2)
    xn[:, 8] = np.clip(xn[:, 8], 0.02, 1.0)
    xn[:, IDX_S:IDX_S + 2] = np.clip(xn[:, IDX_S:IDX_S + 2], 0.0, 1.0)
    if rng is not None:
        xn[:, IDX_XI] = ring.ou_step(x[:, IDX_XI], dt, rng)
        xn[:, IDX_XI + 1] = ring.ou_step(x[:, IDX_XI + 1], dt, rng)
    return xn


def make_env(n, wind_x=0.0, **overrides):
    env = {"wind_x": np.full(n, float(wind_x))
           if np.isscalar(wind_x) else np.asarray(wind_x, float)}
    env.update(overrides)
    return env


def hover_state(n=1, soc=0.95, h=60.0):
    x = np.zeros((n, 13))
    x[:, 4] = h
    x[:, 5] = x[:, 6] = 700.0
    x[:, 7] = np.pi / 2
    x[:, 8] = soc
    return x
