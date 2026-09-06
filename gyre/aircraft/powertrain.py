"""Electric propulsion chain in the KDS convention (battery -> ESC -> DC-model
BLDC -> advance-ratio propeller), vectorized over batch and channel — 296
lineage — with the GYRE extension: the propeller wrench accepts a ring-state
intensity and a buffet carrier and applies

    T = CT(J) rho n^2 D^4 * (1 - k_t s + c_fl s xi)
    Q = CP(J)/(2 pi) rho n^2 D^5 * (1 + k_p s)

Windings are at reference temperature (the electro-thermal axis is #296's);
J is allowed to run to the deep-descent extension J_LO = -0.45 — GYRE removes
the family's v_ax >= -5 m/s clamp because modeling that region is the project.
"""
from __future__ import annotations

import numpy as np

from ..config import RHO0, VRS
from ..tables import tables


def rotor_wrench(omega, v_axial, s=0.0, xi=0.0, rho=RHO0,
                 k_t=None, k_p=None, c_fl=None):
    """Per-rotor thrust and aero torque with the VRS multipliers.  All args
    broadcast; s, xi are the pair's ring intensity and buffet carrier."""
    t = tables()
    k_t = VRS.k_t if k_t is None else k_t
    k_p = VRS.k_p if k_p is None else k_p
    c_fl = VRS.c_fl if c_fl is None else c_fl
    n = np.maximum(np.abs(omega), 1.0) / (2.0 * np.pi)
    J = v_axial / (n * t.D)
    t_base = t.CT_J(J) * rho * n * n * t.D ** 4
    q_base = t.CP_J(J) / (2.0 * np.pi) * rho * n * n * t.D ** 5
    s = np.clip(s, 0.0, 1.0)
    thrust = t_base * (1.0 - k_t * s + c_fl * s * xi)
    q_aero = q_base * (1.0 + k_p * s)
    return thrust, q_aero


def rotor_wrench_clean(omega, v_axial, rho=RHO0):
    """Momentum-branch wrench with no VRS effects (parity checks, trim of the
    normalization thrust)."""
    return rotor_wrench(omega, v_axial, 0.0, 0.0, rho)


def bus_solve(duty, omega, soc, rho=RHO0):
    """Resolve the sag-coupled bus.  duty, omega: (..., 4); soc: (...).
    Damped fixed point on the pack current (loop gain << 1)."""
    t = tables()
    duty_eff = np.minimum(t.k1 * np.clip(duty, 0.0, 1.0), 1.0)
    emf = t.n_cell * t.ocv_cell(soc)
    i_pack = np.zeros(np.shape(soc))
    i_ch = np.zeros(np.shape(duty))
    for _ in range(6):
        v_bus = emf - i_pack * t.r_pack
        i_ch = np.maximum((duty_eff * v_bus[..., None] - omega / t.Kv_si)
                          / (t.R_m + t.R_esc), 0.0)
        i_ch = np.minimum(i_ch, t.i_esc_max)
        i_pack = 0.6 * i_pack + 0.4 * i_ch.sum(axis=-1)
    v_bus = emf - i_pack * t.r_pack
    v_m = duty_eff * v_bus[..., None] - i_ch * t.R_esc
    return {"I": i_ch, "V_m": v_m, "V_bus": v_bus, "I_pack": i_pack,
            "duty_eff": duty_eff}


def shaft_torque(i_ch):
    t = tables()
    return t.Kt * np.maximum(i_ch - t.I0, 0.0)


def duty_for_current(i_target, omega, v_bus):
    """Invert the chain: duty producing current i_target at speed omega
    (governor: certified current ceiling -> duty ceiling, analytic)."""
    t = tables()
    duty_eff = (omega / t.Kv_si + i_target * (t.R_m + t.R_esc)) \
        / np.maximum(v_bus, 1e-6)
    return np.clip(duty_eff / t.k1, 0.0, 1.0)


def steady_omega(duty, v_axial, soc, s=0.0, rho=RHO0, iters=40):
    """Steady rotor speed: solve Q_m(omega) = Q_a(omega) per channel with a
    damped secant (batched).  The VRS torque inflation shifts the balance."""
    t = tables()
    omega = np.full(np.broadcast(np.asarray(duty), np.asarray(v_axial)).shape,
                    600.0)

    def resid(om):
        bus = bus_solve(duty, om, soc, rho)
        q_m = t.Kt * np.maximum(bus["I"] - t.I0, 0.0)
        _, q_a = rotor_wrench(om, v_axial, s, 0.0, rho)
        return q_m - q_a, bus

    om0 = omega * 0.9 + 1.0
    r0, _ = resid(om0)
    om1 = omega
    for _ in range(iters):
        r1, bus = resid(om1)
        denom = np.where(np.abs(r1 - r0) < 1e-12, 1e-12, r1 - r0)
        om2 = om1 - r1 * (om1 - om0) / denom
        om2 = np.clip(om2, 1.0, 1300.0)
        om0, r0, om1 = om1, r1, 0.5 * om1 + 0.5 * om2
    _, bus = resid(om1)
    return om1, bus
