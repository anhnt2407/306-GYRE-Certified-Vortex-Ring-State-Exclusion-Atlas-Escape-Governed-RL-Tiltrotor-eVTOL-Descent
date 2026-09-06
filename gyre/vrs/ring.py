"""Ring-state intensity model: the new physics of project #298.

Per rotor pair, a scalar intensity s in [0, 1] tracks how developed the
vortex-ring state is.  A quasi-steady severity field s_qs on the normalized
disk-velocity plane (descent ratio Vz_bar = -v_ax / v_h, edgewise ratio
Vx_bar = v_pp / v_h) encodes the classical VRS region (Johnson
NASA/TP-2005-213477; ONERA wake-transport criterion) with smoothstep edges;
the intensity lags it with formation/collapse asymmetry, and the washout
(exit) edge is displaced outward in proportion to the current intensity —
the documented entry/exit hysteresis (escaping needs more edgewise flow
than avoiding entry).

Effects at intensity s (applied by the powertrain/plant):
    thrust   T = T_table(J) * (1 - k_t * s + c_fl * s * xi)
    torque   Q = Q_table(J) * (1 + k_p * s)
with xi a unit-variance OU buffet carrier (band-limited at tau_xi).

Certification interface: the factors of s_qs depend on *disjoint* variables
(B_z on Vz_bar, B_x on Vx_bar), each piecewise monotone, so the supremum of
s_qs over an axis-aligned box in (Vz_bar, Vx_bar) is the product of exact
per-factor suprema — no interval slack.  ``s_qs_sup`` implements that and is
the only path the atlas certificates use.
"""
from __future__ import annotations

import numpy as np

from ..config import VRS

_EPS = 1e-12


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def b_z(zbar):
    """Descent-ratio factor: 0 below onset, 1 in the core, 0 past windmill."""
    up = smoothstep((zbar - VRS.z_on) / (VRS.z_core0 - VRS.z_on))
    dn = 1.0 - smoothstep((zbar - VRS.z_core1) / (VRS.z_wm - VRS.z_core1))
    return up * dn


def x_out_eff(s):
    """Hysteresis-displaced washout edge."""
    return VRS.x_out * (1.0 + VRS.hyst * np.clip(s, 0.0, 1.0))


def b_x(xbar, s):
    """Edgewise washout factor, 1 at axial flow, 0 beyond the (displaced)
    washout edge."""
    xo = x_out_eff(s)
    return 1.0 - smoothstep((xbar - VRS.x_in) / (xo - VRS.x_in))


def s_qs(zbar, xbar, s):
    return b_z(zbar) * b_x(xbar, s)


# ----------------------------------------------------------------------------
# disk kinematics and normalization
# ----------------------------------------------------------------------------
def disk_velocities(ua, w, sigma):
    """Per-pair disk-frame velocities (family conventions, 296 lineage).
    ua: air-relative body-x speed; w: body down speed; sigma: tilt (pi/2 =
    hover).  Returns v_ax_f, v_pp_f, v_ax_r, v_pp_r.  Positive v_ax = flow
    entering the disk from above (climb side); descent makes it negative."""
    cs, sn = np.cos(sigma), np.sin(sigma)
    v_ax_f = ua * cs - w * sn
    v_pp_f = np.abs(ua * sn + w * cs)
    v_ax_r = -w
    v_pp_r = np.abs(ua)
    return v_ax_f, v_pp_f, v_ax_r, v_pp_r


_VH0_CACHE = {}


def v_hover0(rho):
    """Weight-based hover induced velocity v_h = sqrt((m g / 4)/(2 rho A)) —
    the standard chart normalization (Johnson TP-2005-213477 normalizes the
    VRS region by hover induced velocity at the operating weight, a constant
    per loading).  Using the constant kills the vh <- thrust <- omega
    feedback loop that would otherwise manufacture spurious trim branches;
    the certified region bounds then depend on kinematics alone."""
    from ..config import G0
    from ..tables import tables
    key = float(rho)
    if key not in _VH0_CACHE:
        t = tables()
        _VH0_CACHE[key] = float(np.sqrt((t.mass * G0 / 4.0)
                                        / (2.0 * rho * t.A_disc)))
    return _VH0_CACHE[key]


def v_hover(thrust_pair, rho):
    """Momentum hover induced velocity at a given per-rotor thrust (kept for
    diagnostics; the severity normalization uses ``v_hover0``)."""
    from ..tables import tables
    t = tables()
    return np.sqrt(np.maximum(thrust_pair, VRS.t_floor_n)
                   / (2.0 * rho * t.A_disc))


def normalized_plane(v_ax, v_pp, vh):
    zbar = -v_ax / np.maximum(vh, _EPS)
    xbar = v_pp / np.maximum(vh, _EPS)
    return zbar, xbar


# ----------------------------------------------------------------------------
# intensity dynamics
# ----------------------------------------------------------------------------
def s_dot(s, sq, tau_form=None, tau_coll=None):
    """First-order lag with formation/collapse asymmetry."""
    tf = VRS.tau_form if tau_form is None else tau_form
    tc = VRS.tau_coll if tau_coll is None else tau_coll
    tau = np.where(sq > s, tf, tc)
    return (sq - s) / tau


def ou_step(xi, dt, rng):
    """Exact discretization of the unit-variance OU buffet carrier."""
    a = np.exp(-dt / VRS.tau_xi)
    return a * xi + np.sqrt(1.0 - a * a) * rng.standard_normal(xi.shape)


# ----------------------------------------------------------------------------
# certification interface: exact suprema over boxes
# ----------------------------------------------------------------------------
def _bz_sup(z_lo, z_hi):
    """Exact sup of b_z over [z_lo, z_hi] (piecewise monotone: rises to the
    core, flat, falls).  Vectorized over leading dims."""
    z_lo = np.asarray(z_lo, float); z_hi = np.asarray(z_hi, float)
    # if the interval touches the core plateau the sup is 1
    touches = (z_hi >= VRS.z_core0) & (z_lo <= VRS.z_core1)
    # otherwise the interval is entirely on one monotone side
    on_rise = z_hi < VRS.z_core0          # increasing side: sup at z_hi
    val = np.where(on_rise, b_z(z_hi), b_z(z_lo))
    return np.where(touches, 1.0, val)


def _bx_sup(x_lo, s_hi):
    """Exact sup of b_x over [x_lo, x_hi] x [0, s_hi]: decreasing in x, and
    increasing in s (larger intensity keeps the ring attached longer), so the
    sup sits at (x_lo, s_hi)."""
    return b_x(np.asarray(x_lo, float), np.asarray(s_hi, float))


def s_qs_sup(z_lo, z_hi, x_lo, s_hi=1.0):
    """Exact supremum of the quasi-steady severity over the box
    [z_lo, z_hi] x [x_lo, inf) x [0, s_hi].  The two factors depend on
    disjoint variables, so the product of exact suprema is the exact sup."""
    return _bz_sup(z_lo, z_hi) * _bx_sup(x_lo, s_hi)


def ring_budget(s0, s_ub, s_star=None, tau=None):
    """Certified minimum time for the intensity to travel s0 -> s_star while
    the quasi-steady severity is bounded by s_ub (comparison lemma on the
    scalar lag with the fastest credible formation time).  inf where the
    bound never reaches s_star."""
    s_star = VRS.s_star if s_star is None else s_star
    tau = VRS.tau_form_lo if tau is None else tau
    s0 = np.asarray(s0, float); s_ub = np.asarray(s_ub, float)
    reach = s_ub > s_star
    num = np.maximum(s_ub - s0, _EPS)
    den = np.maximum(s_ub - s_star, _EPS)
    t = tau * np.log(num / den)
    return np.where(reach, np.where(s0 >= s_star, 0.0, t), np.inf)
