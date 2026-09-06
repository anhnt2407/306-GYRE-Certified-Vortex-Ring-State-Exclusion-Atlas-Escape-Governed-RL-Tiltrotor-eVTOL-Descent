"""GYRE configuration: every physical constant, certification threshold and
solver knob in one place (family convention, 296-EMBER lineage).

Flight-side constants are read from ``data/kp2_parsed.json`` (first-hand parse
of the authoritative KFDS KP-2C model, ``scripts/parse_kp2.py``).  The
vortex-ring-state layer is GYRE's new physics; its parameters are frozen here
with provenance notes and justified in ``docs/01-proposal.md`` /
``DESIGN.md``.  Certificates are conditional on this frozen disturbance class
and say so in the paper.

Provenance of the VRS constants (engineering calibration, not new aero data):
  * Region edges (normalized by hover induced velocity v_h): the classical
    vortex-ring region of Johnson, "Model for Vortex Ring State Influence on
    Rotorcraft Flight Dynamics", NASA/TP-2005-213477 (onset descent ratio
    ~0.3, deepest extent to ~1.5-2.0, edgewise washout by ~0.9-1.0), and the
    ONERA wake-transport criterion (Jimenez/Taghizad).  GYRE uses smoothstep
    edges spanning those bands.
  * Mean thrust deficit at full intensity K_T ~ 0.2-0.4 and power inflation:
    V-22/HROD and model-rotor test magnitudes reported in the VRS literature
    (Betzina; Johnson TP survey).  Frozen mid-band with an OOD ladder above.
  * Buffet: thrust-fluctuation RMS growing with intensity, band-limited at a
    wake time scale; model tests report ~10-20 % peak excursions mid-VRS.
  * Formation/collapse times: wake development over tens of wake transits
    R/v_h (sub-second to seconds at model scale); collapse on washout is
    faster than formation.  Frozen with [lo, hi] bands used by certificates
    in the sound direction (rise fast / decay slow for upper bounds).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

G0 = 9.80665
RHO0 = 1.225            # sea-level ISA [kg/m^3]


def load_kp2() -> dict:
    with open(DATA / "kp2_parsed.json") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# vortex-ring-state layer (NEW physics, per rotor pair)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class VRSParams:
    """Ring-intensity model s in [0, 1] per rotor pair.

    Quasi-steady severity  s_qs = B_z(Vz_bar) * B_x(Vx_bar; s)  with
    smoothstep edges; first-order intensity lag with formation/collapse
    asymmetry; exit-edge hysteresis via the (1 + hyst * s) displacement of
    the washout edge.  Effects on the pair: mean thrust deficit, power
    inflation, band-limited buffet, all proportional to s.
    """
    # normalized region edges (units of v_h)
    z_on: float = 0.28       # descent ratio where severity starts
    z_core0: float = 0.60    # start of full-severity core
    z_core1: float = 1.20    # end of full-severity core
    z_wm: float = 1.70       # windmill-brake exit (deep descent)
    x_in: float = 0.35       # edgewise ratio where washout starts
    x_out: float = 0.95      # edgewise ratio that fully washes the ring out
    hyst: float = 0.20       # exit-edge displacement factor (entry/exit hysteresis)
    # intensity dynamics [s]
    tau_form: float = 1.10   # formation lag (nominal)
    tau_form_lo: float = 0.70   # fastest credible formation (cert upper bounds)
    tau_form_hi: float = 1.60
    tau_coll: float = 0.45   # collapse lag on exit (nominal)
    tau_coll_lo: float = 0.30
    tau_coll_hi: float = 0.80   # slowest credible collapse (cert upper bounds)
    # effect magnitudes at s = 1
    k_t: float = 0.30        # mean thrust deficit fraction (nominal)
    k_t_hi: float = 0.40     # worst credible deficit (escape DP, OOD ladder)
    k_p: float = 0.18        # torque/power inflation fraction
    c_fl: float = 0.12       # buffet thrust RMS fraction of local mean thrust
    c_fl_hi: float = 0.18
    tau_xi: float = 0.35     # buffet OU correlation time [s]
    # thresholds (the certified ladder)
    s_star: float = 0.35     # exclusion threshold: placarded max intensity
    s_trig: float = 0.22     # escape-commit trigger on the UCB estimate
    s_safe: float = 0.15     # recovered set: intensity at/below this
    margin_cell: float = 0.04    # atlas cell margin on s_qs upper bounds
    # normalization
    t_floor_n: float = 4.0   # thrust floor for v_h normalization [N]

    def tau(self, rising):
        import numpy as np
        return np.where(rising, self.tau_form, self.tau_coll)


@dataclass(frozen=True)
class DescentTaskParams:
    """Steep decelerating approach task and atlas cube."""
    vx_max: float = 22.0
    vx_step: float = 1.0
    vzd_max: float = 12.0        # descent rate grid ceiling [m/s]
    vzd_step: float = 0.5
    sig_step_deg: float = 6.0
    alpha_stall_deg: float = 10.0    # family placard (CL_max at alpha = 10)
    alpha_min_deg: float = -8.0
    uc_max: float = 0.95
    i_mot_cert: float = 72.0     # family per-motor placard [A]
    h_gate: float = 25.0         # hover-capture gate altitude AGL [m]
    gate_vx: float = 1.5
    gate_sink: float = 1.0
    gate_sig_deg: float = 80.0
    tilt_rate_max_dps: float = 15.0  # family tilt-rate cap
    # episode entry ranges (ID)
    h0_range: tuple = (60.0, 120.0)
    v0_range: tuple = (14.0, 24.0)
    gamma0_range_deg: tuple = (-13.0, -5.0)
    wind_id: tuple = (-3.0, 3.0)     # headwind range, nuisance level only
    gust_sd_id: float = 0.6          # OU gust sd [m/s] (wind axis is #289's)


@dataclass(frozen=True)
class GovernorParams:
    s_star_margin: float = 0.05      # governor enforces s_ucb <= s_star - this
    beta_ucb: float = 2.0            # EKF UCB multiplier
    cert_rate_hz: float = 2.0        # placard refresh rate (family convention)
    vz_floor: float = 0.4            # never clamp commanded sink below [m/s]
    lead_time: float = 2.5           # ring-budget lead required before RESTRICT
    hold_recover_s: float = 2.0      # sustained recovered-set dwell to release
    escape_lock_s: float = 1.0       # min escape dwell (anti-chatter)


@dataclass(frozen=True)
class ObserverParams:
    ekf_decim: int = 2               # 100 Hz plant -> 50 Hz EKF (287 convention)
    q_s: float = 0.060               # intensity random-walk PSD [1/s]
    q_baz: float = 1.0e-4            # accel-bias random walk
    r_tq: float = 0.055              # torque residual noise [N m]
    r_az: float = 0.35               # specific-force noise [m/s^2]
    p0_s: float = 0.010
    p0_baz: float = 0.010
    chi2_gate: float = 25.0
    sd_omega: float = 2.0            # rotor-speed sensor noise [rad/s]
    sd_current: float = 0.8          # ESC current telemetry noise [A]
    sd_az: float = 0.15              # accelerometer noise [m/s^2]
    sd_hdot: float = 0.12
    # GLRT incipience detector (windowed innovation energy, m-of-k persist)
    glrt_win: int = 40               # samples at 50 Hz (0.8 s window)
    glrt_thresh: float = 2.6         # energy ratio threshold vs H0 floor
    glrt_m: int = 4
    glrt_k: int = 6
    beta_quant: float = 0.95         # held-out inflation quantile (SLEET lane)


@dataclass(frozen=True)
class SimParams:
    fdm_dt: float = 0.01
    agent_dt: float = 0.10
    episode_t_max: float = 60.0


@dataclass(frozen=True)
class RLParams:
    """Constrained SAC (cost critic + Lagrangian dual), 296 SAC lineage minus
    the Lyapunov critic (that is EMBER's LAC lane)."""
    hidden: tuple = (256, 256)
    gamma: float = 0.99
    tau: float = 5.0e-3
    lr_actor: float = 1.0e-4
    lr_critic: float = 3.0e-4
    lr_lambda: float = 8.0e-4
    cost_budget: float = 0.10        # per-episode expected cost target
    batch_size: int = 256
    buffer_size: int = 400_000
    total_steps: int = 150_000
    start_random: int = 4_000
    seed_list: tuple = (0, 1, 2, 3, 4)


VRS = VRSParams()
TASK = DescentTaskParams()
GOV = GovernorParams()
OBS = ObserverParams()
SIM = SimParams()
RL = RLParams()
