"""Edge-clamped linear interpolation kernels and the parsed KP-2C table set
(own implementation, family style: ``searchsorted`` + gather, scalars or (N,)
arrays).

GYRE delta vs 296: the CT/CP advance-ratio tables are linearly extended to
J = -0.45 on the descent side (the family precedent extends to -0.30; steep
descent at hover rotor speeds reaches J ~ -0.35).  The extension is the
*momentum branch* only — everything the momentum picture misses in the ring
band is carried by the VRS intensity layer (``gyre/vrs/ring.py``), which
multiplies these tables.  The extension slope is frozen at the table's own
end slope; the deep-descent side is exactly where the intensity layer is
active, so certificate soundness never rests on the extension alone.
"""
from __future__ import annotations

import numpy as np

from .config import load_kp2


class Interp1:
    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x not strictly increasing")

    def __call__(self, q):
        q = np.asarray(q, float)
        qc = np.clip(q, self.x[0], self.x[-1])
        i = np.clip(np.searchsorted(self.x, qc) - 1, 0, len(self.x) - 2)
        t = (qc - self.x[i]) / (self.x[i + 1] - self.x[i])
        return (1 - t) * self.y[i] + t * self.y[i + 1]

    def deriv(self, q):
        q = np.asarray(q, float)
        qc = np.clip(q, self.x[0], self.x[-1])
        i = np.clip(np.searchsorted(self.x, qc) - 1, 0, len(self.x) - 2)
        return (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])


def _extend(J, C, j_lo, j_hi):
    """Linear extension on both ends with the table's own end slopes."""
    J = list(map(float, J)); C = list(map(float, C))
    sl_lo = (C[1] - C[0]) / (J[1] - J[0])
    sl_hi = (C[-1] - C[-2]) / (J[-1] - J[-2])
    return ([j_lo] + J + [j_hi],
            [C[0] + sl_lo * (j_lo - J[0])] + C + [C[-1] + sl_hi * (j_hi - J[-1])])


class KP2Tables:
    """Longitudinal slice (beta = 0) of the parsed KP-2C database plus the
    propeller, motor, ESC and battery maps."""

    J_LO = -0.45     # GYRE deep-descent extension (family used -0.30)
    J_HI = 1.05

    def __init__(self):
        d = load_kp2()
        self.raw = d
        b0 = d["tables2d"]["CZ"]["cols"].index(0.0)

        def slice_b0(key):
            t = d["tables2d"][key]
            return Interp1(t["rows"], [row[b0] for row in t["vals"]])

        self.CZ = slice_b0("CZ")
        self.CX = slice_b0("CX")
        self.CMY = slice_b0("CMY")

        ct = d["ctrl_tables"]

        def ctrl(key):
            t = ct[key]
            return Interp1(t["x"], t["y"])

        # symmetric ruddervator: left and right tables applied at the same
        # deflection (exactly what JSBSim sums in the axis buildup)
        self.CZ_dlv, self.CZ_drv = ctrl("CZdlv"), ctrl("CZdrv")
        self.CX_dlv, self.CX_drv = ctrl("CXdlv"), ctrl("CXdrv")
        self.CMY_dlv, self.CMY_drv = ctrl("CMYdlv"), ctrl("CMYdrv")

        dampers = d["dampers"]
        self.CXq = dampers["CXq"]; self.CZq = dampers["CZq"]
        self.CMYq = dampers["CMYq"]

        p = d["propeller"]
        self.D = p["D_m"]
        self.A_disc = np.pi * self.D ** 2 / 4.0
        jt, ct_y = _extend(p["CT_J"]["x"], p["CT_J"]["y"], self.J_LO, self.J_HI)
        jp, cp_y = _extend(p["CP_J"]["x"], p["CP_J"]["y"], self.J_LO, self.J_HI)
        self.CT_J = Interp1(jt, ct_y)
        self.CP_J = Interp1(jp, cp_y)
        self.I_rotor = p["Ixx_kgm2"] + 1.35e-4   # prop + motor-rotor inertia

        b = d["battery"]
        self.ocv_cell = Interp1(b["ocv_soc"], b["ocv_v_cell"])
        self.n_cell = b["ncell"]
        wiring = d["battery_wiring"]
        self.series, self.parallel = wiring["series"], wiring["parallel"]
        self.r_pack = b["r_cell"] * self.n_cell * self.series / self.parallel
        self.cap_As = b["cap_mAh"] * 3.6 * self.parallel

        m = d["metrics"]
        self.S = m["S_m2"]; self.b = m["b_m"]; self.cbar = m["cbar_m"]
        self.mass = d["mass"]["empty_kg"]; self.Iyy = d["mass"]["Iyy"]

        cg = np.array(d["cg_struct_m"]); rp = np.array(d["aerorp_struct_m"])
        # structural -> body: x_b = -(x_s - x_cg), z_b = -(z_s - z_cg)
        self.r_rp_body = np.array([-(rp[0] - cg[0]), 0.0, -(rp[2] - cg[2])])
        rot = d["rotors"]
        self.r_rotor_body = np.array(
            [[-(r["loc_struct_m"][0] - cg[0]), r["loc_struct_m"][1],
              -(r["loc_struct_m"][2] - cg[2])] for r in rot])
        # channel order [FR(tilt), RL, FL(tilt), RR] -> pair index F/R
        self.tilting = np.array([True, False, True, False])

        mo = d["motor"]
        self.Kt = mo["Kt"]
        self.Kv_si = mo["Kv_rpm_per_V"] * 2.0 * np.pi / 60.0
        self.R_m = mo["R"]; self.I0 = mo["i_idle"]; self.i_peak = mo["i_peak"]
        e = d["esc"]
        self.R_esc = e["R"]; self.k1 = e["k1"]; self.i_esc_max = e["i_max"]


TAB = None


def tables() -> KP2Tables:
    global TAB
    if TAB is None:
        TAB = KP2Tables()
    return TAB
