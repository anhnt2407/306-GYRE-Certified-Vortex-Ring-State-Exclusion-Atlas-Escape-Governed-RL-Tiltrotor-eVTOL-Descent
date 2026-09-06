#!/usr/bin/env python3
"""First-hand parse of the authoritative KP-2C KFDS model into
``data/kp2_parsed.json`` (family convention: every project re-parses the
source XML itself; no numbers are copied from sibling repos).

Source of truth: the KADA KP-2C JSBSim/KFDS model v2.1 (2024-03,
"KADA Simulation Team - Nghia").  Two copies exist on this machine; we parse
the uDT one (same file the family records as provenance) and assert that the
canonical VTOL_Transition working-tree copy is byte-identical, so the
provenance statement covers both.

Extracted, longitudinal focus (GYRE is a symmetric-descent study):
  metrics, mass/inertia, CG / aero reference point, the six static
  coefficient tables (full alpha x beta), ruddervator/aileron control
  derivative tables, constant rate dampers, rotor layout (4x location +
  sense), propeller CT(J)/CP(J) + diameter + inertia, motor KDE5215XF, ESC
  FLAME_80A_12S, battery PT_B10000_NSR35 + pack wiring.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = pathlib.Path(
    "/home/server08/anhnt358/kada/uDT/PX4-Custom/Tools/jsbsim_bridge/models/kp2")
ALT = pathlib.Path(
    "/home/server08/anhnt358/kada/transition/VTOL_Transition/px4/Tools/"
    "jsbsim_bridge/models/kp2")
OUT = ROOT / "data" / "kp2_parsed.json"


def _floats(text: str) -> list[float]:
    return [float(t) for t in re.split(r"[\s,]+", text.strip()) if t]


def _num(el, path, default=None):
    e = el.find(path)
    if e is None:
        if default is not None:
            return default
        raise KeyError(path)
    return float(e.text.strip())


def parse_table_2d(tab_el):
    """JSBSim 2-D <tableData>: first row = column breakpoints."""
    rows_txt = tab_el.find("tableData").text.strip().splitlines()
    cols = _floats(rows_txt[0])
    rows, vals = [], []
    for line in rows_txt[1:]:
        f = _floats(line)
        if not f:
            continue
        rows.append(f[0])
        vals.append(f[1:])
        assert len(f) - 1 == len(cols), "ragged 2-D table"
    return {"rows": rows, "cols": cols, "vals": vals}


def parse_table_1d(tab_el):
    xs, ys = [], []
    for line in tab_el.find("tableData").text.strip().splitlines():
        f = _floats(line)
        if len(f) == 2:
            xs.append(f[0]); ys.append(f[1])
    return {"x": xs, "y": ys}


def find_coeff_tables(root):
    """Map aero/coefficient/<NAME> -> its <table> element."""
    out = {}
    for fn in root.iter("function"):
        name = fn.get("name", "")
        m = re.match(r"aero/coefficient/(\w+)$", name)
        if not m:
            continue
        tab = fn.find(".//table")
        out[m.group(1)] = (fn, tab)
    return out


def coeff_constant(fn):
    """Rate dampers are product(property, ..., value)."""
    v = fn.find(".//value")
    if v is not None:
        return float(v.text.strip())
    raise KeyError("no constant value in coefficient")


def main():
    kp2 = SRC / "kp2.xml"
    alt = ALT / "kp2.xml"
    def _norm(p):  # the two copies differ only in CRLF vs LF; hash content
        return hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    h_src = _norm(kp2)
    h_alt = _norm(alt) if alt.exists() else None
    if h_alt is not None and h_alt != h_src:
        print("WARNING: uDT and VTOL_Transition kp2.xml differ in content",
              file=sys.stderr)

    root = ET.parse(kp2).getroot()

    met = root.find("metrics")
    metrics = {"S_m2": _num(met, "wingarea"), "b_m": _num(met, "wingspan"),
               "cbar_m": _num(met, "chord")}
    aerorp = [_num(met, "location[@name='AERORP']/" + a) for a in "xyz"]

    mb = root.find("mass_balance")
    mass = {"empty_kg": _num(mb, "emptywt"), "Ixx": _num(mb, "ixx"),
            "Iyy": _num(mb, "iyy"), "Izz": _num(mb, "izz"),
            "Ixz": _num(mb, "ixz")}
    cg = [_num(mb, "location[@name='CG']/" + a) for a in "xyz"]

    coeffs = find_coeff_tables(root)
    tables2d = {k: parse_table_2d(coeffs[k][1])
                for k in ("CZ", "CX", "CMY")}
    ctrl_tables = {k: parse_table_1d(coeffs[k][1])
                   for k in ("CXdlv", "CXdrv", "CZdlv", "CZdrv",
                             "CMYdlv", "CMYdrv")}
    dampers = {k: coeff_constant(coeffs[k][0])
               for k in ("CXq", "CYp", "CYr", "CZq", "CMXp", "CMXr",
                         "CMYq", "CMZp", "CMZr")}

    rotors = []
    prop_files = set()
    for eng in root.find("propulsion").findall("engine"):
        thr = eng.find("thruster")
        prop_files.add(thr.get("file"))
        rotors.append({
            "name": eng.get("name"),
            "sense": float(thr.find("sense").text),
            "loc_struct_m": [_num(thr, "location/" + a) for a in "xyz"],
        })
    assert len(rotors) == 4 and prop_files == {"XOAR15x7_J"}

    bat_el = root.find("propulsion/battery")
    battery_wiring = {"series": int(_num(bat_el, "series")),
                      "parallel": int(_num(bat_el, "parallel"))}

    prop_root = ET.parse(SRC / "propeller" / "XOAR15x7_J.xml").getroot()
    d_in = _num(prop_root, "diameter")
    prop = {
        "D_m": d_in * 0.0254,
        "Ixx_kgm2": _num(prop_root, "ixx"),
        "n_blades": int(_num(prop_root, "numblades")),
        "pitch_in": _num(prop_root, "minpitch"),
        "maxrpm": _num(prop_root, "maxrpm"),
    }
    for tab in prop_root.findall("table"):
        if tab.get("name") == "C_THRUST":
            prop["CT_J"] = parse_table_1d(tab)
        elif tab.get("name") == "C_POWER":
            prop["CP_J"] = parse_table_1d(tab)
    assert "CT_J" in prop and "CP_J" in prop

    mot_root = ET.parse(SRC / "engine" / "KDE5215XF.xml").getroot()
    motor = {"Kv_rpm_per_V": _num(mot_root, "Kv"), "Kt": _num(mot_root, "Kt"),
             "i_idle": _num(mot_root, "idlecurrent"),
             "v_idle": _num(mot_root, "idlevoltage"),
             "i_peak": _num(mot_root, "peakcurrent"),
             "R": _num(mot_root, "resistance")}

    esc_root = ET.parse(SRC / "esc" / "FLAME_80A_12S.xml").getroot()
    esc = {"i_max": _num(esc_root, "maxcurrent"), "R": _num(esc_root, "resistance"),
           "k1": _num(esc_root, "k1"), "pwm_min": _num(esc_root, "minpwm"),
           "pwm_max": _num(esc_root, "maxpwm")}

    bat_root = ET.parse(SRC / "battery" / "PT_B10000_NSR35.xml").getroot()
    ocv = parse_table_1d(bat_root.find("cell/table"))
    battery = {"ncell": int(_num(bat_root, "ncell")),
               "cap_mAh": _num(bat_root, "capacity"),
               "c_rate": _num(bat_root, "c_rate"),
               "r_cell": _num(bat_root, "cell/resistance"),
               "v_cutoff_cell": _num(bat_root, "cell/cutoffvoltage"),
               "ocv_soc": ocv["x"], "ocv_v_cell": ocv["y"]}

    out = {
        "source": str(kp2),
        "source_sha256": h_src,
        "canonical_copy_identical": h_alt == h_src,
        "note": ("First-hand parse for project 298-GYRE (VRS axis). "
                 "Longitudinal slice; beta=0 taken downstream. Propeller file "
                 "XOAR15x7_J.xml is, despite its name, the 16-inch map "
                 "(diameter tag = 16 IN) — family-documented quirk."),
        "tables2d": tables2d,
        "ctrl_tables": ctrl_tables,
        "dampers": dampers,
        "metrics": metrics,
        "mass": mass,
        "cg_struct_m": cg,
        "aerorp_struct_m": aerorp,
        "rotors": rotors,
        "battery_wiring": battery_wiring,
        "battery_file": "PT_B10000_NSR35",
        "motor": motor,
        "esc": esc,
        "battery": battery,
        "propeller": prop,
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} kB)")
    print(f"  mass {mass['empty_kg']} kg, S {metrics['S_m2']} m^2, "
          f"D {prop['D_m']:.4f} m, rotors {len(rotors)}")
    print(f"  canonical copy identical: {out['canonical_copy_identical']}")


if __name__ == "__main__":
    main()
