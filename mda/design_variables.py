"""Design variables DV1-DV9: domain validation and DES-support check.

The reference throughput / energy values and coefficients of config/design_variables.json are used ONLY as
calibration targets of single alternatives (mda/calibration.py). This module never combines them: no
additive or multiplicative response model, no surrogate; combinations are computed by the DES.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..des.config import PACKAGE_DIR

DV_PATH = os.path.join(PACKAGE_DIR, "config", "design_variables.json")


def load_design_variables(path: str = DV_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def n_configurations(spec: Dict[str, Any]) -> int:
    return math.prod(len(d["domain"]) for d in spec["design_variables"].values())


def validate_design_vector(z: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    """Domain check (design-space definition, not an operational constraint). Returns error messages."""
    dvs = spec["design_variables"]
    errors = [f"missing design variable {k}" for k in dvs if k not in z]
    errors += [f"unknown design variable {k}" for k in z if k not in dvs]
    for k, v in z.items():
        if k in dvs and (isinstance(v, bool) or not isinstance(v, int) or v not in dvs[k]["domain"]):
            errors.append(f"{k} = {v!r} outside its domain {dvs[k]['domain']}")
    return errors


SUPPORTED = ("implemented", "implemented_calibrated")


def level_status(dv: str, level: int, spec: Dict[str, Any], cal: Dict[str, Any] = None) -> Tuple[str, str]:
    """(status, representation or reason). A calibrated level is supported only if its calibration is admissible."""
    s = spec["design_variables"][dv]["levels"][str(level)]["des_support"]
    if s["status"] == "implemented_calibrated":
        c = (cal or {}).get("alternatives", {}).get(s["alternative_id"])
        if c is None:
            return "not_implemented", "no calibration available (run python -B -m cs2_mdo.mda.calibration)"
        if not c["physically_admissible"]:
            what = (f"equipment / robot active power {c['active_W']:.1f} W" if "active_W" in c
                    else f"idle power factor {c['idle_power_factor']:.3f}")
            return "calibration_inadmissible", (f"the energy coefficient {c['energy_coefficient']} needs {what} (< 0): "
                                                f"minimum achievable energy ratio {c.get('min_energy_ratio_at_0W', float('nan')):.4f}")
    return s["status"], s.get("des_representation", s.get("reason"))


def support_table(z: Dict[str, int], spec: Dict[str, Any], cal: Dict[str, Any] = None) -> pd.DataFrame:
    rows = []
    for k, d in spec["design_variables"].items():
        lv = d["levels"][str(z[k])]
        status, text = level_status(k, z[k], spec, cal)
        rows.append({"design_variable": k, "name": d["name"], "primary_discipline": d["primary_discipline"], "level": z[k],
                     "alternative": lv["alternative"], "is_baseline_level": lv["is_baseline_level"],
                     "des_status": status, "des_representation_or_reason": text})
    return pd.DataFrame(rows)


def des_mapping(z: Dict[str, int], spec: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Configuration overrides and energy policy of a fully supported design vector."""
    overrides: Dict[str, Any] = {}
    policy = "standard_powered_operation"
    for k, d in spec["design_variables"].items():
        s = d["levels"][str(z[k])]["des_support"]
        if s["status"] not in SUPPORTED:
            raise ValueError(f"{k} level {z[k]} is not implemented in the DES")
        overrides.update(s.get("config_overrides", {}))
        policy = s.get("energy_policy", policy)
    return overrides, policy


def reference_values_table(spec: Dict[str, Any]) -> pd.DataFrame:
    """The supplied reference table, one row per DV alternative (stored values, unmodified)."""
    return pd.DataFrame([{"design_variable": k, "name": d["name"], "primary_discipline": d["primary_discipline"],
                          "level": int(l), "alternative": v["alternative"],
                          "reference_throughput_products_per_h": v["reference_throughput_products_per_h"],
                          "reference_energy_kWh_8h": v["reference_energy_kWh_8h"],
                          "reference_throughput_coefficient": v.get("reference_throughput_coefficient"),
                          "reference_energy_coefficient": v.get("reference_energy_coefficient"),
                          "is_baseline_level": v["is_baseline_level"],
                          "des_status": v["des_support"]["status"]}
                         for k, d in spec["design_variables"].items() for l, v in d["levels"].items()])
