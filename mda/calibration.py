"""Calibrate the DES parameters of the non-baseline DV alternatives to the reference coefficients.

    python -B -m cs2_mdo.mda.calibration          (from the project root; writes config/alternatives_calibration.json)

Basis: 1 shuttle, 8 h, all other DVs at baseline (the conditions of the coefficient table).
  * Throughput: for time = "calibrated", the steady-state shuttle cycle must be C0 / c_T (C0 = baseline cycle of
    the current DES); the operation time (DV9: the wake-up delay) is found by bisection on the DES repeated-cycle time.
    For time = "speed_up" / "fixed" (DV6, DV8 level 1) the time follows from the physical assumption and the
    coefficient in the table is the DES result; it is checked to within the table's rounding.
    The 8 h product count moves in whole batches (9 products = 1.125 products/h), so the 8 h throughput is reproduced
    to within one batch; the cycle-time (rate) ratio is reproduced exactly.
  * Energy: the 8 h DES energy must be c_E * E0 (E0 = baseline DES energy). Energy is linear in the calibrated power
    (station equipment, collaborative robot mode, UR3e_WS9 or assistance equipment; DV9 idle power factor), which is
    solved exactly from two DES runs.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Callable, Dict

from ..des.config import PACKAGE_DIR, apply_overrides, load_config
from .alternatives import ALTERNATIVES, POLICIES, alternative_id, apply_design, fixed_time
from .design_variables import load_design_variables

CALIBRATION_PATH = os.path.join(PACKAGE_DIR, "config", "alternatives_calibration.json")
ROUNDING_TOL = 0.0005           # coefficients are given with 3 decimals
TIME_KEY = {"ws6_ur10_load_two_trays": "ws6_ur10_load_two_trays_s", "ws9_load_workpieces": "ws9_load_workpieces_s",
            "ws4_ur3e_assembly": "ws4_ur3e_assembly_s", "fanuc_conveyor_sequence": "fanuc_conveyor_sequence_s",
            "ws8_quality_inspection": "ws8_quality_inspection_s", "ws8_ur10_unload_three_trays": "ws8_ur10_unload_three_trays_s"}


def _run(cfg: Dict[str, Any], z: Dict[str, int], params) -> Dict[str, Any]:
    from .coordinator import MDA          # local import: coordinator imports this module
    pol = POLICIES.get(("DV9", z["DV9"]), "standard_powered_operation")
    m = MDA(apply_design(cfg, z, params), energy_policy=pol).run()
    return m.aggregate()["results"]


def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float, tol: float = 1e-9) -> float:
    flo, fhi = f(lo) - target, f(hi) - target
    if flo > 0 or fhi < 0:
        raise ValueError(f"target {target:.4f} s not bracketed: f({lo})={flo + target:.4f}, f({hi})={fhi + target:.4f}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - target
        if abs(fm) <= tol or hi - lo < 1e-12:
            return mid
        lo, hi = (mid, hi) if fm < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def calibrate(write: bool = True) -> Dict[str, Any]:
    spec = load_design_variables()
    z0 = dict(spec["baseline"])
    base = apply_overrides(load_config(), ["simulation.n_shuttles=1", "simulation.horizon_h=8"])
    r0 = _run(base, z0, {})
    C0, E0 = r0["kpis"]["repeated_cycle_time_s_mean"], r0["kpis"]["energy_total_Wh"]
    robot_W = {"UR10": base["energy"]["robot_systems"]["UR10"]["active_W"],
               "UR3e": base["energy"]["robot_systems"]["UR3e"]["active_W"], "FANUC": base["energy"]["fanuc_cell"]["active_W"]}
    out: Dict[str, Any] = {}
    for key in list(ALTERNATIVES) + list(POLICIES):
        dv, lvl = key
        aid = alternative_id(dv, lvl)
        L = spec["design_variables"][dv]["levels"][str(lvl)]
        cT, cE = L["reference_throughput_coefficient"], L["reference_energy_coefficient"]
        z = dict(z0, **{dv: lvl})
        C_t, E_t = C0 / cT, cE * E0
        energy = lambda prm: _run(base, z, {aid: prm})["kpis"]["energy_total_Wh"]
        if key in ALTERNATIVES:
            a = ALTERNATIVES[key]
            rec = {"operation": a["operation"], "mode": a["mode"], "resources": a["resources"], "time_rule": a["time"],
                   "power_rule": a["power"], "baseline_time_s": base["process_times_s"][TIME_KEY[a["operation"]]]}
            params: Dict[str, float] = {}
            if a["time"] == "calibrated":
                cyc = lambda t: _run(base, z, {aid: {"time_s": t, "active_W": 0.0}})["kpis"]["repeated_cycle_time_s_mean"]
                params["time_s"] = _bisect(cyc, C_t, 0.5, 1500.0)
                rec["time_s"] = params["time_s"]
            elif a["time"] == "speed_up":
                rec.update(speed_up=a["speed_up"], time_s=rec["baseline_time_s"] / a["speed_up"])
            else:
                rec.update(sampled_products=a["sampled_products"], time_s=fixed_time(base, a))
            if a["power"] is not None:
                e0, e1 = energy(dict(params, active_W=0.0)), energy(dict(params, active_W=100.0))
                params["active_W"] = (E_t - e0) * 100.0 / (e1 - e0)
                rec["min_energy_ratio_at_0W"] = e0 / E0
                if a["power"] == "robot_mode":
                    rec["robot"], rec["robot_normal_active_W"] = a["robot"], robot_W[a["robot"]]
                elif a["power"] == "robot":
                    rec["robot"], rec["reference_ur3e_active_W"] = "UR3e_WS9", robot_W["UR3e"]
                rec.update(params)
            rec["calibrated_parameters"] = sorted(params)
        else:
            cyc = lambda d: _run(base, z, {aid: {"idle_power_factor": 1.0, "wake_up_delay_s": d}})["kpis"]["repeated_cycle_time_s_mean"]
            d = _bisect(cyc, C_t, 0.0, 300.0)
            e = {f: energy({"idle_power_factor": f, "wake_up_delay_s": d}) for f in (0.0, 1.0)}
            params = {"idle_power_factor": (E_t - e[0.0]) / (e[1.0] - e[0.0]), "wake_up_delay_s": d}
            rec = {"policy": POLICIES[key], "covered": "robots and FANUC cell", "time_rule": "calibrated",
                   "power_rule": "idle_power_factor", "energy_ratio_range": [e[0.0] / E0, e[1.0] / E0],
                   "calibrated_parameters": sorted(params), **params}
        ok = bool(params.get("active_W", 0.0) >= 0 and 0.0 <= params.get("idle_power_factor", 0.0) <= 1.0)
        safe = dict(params, **({"active_W": max(params["active_W"], 0.0)} if "active_W" in params else {}),
                    **({"idle_power_factor": min(max(params["idle_power_factor"], 0.0), 1.0)} if "idle_power_factor" in params else {}))
        k = _run(base, z, {aid: safe})["kpis"]
        rate, en = C0 / k["repeated_cycle_time_s_mean"], k["energy_total_Wh"] / E0
        out[aid] = {"dv": dv, "level": lvl, "alternative": L["alternative"], **rec,
                    "throughput_coefficient": cT, "energy_coefficient": cE, "target_cycle_s": C_t, "target_energy_Wh": E_t,
                    "achieved_cycle_s": k["repeated_cycle_time_s_mean"], "achieved_rate_ratio": rate,
                    "achieved_energy_Wh": k["energy_total_Wh"], "achieved_energy_ratio": en,
                    "achieved_throughput_8h_products_per_h": k["throughput_products_per_h"],
                    "reference_throughput_products_per_h": L["reference_throughput_products_per_h"],
                    "throughput_within_rounding": abs(rate - cT) <= ROUNDING_TOL,
                    "energy_within_rounding": abs(en - cE) <= ROUNDING_TOL,
                    "physically_admissible": ok}
        print(f"  {'OK ' if ok else 'INADMISSIBLE '}{aid}: {', '.join(f'{p}={v:.4f}' for p, v in params.items()) or 'no free parameter'}; "
              f"rate ratio {rate:.4f} (table {cT}), energy ratio {en:.4f} (table {cE})")
    res = {"meta": {"date": datetime.date.today().isoformat(), "generated_by": "cs2_mdo/mda/calibration.py",
                    "basis": "1 shuttle, 8 h, other DVs at baseline; targets = coefficients of config/design_variables.json v"
                             + spec["meta"]["version"],
                    "method": __doc__.strip().split("\n\n", 1)[1],
                    "baseline_cycle_s": C0, "baseline_energy_Wh": E0, "rounding_tolerance": ROUNDING_TOL},
           "alternatives": out}
    if write:
        with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    return res


def load_calibration(path: str = CALIBRATION_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calibrated_params(cal: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return {aid: {k: a[k] for k in a.get("calibrated_parameters", [])} for aid, a in cal["alternatives"].items()}


if __name__ == "__main__":
    calibrate()
    print("written", CALIBRATION_PATH)
