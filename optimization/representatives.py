"""Phase D - representative configurations R0-R3 (and the Pareto designs that dominate the baseline).

R0 baseline (1,3,1,3,3,1,3,2,1); R1 minimum-energy Pareto design; R2 maximum-throughput Pareto design;
R3 compromise: Pareto design minimising the Euclidean distance to the ideal point in objective space normalised by the
ranges of the exact front,
    d_i = sqrt( ((TH_max - TH_i) / (TH_max - TH_min))^2 + ((E_i - E_min) / (E_max - E_min))^2 ).
This equal weighting in normalised objective space is a transparent convention, not a universally optimal design.
Every representative is re-simulated with the DES to report utilisations and bottleneck indicators (and to confirm the
dataset values); nothing is invented beyond the DES outputs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..mda.coordinator import evaluate_design
from ..sensitivity.common import DV_ORDER, as_dict, baseline, config_id
from .common import OUT_DIR, ensure_dirs

DV_LABEL = {"DV1": "shuttles", "DV2": "tray loading", "DV3": "workpiece setup", "DV4": "WS4 assembly",
            "DV5": "FANUC assembly", "DV6": "QC assistance", "DV7": "dispatch", "DV8": "inspection policy",
            "DV9": "energy management"}


def _front() -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUT_DIR, "exact_pareto_designs.csv"), float_precision="round_trip")


def choose() -> Dict[str, str]:
    f = _front()
    th_max, th_min, e_max, e_min = f.TH.max(), f.TH.min(), f.E.max(), f.E.min()
    if th_max == th_min or e_max == e_min:
        raise ValueError("zero objective range on the Pareto front: compromise distance undefined")
    d = np.sqrt(((th_max - f.TH) / (th_max - th_min)) ** 2 + ((f.E - e_min) / (e_max - e_min)) ** 2)
    f = f.assign(d_ideal=d)
    pick = lambda s: s.sort_values("config_id").iloc[0].config_id             # deterministic tie-break
    return {"R0": config_id(baseline()), "R1": pick(f[f.E == e_min]), "R2": pick(f[f.TH == th_max]),
            "R3": pick(f[f.d_ideal == f.d_ideal.min()]), "_d_min": float(d.min())}


def describe(cid: str) -> Dict[str, Any]:
    z = tuple(int(c) for c in cid[1:])
    out = evaluate_design(as_dict(z))
    k = out["results"]["kpis"]
    act = out["results"]["activity"].set_index("resource")
    st = out["results"]["stations"].set_index("station")
    shut = act[act.index.str.startswith("Shuttle_")]
    ops = [r for r in act.index if r.startswith("Operator_")]
    return {
        "config_id": cid, **as_dict(z), "TH": k["throughput_products_per_h"], "E": k["energy_total_kWh"],
        "finished_products": k["finished_products_storage_b"], "Wh_per_product": k["specific_energy_Wh_per_product"],
        "TH_rate": 9 * 3600.0 / k["mean_batch_completion_interval_s"], "feasible": bool(out["feasible"]),
        "util_UR10_pct": act.at["UR10", "active_pct"], "util_UR3e_pct": act.at["UR3e", "active_pct"],
        "util_FANUC_cell_pct": act.at["FANUC_cell", "active_pct"], "util_shuttles_mean_pct": float(shut.active_pct.mean()),
        "util_operators": "; ".join(f"{o} {act.at[o, 'active_pct']:.1f} %" for o in ops if act.at[o, "active_s"] > 0),
        "util_UR3e_WS9_pct": act.at["UR3e_WS9", "active_pct"] if "UR3e_WS9" in act.index else np.nan,
        "mean_cycle_time_s": k["mean_cycle_time_s"], "blocking_wait_s_per_cycle": k["mean_wait_blocking_s_per_cycle"],
        "fanuc_sync_wait_s_per_cycle": k["mean_wait_fanuc_sync_s_per_cycle"],
        "resource_wait_s_per_cycle": k["mean_wait_resource_s_per_cycle"],
        "fanuc_queue_length_mean": k["fanuc_queue_length_mean"], "highest_occupancy": k["highest_occupancy"],
        "S4_occupancy_pct": st.at["S4", "occupancy_pct_of_capacity"], "S6_occupancy_pct": st.at["S6", "occupancy_pct_of_capacity"],
        "energy_shuttles_Wh": k["energy_shuttles_Wh"], "energy_robot_systems_Wh": k["energy_robot_systems_Wh"],
        "energy_workstation_equipment_Wh": k["energy_workstation_equipment_Wh"],
        "energy_fixed_Wh": k["energy_montrac_infrastructure_Wh"] + k["energy_continuous_equipment_Wh"],
    }


def select_representatives(write: bool = True) -> pd.DataFrame:
    ensure_dirs()
    ch = choose()
    rows = []
    for tag, desc in (("R0", "reference baseline"), ("R1", "minimum-energy Pareto design"),
                      ("R2", "maximum-throughput Pareto design"), ("R3", "compromise Pareto design (min distance to ideal)")):
        rows.append({"representative": tag, "description": desc, **describe(ch[tag])})
    df = pd.DataFrame(rows)
    b = df.iloc[0]
    df["dTH_pct_vs_R0"] = 100.0 * (df.TH - b.TH) / b.TH
    df["dE_pct_vs_R0"] = 100.0 * (df.E - b.E) / b.E
    df["dWh_per_product_pct_vs_R0"] = 100.0 * (df.Wh_per_product - b.Wh_per_product) / b.Wh_per_product
    front = _front()
    df["pareto_optimal"] = df.config_id.isin(front.config_id)
    ref = pd.read_csv(os.path.join(os.path.dirname(OUT_DIR), "sensitivity", "factorial_results.csv"),
                      float_precision="round_trip").set_index("config_id")
    df["matches_factorial"] = [r.TH == ref.at[r.config_id, "TH"] and abs(r.E - ref.at[r.config_id, "E"]) <= 1e-9
                               for r in df.itertuples()]
    dom = front[(front.TH >= b.TH) & (front.E <= b.E)].copy()
    dom["dTH_pct_vs_R0"] = 100.0 * (dom.TH - b.TH) / b.TH
    dom["dE_pct_vs_R0"] = 100.0 * (dom.E - b.E) / b.E
    if write:
        df.to_csv(os.path.join(OUT_DIR, "representative_designs.csv"), index=False, float_format="%.17g")
        dom[["point_id", "config_id"] + list(DV_ORDER) + ["TH", "E", "dTH_pct_vs_R0", "dE_pct_vs_R0"]].to_csv(
            os.path.join(OUT_DIR, "pareto_designs_dominating_baseline.csv"), index=False, float_format="%.17g")
        with open(os.path.join(OUT_DIR, "representatives_selection.json"), "w", encoding="utf-8") as fh:
            json.dump({"choice": ch, "compromise_rule": "min Euclidean distance to the ideal point, objectives normalised by "
                                                         "the exact-front ranges (equal weights)"}, fh, indent=2)
    return df
