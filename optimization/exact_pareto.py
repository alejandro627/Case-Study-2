"""Phase A - exact discrete Pareto front of the deterministic model from the full-factorial dataset.

Objectives: maximize TH (finished products in 8 h / 8, products/h), minimize E (kWh / 8 h).
Design a dominates b  <=>  TH_a >= TH_b and E_a <= E_b, with at least one strict inequality.

Dominance is decided on the stored full-precision values with exact comparisons, no tolerance: every value is
produced by the same deterministic DES and TH is a multiple of 1.125 (exactly representable). The factorial CSV
stores 16 significant digits (energies within 1 ULP of the in-memory DES values); extract_exact_front verifies that
the front is unchanged when energies are compared at a 1e-9 kWh resolution, so this cannot alter dominance.
Designs that share an objective point are all kept (traceable by config_id).
The factorial dataset is used ONLY as the independent reference, never inside the direct-DES optimizer.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..sensitivity.common import DV_ORDER, baseline, config_id, design_space, domains
from ..sensitivity.factorial import FINAL as FACTORIAL_CSV
from .common import OUT_DIR, ensure_dirs


def nondominated_mask(th: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Non-dominated mask for (max th, min e), O(n log n), exact comparisons, ties kept."""
    th, e = np.asarray(th, float), np.asarray(e, float)
    order = np.lexsort((e, -th))                       # TH descending, then E ascending
    keep = np.zeros(len(th), bool)
    best_e = np.inf                                    # lowest E among designs with strictly higher TH
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and th[order[j]] == th[order[i]]:
            j += 1
        grp = order[i:j]
        e_min = e[grp].min()
        if e_min < best_e:                             # not dominated by a strictly higher-TH design
            keep[grp[e[grp] == e_min]] = True          # within the TH group only the lowest E survives (ties kept)
        best_e = min(best_e, e_min)
        i = j
    return keep


def nondominated_bruteforce(th, e) -> np.ndarray:
    """O(n^2) reference implementation of the same definition (tests)."""
    th, e = np.asarray(th, float), np.asarray(e, float)
    keep = np.ones(len(th), bool)
    for k in range(len(th)):
        dom = (th >= th[k]) & (e <= e[k]) & ((th > th[k]) | (e < e[k]))
        keep[k] = not dom.any()
    return keep


def load_factorial_validated() -> pd.DataFrame:
    df = pd.read_csv(FACTORIAL_CSV, float_precision="round_trip")
    space = design_space()
    ids = [config_id(z) for z in space]
    dom = domains()
    problems = []
    if len(df) != len(space) or df.config_id.nunique() != len(space) or set(df.config_id) != set(ids):
        problems.append("design vectors missing or duplicated")
    if df[list(DV_ORDER)].isna().any().any() or any(not set(df[d]) <= set(dom[d]) for d in DV_ORDER):
        problems.append("incomplete or out-of-domain DV values")
    if [config_id(tuple(r)) for r in df[list(DV_ORDER)].itertuples(index=False)] != list(df.config_id):
        problems.append("config_id inconsistent with DV values")
    if not ((df.status == "evaluated").all() and np.isfinite(df.TH).all() and np.isfinite(df.E).all()):
        problems.append("failed evaluations or invalid responses")
    if not df.feasible.astype(bool).all():
        problems.append(f"{int((~df.feasible.astype(bool)).sum())} infeasible configurations")
    if problems:
        raise ValueError("factorial dataset invalid: " + "; ".join(problems))
    return df


def extract_exact_front(write: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    df = load_factorial_validated()
    mask = nondominated_mask(df.TH.values, df.E.values)
    robust = bool((nondominated_mask(df.TH.values, np.round(df.E.values, 9)) == mask).all())
    if not robust:
        raise RuntimeError("exact front changes at 1e-9 kWh resolution: near-ties require an explicit tolerance decision")
    front = df[mask].copy()
    pts = (front.groupby(["TH", "E"], sort=False).agg(n_designs=("config_id", "size"), config_ids=("config_id", lambda s: ";".join(sorted(s))))
           .reset_index().sort_values("TH").reset_index(drop=True))
    pts.insert(0, "point_id", [f"P{i + 1:02d}" for i in range(len(pts))])
    front = front.merge(pts[["point_id", "TH", "E"]], on=["TH", "E"]).sort_values(["TH", "config_id"]).reset_index(drop=True)
    base_id = config_id(baseline())
    b = df[df.config_id == base_id].iloc[0]
    doms = df[(df.TH >= b.TH) & (df.E <= b.E) & ((df.TH > b.TH) | (df.E < b.E))]
    doms_front = front[front.config_id.isin(doms.config_id)]
    minE = front[front.E == front.E.min()]
    maxTH = front[front.TH == front.TH.max()]
    summary = {
        "source": os.path.relpath(FACTORIAL_CSV, os.path.dirname(OUT_DIR)),
        "n_configurations": int(len(df)), "all_feasible": bool(df.feasible.astype(bool).all()),
        "objectives": "maximize TH (finished products / 8 h, products/h); minimize E (kWh / 8 h)",
        "dominance": "a dominates b iff TH_a >= TH_b and E_a <= E_b with one strict; exact float comparison (no tolerance)",
        "robust_to_1e-9_kWh": robust,
        "storage_precision_note": "factorial CSV stores 16 significant digits (<= 1 ULP from the DES values); NSGA-II points "
                                  "are matched to the reference with |dE| <= 1e-9 kWh and exact TH",
        "n_pareto_designs": int(len(front)), "n_pareto_points": int(len(pts)),
        "max_designs_per_point": int(pts.n_designs.max()),
        "min_energy": {"TH": float(minE.TH.iloc[0]), "E": float(minE.E.iloc[0]), "config_ids": sorted(minE.config_id)},
        "max_throughput": {"TH": float(maxTH.TH.iloc[0]), "E": float(maxTH.E.iloc[0]), "config_ids": sorted(maxTH.config_id)},
        "baseline": {"config_id": base_id, "TH": float(b.TH), "E": float(b.E), "pareto_optimal": bool(base_id in set(front.config_id)),
                     "n_dominating_designs": int(len(doms)), "n_dominating_pareto_designs": int(len(doms_front)),
                     "example_dominating_pareto_designs": [
                         {"config_id": r.config_id, "TH": float(r.TH), "E": float(r.E)} for r in
                         pd.concat([doms_front.nsmallest(1, "E"), doms_front.nlargest(1, "TH")]).drop_duplicates("config_id").itertuples()]},
        "objective_ranges_all_designs": {"TH": [float(df.TH.min()), float(df.TH.max())], "E": [float(df.E.min()), float(df.E.max())]},
    }
    if write:
        cols = ["point_id", "config_id"] + list(DV_ORDER) + ["TH", "E", "TH_rate", "finished_products", "specific_energy_Wh_per_product",
                                                             "util_UR10_pct", "util_UR3e_pct", "util_FANUC_cell_pct",
                                                             "mean_wait_blocking_s_per_cycle", "mean_wait_fanuc_s_per_cycle",
                                                             "fanuc_queue_length_mean", "highest_occupancy"]
        front[cols].to_csv(os.path.join(OUT_DIR, "exact_pareto_designs.csv"), index=False, float_format="%.17g")
        pts.to_csv(os.path.join(OUT_DIR, "exact_pareto_objectives.csv"), index=False, float_format="%.17g")
        with open(os.path.join(OUT_DIR, "pareto_reference_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return {"data": df, "front": front, "points": pts, "summary": summary}
