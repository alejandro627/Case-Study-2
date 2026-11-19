"""Phase C of MDF 1.0: comparison and statistics of the NSGA-II runs (all seeds) against the exact Pareto front."""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .common import OUT_DIR, RUN_DIR, ensure_dirs

SPACE = 26244
KEY = ["des_calls", "requests", "duplicates_avoided", "runtime_s", "archive_hv_gap_pct", "archive_igd_plus",
       "archive_designs_recovered", "archive_designs_recovered_pct", "archive_points_recovered_pct",
       "archive_false_pareto_designs", "pop_hv_gap_pct", "pop_igd_plus", "pop_designs_recovered", "pop_false_pareto_designs",
       "des_calls_to_hv_gap_1pct", "des_calls_to_hv_gap_0_1pct", "des_calls_to_full_recovery"]


def load_runs() -> List[Dict[str, Any]]:
    runs = []
    for d in sorted(glob.glob(os.path.join(RUN_DIR, "seed_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
        h = pd.read_csv(os.path.join(d, "history.csv"), float_precision="round_trip")
        runs.append({"dir": d, "summary": s, "history": h})
    return runs


def _first(h: pd.DataFrame, cond) -> float:
    hit = h[cond(h)]
    return float(hit.des_calls.iloc[0]) if len(hit) else np.nan


def compare_runs(write: bool = True) -> pd.DataFrame:
    ensure_dirs()
    # size of the exact front of the current model
    n_front = len(pd.read_csv(os.path.join(OUT_DIR, "exact_pareto_designs.csv")))
    rows = []
    for r in load_runs():
        s, h = r["summary"], r["history"]
        last = h.iloc[-1]
        rows.append({
            "seed": s["seed"], "generations": int(last.generation), "requests": s["requests"], "des_calls": s["des_calls"],
            "duplicates_avoided": s["duplicates_avoided"], "des_calls_pct_of_space": 100.0 * s["des_calls"] / SPACE,
            "runtime_s": s["runtime_s"], "cpu_time_per_des_call_s": s["cpu_time_per_des_call_s"],
            "wall_time_per_des_call_s": s["wall_time_per_des_call_s"], "infeasible": len(s["infeasible_designs"]),
            **{c: last[c] for c in h.columns if c.startswith(("archive_", "pop_"))},
            "des_calls_to_hv_gap_1pct": _first(h, lambda x: x.archive_hv_gap_pct <= 1.0),
            "des_calls_to_hv_gap_0_1pct": _first(h, lambda x: x.archive_hv_gap_pct <= 0.1),
            "des_calls_to_full_recovery": _first(h, lambda x: x.archive_designs_recovered >= n_front),
        })
    df = pd.DataFrame(rows)
    if write and len(df):
        df.to_csv(os.path.join(OUT_DIR, "optimisation_comparison.csv"), index=False, float_format="%.17g")
        statistics(df).to_csv(os.path.join(OUT_DIR, "optimisation_statistics.csv"), index=False, float_format="%.17g")
    return df


def statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in KEY:
        v = df[k].astype(float)
        rows.append({"metric": k, "mean": v.mean(), "std": v.std(ddof=1) if v.notna().sum() > 1 else np.nan,
                     "min": v.min(), "max": v.max(), "n_seeds_defined": int(v.notna().sum())})
    return pd.DataFrame(rows)


def union_recovery() -> Dict[str, Any]:
    ex = pd.read_csv(os.path.join(OUT_DIR, "exact_pareto_designs.csv"))
    found = set()
    per_seed = {}
    for r in load_runs():
        a = pd.read_csv(os.path.join(r["dir"], "archive_front.csv"))
        got = set(a.config_id) & set(ex.config_id)
        per_seed[r["summary"]["seed"]] = sorted(set(ex.config_id) - got)
        found |= got
    return {"union_recovered": len(found), "missed_by_seed": per_seed,
            "never_recovered": sorted(set(ex.config_id) - found)}

