"""Phase B - NSGA-II within the MDF architecture (direct DES evaluation).

Operators for categorical integer genes (no continuous operators, no rounding):
  * sampling  : IntegerRandomSampling - every gene uniform over its level indices;
  * crossover : uniform crossover UX (p = 0.9) - each child gene is copied from one parent, so it is always admissible;
  * mutation  : CategoricalMutation - each gene mutates with p = 1/9 to a DIFFERENT admissible level chosen uniformly;
  * duplicate elimination on the decision vectors (pymoo default), plus the per-run DES cache in problem.py.
All randomness comes from the pymoo random state seeded by `seed`, so a run is reproducible.
Metrics are computed after every generation on the archive of all designs evaluated so far (each by the DES) and on
the current population, against the exact front of Phase A (used only as the reference).
"""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ux import UX
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from ..des.simulation import DES_VERSION
from ..mda.coordinator import MDA_VERSION
from ..sensitivity.common import DV_ORDER, config_id
from .common import MDF_VERSION, OUT_DIR, RUN_DIR, ensure_dirs
from .exact_pareto import nondominated_mask
from .metrics import REF_POINT, bounds_from_front, hypervolume_2d, igd_plus, normalise, recovery
from .problem import CS2MDFProblem, DESEvaluator, decode


class CategoricalMutation(Mutation):
    """Each gene, with probability prob_var, is replaced by a different admissible level chosen uniformly."""

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        X = np.asarray(X).astype(int).copy()
        n_lv = (problem.xu - problem.xl + 1).astype(int)
        pv = self.get_prob_var(problem, size=len(X))
        mask = random_state.random(X.shape) < np.asarray(pv).reshape(-1, 1)
        shift = random_state.integers(1, np.maximum(n_lv, 2), size=X.shape)       # 1 .. n_j - 1
        X[mask] = ((X - problem.xl.astype(int) + shift) % n_lv + problem.xl.astype(int))[mask]
        return X


def make_algorithm(pop_size: int = 100) -> NSGA2:
    return NSGA2(pop_size=pop_size, sampling=IntegerRandomSampling(), crossover=UX(prob=0.9),
                 mutation=CategoricalMutation(prob=1.0, prob_var=1.0 / 9), eliminate_duplicates=True)


def load_reference() -> Dict[str, Any]:
    d = pd.read_csv(os.path.join(OUT_DIR, "exact_pareto_designs.csv"), float_precision="round_trip")
    p = pd.read_csv(os.path.join(OUT_DIR, "exact_pareto_objectives.csv"), float_precision="round_trip")
    b = bounds_from_front(p.TH.values, p.E.values)
    Zn = normalise(p.TH.values, p.E.values, b)
    return {"ids": list(d.config_id), "points": list(zip(p.TH, p.E)), "bounds": b, "Zn": Zn,
            "hv_exact": hypervolume_2d(Zn, REF_POINT)}


def front_metrics(th, e, ids, ref: Dict[str, Any]) -> Dict[str, Any]:
    th, e = np.asarray(th, float), np.asarray(e, float)
    m = nondominated_mask(th, e)
    An = normalise(th[m], e[m], ref["bounds"])
    hv = hypervolume_2d(An, REF_POINT)
    return {"n_nondominated_designs": int(m.sum()), "hv": hv, "hv_gap_pct": 100.0 * (ref["hv_exact"] - hv) / ref["hv_exact"],
            "igd_plus": igd_plus(An, ref["Zn"]),
            **recovery(np.asarray(ids)[m], zip(th[m], e[m]), ref["ids"], ref["points"])}


class Recorder(Callback):
    def __init__(self, evaluator: DESEvaluator, ref: Dict[str, Any], t0: float):
        super().__init__()
        self.ev, self.ref, self.t0, self.rows = evaluator, ref, t0, []

    def notify(self, algorithm):
        c = self.ev.cache
        ids = list(c)
        arch = front_metrics([c[i]["TH"] for i in ids], [c[i]["E"] for i in ids], ids, self.ref)
        pop = algorithm.pop
        F = pop.get("F")
        pids = [config_id(decode(x, algorithm.problem.doms)) for x in pop.get("X")]
        pm = front_metrics(-F[:, 0], F[:, 1], pids, self.ref)
        s = self.ev.stats()
        self.rows.append({"generation": algorithm.n_gen, "requests": s["requests"], "des_calls": s["des_calls"],
                          "duplicates_avoided": s["duplicates_avoided"], "elapsed_s": time.perf_counter() - self.t0,
                          **{f"archive_{k}": v for k, v in arch.items()}, **{f"pop_{k}": v for k, v in pm.items()}})


def run_nsga2(seed: int, pop_size: int = 100, n_gen: int = 100, workers: int = 10, label: Optional[str] = None,
              write: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    ref = load_reference()
    ev = DESEvaluator(workers)
    problem = CS2MDFProblem(ev)
    t0 = time.perf_counter()
    rec = Recorder(ev, ref, t0)
    try:
        res = minimize(problem, make_algorithm(pop_size), ("n_gen", n_gen), seed=seed, callback=rec, verbose=False)
    finally:
        ev.close()
    runtime = time.perf_counter() - t0
    hist = pd.DataFrame(rec.rows)
    evaluated = pd.DataFrame([ev.cache[i] for i in ev.eval_order])
    m = nondominated_mask(evaluated.TH.values, evaluated.E.values)
    archive_front = evaluated[m].sort_values(["TH", "config_id"])
    final_pop = pd.DataFrame([dict(zip(DV_ORDER, decode(x, problem.doms)), TH=-f[0], E=f[1]) for x, f in
                              zip(res.pop.get("X"), res.pop.get("F"))])
    s = ev.stats()
    summary = {
        "label": label or f"seed_{seed}", "seed": seed, "pop_size": pop_size, "n_gen": n_gen, "nominal_evaluations": pop_size * n_gen,
        "workers": workers, "runtime_s": runtime, **s,
        "cpu_time_per_des_call_s": float(evaluated.eval_time_s.mean()),        # one simulation, single core
        "wall_time_per_des_call_s": runtime / max(s["des_calls"], 1),          # whole run incl. NSGA-II and metrics
        "final_archive": hist.filter(like="archive_").iloc[-1].to_dict(),
        "final_population": hist.filter(like="pop_").iloc[-1].to_dict(),
        "hv_exact": ref["hv_exact"], "reference_point_normalised": REF_POINT, "normalisation_bounds": ref["bounds"],
        "algorithm": {"name": "NSGA-II", "library": f"pymoo {pymoo.__version__}", "sampling": "IntegerRandomSampling",
                      "crossover": "UX uniform crossover, p = 0.9", "mutation": "CategoricalMutation, p_var = 1/9",
                      "eliminate_duplicates": True, "termination": f"n_gen = {n_gen}"},
        "evaluator": "direct DES (evaluate_design); per-run cache starts empty; factorial dataset not used",
        "versions": {"MDF": MDF_VERSION, "MDA": MDA_VERSION, "DES": DES_VERSION, "python": platform.python_version()},
    }
    if write:
        d = os.path.join(RUN_DIR, summary["label"])
        os.makedirs(d, exist_ok=True)
        for name, frame in (("history", hist), ("evaluated_designs", evaluated), ("archive_front", archive_front),
                            ("final_population", final_pop)):
            frame.to_csv(os.path.join(d, f"{name}.csv"), index=False, float_format="%.17g")    # round-trip exact
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)
    return {"summary": summary, "history": hist, "evaluated": evaluated, "archive_front": archive_front,
            "final_population": final_pop, "result": res}
