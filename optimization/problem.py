"""MDF problem: direct DES evaluation of discrete design vectors for NSGA-II (pymoo).

Every objective value of the primary experiment comes from a DES run (cs2_mdo.mda.coordinator.evaluate_design via
cs2_mdo.sensitivity.common.evaluate_record): the six disciplines run in one fresh SimPy environment; couplings are
resolved by the event-driven simulation (no fixed-point iterations). The per-run cache is EMPTY at the start and is
filled only by these DES runs - nothing is preloaded from the factorial dataset - so repeated requests of the same
design within a run are counted but not simulated again.

Encoding: gene j is the index of the level in the domain of DV j (0 .. n_j - 1), decoded to the actual level.
Objectives (minimised): F = [-TH, E]; TH = finished products in 8 h / 8 (products/h), E = kWh / 8 h.
Constraint: G = number of violated feasibility conditions C1-C6 (feasible <=> G <= 0).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.core.problem import Problem

from ..sensitivity.common import CONDITIONS, DV_ORDER, config_id, domains, evaluate_record


def decode(x, doms: Dict[str, List[int]]) -> Tuple[int, ...]:
    return tuple(int(doms[d][int(round(v))]) for d, v in zip(DV_ORDER, x))


class DESEvaluator:
    """Direct DES evaluation with a per-run cache and exact counting of physical simulations."""

    def __init__(self, workers: int = 1):
        self.workers = workers
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.eval_order: List[str] = []
        self.requests = 0
        self.des_calls = 0
        self.des_wall_s = 0.0
        self.infeasible: List[str] = []
        self._pool: Optional[ProcessPoolExecutor] = None

    def _pool_get(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self._pool

    def evaluate_many(self, zs: List[Tuple[int, ...]]) -> List[Dict[str, Any]]:
        self.requests += len(zs)
        missing = list(dict.fromkeys(z for z in zs if config_id(z) not in self.cache))
        if missing:
            t0 = time.perf_counter()
            if self.workers > 1 and len(missing) > 1:
                recs = list(self._pool_get().map(evaluate_record, missing, chunksize=1))
            else:
                recs = [evaluate_record(z) for z in missing]
            self.des_wall_s += time.perf_counter() - t0
            for r in recs:
                if r["status"] != "evaluated":
                    raise RuntimeError(f"DES evaluation of {r['config_id']} failed or is unsupported: "
                                       f"{r['status']} {r.get('error', '')}")
                r["des_call_index"] = self.des_calls
                self.des_calls += 1
                self.cache[r["config_id"]] = r
                self.eval_order.append(r["config_id"])
                if not r["feasible"]:
                    self.infeasible.append(r["config_id"])
        return [self.cache[config_id(z)] for z in zs]

    def stats(self) -> Dict[str, Any]:
        return {"requests": self.requests, "unique_designs": len(self.cache), "des_calls": self.des_calls,
                "duplicates_avoided": self.requests - self.des_calls, "des_wall_s": self.des_wall_s,
                "infeasible_designs": list(self.infeasible)}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


class CS2MDFProblem(Problem):
    """Bi-objective discrete MDF problem over the nine DVs (vectorised evaluation of a whole population)."""

    def __init__(self, evaluator: DESEvaluator):
        self.doms = domains()
        self.evaluator = evaluator
        n = [len(self.doms[d]) for d in DV_ORDER]
        super().__init__(n_var=9, n_obj=2, n_ieq_constr=1, xl=np.zeros(9), xu=np.array(n) - 1, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        zs = [decode(x, self.doms) for x in X]
        recs = self.evaluator.evaluate_many(zs)
        out["F"] = np.array([[-r["TH"], r["E"]] for r in recs], float)
        out["G"] = np.array([[sum(not r[f"{c}_ok"] for c in CONDITIONS)] for r in recs], float)
