"""Front-quality metrics against the exact Pareto reference.

Normalisation (fixed for every run): bounds of the exact Pareto front,
    f1 = (TH_ideal - TH) / (TH_ideal - TH_nadir),   f2 = (E - E_ideal) / (E_nadir - E_ideal)
with TH_ideal = max TH and E_ideal = min E on the front, TH_nadir = TH of the min-E point and E_nadir = E of the
max-TH point. Both normalised objectives are minimised; the exact front spans [0, 1] x [0, 1].
Hypervolume: exact 2-D computation with the reference point r = (1.1, 1.1) in normalised space; points not strictly
better than r in both objectives contribute nothing. IGD+ (Ishibuchi et al. 2015) in the same normalised space.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

REF_POINT = (1.1, 1.1)
# Matching tolerance for energy when a DES value is compared with the stored reference. The factorial CSV stores 16
# significant digits, so a stored energy can differ from the in-memory DES value by 1 ULP (~1e-15 kWh). Throughput is
# a multiple of 1.125 and is compared exactly. Dominance itself is always decided with exact comparisons.
E_MATCH_TOL_KWH = 1e-9


def bounds_from_front(th: np.ndarray, e: np.ndarray) -> Dict[str, float]:
    th, e = np.asarray(th, float), np.asarray(e, float)
    i_th, i_e = int(np.argmax(th)), int(np.argmin(e))
    b = {"TH_ideal": float(th.max()), "TH_nadir": float(th[i_e]), "E_ideal": float(e.min()), "E_nadir": float(e[i_th])}
    if b["TH_ideal"] == b["TH_nadir"] or b["E_ideal"] == b["E_nadir"]:
        raise ValueError("zero objective range on the reference front: normalisation undefined")
    return b


def normalise(th, e, b: Dict[str, float]) -> np.ndarray:
    th, e = np.asarray(th, float), np.asarray(e, float)
    return np.column_stack([(b["TH_ideal"] - th) / (b["TH_ideal"] - b["TH_nadir"]),
                            (e - b["E_ideal"]) / (b["E_nadir"] - b["E_ideal"])])


def hypervolume_2d(F: np.ndarray, ref: Tuple[float, float] = REF_POINT) -> float:
    """Exact dominated area (minimisation) between the non-dominated points of F and the reference point."""
    F = np.asarray(F, float)
    F = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if len(F) == 0:
        return 0.0
    F = F[np.lexsort((F[:, 1], F[:, 0]))]          # f1 ascending, then f2 ascending
    pts, best = [], ref[1]                          # staircase of non-dominated points
    for f1, f2 in F:
        if f2 < best:
            pts.append((f1, f2))
            best = f2
    hv = 0.0
    for k, (f1, f2) in enumerate(pts):
        x_next = pts[k + 1][0] if k + 1 < len(pts) else ref[0]
        hv += (x_next - f1) * (ref[1] - f2)
    return float(hv)


def igd_plus(A: np.ndarray, Z: np.ndarray) -> float:
    """IGD+ of the approximation A w.r.t. the reference front Z (both normalised, minimisation)."""
    A, Z = np.asarray(A, float), np.asarray(Z, float)
    if len(A) == 0:
        return float("inf")
    d = np.sqrt((np.maximum(A[None, :, :] - Z[:, None, :], 0.0) ** 2).sum(axis=2))   # |Z| x |A|
    return float(d.min(axis=1).mean())


def recovery(found_ids: Iterable[str], found_pts: Iterable[Tuple[float, float]], exact_ids: Iterable[str],
             exact_pts: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    found_ids, exact_ids = set(found_ids), set(exact_ids)
    found_pts, exact_pts = list(map(tuple, found_pts)), list(dict.fromkeys(map(tuple, exact_pts)))
    nd = len(found_ids & exact_ids)
    npnt = sum(any(th == ft and abs(e - fe) <= E_MATCH_TOL_KWH for ft, fe in found_pts) for th, e in exact_pts)
    return {"designs_recovered": nd, "designs_recovered_pct": 100.0 * nd / len(exact_ids),
            "points_recovered": npnt, "points_recovered_pct": 100.0 * npnt / len(exact_pts),
            "false_pareto_designs": len(found_ids - exact_ids)}
