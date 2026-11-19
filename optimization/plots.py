"""MDF 1.0 figures in two sizes: full page width (figures/full_width) and single column (figures/single_column).

Every figure is exported as PNG (300 dpi) and PDF (vector). Colour roles: exact front = blue, NSGA-II = orange,
background designs = neutral grey; representatives use distinct marker shapes plus direct labels.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

from ..sensitivity.common import DV_ORDER                          # noqa: E402
from ..sensitivity.factorial import FINAL as FACTORIAL_CSV         # noqa: E402
from .common import FIG_DIR, OUT_DIR, ensure_dirs                  # noqa: E402
from .reporting import load_runs                                   # noqa: E402

SURFACE, INK, INK2, GRID, BG = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MARK = {"R0": ("s", INK), "R1": ("v", AQUA), "R2": ("^", ORANGE), "R3": ("D", "#6250d6")}
SIZES = {"full_width": {"w": 6.3, "h": 4.0, "fs": 9}, "single_column": {"w": 3.4, "h": 2.5, "fs": 7.5}}


def _style(fs: float) -> None:
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": fs, "axes.titlesize": fs + 1, "legend.fontsize": fs - 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def _grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, name: str, variant: str) -> List[str]:
    d = os.path.join(FIG_DIR, variant)
    os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(d, f"{name}.{ext}")
        fig.savefig(p, dpi=300)
        out.append(p)
    plt.close(fig)
    return out


def _data():
    all_ = pd.read_csv(FACTORIAL_CSV, float_precision="round_trip", usecols=["config_id", "TH", "E", "DV1"])
    front = pd.read_csv(os.path.join(OUT_DIR, "exact_pareto_designs.csv"), float_precision="round_trip")
    reps = pd.read_csv(os.path.join(OUT_DIR, "representative_designs.csv"), float_precision="round_trip")
    return all_, front.sort_values("TH"), reps


def fig_exact_front(variant: str) -> List[str]:
    sz = SIZES[variant]
    _style(sz["fs"])
    all_, front, reps = _data()
    fig, ax = plt.subplots(figsize=(sz["w"], sz["h"]))
    ax.scatter(all_.TH, all_.E, s=2 if variant == "full_width" else 1, color=BG, alpha=0.35, linewidths=0,
               label=f"all {len(all_):,} designs", rasterized=True)
    ax.plot(front.TH, front.E, color=BLUE, linewidth=1.0, zorder=3)
    ax.scatter(front.TH, front.E, s=14 if variant == "full_width" else 8, color=BLUE, zorder=4,
               label=f"exact Pareto front ({len(front)})")
    for r in reps.itertuples():
        m, c = MARK[r.representative]
        ax.scatter(r.TH, r.E, s=60 if variant == "full_width" else 34, marker=m, facecolor="none", edgecolor=c, linewidth=1.4, zorder=5)
        dx, dy = {"R0": (-6, 8), "R1": (6, -2), "R2": (-16, 6), "R3": (6, -10)}[r.representative]
        ax.annotate(r.representative, (r.TH, r.E), xytext=(dx, dy), textcoords="offset points", fontsize=sz["fs"], color=c)
    ax.set_xlabel("throughput TH (products/h)")
    ax.set_ylabel("total energy $E_{tot}$ (kWh / 8 h)")
    ax.legend(frameon=False, loc="upper left", markerscale=2 if variant == "full_width" else 3)
    _grid(ax)
    return _save(fig, "fig1_exact_pareto_front", variant)


def _best_run(runs):
    return min(runs, key=lambda r: (r["summary"]["final_archive"]["archive_hv_gap_pct"], r["summary"]["des_calls"]))


def fig_nsga_vs_exact(variant: str) -> List[str]:
    sz = SIZES[variant]
    _style(sz["fs"])
    _, front, _ = _data()
    runs = load_runs()
    best = _best_run(runs)
    arch = pd.read_csv(os.path.join(best["dir"], "archive_front.csv"), float_precision="round_trip").sort_values("TH")
    pop = pd.read_csv(os.path.join(best["dir"], "final_population.csv"), float_precision="round_trip")
    fig, ax = plt.subplots(figsize=(sz["w"], sz["h"]))
    ax.scatter(pop.TH, pop.E, s=10 if variant == "full_width" else 6, marker="+", color=INK2, linewidths=0.7,
               label="NSGA-II final population", zorder=2)
    ax.plot(front.TH, front.E, color=BLUE, linewidth=1.0, zorder=3)
    ax.scatter(front.TH, front.E, s=14 if variant == "full_width" else 8, color=BLUE, label="exact Pareto front", zorder=3)
    ax.scatter(arch.TH, arch.E, s=48 if variant == "full_width" else 26, facecolor="none", edgecolor=ORANGE, linewidth=1.1,
               label="NSGA-II non-dominated archive", zorder=4)
    s = best["summary"]
    ax.set_title(f"seed {s['seed']}: {s['des_calls']:,} DES evaluations ({100 * s['des_calls'] / 26244:.1f} % of the space)",
                 loc="left", fontsize=sz["fs"])
    ax.set_xlabel("throughput TH (products/h)")
    ax.set_ylabel("total energy $E_{tot}$ (kWh / 8 h)")
    ax.legend(frameon=False, loc="upper left")
    _grid(ax)
    return _save(fig, "fig2_nsga2_vs_exact", variant)


def _on_grid(runs, col):
    xmax = max(r["history"].des_calls.max() for r in runs)
    grid = np.linspace(min(r["history"].des_calls.min() for r in runs), xmax, 400)
    Y = []
    for r in runs:
        h = r["history"]
        x, y = h.des_calls.values, h[col].values
        idx = np.searchsorted(x, grid, side="right") - 1                 # value after the last completed generation
        Y.append(np.where(idx >= 0, y[np.clip(idx, 0, None)], np.nan))
    return grid, np.array(Y)


def fig_convergence(variant: str) -> List[str]:
    sz = SIZES[variant]
    _style(sz["fs"])
    runs = load_runs()
    panels = [("archive_hv_gap_pct", "HV gap (%)\n[0 drawn at $10^{-3}$]", True), ("archive_igd_plus", "IGD+ (normalised)", True),
              ("archive_designs_recovered_pct", "exact Pareto designs\nrecovered (%)", False)]
    if variant == "single_column":
        panels = [panels[0], panels[2]]
    fig, axes = plt.subplots(len(panels), 1, figsize=(sz["w"], sz["h"] * (1.45 if variant == "full_width" else 1.35)), sharex=True)
    for ax, (col, lab, log) in zip(axes, panels):
        g, Y = _on_grid(runs, col)
        floor = 1e-3 if col == "archive_hv_gap_pct" else 1e-4
        Yp = np.maximum(Y, floor) if log else Y
        ax.fill_between(g, np.nanmin(Yp, 0), np.nanmax(Yp, 0), color="#cde2fb", linewidth=0, label="min–max over 5 seeds")
        ax.plot(g, np.nanmean(Yp, 0), color=BLUE, linewidth=1.4, label="mean")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(lab)
        _grid(ax)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("cumulative unique DES evaluations")
    return _save(fig, "fig3_convergence", variant)


def fig_representatives(variant: str) -> List[str]:
    sz = SIZES[variant]
    _style(sz["fs"])
    _, _, reps = _data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(sz["w"], sz["h"] * 0.95), gridspec_kw={"width_ratios": [1, 1.35]})
    r = reps[reps.representative != "R0"]
    x = np.arange(len(r))
    a.bar(x - 0.19, r.dTH_pct_vs_R0, width=0.36, color=BLUE, label="ΔTH")
    a.bar(x + 0.19, r.dE_pct_vs_R0, width=0.36, color=ORANGE, label="ΔE")
    for xi, v1, v2 in zip(x, r.dTH_pct_vs_R0, r.dE_pct_vs_R0):
        a.text(xi - 0.19, v1, f"{v1:+.0f}", ha="center", va="bottom" if v1 >= 0 else "top", fontsize=sz["fs"] - 1)
        a.text(xi + 0.19, v2, f"{v2:+.0f}", ha="center", va="bottom" if v2 >= 0 else "top", fontsize=sz["fs"] - 1)
    a.axhline(0, color=INK2, linewidth=0.7)
    a.set_xticks(x, r.representative)
    a.set_ylabel("change vs baseline R0 (%)")
    a.legend(frameon=False, loc="upper left")
    _grid(a, "y")
    M = reps[list(DV_ORDER)].values.astype(float)
    cmap = ListedColormap(["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    b.imshow(M, cmap=cmap, vmin=0.5, vmax=4.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            b.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=sz["fs"], color="white" if M[i, j] >= 3 else INK)
    b.set_xticks(range(9), [d.replace("DV", "") for d in DV_ORDER])
    b.set_yticks(range(len(reps)), reps.representative)
    b.set_xlabel("design variable (level)")
    b.spines[:].set_visible(False)
    b.tick_params(length=0)
    return _save(fig, "fig4_representatives", variant)


def fig_design_map() -> List[str]:
    sz = SIZES["full_width"]
    _style(sz["fs"])
    _, front, _ = _data()
    M = front[list(DV_ORDER)].values.T.astype(float)
    fig, ax = plt.subplots(figsize=(sz["w"] * 1.25, 3.2))
    cmap = ListedColormap(["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    ax.imshow(M, cmap=cmap, vmin=0.5, vmax=4.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] >= 3 else INK)
    ax.set_yticks(range(9), DV_ORDER)
    ax.set_xticks(range(len(front)), [f"{t:g}" for t in front.TH], rotation=90, fontsize=6.5)
    ax.set_xlabel("Pareto designs ordered by throughput (products/h)")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("DV levels along the exact Pareto front (1 = lowest level, 4 = DV9 only)", loc="left")
    return _save(fig, "fig5_pareto_design_map", "full_width")


def fig_per_seed() -> List[str]:
    sz = SIZES["full_width"]
    _style(sz["fs"])
    runs = load_runs()
    fig, ax = plt.subplots(figsize=(sz["w"], sz["h"]))
    for r, c in zip(runs, (BLUE, ORANGE, AQUA, "#6250d6", "#e34948")):
        h = r["history"]
        ax.plot(h.des_calls, np.maximum(h.archive_hv_gap_pct, 1e-3), color=c, linewidth=1.1, label=f"seed {r['summary']['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative unique DES evaluations")
    ax.set_ylabel("HV gap of the archive (%)  [0 drawn at 1e-3]")
    ax.legend(frameon=False)
    _grid(ax)
    return _save(fig, "fig6_hv_gap_per_seed", "full_width")


def fig_specific_energy() -> List[str]:
    sz = SIZES["full_width"]
    _style(sz["fs"])
    _, front, _ = _data()
    fig, ax = plt.subplots(figsize=(sz["w"], sz["h"] * 0.85))
    for lvl, c in zip((1, 2, 3), (BLUE, ORANGE, AQUA)):
        d = front[front.DV1 == lvl]
        ax.scatter(d.TH, d.specific_energy_Wh_per_product, s=22, color=c, label=f"{lvl} shuttle{'s' if lvl > 1 else ''}")
    ax.set_xlabel("throughput TH (products/h)")
    ax.set_ylabel("specific energy (Wh/product)")
    ax.legend(frameon=False)
    ax.set_title("Exact Pareto designs: energy per product", loc="left")
    _grid(ax)
    return _save(fig, "fig7_pareto_specific_energy", "full_width")


def all_figures() -> List[str]:
    ensure_dirs()
    out = []
    for v in ("full_width", "single_column"):
        out += fig_exact_front(v) + fig_nsga_vs_exact(v) + fig_convergence(v) + fig_representatives(v)
    out += fig_design_map() + fig_per_seed() + fig_specific_energy()
    return out
