"""MDF 1.0 - deterministic bi-objective optimization of CS2 (maximize TH, minimize E) - command line.

Run from the repository root:

    python -B cs2_mdo/run_mdf.py exact                                  # Phase A: exact Pareto front (factorial dataset)
    python -B cs2_mdo/run_mdf.py pilot [--gen 10] [--workers 10]        # small direct-DES NSGA-II pilot + cost estimate
    python -B cs2_mdo/run_mdf.py nsga2 --seeds 1 2 3 4 5 [--pop 100 --gen 100 --workers 10]   # multi-seed experiment
    python -B cs2_mdo/run_mdf.py compare | representatives | plots
    python -B cs2_mdo/run_mdf.py all --seeds 1 2 3 4 5                  # complete workflow

NSGA-II evaluates every new design with the DES; the factorial dataset is only the reference for the metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CS2 MDF 1.0 (NSGA-II with direct DES evaluation)")
    p.add_argument("phase", choices=["exact", "pilot", "nsga2", "compare", "representatives", "plots", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gen", type=int, default=None, help="generations (default 100; pilot 10)")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args(argv)

    from cs2_mdo.optimisation import exact_pareto, nsga2

    if a.phase in ("exact", "all"):
        r = exact_pareto.extract_exact_front()
        s = r["summary"]
        print(f"exact front: {s['n_pareto_designs']} designs, {s['n_pareto_points']} points; baseline Pareto-optimal: "
              f"{s['baseline']['pareto_optimal']} (dominated by {s['baseline']['n_dominating_designs']} designs)")
    if a.phase == "pilot":
        gen = a.gen or 10
        r = nsga2.run_nsga2(seed=a.seeds[0], pop_size=a.pop, n_gen=gen, workers=a.workers, label=f"pilot_seed_{a.seeds[0]}_gen_{gen}")
        s = r["summary"]
        rate = s["des_calls"] / s["runtime_s"]
        h = r["history"]
        print(json.dumps({k: s[k] for k in ("label", "requests", "unique_designs", "des_calls", "duplicates_avoided", "runtime_s",
                                            "cpu_time_per_des_call_s", "wall_time_per_des_call_s", "infeasible_designs",
                                            "final_archive")}, indent=1, default=float))
        print(h[["generation", "requests", "des_calls", "elapsed_s", "archive_hv_gap_pct", "archive_igd_plus",
                 "archive_designs_recovered", "pop_hv_gap_pct"]].round(4).to_string(index=False))
        print(f"DES throughput {rate:.1f} evaluations/s; upper bound for one 100 x 100 run (10,000 DES calls): "
              f"{10000 / rate / 60:.0f} min")
    if a.phase in ("nsga2", "all"):
        for seed in a.seeds:
            r = nsga2.run_nsga2(seed=seed, pop_size=a.pop, n_gen=a.gen or 100, workers=a.workers)
            s = r["summary"]
            print(f"seed {seed}: {s['des_calls']} DES calls, {s['runtime_s'] / 60:.1f} min, HV gap "
                  f"{s['final_archive']['archive_hv_gap_pct']:.3f} %, recovered {s['final_archive']['archive_designs_recovered']} / {s['final_archive'].get('n_exact_designs', 'exact front')}")
    if a.phase in ("compare", "all"):
        from cs2_mdo.optimisation import reporting
        print(reporting.compare_runs().to_string(index=False))
    if a.phase in ("representatives", "all"):
        from cs2_mdo.optimisation import representatives
        print(representatives.select_representatives().to_string(index=False))
    if a.phase in ("plots", "all"):
        from cs2_mdo.optimisation import plots
        print(f"{len(plots.all_figures())} figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
