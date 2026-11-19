"""MDA 1.4 deterministic sensitivity analysis - command line.

Run from the repository root:

    python -B cs2_mdo/run_sensitivity.py benchmark                 # Phase A: runtime benchmark + cost estimate
    python -B cs2_mdo/run_sensitivity.py ofat                      # Phase B: one-factor-at-a-time + plots
    python -B cs2_mdo/run_sensitivity.py ofat-constraints          #          OFAT of the constraints C1-C6, capacity use, waits
    python -B cs2_mdo/run_sensitivity.py factorial [--workers 10]  # Phase C: all 26,244 configurations (resumable)
    python -B cs2_mdo/run_sensitivity.py factorial --limit 500     #          ... only the next 500 (checkpointed)
    python -B cs2_mdo/run_sensitivity.py indices                   # Phases D-E: Sobol and pairwise indices
    python -B cs2_mdo/run_sensitivity.py plots                     # all figures from the stored results
    python -B cs2_mdo/run_sensitivity.py all [--run-factorial]     # benchmark, OFAT, (factorial), indices, plots

The full enumeration never starts unless 'factorial' or 'all --run-factorial' is given. An interrupted factorial
run resumes from results/sensitivity/factorial_results_partial.csv (use --no-resume to start again).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CS2 MDA 1.4 deterministic sensitivity analysis")
    p.add_argument("phase", choices=["benchmark", "ofat", "ofat-constraints", "factorial", "indices", "plots", "all"])
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPU count - 2)")
    p.add_argument("--limit", type=int, default=None, help="factorial: evaluate at most this many missing configurations")
    p.add_argument("--no-resume", action="store_true", help="factorial: discard the checkpoint and start again")
    p.add_argument("--run-factorial", action="store_true", help="all: include the full enumeration")
    p.add_argument("--n-benchmark", type=int, default=36)
    a = p.parse_args(argv)

    from cs2_mdo.sensitivity import benchmark, factorial, interactions, ofat, plots

    if a.phase in ("benchmark", "all"):
        r = benchmark.run_benchmark(a.n_benchmark, a.workers)
        t, est = r["eval_time_s"], r["full_enumeration_estimate"]
        print(f"benchmark: {r['successful']}/{r['n_configurations_benchmarked']} evaluated, mean {t['mean']:.3f} s, "
              f"median {t['median']:.3f} s, max {t['max']:.3f} s; deterministic {r['deterministic_repeat_identical']}")
        print(f"  full enumeration ({est['n_configurations']:,}): ~{est['serial_hours']:.2f} h serial, "
              f"~{est['parallel_hours'] * 60:.0f} min with {est['parallel_workers']} workers")
    if a.phase in ("ofat", "all"):
        o = ofat.run_ofat()
        print(o["ranges"][["design_variable", "R_TH", "R_TH_pct_of_baseline", "rank_TH", "R_E", "R_E_pct_of_baseline",
                           "rank_E"]].to_string(index=False))
        for f in plots.all_ofat(o):
            print("  figure", f)
    if a.phase in ("ofat-constraints", "all"):
        from cs2_mdo.sensitivity import ofat_constraints
        oc = ofat_constraints.run_ofat_constraints()
        print(f"OFAT constraints: {oc['n_simulations']} DES runs; C1-C6 checks satisfied "
              f"{int(oc['checks'].satisfied.sum())} / {len(oc['checks'])}")
        for f in plots.ofat_constraint_figures(oc["checks"], oc["use"]):
            print("  figure", f)
    if a.phase == "factorial" or (a.phase == "all" and a.run_factorial):
        r = factorial.run_factorial(a.workers, resume=not a.no_resume, limit=a.limit)
        print(json.dumps(r, indent=2))
        from cs2_mdo.sensitivity.common import OUT_DIR
        with open(os.path.join(OUT_DIR, "factorial_run.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)
    if a.phase in ("indices", "all"):
        if not os.path.exists(factorial.FINAL):
            print("indices: the full-factorial dataset is not available yet (run the 'factorial' phase).")
        else:
            g = interactions.run_global()
            print(g["indices"][["response", "design_variable", "S1", "ST", "ST_minus_S1"]].round(4).to_string(index=False))
            print(json.dumps({y: s.get("checks") for y, s in g["summary"].items() if y != "population"}, indent=1))
    if a.phase in ("plots", "all"):
        import pandas as pd
        from cs2_mdo.sensitivity.common import OUT_DIR
        o = {"ofat": pd.read_csv(os.path.join(OUT_DIR, "ofat_results.csv")),
             "ranges": pd.read_csv(os.path.join(OUT_DIR, "ofat_ranges.csv"))}
        figs = plots.all_ofat(o)
        if os.path.exists(factorial.FINAL):
            figs += plots.all_global(interactions.run_global(), o["ofat"])
        print(f"{len(figs)} figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
