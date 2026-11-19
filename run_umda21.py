"""UMDA 2.1 (UDES 2.1: time_lag catch-up, R_phi(t_f) at restoration + 30 min) + reference-configuration study.

    python -B cs2_mdo/run_umda21.py plan         # run counts
    python -B cs2_mdo/run_umda21.py run          # stages A, C, S (resumable)
    python -B cs2_mdo/run_umda21.py reference    # reference design: DS1-DS4, DS15 x N = 500; DS15 convergence 200/500/1500 x 3
    python -B cs2_mdo/run_umda21.py report
Outputs: cs2_mdo/results/umda_2_1/. The UDES 2.0 settings (count rule, R_phi at restoration + 60 min) are selected with
UDES2_CONFIG=udes_2_0_config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "results", "umda_2_1")
RUNS = os.path.join(OUT, "UMDA2_1_runs.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["plan", "run", "reference", "report"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    from cs2_mdo.udes_2_0 import coefficients as C2
    if C2.load_config()["meta"]["version"] != "UDES 2.1":
        raise SystemExit(f"UMDA 2.1 needs the UDES 2.1 configuration, found {C2.CONFIG_PATH}")
    from cs2_mdo.umda_2_1 import plan as P
    from cs2_mdo.umda_2_1 import reference as RF
    J = P.jobs()
    if a.phase == "plan":
        print(J.groupby(["stage", "strategies"]).size().rename("runs").to_string())
        print(f"main {len(J):,} runs; reference {len(RF.jobs()):,} runs")
        return 0
    if a.phase == "run":
        t0 = time.perf_counter()
        info = P.execute(J, RUNS, workers=a.workers)
        info.update(total_planned=len(J), start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - (time.perf_counter() - t0))),
                    end=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(info, open(os.path.join(OUT, "UMDA2_1_runtime.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(info, indent=1))
        return 0
    if a.phase == "reference":
        t0 = time.perf_counter()
        info = RF.execute(OUT, workers=a.workers)
        info.update(start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - (time.perf_counter() - t0))),
                    end=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(info, open(os.path.join(OUT, "UMDA2_1_ref_runtime.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(info, indent=1))
        return 0
    from cs2_mdo.umda_2_1.report import build
    print(build(RUNS, OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
