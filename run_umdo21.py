"""UMDO 2.1 (UDES 2.1): NSGA-II search at N = 50, verification at N = 100, high-fidelity representatives at N = 500.

    python -B cs2_mdo/run_umdo21.py search  [--pop 60 --gen 40 --seed 1]   # resumable (checkpoint)
    python -B cs2_mdo/run_umdo21.py verify                                  # archive front at N = 100 (independent samples)
    python -B cs2_mdo/run_umdo21.py hifi                                    # representatives at N = 500 + trajectories
    python -B cs2_mdo/run_umdo21.py report
Outputs: cs2_mdo/results/umdo_2_1/.
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
OUT = os.path.join(HERE, "results", "umdo_2_1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["search", "verify", "hifi", "report"])
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import pandas as pd
    from cs2_mdo.udes_2_0 import coefficients as C2
    if C2.load_config()["meta"]["version"] != "UDES 2.1":
        raise SystemExit("UMDO 2.1 needs the UDES 2.1 configuration")
    from cs2_mdo.umdo_2_1 import post as PO
    from cs2_mdo.umdo_2_1.evaluator import Evaluator21
    from cs2_mdo.umdo_2_1.optimizer import run
    log_path = os.path.join(a.out, f"UMDO2_1_{a.phase}.log")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        open(log_path, "a", encoding="utf-8").write(line + "\n")

    if a.phase == "search":
        store = os.path.join(a.out, "UMDO2_1_search_runs.csv")
        t_start = time.strftime("%Y-%m-%d %H:%M:%S")
        with Evaluator21(store, "A", 50, a.workers) as ev:
            if os.path.exists(store):                                    # resume: designs already simulated
                n = ev.seed_from_rows(pd.read_csv(store, low_memory=False), count_as_seed=False)
                log(f"cache restored from the run store: {n} designs")
            umda = os.path.join(HERE, "results", "umda_2_1", "UMDA2_1_runs.csv")
            if os.path.exists(umda):                                     # UMDA 2.1 stage A used the same sample set
                R = pd.read_csv(umda, low_memory=False)
                R["strategies"] = R["strategies"].astype(str).str.replace(".0", "", regex=False).str.zfill(3)
                A = R[(R.stage == "A") & (R.strategies == "111")]
                log(f"seeded {ev.seed_from_rows(A)} designs ({ev.seed_runs_reused} runs) from UMDA 2.1 stage A")
            res = run(ev, a.pop, a.gen, a.seed, os.path.join(a.out, "UMDO2_1_checkpoint.pkl"), log)
            res["monitor"].to_csv(os.path.join(a.out, "UMDO2_1_search_monitor.csv"), index=False)
            arch = pd.DataFrame([{k: v for k, v in d.items() if k not in ("F", "G")} | {f"G{i}": g for i, g in enumerate(d["G"])}
                                 for d in ev.designs.values()])
            arch.to_csv(os.path.join(a.out, "UMDO2_1_search_archive.csv"), index=False, float_format="%.10g")
            m = res["monitor"]
            info = {"start": t_start, "end": time.strftime("%Y-%m-%d %H:%M:%S"), "pop": a.pop, "max_gen": a.gen, "seed": a.seed,
                    "generations": int(len(m)), "stopped_early": res["stopped_early"], "wall_s": float(m.elapsed_s.iloc[-1]),
                    "requests": ev.requests, "unique_designs": len(ev.designs), "des_runs_executed": ev.des_runs_executed,
                    "seed_designs_reused": ev.seed_designs_reused, "seed_runs_reused": ev.seed_runs_reused, "workers": a.workers}
            json.dump(info, open(os.path.join(a.out, "UMDO2_1_search_runtime.json"), "w"), indent=1)
            log(json.dumps(info))
        return 0
    if a.phase == "verify":
        log(json.dumps(PO.verify(a.out, a.workers)))
        return 0
    if a.phase == "hifi":
        log(json.dumps(PO.hifi(a.out, a.workers)))
        return 0
    from cs2_mdo.umdo_2_1.report import build
    print(build(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
