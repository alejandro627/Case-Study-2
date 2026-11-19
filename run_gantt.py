"""Gantt charts of the production process (DES v10.0) for 1, 2 and 3 shuttles.

    python -B cs2_mdo/run_gantt.py                    # cycle 1 (with start-up) and cycle 10 (steady state), 1-3 shuttles
    python -B cs2_mdo/run_gantt.py --cycles 1 5       # other cycles
    python -B cs2_mdo/run_gantt.py --dv DV5=1         # any other design vector (default: baseline)

Output: cs2_mdo/results/gantt/  gantt_flow_<n>sh_cycle<k>.png/pdf, gantt_activity_<n>sh_cycle<k>.png/pdf and
gantt_bars_<n>sh_cycle<k>.csv (every bar: shuttle, lane, activity, start, end, duration - for Excel).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs2_mdo.des.config import RESULTS_DIR                    # noqa: E402
from cs2_mdo.mda.coordinator import evaluate_design            # noqa: E402
from cs2_mdo.mda.design_variables import load_design_variables  # noqa: E402
from cs2_mdo.mda.gantt import make_gantts                      # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Gantt charts of the CS2 production process from the DES")
    p.add_argument("--shuttles", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cycles", type=int, nargs="+", default=[1, 10])
    p.add_argument("--dv", action="append", default=[], metavar="DVk=LEVEL")
    p.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "gantt"))
    a = p.parse_args(argv)
    z0 = dict(load_design_variables()["baseline"])
    for item in a.dv:
        k, _, v = item.partition("=")
        z0[k.strip()] = int(v)
    for n in a.shuttles:
        out = evaluate_design(dict(z0, DV1=n))
        if out["status"] != "evaluated":
            print(f"{n} shuttles: {out['status']} {out.get('messages')}")
            continue
        for k in a.cycles:
            for f in make_gantts(out["results"], n, a.out_dir, k):
                print("written", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
