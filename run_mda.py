"""Run the six-discipline MDA for one design vector.

    python -B cs2_mdo/run_mda.py                                  # baseline design vector, 8 h
    python -B cs2_mdo/run_mda.py --dv DV1=3 --hours 4
    python -B cs2_mdo/run_mda.py --dv DV1=2 --travel "S8->S4=17.10" --set process_times_s.ws4_ur3e_assembly_s=175

Unsupported alternatives (e.g. --dv DV9=2) return status 'unsupported_configuration' without simulating.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs2_mdo.des.config import load_config                                     # noqa: E402
from cs2_mdo.mda.coordinator import evaluate_design, write_outputs, discipline_summary  # noqa: E402
from cs2_mdo.mda.coupling_values import coupling_values                         # noqa: E402
from cs2_mdo.mda.design_variables import load_design_variables                 # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CS2 six-discipline MDA (integrated SimPy DES)")
    p.add_argument("--dv", action="append", default=[], metavar="DVk=LEVEL", help="design-variable level (default: baseline)")
    p.add_argument("--config", default=None, help="DES configuration JSON (default: cs2_mdo/config/baseline.json)")
    p.add_argument("--hours", type=float, help="simulation horizon in hours")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="config override dotted.path=value")
    p.add_argument("--travel", action="append", default=[], metavar="SEG=SECONDS", help='travel segment time, e.g. "S8->S4=16.96"')
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)

    z = dict(load_design_variables()["baseline"])
    for item in a.dv:
        k, _, v = item.partition("=")
        z[k.strip()] = int(v)
    ov = list(a.overrides)
    if a.hours is not None:
        ov.append(f"simulation.horizon_h={a.hours}")
    cfg = load_config(a.config) if a.config else load_config()
    for item in a.travel:
        seg, sep, val = item.partition("=")
        if not sep or seg.strip() not in cfg["travel_segments_s"]:
            print(f"--travel {item!r}: expected SEG=SECONDS with SEG in {sorted(cfg['travel_segments_s'])}", file=sys.stderr)
            return 2
        ov.append(f"travel_segments_s.{seg.strip()}.value={val}")
    try:
        out = evaluate_design(z, cfg, overrides=ov)
    except (ValueError, KeyError, NotImplementedError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"design vector {z}\nstatus: {out['status']}")
    if out["status"] != "evaluated":
        for m in out.get("messages", []):
            print(f"  {m}")
        return 1
    for key, v in out["system_responses"].items():
        print(f"  {key:34s} {v:.4f}" if isinstance(v, float) else f"  {key:34s} {v}")
    print(f"  feasible (C1-C6)                   {out['feasible']}")
    print(discipline_summary(out).to_string(index=False))
    md = out["metadata"]
    print(f"\nMDA iterations: {md['coupling_iterations']}  (couplings resolved by: {md['coupling_resolution']})")
    summary, per_batch = coupling_values(out)
    print("\nCoupling variables (CG1-CG6):")
    for g, grp in summary.groupby("group", sort=False):
        for r in grp.itertuples():
            v = f"{r.value:,.3f}" if isinstance(r.value, float) else str(r.value)
            print(f"  {r.group}  {r.coupling_variable:58s} {v:>14s}  {r.unit:12s} {r.producer_to_consumer}")
    paths = write_outputs(out, a.out_dir, a.tag)
    pre = os.path.commonprefix([os.path.basename(p) for p in paths.values()]).rstrip("_")
    for name, df in (("coupling_values", summary), ("coupling_values_per_batch", per_batch)):
        df.to_csv(os.path.join(a.out_dir, f"{pre}_{name}.csv"), index=False)
    print(f"\n{len(paths) + 2} files written to {a.out_dir} (coupling values in {pre}_coupling_values*.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
