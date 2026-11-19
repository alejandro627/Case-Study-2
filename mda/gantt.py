"""Gantt charts of the production process from the DES event log (post-processing; no effect on the model).

Two chart types for a chosen cycle of every shuttle (loop index k, all times are DES times in s):
  * flow    : process-flow lanes per shuttle - a middle lane for the sequential steps, which splits at the departure
              from WS6 into the shuttle branch (above: WS9 setup, UR3e assembly, travels) and the FANUC branch (below:
              UR10 input transfer, FANUC sequence), and merges at the UR10 transfer of the third tray onto the shuttle;
  * activity: one row per activity in process order, bars at their start and end times (one sub-lane per shuttle).
Waiting (blocked track position, clearance, busy resource, FANUC tray / shuttle synchronisation) is drawn in grey.
Every bar is also written to a CSV (so the charts can be rebuilt in Excel).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.patches import Patch, Rectangle                    # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KIND_COLOR = {"Shuttle travel": "#1c2b4a", "UR10": "#eb6834", "UR3e": "#2a78d6", "FANUC cell": "#6250d6",
              "Operator WS9 (setup)": "#1baf7a", "Operator WS8 (QC)": "#008300", "Waiting": "#c9c8c2"}
OP_KIND = {"ws6_ur10_load_two_trays": "UR10", "fanuc_input_transfer": "UR10", "ws6_sync_retrieve_fanuc_tray": "UR10",
           "ws8_ur10_unload_three_trays": "UR10", "ws4_ur3e_assembly": "UR3e", "fanuc_conveyor_sequence": "FANUC cell",
           "ws9_load_workpieces": "Operator WS9 (setup)", "ws8_quality_inspection": "Operator WS8 (QC)"}
LEG_LABEL = {"L1_WS6_to_WS9": "The shuttle travels from WS6 to WS9", "L2_WS9_to_WS4": "The shuttle travels from WS9 to WS4",
             "L3_WS4_to_WS6": "The shuttle travels from WS4 to WS6", "L4_WS6_to_WS8": "The shuttle travels from WS6 to WS8",
             "L5_WS8_to_WS6_return": "The shuttle travels from WS8 to WS6", "Linit_WS5_to_WS6": "Start: the shuttle enters WS6 from WS5"}
OP_LABEL = {"ws6_ur10_load_two_trays": "UR10 transfers two blisters from WH to the shuttle",
            "ws9_load_workpieces": "Setup of the workpieces (operator WS9)",
            "ws4_ur3e_assembly": "UR3e performs the assembly",
            "fanuc_input_transfer": "UR10 transfers one blister from WH to the conveyor",
            "fanuc_conveyor_sequence": "FANUC performs the assembly + conveyor movement",
            "ws6_sync_retrieve_fanuc_tray": "UR10 transfers the blister from the conveyor to the shuttle",
            "ws8_quality_inspection": "Quality inspection (operator WS8)",
            "ws8_ur10_unload_three_trays": "UR10 transfers the three blister trays from the shuttle to dispatch"}
ACTIVITY_ORDER = ["Start: the shuttle enters WS6 from WS5", OP_LABEL["ws6_ur10_load_two_trays"], LEG_LABEL["L1_WS6_to_WS9"],
                  OP_LABEL["ws9_load_workpieces"], LEG_LABEL["L2_WS9_to_WS4"], OP_LABEL["ws4_ur3e_assembly"],
                  LEG_LABEL["L3_WS4_to_WS6"], OP_LABEL["fanuc_input_transfer"], OP_LABEL["fanuc_conveyor_sequence"],
                  OP_LABEL["ws6_sync_retrieve_fanuc_tray"], LEG_LABEL["L4_WS6_to_WS8"], OP_LABEL["ws8_quality_inspection"],
                  OP_LABEL["ws8_ur10_unload_three_trays"], LEG_LABEL["L5_WS8_to_WS6_return"],
                  "Shuttle waits", "FANUC tray / batch waits"]
WAIT_EVENTS = {"WAIT_FOR_NODE": "blocked: next track position occupied", "STATION_RELEASE_CLEARANCE": "station-release clearance",
               "WAIT_FOR_RESOURCE": "resource busy", "WAIT_FOR_FANUC_TRAY": "shuttle waits for the FANUC tray",
               "WAIT_FOR_FANUC_CELL": "batch waits for the free FANUC cell",
               "FANUC_TRAY_WAITS_FOR_SHUTTLE": "FANUC tray waits for the shuttle",
               "FANUC_TRAY_WAITS_FOR_UR10": "FANUC tray waits for the UR10", "CONTROL_DELAY": "control delay",
               "WAKE_UP": "equipment wake-up"}


def cycle_bars(res: Dict[str, Any], loop_index: int = 1) -> pd.DataFrame:
    """All bars of loop `loop_index` of every shuttle (incl. its FANUC branch and, for loop 1, the start positioning)."""
    ev, loops, jobs = res["events"], res["loops"], res["jobs"].set_index("batch_id")
    rows: List[Dict[str, Any]] = []
    for sid, lp in loops.groupby("shuttle_id"):
        sel = lp[lp.shuttle_loop_index == loop_index]
        if sel.empty:
            continue
        b = int(sel.batch_id.iloc[0])
        t0, t1 = float(sel.cycle_start_s.iloc[0]), float(sel.cycle_end_s.iloc[0])
        j = jobs.loc[b]
        split, merge = float(j.t_shuttle_departs), float(j.t_retrieval_start)
        e = ev[(ev.batch_id == b) | ((loop_index == 1) & (ev.shuttle_id == sid) & ev.batch_id.isna() & (ev.end_s <= t0 + 1e-9))]
        for r in e.itertuples():
            if r.event in ("ARRIVE", "PASS_THROUGH", "CYCLE_START", "CYCLE_END") or r.end_s - r.start_s <= 1e-9:
                continue
            fanuc = (isinstance(r.operation, str) and r.operation.startswith("fanuc")) or r.event in (
                "WAIT_FOR_FANUC_CELL", "FANUC_TRAY_WAITS_FOR_SHUTTLE", "FANUC_TRAY_WAITS_FOR_UR10")
            if r.event == "TRAVEL":
                kind, label = "Shuttle travel", LEG_LABEL.get(r.leg, r.leg)
                detail = f"{r.from_node} -> {r.to_node}"
            elif r.event == "PROCESS":
                kind, label, detail = OP_KIND.get(r.operation, r.resource), OP_LABEL.get(r.operation, r.operation), r.resource
            elif r.event in WAIT_EVENTS:
                kind = "Waiting"
                label = "FANUC tray / batch waits" if fanuc else "Shuttle waits"
                detail = WAIT_EVENTS[r.event] + (f" ({r.resource})" if isinstance(r.resource, str) and r.resource else "")
            else:
                continue
            if fanuc:
                lane = "FANUC branch"
            elif r.start_s >= split - 1e-9 and r.end_s <= merge + 1e-9:
                lane = "shuttle branch"
            else:
                lane = "sequential"
            rows.append({"shuttle_id": int(sid), "loop_index": loop_index, "batch_id": b, "lane": lane, "activity": label,
                         "kind": kind, "detail": detail, "start_s": float(r.start_s), "end_s": float(r.end_s),
                         "duration_s": float(r.end_s - r.start_s), "cycle_start_s": t0, "cycle_end_s": t1,
                         "split_s": split, "merge_s": merge})
    df = pd.DataFrame(rows).sort_values(["shuttle_id", "start_s"]).reset_index(drop=True)
    merged: List[Dict[str, Any]] = []                     # contiguous hops of one leg -> one travel bar
    last_travel: Dict[int, Dict[str, Any]] = {}            # per shuttle (parallel FANUC rows interleave in time)
    for r in df.to_dict("records"):
        p = last_travel.get(r["shuttle_id"]) if r["kind"] == "Shuttle travel" else None
        if r["kind"] == "Shuttle travel" and (p is None or p["activity"] != r["activity"] or abs(p["end_s"] - r["start_s"]) > 1e-9):
            p = None
        if p is not None:
            p["end_s"], p["duration_s"] = r["end_s"], r["end_s"] - p["start_s"]
            p["detail"] = p["detail"].rsplit(" -> ", 1)[0] + " -> " + r["detail"].split(" -> ")[-1]
        else:
            merged.append(dict(r))
            if r["kind"] == "Shuttle travel":
                last_travel[r["shuttle_id"]] = merged[-1]
    out = pd.DataFrame(merged)
    out["short_label"] = [_short(r) for r in out.itertuples()]
    return out


SHORT_OP = {OP_LABEL["ws6_ur10_load_two_trays"]: "UR10 load", OP_LABEL["ws9_load_workpieces"]: "Setup",
            OP_LABEL["ws4_ur3e_assembly"]: "UR3e assembly", OP_LABEL["fanuc_input_transfer"]: "UR10 input",
            OP_LABEL["fanuc_conveyor_sequence"]: "FANUC + conveyor", OP_LABEL["ws6_sync_retrieve_fanuc_tray"]: "UR10 retrieve",
            OP_LABEL["ws8_quality_inspection"]: "QC", OP_LABEL["ws8_ur10_unload_three_trays"]: "UR10 unload"}


def _short(r) -> str:
    if r.kind == "Shuttle travel":
        a = r.activity.replace("Start: the shuttle enters WS6 from WS5", "from WS5 to WS6")
        p = a.split("from ")[-1].split(" to ")
        return f"{p[0]}-{p[1]}" if len(p) == 2 else a
    if r.kind == "Waiting":
        return "wait"
    return SHORT_OP.get(r.activity, r.activity)


def _text(ax, r, y, fontsize=6.5):
    """Task name inside the bar when it fits, else nothing (durations are in the CSV)."""
    if r.end_s - r.start_s < 14:
        return
    ax.text((r.start_s + r.end_s) / 2, y, r.short_label, ha="center", va="center", fontsize=fontsize, zorder=4,
            color=INK if r.kind == "Waiting" else "white", linespacing=0.95)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _bar(ax, x0, x1, y, h, kind):
    c = KIND_COLOR[kind]
    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=c, edgecolor="#22272e", linewidth=0.7,
                           hatch="////" if kind == "Waiting" else None, zorder=3))


def flow_chart(bars: pd.DataFrame, title: str, path: str) -> None:
    sh = sorted(bars.shuttle_id.unique())
    fig, ax = plt.subplots(figsize=(13, 1.0 + 1.55 * len(sh)), facecolor=SURFACE)
    yt, yl = [], []
    for k, sid in enumerate(sh):
        d = bars[bars.shuttle_id == sid]
        base = (len(sh) - 1 - k) * 3.2
        y = {"shuttle branch": base + 1.0, "sequential": base, "FANUC branch": base - 1.0}
        for r in d.itertuples():
            _bar(ax, r.start_s, r.end_s, y[r.lane], 0.62, r.kind)
            _text(ax, r, y[r.lane], 6 if len(sh) > 1 else 6.5)
        split, merge = d.split_s.iloc[0], d.merge_s.iloc[0]
        for lane in ("shuttle branch", "FANUC branch"):                    # split and merge connectors
            dl = d[d.lane == lane]
            if len(dl):
                ax.plot([split, split, dl.start_s.min()], [base, y[lane], y[lane]], color="#9aa4b2", lw=1.0, zorder=2)
                ax.plot([dl.end_s.max(), merge, merge], [y[lane], y[lane], base], color="#9aa4b2", lw=1.0, zorder=2)
        yt += [y["shuttle branch"], y["sequential"], y["FANUC branch"]]
        yl += [f"Shuttle {sid}: shuttle branch", f"Shuttle {sid}: sequential", f"Shuttle {sid}: FANUC branch"]
    ax.set_yticks(yt, yl, fontsize=8)
    ax.set_xlim(bars.start_s.min() - 5, bars.end_s.max() + 5)
    ax.set_ylim(-1.8, (len(sh) - 1) * 3.2 + 1.8)
    ax.set_xlabel("time (s)")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    _style(ax)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#22272e", hatch="////" if k == "Waiting" else None, label=k)
                       for k, c in KIND_COLOR.items()], frameon=False, ncol=7, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.14 if len(sh) > 1 else -0.36))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def activity_chart(bars: pd.DataFrame, title: str, path: str) -> None:
    rows = [a for a in ACTIVITY_ORDER if a in set(bars.activity)]
    sh = sorted(bars.shuttle_id.unique())
    n = len(sh)
    fig, ax = plt.subplots(figsize=(13, 0.9 + 0.48 * len(rows) * max(1, 0.75 * n)), facecolor=SURFACE)
    h = 0.8 / n
    for i, a in enumerate(rows):
        for k, sid in enumerate(sh):
            y = i - 0.4 + h * (k + 0.5)
            for r in bars[(bars.activity == a) & (bars.shuttle_id == sid)].itertuples():
                _bar(ax, r.start_s, r.end_s, y, h * 0.9, r.kind)
                w = r.end_s - r.start_s
                if n > 1 and w > 6:
                    ax.text((r.start_s + r.end_s) / 2, y, f"S{sid}" + (f" {r.short_label}" if w > 25 else ""), ha="center",
                            va="center", fontsize=6, color="white" if r.kind != "Waiting" else INK, zorder=4)
                elif n == 1 and w > 12:
                    lab = r.short_label
                    ax.text((r.start_s + r.end_s) / 2, y, lab, ha="center", va="center", fontsize=7,
                            color="white" if r.kind != "Waiting" else INK, zorder=4)
        ax.axhline(i + 0.5, color=GRID, lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_ylim(len(rows) - 0.4, -0.6)                     # first activity at the top
    ax.set_xlim(bars.start_s.min() - 5, bars.end_s.max() + 5)
    ax.set_xlabel("time (s)")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    _style(ax)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#22272e", hatch="////" if k == "Waiting" else None, label=k)
                       for k, c in KIND_COLOR.items()], frameon=False, ncol=7, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.07 if len(rows) > 10 else -0.12))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_gantts(res: Dict[str, Any], n_shuttles: int, out_dir: str, loop_index: int = 1) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    bars = cycle_bars(res, loop_index)
    tag = f"{n_shuttles}sh_cycle{loop_index}"
    what = "first cycle (incl. start-up)" if loop_index == 1 else f"cycle {loop_index} of every shuttle (steady state)"
    base = f"{n_shuttles} shuttle{'s' if n_shuttles > 1 else ''} – {what}"
    bars.to_csv(os.path.join(out_dir, f"gantt_bars_{tag}.csv"), index=False)
    p1 = os.path.join(out_dir, f"gantt_flow_{tag}")
    p2 = os.path.join(out_dir, f"gantt_activity_{tag}")
    flow_chart(bars, f"Process flow, {base}", p1)
    activity_chart(bars, f"Production activities, {base}", p2)
    return [p1, p2, os.path.join(out_dir, f"gantt_bars_{tag}.csv")]
