"""Operational feasibility conditions C1-C6 (verification of a finished run).

  C1-C4  operational inequality constraints
  C5     physical conservation equality (model-verification condition, no optimizer penalty)
  C6     temporal precedence condition
Design-variable bounds belong to the design domain (mda/design_variables.py), not to C1-C6.

For each condition the table states how it is enforced inside the DES (SimPy resources, assertions) and
which additional post-run verification is performed here. The checks read plain data (ConstraintInputs),
so the tests can feed deliberately violated data and confirm that each check detects it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

EPS = 1e-9


@dataclass
class ConstraintInputs:
    horizon_s: float
    events: pd.DataFrame
    node_intervals: Dict[str, List[List[Any]]]
    node_capacity: Dict[str, int]
    res_intervals: Dict[str, List[Tuple]]
    res_capacity: Dict[str, int]
    cell_intervals: List[Tuple[float, float]]
    cell_capacity: int
    material_monitor: Dict[str, int]
    end_trays_per_shuttle: List[int]
    material_cfg: Dict[str, Any]
    counters: Dict[str, int]
    trays: List[Any]
    cycle_steps: List[Dict[str, Any]]
    rejected_products: int = 0

    @classmethod
    def from_mda(cls, mda) -> "ConstraintInputs":
        core, d1, d3 = mda.core, mda.disciplines["D1"], mda.disciplines["D3"]
        H = core.horizon_s
        return cls(
            horizon_s=H,
            events=pd.DataFrame(core.events),
            node_intervals=d1.node_intervals, node_capacity=d1.node_capacity,
            res_intervals=core.res_intervals,
            res_capacity={**{r: res.capacity for r, res in core.resources.items()},
                          **{eq: 1 for eq in core.equipment}},          # station equipment serves one task at a time
            cell_intervals=[(j.t_position_acquired, j.t_cell_released if j.t_cell_released is not None else H)
                            for j in core.jobs.values() if j.t_position_acquired is not None],
            cell_capacity=d3.fanuc_positions.capacity,
            material_monitor=dict(core.material_monitor),
            end_trays_per_shuttle=[len(s.trays) for s in core.shuttles],
            material_cfg=core.cfg["material"], counters=dict(core.c), trays=list(core.trays.values()),
            cycle_steps=core.cfg["cycle_steps"],
            rejected_products=0,   # defect_disposition = correct_and_continue: no product is rejected or removed
        )


def peak_concurrency(intervals) -> int:
    """Maximum number of simultaneously open intervals; an interval ending at t frees its slot before one starting at t."""
    pts = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in pts:
        cur += d
        peak = max(peak, cur)
    return peak


def _row(cid, name, ctype, enforcement, measured, limit, ok, evidence):
    return {"condition": cid, "name": name, "type": ctype, "enforced_by_simpy": enforcement, "measured": measured,
            "limit": limit, "satisfied": bool(ok), "evidence": evidence}


def check_c1(inp: ConstraintInputs) -> List[Dict[str, Any]]:
    H = inp.horizon_s
    worst, worst_node = 0, ""
    excl = {n for n, c in inp.node_capacity.items() if c == 1}
    viol = []
    for n, ivs in inp.node_intervals.items():
        p = peak_concurrency([(s, e if e is not None else H) for s, e, _ in ivs])
        if p > inp.node_capacity[n]:
            viol.append(n)
        if n in excl and p > worst:
            worst, worst_node = p, n
    ev = inp.events
    tr = ev[ev.event == "TRAVEL"] if len(ev) else ev
    edge_peak, edge_worst = 0, ""
    for (a, b), g in (tr.groupby(["from_node", "to_node"]) if len(tr) else []):
        p = peak_concurrency(list(zip(g.start_s, g.end_s)))
        if p > edge_peak:
            edge_peak, edge_worst = p, f"{a}->{b}"
    return [
        _row("C1", "track position exclusivity N_occ(t) <= capacity", "inequality",
             "yes: one simpy.Resource per track position (capacity 1; WS5 buffer = n_shuttles); the next position is "
             "reserved before moving", worst, 1, not viol and worst <= 1,
             f"peak shuttles on an exclusive position = {worst} ({worst_node}); {len(excl)} exclusive positions; "
             f"positions over capacity: {viol or 'none'}"),
        _row("C1", "track segment exclusivity N_occ(segment, t) <= 1", "inequality",
             "yes: a segment is entered only with its end position reserved", edge_peak, 1, edge_peak <= 1,
             f"peak shuttles travelling on one segment = {edge_peak} ({edge_worst}); {len(tr)} traversals checked"),
    ]


def check_c2(inp: ConstraintInputs) -> List[Dict[str, Any]]:
    peaks = {r: peak_concurrency([(s, e) for s, e, *_ in ivs]) for r, ivs in inp.res_intervals.items()}
    peaks["FANUC_cell_tray_position"] = peak_concurrency(inp.cell_intervals)
    caps = dict(inp.res_capacity, FANUC_cell_tray_position=inp.cell_capacity)
    over = [r for r in peaks if peaks[r] > caps[r]]
    return [_row("C2", "resource capacity N_busy(t) <= C_r", "inequality",
                 "yes: simpy.Resource(capacity) for robots, conveyor, operators and the cell tray position",
                 max(peaks[r] - caps[r] for r in peaks), 0, not over,
                 "; ".join(f"{r} {peaks[r]}/{caps[r]}" for r in peaks) + f"; over capacity: {over or 'none'}")]


def check_c3(inp: ConstraintInputs) -> List[Dict[str, Any]]:
    lim = inp.material_cfg["trays_per_shuttle"]
    m = max(inp.material_monitor["max_trays_per_shuttle"], max(inp.end_trays_per_shuttle, default=0))
    return [_row("C3", "trays per shuttle N_trays(t) <= 3", "inequality",
                 "by construction (2 trays loaded at WS6 + 1 FANUC tray; assertions at WS6, WS9, WS6 retrieval, WS8)",
                 m, lim, m <= lim, f"maximum trays on a shuttle at any logged event = {m}")]


def check_c4(inp: ConstraintInputs) -> List[Dict[str, Any]]:
    cap = inp.material_cfg["workpieces_per_tray_capacity"]
    prod_cap = max(list(inp.material_cfg["ur3e_products_per_tray"]) + [inp.material_cfg["tray3_male_workpieces"]])
    mm = inp.material_monitor
    end_pieces = max((t.male + t.female for t in inp.trays), default=0)
    end_prod = max((len(t.products) for t in inp.trays), default=0)
    wp, pr, items = max(mm["max_workpieces_per_tray"], end_pieces), max(mm["max_products_per_tray"], end_prod), mm["max_items_per_tray"]
    return [
        _row("C4", "workpieces per tray before assembly N_pieces(t) <= 6", "inequality",
             "by construction (6 per tray at WS9, 3 Male + 3 Female on tray 3); validated configuration", wp, cap, wp <= cap,
             f"maximum workpieces on a tray at any logged event = {wp}"),
        _row("C4", "assembled products per tray <= 3", "inequality", "by construction (ur3e_products_per_tray, FANUC 3)",
             pr, prod_cap, pr <= prod_cap and items <= cap, f"maximum products on a tray = {pr}; maximum workpieces + products = {items}"),
    ]


def check_c5(inp: ConstraintInputs) -> List[Dict[str, Any]]:
    c = inp.counters
    trays = inp.trays
    wip = [t for t in trays if t.location != "STORAGE_B"]
    male_in = c["workpieces_loaded_ws9_male"] + c["workpieces_preloaded_tray3_male"]
    female_in = c["workpieces_loaded_ws9_female"] + c["workpieces_preloaded_tray3_female"]
    male_wip, female_wip = sum(t.male for t in wip), sum(t.female for t in wip)
    assembled = c["products_assembled_ur3e"] + c["products_assembled_fanuc"]
    prod_wip = sum(len(t.products) for t in wip)
    finished = c["products_finished_storage_b"]
    prod_b = sum(len(t.products) for t in trays if t.location == "STORAGE_B")
    rejected = inp.rejected_products
    tray_in = c["trays_loaded_ws6"] + c["trays_fanuc_input"]
    loc = {"on_shuttles": sum(1 for t in trays if t.location.startswith("SHUTTLE_")),
           "in_fanuc_cell": sum(1 for t in trays if t.location.startswith("FANUC_CELL")),
           "storage_b": sum(1 for t in trays if t.location == "STORAGE_B")}
    ids = [p for t in trays for p in t.products]
    initial_wip = 0     # material.initial_trays_on_shuttle = 0: the line starts empty
    eqs = {
        "Male: introduced + initial WIP = unassembled WIP + assembled": (male_in + initial_wip, male_wip + assembled),
        "Female: introduced + initial WIP = unassembled WIP + assembled": (female_in + initial_wip, female_wip + assembled),
        "products: assembled = WIP + finished + rejected": (assembled, prod_wip + finished + rejected),
        "finished counted = products in Storage B": (finished, prod_b),
        "finished = 9 x completed batches": (finished, 9 * c["batches_completed"]),
        "trays: introduced = on shuttles + in FANUC cell + Storage B": (tray_in, sum(loc.values())),
        "unique product ids = assembled": (len(set(ids)), assembled),
    }
    res = {k: a - b for k, (a, b) in eqs.items()}
    ok = all(v == 0 for v in res.values()) and len(ids) == len(set(ids))
    return [_row("C5", "material conservation (2 workpieces -> 1 product; 9 products per batch)", "equality",
                 "partly: material changes state atomically at the end of each operation; balances verified here",
                 max(abs(v) for v in res.values()), 0, ok,
                 f"Male in {male_in}, Female in {female_in}, unassembled WIP {male_wip}M/{female_wip}F, assembled {assembled}, "
                 f"product WIP {prod_wip}, finished {finished}, rejected {rejected}; trays {tray_in} = {loc}; "
                 f"residuals: {sorted(set(res.values()))}")]


_PRECEDENCE = (
    ("P1", "arrival at WS6 before initial tray loading", "arr6", "load"),
    ("P2", "initial tray loading before departure to WS9", "load", "dep6"),
    ("P3", "workpiece setup at WS9 before UR3e assembly at WS4", "ws9", "ws4"),
    ("P4", "FANUC input transfer before FANUC assembly", "fin", "fseq"),
    ("P5", "FANUC assembly completion before output transfer", "fseq", "ret"),
    ("P6", "shuttle back at WS6 before the FANUC tray is transferred onto it", "back6", "ret"),
    ("P7a", "UR3e assembly before quality inspection", "ws4", "qc"),
    ("P7b", "FANUC tray on the shuttle before quality inspection", "ret", "qc"),
    ("P8", "inspection completion before final unloading", "qc", "unl"),
    ("P9", "final unloading before the loop is completed", "unl", "loop"),
)


def precedence_table(inp: ConstraintInputs) -> pd.DataFrame:
    ev = inp.events
    prev_move, move, depart = {}, None, None
    for st in inp.cycle_steps:
        if st["type"] == "move":
            move = st["name"]
            if st.get("on_departure") == "start_fanuc_branch":
                depart = move
        else:
            prev_move[st["name"]] = move
    # Arrival at WS6 before loading: the arrival leg of the same cycle (DES v9.3, cycle starting at WS5) or, when the
    # loading is the first step of the cycle (DES v10.0, cycle starting at WS6), the cycle start of the shuttle at WS6.
    arr6 = ((ev.event == "ARRIVE") & (ev.leg == prev_move["ws6_ur10_load_two_trays"]) if prev_move["ws6_ur10_load_two_trays"]
            else (ev.event == "CYCLE_START") & (ev.node == "S6"))
    sel = {
        "arr6": arr6,
        "load": (ev.event == "PROCESS") & (ev.operation == "ws6_ur10_load_two_trays"),
        "dep6": (ev.event == "TRAVEL") & (ev.leg == depart),
        "ws9": (ev.event == "PROCESS") & (ev.operation == "ws9_load_workpieces"),
        "ws4": (ev.event == "PROCESS") & (ev.operation == "ws4_ur3e_assembly"),
        "fin": (ev.event == "PROCESS") & (ev.operation == "fanuc_input_transfer"),
        "fseq": (ev.event == "PROCESS") & (ev.operation == "fanuc_conveyor_sequence"),
        "back6": (ev.event == "ARRIVE") & (ev.leg == prev_move["ws6_sync_retrieve_fanuc_tray"]),
        "ret": (ev.event == "PROCESS") & (ev.operation == "ws6_sync_retrieve_fanuc_tray"),
        "qc": (ev.event == "PROCESS") & (ev.operation == "ws8_quality_inspection"),
        "unl": (ev.event == "PROCESS") & (ev.operation == "ws8_ur10_unload_three_trays"),
        "loop": ev.event == "CYCLE_END",
    }
    frames = []
    for k, m in sel.items():
        g = ev[m & ev.batch_id.notna()].groupby("batch_id").agg(start=("start_s", "min"), end=("end_s", "max"))
        if k == "loop":            # CYCLE_END spans the whole loop: the loop is completed at its end
            g["start"] = g["end"]
        frames.append(g.add_prefix(f"{k}_"))
    return pd.concat(frames, axis=1)


def check_c6(inp: ConstraintInputs) -> List[Dict[str, Any]]:
    t = precedence_table(inp)
    rows, bad_total = [], 0
    for pid, name, a, b in _PRECEDENCE:
        has_b = t[f"{b}_start"].notna()
        missing = int((has_b & t[f"{a}_end"].isna()).sum())
        both = has_b & t[f"{a}_end"].notna()
        late = int((t.loc[both, f"{a}_end"] > t.loc[both, f"{b}_start"] + EPS).sum())
        bad_total += missing + late
        rows.append(f"{pid} {int(both.sum())} pairs, {missing + late} violations")
    # The two assembly branches must be able to run concurrently
    br = t[t.fseq_start.notna() & t.ws9_start.notna() & t.ws4_end.notna()]
    conc = int(((br.fseq_start < br.ws4_end) & (br.fseq_end > br.ws9_start)).sum())
    return [
        _row("C6", "scheduling precedence P1-P9", "precedence",
             "yes: sequential shuttle process (cycle_steps) and the all_of(shuttle_back, tray_ready) synchronisation",
             bad_total, 0, bad_total == 0, "; ".join(rows)),
        _row("C6", "WS9-WS4 branch and FANUC branch run concurrently", "precedence",
             "yes: FANUC branch is a separate simpy.Process started on the departure from WS6",
             conc, ">= 1", conc >= 1 or len(br) == 0,
             f"FANUC sequence overlaps the WS9 setup-to-UR3e interval in {conc} of {len(br)} batches"),
    ]


def check_constraints(inp: ConstraintInputs) -> pd.DataFrame:
    return pd.DataFrame(check_c1(inp) + check_c2(inp) + check_c3(inp) + check_c4(inp) + check_c5(inp) + check_c6(inp))
