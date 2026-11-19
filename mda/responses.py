"""CG6 - System-level response aggregation (D1-D6 -> MDA coordinator).

Collects production counts, throughput, batches and loops, resource activity and utilisation, waiting
times, station occupancy, energy by consumer, total and specific energy, and the material-conservation
checks. The aggregation code (activity_times, conservation, results) is copied verbatim from
montrac_des_v8_8_tlpp2.py; ResponseAggregator only states which discipline supplies each quantity.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..des.config import HUMAN_OPERATORS, PRODUCTION_PROCESS, ROBOTS, SHUTTLE_OPERATIONS
from ..des.energy import energy_summary, energy_table, infrastructure_power_W
from ..des.simulation import DES_VERSION

MODEL_VERSION = DES_VERSION


class ResponseAggregator:
    """Read-only view of one finished run, with the attribute names used by the result code."""

    def __init__(self, core, disciplines: Dict[str, Any]):
        d1, d6 = disciplines["D1"], disciplines["D6"]
        # shared core (all disciplines)
        self.cfg, self.env, self.c, self.warnings = core.cfg, core.env, core.c, core.warnings
        self.n_shuttles, self.horizon_h, self.horizon_s, self.seed = core.n_shuttles, core.horizon_h, core.horizon_s, core.seed
        self.dur, self.ref_pt, self.repos, self.quality = core.dur, core.ref_pt, core.repos, core.quality
        self.cap_tray, self.products_per_batch = core.cap_tray, core.products_per_batch
        self.events, self.res_intervals, self.res_waits, self.resources = core.events, core.res_intervals, core.res_waits, core.resources
        self.shuttles, self.trays, self.cycle_start_station, self._ran = core.shuttles, core.trays, core.cycle_start_station, core._ran
        # D1 transportation: loops, track positions, movement
        self.loop_records = core.loop_records
        self.move_intervals, self.node_intervals, self.node_waits = d1.move_intervals, d1.node_intervals, d1.node_waits
        self.node_capacity, self.leg_hops, self.seg_value = d1.node_capacity, d1.leg_hops, d1.seg_value
        self.clearance_events = d1.clearance_events
        # D3 assembly: FANUC jobs; D4 quality and dispatch: completed batches
        self.jobs, self.batch_records = core.jobs, core.batch_records
        # D6 coordination: control delays
        self.control_delays = d6.control_delays
        # D6 energy-management policy (DV9): standby time per covered consumer (empty in the baseline)
        self.standby = d6.standby_times(core.horizon_s)
        # collaborative robot modes (DV levels 2): {consumer: [(time in the collaborative task, power)]}
        self.modes: Dict[str, List[Tuple[float, float]]] = {}
        for op, a in core.alternatives.items():
            rm = a.get("robot_mode")
            if rm:
                ivs = self._clip([(s, e) for s, e, o, *_ in core.res_intervals[rm["resource"]] if o == op], core.horizon_s)
                self.modes.setdefault(rm["consumer"], []).append((sum(e - s for s, e in ivs), float(rm["active_W"])))

    # ------------------------------------------------------------------ result tables, with the
    # MDA extensions marked "DV alternatives": FANUC_cell activity from the FANUC robot's intervals (identical
    # to the 196 s job times in the baseline), activity rows for alternative resources and station equipment,
    # and the DV9 standby times passed to the energy table.

    @staticmethod
    def _clip(intervals, H: float) -> List[Tuple[float, float]]:
        out = sorted((max(0.0, s), min(e, H)) for s, e in intervals if min(e, H) > max(0.0, s))
        merged: List[List[float]] = []
        for s, e in out:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    @staticmethod
    def _intersection(a, b) -> float:
        return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)

    def activity_times(self, ev: pd.DataFrame, H: float) -> pd.DataFrame:
        """Active / idle time per energy consumer; active + idle = H (intervals clipped to H)."""
        def total(ivs) -> float:
            return sum(e - s_ for s_, e in self._clip([(a, b) for a, b in ivs if a is not None and b is not None], H))

        def row(name, active, definition, **parts):
            return {"resource": name, "active_s": active, "idle_s": H - active, "active_pct": 100.0 * active / H,
                    "idle_pct": 100.0 * (H - active) / H, "active_definition": definition, **parts}

        rows = []
        for sh in self.shuttles:
            sid = sh.shuttle_id
            ops = [(a, b) for r in self.res_intervals.values() for a, b, op, _, who, _ in r
                   if who == sid and op in SHUTTLE_OPERATIONS]
            active = total(self.move_intervals[sid] + ops)
            # FANUC-branch events carry the shuttle_id of their batch; they are the tray's waits, not the shuttle's
            mine = ev[(ev.shuttle_id == sid) & ~ev.operation.fillna("").str.startswith("fanuc_") & ev.category.isin(
                ["waiting_blocking", "clearance", "waiting_resource", "waiting_fanuc_sync"])] if len(ev) else ev
            w = {c: total(list(zip(g.start_s, g.end_s))) for c, g in mine.groupby("category")} if len(mine) else {}
            rows.append(row(f"Shuttle_{sid}", active, "moving + operation performed on the shuttle at a station",
                            active_moving_s=total(self.move_intervals[sid]), active_at_station_operation_s=total(ops),
                            idle_blocked_by_other_shuttle_s=w.get("waiting_blocking", 0.0),
                            idle_station_release_clearance_s=w.get("clearance", 0.0),
                            idle_waiting_for_robot_or_operator_s=w.get("waiting_resource", 0.0),
                            idle_waiting_for_fanuc_tray_s=w.get("waiting_fanuc_sync", 0.0),
                            idle_unlogged_wait_at_horizon_end_s=max(0.0, H - active - sum(w.values()))))
        for r in ("UR10", "UR3e"):
            ivs = self.res_intervals[r]
            per_task = {f"active_{op}_s": total([(a, b) for a, b, o, *_ in ivs if o == op])
                        for op in dict.fromkeys(o for _, _, o, *_ in ivs)}
            rows.append(row(r, total([(a, b) for a, b, *_ in ivs]), "executing a task", **per_task))
        for r, st in HUMAN_OPERATORS.items():
            ivs = self.res_intervals[r]
            rows.append(row(r, total([(a, b) for a, b, *_ in ivs]), f"performing its task at {st} (human: no electrical consumption)",
                            busy_intervals=len(ivs), task_in_progress_at_horizon=sum(1 for a, b, *_ in ivs if a < H < b),
                            shuttle_wait_for_operator_s=float(sum(self.res_waits[r]))))
        jobs = [j for j in self.jobs.values() if j.t_input_start is not None]

        def jt(a: str, b: str):
            return [(getattr(j, a), getattr(j, b) if getattr(j, b) is not None else H) for j in jobs]   # unfinished at H -> H
        occupied = total(jt("t_input_start", "t_cell_released"))
        # DV alternatives: the cell consumer is active while the FANUC robot works (none in manual FANUC assembly)
        rows.append(row("FANUC_cell", total([(a, b) for a, b, *_ in self.res_intervals["FANUC"]]),
                        "196 s sequence: conveyor in + FANUC assembly + conveyor out to UR10 pick point",
                        occupied_by_tray_s=occupied,
                        idle_occupied_ur10_input_31s_s=total(jt("t_input_start", "t_input_end")),
                        idle_occupied_tray_waiting_on_conveyor_s=total(jt("t_ready", "t_retrieval_start")),
                        idle_occupied_ur10_transfer_to_shuttle_18s_s=total(jt("t_retrieval_start", "t_retrieval_end")),
                        idle_cell_empty_s=H - occupied))
        # DV alternatives: added resources (operators, UR3e_WS9) and station equipment (WSE_*)
        for r, ivs in self.res_intervals.items():
            if r in ("UR10", "UR3e", "FANUC", "Conveyor") or r in HUMAN_OPERATORS:
                continue
            spec = self.cfg["resources"].get(r)
            what = ("performing its task (human: no electrical consumption)" if spec and spec["type"] == "human" else
                    "executing a task" if spec else "station equipment in use during its task")
            rows.append(row(r, total([(a, b) for a, b, *_ in ivs]), what, busy_intervals=len(ivs)))
        df = pd.DataFrame(rows)
        base = ["resource", "active_s", "idle_s", "active_pct", "idle_pct", "active_definition"]
        return df[base + [c for c in df.columns if c not in base]]

    def conservation(self) -> Dict[str, Any]:
        on_sh = sum(len(s.trays) for s in self.shuttles)
        in_cell = sum(1 for t in self.trays.values() if t.location.startswith("FANUC_CELL"))
        in_b = sum(1 for t in self.trays.values() if t.location == "STORAGE_B")
        active = [t for t in self.trays.values() if t.location != "STORAGE_B"]
        c = self.c
        male = c["workpieces_loaded_ws9_male"] + c["workpieces_preloaded_tray3_male"]
        female = c["workpieces_loaded_ws9_female"] + c["workpieces_preloaded_tray3_female"]
        wp_in = male + female
        wp_cons = c["workpieces_consumed_ur3e"] + c["workpieces_consumed_fanuc"]
        wp_wip = sum(t.male + t.female for t in active)
        assembled = c["products_assembled_ur3e"] + c["products_assembled_fanuc"]
        prod_wip = sum(len(t.products) for t in active)
        prod_b = sum(len(t.products) for t in self.trays.values() if t.location == "STORAGE_B")
        tray_in = c["trays_loaded_ws6"] + c["trays_fanuc_input"]
        all_ids = [p for t in self.trays.values() for p in t.products]
        checks = {
            "tray_balance_ok": tray_in == on_sh + in_cell + in_b and len(self.trays) == tray_in,
            "trays_per_shuttle_ok": all(len(s.trays) <= 3 for s in self.shuttles),
            "tray_capacity_ok": all(t.male + t.female + len(t.products) <= self.cap_tray for t in self.trays.values()),
            "workpiece_balance_ok": wp_in == wp_cons + wp_wip,
            "male_female_equal_ok": male == female,
            "consumption_ratio_ok": wp_cons == 2 * assembled,
            "product_balance_ok": assembled == prod_b + prod_wip,
            "no_duplicate_products_ok": len(all_ids) == len(set(all_ids)) == assembled,
            "finished_count_ok": prod_b == c["products_finished_storage_b"] == self.products_per_batch * c["batches_completed"],
            "trays_per_batch_ok": c["trays_unloaded_ws8"] == 3 * c["batches_completed"] == in_b,
            "batch_matching_ok": c["batch_match_failures"] == 0,
        }
        return {"tray_inputs_ws6": c["trays_loaded_ws6"], "tray_inputs_fanuc": c["trays_fanuc_input"],
                "trays_on_shuttles_end": on_sh, "trays_in_fanuc_cell_end": in_cell, "trays_in_storage_b_end": in_b,
                "workpieces_input": wp_in, "workpieces_consumed": wp_cons, "workpieces_in_wip_end": wp_wip,
                "products_assembled": assembled, "products_in_wip_end": prod_wip, "products_in_storage_b_end": prod_b,
                **checks, "all_conservation_ok": all(checks.values())}

    def results(self) -> Dict[str, Any]:
        if not self._ran:
            raise RuntimeError("Call run() first")
        H, hours = self.horizon_s, self.horizon_h
        ev = pd.DataFrame(self.events).sort_values(["start_s", "end_s"], kind="stable").reset_index(drop=True)
        loops = pd.DataFrame(self.loop_records)
        batches = pd.DataFrame(self.batch_records)

        job_rows = []
        for j in self.jobs.values():
            row = {k: getattr(j, k) for k in (
                "batch_id", "shuttle_id", "t_shuttle_departs", "t_position_acquired", "t_input_start", "t_input_end", "t_cell_start", "t_cell_end",
                "t_ready", "t_shuttle_back", "t_sync_complete", "t_retrieval_start", "t_retrieval_end", "t_ur3e_start", "t_ur3e_end", "t_cell_released")}
            done = j.t_retrieval_start is not None
            row["ur3e_branch_s"] = j.t_shuttle_back - j.t_shuttle_departs if j.t_shuttle_back is not None else np.nan
            row["fanuc_branch_s"] = j.t_ready - j.t_shuttle_departs if j.t_ready is not None else np.nan
            row["tray_wait_at_output_s"] = j.t_retrieval_start - j.t_ready if done else np.nan
            row["fanuc_wait_for_cell_s"] = j.t_position_acquired - j.t_shuttle_departs if j.t_position_acquired is not None else np.nan
            row["output_wait_for_shuttle_s"] = max(0.0, min(j.t_retrieval_start, j.t_shuttle_back) - j.t_ready) if done else np.nan
            row["output_wait_for_ur10_s"] = j.t_retrieval_start - max(j.t_ready, j.t_shuttle_back) if done else np.nan
            row["cell_position_held_s"] = j.t_cell_released - j.t_position_acquired if j.t_cell_released is not None else np.nan
            row["shuttle_wait_for_fanuc_s"] = j.t_sync_complete - j.t_shuttle_back if j.t_sync_complete is not None else np.nan
            row["shuttle_wait_for_ur10_s"] = j.t_retrieval_start - j.t_sync_complete if done else np.nan
            row["assembly_overlap_s"] = (self._intersection([(j.t_ur3e_start, j.t_ur3e_end)], [(j.t_cell_start, j.t_cell_end)])
                                         if j.t_ur3e_end is not None and j.t_cell_end is not None else np.nan)
            row["tray_id"] = j.tray.tray_id if j.tray else None
            row["tray_batch_id"] = j.tray.batch_id if j.tray else None
            job_rows.append(row)
        jobs_df = pd.DataFrame(job_rows)

        res_rows = []
        for r in self.resources:
            occ = sum(e - s for s, e in self._clip([(s, e) for s, e, *_ in self.res_intervals[r]], H))
            waits = self.res_waits[r]
            res_rows.append({"resource": r, "occupied_time_s": occ, "occupancy_pct": 100.0 * occ / H,
                             "busy_intervals": len(self.res_intervals[r]), "requests": len(waits),
                             "mean_wait_s": float(np.mean(waits)) if waits else 0.0,
                             "max_wait_s": float(np.max(waits)) if waits else 0.0,
                             "total_wait_s": float(np.sum(waits)) if waits else 0.0})
        for sh in self.shuttles:
            occ = sum(e - s for s, e in self._clip(self.move_intervals[sh.shuttle_id], H))
            res_rows.append({"resource": f"Shuttle_{sh.shuttle_id}", "occupied_time_s": occ, "occupancy_pct": 100.0 * occ / H,
                             "busy_intervals": len(self.move_intervals[sh.shuttle_id]), "requests": np.nan,
                             "mean_wait_s": np.nan, "max_wait_s": np.nan, "total_wait_s": np.nan})
        resources_df = pd.DataFrame(res_rows)
        # occupied_time_s: busy time of the resource; for shuttles it is moving time (energy uses the activity table)
        resources_df["occupied_basis"] = ["moving" if r.startswith("Shuttle_") else "busy" for r in resources_df["resource"]]

        st_rows = []
        for nid, ivs in self.node_intervals.items():
            occ = sum(max(0.0, min(e if e is not None else H, H) - max(s, 0.0)) for s, e, _ in ivs)
            cap = self.node_capacity[nid]
            pts = sorted([(s, 1) for s, _, _ in ivs] + [(e if e is not None else H, -1) for _, e, _ in ivs],
                         key=lambda x: (x[0], x[1]))
            cur = peak = 0
            for _, d in pts:
                cur += d
                peak = max(peak, cur)
            waits = self.node_waits[nid]
            st_rows.append({"station": nid, "capacity": cap,
                            "production_operations": ", ".join(o for o, s in self.cfg["operations"].items() if s["station"] == nid) or "none (pass-through/buffer)",
                            "occupancy_time_s": occ, "occupancy_pct_of_capacity": 100.0 * occ / (H * cap),
                            "entries": len(ivs), "peak_concurrent_shuttles": peak,
                            "mean_entry_wait_s": float(np.mean(waits)) if waits else 0.0,
                            "max_entry_wait_s": float(np.max(waits)) if waits else 0.0,
                            "total_entry_wait_s": float(np.sum(waits)) if waits else 0.0})
        stations_df = pd.DataFrame(st_rows)

        finished = self.c["products_finished_storage_b"]
        cons = self.conservation()
        activity_df = self.activity_times(ev, H)
        energy_df = energy_table(self.cfg["energy"], {r.resource: (r.active_s, r.idle_s) for r in activity_df.itertuples()}, H,
                                 standby=self.standby or None, modes=self.modes or None)
        energy_kpis = energy_summary(energy_df, finished, hours)

        prod_res = resources_df[~resources_df["resource"].str.startswith("Shuttle_")]
        cand = [(r["resource"], float(r["occupancy_pct"]), "resource occupancy") for _, r in prod_res.iterrows()]
        cand += [(f"{r['station']} station", float(r["occupancy_pct_of_capacity"]), "station occupancy")
                 for _, r in stations_df.iterrows() if r["capacity"] == 1]
        bott_name, bott_pct, bott_kind = max(cand, key=lambda x: x[1])
        # Highest occupancy is reported as such: it does not identify the throughput
        # constraint when rail blocking starves stations (see the bottleneck probe).
        highest = f"{bott_name} ({bott_kind} {bott_pct:.1f}%)"

        d = self.dur
        t_a = (sum(h[2] for s in ("L1_WS6_to_WS9", "L2_WS9_to_WS4", "L3_WS4_to_WS6") for h in self.leg_hops[s])
               + d["ws9_load_workpieces"] + d["ws4_ur3e_assembly"])
        t_b = d["fanuc_input_transfer"] + d["fanuc_conveyor_sequence"]
        travel = sum(h[2] for s in self.cfg["cycle_steps"] if s["type"] == "move" for h in self.leg_hops[s["name"]])
        shuttle_proc = sum(d[k] for k in SHUTTLE_OPERATIONS)
        loop_s = shuttle_proc + travel + max(0.0, t_b - t_a)
        demand = {"UR10": d["ws6_ur10_load_two_trays"] + d["fanuc_input_transfer"] + d["ws6_sync_retrieve_fanuc_tray"] + d["ws8_ur10_unload_three_trays"],
                  "UR3e": d["ws4_ur3e_assembly"], "FANUC": d["fanuc_conveyor_sequence"],
                  "Operator_WS9": d["ws9_load_workpieces"], "Operator_WS8": d["ws8_quality_inspection"]}
        res_b = max(demand, key=demand.get)
        bound = min(self.n_shuttles * 3600.0 / loop_s, 3600.0 / demand[res_b])

        ur3 = self._clip([(s, e) for s, e, *_ in self.res_intervals["UR3e"]], H)
        fan = self._clip([(s, e) for s, e, *_ in self.res_intervals["FANUC"]], H)
        both_s = self._intersection(ur3, fan)
        both_ops_s = self._intersection(*(self._clip([(s, e) for s, e, *_ in self.res_intervals[r]], H) for r in HUMAN_OPERATORS))
        act = activity_df.set_index("resource")
        # FANUC queue = batches released by their shuttle that are still waiting for a cell position
        queue_ivs = [(j.t_shuttle_departs, j.t_position_acquired if j.t_position_acquired is not None else H) for j in self.jobs.values()]
        pos_ivs = [(j.t_position_acquired, j.t_cell_released if j.t_cell_released is not None else H)
                   for j in self.jobs.values() if j.t_position_acquired is not None]
        starts = sorted(j.t_cell_start for j in self.jobs.values() if j.t_cell_start is not None)
        seq = sorted((j.t_cell_start, j.t_cell_end) for j in self.jobs.values() if j.t_cell_end is not None)
        gaps = [b[0] - a[1] for a, b in zip(seq, seq[1:])]
        b_done = sorted(r["completed_s"] for r in self.batch_records)
        q_time = sum(max(0.0, min(e, H) - max(s, 0.0)) for s, e in queue_ivs)
        pts = sorted([(s, 1) for s, e in queue_ivs if e > s] + [(e, -1) for s, e in queue_ivs if e > s], key=lambda x: (x[0], x[1]))
        cur = q_max = 0
        for _, dq in pts:
            cur += dq
            q_max = max(q_max, cur)

        ct = loops["cycle_time_s"] if not loops.empty else pd.Series(dtype=float)
        first = loops[loops.shuttle_loop_index == 1] if not loops.empty else loops
        rep = loops[loops.shuttle_loop_index > 1] if not loops.empty else loops
        jd = jobs_df[jobs_df.t_retrieval_start.notna()] if not jobs_df.empty else jobs_df

        def m(frame, col):
            return float(frame[col].mean()) if len(frame) else np.nan

        kpis = {
            "model_version": MODEL_VERSION, "production_process": PRODUCTION_PROCESS,
            "n_shuttles": self.n_shuttles, "horizon_h": hours, "horizon_s": H, "random_seed": self.seed, "env_time_end_s": self.env.now,
            "cycles_started": self.c["cycles_started"], "completed_shuttle_loops": self.c["loops_completed"],
            "completed_production_batches": self.c["batches_completed"],
            "finished_products_storage_b": finished, "finished_trays_storage_b": self.c["trays_unloaded_ws8"],
            "throughput_products_per_h": finished / hours, "throughput_batches_per_h": self.c["batches_completed"] / hours,
            "startup_to_first_ws6_arrival_s_mean": float(np.mean([s.t_first_arrival_ws6 for s in self.shuttles if s.t_first_arrival_ws6 is not None])),
            "mean_cycle_time_s": float(ct.mean()) if len(ct) else np.nan,
            "cycle_boundary": f"{self.cycle_start_station} -> {self.cycle_start_station}",
            "first_cycle_s_mean": m(first, "cycle_time_s"),
            **{f"first_cycle_end_s_shuttle_{int(r.shuttle_id)}": float(r.cycle_end_s) for r in first.itertuples()},
            "repeated_cycle_time_s_mean": m(rep, "cycle_time_s"),
            "min_cycle_time_s": float(ct.min()) if len(ct) else np.nan, "max_cycle_time_s": float(ct.max()) if len(ct) else np.nan,
            "mean_processing_s_per_cycle": m(loops, "processing_s"), "mean_travel_s_per_cycle": m(loops, "travel_s"),
            "mean_sync_control_s_per_cycle": m(loops, "sync_s"), "mean_wait_resource_s_per_cycle": m(loops, "wait_resource_s"),
            "mean_wait_blocking_s_per_cycle": m(loops, "wait_node_s"), "mean_wait_fanuc_sync_s_per_cycle": m(loops, "wait_fanuc_s"),
            "ur3e_branch_s_mean": m(jd, "ur3e_branch_s"), "fanuc_branch_s_mean": m(jd, "fanuc_branch_s"),
            "assembly_overlap_s_mean": m(jd, "assembly_overlap_s"),
            "fanuc_tray_wait_at_output_s_mean": m(jd, "tray_wait_at_output_s"),
            "fanuc_output_wait_for_shuttle_s_mean": m(jd, "output_wait_for_shuttle_s"),
            "fanuc_output_wait_for_ur10_s_mean": m(jd, "output_wait_for_ur10_s"),
            "fanuc_batch_wait_for_cell_s_mean": m(jobs_df[jobs_df.t_position_acquired.notna()], "fanuc_wait_for_cell_s") if not jobs_df.empty else np.nan,
            "fanuc_tray_wait_for_output_position_s_total": sum(e["duration_s"] for e in self.events if e["event"] == "WAIT_FOR_FANUC_OUTPUT"),
            "fanuc_cell_turnover_s_mean": float(np.mean(np.diff(starts))) if len(starts) > 1 else np.nan,
            "fanuc_gap_between_sequences_s_mean": float(np.mean(gaps)) if gaps else np.nan,
            "fanuc_gap_between_sequences_s_min": float(np.min(gaps)) if gaps else np.nan,
            "fanuc_cell_position_occupancy_pct": 100.0 * sum(max(0.0, min(e, H) - max(s_, 0.0)) for s_, e in pos_ivs) / (H * self.cfg["fanuc_cell"]["tray_positions"]),
            "mean_batch_completion_interval_s": float(np.mean(np.diff(b_done))) if len(b_done) > 1 else np.nan,
            "clearance_events": self.clearance_events,
            "mean_clearance_s_per_cycle": m(loops, "clearance_s"),
            "shuttle_wait_for_fanuc_at_ws6_s_mean": m(jd, "shuttle_wait_for_fanuc_s"),
            "shuttle_wait_for_ur10_at_ws6_s_mean": m(jd, "shuttle_wait_for_ur10_s"),
            "fanuc_queue_length_mean": q_time / H, "fanuc_queue_length_max": q_max,
            "pct_time_both_assemblies_simultaneous": 100.0 * both_s / H,
            **{f"operator.{r}.{k}": float(act.at[r, c]) for r in HUMAN_OPERATORS for k, c in (
                ("active_s", "active_s"), ("idle_s", "idle_s"), ("utilization_pct", "active_pct"),
                ("shuttle_wait_for_operator_s", "shuttle_wait_for_operator_s"))},
            "operators_simultaneously_active_s": both_ops_s,
            "pct_time_both_operators_simultaneous": 100.0 * both_ops_s / H,
            "analytic_branch_a_s": t_a, "analytic_branch_b_s": t_b,
            "analytic_single_shuttle_loop_s": loop_s, "theoretical_batch_bound_per_h": bound,
            "most_loaded_resource_per_batch": res_b, "most_loaded_resource_demand_s": demand[res_b],
            "highest_occupancy": highest,
            **{f"active_s.{r.resource}": r.active_s for r in activity_df.itertuples()},
            **{f"energy_Wh.{r.consumer}": r.total_energy_Wh for r in energy_df.itertuples() if r.group != "total"},
            **energy_kpis,
            "defects_detected": self.c["defects_detected"], "defects_corrected": self.c["defects_corrected"],
            "end_trays_on_shuttles": cons["trays_on_shuttles_end"], "end_trays_in_fanuc_cell": cons["trays_in_fanuc_cell_end"],
            "end_workpieces_wip": cons["workpieces_in_wip_end"], "end_products_wip_not_finished": cons["products_in_wip_end"],
            "end_cycles_in_progress": self.c["cycles_started"] - self.c["loops_completed"],
            "warnings": " | ".join(self.warnings),
            **{f"conservation.{k}": v for k, v in cons.items()},
            **{f"counter.{k}": v for k, v in self.c.items()},
            **{f"input.process_times_s.{k}": v for k, v in self.ref_pt.items()},
            **{f"effective_duration_s.{k}": v for k, v in self.dur.items()},
            **{f"input.robot_speed.{r}": self.cfg["robot_speed"]["setting"][r] for r in ROBOTS},
            "input.shuttle_speed": self.cfg["shuttle_speed"]["setting"],
            **{f"input.travel_s.{k}": v for k, v in self.seg_value.items()},
            **{f"input.control_delay_s.{k}": v for k, v in self.control_delays.items()},
            **{f"input.reposition_s.{k}": v for k, v in self.repos.items()},
            "input.energy.transfer_line_infrastructure_W": infrastructure_power_W(self.cfg["energy"]),
            **{f"input.energy.{r.consumer}.{k}_W": getattr(r, f"{k}_power_W") for r in energy_df.itertuples()
               if r.group not in ("total", "montrac_infrastructure") and not r.consumer.startswith("Shuttle_") for k in ("active", "idle")},
            "input.energy.shuttle.active_W": self.cfg["energy"]["shuttle"]["active_W"],
            "input.energy.shuttle.idle_W": self.cfg["energy"]["shuttle"]["idle_W"],
            "input.defect_probability_per_product": self.quality["defect_probability_per_product"],
            "input.tray3_preloaded_from_warehouse": self.cfg["tray3_workpieces"]["preloaded_from_warehouse"],
            "input.initial_positions": json.dumps({k: v for k, v in self.cfg["initial_positions"].items() if int(k) <= self.n_shuttles}),
            **{f"input.node_capacity.{k}": v for k, v in self.node_capacity.items()},
            **{f"input.fanuc_cell.{k}": v for k, v in self.cfg["fanuc_cell"].items() if k != "note"},
        }
        return {"kpis": kpis, "events": ev, "loops": loops, "batches": batches, "jobs": jobs_df, "resources": resources_df,
                "stations": stations_df, "activity": activity_df, "energy": energy_df,
                "conservation": cons, "warnings": list(self.warnings)}
