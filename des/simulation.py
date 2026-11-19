"""Shared SimPy simulation core of the six-discipline MDA (DES v9.0).

One SimPy environment, the shared capacity-limited resources, the event log, material counters and the
resource helpers (acquire / work / release) used by every discipline. The production logic itself lives in
the disciplines (cs2_mdo/disciplines); D6 starts one dispatcher process per shuttle.

The code of the single-file DES is split by discipline without changing its behaviour (the regression tests compare
event logs and all result tables).

Coupling signals (signal()) only record that an interface exchange happened. They never yield to SimPy,
so they cannot change the event order.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import simpy

from .config import (REPOSITION_KEYS, RESOURCE_ORDER, ROBOTS, cycle_start_station, speed_factor,
                     validate_config)
from .resources import Shuttle, Tray

DES_VERSION = "10.0"


class SimulationCore:
    """Shared environment, resources and records of one DES run."""

    def __init__(self, cfg: Dict[str, Any]):
        validate_config(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.warnings: List[str] = []
        sim = self.cfg["simulation"]
        self.n_shuttles = int(sim["n_shuttles"])
        self.horizon_h = float(sim["horizon_h"])
        self.horizon_s = self.horizon_h * 3600.0
        self.seed = int(sim["random_seed"])
        self.rng = random.Random(self.seed)
        self.env = simpy.Environment()

        rs = self.cfg["robot_speed"]
        self.robot_factor = {r: speed_factor(rs["duration_factor"][r], rs["setting"][r], r) for r in ROBOTS}
        ss = self.cfg["shuttle_speed"]
        self.shuttle_factor = speed_factor(ss["travel_time_factor"], ss["setting"], "Shuttle")
        pt = {k: float(v) for k, v in self.cfg["process_times_s"].items()}
        self.ref_pt = dict(pt)
        f = self.robot_factor
        self.dur = {
            "ws6_ur10_load_two_trays": pt["ws6_ur10_load_two_trays_s"] * f["UR10"],
            "ws9_load_workpieces": pt["ws9_load_workpieces_s"],
            "ws4_ur3e_assembly": pt["ws4_ur3e_assembly_s"] * f["UR3e"],
            "fanuc_input_transfer": pt["ws6_ur10_fanuc_input_s"] * f["UR10"],
            "fanuc_conveyor_sequence": pt["fanuc_conveyor_sequence_s"],
            "ws6_sync_retrieve_fanuc_tray": pt["ws6_ur10_fanuc_output_s"] * f["UR10"],
            "ws8_quality_inspection": pt["ws8_quality_inspection_s"],
            "ws8_ur10_unload_three_trays": pt["ws8_ur10_unload_three_trays_s"] * f["UR10"],
        }
        self.repos = {r: float(self.cfg["resource_repositioning_s"][k]) for r, k in REPOSITION_KEYS.items()}
        # DV alternatives (absent in the baseline): replacement resources, duration and station equipment per operation
        self.alternatives: Dict[str, Dict[str, Any]] = self.cfg.get("alternatives", {})
        for op, a in self.alternatives.items():
            self.dur[op] = float(a["time_s"])

        self.mat = self.cfg["material"]
        self.cap_tray = int(self.mat["workpieces_per_tray_capacity"])
        self.n_ur3e = int(self.mat["tray1_male_workpieces"])
        self.n_fanuc = int(self.mat["tray3_male_workpieces"])
        self.products_per_batch = self.n_ur3e + self.n_fanuc
        self.workpieces_per_batch = 2 * self.products_per_batch
        self.quality = self.cfg["quality"]

        # Shared capacity-limited equipment and operators, registered in configuration order.
        # Ownership by discipline: cs2_mdo/mda/architecture.py (RESOURCE_OWNER).
        self.resources = {name: simpy.Resource(self.env, capacity=int(s["capacity"])) for name, s in self.cfg["resources"].items()}
        self.res_loc: Dict[str, Optional[str]] = {r: None for r in self.resources}

        self.events: List[Dict[str, Any]] = []
        self.res_intervals: Dict[str, List[Tuple[float, float, str, str, int, Optional[int]]]] = {r: [] for r in self.resources}
        self.res_waits: Dict[str, List[float]] = {r: [] for r in self.resources}
        # station equipment of manual / collaborative alternatives: an energy consumer, in use during its task
        self.equipment = [a["equipment"] for a in self.alternatives.values() if a.get("equipment")]
        for eq in self.equipment:
            self.res_intervals[eq] = []
        self.loop_records: List[Dict[str, Any]] = []
        self.batch_records: List[Dict[str, Any]] = []
        self.jobs: Dict[int, Any] = {}
        self.trays: Dict[int, Tray] = {}
        self._tray_counter = self._product_counter = self._cycle_counter = 0
        self.c = {k: 0 for k in (
            "trays_loaded_ws6", "trays_fanuc_input", "trays_retrieved_from_fanuc", "trays_unloaded_ws8",
            "workpieces_loaded_ws9_male", "workpieces_loaded_ws9_female",
            "workpieces_preloaded_tray3_male", "workpieces_preloaded_tray3_female",
            "workpieces_consumed_ur3e", "workpieces_consumed_fanuc",
            "products_assembled_ur3e", "products_assembled_fanuc", "products_inspected",
            "defects_detected", "defects_corrected", "products_finished_storage_b",
            "cycles_started", "loops_completed", "batches_completed",
            "ur10_load_operations", "ur10_fanuc_input_operations", "ur10_fanuc_output_operations",
            "ur10_unload_operations", "ur3e_assemblies", "fanuc_assemblies", "ws9_operations", "ws8_inspections",
            "batch_match_failures")}

        self.shuttles: List[Shuttle] = [Shuttle(shuttle_id=sid, node=self.cfg["initial_positions"][str(sid)])
                                        for sid in range(1, self.n_shuttles + 1)]
        self.cycle_start_station = cycle_start_station(self.cfg)
        self.warnings.append(
            f"FANUC cell (robot + conveyor) is one energy consumer, active during the {self.dur['fanuc_conveyor_sequence']:.0f} s "
            "sequence and idle otherwise.")

        # Interface diagnostics (no effect on the simulation)
        self.coupling_log: Dict[str, List[Any]] = {}             # id -> [count, first_t, last_t]
        self.material_monitor = {"max_trays_per_shuttle": 0, "max_workpieces_per_tray": 0, "max_products_per_tray": 0,
                                 "max_items_per_tray": 0}
        self.pre_work: Optional[Callable[..., Any]] = None    # D6 energy-management hook (DV9 wake-up delay)
        # Uncertainty extension (UDES 1.0, cs2_mdo/uncertainty): resource breakdown model. None in deterministic mode,
        # where _work / _reposition / D1.move execute exactly the original code path.
        self.failure: Optional[Any] = None
        self._ran = False

    # ------------------------------------------------------------------ DV alternatives
    def task_resources(self, op: str, default: List[str]) -> Tuple[List[str], List[str]]:
        """(resources to acquire, resources + station equipment in use) of operation `op`."""
        a = self.alternatives.get(op)
        if a is None:
            return default, default
        return list(a["resources"]), list(a["resources"]) + ([a["equipment"]] if a.get("equipment") else [])

    # ------------------------------------------------------------------ interface diagnostics
    def signal(self, cid: str) -> None:
        """Record one exchange of coupling `cid` (see mda/couplings.py). Never yields."""
        rec = self.coupling_log.get(cid)
        if rec is None:
            self.coupling_log[cid] = [1, self.env.now, self.env.now]
        else:
            rec[0] += 1
            rec[2] = self.env.now

    def _monitor_material(self, trays: List[Tray], on_shuttle: bool) -> None:
        mm = self.material_monitor
        if on_shuttle:
            mm["max_trays_per_shuttle"] = max(mm["max_trays_per_shuttle"], len(trays))
        for t in trays:
            mm["max_workpieces_per_tray"] = max(mm["max_workpieces_per_tray"], t.male + t.female)
            mm["max_products_per_tray"] = max(mm["max_products_per_tray"], len(t.products))
            mm["max_items_per_tray"] = max(mm["max_items_per_tray"], t.male + t.female + len(t.products))

    # ------------------------------------------------------------------ helpers
    def sample_duration(self, key: str) -> float:
        """Hook for future stochastic processing times. Deterministic now."""
        return self.dur[key]

    def _new_tray(self, state: str, location: str, origin: str, batch_id: int) -> Tray:
        self._tray_counter += 1
        t = Tray(tray_id=self._tray_counter, state=state, batch_id=batch_id, location=location, origin=origin)
        self.trays[t.tray_id] = t
        return t

    @staticmethod
    def _tray_text(trays: List[Tray]) -> str:
        return "; ".join(f"T{t.tray_id}:{t.state}(M{t.male},F{t.female},P{len(t.products)})" for t in trays)

    def log(self, who: Any, event: str, start: float, end: float, **kw: Any) -> None:
        is_shuttle = isinstance(who, Shuttle)
        self.events.append({
            "start_s": start, "end_s": end, "duration_s": end - start,
            "shuttle_id": who.shuttle_id if who else None, "batch_id": who.cycle_id if who else None,
            "event": event, "node": kw.get("node", who.node if is_shuttle else "FANUC_CELL"),
            "from_node": kw.get("from_node"), "to_node": kw.get("to_node"), "leg": kw.get("leg"),
            "operation": kw.get("operation"), "resource": kw.get("resource"), "category": kw.get("category"),
            "trays_on_shuttle": len(who.trays) if is_shuttle else None,
            "tray_states": self._tray_text(who.trays) if is_shuttle else (self._tray_text([who.tray]) if who and who.tray else None),
            "note": kw.get("note", ""),
        })
        if is_shuttle:
            self._monitor_material(who.trays, True)
        elif who is not None and who.tray is not None:
            self._monitor_material([who.tray], False)

    @staticmethod
    def _acc(who: Any, key: str, value: float) -> None:
        who.acc[key] = who.acc.get(key, 0.0) + value

    # ------------------------------------------------------------------ resource helpers (who = Shuttle or FanucJob)
    def _acquire(self, who: Any, names: List[str], op: str, node: Optional[str] = None):
        reqs = []
        for n in sorted(names, key=RESOURCE_ORDER.index):     # D6 policy: fixed acquisition order, FIFO per resource
            t0 = self.env.now
            r = self.resources[n].request()
            yield r
            w = self.env.now - t0
            self.res_waits[n].append(w)
            if w > 0:
                kw = {"node": node} if node else {}
                self.log(who, "WAIT_FOR_RESOURCE", t0, self.env.now, operation=op, resource=n,
                         category="waiting_resource", **kw)
                self._acc(who, "wait_resource_s", w)
            reqs.append((n, r))
        self.signal("CG5.4")
        return reqs

    def _release(self, reqs) -> None:
        for n, r in reqs:
            self.resources[n].release(r)

    def _reposition(self, who: Any, name: str, station: str, op: str):
        if name not in REPOSITION_KEYS:
            return
        loc = self.res_loc[name]
        if loc is not None and loc != station and self.repos[name] > 0:
            d = self.repos[name]
            t0 = self.env.now
            if self.failure is None:
                self.res_intervals[name].append((t0, t0 + d, "reposition", f"{loc}->{station}", who.shuttle_id, who.cycle_id))
                yield self.env.timeout(d)
            else:   # uncertain mode: preempt-resume on a breakdown of the robot
                yield from self.failure.run_segment(who, [name], d, "reposition", f"{loc}->{station}", station)
            self.log(who, "RESOURCE_REPOSITION", t0, self.env.now, node=station, operation=op, resource=name,
                     category="synchronization", note=f"{name} {loc} -> {station}")
            self._acc(who, "sync_s", d)
        self.res_loc[name] = station

    def _work(self, who: Any, names: List[str], duration: float, op: str, category: str = "processing",
              note_fn=None, node: Optional[str] = None):
        if self.pre_work is not None:
            yield from self.pre_work(who, names, op, node)
        t0 = self.env.now
        if self.failure is None:
            for n in names:
                self.res_intervals[n].append((t0, t0 + duration, op, op, who.shuttle_id, who.cycle_id))
            yield self.env.timeout(duration)
        else:   # uncertain mode: no start on a failed resource; preempt-resume if one fails during the task
            yield from self.failure.run_segment(who, names, duration, op, op, node)
        note = note_fn() if note_fn else ""
        kw = {"node": node} if node else {}
        self.log(who, "PROCESS", t0, self.env.now, operation=op, resource="+".join(names), category=category, note=note, **kw)
        self._acc(who, "processing_s", duration)

    # ------------------------------------------------------------------ execution
    def run(self, shuttle_process: Callable[[Shuttle], Any]) -> "SimulationCore":
        """Start one dispatcher process per shuttle (provided by D6) and run until the horizon."""
        if self._ran:
            raise RuntimeError("A model instance can only be run once")
        for sh in self.shuttles:
            self.env.process(shuttle_process(sh))
        self.env.run(until=self.horizon_s)
        self._ran = True
        return self
