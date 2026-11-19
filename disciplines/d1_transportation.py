"""D1 - Shuttle transportation (design variable DV1: number of active Montrac shuttles).

Shuttle movement and routing over the rail graph, exclusive track positions (one SimPy Resource per node),
station arrival / departure, blocking waits and the station-release clearance, shuttle engagement, station
occupancy and shuttle loops. Energy consumers: transfer-line infrastructure and shuttles.
Code from montrac_des_v8_8_tlpp2.py (move, _occupy, _release_node, start of shuttle_process, loop record).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import simpy

from ..des.config import resolve_capacity
from ..des.resources import Shuttle
from .base import Discipline


class D1Transportation(Discipline):
    ID = "D1"
    NAME = "Shuttle transportation"
    DESIGN_VARIABLES = ("DV1",)
    RESOURCES = ("Shuttle_*",)
    ENERGY_CONSUMERS = ("Transfer line infrastructure", "Shuttle_*")
    KPIS = ("n_shuttles", "completed_shuttle_loops", "cycles_started", "mean_cycle_time_s", "first_cycle_*",
            "repeated_cycle_time_s_mean", "min_cycle_time_s", "max_cycle_time_s", "mean_travel_s_per_cycle",
            "mean_wait_blocking_s_per_cycle", "clearance_events", "mean_clearance_s_per_cycle",
            "startup_to_first_ws6_arrival_s_mean", "active_s.Shuttle_*", "energy_Wh.Shuttle_*",
            "energy_Wh.Transfer line infrastructure", "energy_shuttles_Wh", "energy_montrac_infrastructure_Wh",
            "energy_montrac_system_Wh", "input.travel_s.*", "input.node_capacity.*", "input.shuttle_speed",
            "input.initial_positions")
    INPUTS = (
        ("DV1 n_shuttles", "design vector", "number of active shuttles (simulation.n_shuttles)"),
        ("travel_segments_s, route_legs, topology", "configuration", "segment times, legs, rail graph, node capacities"),
        ("move command (leg, on_departure)", "D6 (CG5.1)", "next cycle step issued by the dispatcher"),
        ("operation completion", "D2-D5 (CG1.4)", "the shuttle may depart from the station"),
    )
    OUTPUTS = (
        ("shuttle arrival at station", "D2-D5 (CG1.1)", "ARRIVE event; operation can start"),
        ("station access authorization", "D1/D2-D5 (CG1.2)", "node Resource request granted"),
        ("station release", "D1 other shuttles (CG1.5)", "node released when the shuttle departs"),
        ("loop completed at the cycle start station (WS6)", "D6 (CG5.7)", "loop record; next cycle starts"),
        ("shuttle activity, waits, station occupancy, loops", "CG6", "for responses and energy"),
    )

    def __init__(self, core):
        super().__init__(core)
        c = core
        topo = c.cfg["topology"]
        self.edges = {tuple(e) for e in topo["directed_edges"]}
        self.node_capacity = {nid: resolve_capacity(s["capacity"], c.n_shuttles) for nid, s in topo["nodes"].items()}
        self.pass_through_blocking = bool(topo.get("pass_through_requires_free_node", True))
        self.nodes = {nid: simpy.Resource(c.env, capacity=cap) for nid, cap in self.node_capacity.items()}
        self.seg_value = {k: float(v["value"]) for k, v in c.cfg["travel_segments_s"].items()}
        self.leg_hops: Dict[str, List[Tuple[str, str, float]]] = {}
        for leg_name, leg in c.cfg["route_legs"].items():
            hops = []
            for seg_key in leg["segments"]:
                seg = c.cfg["travel_segments_s"][seg_key]
                a, b = seg_key.split("->")
                chain = [a] + list(seg.get("via", [])) + [b]
                fr = [0.0] + list(seg.get("via_time_fractions", [])) + [1.0]
                total = self.seg_value[seg_key] * c.shuttle_factor
                for i in range(len(chain) - 1):
                    hops.append((chain[i], chain[i + 1], total * (fr[i + 1] - fr[i])))
            self.leg_hops[leg_name] = hops
        self.clearance_s = float(c.cfg["station_release_clearance_s"])
        self.clearance_scope = c.cfg["station_release_clearance_scope"]
        self.node_intervals: Dict[str, List[List[Any]]] = {n_: [] for n_ in self.nodes}
        self.node_waits: Dict[str, List[float]] = {n_: [] for n_ in self.nodes}
        self.move_intervals: Dict[int, List[Tuple[float, float]]] = {sh.shuttle_id: [] for sh in c.shuttles}
        self.clearance_events = 0
        for leg_name, leg in c.cfg["route_legs"].items():
            step = leg.get("process_step_time_s")
            t_leg = sum(h[2] for h in self.leg_hops[leg_name])
            if step is not None and abs(t_leg - step) > 0.005:
                c.warnings.append(f"Leg {leg_name} = {t_leg:.2f} s ({' + '.join(leg['segments'])}) differs from the "
                                  f"Process 2.0 step time {step:.2f} s.")

    # ------------------------------------------------------------------ track positions
    def _occupy(self, sh: Shuttle, node: str, req: Any) -> None:
        sh.node, sh.node_req = node, req
        if req is not None:
            self.node_intervals[node].append([self.core.env.now, None, sh.shuttle_id])

    def _release_node(self, sh: Shuttle) -> None:
        if sh.node_req is not None:
            self.nodes[sh.node].release(sh.node_req)
            for iv in reversed(self.node_intervals[sh.node]):
                if iv[2] == sh.shuttle_id and iv[1] is None:
                    iv[1] = self.core.env.now
                    break
            sh.node_req = None
            self.core.signal("CG1.5")

    def enter_line(self, sh: Shuttle):
        """Occupy the initial position and, if needed, drive to the cycle start station."""
        c = self.core
        req = self.nodes[sh.node].request()
        yield req
        self._occupy(sh, sh.node, req)
        if sh.node != c.cycle_start_station:
            sh.phase = "INITIAL_POSITIONING"
            yield from self.move(sh, c.cfg["initial_positioning_legs"][sh.node])

    # ------------------------------------------------------------------ movement
    def move(self, sh: Shuttle, leg_name: str, on_departure=None):
        c = self.core
        env = c.env
        hops = self.leg_hops[leg_name]
        final = hops[-1][1]
        sh.phase = f"MOVE_{leg_name}"
        for i, (a, b, dur) in enumerate(hops):
            assert sh.node == a and (a, b) in self.edges, f"Illegal hop {a}->{b} for shuttle {sh.shuttle_id} at {sh.node}"
            req = None
            if b == final or self.pass_through_blocking:
                t0 = env.now
                req = self.nodes[b].request()
                yield req
                paused = 0.0
                if c.failure is not None and c.failure.is_down(f"Shuttle_{sh.shuttle_id}"):
                    # uncertain mode: a failed shuttle stays at `a` until restored. It keeps its FIFO place for `b`
                    # (and the granted reservation), so no shuttle behind it can overtake it - also not in the WS5
                    # buffer; the stop is recorded as a breakdown wait, not as a node wait.
                    t_grant = env.now
                    yield from c.failure.shuttle_gate(sh, leg_name)
                    paused = env.now - t_grant
                c.signal("CG1.2")
                wait = env.now - t0 - paused
                self.node_waits[b].append(wait)
                if wait > 0:
                    c.log(sh, "WAIT_FOR_NODE", t0, t0 + wait if paused else env.now, node=a, to_node=b, leg=leg_name,
                          category="waiting_blocking", note=f"next position {b} occupied")
                    c._acc(sh, "wait_node_s", wait)
                    if self.clearance_s > 0 and (b == final or self.clearance_scope == "all_positions"):
                        # Station-release clearance after the occupant left. The reservation of b is already
                        # held, so the clearance is charged once and no other shuttle can take b meanwhile.
                        t1 = env.now
                        yield env.timeout(self.clearance_s)
                        c.log(sh, "STATION_RELEASE_CLEARANCE", t1, env.now, node=a, to_node=b, leg=leg_name,
                              category="clearance", note=f"{b} was released by another shuttle")
                        c._acc(sh, "clearance_s", self.clearance_s)
                        self.clearance_events += 1
            elif c.failure is not None:     # uncertain mode, hop without reservation: a failed shuttle stays at `a`
                yield from c.failure.shuttle_gate(sh, leg_name)
            self._release_node(sh)
            self._occupy(sh, b, req)
            if i == 0 and on_departure is not None:
                on_departure()
            t0 = env.now
            self.move_intervals[sh.shuttle_id].append((t0, t0 + dur))
            yield env.timeout(dur)
            c.log(sh, "TRAVEL", t0, env.now, node=b, from_node=a, to_node=b, leg=leg_name, category="travel")
            c._acc(sh, "travel_s", dur)
            if b != final:
                c.log(sh, "PASS_THROUGH", env.now, env.now, node=b, leg=leg_name, category="pass_through",
                      note="no production operation")
        c.log(sh, "ARRIVE", env.now, env.now, node=final, leg=leg_name, category="event")
        c.signal("CG1.1")

    # ------------------------------------------------------------------ shuttle loops
    def complete_loop(self, sh: Shuttle) -> None:
        """Loop record when the shuttle is back at the cycle start station (WS6 in DES v10.0)."""
        c = self.core
        end = c.env.now
        rec = {"batch_id": sh.cycle_id, "shuttle_id": sh.shuttle_id, "shuttle_loop_index": sh.loops_completed + 1,
               "cycle_start_s": sh.cycle_start, "cycle_end_s": end, "cycle_time_s": end - sh.cycle_start,
               "cycle_time_from_t0_s": end if sh.loops_completed == 0 else np.nan}
        for k in ("processing_s", "travel_s", "sync_s", "wait_resource_s", "wait_node_s", "clearance_s", "wait_fanuc_s",
                  "wait_ur10_retrieval_s", "scenario_hold_s"):
            rec[k] = sh.acc.get(k, 0.0)
        rec["sum_components_s"] = (rec["processing_s"] + rec["travel_s"] + rec["sync_s"] + rec["wait_resource_s"]
                                   + rec["wait_node_s"] + rec["clearance_s"] + rec["wait_fanuc_s"] + rec["scenario_hold_s"])
        if c.failure is not None:   # uncertain mode only (keeps the deterministic loop table unchanged)
            rec["breakdown_wait_s"] = sh.acc.get("breakdown_wait_s", 0.0)
            rec["sum_components_s"] += rec["breakdown_wait_s"]
        c.loop_records.append(rec)
        sh.loops_completed += 1
        c.c["loops_completed"] += 1
        c.log(sh, "CYCLE_END", sh.cycle_start, end, category="event", note=f"cycle_time={end - sh.cycle_start:.4f}s")

    def extra_outputs(self, res):
        return {"stations": res["stations"], "loops": res["loops"]}
