"""D3 - Assembly process (DV4: WS4 assembly automation, DV5: FANUC assembly automation).

Two assembly branches that run in parallel in simulation time:
  * UR3e assembly at WS4 (6 products from trays 1 and 2, 180 s), on the shuttle branch;
  * FANUC branch, started when the shuttle departs WS6: free cell tray position -> UR10 input transfer (D2)
    -> FANUC robot + conveyor sequence (196 s) -> tray waits at the output until the UR10 retrieves it.
The cell (conveyor included, one tray) is free only after the 18 s UR10 transfer onto the shuttle.
Energy consumers: UR3e and the FANUC cell (robot + conveyor = one consumer).
Code from montrac_des_v8_8_tlpp2.py (op_ws4, _start_fanuc_branch, fanuc_branch, _eff_ur3e, _eff_fanuc).
"""
from __future__ import annotations

import simpy

from ..des.config import (T_AT_OUTPUT, T_FANUC_DONE, T_FEMALE, T_IN_FANUC_CELL, T_MALE, T_PRODUCTS_UR3E,
                          T_UR3E_ASSEMBLING)
from ..des.resources import FanucJob, Shuttle
from .base import Discipline


class D3Assembly(Discipline):
    ID = "D3"
    NAME = "Assembly process"
    DESIGN_VARIABLES = ("DV4", "DV5")
    RESOURCES = ("UR3e", "FANUC_cell", "Operator_WS4", "Operator_FANUC", "WSE_WS4_*", "WSE_FANUC_*")
    ENERGY_CONSUMERS = ("UR3e", "FANUC_cell", "WSE_WS4_*", "WSE_FANUC_*")
    KPIS = ("ur3e_branch_s_mean", "fanuc_branch_s_mean", "assembly_overlap_s_mean", "pct_time_both_assemblies_simultaneous",
            "fanuc_*", "active_s.UR3e", "active_s.FANUC_cell", "energy_Wh.UR3e", "energy_Wh.FANUC_cell",
            "shuttle_wait_for_fanuc_at_ws6_s_mean", "mean_wait_fanuc_sync_s_per_cycle", "analytic_branch_*",
            "counter.ur3e_assemblies", "counter.fanuc_assemblies", "counter.products_assembled_*",
            "input.robot_speed.UR3e", "input.robot_speed.FANUC", "input.fanuc_cell.*", "input.energy.UR3e.*",
            "input.energy.FANUC_cell.*")
    INPUTS = (
        ("DV4 WS4 level, DV5 FANUC level", "design vector", "level 3 = UR3e / FANUC cell (implemented)"),
        ("prepared workpieces (6 Male, 6 Female)", "D5 (CG4.1, CG4.2) via D1", "trays 1 and 2 at WS4"),
        ("branch start command", "D6 (CG5.3)", "shuttle departs WS6"),
        ("FANUC input-transfer completion", "D2 (CG2.2)", "tray 3 on the conveyor"),
        ("output-transfer synchronisation, cell release", "D2 (CG2.5, CG2.6)", "tray placed on the shuttle"),
    )
    OUTPUTS = (
        ("assembly prerequisites verified", "D3 (CG4.3)", "trays 1 and 2 loaded before the UR3e starts"),
        ("cell tray position available", "D2 (CG2.1)", "UR10 may load the conveyor"),
        ("FANUC assembly completion", "D2 (CG2.3)", "end of the 196 s sequence"),
        ("FANUC output-tray readiness", "D2 (CG2.4)", "job.tray_ready event"),
        ("6 products (UR3e) on trays 1-2, 3 products (FANUC) on tray 3", "D4 via D1/D2", "assembled products"),
        ("branch durations, overlap, queue, UR3e/FANUC activity", "CG6", "responses and energy"),
    )

    def __init__(self, core):
        super().__init__(core)
        c = core
        self.ur3e, self.fanuc, self.conveyor = (c.resources[r] for r in ("UR3e", "FANUC", "Conveyor"))
        fc = c.cfg["fanuc_cell"]
        # Physical tray positions of the conveyor/FANUC cell (1 = single-tray interpretation)
        self.fanuc_positions = simpy.Resource(c.env, capacity=fc["tray_positions"])
        # Cell release rule; the separate pick position is only used by 'end_of_sequence' (v8.4)
        self.cell_release = fc["position_released_at"]
        self.fanuc_output = simpy.Resource(c.env, capacity=fc["output_positions"])
        self.hold_ws4 = {int(k): float(v) for k, v in c.cfg["scenario_hooks"]["extra_hold_at_ws4_s_by_batch"].items()}
        if self.hold_ws4:
            c.warnings.append(f"TEST scenario hook active: extra WS4 holds {self.hold_ws4}")

    # ------------------------------------------------------------------ material effects
    def _eff_ur3e(self, sh: Shuttle) -> str:
        c = self.core
        t1, t2 = sh.trays
        n = min(t1.male, t2.female)
        t1.male -= n
        t2.female -= n
        for t, k in zip((t1, t2), c.mat["ur3e_products_per_tray"]):
            for _ in range(k):
                c._product_counter += 1
                t.products.append(c._product_counter)
            t.state = T_PRODUCTS_UR3E
        c.c["workpieces_consumed_ur3e"] += 2 * n
        c.c["products_assembled_ur3e"] += n
        c.c["ur3e_assemblies"] += 1
        return f"UR3e: {2 * n} workpieces -> {n} products ({len(t1.products)} in T{t1.tray_id}, {len(t2.products)} in T{t2.tray_id})"

    def _eff_fanuc(self, job: FanucJob) -> str:
        c = self.core
        t = job.tray
        n = min(t.male, t.female)
        t.male -= n
        t.female -= n
        for _ in range(n):
            c._product_counter += 1
            t.products.append(c._product_counter)
        t.state = T_FANUC_DONE
        c.c["workpieces_consumed_fanuc"] += 2 * n
        c.c["products_assembled_fanuc"] += n
        c.c["fanuc_assemblies"] += 1
        return f"FANUC: {2 * n} workpieces -> {n} products in T{t.tray_id}"

    # ------------------------------------------------------------------ branch A: UR3e at WS4
    def op_ws4(self, sh: Shuttle, op: str):
        c = self.core
        env = c.env
        assert [t.state for t in sh.trays] == [T_MALE, T_FEMALE]
        c.signal("CG4.3")
        yield from self.d["D6"].control_delay(sh, op)
        acq, use = c.task_resources(op, ["UR3e"])            # DV4: UR3e | Operator_WS4 | UR3e + Operator_WS4
        reqs = yield from c._acquire(sh, acq, op)
        for t in sh.trays:
            t.state = T_UR3E_ASSEMBLING
        sh.job.t_ur3e_start = env.now
        yield from c._work(sh, use, c.sample_duration(op), op, note_fn=lambda: self._eff_ur3e(sh))
        sh.job.t_ur3e_end = env.now
        c._release(reqs)
        hold = self.hold_ws4.get(sh.cycle_id, 0.0)
        if hold > 0:  # TEST ONLY scenario hook
            t1 = env.now
            yield env.timeout(hold)
            c.log(sh, "SCENARIO_HOLD", t1, env.now, operation=op, category="scenario_hook", note="TEST: delayed return to WS6")
            c._acc(sh, "scenario_hold_s", hold)

    # ------------------------------------------------------------------ branch B: FANUC process
    def start_fanuc_branch(self, sh: Shuttle) -> None:
        """D6 command on the shuttle's departure from WS6 (CG5.3)."""
        c = self.core
        job = FanucJob(batch_id=sh.cycle_id, shuttle_id=sh.shuttle_id, shuttle_back=c.env.event(), tray_ready=c.env.event())
        job.t_shuttle_departs = c.env.now
        c.jobs[job.batch_id] = job
        sh.job = job
        c.signal("CG5.3")
        c.env.process(self.fanuc_branch(job))

    def fanuc_branch(self, job: FanucJob):
        """Runs independently of the shuttle from its departure until the completed tray waits at the output.
        The batch first needs a free cell tray position. With 'end_of_sequence' the position is released when the
        196 s sequence has brought the tray out of the cell to the UR10 pick position (if that position is free);
        with 'after_retrieval' (v8.2/v8.3) it is held until the UR10 has put the tray on the shuttle."""
        c = self.core
        env = c.env
        op_in, op_cell = "fanuc_input_transfer", "fanuc_conveyor_sequence"
        t0 = env.now
        job.cell_req = self.fanuc_positions.request()
        yield job.cell_req
        job.t_position_acquired = env.now
        if env.now > t0:
            c.log(job, "WAIT_FOR_FANUC_CELL", t0, env.now, node="S6", operation=op_in, category="fanuc_wait_cell",
                  note="previous FANUC tray still in the cell")
        c.signal("CG2.1")
        yield from self.d["D2"].transfer_to_fanuc_input(job)

        acq, use = c.task_resources(op_cell, ["Conveyor", "FANUC"])   # DV5: FANUC | Operator_FANUC | both
        cell = yield from c._acquire(job, acq, op_cell)
        job.t_cell_start = env.now
        job.tray.state = T_IN_FANUC_CELL
        yield from c._work(job, use, c.sample_duration(op_cell), op_cell, "fanuc_branch",
                           lambda: self._eff_fanuc(job))
        job.t_cell_end = env.now
        c._release(cell)          # FANUC and conveyor idle
        c.signal("CG2.3")
        if self.cell_release == "end_of_sequence":
            t0 = env.now
            job.out_req = self.fanuc_output.request()
            yield job.out_req
            if env.now > t0:    # pick position still taken: the tray stays in the cell
                c.log(job, "WAIT_FOR_FANUC_OUTPUT", t0, env.now, category="fanuc_wait_output",
                      note="UR10 pick position occupied by the previous tray")
            self.fanuc_positions.release(job.cell_req)   # tray left the cell: next tray can be loaded
            job.t_cell_released = env.now
        job.tray.state = T_AT_OUTPUT
        job.t_ready = env.now
        job.tray_ready.succeed()
        c.signal("CG2.4")

    def release_cell_after_retrieval(self, job: FanucJob) -> None:
        """Called by D2 at the end of the 18 s transfer onto the shuttle (CG2.6)."""
        if self.cell_release == "end_of_sequence":
            self.fanuc_output.release(job.out_req)       # pick position free
        else:
            self.fanuc_positions.release(job.cell_req)   # v8.3 rule: cell position free only now
            job.t_cell_released = self.core.env.now
        self.core.signal("CG2.6")

    def extra_outputs(self, res):
        return {"fanuc_jobs": res["jobs"]}
