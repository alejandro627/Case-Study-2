"""D2 - Robotic material handling (DV2: tray-loading automation, DV7: finished-product dispatch automation).

The UR10 is ONE shared FIFO resource for: loading two empty trays at WS6, transferring tray 3 to the FANUC
conveyor input, transferring the assembled tray from the FANUC output onto the shuttle (synchronisation at
WS6), and dispatching the three trays at WS8 after the D4 release. Energy consumer: UR10.
Code from montrac_des_v8_8_tlpp2.py (op_ws6_load, UR10 part of fanuc_branch, op_ws6_retrieve, op_ws8_unload,
_eff_load_trays, _eff_tray_on_conveyor, _eff_retrieve, physical part of _eff_unload).
"""
from __future__ import annotations

from ..des.config import T_CONVEYOR_INPUT, T_EMPTY, T_PRODUCTS_FANUC, T_STORAGE_B, T_UNLOADING
from ..des.resources import FanucJob, Shuttle
from .base import Discipline


class D2MaterialHandling(Discipline):
    ID = "D2"
    NAME = "Robotic material handling"
    DESIGN_VARIABLES = ("DV2", "DV7")
    RESOURCES = ("UR10", "Operator_WS6", "Operator_WS8_dispatch", "WSE_WS6_*", "WSE_WS8_dispatch_*")
    ENERGY_CONSUMERS = ("UR10", "WSE_WS6_*", "WSE_WS8_dispatch_*")
    KPIS = ("active_s.UR10", "energy_Wh.UR10", "shuttle_wait_for_ur10_at_ws6_s_mean", "fanuc_output_wait_for_ur10_s_mean",
            "mean_wait_resource_s_per_cycle", "counter.ur10_*", "input.reposition_s.UR10",
            "effective_duration_s.ws6_ur10_load_two_trays", "effective_duration_s.fanuc_input_transfer",
            "effective_duration_s.ws6_sync_retrieve_fanuc_tray", "effective_duration_s.ws8_ur10_unload_three_trays",
            "input.robot_speed.UR10", "input.energy.UR10.*")
    INPUTS = (
        ("DV2 tray-loading level, DV7 dispatch level", "design vector", "level 3 = UR10 (implemented)"),
        ("shuttle at WS6 / WS8", "D1 (CG1.1, CG1.3)", "shuttle stopped and handed over"),
        ("FANUC cell position free", "D3 (CG2.1)", "the UR10 may place tray 3 on the conveyor"),
        ("FANUC output tray ready", "D3 (CG2.4)", "job.tray_ready event"),
        ("product-release authorization", "D4 (CG3.2)", "trays inspected"),
        ("control delay, acquisition order", "D6 (CG5.2, CG5.4)", "existing coordination settings"),
    )
    OUTPUTS = (
        ("two empty trays on the shuttle", "D5 via D1", "trays for the workpiece setup"),
        ("FANUC input-transfer completion", "D3 (CG2.2)", "tray 3 (3 Male + 3 Female) on the conveyor"),
        ("output-transfer synchronisation", "D3 (CG2.5)", "shuttle back AND tray ready, UR10 acquired"),
        ("FANUC cell released", "D3 (CG2.6)", "end of the 18 s transfer onto the shuttle"),
        ("unloading readiness / completion", "D4 (CG3.3, CG3.4)", "UR10 at WS8 / trays in Storage B"),
        ("operation completion", "D1 (CG1.4)", "shuttle may depart"),
        ("UR10 activity and waits", "CG6", "utilisation and energy"),
    )

    def __init__(self, core):
        super().__init__(core)
        self.ur10 = core.resources["UR10"]

    # ------------------------------------------------------------------ material effects
    def _eff_load_trays(self, sh: Shuttle) -> str:
        c = self.core
        ids = []
        for slot in (1, 2):
            t = c._new_tray(T_EMPTY, f"SHUTTLE_{sh.shuttle_id}", f"ws6_load_{slot}", sh.cycle_id)
            sh.trays.append(t)
            ids.append(t.tray_id)
        c.c["trays_loaded_ws6"] += 2
        c.c["ur10_load_operations"] += 1
        return f"UR10 loaded empty trays T{ids[0]}, T{ids[1]}; shuttle carries {len(sh.trays)} trays"

    def _eff_tray_on_conveyor(self, job: FanucJob) -> str:
        c = self.core
        t = c._new_tray(T_CONVEYOR_INPUT, f"FANUC_CELL_B{job.batch_id}", "fanuc_input", job.batch_id)
        t.male = t.female = c.n_fanuc
        job.tray = t
        c.c["trays_fanuc_input"] += 1
        c.c["workpieces_preloaded_tray3_male"] += c.n_fanuc
        c.c["workpieces_preloaded_tray3_female"] += c.n_fanuc
        c.c["ur10_fanuc_input_operations"] += 1
        return f"UR10 placed T{t.tray_id} (3M+3F, batch {job.batch_id}) on the conveyor"

    def _eff_retrieve(self, sh: Shuttle) -> str:
        c = self.core
        job = sh.job
        t = job.tray
        if not (t.batch_id == job.batch_id == sh.cycle_id and job.shuttle_id == sh.shuttle_id):
            c.c["batch_match_failures"] += 1
            raise AssertionError(f"Batch mismatch: tray T{t.tray_id} batch {t.batch_id}, shuttle {sh.shuttle_id} batch {sh.cycle_id}")
        t.location, t.state = f"SHUTTLE_{sh.shuttle_id}", T_PRODUCTS_FANUC
        sh.trays.append(t)
        c.c["trays_retrieved_from_fanuc"] += 1
        c.c["ur10_fanuc_output_operations"] += 1
        return f"UR10 placed T{t.tray_id} (batch {t.batch_id}) on shuttle {sh.shuttle_id}; shuttle carries {len(sh.trays)} trays"

    def _eff_unload(self, sh: Shuttle) -> str:
        """Physical dispatch to Storage B; D4 confirms and counts the finished products (CG3.5)."""
        c = self.core
        out = list(sh.trays)
        n = sum(len(t.products) for t in out)
        for t in out:
            t.location, t.state = "STORAGE_B", T_STORAGE_B
        sh.trays.clear()
        c.c["trays_unloaded_ws8"] += len(out)
        c.c["ur10_unload_operations"] += 1
        self.d["D4"].confirm_dispatch(sh, out, n)
        return f"UR10 dispatched {len(out)} trays with {n} finished products to Storage B"

    # ------------------------------------------------------------------ operations
    def op_ws6_load(self, sh: Shuttle, op: str):
        c = self.core
        assert not sh.trays, "WS6: shuttle must arrive empty"
        if sh.t_first_arrival_ws6 is None:
            sh.t_first_arrival_ws6 = c.env.now
        yield from self.d["D6"].control_delay(sh, op)
        acq, use = c.task_resources(op, ["UR10"])            # DV2: UR10 | Operator_WS6 | UR10 + Operator_WS6
        reqs = yield from c._acquire(sh, acq, op)
        if "UR10" in acq:
            yield from c._reposition(sh, "UR10", "S6", op)
        yield from c._work(sh, use, c.sample_duration(op), op, note_fn=lambda: self._eff_load_trays(sh))
        c._release(reqs)

    def transfer_to_fanuc_input(self, job: FanucJob):
        """UR10 places tray 3 on the FANUC conveyor (31 s); called by D3 once the cell position is free."""
        c = self.core
        op_in = "fanuc_input_transfer"
        yield from self.d["D6"].control_delay(job, op_in, node="S6")
        reqs = yield from c._acquire(job, ["UR10"], op_in, node="S6")
        yield from c._reposition(job, "UR10", "S6", op_in)
        job.t_input_start = c.env.now
        yield from c._work(job, ["UR10"], c.sample_duration(op_in), op_in, "fanuc_branch",
                           lambda: self._eff_tray_on_conveyor(job), node="S6")
        job.t_input_end = c.env.now
        c._release(reqs)
        c.signal("CG2.2")

    def op_ws6_retrieve(self, sh: Shuttle, op: str):
        """Synchronization point: shuttle back AND tray ready AND UR10 free."""
        c = self.core
        env = c.env
        job = sh.job
        assert job.batch_id == sh.cycle_id and len(sh.trays) == 2
        t0 = env.now
        job.t_shuttle_back = t0
        job.shuttle_back.succeed()
        yield env.all_of([job.shuttle_back, job.tray_ready])
        job.t_sync_complete = env.now
        c.signal("CG2.5")
        w = env.now - t0
        if w > 0:
            c.log(sh, "WAIT_FOR_FANUC_TRAY", t0, env.now, operation=op, category="waiting_fanuc_sync",
                  note=f"waiting for batch {job.batch_id} tray")
            c._acc(sh, "wait_fanuc_s", w)
        yield from self.d["D6"].control_delay(sh, op)
        reqs = yield from c._acquire(sh, ["UR10"], op)
        job.t_retrieval_start = env.now
        c._acc(sh, "wait_ur10_retrieval_s", env.now - job.t_sync_complete)
        yield from c._reposition(sh, "UR10", "S6", op)
        yield from c._work(sh, ["UR10"], c.sample_duration(op), op, note_fn=lambda: self._eff_retrieve(sh))
        job.t_retrieval_end = env.now
        self.d["D3"].release_cell_after_retrieval(job)
        c._release(reqs)
        back = max(job.t_ready, job.t_shuttle_back)
        if job.t_shuttle_back > job.t_ready:
            c.log(job, "FANUC_TRAY_WAITS_FOR_SHUTTLE", job.t_ready, job.t_shuttle_back, category="fanuc_wait_shuttle")
        if job.t_retrieval_start > back:
            c.log(job, "FANUC_TRAY_WAITS_FOR_UR10", back, job.t_retrieval_start, category="fanuc_wait_ur10")

    def op_ws8_unload(self, sh: Shuttle, op: str):
        c = self.core
        assert self.d["D4"].release_authorized(sh)
        yield from self.d["D6"].control_delay(sh, op)
        acq, use = c.task_resources(op, ["UR10"])            # DV7: UR10 | Operator_WS8_dispatch | both
        reqs = yield from c._acquire(sh, acq, op)
        c.signal("CG3.3")
        if "UR10" in acq:
            yield from c._reposition(sh, "UR10", "S8", op)
        for t in sh.trays:
            t.state = T_UNLOADING
        yield from c._work(sh, use, c.sample_duration(op), op, note_fn=lambda: self._eff_unload(sh))
        c._release(reqs)
        c.signal("CG3.4")
