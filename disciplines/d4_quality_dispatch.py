"""D4 - Quality inspection and dispatch (DV6: QC assistance level, DV8: inspection policy level).

Operator_WS8 (dedicated, capacity 1) inspects the 9 products at WS8 (20 s) and authorises their release;
D2 (UR10) performs the physical unloading; D4 confirms the dispatch and counts finished products and
completed batches at Storage B. No electrical consumer (human operator).
Code from montrac_des_v8_8_tlpp2.py (op_ws8_qc, _eff_inspect, accounting part of _eff_unload).
"""
from __future__ import annotations

from typing import List

from ..des.config import T_INSPECTED
from ..des.resources import Shuttle, Tray
from .base import Discipline


class D4QualityDispatch(Discipline):
    ID = "D4"
    NAME = "Quality inspection and dispatch"
    DESIGN_VARIABLES = ("DV6", "DV8")
    RESOURCES = ("Operator_WS8", "WSE_WS8_inspection_*")
    ENERGY_CONSUMERS = ("WSE_WS8_inspection_*", "WSE_WS8_assistance_*")
    KPIS = ("completed_production_batches", "finished_products_storage_b", "finished_trays_storage_b",
            "throughput_products_per_h", "throughput_batches_per_h", "mean_batch_completion_interval_s",
            "operator.Operator_WS8.*", "active_s.Operator_WS8", "defects_*", "counter.products_inspected",
            "counter.ws8_inspections", "input.defect_probability_per_product", "effective_duration_s.ws8_quality_inspection")
    INPUTS = (
        ("DV6 QC assistance level, DV8 inspection policy level", "design vector", "DV6 = 1, DV8 = 2 (implemented)"),
        ("shuttle at WS8 with 3 trays / 9 products", "D1 (CG1.1, CG1.3)", "assembled products from D3"),
        ("unloading readiness / completion", "D2 (CG3.3, CG3.4)", "UR10 dispatch"),
        ("control delay", "D6 (CG5.2)", "existing coordination setting"),
    )
    OUTPUTS = (
        ("inspection completion", "D2 (CG3.1)", "products inspected"),
        ("product-release authorization", "D2 (CG3.2)", "trays in state PRODUCTS_INSPECTED"),
        ("dispatch confirmation", "CG6 (CG3.5)", "finished products and batches counted at Storage B"),
        ("operation completion", "D1 (CG1.4)", "shuttle may continue to the UR10 dispatch"),
        ("Operator_WS8 activity", "CG6", "utilisation"),
    )

    def __init__(self, core):
        super().__init__(core)
        self.operator = core.resources["Operator_WS8"]

    def _eff_inspect(self, sh: Shuttle) -> str:
        c = self.core
        n = 0
        for t in sh.trays:
            t.state = T_INSPECTED
            n += len(t.products)
        c.c["products_inspected"] += n
        c.c["defects_detected"] += sh.acc.get("_defects", 0)
        c.c["ws8_inspections"] += 1
        return f"{n} products inspected; defects detected={sh.acc.get('_defects', 0)}"

    def op_ws8_qc(self, sh: Shuttle, op: str):
        c = self.core
        assert len(sh.trays) == 3 and sum(len(t.products) for t in sh.trays) == c.products_per_batch
        yield from self.d["D6"].control_delay(sh, op)
        acq, use = c.task_resources(op, ["Operator_WS8"])    # DV8 level 3: longer inspection + inspection equipment
        reqs = yield from c._acquire(sh, acq, op)
        p = float(c.quality["defect_probability_per_product"])
        sh.acc["_defects"] = sum(1 for _ in range(c.products_per_batch) if c.rng.random() < p)
        yield from c._work(sh, use, c.sample_duration(op), op, note_fn=lambda: self._eff_inspect(sh))
        n_def = int(sh.acc.pop("_defects", 0))
        corr = n_def * float(c.quality["correction_time_per_defective_product_s"])
        if corr > 0:
            yield from c._work(sh, acq, corr, op, note_fn=lambda: f"{n_def} corrected")
        c.c["defects_corrected"] += n_def
        c._release(reqs)
        c.signal("CG3.1")

    def release_authorized(self, sh: Shuttle) -> bool:
        """Product-release authorization read by D2 before unloading (CG3.2)."""
        self.core.signal("CG3.2")
        return len(sh.trays) == 3 and all(t.state == T_INSPECTED for t in sh.trays)

    def confirm_dispatch(self, sh: Shuttle, out: List[Tray], n: int) -> None:
        """Finished-product accounting at Storage B, called at the end of the D2 unloading (CG3.5)."""
        c = self.core
        c.c["products_finished_storage_b"] += n
        c.c["batches_completed"] += 1
        sh.batches_completed += 1
        c.batch_records.append({"batch_id": sh.cycle_id, "shuttle_id": sh.shuttle_id, "shuttle_batch_index": sh.batches_completed,
                                "completed_s": c.env.now, "tray_ids": [t.tray_id for t in out],
                                "tray_batch_ids": [t.batch_id for t in out], "products_per_tray": [len(t.products) for t in out],
                                "products_dispatched": n})
        c.signal("CG3.5")

    def extra_outputs(self, res):
        return {"batches": res["batches"]}
