"""D5 - Workpiece setup (DV3: workpiece-setup automation level).

Operator_WS9 (dedicated, capacity 1) loads 6 Male workpieces into tray 1 and 6 Female workpieces into
tray 2 at WS9 (25 s) and releases the shuttle. The prepared trays reach the UR3e at WS4 by the D1 transport
(CG4). DV3 levels 2-3 add UR3e_WS9, a robot at WS9 separate from the UR3e at WS4 (collaborative with
Operator_WS9, or alone); its active power is calibrated. Energy consumer: UR3e_WS9 when present.
Code from montrac_des_v8_8_tlpp2.py (op_ws9, _eff_ws9).
"""
from __future__ import annotations

from ..des.config import T_EMPTY, T_FEMALE, T_MALE
from ..des.resources import Shuttle
from .base import Discipline


class D5WorkpieceSetup(Discipline):
    ID = "D5"
    NAME = "Workpiece setup"
    DESIGN_VARIABLES = ("DV3",)
    RESOURCES = ("Operator_WS9", "UR3e_WS9")
    ENERGY_CONSUMERS = ("UR3e_WS9",)
    KPIS = ("operator.Operator_WS9.*", "active_s.Operator_WS9", "counter.ws9_operations", "counter.workpieces_loaded_ws9_*",
            "effective_duration_s.ws9_load_workpieces")
    INPUTS = (
        ("DV3 workpiece-setup level", "design vector", "level 1 = manual, Operator_WS9 (implemented)"),
        ("shuttle at WS9 with 2 empty trays", "D1 (CG1.1, CG1.3), D2", "trays loaded at WS6"),
        ("control delay", "D6 (CG5.2)", "existing coordination setting"),
    )
    OUTPUTS = (
        ("workpiece-setup completion", "D3 (CG4.1)", "end of the 25 s setup"),
        ("prepared workpieces", "D3 (CG4.2)", "tray 1: 6 Male, tray 2: 6 Female"),
        ("operation completion", "D1 (CG1.4)", "shuttle may depart to WS4"),
        ("Operator_WS9 activity", "CG6", "utilisation"),
    )

    def __init__(self, core):
        super().__init__(core)
        self.operator = core.resources["Operator_WS9"]

    def _eff_ws9(self, sh: Shuttle) -> str:
        c = self.core
        t1, t2 = sh.trays
        t1.male, t1.state = c.n_ur3e, T_MALE
        t2.female, t2.state = c.n_ur3e, T_FEMALE
        c.c["workpieces_loaded_ws9_male"] += c.n_ur3e
        c.c["workpieces_loaded_ws9_female"] += c.n_ur3e
        c.c["ws9_operations"] += 1
        return f"{c.n_ur3e} Male -> T{t1.tray_id}, {c.n_ur3e} Female -> T{t2.tray_id}"

    def op_ws9(self, sh: Shuttle, op: str):
        c = self.core
        assert len(sh.trays) == 2 and all(t.state == T_EMPTY for t in sh.trays)
        yield from self.d["D6"].control_delay(sh, op)
        acq, use = c.task_resources(op, ["Operator_WS9"])    # DV3: Operator_WS9 | UR3e_WS9 + Operator_WS9 | UR3e_WS9
        reqs = yield from c._acquire(sh, acq, op)
        yield from c._work(sh, use, c.sample_duration(op), op, note_fn=lambda: self._eff_ws9(sh))
        c._release(reqs)
        c.signal("CG4.1")
        c.signal("CG4.2")
