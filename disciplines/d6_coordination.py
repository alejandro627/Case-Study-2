"""D6 - Digital coordination and control (DV9: energy-management policy).

Represents ONLY the coordination already present in the validated DES:
  * production-cycle dispatcher: the cycle_steps sequence (DES v10.0: WS6 -> WS9 -> WS4 -> WS6 -> WS8 -> WS6),
    issuing move commands to D1 and operation commands to D2-D5 (shuttle process of the single-file DES);
  * the FANUC-branch start command on the departure from WS6;
  * task-release control delays (control_delays_s, all 0 s in the baseline);
  * resource acquisition policy: fixed acquisition order, FIFO per resource, no preemption (RESOURCE_ORDER);
  * status collection (equipment, workstation and shuttle state) for the coordinator;
  * energy-management policy: standard powered operation (energy section of the configuration, unchanged).
Energy consumers: Endian, HMI and auxiliary equipment (continuously powered).
No Digital Twin, communication-delay, maintenance or advanced control functionality is modelled.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..des.config import RESOURCE_ORDER
from ..des.resources import Shuttle
from .base import Discipline

# Operation -> (owning discipline, handler)
OPERATION_OWNER = {
    "ws6_ur10_load_two_trays": ("D2", "op_ws6_load"),
    "ws9_load_workpieces": ("D5", "op_ws9"),
    "ws4_ur3e_assembly": ("D3", "op_ws4"),
    "ws6_sync_retrieve_fanuc_tray": ("D2", "op_ws6_retrieve"),
    "ws8_quality_inspection": ("D4", "op_ws8_qc"),
    "ws8_ur10_unload_three_trays": ("D2", "op_ws8_unload"),
}
# Energy-management policies implemented in the DES (DV9). Levels 2-4 act on the robots and the FANUC cell only:
# after an idle period longer than standby_threshold_s the equipment draws idle_power_factor * idle_W (standby) and
# needs wake_up_delay_s (at active power) before its next task. Factor and delay are calibrated per level.
ENERGY_POLICIES = {
    "standard_powered_operation": "idle powers of the energy section apply whenever equipment is not active; "
                                  "Endian, HMI, auxiliary and infrastructure are always on",
    "reduced_idle_power_operation": "DV9 level 2: reduced idle power with wake-up delay (calibrated)",
    "inactivity_triggered_standby_sleep": "DV9 level 3: standby/sleep after inactivity with wake-up delay (calibrated)",
    "state_dependent_adaptive_energy_management": "DV9 level 4: adaptive standby with wake-up delay (calibrated)",
}
# Resource -> energy consumer covered by a DV9 policy (robots and the FANUC cell)
POLICY_CONSUMER = {"UR10": "UR10", "UR3e": "UR3e", "FANUC": "FANUC_cell", "UR3e_WS9": "UR3e_WS9"}


class D6Coordination(Discipline):
    ID = "D6"
    NAME = "Digital coordination and control"
    DESIGN_VARIABLES = ("DV9",)
    RESOURCES = ()
    ENERGY_CONSUMERS = ("Endian", "HMI", "Auxiliary")
    KPIS = ("energy_continuous_equipment_Wh", "energy_Wh.Endian", "energy_Wh.HMI", "energy_Wh.Auxiliary",
            "mean_sync_control_s_per_cycle", "input.control_delay_s.*", "cycle_boundary", "highest_occupancy",
            "most_loaded_resource_*", "theoretical_batch_bound_per_h", "analytic_single_shuttle_loop_s",
            "input.energy.Endian.*", "input.energy.HMI.*", "input.energy.Auxiliary.*")
    INPUTS = (
        ("DV9 energy-management policy", "design vector", "level 1 = standard powered operation (implemented)"),
        ("cycle_steps, control_delays_s, energy section", "configuration", "existing coordination settings"),
        ("equipment / workstation / shuttle status", "D1-D5 (CG5.5)", "status collection"),
        ("loop completed at the cycle start station (WS6)", "D1 (CG5.7)", "start of the next production cycle"),
    )
    OUTPUTS = (
        ("operation and move commands", "D1-D5 (CG5.1)", "next step of the production cycle"),
        ("task-release control delay", "D2-D5 (CG5.2)", "0 s in the baseline"),
        ("FANUC-branch start command", "D3 (CG5.3)", "on the shuttle's departure from WS6"),
        ("resource acquisition policy", "D2-D5 (CG5.4)", "fixed order, FIFO, no preemption"),
        ("energy-management setting", "energy accounting (CG5.6)", "standard powered operation"),
    )

    def __init__(self, core, energy_policy: str = "standard_powered_operation"):
        super().__init__(core)
        if energy_policy not in ENERGY_POLICIES:
            raise ValueError(f"Energy-management policy {energy_policy!r} is not implemented in the DES "
                             f"(implemented: {sorted(ENERGY_POLICIES)})")
        self.energy_policy = energy_policy
        self.em = core.cfg.get("energy_management")
        if (self.em is not None) != (energy_policy != "standard_powered_operation") or (
                self.em is not None and self.em["policy"] != energy_policy):
            raise ValueError(f"energy_management section does not match the policy {energy_policy!r}")
        if self.em is not None:
            core.pre_work = self.wake_up
        self.control_delays = {k: float(v) for k, v in core.cfg["control_delays_s"].items() if k != "note"}
        self.cycle_steps = core.cfg["cycle_steps"]
        self.acquisition_order = RESOURCE_ORDER

    # ------------------------------------------------------------------ coordination commands
    def control_delay(self, who: Any, op: str, node: Optional[str] = None):
        c = self.core
        d = self.control_delays.get(op, 0.0)
        c.signal("CG5.2")
        if d > 0:
            t0 = c.env.now
            yield c.env.timeout(d)
            kw = {"node": node} if node else {}
            c.log(who, "CONTROL_DELAY", t0, c.env.now, operation=op, category="synchronization", **kw)
            c._acc(who, "sync_s", d)

    def _idle_gap(self, r: str) -> float:
        ivs = self.core.res_intervals[r]
        return self.core.env.now - (ivs[-1][1] if ivs else 0.0)

    def wake_up(self, who: Any, names, op: str, node: Optional[str] = None):
        """DV9: equipment idle for longer than the threshold needs a wake-up delay before its task (core.pre_work)."""
        c = self.core
        em = self.em
        asleep = [n for n in names if n in em["covered_resources"] and self._idle_gap(n) > em["standby_threshold_s"]]
        d = float(em["wake_up_delay_s"])
        if not asleep or d <= 0:
            return
        t0 = c.env.now
        if c.failure is None:
            for n in asleep:
                c.res_intervals[n].append((t0, t0 + d, "wake_up", op, who.shuttle_id, who.cycle_id))
            yield c.env.timeout(d)
        else:   # uncertain mode: no wake-up on a failed resource; preempt-resume as for tasks
            yield from c.failure.run_segment(who, asleep, d, "wake_up", op, node)
        kw = {"node": node} if node else {}
        c.log(who, "WAKE_UP", t0, c.env.now, operation=op, resource="+".join(asleep), category="energy_policy",
              note=f"{self.energy_policy}: wake-up after standby", **kw)
        c._acc(who, "sync_s", d)

    def standby_times(self, H: float) -> Dict[str, Any]:
        """{consumer: (standby time, idle power factor)}: idle periods beyond the threshold, within [0, H]."""
        if self.em is None:
            return {}
        c, th, f = self.core, float(self.em["standby_threshold_s"]), float(self.em["idle_power_factor"])
        out = {}
        for r in self.em["covered_resources"]:
            busy = [(a, b) for a, b, *_ in c.res_intervals[r]]
            if c.failure is not None:   # uncertain mode: down time is a breakdown state, not standby
                busy += c.failure.down_intervals(r)
            ivs = sorted((max(0.0, a), min(b, H)) for a, b in busy if min(b, H) > max(0.0, a))
            t, sb = 0.0, 0.0
            for a, b in ivs:
                sb += max(0.0, a - t - th)
                t = max(t, b)
            sb += max(0.0, H - t - th)
            out[POLICY_CONSUMER[r]] = (sb, f)
        return out

    def shuttle_process(self, sh: Shuttle):
        """Production-cycle dispatcher of one shuttle (shuttle process)."""
        c = self.core
        d1 = self.d["D1"]
        yield from d1.enter_line(sh)
        while True:
            c._cycle_counter += 1
            sh.cycle_id, sh.cycle_start, sh.acc, sh.job = c._cycle_counter, c.env.now, {}, None
            c.c["cycles_started"] += 1
            c.log(sh, "CYCLE_START", c.env.now, c.env.now, category="event")
            for step in self.cycle_steps:
                c.signal("CG5.1")
                if step["type"] == "operation":
                    sh.phase = step["name"].upper()
                    owner, handler = OPERATION_OWNER[step["name"]]
                    c.signal("CG1.3")
                    yield from getattr(self.d[owner], handler)(sh, step["name"])
                    c.signal("CG1.4")
                else:
                    cb = (lambda: self.d["D3"].start_fanuc_branch(sh)) if step.get("on_departure") == "start_fanuc_branch" else None
                    yield from d1.move(sh, step["name"], cb)
            d1.complete_loop(sh)
            c.signal("CG5.7")

    # ------------------------------------------------------------------ status and settings
    def collect_status(self) -> Dict[str, Any]:
        """Equipment, workstation and shuttle status at the current simulation time (CG5.5)."""
        c = self.core
        d1, d3 = self.d["D1"], self.d["D3"]
        c.signal("CG5.5")
        return {
            "time_s": c.env.now,
            "equipment": {r: {"busy": res.count, "capacity": res.capacity, "queue": len(res.queue)} for r, res in c.resources.items()},
            "fanuc_cell_positions": {"occupied": d3.fanuc_positions.count, "capacity": d3.fanuc_positions.capacity,
                                     "queue": len(d3.fanuc_positions.queue)},
            "workstations": {n: {"occupied": r.count, "capacity": r.capacity, "queue": len(r.queue)} for n, r in d1.nodes.items()},
            "shuttles": {sh.shuttle_id: {"node": sh.node, "phase": sh.phase, "trays": len(sh.trays), "batch_id": sh.cycle_id}
                         for sh in c.shuttles},
        }

    def energy_settings(self) -> Dict[str, Any]:
        """Energy-management setting passed to the energy accounting (CG5.6)."""
        self.core.signal("CG5.6")
        return {"policy": self.energy_policy, "description": ENERGY_POLICIES[self.energy_policy],
                "energy_section": self.core.cfg["energy"], "energy_management": self.em,
                "standby": self.standby_times(self.core.horizon_s)}

    def extra_outputs(self, res):
        return {"energy_policy": self.energy_policy, "control_delays_s": dict(self.control_delays),
                "acquisition_order": list(self.acquisition_order)}
