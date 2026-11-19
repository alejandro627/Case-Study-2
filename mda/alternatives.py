"""Physical representation of the non-baseline DV alternatives in the DES (structure only).

Each alternative replaces the resources of ONE operation and states how its time and power are obtained
as follows:
  time   "calibrated"  operation time solved so that, with 1 shuttle, the cycle rate equals the throughput coefficient;
         "speed_up"    task-level speed-up of the inspection (DV6: x1.30 guided, x2.00 smart adaptive);
         "fixed"       physical assumption (DV8 level 1: sample 3 of 9 products -> 20 s x 3/9).
  power  "equipment"   manual station equipment (tools, lighting), in use during the task;
         "robot_mode"  collaborative levels 2: the level-3 robot works with an operator at a calibrated power;
         "robot"       DV3 levels 2-3: UR3e_WS9, a robot at WS9 separate from the UR3e at WS4, switched off between
                       shuttles (idle 0 W), active power calibrated;
         "always_on"   DV6: inspection-assistance equipment (display / camera and vision PC), always powered;
         None          no electrical change (DV8 levels 1 and 3: manual inspection, the operator uses 0 W).
Powers are calibrated so that the 8 h energy, with 1 shuttle, equals the energy coefficient
(cs2_mdo/mda/calibration.py). Operators use no electricity. DV9 acts on the robots and the FANUC cell only.
Inspection: DV8 sets the inspection workload (low 6.67 s / moderate 20 s / full, calibrated), DV6 divides it by its
speed-up.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

OPERATORS = {
    "Operator_WS6": {"capacity": 1, "type": "human", "station": "S6",
                     "note": "DV2 levels 1-2: manual / collaborative tray loading at WS6 (no electrical consumption)."},
    "Operator_WS4": {"capacity": 1, "type": "human", "station": "S4",
                     "note": "DV4 levels 1-2: manual / collaborative assembly at WS4 (no electrical consumption)."},
    "Operator_FANUC": {"capacity": 1, "type": "human", "station": "FANUC_CELL",
                       "note": "DV5 levels 1-2: manual / collaborative assembly at the FANUC station (no electrical consumption)."},
    "Operator_WS8_dispatch": {"capacity": 1, "type": "human", "station": "S8",
                              "note": "DV7 levels 1-2: manual / collaborative finished-product dispatch at WS8."},
    "UR3e_WS9": {"capacity": 1, "type": "robot", "station": "S9",
                 "note": "DV3 levels 2-3: robot at WS9, independent of the UR3e at WS4."},
}
UR3E_WS9_IDLE_W = 0.0           # switched off between shuttles (no start-up delay modelled)
ROBOT_CONSUMER = {"UR10": "UR10", "UR3e": "UR3e", "FANUC": "FANUC_cell"}
INSPECTION = "ws8_quality_inspection"

# (DV, level) -> alternative
ALTERNATIVES: Dict[tuple, Dict[str, Any]] = {
    ("DV2", 1): dict(operation="ws6_ur10_load_two_trays", mode="manual", resources=["Operator_WS6"],
                     time="calibrated", power="equipment", equipment="WSE_WS6_manual"),
    ("DV2", 2): dict(operation="ws6_ur10_load_two_trays", mode="collaborative", resources=["UR10", "Operator_WS6"],
                     time="calibrated", power="robot_mode", robot="UR10"),
    ("DV3", 2): dict(operation="ws9_load_workpieces", mode="collaborative", resources=["UR3e_WS9", "Operator_WS9"],
                     time="calibrated", power="robot"),
    ("DV3", 3): dict(operation="ws9_load_workpieces", mode="automated", resources=["UR3e_WS9"],
                     time="calibrated", power="robot"),
    ("DV4", 1): dict(operation="ws4_ur3e_assembly", mode="manual", resources=["Operator_WS4"],
                     time="calibrated", power="equipment", equipment="WSE_WS4_manual"),
    ("DV4", 2): dict(operation="ws4_ur3e_assembly", mode="collaborative", resources=["UR3e", "Operator_WS4"],
                     time="calibrated", power="robot_mode", robot="UR3e"),
    ("DV5", 1): dict(operation="fanuc_conveyor_sequence", mode="manual", resources=["Conveyor", "Operator_FANUC"],
                     time="calibrated", power="equipment", equipment="WSE_FANUC_manual"),
    ("DV5", 2): dict(operation="fanuc_conveyor_sequence", mode="collaborative", resources=["Conveyor", "FANUC", "Operator_FANUC"],
                     time="calibrated", power="robot_mode", robot="FANUC"),
    ("DV6", 2): dict(operation=INSPECTION, mode="guided_assistance", resources=["Operator_WS8"],
                     time="speed_up", speed_up=1.30, power="always_on", equipment="WSE_WS8_assistance_guided"),
    ("DV6", 3): dict(operation=INSPECTION, mode="smart_adaptive_assistance", resources=["Operator_WS8"],
                     time="speed_up", speed_up=2.00, power="always_on", equipment="WSE_WS8_assistance_smart"),
    ("DV7", 1): dict(operation="ws8_ur10_unload_three_trays", mode="manual", resources=["Operator_WS8_dispatch"],
                     time="calibrated", power="equipment", equipment="WSE_WS8_dispatch_manual"),
    ("DV7", 2): dict(operation="ws8_ur10_unload_three_trays", mode="collaborative", resources=["UR10", "Operator_WS8_dispatch"],
                     time="calibrated", power="robot_mode", robot="UR10"),
    ("DV8", 1): dict(operation=INSPECTION, mode="low_inspection", resources=["Operator_WS8"],
                     time="fixed", sampled_products=3, power=None),
    ("DV8", 3): dict(operation=INSPECTION, mode="full_inspection", resources=["Operator_WS8"],
                     time="calibrated", power=None),
}
POLICIES = {("DV9", 2): "reduced_idle_power_operation", ("DV9", 3): "inactivity_triggered_standby_sleep",
            ("DV9", 4): "state_dependent_adaptive_energy_management"}
POLICY_RESOURCES = ("UR10", "UR3e", "FANUC", "UR3e_WS9")       # robots and the FANUC cell
STANDBY_THRESHOLD_S = 0.0       # no threshold data: standby as soon as the equipment is idle (same for levels 2-4)


def alternative_id(dv: str, level: int) -> str:
    return f"{dv}_L{level}"


def fixed_time(cfg: Dict[str, Any], a: Dict[str, Any]) -> float:
    """DV8 level 1: inspection time proportional to the inspected products (sample of the batch)."""
    m = cfg["material"]
    per_batch = m["tray1_male_workpieces"] + m["tray3_male_workpieces"]
    return float(cfg["process_times_s"]["ws8_quality_inspection_s"]) * a["sampled_products"] / per_batch


def apply_design(cfg: Dict[str, Any], z: Dict[str, int], params: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Configuration of design vector z. params[alternative_id] holds the calibrated values
    (time_s and / or active_W for operation alternatives; idle_power_factor and wake_up_delay_s for DV9)."""
    cfg = copy.deepcopy(cfg)
    alt_eq = cfg["energy"].setdefault("alternative_equipment", {})
    alts = cfg.setdefault("alternatives", {})
    insp = {"workload_s": float(cfg["process_times_s"]["ws8_quality_inspection_s"]), "speed_up": 1.0, "equipment": None,
            "dv": []}
    for dv, level in z.items():
        key = (dv, level)
        if key not in ALTERNATIVES:
            continue
        a, p, aid = ALTERNATIVES[key], params.get(alternative_id(dv, level), {}), alternative_id(dv, level)
        basis = f"calibrated to the {dv} level {level} coefficients (cs2_mdo/config/alternatives_calibration.json)"
        for r in a["resources"]:
            if r not in cfg["resources"]:
                cfg["resources"][r] = copy.deepcopy(OPERATORS[r])
        if a["power"] == "equipment":
            alt_eq[a["equipment"]] = {"active_W": float(p["active_W"]), "idle_W": 0.0, "group": "workstation_equipment",
                                      "basis": basis, "note": f"{a['mode']} station equipment in use during {a['operation']}"}
        elif a["power"] == "always_on":
            alt_eq[a["equipment"]] = {"active_W": float(p["active_W"]), "idle_W": 0.0, "always_on": True,
                                      "group": "workstation_equipment", "basis": basis,
                                      "note": f"{a['mode']} equipment at WS8, always powered"}
        elif a["power"] == "robot":
            alt_eq["UR3e_WS9"] = {"active_W": float(p["active_W"]), "idle_W": UR3E_WS9_IDLE_W, "group": "robot_systems",
                                  "basis": basis + "; switched off between shuttles (idle 0 W)"}
        if a["operation"] == INSPECTION:                       # DV8 workload / DV6 speed-up, composed
            insp["dv"].append(aid)
            if a["time"] == "speed_up":
                insp["speed_up"] = float(a["speed_up"])
            elif a["time"] == "fixed":
                insp["workload_s"] = fixed_time(cfg, a)
            else:
                insp["workload_s"] = float(p["time_s"])
            if a["power"] == "equipment":
                insp["equipment"] = a["equipment"]
            continue
        entry = {"dv": dv, "level": level, "mode": a["mode"], "resources": list(a["resources"]),
                 "time_s": float(p["time_s"]), "equipment": a.get("equipment") if a["power"] == "equipment" else None}
        if a["power"] == "robot_mode":
            entry["robot_mode"] = {"resource": a["robot"], "consumer": ROBOT_CONSUMER[a["robot"]],
                                   "active_W": float(p["active_W"]), "basis": basis}
        alts[a["operation"]] = entry
        if a["operation"] in cfg["operations"]:
            cfg["operations"][a["operation"]]["resources"] = list(a["resources"])
    if insp["dv"]:
        alts[INSPECTION] = {"dv": "+".join(insp["dv"]), "mode": "inspection", "resources": ["Operator_WS8"],
                            "time_s": insp["workload_s"] / insp["speed_up"], "equipment": insp["equipment"],
                            "workload_s": insp["workload_s"], "speed_up": insp["speed_up"]}
    if not alts:
        del cfg["alternatives"]
    if not alt_eq:
        del cfg["energy"]["alternative_equipment"]
    for dv, level in z.items():
        if (dv, level) in POLICIES:
            p = params[alternative_id(dv, level)]
            cfg["energy_management"] = {
                "policy": POLICIES[(dv, level)], "idle_power_factor": float(p["idle_power_factor"]),
                "wake_up_delay_s": float(p["wake_up_delay_s"]), "standby_threshold_s": STANDBY_THRESHOLD_S,
                "covered_resources": [r for r in POLICY_RESOURCES if r in cfg["resources"]]}
    return cfg
