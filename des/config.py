"""DES configuration: constants, JSON loading, overrides and validation.

The default configuration is cs2_mdo/config/baseline.json."""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional

from .energy import validate_energy_config

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_DIR, "config", "baseline.json")
# Output root: Montrac Transfer Line 3.0 with measured times (DES v10.0).
RESULTS_DIR = os.path.join(PACKAGE_DIR, "results")
PRODUCTION_PROCESS = "Transfer Line Production Process 2.0 - 01 November 2026"

REQUIRED_PROCESS_TIME_KEYS = (
    "ws6_ur10_load_two_trays_s",
    "ws9_load_workpieces_s",
    "ws4_ur3e_assembly_s",
    "ws6_ur10_fanuc_input_s",
    "fanuc_conveyor_sequence_s",
    "ws6_ur10_fanuc_output_s",
    "ws8_quality_inspection_s",
    "ws8_ur10_unload_three_trays_s",
)
SHUTTLE_OPERATIONS = (
    "ws6_ur10_load_two_trays",
    "ws9_load_workpieces",
    "ws4_ur3e_assembly",
    "ws6_sync_retrieve_fanuc_tray",
    "ws8_quality_inspection",
    "ws8_ur10_unload_three_trays",
)
ROBOTS = ("UR10", "UR3e", "FANUC")
SPEED_LEVELS = ("low", "normal", "high")
# Fixed acquisition order (baseline resources first in their original order, alternative resources inserted)
RESOURCE_ORDER = ("UR10", "Conveyor", "FANUC", "UR3e", "UR3e_WS9", "Operator_WS6", "Operator_WS9", "Operator_WS4",
                  "Operator_FANUC", "Operator_WS8", "Operator_WS8_dispatch")
CLEARANCE_SCOPES = ("stop_stations", "all_positions")
CELL_RELEASE_MODES = ("end_of_sequence", "after_retrieval")
REPOSITION_KEYS = {"UR10": "UR10_between_WS6_and_WS8"}  # operators are station-bound: no repositioning
HUMAN_OPERATORS = {"Operator_WS9": "S9", "Operator_WS8": "S8"}  # dedicated operator -> its only station
# Operators added only by manual / collaborative DV alternatives (cs2_mdo/mda/alternatives.py)
ALTERNATIVE_OPERATORS = {"Operator_WS6": "S6", "Operator_WS4": "S4", "Operator_FANUC": "FANUC_CELL",
                         "Operator_WS8_dispatch": "S8"}
# Operations whose resources / duration / station equipment a DV alternative may replace
ALTERNATIVE_OPERATIONS = ("ws6_ur10_load_two_trays", "ws9_load_workpieces", "ws4_ur3e_assembly", "fanuc_conveyor_sequence",
                          "ws8_quality_inspection", "ws8_ur10_unload_three_trays")

# Tray states
T_EMPTY = "EMPTY_LOADED_AT_WS6"
T_MALE = "LOADED_6_MALE"
T_FEMALE = "LOADED_6_FEMALE"
T_UR3E_ASSEMBLING = "UR3E_ASSEMBLY_IN_PROGRESS"
T_PRODUCTS_UR3E = "3_PRODUCTS_FROM_UR3E"
T_CONVEYOR_INPUT = "3M3F_AT_CONVEYOR_INPUT"
T_IN_FANUC_CELL = "IN_FANUC_CONVEYOR_SEQUENCE"
T_FANUC_DONE = "3_PRODUCTS_FROM_FANUC_IN_CELL"
T_AT_OUTPUT = "3_PRODUCTS_AT_FANUC_OUTPUT"
T_PRODUCTS_FANUC = "3_PRODUCTS_FROM_FANUC_ON_SHUTTLE"
T_INSPECTED = "PRODUCTS_INSPECTED"
T_UNLOADING = "DISPATCHING_TO_STORAGE_B"
T_STORAGE_B = "TRANSFERRED_TO_STORAGE_B"


class SpeedNotCalibrated(ValueError):
    """A low/high speed setting was selected but no duration factor is configured."""


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_override(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set cfg[a][b][c] = value for 'a.b.c'. The path must exist (typos fail)."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"Override path '{dotted}': '{p}' not found in configuration")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Override path '{dotted}': '{parts[-1]}' not found in configuration")
    node[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must have the form dotted.path=value")
        key, raw = item.split("=", 1)
        apply_override(cfg, key.strip(), _parse_value(raw.strip()))
    return cfg


def _number(name: str, value: Any, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'> 0' if positive else '>= 0'}; got {value!r}")
    return float(value)


def resolve_capacity(raw: Any, n_shuttles: int) -> int:
    if raw == "n_shuttles":
        return n_shuttles
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"Capacity must be a positive int or 'n_shuttles'; got {raw!r}")
    return raw


def speed_factor(table: Dict[str, Any], setting: str, who: str) -> float:
    if setting not in SPEED_LEVELS:
        raise ValueError(f"{who} speed setting must be one of {SPEED_LEVELS}; got {setting!r}")
    f = table.get(setting)
    if f is None:
        raise SpeedNotCalibrated(f"{who} speed '{setting}' is not yet calibrated (duration factor is null).")
    f = _number(f"{who} {setting} duration factor", f, positive=True)
    if setting == "normal" and f != 1.0:
        raise ValueError(f"{who} 'normal' factor must be 1.0; got {f}")
    return f


def cycle_start_station(cfg: Dict[str, Any]) -> str:
    """Where a cycle starts and ends: the origin of the first move, or the station of the first operation."""
    first = cfg["cycle_steps"][0]
    if first["type"] == "move":
        return cfg["route_legs"][first["name"]]["from"]
    return cfg["operations"][first["name"]]["station"]


def validate_config(cfg: Dict[str, Any]) -> None:
    sim = cfg["simulation"]
    n = sim["n_shuttles"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"simulation.n_shuttles must be a positive int; got {n!r}")
    if sim.get("allowed_n_shuttles") and n not in sim["allowed_n_shuttles"]:
        raise ValueError(f"simulation.n_shuttles={n} not in allowed_n_shuttles={sim['allowed_n_shuttles']}")
    _number("simulation.horizon_h", sim["horizon_h"], positive=True)
    if sim.get("duration_model", "deterministic") != "deterministic":
        raise NotImplementedError("Only duration_model='deterministic' is implemented in the DES")

    pt = cfg["process_times_s"]
    for key in REQUIRED_PROCESS_TIME_KEYS:
        if key not in pt:
            raise KeyError(f"process_times_s.{key} is missing")
        _number(f"process_times_s.{key}", pt[key], positive=True)
    for op, d in cfg["control_delays_s"].items():
        if op != "note":
            _number(f"control_delays_s.{op}", d)
    for k, d in cfg["resource_repositioning_s"].items():
        if k != "note":
            _number(f"resource_repositioning_s.{k}", d)

    fc = cfg["fanuc_cell"]
    if fc["concurrent_jobs"] != 1:
        raise ValueError("fanuc_cell.concurrent_jobs must be 1 (the FANUC is one robot)")
    if isinstance(fc["tray_positions"], bool) or not isinstance(fc["tray_positions"], int) or fc["tray_positions"] < 1:
        raise ValueError("fanuc_cell.tray_positions must be a positive int")
    if fc.get("position_released_at") not in CELL_RELEASE_MODES:
        raise ValueError(f"fanuc_cell.position_released_at must be one of {CELL_RELEASE_MODES}")
    op_ = fc.get("output_positions")
    if isinstance(op_, bool) or not isinstance(op_, int) or op_ < 1:
        raise ValueError("fanuc_cell.output_positions must be a positive int")
    _number("station_release_clearance_s", cfg["station_release_clearance_s"])
    if cfg["station_release_clearance_scope"] not in CLEARANCE_SCOPES:
        raise ValueError(f"station_release_clearance_scope must be one of {CLEARANCE_SCOPES}")
    for b, v in cfg["scenario_hooks"]["extra_hold_at_ws4_s_by_batch"].items():
        int(b)
        _number(f"scenario_hooks.extra_hold_at_ws4_s_by_batch.{b}", v)

    rs = cfg["robot_speed"]
    for r in ROBOTS:
        speed_factor(rs["duration_factor"][r], rs["setting"][r], r)
    if rs["setting"]["FANUC"] != "normal":
        raise SpeedNotCalibrated("FANUC speed other than 'normal' needs the 196 s aggregate to be decomposed first.")
    speed_factor(cfg["shuttle_speed"]["travel_time_factor"], cfg["shuttle_speed"]["setting"], "Shuttle")

    topo = cfg["topology"]
    nodes = topo["nodes"]
    for spec in nodes.values():
        resolve_capacity(spec["capacity"], n)
    edges = {tuple(e) for e in topo["directed_edges"]}
    for a, b in edges:
        if a not in nodes or b not in nodes:
            raise ValueError(f"Edge {a}->{b} refers to an unknown node")
    for seg_key, seg in cfg["travel_segments_s"].items():
        _number(f"travel_segments_s.{seg_key}.value", seg["value"], positive=True)
        a, b = seg_key.split("->")
        chain = [a] + list(seg.get("via", [])) + [b]
        for x, y in zip(chain, chain[1:]):
            if (x, y) not in edges:
                raise ValueError(f"Travel segment {seg_key} uses {x}->{y}, which is not a directed rail edge")
        fr = seg.get("via_time_fractions", [])
        if len(fr) != len(seg.get("via", [])) or any(not 0.0 < f < 1.0 for f in fr) or list(fr) != sorted(fr):
            raise ValueError(f"travel_segments_s.{seg_key}: via_time_fractions must match via and increase in (0,1)")
    for leg_name, leg in cfg["route_legs"].items():
        pos = leg["from"]
        for seg_key in leg["segments"]:
            if seg_key not in cfg["travel_segments_s"]:
                raise KeyError(f"Leg {leg_name} uses undefined travel segment {seg_key}")
            a, b = seg_key.split("->")
            if a != pos:
                raise ValueError(f"Leg {leg_name}: segment {seg_key} does not start at {pos}")
            pos = b
        if pos != leg["to"]:
            raise ValueError(f"Leg {leg_name} ends at {pos}, expected {leg['to']}")

    ops = cfg["operations"]
    if sorted(ops) != sorted(SHUTTLE_OPERATIONS):
        raise ValueError(f"operations must be exactly {SHUTTLE_OPERATIONS}")
    for op, spec in ops.items():
        if spec["station"] not in nodes:
            raise ValueError(f"Operation {op} at unknown station {spec['station']}")
        for r in spec["resources"]:
            if r not in cfg["resources"]:
                raise ValueError(f"Operation {op} uses unknown resource {r}")
    humans = {r for r, spec in cfg["resources"].items() if spec.get("type") == "human"}
    allowed = dict(HUMAN_OPERATORS, **ALTERNATIVE_OPERATORS)
    if not set(HUMAN_OPERATORS) <= humans <= set(allowed):
        raise ValueError(f"human resources must include {sorted(HUMAN_OPERATORS)} and may add {sorted(ALTERNATIVE_OPERATORS)} "
                         "(one dedicated operator per station task)")
    alt_ops = set(cfg.get("alternatives", {}))
    for r in sorted(humans):
        st = allowed[r]
        if cfg["resources"][r]["capacity"] != 1 or cfg["resources"][r].get("station") != st:
            raise ValueError(f"{r} must have capacity 1 and station {st}")
        used = {spec["station"] for o, spec in ops.items() if r in spec["resources"]}
        # station-bound; a base operator may be unused only when an alternative replaces its task
        if not used <= {st} or (r in HUMAN_OPERATORS and used != {st} and not alt_ops):
            raise ValueError(f"{r} must be used only at {st}; used at {sorted(used)}")
    validate_alternatives(cfg)

    steps = cfg["cycle_steps"]
    pos = start = cycle_start_station(cfg)
    for st in steps:
        if st["type"] == "operation":
            if ops[st["name"]]["station"] != pos:
                raise ValueError(f"Operation {st['name']} is at {ops[st['name']]['station']} but the shuttle is at {pos}")
        elif st["type"] == "move":
            leg = cfg["route_legs"][st["name"]]
            if leg["from"] != pos:
                raise ValueError(f"Move {st['name']} starts at {leg['from']} but the shuttle is at {pos}")
            pos = leg["to"]
        else:
            raise ValueError(f"Unknown step type {st['type']}")
    if pos != start:
        raise ValueError(f"cycle_steps end at {pos}, not at the start station {start}")
    if sorted(st["name"] for st in steps if st["type"] == "operation") != sorted(SHUTTLE_OPERATIONS):
        raise ValueError("cycle_steps must contain every operation exactly once")
    if sum(1 for st in steps if st.get("on_departure") == "start_fanuc_branch") != 1:
        raise ValueError("Exactly one move must start the FANUC branch")

    for sid in range(1, n + 1):
        p0 = cfg["initial_positions"].get(str(sid))
        if p0 is None or p0 not in nodes:
            raise ValueError(f"initial_positions['{sid}'] must name a node")
        if p0 != start:
            leg_name = cfg["initial_positioning_legs"].get(p0)
            if leg_name is None:
                raise ValueError(f"No initial positioning leg from {p0} to {start}")
            leg = cfg["route_legs"][leg_name]
            if leg["from"] != p0 or leg["to"] != start:
                raise ValueError(f"Initial positioning leg {leg_name} must go {p0}->{start}")

    if any("quattro" in k.lower() for k in cfg["resources"]):
        raise ValueError("Quattro must not appear in resources (removed from production)")
    validate_energy_config(cfg["energy"])

    m = cfg["material"]
    if m["trays_per_shuttle"] != 3 or m["initial_trays_on_shuttle"] != 0 or m["trays_loaded_at_ws6"] != 2:
        raise ValueError("Process 2.0 requires trays_per_shuttle=3, initial_trays_on_shuttle=0, trays_loaded_at_ws6=2")
    cap = m["workpieces_per_tray_capacity"]
    if m["tray1_male_workpieces"] != m["tray2_female_workpieces"] or m["tray1_male_workpieces"] > cap:
        raise ValueError("material: tray 1 Male count must equal tray 2 Female count, within tray capacity")
    if m["tray3_male_workpieces"] != m["tray3_female_workpieces"] or m["tray3_male_workpieces"] * 2 > cap:
        raise ValueError("material: tray 3 needs equal Male/Female counts within tray capacity")
    if sum(m["ur3e_products_per_tray"]) != m["tray1_male_workpieces"] or len(m["ur3e_products_per_tray"]) != 2:
        raise ValueError("material.ur3e_products_per_tray must split the UR3e products over trays 1 and 2")

    q = cfg["quality"]
    p = q["defect_probability_per_product"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
        raise ValueError("quality.defect_probability_per_product must be in [0, 1]")
    _number("quality.correction_time_per_defective_product_s", q["correction_time_per_defective_product_s"])
    if q["defect_disposition"] != "correct_and_continue":
        raise ValueError("Only defect_disposition='correct_and_continue' is supported")


def validate_alternatives(cfg: Dict[str, Any]) -> None:
    """Optional sections written by the MDA coordinator for non-baseline DV levels (absent in the baseline)."""
    for op, a in cfg.get("alternatives", {}).items():
        if op not in ALTERNATIVE_OPERATIONS:
            raise ValueError(f"alternatives: operation {op} cannot be replaced (allowed: {ALTERNATIVE_OPERATIONS})")
        if not a["resources"]:
            raise ValueError(f"alternatives.{op}: at least one resource is required")
        for r in a["resources"]:
            if r not in cfg["resources"]:
                raise ValueError(f"alternatives.{op} uses unknown resource {r}")
        _number(f"alternatives.{op}.time_s", a["time_s"], positive=True)
        eq = a.get("equipment")
        if eq is not None and eq not in cfg["energy"].get("alternative_equipment", {}):
            raise ValueError(f"alternatives.{op}: equipment {eq} has no entry in energy.alternative_equipment")
        rm = a.get("robot_mode")
        if rm is not None:
            if rm["resource"] not in a["resources"]:
                raise ValueError(f"alternatives.{op}: collaborative robot {rm['resource']} does not perform the task")
            if _number(f"alternatives.{op}.robot_mode.active_W", rm["active_W"]) < 0:
                raise ValueError(f"alternatives.{op}.robot_mode.active_W must be >= 0")
    em = cfg.get("energy_management")
    if em is not None:
        f = _number("energy_management.idle_power_factor", em["idle_power_factor"])
        if not 0.0 <= f <= 1.0:
            raise ValueError("energy_management.idle_power_factor must be in [0, 1]")
        if _number("energy_management.wake_up_delay_s", em["wake_up_delay_s"]) < 0:
            raise ValueError("energy_management.wake_up_delay_s must be >= 0")
        if _number("energy_management.standby_threshold_s", em["standby_threshold_s"]) < 0:
            raise ValueError("energy_management.standby_threshold_s must be >= 0")
        for r in em["covered_resources"]:
            if r not in cfg["resources"]:
                raise ValueError(f"energy_management covers unknown resource {r}")
