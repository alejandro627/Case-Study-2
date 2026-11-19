"""Transfer Line DES v10.0 - standalone discrete-event simulation of the Montrac Transfer Line 3.0 (Case Study 2, SMF).

Single-file version of the DES used by the CS2 MDA / MDO (package cs2_mdo, DES v10.0). The simulation logic is that
of the package, combined module by module without changes (sections below), and the baseline (TL3) configuration is
embedded at the end of the file (BASELINE_CONFIG). Same inputs -> identical event log, KPIs and energy table.

    python transfer_line_des_v10_0.py                       # 1, 2 and 3 shuttles, 8 h, writes the result tables
    python transfer_line_des_v10_0.py --shuttles 2 --hours 4
    python transfer_line_des_v10_0.py --set process_times_s.ws4_ur3e_assembly_s=65 --no-write
    python transfer_line_des_v10_0.py --config my_config.json   # e.g. a *_run_config.json written by cs2_mdo/run_mda.py
    python transfer_line_des_v10_0.py --export-config baseline_TL3.json

    from transfer_line_des_v10_0 import run_des
    res = run_des(n_shuttles=2)            # dict: kpis, events, loops, batches, jobs, resources, stations, activity, energy
    res["kpis"]["throughput_products_per_h"], res["kpis"]["energy_total_kWh"]

Requirements: Python >= 3.9, simpy, pandas, numpy.

PRODUCTION PROCESS (one shuttle cycle = one batch of 9 products; the cycle starts and ends at WS6)
  WS6  UR10 loads two empty blister trays from the warehouse onto the shuttle ............ 34.78 s
  WS6 -> WS9 (via WS7, WS8) ........................................................... 27.68 s
  WS9  operator WS9 sets up the workpieces (6 male -> tray 1, 6 female -> tray 2) ......... 17.81 s
  WS9 -> WS4 (via WS10) ............................................................... 21.71 s
  WS4  UR3e assembles 6 products (3 in tray 1, 3 in tray 2) ............................. 59.76 s
  WS4 -> WS6 (via WS5) ................................................................ 15.73 s
  WS6  UR10 transfers the FANUC tray (3 products) from the conveyor onto the shuttle ..... 21.48 s
  WS6 -> WS8 (via WS7) ................................................................ 17.46 s
  WS8  operator WS8 inspects the products (quality control) ............................ 14.91 s
  WS8  UR10 transfers the three trays from the shuttle to dispatch ...................... 54.22 s
  WS8 -> WS6 return (via WS9, WS10, WS4, WS5) .......................................... 33.86 s
  Parallel FANUC branch, started when the shuttle leaves WS6:
       UR10 places the third tray (3 male + 3 female) on the conveyor ..................... 19.84 s
       FANUC assembly + conveyor movement (3 products) .................................... 80.54 s
       the tray waits at the conveyor output until the shuttle is back at WS6.
  Measured times = means of 20 repetitions. Nominal 1-shuttle cycle 319.40 s (computed, not set).

RESOURCES: shared UR10 (WS6 and WS8, no repositioning time), UR3e (WS4), FANUC cell (robot + conveyor), independent
operators at WS9 and WS8, capacity-1 track positions S4-S10 with station-release clearance; FIFO, fixed acquisition
order, no preemption. Shuttles start at WS6 (shuttle 1) and queue at WS5 (shuttles 2-3).
ENERGY: E = sum over consumers of (P_active * t_active + P_idle * t_idle) / 3600 over the horizon; Endian, HMI,
auxiliary equipment, shuttles and the track infrastructure are always on.

Reference results (8 h):  1 shuttle 101.25 products/h, 5.561 kWh | 2 shuttles 173.25, 7.190 kWh | 3 shuttles 195.75, 7.723 kWh.
Generated from the cs2_mdo package (DES v10.0); the logic is maintained in the package and this file is regenerated from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import Any, Dict, List, Optional
from typing import Any, Dict, List, Optional, Tuple
from typing import Any, Dict, List, Tuple
from typing import Any, Dict, Optional
from typing import Any, Dict, Tuple
from typing import List
import argparse
import copy
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import simpy


# ====================================================================================================================
# ENERGY ACCOUNTING (active / idle power x time per consumer)
#   (from cs2_mdo/des/energy.py)
# ====================================================================================================================
# Electrical energy accounting (active and idle power x time per consumer).
#
# E_active = P_active * t_active / 3600, E_idle = P_idle * t_idle / 3600 [Wh]. Parameters only from the JSON
# section "energy".

def validate_energy_config(e: Dict[str, Any]) -> None:
    infra = e["transfer_line_infrastructure"]
    for name, c in infra["components"].items():
        q = c["quantity"]
        if isinstance(q, bool) or not isinstance(q, int) or q < 0:
            raise ValueError(f"energy.transfer_line_infrastructure.components.{name}.quantity must be an int >= 0")
        _number(f"energy.transfer_line_infrastructure.components.{name}.power_W_each", c["power_W_each"])
    for group in ("shuttle", "robot_systems", "fanuc_cell", "continuous_equipment"):
        items = {"each": e[group]} if group in ("shuttle", "fanuc_cell") else e[group]
        for name, c in items.items():
            for k in ("active_W", "idle_W"):
                _number(f"energy.{group}.{name}.{k}", c[k])
    if set(e["robot_systems"]) != {"UR10", "UR3e"}:
        raise ValueError("energy.robot_systems must define exactly UR10 and UR3e")
    if set(e["continuous_equipment"]) != {"Endian", "HMI", "Auxiliary"}:
        raise ValueError("energy.continuous_equipment must define exactly Endian, HMI and Auxiliary")
    _number("energy.time_balance_tolerance_s", e["time_balance_tolerance_s"], positive=True)
    for name, c in e.get("alternative_equipment", {}).items():     # added only by DV alternatives
        for k in ("active_W", "idle_W"):
            if _number(f"energy.alternative_equipment.{name}.{k}", c[k]) < 0:
                raise ValueError(f"energy.alternative_equipment.{name}.{k} must be >= 0")
        if c["group"] not in ("robot_systems", "workstation_equipment"):
            raise ValueError(f"energy.alternative_equipment.{name}.group must be robot_systems or workstation_equipment")


def infrastructure_power_W(e: Dict[str, Any]) -> float:
    return sum(c["quantity"] * c["power_W_each"] for c in e["transfer_line_infrastructure"]["components"].values())


def energy_table(e: Dict[str, Any], durations: Dict[str, Tuple[float, float]], H: float,
                 standby: Optional[Dict[str, Tuple[float, float]]] = None,
                 modes: Optional[Dict[str, List[Tuple[float, float]]]] = None) -> pd.DataFrame:
    """Energy per consumer from active/idle durations [s] and the JSON energy section.

    durations = {"Shuttle_1": (active_s, idle_s), ..., "UR10": ..., "UR3e": ..., "FANUC_cell": ...}.
    Infrastructure and continuously powered equipment are active for the whole horizon H.
    Used for the DES run and, with the workbook durations, to reproduce ec_cs2.xlsx.
    energy.alternative_equipment (DV alternatives) adds consumers. standby = {consumer: (t_standby_s, factor)} from an
    energy-management policy (DV9): during t_standby the idle power is factor * idle_W.
    modes = {consumer: [(t_mode_s, P_mode_W), ...]}: active time spent in a collaborative mode (DV levels 2) is
    charged at P_mode instead of active_W. Equipment with always_on = true is active for the whole horizon."""
    tol = e["time_balance_tolerance_s"]
    rows: List[Dict[str, Any]] = []

    def add(consumer, group, p_a, p_i, t_a, t_i, basis):
        e_a = p_a * t_a / 3600.0
        e_i = 0.0 if p_i is None else p_i * t_i / 3600.0
        extra = {}
        if modes and consumer in modes:
            t_m = sum(t for t, _ in modes[consumer])
            e_a = p_a * (t_a - t_m) / 3600.0 + sum(p * t for t, p in modes[consumer]) / 3600.0
            extra["collaborative_mode_time_s"] = t_m
            extra["collaborative_mode_power_W"] = "; ".join(f"{p:.2f}" for _, p in modes[consumer])
        if standby and consumer in standby:
            t_sb, f = standby[consumer]
            e_i = p_i * (t_i - t_sb) / 3600.0 + f * p_i * t_sb / 3600.0
            extra = {"standby_time_s": t_sb, "standby_idle_power_factor": f}
        rows.append({**extra, "consumer": consumer, "group": group, "active_power_W": p_a, "idle_power_W": p_i,
                     "active_time_s": t_a, "idle_time_s": t_i, "active_energy_Wh": e_a, "idle_energy_Wh": e_i,
                     "total_energy_Wh": e_a + e_i, "total_energy_kWh": (e_a + e_i) / 1000.0,
                     "time_balance_error_s": t_a + t_i - H, "time_balance_ok": abs(t_a + t_i - H) <= tol,
                     "power_basis": basis})

    infra = e["transfer_line_infrastructure"]
    add("Transfer line infrastructure", "montrac_infrastructure", infrastructure_power_W(e), None, H, 0.0, infra["basis"])
    sh = e["shuttle"]
    for name in sorted((k for k in durations if k.startswith("Shuttle_")), key=lambda k: int(k.split("_")[1])):
        add(name, "shuttles", sh["active_W"], sh["idle_W"], *durations[name], sh["basis"])
    for name, c in e["robot_systems"].items():
        add(name, "robot_systems", c["active_W"], c["idle_W"], *durations[name], c["basis"])
    fc = e["fanuc_cell"]
    add("FANUC_cell", "robot_systems", fc["active_W"], fc["idle_W"], *durations["FANUC_cell"], fc["basis"])
    for name, c in e.get("alternative_equipment", {}).items():
        if c.get("always_on"):
            add(name, c["group"], c["active_W"], None, H, 0.0, c["basis"])
        else:
            add(name, c["group"], c["active_W"], c["idle_W"], *durations[name], c["basis"])
    for name, c in e["continuous_equipment"].items():
        add(name, "continuous_equipment", c["active_W"], c["idle_W"], H, 0.0, c["basis"])
    df = pd.DataFrame(rows)
    total = {"consumer": "Total system", "group": "total",
             **{k: float(df[k].sum()) for k in ("active_energy_Wh", "idle_energy_Wh", "total_energy_Wh", "total_energy_kWh")},
             "time_balance_ok": bool(df["time_balance_ok"].all())}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def energy_summary(df: pd.DataFrame, finished: int, hours: float) -> Dict[str, Any]:
    c = df[df["group"] != "total"]

    def g(*groups):
        return float(c[c["group"].isin(groups)]["total_energy_Wh"].sum())

    total = float(c["total_energy_Wh"].sum())
    return {
        "energy_montrac_infrastructure_Wh": g("montrac_infrastructure"),
        "energy_shuttles_Wh": g("shuttles"),
        "energy_montrac_system_Wh": g("montrac_infrastructure", "shuttles"),
        "energy_robot_systems_Wh": g("robot_systems"),
        "energy_continuous_equipment_Wh": g("continuous_equipment"),
        "energy_workstation_equipment_Wh": g("workstation_equipment"),
        "energy_active_Wh": float(c["active_energy_Wh"].sum()),
        "energy_idle_Wh": float(c["idle_energy_Wh"].sum()),
        "energy_total_Wh": total, "energy_total_kWh": total / 1000.0,
        "energy_time_balance_ok": bool(c["time_balance_ok"].all()),
        "energy_completed_products": finished,
        "energy_throughput_products_per_h": finished / hours,
        "specific_energy_Wh_per_product": total / finished if finished else "undefined (no completed products)",
    }


# ====================================================================================================================
# CONFIGURATION: constants, loading, overrides and validation
#   (from cs2_mdo/des/config.py)
# ====================================================================================================================
# DES configuration: constants, JSON loading, overrides and validation.
#
# The default configuration is cs2_mdo/config/baseline.json (parameters identical to montrac_des_v8_8_config.json).

# standalone: the baseline configuration is embedded (BASELINE_CONFIG, end of file)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = None
# Output root: Montrac Transfer Line 3.0 with measured times (DES v10.0).
RESULTS_DIR = os.path.join(PACKAGE_DIR, "transfer_line_des_v10_0_results")
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
    if path is None:
        return copy.deepcopy(BASELINE_CONFIG)
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


# ====================================================================================================================
# ENTITIES: trays, FANUC jobs, shuttles
#   (from cs2_mdo/des/resources.py)
# ====================================================================================================================
# Shared DES entities: trays, FANUC jobs (one per batch) and shuttles.

@dataclass
class Tray:
    tray_id: int
    state: str
    batch_id: int
    male: int = 0
    female: int = 0
    products: List[int] = field(default_factory=list)
    location: str = ""
    origin: str = ""


@dataclass
class FanucJob:
    """One FANUC branch execution, tied to one shuttle batch."""
    batch_id: int
    shuttle_id: int
    shuttle_back: Any
    tray_ready: Any
    tray: Optional[Tray] = None
    cell_req: Any = None
    out_req: Any = None
    acc: Dict[str, float] = field(default_factory=dict)
    t_shuttle_departs: Optional[float] = None
    t_position_acquired: Optional[float] = None
    t_input_start: Optional[float] = None
    t_input_end: Optional[float] = None
    t_cell_start: Optional[float] = None
    t_cell_end: Optional[float] = None
    t_cell_released: Optional[float] = None
    t_ready: Optional[float] = None
    t_shuttle_back: Optional[float] = None
    t_sync_complete: Optional[float] = None
    t_retrieval_start: Optional[float] = None
    t_retrieval_end: Optional[float] = None
    t_ur3e_start: Optional[float] = None
    t_ur3e_end: Optional[float] = None

    @property
    def cycle_id(self) -> int:
        return self.batch_id


@dataclass
class Shuttle:
    shuttle_id: int
    node: str
    node_req: Any = None
    trays: List[Tray] = field(default_factory=list)
    job: Optional[FanucJob] = None
    cycle_id: Optional[int] = None
    cycle_start: Optional[float] = None
    t_first_arrival_ws6: Optional[float] = None
    phase: str = "IDLE"
    acc: Dict[str, float] = field(default_factory=dict)
    loops_completed: int = 0
    batches_completed: int = 0


# ====================================================================================================================
# SIMULATION CORE: SimPy environment, shared resources, event log, resource helpers
#   (from cs2_mdo/des/simulation.py)
# ====================================================================================================================
# Shared SimPy simulation core of the six-discipline MDA (DES v9.0).
#
# One SimPy environment, the shared capacity-limited resources, the event log, material counters and the
# resource helpers (acquire / work / release) used by every discipline. The production logic itself lives in
# the disciplines (cs2_mdo/disciplines); D6 starts one dispatcher process per shuttle.
#
# The code of the single-file DES is split by discipline without changing its behaviour (the regression tests compare
# event logs and all result tables).
#
# Coupling signals (signal()) only record that an interface exchange happened. They never yield to SimPy,
# so they cannot change the event order.

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
            self.res_intervals[name].append((t0, t0 + d, "reposition", f"{loc}->{station}", who.shuttle_id, who.cycle_id))
            yield self.env.timeout(d)
            self.log(who, "RESOURCE_REPOSITION", t0, self.env.now, node=station, operation=op, resource=name,
                     category="synchronization", note=f"{name} {loc} -> {station}")
            self._acc(who, "sync_s", d)
        self.res_loc[name] = station

    def _work(self, who: Any, names: List[str], duration: float, op: str, category: str = "processing",
              note_fn=None, node: Optional[str] = None):
        if self.pre_work is not None:
            yield from self.pre_work(who, names, op, node)
        t0 = self.env.now
        for n in names:
            self.res_intervals[n].append((t0, t0 + duration, op, op, who.shuttle_id, who.cycle_id))
        yield self.env.timeout(duration)
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


# ====================================================================================================================
# DISCIPLINE BASE CLASS
#   (from cs2_mdo/disciplines/base.py)
# ====================================================================================================================
# Common interface of the six MDA disciplines.
#
# Each discipline owns part of the production logic and of the shared SimPy resources. All disciplines act on
# ONE SimulationCore (one SimPy environment); couplings are resolved by the shared event-driven simulation,
# without fixed-point iterations.

class Discipline:
    ID = ""
    NAME = ""
    DESIGN_VARIABLES: Tuple[str, ...] = ()
    RESOURCES: Tuple[str, ...] = ()            # rows of the activity table owned by this discipline
    ENERGY_CONSUMERS: Tuple[str, ...] = ()     # rows of the energy table (fnmatch patterns)
    KPIS: Tuple[str, ...] = ()                 # system KPIs reported by this discipline (fnmatch patterns)
    # Interface table: (name, counterpart, description)
    INPUTS: Tuple[Tuple[str, str, str], ...] = ()
    OUTPUTS: Tuple[Tuple[str, str, str], ...] = ()

    def __init__(self, core: SimulationCore):
        self.core = core
        self.d: Dict[str, "Discipline"] = {}

    def connect(self, disciplines: Dict[str, "Discipline"]) -> None:
        """Give the discipline access to the other disciplines (coupling interfaces)."""
        self.d = disciplines

    def outputs(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """Discipline-level outputs extracted from the aggregated results."""
        act = res["activity"]
        en = res["energy"]
        en = en[en["group"] != "total"]
        mine_en = en[[any(fnmatch(c, p) for p in self.ENERGY_CONSUMERS) for c in en["consumer"]]]
        mine_act = act[[any(fnmatch(r, p) for p in self.RESOURCES) for r in act["resource"]]]
        return {
            "discipline": self.ID, "name": self.NAME, "design_variables": list(self.DESIGN_VARIABLES),
            "kpis": {k: v for k, v in res["kpis"].items() if any(fnmatch(k, p) for p in self.KPIS)},
            "activity": mine_act.reset_index(drop=True),
            "energy": mine_en.reset_index(drop=True),
            "energy_Wh": float(mine_en["total_energy_Wh"].sum()) if len(mine_en) else 0.0,
            **self.extra_outputs(res),
        }

    def extra_outputs(self, res: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    @staticmethod
    def interface_rows(cls) -> pd.DataFrame:
        rows = [{"discipline": cls.ID, "direction": "input", "name": n, "counterpart": c, "description": d}
                for n, c, d in cls.INPUTS]
        rows += [{"discipline": cls.ID, "direction": "output", "name": n, "counterpart": c, "description": d}
                 for n, c, d in cls.OUTPUTS]
        return pd.DataFrame(rows)


# ====================================================================================================================
# D1 - TRANSPORTATION (Montrac shuttles, track positions, legs)
#   (from cs2_mdo/disciplines/d1_transportation.py)
# ====================================================================================================================
# D1 - Shuttle transportation (design variable DV1: number of active Montrac shuttles).
#
# Shuttle movement and routing over the rail graph, exclusive track positions (one SimPy Resource per node),
# station arrival / departure, blocking waits and the station-release clearance, shuttle engagement, station
# occupancy and shuttle loops. Energy consumers: transfer-line infrastructure and shuttles.
# Code from montrac_des_v8_8_tlpp2.py (move, _occupy, _release_node, start of shuttle_process, loop record).

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
                c.signal("CG1.2")
                wait = env.now - t0
                self.node_waits[b].append(wait)
                if wait > 0:
                    c.log(sh, "WAIT_FOR_NODE", t0, env.now, node=a, to_node=b, leg=leg_name,
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
        c.loop_records.append(rec)
        sh.loops_completed += 1
        c.c["loops_completed"] += 1
        c.log(sh, "CYCLE_END", sh.cycle_start, end, category="event", note=f"cycle_time={end - sh.cycle_start:.4f}s")

    def extra_outputs(self, res):
        return {"stations": res["stations"], "loops": res["loops"]}


# ====================================================================================================================
# D2 - MATERIAL HANDLING (UR10 at WS6 and WS8)
#   (from cs2_mdo/disciplines/d2_material_handling.py)
# ====================================================================================================================
# D2 - Robotic material handling (DV2: tray-loading automation, DV7: finished-product dispatch automation).
#
# The UR10 is ONE shared FIFO resource for: loading two empty trays at WS6, transferring tray 3 to the FANUC
# conveyor input, transferring the assembled tray from the FANUC output onto the shuttle (synchronisation at
# WS6), and dispatching the three trays at WS8 after the D4 release. Energy consumer: UR10.
# Code from montrac_des_v8_8_tlpp2.py (op_ws6_load, UR10 part of fanuc_branch, op_ws6_retrieve, op_ws8_unload,
# _eff_load_trays, _eff_tray_on_conveyor, _eff_retrieve, physical part of _eff_unload).

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


# ====================================================================================================================
# D3 - ASSEMBLY (UR3e at WS4, parallel FANUC branch)
#   (from cs2_mdo/disciplines/d3_assembly.py)
# ====================================================================================================================
# D3 - Assembly process (DV4: WS4 assembly automation, DV5: FANUC assembly automation).
#
# Two assembly branches that run in parallel in simulation time:
#   * UR3e assembly at WS4 (6 products from trays 1 and 2, 180 s), on the shuttle branch;
#   * FANUC branch, started when the shuttle departs WS6: free cell tray position -> UR10 input transfer (D2)
#     -> FANUC robot + conveyor sequence (196 s) -> tray waits at the output until the UR10 retrieves it.
# The cell (conveyor included, one tray) is free only after the 18 s UR10 transfer onto the shuttle.
# Energy consumers: UR3e and the FANUC cell (robot + conveyor = one consumer).
# Code from montrac_des_v8_8_tlpp2.py (op_ws4, _start_fanuc_branch, fanuc_branch, _eff_ur3e, _eff_fanuc).

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


# ====================================================================================================================
# D4 - QUALITY CONTROL AND DISPATCH (operator WS8)
#   (from cs2_mdo/disciplines/d4_quality_dispatch.py)
# ====================================================================================================================
# D4 - Quality inspection and dispatch (DV6: QC assistance level, DV8: inspection policy level).
#
# Operator_WS8 (dedicated, capacity 1) inspects the 9 products at WS8 (20 s) and authorises their release;
# D2 (UR10) performs the physical unloading; D4 confirms the dispatch and counts finished products and
# completed batches at Storage B. No electrical consumer (human operator).
# Code from montrac_des_v8_8_tlpp2.py (op_ws8_qc, _eff_inspect, accounting part of _eff_unload).

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


# ====================================================================================================================
# D5 - WORKPIECE SETUP (operator WS9)
#   (from cs2_mdo/disciplines/d5_workpiece_setup.py)
# ====================================================================================================================
# D5 - Workpiece setup (DV3: workpiece-setup automation level).
#
# Operator_WS9 (dedicated, capacity 1) loads 6 Male workpieces into tray 1 and 6 Female workpieces into
# tray 2 at WS9 (25 s) and releases the shuttle. The prepared trays reach the UR3e at WS4 by the D1 transport
# (CG4). DV3 levels 2-3 add UR3e_WS9, a robot at WS9 separate from the UR3e at WS4 (collaborative with
# Operator_WS9, or alone); its active power is calibrated. Energy consumer: UR3e_WS9 when present.
# Code from montrac_des_v8_8_tlpp2.py (op_ws9, _eff_ws9).

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


# ====================================================================================================================
# D6 - COORDINATION (cycle dispatcher, control delays, energy policy)
#   (from cs2_mdo/disciplines/d6_coordination.py)
# ====================================================================================================================
# D6 - Digital coordination and control (DV9: energy-management policy).
#
# Represents ONLY the coordination already present in the validated DES:
#   * production-cycle dispatcher: the cycle_steps sequence (DES v10.0: WS6 -> WS9 -> WS4 -> WS6 -> WS8 -> WS6),
#     issuing move commands to D1 and operation commands to D2-D5 (shuttle process of the single-file DES);
#   * the FANUC-branch start command on the departure from WS6;
#   * task-release control delays (control_delays_s, all 0 s in the baseline);
#   * resource acquisition policy: fixed acquisition order, FIFO per resource, no preemption (RESOURCE_ORDER);
#   * status collection (equipment, workstation and shuttle state) for the coordinator;
#   * energy-management policy: standard powered operation (energy section of the configuration, unchanged).
# Energy consumers: Endian, HMI and auxiliary equipment (continuously powered).
# No Digital Twin, communication-delay, maintenance or advanced control functionality is modelled.

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
        for n in asleep:
            c.res_intervals[n].append((t0, t0 + d, "wake_up", op, who.shuttle_id, who.cycle_id))
        yield c.env.timeout(d)
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
            ivs = sorted((max(0.0, a), min(b, H)) for a, b, *_ in c.res_intervals[r] if min(b, H) > max(0.0, a))
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


# ====================================================================================================================
# RESULTS: activity times, conservation checks, KPIs, energy table
#   (from cs2_mdo/mda/responses.py)
# ====================================================================================================================
# CG6 - System-level response aggregation (D1-D6 -> MDA coordinator).
#
# Collects production counts, throughput, batches and loops, resource activity and utilisation, waiting
# times, station occupancy, energy by consumer, total and specific energy, and the material-conservation
# checks. The aggregation code (activity_times, conservation, results) is copied verbatim from
# montrac_des_v8_8_tlpp2.py; ResponseAggregator only states which discipline supplies each quantity.

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


# ======================================================================================================================
# MODEL ASSEMBLY AND RUNNER (standalone)
#   One SimulationCore (one SimPy environment) shared by the six disciplines, as in cs2_mdo/mda/coordinator.py (MDA).
# ======================================================================================================================
DISCIPLINE_CLASSES = {"D1": D1Transportation, "D2": D2MaterialHandling, "D3": D3Assembly,
                      "D4": D4QualityDispatch, "D5": D5WorkpieceSetup, "D6": D6Coordination}


class TransferLineDES:
    """The transfer-line DES: six disciplines acting on one shared SimulationCore."""

    def __init__(self, cfg: Dict[str, Any]):
        validate_config(cfg)
        em = cfg.get("energy_management")
        policy = em["policy"] if em else "standard_powered_operation"
        self.core = SimulationCore(cfg)
        self.disciplines = {did: (cls(self.core, policy) if did == "D6" else cls(self.core))
                            for did, cls in DISCIPLINE_CLASSES.items()}
        for d in self.disciplines.values():
            d.connect(self.disciplines)

    def run(self) -> "TransferLineDES":
        self.core.run(self.disciplines["D6"].shuttle_process)
        return self

    def results(self) -> Dict[str, Any]:
        d6 = self.disciplines["D6"]
        d6.collect_status()
        settings = d6.energy_settings()
        res = ResponseAggregator(self.core, self.disciplines).results()
        res["kpis"]["energy_management_policy"] = settings["policy"]
        return res


def build_config(n_shuttles: Optional[int] = None, horizon_h: Optional[float] = None, config: Any = None,
                 overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Configuration of one run: embedded baseline (or a JSON path / dict) + shuttles, horizon and dotted overrides."""
    cfg = load_config(config) if isinstance(config, str) else (copy.deepcopy(config) if config else load_config())
    ov = []
    if n_shuttles is not None:
        ov.append(f"simulation.n_shuttles={int(n_shuttles)}")
    if horizon_h is not None:
        ov.append(f"simulation.horizon_h={float(horizon_h)}")
    cfg = apply_overrides(cfg, ov + list(overrides or []))
    validate_config(cfg)
    return cfg


def run_des(n_shuttles: Optional[int] = 1, horizon_h: Optional[float] = 8.0, config: Any = None,
            overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the DES and return the results (kpis, events, loops, batches, jobs, resources, stations, activity, energy)."""
    return TransferLineDES(build_config(n_shuttles, horizon_h, config, overrides)).run().results()


def write_outputs(res: Dict[str, Any], out_dir: str = RESULTS_DIR, tag: Optional[str] = None) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    k = res["kpis"]
    pre = os.path.join(out_dir, f"tl_des_v10_0_{tag or str(k['n_shuttles']) + 'sh_' + format(float(k['horizon_h']), 'g') + 'h'}")
    paths = [f"{pre}_kpis.csv"]
    pd.DataFrame(list(k.items()), columns=["metric", "value"]).to_csv(paths[0], index=False)
    for name, key in (("event_log", "events"), ("loops", "loops"), ("batches", "batches"), ("fanuc_jobs", "jobs"),
                      ("resources", "resources"), ("stations", "stations"), ("activity", "activity"), ("energy", "energy")):
        paths.append(f"{pre}_{name}.csv")
        res[key].to_csv(paths[-1], index=False)
    paths.append(f"{pre}_warnings.csv")
    pd.DataFrame({"warning": res["warnings"]}).to_csv(paths[-1], index=False)
    return paths


def print_report(res: Dict[str, Any]) -> None:
    k = res["kpis"]
    print(f"\n=== Transfer Line DES v{DES_VERSION}: {k['n_shuttles']} shuttle(s), {float(k['horizon_h']):g} h ===")
    for key in ("throughput_products_per_h", "finished_products_storage_b", "completed_production_batches",
                "repeated_cycle_time_s_mean", "energy_total_kWh", "specific_energy_Wh_per_product"):
        v = k.get(key)
        print(f"  {key:34s} {v:.4f}" if isinstance(v, float) else f"  {key:34s} {v}")
    e = res["energy"]
    print("  energy per consumer (Wh): " + ", ".join(f"{r.consumer} {r.total_energy_Wh:.1f}"
                                                     for r in e[e.group != "total"].itertuples()))
    act = res["activity"]
    if "active_pct" in act:
        top = act.sort_values("active_pct", ascending=False).head(4)
        print("  most used resources: " + ", ".join(f"{r.resource} {r.active_pct:.0f} %" for r in top.itertuples()))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Transfer Line 3.0 (CS2, SMF) - standalone DES v10.0")
    p.add_argument("--shuttles", type=int, nargs="+", default=[1, 2, 3], help="number(s) of active shuttles (1-3)")
    p.add_argument("--hours", type=float, default=8.0, help="simulation horizon in hours")
    p.add_argument("--config", default=None, help="configuration JSON (default: embedded TL3 baseline)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="dotted.key=value",
                   help="configuration override, e.g. process_times_s.ws9_load_workpieces_s=20")
    p.add_argument("--out-dir", default=RESULTS_DIR)
    p.add_argument("--no-write", action="store_true", help="print the results only")
    p.add_argument("--export-config", metavar="PATH", help="write the embedded baseline configuration to PATH and exit")
    a = p.parse_args(argv)
    if a.export_config:
        with open(a.export_config, "w", encoding="utf-8") as fh:
            json.dump(BASELINE_CONFIG, fh, indent=2, ensure_ascii=False)
        print("written", a.export_config)
        return 0
    for n in a.shuttles:
        try:
            res = run_des(n, a.hours, a.config, a.overrides)
        except (ValueError, KeyError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        print_report(res)
        if not a.no_write:
            paths = write_outputs(res, a.out_dir)
            print(f"  {len(paths)} files written to {a.out_dir}")
    return 0


# ======================================================================================================================
# EMBEDDED BASELINE CONFIGURATION - Transfer Line 3.0 (copy of cs2_mdo/config/baseline.json, DES v10.0)
#   simulation.n_shuttles is set per run (--shuttles / run_des); measured times in process_times_s and route_legs.
# ======================================================================================================================
BASELINE_CONFIG: Dict[str, Any] = json.loads(r'''
{
 "meta": {
  "model": "Montrac Transfer Line DES",
  "model_version": "10.0",
  "production_process": "Transfer Line Production Process 2.0 - 01 November 2026",
  "derived_from": "Process 2.0 model of the line. Updated for Transfer Line 3.0: process times, travel segments / legs, cycle start at WS6, initial positions. Energy, resources, topology, capacities and DV alternatives unchanged.",
  "time_unit": "s",
  "power_unit": "W",
  "energy_unit": "Wh",
  "notes": "Central configuration. Every value can be overridden with --set dotted.path=value. null means 'not measured / not available'. ALL TIMES ARE PRELIMINARY: replace them here after re-measurement, or at run time with --travel \"SEG=SECONDS\" / --set dotted.path=value.",
  "process": "Montrac Transfer Line 3.0 - revised production sequence WS6 -> WS9 -> WS4 -> WS6 -> WS8 -> WS6 with experimentally measured processing and transport times (means of 20 repetitions)"
 },
 "simulation": {
  "n_shuttles": 3,
  "allowed_n_shuttles": [
   1,
   2,
   3
  ],
  "horizon_h": 8.0,
  "standard_horizons_h": [
   1,
   2,
   4,
   8
  ],
  "random_seed": 42,
  "duration_model": "deterministic"
 },
 "process_times_s": {
  "ws6_ur10_load_two_trays_s": 34.78,
  "ws9_load_workpieces_s": 17.81,
  "ws4_ur3e_assembly_s": 59.76,
  "ws6_ur10_fanuc_input_s": 19.84,
  "fanuc_conveyor_sequence_s": 80.54,
  "ws6_ur10_fanuc_output_s": 21.48,
  "ws8_quality_inspection_s": 14.91,
  "ws8_ur10_unload_three_trays_s": 54.22
 },
 "process_time_provenance": {
  "ws6_ur10_load_two_trays_s": {
   "step": 2,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/shuttle",
   "note": "UR10 transfers two blister trays warehouse -> shuttle at WS6.",
   "previous_value_DES9_3": 34.78
  },
  "ws9_load_workpieces_s": {
   "step": 4,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/shuttle",
   "note": "Operator_WS9 (dedicated): 6 Male -> tray 1, 6 Female -> tray 2 (12 workpieces).",
   "previous_value_DES9_3": 25.0
  },
  "ws4_ur3e_assembly_s": {
   "step": 6,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/batch",
   "note": "UR3e assembles 6 products; 3 end up in each of trays 1 and 2.",
   "previous_value_DES9_3": 180.0
  },
  "ws6_ur10_fanuc_input_s": {
   "step": 8,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/batch",
   "note": "UR10 transfers the third tray warehouse -> FANUC conveyor. Runs in parallel with the shuttle branch.",
   "previous_value_DES9_3": 31.0
  },
  "fanuc_conveyor_sequence_s": {
   "step": 9,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/batch",
   "note": "AGGREGATE (measured): conveyor transport of tray 3 to the FANUC robot, FANUC assembly of 3 products and conveyor return to the UR10 pick position (80.54 s).",
   "previous_value_DES9_3": 196.0
  },
  "ws6_ur10_fanuc_output_s": {
   "step": 10,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/batch",
   "note": "UR10 picks the completed FANUC tray from the conveyor and places it on the shuttle (step 10 of Process 2.0). The cell is free when it ends.",
   "previous_value_DES9_3": 18.0
  },
  "ws8_quality_inspection_s": {
   "step": 12,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/batch",
   "note": "Operator_WS8 (dedicated) inspects 9 products.",
   "previous_value_DES9_3": 20.0
  },
  "ws8_ur10_unload_three_trays_s": {
   "step": 13,
   "category": "measured_mean_of_20_repetitions",
   "source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions",
   "unit": "s/shuttle",
   "note": "UR10 dispatches the three completed trays from the shuttle at WS8 to the warehouse (54.22 s, measured; the UR10 is modelled as reaching WS8 and WS6 without a separate transport, as in DES v9.3).",
   "previous_value_DES9_3": 42.0
  }
 },
 "fanuc_cell": {
  "concurrent_jobs": 1,
  "tray_positions": 1,
  "note": "The conveyor is part of the FANUC cell, which holds ONE tray . The 80.54 s covers the conveyor transport to the FANUC robot, the assembly and the conveyor return to the UR10 pick position. A batch tray occupies the cell from the start of its 19.84 s UR10 input transfer until the 21.48 s UR10 transfer onto the shuttle has finished. The FANUC robot and the conveyor are ONE energy consumer.",
  "position_released_at": "after_retrieval",
  "output_positions": 1,
  "output_positions_note": "Used only with position_released_at = 'end_of_sequence'. With 'after_retrieval' the tray waits on the conveyor inside the cell."
 },
 "tray3_workpieces": {
  "preloaded_from_warehouse": true,
  "note": "the third tray arrives from the warehouse with 3 Male + 3 Female workpieces already positioned; no setup operation."
 },
 "robot_speed": {
  "setting": {
   "UR10": "normal",
   "UR3e": "normal",
   "FANUC": "normal"
  },
  "duration_factor": {
   "UR10": {
    "low": null,
    "normal": 1.0,
    "high": null
   },
   "UR3e": {
    "low": null,
    "normal": 1.0,
    "high": null
   },
   "FANUC": {
    "low": null,
    "normal": 1.0,
    "high": null
   }
  },
  "note": "Reference settings. A low/high setting with a null factor is rejected as 'not yet calibrated'. FANUC non-normal is rejected while the 196 s remains an aggregate."
 },
 "shuttle_speed": {
  "setting": "normal",
  "travel_time_factor": {
   "low": null,
   "normal": 1.0,
   "high": null
  }
 },
 "control_delays_s": {
  "ws6_ur10_load_two_trays": 0.0,
  "ws9_load_workpieces": 0.0,
  "ws4_ur3e_assembly": 0.0,
  "ws6_sync_retrieve_fanuc_tray": 0.0,
  "ws8_quality_inspection": 0.0,
  "ws8_ur10_unload_three_trays": 0.0,
  "fanuc_input_transfer": 0.0,
  "note": "Communication/task-release delay before each operation. Not separated in the process description -> 0.0."
 },
 "resource_repositioning_s": {
  "UR10_between_WS6_and_WS8": 0.0,
  "note": "Applied when a shared resource starts at a different station from its previous one. Not measured -> 0.0. Operators are station-bound, so only the UR10 can reposition."
 },
 "travel_segments_s": {
  "S6->S8": {
   "value": 17.46,
   "category": "measured_leg",
   "via": [
    "S7"
   ],
   "via_time_fractions": [
    0.706411
   ],
   "source": "WS6 -> WS8 leg: Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions. WS7 pass-through fraction retained from DES v9.3 (blocking timing only)."
  },
  "S8->S9": {
   "value": 10.22,
   "category": "derived_from_measured_legs",
   "via": [],
   "via_time_fractions": [],
   "source": "WS6 -> WS9 (27.68 s, measured) minus WS6 -> WS8 (17.46 s, measured): the shuttle passes WS8."
  },
  "S9->S4": {
   "value": 21.71,
   "category": "measured_leg",
   "via": [
    "S10"
   ],
   "via_time_fractions": [
    0.5
   ],
   "source": "WS9 -> WS4 leg: Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions. WS10 pass-through fraction retained from DES v9.3."
  },
  "S4->S6": {
   "value": 15.73,
   "category": "measured_leg",
   "via": [
    "S5"
   ],
   "via_time_fractions": [
    0.571338
   ],
   "source": "WS4 -> WS6 leg: Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions. WS5 pass-through fraction 0.571338 = 8.97 / (8.97 + 6.73) from the DES v9.3 segments (assumption; affects blocking timing at WS5 only, not the leg time)."
  },
  "S8->S4": {
   "value": 18.13,
   "category": "derived_from_measured_legs",
   "via": [],
   "via_time_fractions": [],
   "source": "WS8 -> WS6 (33.86 s, measured, via WS4 and WS5) minus WS4 -> WS6 (15.73 s, measured)."
  },
  "S5->S6": {
   "value": 6.7429,
   "category": "assumed_initial_positioning_only",
   "via": [],
   "via_time_fractions": [],
   "source": "Only used to bring shuttles 2-3 from the WS5 queue to WS6 at t = 0: (1 - fraction) x 15.73 s. Not measured."
  }
 },
 "travel_segments_note": "every route leg equals its measured time exactly (sum of segments). S8->S9 and S8->S4 are derived from measured legs; the WS7 / WS10 / WS5 pass-through fractions only time the reservation of the intermediate position.",
 "topology": {
  "source": "PP7/PP8 topology, consistent with the PP8 layout figure. Unchanged.",
  "nodes": {
   "S4": {
    "capacity": 1,
    "role": "WS4 - UR3e assembly; pass-through on return",
    "capacity_basis": "single position"
   },
   "S5": {
    "capacity": "n_shuttles",
    "role": "WS5 - pass-through and queueing buffer (start queue of shuttles 2-3)",
    "capacity_basis": "only WS5 queues shuttles"
   },
   "S6": {
    "capacity": 1,
    "role": "WS6 - cycle start / end; tray loading and FANUC tray retrieval",
    "capacity_basis": "single-shuttle capacity"
   },
   "S7": {
    "capacity": 1,
    "role": "WS7 - pass-through",
    "capacity_basis": "single position"
   },
   "S8": {
    "capacity": 1,
    "role": "WS8 - quality inspection and dispatch; pass-through towards spur",
    "capacity_basis": "single-shuttle capacity"
   },
   "S9": {
    "capacity": 1,
    "role": "WS9 - workpiece setup (spur)",
    "capacity_basis": "single human workstation"
   },
   "S10": {
    "capacity": 1,
    "role": "WS10 - pass-through on spur",
    "capacity_basis": "no queueing at WS10"
   }
  },
  "directed_edges": [
   [
    "S6",
    "S7"
   ],
   [
    "S7",
    "S8"
   ],
   [
    "S8",
    "S9"
   ],
   [
    "S9",
    "S10"
   ],
   [
    "S10",
    "S4"
   ],
   [
    "S8",
    "S4"
   ],
   [
    "S4",
    "S5"
   ],
   [
    "S5",
    "S6"
   ]
  ],
  "pass_through_requires_free_node": true,
  "blocking_rule": "No passing at occupied positions ; block signalling; FIFO at merges; every wait chain ends at the WS5 buffer."
 },
 "route_legs": {
  "L1_WS6_to_WS9": {
   "from": "S6",
   "to": "S9",
   "segments": [
    "S6->S8",
    "S8->S9"
   ],
   "process_step_time_s": 27.68,
   "process_step_source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions"
  },
  "L2_WS9_to_WS4": {
   "from": "S9",
   "to": "S4",
   "segments": [
    "S9->S4"
   ],
   "process_step_time_s": 21.71,
   "process_step_source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions"
  },
  "L3_WS4_to_WS6": {
   "from": "S4",
   "to": "S6",
   "segments": [
    "S4->S6"
   ],
   "process_step_time_s": 15.73,
   "process_step_source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions"
  },
  "L4_WS6_to_WS8": {
   "from": "S6",
   "to": "S8",
   "segments": [
    "S6->S8"
   ],
   "process_step_time_s": 17.46,
   "process_step_source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions"
  },
  "L5_WS8_to_WS6_return": {
   "from": "S8",
   "to": "S6",
   "segments": [
    "S8->S4",
    "S4->S6"
   ],
   "process_step_time_s": 33.86,
   "process_step_source": "Montrac Transfer Line 3.0 measurement campaign: sample mean of 20 repetitions (via WS4 and WS5)"
  },
  "Linit_WS5_to_WS6": {
   "from": "S5",
   "to": "S6",
   "segments": [
    "S5->S6"
   ],
   "process_step_source": "initial positioning of shuttles 2-3 only (assumed time, see S5->S6)"
  }
 },
 "operations": {
  "ws6_ur10_load_two_trays": {
   "station": "S6",
   "resources": [
    "UR10"
   ]
  },
  "ws9_load_workpieces": {
   "station": "S9",
   "resources": [
    "Operator_WS9"
   ]
  },
  "ws4_ur3e_assembly": {
   "station": "S4",
   "resources": [
    "UR3e"
   ]
  },
  "ws6_sync_retrieve_fanuc_tray": {
   "station": "S6",
   "resources": [
    "UR10"
   ]
  },
  "ws8_quality_inspection": {
   "station": "S8",
   "resources": [
    "Operator_WS8"
   ]
  },
  "ws8_ur10_unload_three_trays": {
   "station": "S8",
   "resources": [
    "UR10"
   ]
  }
 },
 "cycle_steps": [
  {
   "type": "operation",
   "name": "ws6_ur10_load_two_trays"
  },
  {
   "type": "move",
   "name": "L1_WS6_to_WS9",
   "on_departure": "start_fanuc_branch"
  },
  {
   "type": "operation",
   "name": "ws9_load_workpieces"
  },
  {
   "type": "move",
   "name": "L2_WS9_to_WS4"
  },
  {
   "type": "operation",
   "name": "ws4_ur3e_assembly"
  },
  {
   "type": "move",
   "name": "L3_WS4_to_WS6"
  },
  {
   "type": "operation",
   "name": "ws6_sync_retrieve_fanuc_tray"
  },
  {
   "type": "move",
   "name": "L4_WS6_to_WS8"
  },
  {
   "type": "operation",
   "name": "ws8_quality_inspection"
  },
  {
   "type": "operation",
   "name": "ws8_ur10_unload_three_trays"
  },
  {
   "type": "move",
   "name": "L5_WS8_to_WS6_return"
  }
 ],
 "initial_positions": {
  "1": "S6",
  "2": "S5",
  "3": "S5"
 },
 "initial_positions_note": "DES v10.0: shuttles start at WS6. WS6 holds one shuttle, so shuttle 1 starts at WS6 and shuttles 2-3 start in the WS5 queue directly upstream of WS6 and enter WS6 in order (assumption: the physical start arrangement for 2-3 shuttles is not specified).",
 "initial_positioning_legs": {
  "S5": "Linit_WS5_to_WS6"
 },
 "resources": {
  "UR10": {
   "capacity": 1,
   "type": "robot",
   "note": "ONE shared robot: WS6 tray loading, FANUC input, FANUC output, WS8 dispatch."
  },
  "UR3e": {
   "capacity": 1,
   "type": "cobot",
   "note": "Assembly of 6 products at WS4."
  },
  "FANUC": {
   "capacity": 1,
   "type": "robot",
   "note": "Assembly of 3 products; occupied with the conveyor during the 196 s aggregate."
  },
  "Conveyor": {
   "capacity": 1,
   "type": "conveyor",
   "note": "FANUC input/output transport; occupied during the 196 s aggregate."
  },
  "Operator_WS9": {
   "capacity": 1,
   "type": "human",
   "station": "S9",
   "task": "workpiece setup (ws9_load_workpieces)",
   "note": "Dedicated to WS9: 6 Male -> tray 1, 6 Female -> tray 2 (12 workpieces), then releases the shuttle."
  },
  "Operator_WS8": {
   "capacity": 1,
   "type": "human",
   "station": "S8",
   "task": "quality inspection (ws8_quality_inspection)",
   "note": "Dedicated to WS8: inspects the 9 products and identifies nonconforming ones, then releases them for the UR10 dispatch."
  }
 },
 "energy": {
  "source": "ec_cs2.xlsx (energy consumption data sheet of the line)",
  "note": "Single source of the electrical power parameters. Values exactly as in ec_cs2.xlsx. E_active [Wh] = active_W * t_active[s] / 3600; E_idle [Wh] = idle_W * t_idle[s] / 3600. Active/idle times come from the DES activity table (montrac_des_v8_6_tlpp2.py, activity_times).",
  "time_balance_tolerance_s": 1e-06,
  "transfer_line_infrastructure": {
   "always_on": true,
   "basis": "SmartRouter + electrical turnout: manufacturer-informed proxies; controller: engineering assumption",
   "note": "Fixed load, independent of the number of shuttles, powered for the whole horizon. Not repeated under Auxiliary.",
   "components": {
    "SmartRouter": {
     "quantity": 1,
     "power_W_each": 4.8,
     "basis": "manufacturer-informed proxy"
    },
    "electrical_turnout": {
     "quantity": 1,
     "power_W_each": 8.64,
     "basis": "manufacturer-informed proxy"
    },
    "Montrac_controller_and_electronics": {
     "quantity": 1,
     "power_W_each": 20.0,
     "basis": "engineering assumption"
    }
   },
   "reference_total_W": 33.44
  },
  "shuttle": {
   "active_W": 27.0,
   "idle_W": 5.0,
   "per": "shuttle",
   "basis": "as in ec_cs2.xlsx",
   "activity_definition": "active = moving + operation performed on the shuttle at a station (engaged time); idle = horizon - active"
  },
  "robot_systems": {
   "UR10": {
    "active_W": 350.0,
    "idle_W": 50.0,
    "includes_controller": true,
    "basis": "as in ec_cs2.xlsx"
   },
   "UR3e": {
    "active_W": 150.0,
    "idle_W": 50.0,
    "includes_controller": true,
    "basis": "ec_cs2.xlsx"
   }
  },
  "fanuc_cell": {
   "active_W": 550.0,
   "idle_W": 100.0,
   "covers": "FANUC robot + conveyor (one consumer)",
   "basis": "ec_cs2.xlsx",
   "activity_definition": "active = 196 s sequence; idle = horizon - active"
  },
  "continuous_equipment": {
   "Endian": {
    "active_W": 60.0,
    "idle_W": 10.0,
    "basis": "manufacturer figure (ec_cs2.xlsx)"
   },
   "HMI": {
    "active_W": 80.0,
    "idle_W": 15.0,
    "basis": "manufacturer figure (ec_cs2.xlsx)"
   },
   "Auxiliary": {
    "active_W": 40.0,
    "idle_W": 7.0,
    "basis": "manufacturer figure (ec_cs2.xlsx); other consumers not included elsewhere"
   }
  },
  "continuous_equipment_note": "Always on: active time = horizon, idle time = 0. idle_W is retained for future scenarios and is not applied.",
  "human_operators_note": "Operator_WS9 and Operator_WS8 are not electrical consumers; their activity is reported but no power is assigned."
 },
 "material": {
  "trays_per_shuttle": 3,
  "workpieces_per_tray_capacity": 6,
  "initial_trays_on_shuttle": 0,
  "trays_loaded_at_ws6": 2,
  "tray1_male_workpieces": 6,
  "tray2_female_workpieces": 6,
  "tray3_male_workpieces": 3,
  "tray3_female_workpieces": 3,
  "ur3e_products_per_tray": [
   3,
   3
  ],
  "note": "trays 1 and 2 hold 3 UR3e products each; tray 3 holds 3 FANUC products; all three trays leave at WS8; shuttles start and return empty."
 },
 "quality": {
  "defect_probability_per_product": 0.0,
  "correction_time_per_defective_product_s": 0.0,
  "correction_included_in_inspection_time": true,
  "defect_disposition": "correct_and_continue",
  "note": "Nominal: defect-free."
 },
 "analytical_reference": {
  "cycle_ws6_to_ws6_single_shuttle_s": 319.4,
  "cycle_composition": "34.78 + 27.68 + 17.81 + 21.71 + 59.76 + 15.73 + 21.48 + 17.46 + 14.91 + 54.22 + 33.86 = 319.40",
  "branch_a_shuttle_s": 142.69,
  "branch_a_composition": "27.68 + 17.81 + 21.71 + 59.76 + 15.73",
  "branch_b_fanuc_s": 100.38,
  "branch_b_composition": "19.84 + 80.54",
  "note": "Comparison values, never imposed: the DES computes cycle times from events and resources."
 },
 "station_release_clearance_s": 2.0,
 "station_release_clearance_note": "Rule: a shuttle that had to wait because its next STOP position was occupied waits this extra time after the occupying shuttle has departed (its release) before it starts moving. Charged once per blocked move; never applied when the position was free. Scope 'stop_stations': only the destination of a route leg (WS6, WS9, WS4, WS8 and WS5), never a pass-through position (WS7, WS10, WS4 on the return leg). 'all_positions' applies it at every position.",
 "scenario_hooks": {
  "extra_hold_at_ws4_s_by_batch": {},
  "note": "TEST ONLY. Keeps the shuttle of the given batch at WS4 for extra seconds after UR3e assembly (used to verify delayed returns). Empty = nominal."
 },
 "cycle_steps_note": "DES v10.0: a cycle starts at WS6 with the UR10 loading of two trays and ends when the shuttle is back at WS6 (WS6 -> WS9 -> WS4 -> WS6 -> WS8 -> WS6). The FANUC branch starts on the departure from WS6.",
 "station_release_clearance_scope": "stop_stations",
 "resources_note": "one operator per station, capacity 1 each, independent (no shared lock, pool or synchronisation). A human resource used at another station is rejected by validate_config.",
 "model_assumptions": {
  "route": "WS6 -> WS9 -> WS4 -> WS6 -> WS8 -> WS6; shuttles start at WS6 and start and return empty",
  "process_times": "means of 20 repetitions on Transfer Line 3.0; deterministic in the baseline model",
  "trays": "trays 1 and 2 hold 3 UR3e products each; tray 3 arrives from the warehouse with 3 male + 3 female workpieces and holds 3 FANUC products; all three trays leave at WS8",
  "fanuc_cell": "robot and conveyor form one cell that holds one tray; the cell is free once the UR10 has transferred the tray from the conveyor onto the shuttle",
  "ur10": "shared resource, first come first served, no preemption; the shuttle waits at WS6 for the FANUC tray",
  "clearance": "the station release clearance applies only at stop stations, never at pass-through positions",
  "operators": "one independent operator at WS9 (workpiece setup) and one at WS8 (inspection), capacity 1 each, no electrical consumption",
  "energy": "parameters from ec_cs2.xlsx; FANUC robot and conveyor are one consumer; a shuttle is active while moving or while an operation is performed on it and at idle power while blocked or waiting; infrastructure (Endian, HMI, auxiliary) always on",
  "nominal_single_shuttle_cycle_s": 319.4
 }
}
''')


if __name__ == "__main__":
    sys.exit(main())
