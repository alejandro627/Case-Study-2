"""Electrical energy accounting (active and idle power x time per consumer).

E_active = P_active * t_active / 3600, E_idle = P_idle * t_idle / 3600 [Wh]. Parameters only from the JSON
section "energy"."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _number(name: str, value: Any, positive: bool = False) -> float:
    from .config import _number as n
    return n(name, value, positive)


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
