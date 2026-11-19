"""Coupling-variable values of one evaluated design (post-processing of an MDA result; no effect on the model).

In this MDA the couplings are exchanged as events inside one shared SimPy simulation, so they are time series, not
single values iterated to convergence (0 coupling iterations). This module reports them in two forms:
  * coupling_summary   : one row per coupling variable (CG1-CG6) - value, unit, producer -> consumer, source;
  * coupling_per_batch : the exchange times of every production batch (when each coupling event happened).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _proc(ev: pd.DataFrame, op: str) -> pd.DataFrame:
    p = ev[(ev.event == "PROCESS") & (ev.operation == op) & ev.batch_id.notna()]
    return p.groupby("batch_id").agg(start=("start_s", "min"), end=("end_s", "max"))


def coupling_per_batch(out: Dict[str, Any]) -> pd.DataFrame:
    res = out["results"]
    ev = res["events"]
    steps = {"ws6_load": "ws6_ur10_load_two_trays", "ws9_setup": "ws9_load_workpieces", "ur3e": "ws4_ur3e_assembly",
             "fanuc_input": "fanuc_input_transfer", "fanuc_cell": "fanuc_conveyor_sequence",
             "retrieval": "ws6_sync_retrieve_fanuc_tray", "inspection": "ws8_quality_inspection",
             "unloading": "ws8_ur10_unload_three_trays"}
    t = pd.concat([_proc(ev, op).add_prefix(f"{k}_") for k, op in steps.items()], axis=1)
    jobs = res["jobs"].set_index("batch_id")[["shuttle_id", "t_shuttle_departs", "t_ready", "t_shuttle_back", "t_sync_complete",
                                             "t_cell_released"]]
    df = jobs.join(t, how="outer")
    batches = res["batches"].set_index("batch_id")[["completed_s", "products_dispatched"]] if len(res["batches"]) else None
    if batches is not None:
        df = df.join(batches, how="left")
    df.index.name = "batch_id"
    df = df.reset_index()
    # coupling delays (consumer start - producer event)
    df["CG4_setup_to_assembly_s"] = df.ur3e_start - df.ws9_setup_end                  # D5 -> D3 (includes WS9 -> WS4 travel)
    df["CG2_fanuc_ready_minus_shuttle_back_s"] = df.t_ready - df.t_shuttle_back       # > 0: shuttle waits for the tray
    df["CG2_sync_to_retrieval_s"] = df.retrieval_start - df.t_sync_complete           # wait for the UR10
    df["CG3_inspection_to_unloading_s"] = df.unloading_start - df.inspection_end      # D4 -> D2
    return df


def coupling_summary(out: Dict[str, Any]) -> pd.DataFrame:
    res = out["results"]
    k = res["kpis"]
    ev = res["events"]
    diag = out["coupling_diagnostics"].set_index("id").exchanges
    st = res["stations"]
    pb = coupling_per_batch(out)
    done = pb[pb.unloading_end.notna()]

    def mean(s):
        s = pd.Series(s).dropna()
        return float(s.mean()) if len(s) else float("nan")

    dur = lambda e: float(ev.loc[ev.event == e, "duration_s"].sum())
    rows = [
        # CG1 - transportation and workstation synchronisation
        ("CG1", "shuttle arrivals at stop stations", diag["CG1.1"], "count", "D1 -> D2-D5", "CG1.1 ARRIVE events"),
        ("CG1", "station-access authorisations", diag["CG1.2"], "count", "D1 -> D1, D2-D5", "CG1.2 node requests granted"),
        ("CG1", "mean wait for station access", float(st.total_entry_wait_s.sum() / max(st.entries.sum(), 1)), "s / access",
         "D1 -> D2-D5", "node Resource queues"),
        ("CG1", "mean blocking wait per shuttle cycle", k["mean_wait_blocking_s_per_cycle"], "s / cycle", "D1 -> D1",
         "WAIT_FOR_NODE events"),
        ("CG1", "mean station-release clearance per cycle", k["mean_clearance_s_per_cycle"], "s / cycle", "D1 -> D1",
         "STATION_RELEASE_CLEARANCE events"),
        ("CG1", "mean operation time handed to D2-D5 per cycle", k["mean_processing_s_per_cycle"], "s / cycle", "D2-D5 -> D1",
         "CG1.3 / CG1.4"),
        # CG2 - robotic material handling and assembly
        ("CG2", "FANUC input transfers completed", diag["CG2.2"], "count", "D2 -> D3", "CG2.2"),
        ("CG2", "mean wait of a batch for the free FANUC cell", k["fanuc_batch_wait_for_cell_s_mean"], "s / batch",
         "D3 -> D2", "fanuc_positions Resource"),
        ("CG2", "mean FANUC branch duration (WS6 departure -> tray ready)", k["fanuc_branch_s_mean"], "s / batch", "D2 + D3",
         "job timestamps"),
        ("CG2", "mean UR3e branch duration (WS6 departure -> shuttle back)", k["ur3e_branch_s_mean"], "s / batch", "D1 + D5 + D3",
         "job timestamps"),
        ("CG2", "mean tray wait at the FANUC output", k["fanuc_tray_wait_at_output_s_mean"], "s / batch", "D3 -> D2",
         "tray_ready -> retrieval start"),
        ("CG2", "mean shuttle wait for the FANUC tray at WS6", k["shuttle_wait_for_fanuc_at_ws6_s_mean"], "s / batch",
         "D3 -> D1/D2", "all_of(shuttle_back, tray_ready)"),
        ("CG2", "mean wait for the UR10 after synchronisation", k["shuttle_wait_for_ur10_at_ws6_s_mean"], "s / batch",
         "D2 -> D3", "UR10 Resource queue"),
        ("CG2", "mean overlap of UR3e and FANUC assembly", k["assembly_overlap_s_mean"], "s / batch", "D3", "job timestamps"),
        # CG3 - quality inspection and dispatch
        ("CG3", "inspections completed (release authorisations)", diag["CG3.1"], "count", "D4 -> D2", "CG3.1 / CG3.2"),
        ("CG3", "mean delay inspection end -> unloading start", mean(done.CG3_inspection_to_unloading_s), "s / batch",
         "D4 -> D2", "event log"),
        ("CG3", "dispatch confirmations (completed batches)", diag["CG3.5"], "count", "D2 -> D4", "CG3.4 / CG3.5"),
        ("CG3", "finished products dispatched to Storage B", k["finished_products_storage_b"], "products", "D4 -> CG6",
         "batch records"),
        # CG4 - workpiece preparation and assembly
        ("CG4", "workpiece setups completed", diag["CG4.1"], "count", "D5 -> D3", "CG4.1"),
        ("CG4", "prepared workpieces per batch (Male + Female)", 2 * 6, "workpieces", "D5 -> D3", "tray states"),
        ("CG4", "mean delay setup end -> UR3e assembly start", mean(pb.CG4_setup_to_assembly_s), "s / batch", "D5 -> D3",
         "event log (includes WS9 -> WS4 travel)"),
        # CG5 - supervisory coordination
        ("CG5", "cycle-step commands issued", diag["CG5.1"], "count", "D6 -> D1-D5", "CG5.1"),
        ("CG5", "resource acquisitions under the D6 policy", diag["CG5.4"], "count", "D6 -> D2-D5", "CG5.4"),
        ("CG5", "total task-release control delay", dur("CONTROL_DELAY"), "s / 8 h", "D6 -> D2-D5", "CONTROL_DELAY events"),
        ("CG5", "total DV9 wake-up delay", dur("WAKE_UP"), "s / 8 h", "D6 -> D2, D3, D5", "WAKE_UP events"),
        ("CG5", "energy-management policy", k.get("energy_management_policy", "standard_powered_operation"), "-", "D6 -> CG6",
         "CG5.6"),
        # CG6 - system-level responses
        ("CG6", "throughput TH", k["throughput_products_per_h"], "products/h", "D4 -> MDA", "CG6.1"),
        ("CG6", "total energy E", k["energy_total_kWh"], "kWh / 8 h", "D1, D2, D3, D6 -> MDA", "CG6.4"),
        ("CG6", "specific energy", k["specific_energy_Wh_per_product"], "Wh/product", "MDA", "CG6.5"),
        ("CG6", "feasible (C1-C6)", bool(out["feasible"]), "-", "D1-D5 -> MDA", "CG6.6"),
    ]
    return pd.DataFrame(rows, columns=["group", "coupling_variable", "value", "unit", "producer_to_consumer", "source"])


def coupling_values(out: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return coupling_summary(out), coupling_per_batch(out)
