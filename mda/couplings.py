"""Coupling registry CG1-CG6 and runtime coupling diagnostics.

All couplings are resolved inside the ONE shared SimPy simulation (events, resources, generator calls);
no coupling iterations are performed. Each exchange is recorded by SimulationCore.signal(<id>), which never
yields, so the diagnostics cannot change the simulation. CG6 is the output-aggregation interface.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

_F = ("id", "group", "producer", "consumer", "name", "data_type", "unit", "trigger", "mechanism")
_R = [
    # CG1 - transportation and workstation synchronisation (D1 <-> D2-D5)
    ("CG1.1", "CG1", "D1", "D2, D3, D4, D5", "shuttle_arrival", "event (Shuttle, node)", "-",
     "end of a route leg at a stop station", "D1.move: final hop timeout, ARRIVE log"),
    ("CG1.2", "CG1", "D1", "D1, D2-D5", "station_access_authorization", "SimPy request granted", "-",
     "next track position becomes free", "D1.nodes[node].request() (simpy.Resource, capacity 1 except WS5)"),
    ("CG1.3", "CG1", "D1", "D2, D3, D4, D5", "shuttle_availability", "Shuttle object (trays, batch)", "-",
     "dispatcher hands the stopped shuttle to the station operation", "D6.shuttle_process -> operation handler"),
    ("CG1.4", "CG1", "D2, D3, D4, D5", "D1", "workstation_operation_completion", "generator return", "-",
     "station operation finished, shuttle may depart", "return of the operation handler (yield from)"),
    ("CG1.5", "CG1", "D1", "D1 (other shuttles)", "station_release", "SimPy release", "-",
     "shuttle starts moving to its already reserved next position", "D1._release_node -> nodes[node].release()"),
    # CG2 - robotic material handling and assembly (D2 <-> D3)
    ("CG2.1", "CG2", "D3", "D2", "fanuc_tray_position_available", "SimPy request granted", "-",
     "the cell (conveyor, 1 tray) is free", "D3.fanuc_positions.request()"),
    ("CG2.2", "CG2", "D2", "D3", "fanuc_input_transfer_complete", "Tray (3 Male + 3 Female)", "s (31)",
     "end of the UR10 input transfer", "D2.transfer_to_fanuc_input (yield from in D3.fanuc_branch)"),
    ("CG2.3", "CG2", "D3", "D2", "fanuc_assembly_complete", "Tray (3 products)", "s (196)",
     "end of the FANUC robot + conveyor sequence", "D3.fanuc_branch: release of FANUC and Conveyor"),
    ("CG2.4", "CG2", "D3", "D2", "fanuc_output_tray_ready", "simpy.Event", "-",
     "assembled tray at the UR10 pick position", "job.tray_ready.succeed()"),
    ("CG2.5", "CG2", "D2", "D3", "output_transfer_synchronisation", "simpy.AllOf", "-",
     "shuttle back at WS6 AND tray ready", "env.all_of([job.shuttle_back, job.tray_ready]) in D2.op_ws6_retrieve"),
    ("CG2.6", "CG2", "D2", "D3", "fanuc_cell_released", "SimPy release", "s (18)",
     "end of the UR10 transfer onto the shuttle", "D3.release_cell_after_retrieval -> fanuc_positions.release()"),
    # CG3 - quality inspection and dispatch (D4 <-> D2)
    ("CG3.1", "CG3", "D4", "D2", "inspection_complete", "Trays (9 products)", "s (20)",
     "end of the Operator_WS8 inspection", "D4.op_ws8_qc: release of Operator_WS8"),
    ("CG3.2", "CG3", "D4", "D2", "product_release_authorization", "bool (trays PRODUCTS_INSPECTED)", "-",
     "before the UR10 unloading", "D4.release_authorized(sh), asserted in D2.op_ws8_unload"),
    ("CG3.3", "CG3", "D2", "D4", "unloading_ready", "SimPy request granted (UR10)", "-",
     "UR10 acquired at WS8", "D2.op_ws8_unload: UR10 request"),
    ("CG3.4", "CG3", "D2", "D4", "unloading_complete", "Trays in Storage B", "s (42)",
     "end of the UR10 dispatch", "D2.op_ws8_unload: release of UR10"),
    ("CG3.5", "CG3", "D4", "MDA coordinator", "dispatch_confirmation", "int (finished products), batch record", "products",
     "trays reach Storage B", "D4.confirm_dispatch (counters, batch_records)"),
    # CG4 - workpiece preparation and assembly (D5 -> D3)
    ("CG4.1", "CG4", "D5", "D3", "workpiece_setup_complete", "event", "s (25)",
     "end of the Operator_WS9 setup", "D5.op_ws9: release of Operator_WS9; shuttle then travels WS9 -> WS4 (D1)"),
    ("CG4.2", "CG4", "D5", "D3", "prepared_workpieces", "Tray states (6 Male, 6 Female)", "workpieces",
     "end of the setup", "tray 1 LOADED_6_MALE, tray 2 LOADED_6_FEMALE on the shuttle"),
    ("CG4.3", "CG4", "D5", "D3", "assembly_prerequisites_verified", "assertion", "-",
     "shuttle at WS4, before the UR3e starts", "D3.op_ws4: tray-state assertion"),
    # CG5 - supervisory coordination (D1-D5 <-> D6)
    ("CG5.1", "CG5", "D6", "D1, D2, D3, D4, D5", "operation_command", "cycle step (move / operation)", "-",
     "next step of the production cycle", "D6.shuttle_process over cycle_steps"),
    ("CG5.2", "CG5", "D6", "D2, D3, D4, D5", "task_release_control_delay", "float", "s (0 in baseline)",
     "before every station / FANUC operation", "D6.control_delay (control_delays_s)"),
    ("CG5.3", "CG5", "D6", "D3", "fanuc_branch_start_command", "callback -> simpy.Process", "-",
     "shuttle departs WS6 (on_departure)", "D1.move on_departure -> D3.start_fanuc_branch -> env.process"),
    ("CG5.4", "CG5", "D6", "D2, D3, D4, D5", "resource_acquisition_policy", "ordered SimPy requests", "-",
     "every resource acquisition", "SimulationCore._acquire (RESOURCE_ORDER, FIFO, no preemption)"),
    ("CG5.5", "CG5", "D1, D2, D3, D4, D5", "D6", "status_report", "dict (busy, queue, occupancy, shuttle state)", "-",
     "on request of the coordinator (end of run)", "D6.collect_status (Resource.count / queue)"),
    ("CG5.6", "CG5", "D6", "energy accounting", "energy_management_setting", "str + energy section", "W",
     "aggregation of the run", "D6.energy_settings (standard powered operation)"),
    ("CG5.7", "CG5", "D1", "D6", "cycle_completion", "loop record", "s",
     "shuttle back at the cycle start station (WS6 in DES v10.0)", "D1.complete_loop; dispatcher starts the next cycle"),
    # CG6 - system-level response aggregation (D1-D6 -> MDA coordinator)
    ("CG6.1", "CG6", "D4", "MDA coordinator", "production_counts_and_throughput", "int, float", "products, products/h",
     "end of run", "counters products_finished_storage_b, batches_completed"),
    ("CG6.2", "CG6", "D1", "MDA coordinator", "loops_waits_station_occupancy", "DataFrames", "s, %",
     "end of run", "loop_records, node_intervals, event log"),
    ("CG6.3", "CG6", "D1, D2, D3, D4, D5", "MDA coordinator", "resource_utilization", "DataFrame (activity table)", "s, %",
     "end of run", "res_intervals, move_intervals -> ResponseAggregator.activity_times"),
    ("CG6.4", "CG6", "D1, D2, D3, D6", "MDA coordinator", "energy_by_consumer", "DataFrame (energy table)", "Wh, kWh",
     "end of run", "energy_table(activity, energy section)"),
    ("CG6.5", "CG6", "MDA coordinator", "MDF (future)", "system_responses", "dict", "products/h, kWh, Wh/product",
     "end of run", "throughput, total energy, specific energy"),
    ("CG6.6", "CG6", "D1, D2, D3, D4, D5", "MDA coordinator", "feasibility_checks", "DataFrame (C1-C6)", "-",
     "end of run", "mda/constraints.py"),
]
REGISTRY: List[Dict[str, Any]] = [dict(zip(_F, r)) for r in _R]
REGISTRY_IDS = [r["id"] for r in REGISTRY]
GROUPS = {"CG1": "Transportation and workstation synchronisation (D1 <-> D2-D5)",
          "CG2": "Robotic material handling and assembly (D2 <-> D3)",
          "CG3": "Quality inspection and dispatch (D4 <-> D2)",
          "CG4": "Workpiece preparation and assembly (D5 -> D3)",
          "CG5": "Supervisory coordination (D1-D5 <-> D6)",
          "CG6": "System-level response aggregation (D1-D6 -> MDA coordinator)"}


def registry_df() -> pd.DataFrame:
    return pd.DataFrame(REGISTRY)


def diagnostics(coupling_log: Dict[str, List[Any]]) -> pd.DataFrame:
    """Exchanges per registered coupling: count and first / last simulation time."""
    unknown = sorted(set(coupling_log) - set(REGISTRY_IDS))
    if unknown:
        raise KeyError(f"Unregistered coupling ids signalled: {unknown}")
    rows = []
    for r in REGISTRY:
        n, t0, t1 = coupling_log.get(r["id"], [0, None, None])
        rows.append({"id": r["id"], "group": r["group"], "producer": r["producer"], "consumer": r["consumer"],
                     "name": r["name"], "exchanges": n, "first_s": t0, "last_s": t1})
    return pd.DataFrame(rows)
