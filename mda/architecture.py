"""Architecture data: disciplines, resource ownership and the mapping of the single-file DES code to D1-D6."""
from __future__ import annotations

import pandas as pd

from ..disciplines.base import Discipline
from ..disciplines.d1_transportation import D1Transportation
from ..disciplines.d2_material_handling import D2MaterialHandling
from ..disciplines.d3_assembly import D3Assembly
from ..disciplines.d4_quality_dispatch import D4QualityDispatch
from ..disciplines.d5_workpiece_setup import D5WorkpieceSetup
from ..disciplines.d6_coordination import D6Coordination

DISCIPLINE_CLASSES = {"D1": D1Transportation, "D2": D2MaterialHandling, "D3": D3Assembly,
                      "D4": D4QualityDispatch, "D5": D5WorkpieceSetup, "D6": D6Coordination}

# SimPy resources -> owning discipline (all live in the one shared environment)
RESOURCE_OWNER = {
    "nodes S4-S10 (track positions)": "D1", "UR10": "D2", "UR3e": "D3", "FANUC": "D3", "Conveyor": "D3",
    "fanuc_positions (cell tray position)": "D3", "fanuc_output (pick position, v8.4 mode only)": "D3",
    "Operator_WS8": "D4", "Operator_WS9": "D5",
}

# single-file DES element -> (kind, discipline, location in the package)
V88_MAPPING = [
    ("main / run_model / MontracTLPP2.run", "entry point", "MDA", "mda/coordinator.py: evaluate_design, MDA.run; run_mda.py"),
    ("load_config, apply_overrides, validate_config, constants", "configuration", "shared", "des/config.py (unchanged)"),
    ("energy_table, energy_summary, validate_energy_config, infrastructure_power_W", "energy", "shared (CG6)", "des/energy.py (unchanged)"),
    ("Tray, FanucJob, Shuttle", "entities", "shared", "des/resources.py (unchanged)"),
    ("MontracTLPP2.__init__: env, resources, records, counters", "SimPy environment / resources", "shared", "des/simulation.py SimulationCore"),
    ("log, _acc, _new_tray, sample_duration", "helpers", "shared", "des/simulation.py"),
    ("_acquire, _release, _work, _reposition", "resource helpers", "shared (policy D6)", "des/simulation.py"),
    ("nodes, leg_hops, node_intervals, move_intervals, clearance", "SimPy resources / data", "D1", "disciplines/d1_transportation.py"),
    ("move, _occupy, _release_node", "SimPy process code", "D1", "D1.move, D1._occupy, D1._release_node"),
    ("shuttle_process: initial request and positioning", "SimPy process code", "D1", "D1.enter_line"),
    ("shuttle_process: loop record at the cycle start station", "accounting", "D1", "D1.complete_loop"),
    ("op_ws6_load, _eff_load_trays", "operation", "D2", "D2.op_ws6_load"),
    ("fanuc_branch: UR10 input transfer (31 s), _eff_tray_on_conveyor", "operation", "D2", "D2.transfer_to_fanuc_input"),
    ("op_ws6_retrieve, _eff_retrieve (all_of synchronisation)", "operation / SimPy event", "D2", "D2.op_ws6_retrieve"),
    ("op_ws8_unload, _eff_unload (physical part)", "operation", "D2", "D2.op_ws8_unload, D2._eff_unload"),
    ("UR10 resource", "SimPy resource", "D2", "core.resources['UR10']"),
    ("op_ws4, _eff_ur3e, scenario hook extra_hold_at_ws4", "operation", "D3", "D3.op_ws4"),
    ("_start_fanuc_branch, fanuc_branch (cell sequence 196 s), _eff_fanuc", "SimPy process", "D3", "D3.start_fanuc_branch, D3.fanuc_branch"),
    ("fanuc_positions, fanuc_output, cell release rule", "SimPy resources", "D3", "D3 (release_cell_after_retrieval)"),
    ("UR3e, FANUC, Conveyor resources", "SimPy resources", "D3", "core.resources"),
    ("op_ws8_qc, _eff_inspect, defect sampling", "operation", "D4", "D4.op_ws8_qc"),
    ("_eff_unload (Storage B accounting, batch records)", "accounting", "D4", "D4.confirm_dispatch"),
    ("tray state PRODUCTS_INSPECTED checked before unloading", "synchronisation", "D4", "D4.release_authorized"),
    ("Operator_WS8 resource", "SimPy resource", "D4", "core.resources['Operator_WS8']"),
    ("op_ws9, _eff_ws9", "operation", "D5", "D5.op_ws9"),
    ("Operator_WS9 resource", "SimPy resource", "D5", "core.resources['Operator_WS9']"),
    ("shuttle_process: cycle loop over cycle_steps, OPERATION_HANDLERS", "SimPy process / dispatching", "D6", "D6.shuttle_process, OPERATION_OWNER"),
    ("_control_delay, control_delays_s", "coordination", "D6", "D6.control_delay"),
    ("on_departure = start_fanuc_branch", "coordination", "D6", "D6 dispatcher -> D3.start_fanuc_branch"),
    ("RESOURCE_ORDER (fixed acquisition order, FIFO)", "coordination policy", "D6", "des/config.py, applied in core._acquire"),
    ("energy section (powers; standard powered operation)", "energy policy", "D6 (DV9)", "D6.energy_settings"),
    ("activity_times, conservation, results", "aggregation", "CG6", "mda/responses.py ResponseAggregator (verbatim)"),
    ("write_outputs, print_report, CLI", "outputs", "MDA", "mda/coordinator.py write_outputs; run_mda.py"),
]


def mapping_df() -> pd.DataFrame:
    return pd.DataFrame(V88_MAPPING, columns=["v8_8_element", "kind", "discipline", "location_in_cs2_mdo"])


def interface_df() -> pd.DataFrame:
    return pd.concat([Discipline.interface_rows(c) for c in DISCIPLINE_CLASSES.values()], ignore_index=True)
