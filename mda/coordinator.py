"""MDA coordinator: evaluate one design vector with the integrated six-discipline SimPy DES.

    from cs2_mdo.mda.coordinator import evaluate_design
    out = evaluate_design({"DV1": 2, "DV2": 3, "DV3": 1, "DV4": 3, "DV5": 3, "DV6": 1, "DV7": 3, "DV8": 2, "DV9": 1})
    out["system_responses"]["throughput_products_per_h"], out["system_responses"]["energy_total_kWh"]

Steps: (1) validate the DV levels against their domains; (2) check that the DES represents every selected
alternative, otherwise return status 'unsupported_configuration' without simulating; (3) build ONE
SimulationCore (one SimPy environment); (4) create D1-D6 and connect them; (5) the shared resources and
coupling events are registered in the core and the disciplines; (6) run; (7) aggregate (CG6); (8) verify
C1-C6; (9) return structured results.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import simpy

from ..des.config import DEFAULT_CONFIG_PATH, apply_overrides, load_config, validate_config
from ..des.simulation import DES_VERSION, SimulationCore
from .architecture import DISCIPLINE_CLASSES
from .constraints import ConstraintInputs, check_constraints
from .couplings import diagnostics
from .alternatives import apply_design
from .calibration import calibrated_params, load_calibration
from .design_variables import SUPPORTED, des_mapping, load_design_variables, support_table, validate_design_vector
from .responses import ResponseAggregator

MDA_VERSION = "1.5"
MODES = ("des",)


class MDA:
    """Six disciplines acting on one shared SimulationCore."""

    def __init__(self, cfg: Dict[str, Any], energy_policy: str = "standard_powered_operation", uncertainty: Any = None):
        self.core = SimulationCore(cfg)
        if uncertainty is not None:     # UDES 1.0: breakdown/restoration model attached to the shared core
            self.core.failure = uncertainty.attach(self.core)
        self.disciplines = {}
        for did, cls in DISCIPLINE_CLASSES.items():
            self.disciplines[did] = cls(self.core, energy_policy) if did == "D6" else cls(self.core)
        for d in self.disciplines.values():
            d.connect(self.disciplines)

    def run(self) -> "MDA":
        self.core.run(self.disciplines["D6"].shuttle_process)
        return self

    def aggregate(self) -> Dict[str, Any]:
        d6 = self.disciplines["D6"]
        status = d6.collect_status()
        energy_settings = d6.energy_settings()
        res = ResponseAggregator(self.core, self.disciplines).results()
        res["kpis"]["energy_management_policy"] = energy_settings["policy"]
        for cid in ("CG6.1", "CG6.2", "CG6.3", "CG6.4"):
            self.core.signal(cid)
        feas = check_constraints(ConstraintInputs.from_mda(self))
        self.core.signal("CG6.6")
        self.core.signal("CG6.5")
        return {"results": res, "feasibility": feas, "status_snapshot": status, "energy_settings": energy_settings,
                "discipline_outputs": {k: d.outputs(res) for k, d in self.disciplines.items()},
                "coupling_diagnostics": diagnostics(self.core.coupling_log)}


def _load(config: Union[None, str, Dict[str, Any]]) -> (Dict[str, Any], str):
    if config is None:
        return load_config(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH
    if isinstance(config, str):
        return load_config(config), config
    return copy.deepcopy(config), "dict"


def evaluate_design(design_vector: Dict[str, int], config: Union[None, str, Dict[str, Any]] = None, mode: str = "des",
                    overrides: Optional[List[str]] = None, uncertainty: Any = None) -> Dict[str, Any]:
    """Evaluate a candidate configuration using the integrated six-discipline SimPy MDA.

    Returns a dict with: status ('evaluated' | 'unsupported_configuration' | 'invalid_design_vector'), feasible,
    system_responses, discipline_outputs, coupling_diagnostics, feasibility (C1-C6), design_support,
    results (all DES tables and KPIs), status_snapshot and metadata.
    `overrides` are dotted configuration overrides ("simulation.horizon_h=4"); they may not set a DV-controlled key."""
    t_start = time.perf_counter()
    if mode not in MODES:
        raise NotImplementedError(f"mode={mode!r}: only the integrated DES ('des') is implemented (no surrogate)")
    spec = load_design_variables()
    z = dict(design_vector)
    out: Dict[str, Any] = {"design_vector": z, "mode": mode, "feasible": None, "system_responses": None,
                           "discipline_outputs": None, "coupling_diagnostics": None, "feasibility": None, "results": None}
    errors = validate_design_vector(z, spec)
    if errors:
        out.update(status="invalid_design_vector", messages=errors)
        return out
    cal = load_calibration()
    sup = support_table(z, spec, cal)
    out["design_support"] = sup
    unsupported = sup[~sup.des_status.isin(SUPPORTED)]
    if len(unsupported):
        out.update(status="unsupported_configuration",
                   messages=[f"{r.design_variable} level {r.level} ({r.alternative}): {r.des_representation_or_reason}"
                             for r in unsupported.itertuples()])
        return out

    cfg, source = _load(config)
    dv_over, policy = des_mapping(z, spec)
    clash = [o for o in (overrides or []) if o.split("=", 1)[0].strip() in dv_over]
    if clash:
        raise ValueError(f"overrides {clash} set keys controlled by the design vector ({sorted(dv_over)})")
    cfg = apply_overrides(cfg, [f"{k}={json.dumps(v)}" for k, v in dv_over.items()] + list(overrides or []))
    cfg = apply_design(cfg, z, calibrated_params(cal))      # calibrated DV alternatives (none for baseline levels)
    validate_config(cfg)

    mda = MDA(cfg, energy_policy=policy, uncertainty=uncertainty).run()
    agg = mda.aggregate()
    res, feas = agg["results"], agg["feasibility"]
    k = res["kpis"]
    out.update(
        status="evaluated", feasible=bool(feas.satisfied.all()),
        system_responses={
            "throughput_products_per_h": k["throughput_products_per_h"], "energy_total_kWh": k["energy_total_kWh"],
            "energy_total_Wh": k["energy_total_Wh"], "specific_energy_Wh_per_product": k["specific_energy_Wh_per_product"],
            "finished_products": k["finished_products_storage_b"], "completed_production_batches": k["completed_production_batches"],
            "completed_shuttle_loops": k["completed_shuttle_loops"], "n_shuttles": k["n_shuttles"], "horizon_h": k["horizon_h"]},
        discipline_outputs=agg["discipline_outputs"], coupling_diagnostics=agg["coupling_diagnostics"], feasibility=feas,
        results=res, status_snapshot=agg["status_snapshot"], messages=list(res["warnings"]),
        metadata={"des_version": DES_VERSION, "mda_version": MDA_VERSION, "mode": mode, "config_source": source,
                  "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                  "design_variable_overrides": dv_over, "energy_management_policy": policy,
                  "calibrated_alternatives": sorted(cfg.get("alternatives", {})) + (["energy_management"] if "energy_management" in cfg else []),
                  "overrides": list(overrides or []), "runtime_s": time.perf_counter() - t_start,
                  "python": platform.python_version(), "simpy": simpy.__version__, "pandas": pd.__version__,
                  "n_events": len(res["events"]), "coupling_iterations": 0,
                  "coupling_resolution": "single shared SimPy event-driven simulation"})
    out["_cfg"] = cfg
    if uncertainty is not None:     # UDES 1.0: breakdown trace and interval records for the recovery/energy checks
        out["_mda"] = mda
    return out


def write_outputs(out: Dict[str, Any], out_dir: str, tag: Optional[str] = None) -> Dict[str, str]:
    """Write the DES tables plus the MDA outputs of one evaluated design."""
    os.makedirs(out_dir, exist_ok=True)
    res = out["results"]
    k = res["kpis"]
    tag = tag or f"{k['n_shuttles']}sh_{float(k['horizon_h']):g}h"
    pre = os.path.join(out_dir, f"cs2_mda_{tag}")
    paths = {}

    def csv(name, df):
        paths[name] = f"{pre}_{name}.csv"
        df.to_csv(paths[name], index=False)

    csv("kpis", pd.DataFrame(list(k.items()), columns=["metric", "value"]))
    for name, key in (("event_log", "events"), ("loops", "loops"), ("batches", "batches"), ("fanuc_jobs", "jobs"),
                      ("resources", "resources"), ("stations", "stations"), ("activity", "activity"), ("energy", "energy")):
        csv(name, res[key])
    csv("warnings", pd.DataFrame({"warning": res["warnings"]}))
    csv("feasibility_C1_C6", out["feasibility"])
    csv("coupling_diagnostics", out["coupling_diagnostics"])
    csv("design_support", out["design_support"])
    csv("discipline_summary", discipline_summary(out))
    paths["run_config"] = f"{pre}_run_config.json"
    with open(paths["run_config"], "w", encoding="utf-8") as fh:
        json.dump(out["_cfg"], fh, indent=2)
    paths["metadata"] = f"{pre}_metadata.json"
    with open(paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump({"design_vector": out["design_vector"], "status": out["status"], "feasible": out["feasible"],
                   "system_responses": out["system_responses"], **out["metadata"]}, fh, indent=2, default=str)
    return paths


def discipline_summary(out: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for did, o in out["discipline_outputs"].items():
        act = o["activity"]
        rows.append({"discipline": did, "name": o["name"], "design_variables": ", ".join(o["design_variables"]),
                     "resources": ", ".join(act["resource"]) if len(act) else "-",
                     "active_s": float(act["active_s"].sum()) if len(act) else 0.0,
                     "mean_utilisation_pct": float(act["active_pct"].mean()) if len(act) else float("nan"),
                     "energy_consumers": ", ".join(o["energy"]["consumer"]) if len(o["energy"]) else "-",
                     "energy_Wh": o["energy_Wh"], "n_kpis": len(o["kpis"])})
    return pd.DataFrame(rows)
