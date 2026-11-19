"""Common interface of the six MDA disciplines.

Each discipline owns part of the production logic and of the shared SimPy resources. All disciplines act on
ONE SimulationCore (one SimPy environment); couplings are resolved by the shared event-driven simulation,
without fixed-point iterations.
"""
from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Dict, Tuple

import pandas as pd

from ..des.simulation import SimulationCore


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
