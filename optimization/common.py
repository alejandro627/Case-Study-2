"""Shared paths of MDF 1.0."""
from __future__ import annotations

import os

from ..des.config import PACKAGE_DIR, RESULTS_DIR

MDF_VERSION = "1.1"
OUT_DIR = os.path.join(RESULTS_DIR, "optimisation")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RUN_DIR = os.path.join(OUT_DIR, "nsga2_runs")


def ensure_dirs() -> None:
    for d in (OUT_DIR, FIG_DIR, RUN_DIR):
        os.makedirs(d, exist_ok=True)
