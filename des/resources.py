"""Shared DES entities: trays, FANUC jobs (one per batch) and shuttles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
