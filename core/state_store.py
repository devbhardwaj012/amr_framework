"""
core/state_store.py
===================
Centralized, thread-safe state store for the entire system.

All agents READ from and WRITE to the state store.
The store is the single source of truth for:
  - AMR states
  - Charging stations
  - Task queue
  - System metrics history (for dashboard charts)

Uses asyncio.Lock for safe concurrent access.

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.models import (
    AMRState, AMRStatus, ChargingStation, Position,
    SystemSnapshot, Task, TaskStatus
)

logger = logging.getLogger(__name__)


class StateStore:
    """
    The shared memory of the multi-agent system.

    Design rules:
    - All mutations go through update_*() methods (never mutate directly)
    - Provides get_snapshot() to freeze current state for agents
    - Persists history to SQLite for dashboard charts
    """

    def __init__(self, db_path: str = "data/amr_history.db"):
        self._amrs:     Dict[str, AMRState]        = {}
        self._stations: Dict[str, ChargingStation] = {}
        self._tasks:    Dict[str, Task]            = {}
        self._lock:     asyncio.Lock               = asyncio.Lock()

        # Rolling metrics history for dashboard graphs (last 300 ticks)
        self._battery_history:  deque = deque(maxlen=300)
        self._alert_history:    deque = deque(maxlen=300)
        self._task_history:     deque = deque(maxlen=300)

        # SQLite for persistence
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        logger.info("[StateStore] Initialized.")

    # ------------------------------------------------------------------
    # DB Setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                tick        INTEGER PRIMARY KEY,
                timestamp   REAL,
                snapshot_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                event_type  TEXT,
                source      TEXT,
                severity    TEXT,
                payload     TEXT,
                timestamp   REAL
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # AMR State Management
    # ------------------------------------------------------------------

    def register_amr(self, amr: AMRState) -> None:
        """Register a new AMR. Called during simulation setup."""
        self._amrs[amr.amr_id] = amr
        logger.debug(f"[StateStore] Registered AMR: {amr.amr_id}")

    def register_station(self, station: ChargingStation) -> None:
        """Register a charging station."""
        self._stations[station.station_id] = station

    async def update_amr_position(self, amr_id: str, position: Position) -> None:
        """Update an AMR's position and track distance traveled."""
        async with self._lock:
            amr = self._amrs.get(amr_id)
            if amr:
                dist = amr.position.distance_to(position)
                amr.total_distance += dist
                amr.position = position
                amr.last_seen = time.time()

    async def update_amr_status(self, amr_id: str, status: AMRStatus) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                self._amrs[amr_id].status = status

    async def update_amr_battery(self, amr_id: str, battery: float) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                delta = self._amrs[amr_id].battery - battery
                self._amrs[amr_id].battery = max(0.0, min(100.0, battery))
                if delta > 0:
                    self._amrs[amr_id].energy_consumed += delta

    async def update_amr_task(self, amr_id: str, task: Optional[Task]) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                self._amrs[amr_id].current_task = task

    async def set_amr_compromised(self, amr_id: str, compromised: bool) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                self._amrs[amr_id].is_compromised = compromised
                if compromised:
                    self._amrs[amr_id].status = AMRStatus.QUARANTINED
                    self._amrs[amr_id].alerts_triggered += 1

    async def set_amr_target(self, amr_id: str, target: Optional[Position]) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                self._amrs[amr_id].target_position = target

    async def increment_tasks_completed(self, amr_id: str) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                self._amrs[amr_id].tasks_completed += 1

    async def increment_tasks_failed(self, amr_id: str) -> None:
        async with self._lock:
            if amr_id in self._amrs:
                self._amrs[amr_id].tasks_failed += 1

    # ------------------------------------------------------------------
    # Task Queue Management
    # ------------------------------------------------------------------

    async def add_task(self, task: Task) -> None:
        async with self._lock:
            self._tasks[task.task_id] = task

    async def update_task_status(
        self, task_id: str, status: TaskStatus, amr_id: str = None
    ) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                if amr_id:
                    task.assigned_to = amr_id
                if status == TaskStatus.IN_PROGRESS:
                    task.started_at = time.time()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    task.completed_at = time.time()

    async def get_pending_tasks(self) -> List[Task]:
        async with self._lock:
            return [
                t for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
            ]

    # ------------------------------------------------------------------
    # Charging Station Management
    # ------------------------------------------------------------------

    async def assign_station(self, station_id: str, amr_id: str) -> None:
        async with self._lock:
            station = self._stations.get(station_id)
            if station:
                station.is_occupied = True
                station.occupant_id = amr_id

    async def release_station(self, station_id: str) -> None:
        async with self._lock:
            station = self._stations.get(station_id)
            if station:
                station.is_occupied = False
                station.occupant_id = None

    def get_free_station(self) -> Optional[ChargingStation]:
        """Get any unoccupied charging station (no lock — snapshot read only)."""
        for station in self._stations.values():
            if not station.is_occupied:
                return station
        return None

    def get_nearest_free_station(self, position: Position) -> Optional[ChargingStation]:
        """Get the closest free charging station to a given position."""
        free = [s for s in self._stations.values() if not s.is_occupied]
        if not free:
            return None
        return min(free, key=lambda s: s.position.distance_to(position))

    # ------------------------------------------------------------------
    # Read-only views (no lock needed — returns copies for safety)
    # ------------------------------------------------------------------

    def get_amr(self, amr_id: str) -> Optional[AMRState]:
        return self._amrs.get(amr_id)

    def get_all_amrs(self) -> List[AMRState]:
        return list(self._amrs.values())

    def get_all_stations(self) -> List[ChargingStation]:
        return list(self._stations.values())

    def get_all_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    def get_snapshot(self, tick: int = 0) -> SystemSnapshot:
        """
        Freeze current system state into a snapshot.
        Agents receive this on every tick — they never hold live references.
        """
        amrs = {k: v for k, v in self._amrs.items()}

        batteries = [a.battery for a in amrs.values()]
        avg_battery = sum(batteries) / len(batteries) if batteries else 0.0
        active = sum(1 for a in amrs.values() if a.status not in (
            AMRStatus.CHARGING, AMRStatus.ERROR, AMRStatus.QUARANTINED
        ))
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        alerts  = sum(a.alerts_triggered for a in amrs.values())

        # Record metrics history
        self._battery_history.append({"tick": tick, "avg": avg_battery})
        self._task_history.append({"tick": tick, "pending": len(pending)})

        return SystemSnapshot(
            tick=tick,
            amrs=amrs,
            charging_stations=dict(self._stations),
            pending_tasks=pending,
            fleet_avg_battery=avg_battery,
            active_amrs=active,
            tasks_in_queue=len(pending),
            total_alerts=alerts,
        )

    def get_battery_history(self) -> List[dict]:
        return list(self._battery_history)

    def get_task_history(self) -> List[dict]:
        return list(self._task_history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_snapshot(self, snapshot: SystemSnapshot) -> None:
        """Save a snapshot to SQLite (called periodically)."""
        try:
            data = {
                "amrs":    {k: v.to_dict() for k, v in snapshot.amrs.items()},
                "metrics": {
                    "avg_battery": snapshot.fleet_avg_battery,
                    "active_amrs": snapshot.active_amrs,
                    "alerts":      snapshot.total_alerts,
                }
            }
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)",
                (snapshot.tick, snapshot.timestamp, json.dumps(data))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[StateStore] Persist failed: {e}")