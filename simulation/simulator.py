"""
simulation/simulator.py
========================
AMR Fleet Simulator — The Virtual Warehouse Floor

This module:
  1. Creates and initializes AMRs and charging stations
  2. Steps the simulation forward (moves AMRs, generates tasks)
  3. Publishes TICK events so agents can react
  4. Handles AMR movement physics (simple straight-line navigation)

This is NOT where intelligence lives — that's in the agents.
The simulator is just the "world" that agents operate in.

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Dict, List, Optional

from core.event_bus import EventBus
from core.models import (
    AMRState, AMRStatus, ChargingStation, Event, EventType,
    Position, Task, TaskStatus, TaskType
)
from core.state_store import StateStore

logger = logging.getLogger(__name__)

# AMR names for the fleet
AMR_NAMES = ["Atlas", "Bolt", "Cygnus", "Delta", "Echo", "Falcon", "Gemini"]


class AMRSimulator:
    """
    Discrete-time simulation of a warehouse AMR fleet on a 2D grid.

    Each tick:
    1. Move each AMR one step toward its target
    2. Check if tasks are completed
    3. Randomly spawn new tasks
    4. Publish TICK event with current tick number

    The grid is [0, grid_size] x [0, grid_size].
    """

    def __init__(
        self,
        bus:       EventBus,
        store:     StateStore,
        config:    Dict = None,
    ):
        self.bus   = bus
        self.store = store
        self.config = config or {}

        self.grid_size   = self.config.get("grid_size", 20)
        self.num_amrs    = self.config.get("num_amrs", 5)
        self.tick_rate   = self.config.get("tick_rate", 1.0)  # seconds

        self._tick:    int   = 0
        self._running: bool  = False

        # Task generation params
        self._task_spawn_prob  = 0.15   # 15% chance each tick
        self._task_spawn_burst = False  # Can be toggled from dashboard

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create AMRs and charging stations, register with state store."""
        self._create_amrs()
        self._create_charging_stations()
        logger.info(
            f"[Simulator] Initialized: {self.num_amrs} AMRs, "
            f"grid={self.grid_size}x{self.grid_size}"
        )

    def _create_amrs(self) -> None:
        """Spawn AMRs at random grid positions with full batteries."""
        names = AMR_NAMES[:self.num_amrs]
        for i, name in enumerate(names):
            amr_id  = f"AMR_{i+1:02d}"
            # Spread AMRs across the grid initially
            pos = Position(
                x=random.uniform(2, self.grid_size - 2),
                y=random.uniform(2, self.grid_size - 2),
            )
            amr = AMRState(
                amr_id=amr_id,
                name=name,
                position=pos,
                battery=random.uniform(70, 100),  # Start between 70-100%
            )
            self.store.register_amr(amr)
            logger.debug(f"[Simulator] Created AMR: {amr_id} ({name}) at {pos}")

    def _create_charging_stations(self) -> None:
        """
        Place charging stations at fixed positions (corners + center).
        Number of stations = ceil(num_amrs / 2) — ensures some competition.
        """
        g = self.grid_size
        positions = [
            Position(1, 1),              # Bottom-left
            Position(g-1, 1),            # Bottom-right
            Position(1, g-1),            # Top-left
            Position(g-1, g-1),          # Top-right
            Position(g//2, g//2),        # Center
        ]
        n_stations = max(2, self.num_amrs // 2)
        for i, pos in enumerate(positions[:n_stations]):
            station = ChargingStation(
                station_id=f"CS_{i+1:02d}",
                position=pos,
            )
            self.store.register_station(station)
            logger.debug(f"[Simulator] Created station CS_{i+1:02d} at {pos}")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main simulation loop. Run as an asyncio task.
        Each iteration = one tick.
        """
        self._running = True
        logger.info("[Simulator] Starting simulation loop.")

        while self._running:
            tick_start = time.time()

            try:
                await self._step()
            except Exception as e:
                logger.error(f"[Simulator] Error in tick {self._tick}: {e}", exc_info=True)

            # Maintain tick rate
            elapsed = time.time() - tick_start
            sleep_time = max(0, self.tick_rate - elapsed)
            await asyncio.sleep(sleep_time)

    async def stop(self) -> None:
        self._running = False
        logger.info("[Simulator] Stopped.")

    # ------------------------------------------------------------------
    # Single simulation step
    # ------------------------------------------------------------------

    async def _step(self) -> None:
        """Execute one simulation tick."""
        self._tick += 1

        # 1. Move all AMRs toward their targets
        for amr in self.store.get_all_amrs():
            await self._move_amr(amr)

        # 2. Check for task completion
        for amr in self.store.get_all_amrs():
            await self._check_task_completion(amr)

        # 3. Randomly spawn new tasks
        await self._maybe_spawn_task()

        # 4. Persist snapshot every 10 ticks
        if self._tick % 10 == 0:
            snapshot = self.store.get_snapshot(self._tick)
            self.store.persist_snapshot(snapshot)

        # 5. Publish TICK event — triggers all agent on_tick() methods
        await self.bus.publish(Event(
            event_type=EventType.TICK,
            source="simulator",
            payload={"tick": self._tick, "timestamp": time.time()},
        ))

    # ------------------------------------------------------------------
    # AMR Movement
    # ------------------------------------------------------------------

    async def _move_amr(self, amr: AMRState) -> None:
        """
        Move AMR one step toward its target_position.
        Simple linear interpolation — replace with A* if desired.
        """
        if amr.target_position is None:
            return
        if amr.status in (AMRStatus.CHARGING, AMRStatus.ERROR, AMRStatus.QUARANTINED):
            return

        pos    = amr.position
        target = amr.target_position
        dist   = pos.distance_to(target)

        if dist < 0.3:
            # Arrived!
            await self.store.update_amr_position(amr.amr_id, target)
            await self.store.set_amr_target(amr.amr_id, None)

            # If was moving to work, start working
            if amr.status == AMRStatus.MOVING and amr.current_task:
                await self.store.update_amr_status(amr.amr_id, AMRStatus.WORKING)

            return

        # Step toward target at amr.speed units per tick
        step = min(amr.speed, dist)
        dx = (target.x - pos.x) / dist * step
        dy = (target.y - pos.y) / dist * step
        new_pos = Position(
            x=round(pos.x + dx, 3),
            y=round(pos.y + dy, 3),
        )

        await self.store.update_amr_position(amr.amr_id, new_pos)

        # Publish position update
        await self.bus.publish(Event(
            event_type=EventType.AMR_POSITION_UPDATE,
            source="simulator",
            payload={
                "amr_id":   amr.amr_id,
                "position": new_pos.to_dict(),
                "tick":     self._tick,
            },
        ))

    # ------------------------------------------------------------------
    # Task Completion
    # ------------------------------------------------------------------

    async def _check_task_completion(self, amr: AMRState) -> None:
        """Check if an AMR has finished its current task."""
        if amr.status != AMRStatus.WORKING:
            return
        if not amr.current_task:
            await self.store.update_amr_status(amr.amr_id, AMRStatus.IDLE)
            return

        task = amr.current_task

        # Task completes after working for some ticks (simulate work duration)
        # Simple: complete after 3 ticks of WORKING (real: use task metadata)
        work_start = task.started_at or task.created_at
        work_duration = self.config.get("task_work_duration", 3.0)

        if time.time() - work_start >= work_duration:
            await self._complete_task(amr, task)

    async def _complete_task(self, amr: AMRState, task: Task) -> None:
        """Mark a task as completed and free the AMR."""
        await self.store.update_task_status(task.task_id, TaskStatus.COMPLETED)
        await self.store.update_amr_task(amr.amr_id, None)
        await self.store.update_amr_status(amr.amr_id, AMRStatus.IDLE)
        await self.store.increment_tasks_completed(amr.amr_id)

        await self.bus.publish(Event(
            event_type=EventType.TASK_COMPLETED,
            source="simulator",
            payload={
                "task_id": task.task_id,
                "amr_id":  amr.amr_id,
                "tick":    self._tick,
            },
        ))

        # Signal AMR is now idle
        await self.bus.publish(Event(
            event_type=EventType.AMR_STATUS_CHANGE,
            source="simulator",
            payload={
                "amr_id":     amr.amr_id,
                "new_status": AMRStatus.IDLE.value,
            },
        ))

    # ------------------------------------------------------------------
    # Task Spawning
    # ------------------------------------------------------------------

    async def _maybe_spawn_task(self) -> None:
        """Randomly spawn a new task to keep the fleet busy."""
        prob = self._task_spawn_prob * (3 if self._task_spawn_burst else 1)
        if random.random() > prob:
            return

        task_type = random.choice([
            TaskType.NAVIGATE, TaskType.PICKUP, TaskType.DROPOFF, TaskType.INSPECT
        ])
        target = Position(
            x=random.uniform(1, self.grid_size - 1),
            y=random.uniform(1, self.grid_size - 1),
        )
        task = Task(
            task_type=task_type,
            target=target,
            priority=random.randint(1, 4),
        )
        await self.store.add_task(task)

        await self.bus.publish(Event(
            event_type=EventType.TASK_CREATED,
            source="simulator",
            payload={
                "task_id":   task.task_id,
                "task_type": task.task_type.value,
                "priority":  task.priority,
                "target":    target.to_dict(),
            },
        ))

    # ------------------------------------------------------------------
    # External controls (called from dashboard)
    # ------------------------------------------------------------------

    async def inject_task(
        self,
        task_type: TaskType = TaskType.NAVIGATE,
        priority:  int = 3,
        target:    Optional[Position] = None,
    ) -> Task:
        """Manually inject a task (from dashboard UI)."""
        if target is None:
            target = Position(
                x=random.uniform(1, self.grid_size - 1),
                y=random.uniform(1, self.grid_size - 1),
            )
        task = Task(task_type=task_type, target=target, priority=priority)
        await self.store.add_task(task)
        await self.bus.publish(Event(
            event_type=EventType.TASK_CREATED,
            source="dashboard",
            payload={"task_id": task.task_id, "task_type": task_type.value, "priority": priority},
        ))
        return task

    def set_tick_rate(self, rate: float) -> None:
        """Change simulation speed (seconds per tick)."""
        self.tick_rate = max(0.1, rate)

    def set_task_burst(self, enabled: bool) -> None:
        """Enable/disable task burst mode."""
        self._task_spawn_burst = enabled

    @property
    def tick(self) -> int:
        return self._tick