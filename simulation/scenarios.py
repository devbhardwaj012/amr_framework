"""
simulation/scenarios.py
========================
Pre-built Demo Scenarios for Presentations

Each scenario sets up a specific situation that demonstrates agent
capabilities. Useful for demos, testing, and evaluation.

Available scenarios:
  1. NORMAL_OPERATIONS   — baseline healthy fleet
  2. BATTERY_CRISIS      — multiple AMRs running low at once
  3. COLLISION_STRESS    — AMRs converging on same point
  4. NETWORK_ATTACK      — simulated 5G intrusion event
  5. TASK_SURGE          — sudden burst of high-priority tasks
  6. FULL_STRESS_TEST    — all problems simultaneously

Usage:
    from simulation.scenarios import ScenarioEngine, Scenario
    engine = ScenarioEngine(simulator, store, bus)
    await engine.run(Scenario.BATTERY_CRISIS)

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import logging
import random
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from core.models import (
    AlertSeverity, AMRStatus, Event, EventType,
    Position, Task, TaskType
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state_store import StateStore
    from simulation.simulator import AMRSimulator

logger = logging.getLogger(__name__)


class Scenario(str, Enum):
    NORMAL_OPERATIONS = "normal_operations"
    BATTERY_CRISIS    = "battery_crisis"
    COLLISION_STRESS  = "collision_stress"
    NETWORK_ATTACK    = "network_attack"
    TASK_SURGE        = "task_surge"
    FULL_STRESS_TEST  = "full_stress_test"


# Human-readable descriptions for the dashboard UI
SCENARIO_DESCRIPTIONS = {
    Scenario.NORMAL_OPERATIONS: (
        "Standard warehouse operations. AMRs pick up tasks, "
        "charge when needed, and maintain safe distances."
    ),
    Scenario.BATTERY_CRISIS: (
        "3 AMRs simultaneously drop to critical battery levels. "
        "Energy Agent must prioritize charging stations and reassign tasks."
    ),
    Scenario.COLLISION_STRESS: (
        "All AMRs navigate toward the center simultaneously. "
        "Coordinator Agent negotiates paths to prevent collisions."
    ),
    Scenario.NETWORK_ATTACK: (
        "A simulated packet flood + jamming attack targets two AMRs. "
        "Sentinel Agent detects, quarantines, and reports the intrusion."
    ),
    Scenario.TASK_SURGE: (
        "10 high-priority tasks injected simultaneously. "
        "Coordinator Agent races to assign optimally."
    ),
    Scenario.FULL_STRESS_TEST: (
        "Everything at once: low batteries, collision risk, network attack, "
        "and task surge. All three agents must cooperate."
    ),
}


class ScenarioEngine:
    """
    Executes pre-built demo scenarios by directly manipulating
    AMR state and injecting events.
    """

    def __init__(
        self,
        simulator: "AMRSimulator",
        store:     "StateStore",
        bus:       "EventBus",
    ):
        self.simulator = simulator
        self.store     = store
        self.bus       = bus
        self._active_scenario: Optional[Scenario] = None
        self._scenario_log: List[str] = []

    async def run(self, scenario: Scenario) -> str:
        """
        Execute a scenario. Returns a summary string.
        Can be called from the dashboard.
        """
        self._active_scenario = scenario
        self._scenario_log.clear()

        handlers = {
            Scenario.NORMAL_OPERATIONS: self._scenario_normal,
            Scenario.BATTERY_CRISIS:    self._scenario_battery_crisis,
            Scenario.COLLISION_STRESS:  self._scenario_collision_stress,
            Scenario.NETWORK_ATTACK:    self._scenario_network_attack,
            Scenario.TASK_SURGE:        self._scenario_task_surge,
            Scenario.FULL_STRESS_TEST:  self._scenario_full_stress,
        }

        handler = handlers.get(scenario)
        if not handler:
            return f"Unknown scenario: {scenario}"

        self._log(f"▶ Starting scenario: {scenario.value}")
        await handler()
        self._log(f"✅ Scenario setup complete: {scenario.value}")

        return "\n".join(self._scenario_log)

    # ------------------------------------------------------------------
    # Individual Scenarios
    # ------------------------------------------------------------------

    async def _scenario_normal(self) -> None:
        """Inject a mix of normal tasks to keep fleet busy."""
        task_types = [TaskType.NAVIGATE, TaskType.PICKUP, TaskType.DROPOFF, TaskType.INSPECT]
        for i in range(5):
            task = Task(
                task_type=random.choice(task_types),
                target=self._random_pos(),
                priority=random.randint(1, 3),
            )
            await self.store.add_task(task)
            self._log(f"  Injected task: {task.task_type.value} priority={task.priority}")
        self._log("Normal operations scenario active — fleet running standard tasks.")

    async def _scenario_battery_crisis(self) -> None:
        """Force multiple AMRs to critical battery level."""
        amrs = self.store.get_all_amrs()
        targets = amrs[:min(3, len(amrs))]

        for amr in targets:
            # Set battery to 8-12% (critical zone)
            crisis_battery = random.uniform(8.0, 12.0)
            await self.store.update_amr_battery(amr.amr_id, crisis_battery)
            await self.bus.publish(Event(
                event_type=EventType.BATTERY_CRITICAL,
                source="scenario_engine",
                payload={"amr_id": amr.amr_id, "battery": crisis_battery},
                severity=AlertSeverity.CRITICAL,
            ))
            self._log(f"  {amr.name} battery forced to {crisis_battery:.1f}% (CRITICAL)")

        self._log(
            "Battery Crisis: Energy Agent will now race to route AMRs to "
            "charging stations and reassign their tasks."
        )

    async def _scenario_collision_stress(self) -> None:
        """Route all AMRs toward the grid center simultaneously."""
        amrs   = self.store.get_all_amrs()
        grid   = self.simulator.grid_size
        center = Position(grid / 2, grid / 2)

        # Spread AMRs to corners first, then route to center
        corners = [
            Position(2, 2), Position(grid-2, 2),
            Position(2, grid-2), Position(grid-2, grid-2),
            Position(grid//3, grid//3),
        ]

        for i, amr in enumerate(amrs):
            start = corners[i % len(corners)]
            await self.store.update_amr_position(amr.amr_id, start)
            await self.store.set_amr_target(amr.amr_id, center)
            await self.store.update_amr_status(amr.amr_id, AMRStatus.MOVING)
            self._log(f"  {amr.name}: {start} → center {center}")

        self._log(
            "Collision Stress: All AMRs converging on center. "
            "Coordinator Agent will detect proximity and negotiate paths."
        )

    async def _scenario_network_attack(self) -> None:
        """
        Trigger an immediate simulated network attack on 1-2 AMRs.
        The Sentinel Agent's Isolation Forest will detect this once
        trained — or we can force-feed anomalous packets via events.
        """
        amrs = self.store.get_all_amrs()
        targets = random.sample(amrs, min(2, len(amrs)))
        attack_types = ["PACKET_FLOOD", "JAMMING"]

        for i, amr in enumerate(targets):
            attack_type = attack_types[i % len(attack_types)]
            # Force-inject an anomaly detection event (bypasses ML training phase)
            await self.bus.publish(Event(
                event_type=EventType.INTRUSION_ALERT,
                source="scenario_engine",
                payload={
                    "amr_id":      amr.amr_id,
                    "amr_name":    amr.name,
                    "attack_type": attack_type,
                    "score":       -0.45,
                    "severity":    "critical",
                    "packet_size": 65535 if attack_type == "DATA_EXFIL" else 64,
                    "latency_ms":  120.0 if attack_type == "PACKET_FLOOD" else 5.0,
                    "packet_loss": 0.0 if attack_type == "PACKET_FLOOD" else 0.65,
                    "action":      "QUARANTINED",
                    "tick":        self.simulator.tick,
                    "forced_by":   "scenario_engine",
                },
                severity=AlertSeverity.CRITICAL,
            ))
            # Also mark AMR as compromised in state
            await self.store.set_amr_compromised(amr.amr_id, True)
            self._log(f"  {amr.name}: simulated {attack_type} attack → QUARANTINED")

        self._log(
            "Network Attack scenario active. Sentinel Agent will investigate "
            "and auto-restore AMRs after quarantine period."
        )

    async def _scenario_task_surge(self) -> None:
        """Inject 10 high-priority tasks simultaneously."""
        task_types = [TaskType.PICKUP, TaskType.DROPOFF, TaskType.NAVIGATE, TaskType.INSPECT]

        for i in range(10):
            task = Task(
                task_type=random.choice(task_types),
                target=self._random_pos(),
                priority=random.randint(3, 5),  # High priority only
            )
            await self.store.add_task(task)
            await self.bus.publish(Event(
                event_type=EventType.TASK_CREATED,
                source="scenario_engine",
                payload={
                    "task_id":   task.task_id,
                    "task_type": task.task_type.value,
                    "priority":  task.priority,
                },
            ))

        self._log(
            "Task Surge: 10 high-priority tasks injected. "
            "Coordinator Agent will score and assign optimally."
        )

    async def _scenario_full_stress(self) -> None:
        """Run all scenarios concurrently for maximum chaos."""
        self._log("FULL STRESS TEST — engaging all scenarios simultaneously...")
        await self._scenario_battery_crisis()
        await asyncio.sleep(0.1)
        await self._scenario_network_attack()
        await asyncio.sleep(0.1)
        await self._scenario_task_surge()
        self._log(
            "All stress conditions active. Watch agents cooperate: "
            "Energy routes low-battery AMRs, Coordinator reassigns tasks, "
            "Sentinel quarantines compromised AMRs."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_pos(self) -> Position:
        g = self.simulator.grid_size
        return Position(
            x=random.uniform(2, g - 2),
            y=random.uniform(2, g - 2),
        )

    def _log(self, msg: str) -> None:
        self._scenario_log.append(msg)
        logger.info(f"[ScenarioEngine] {msg}")

    @property
    def active_scenario(self) -> Optional[Scenario]:
        return self._active_scenario

    @staticmethod
    def list_scenarios() -> List[dict]:
        """Return all scenarios with descriptions for UI rendering."""
        return [
            {
                "id":          s.value,
                "name":        s.value.replace("_", " ").title(),
                "description": SCENARIO_DESCRIPTIONS[s],
            }
            for s in Scenario
        ]