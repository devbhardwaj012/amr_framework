"""
agents/coordinator_agent.py
============================
Coordinator Agent — Intelligent Fleet Orchestration via LLM + Rules

Every meaningful decision goes through a two-layer pipeline:
  1. Rule engine  — instant, deterministic, handles common cases
  2. LLM layer    — Groq/LLaMA3-70B for complex, ambiguous situations

LLM is used for:
  - Multi-AMR path conflict negotiation (equal priority)
  - Optimal task assignment when scores are close (< 2 pts apart)
  - Fleet rebalancing strategy when >50% of AMRs are idle
  - Generating natural language decision logs for dashboard

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from core.base_agent import BaseAgent
from core.models import (
    AlertSeverity, AMRState, AMRStatus, Event, EventType,
    Position, SystemSnapshot, Task, TaskStatus, TaskType
)

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):

    COLLISION_RADIUS   = 2.0
    COLLISION_CRITICAL = 1.0
    LLM_SCORE_MARGIN   = 2.0   # use LLM if top two AMR scores within this margin

    def __init__(self, bus, store, config=None):
        super().__init__("CoordinatorAgent", bus, store, config)
        self._warned_pairs: set = set()
        self._groq_client = None
        self._llm_enabled = bool(os.getenv("GROQ_API_KEY"))
        self._llm_model   = os.getenv("LLM_MODEL", "llama3-70b-8192")
        self._decisions:  List[dict] = []
        self._llm_calls:  int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        self.bus.subscribe(EventType.TASK_CREATED,      self._on_task_created)
        self.bus.subscribe(EventType.AMR_STATUS_CHANGE, self._on_amr_status_change)
        self.bus.subscribe(EventType.COLLISION_WARNING, self._on_collision_warning)

        if self._llm_enabled:
            self._init_groq()
            await self.log("LLM reasoning ENABLED — Groq/LLaMA3-70B.", AlertSeverity.INFO)
        else:
            await self.log("No GROQ_API_KEY — rule-based mode only.", AlertSeverity.WARNING)

        await self.log("Coordinator Agent online. Monitoring fleet.", AlertSeverity.INFO)

    def _init_groq(self) -> None:
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logger.info("[CoordinatorAgent] Groq client initialized.")
        except ImportError:
            logger.warning("groq package not installed — pip install groq")
            self._groq_client = None
            self._llm_enabled = False

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    async def on_tick(self, snapshot: SystemSnapshot) -> None:
        amrs = list(snapshot.amrs.values())
        await self._check_collisions(amrs)
        if snapshot.pending_tasks:
            await self._assign_pending_tasks(snapshot.pending_tasks, amrs)
        await self._rebalance_idle_amrs(amrs, snapshot)

    # ------------------------------------------------------------------
    # Collision Detection & Avoidance
    # ------------------------------------------------------------------

    async def _check_collisions(self, amrs: List[AMRState]) -> None:
        active = [a for a in amrs if a.status in (
            AMRStatus.MOVING, AMRStatus.WORKING, AMRStatus.IDLE)]

        for i, a1 in enumerate(active):
            for a2 in active[i+1:]:
                dist     = a1.position.distance_to(a2.position)
                pair_key = tuple(sorted([a1.amr_id, a2.amr_id]))

                if dist <= self.COLLISION_CRITICAL:
                    await self._emergency_stop(a1, a2, dist)
                    self._warned_pairs.add(pair_key)
                elif dist <= self.COLLISION_RADIUS:
                    if pair_key not in self._warned_pairs:
                        await self._negotiate_path(a1, a2, dist)
                        self._warned_pairs.add(pair_key)
                else:
                    self._warned_pairs.discard(pair_key)

    async def _emergency_stop(self, a1: AMRState, a2: AMRState, dist: float) -> None:
        msg = (f"EMERGENCY: {a1.name} and {a2.name} are {dist:.2f} units apart! "
               f"Issuing stop command.")
        await self.log(msg, AlertSeverity.CRITICAL)
        await self.emit(EventType.COLLISION_WARNING, {
            "amr_ids": [a1.amr_id, a2.amr_id],
            "distance": dist, "severity": "critical", "action": "emergency_stop",
        }, AlertSeverity.CRITICAL)
        self._decisions_made += 1

    async def _negotiate_path(self, a1: AMRState, a2: AMRState, dist: float) -> None:
        p1 = a1.current_task.priority if a1.current_task else 0
        p2 = a2.current_task.priority if a2.current_task else 0

        if p1 != p2:
            # Rule: lower priority task yields
            yielder = a1 if p1 < p2 else a2
            reason  = f"task priority ({p1} vs {p2})"
        elif self._groq_client:
            # LLM: equal priority — ask LLaMA
            yielder, reason = await self._llm_negotiate_collision(a1, a2, dist)
        else:
            # Fallback rule: higher battery yields (keep lower-battery AMR moving)
            yielder = a1 if a1.battery > a2.battery else a2
            reason  = f"battery level ({a1.battery:.0f}% vs {a2.battery:.0f}%)"

        msg = (f"Path conflict: {a1.name} ↔ {a2.name} ({dist:.2f} units). "
               f"{yielder.name} yields — {reason}.")
        await self.log(msg, AlertSeverity.WARNING)
        await self.emit(EventType.PATH_NEGOTIATION, {
            "amr_ids":    [a1.amr_id, a2.amr_id],
            "yielder_id": yielder.amr_id,
            "distance":   dist,
            "reason":     reason,
            "action":     "yield_and_reroute",
        }, AlertSeverity.WARNING)
        self._decisions_made += 1
        await self.store.update_amr_status(yielder.amr_id, AMRStatus.IDLE)

    async def _llm_negotiate_collision(
        self, a1: AMRState, a2: AMRState, dist: float
    ) -> Tuple[AMRState, str]:
        """Ask LLaMA which AMR should yield in a path conflict."""
        prompt = f"""You are an AI fleet manager for a warehouse autonomous robot system.

Two AMRs are on a collision course and must negotiate who yields (pauses/reroutes).

AMR 1: {a1.name}
  Battery: {a1.battery:.1f}%
  Status: {a1.status.value}
  Current task: {a1.current_task.task_type.value if a1.current_task else "none"}
  Task priority: {a1.current_task.priority if a1.current_task else 0}/5
  Distance traveled today: {a1.total_distance:.1f} units
  Tasks completed: {a1.tasks_completed}

AMR 2: {a2.name}
  Battery: {a2.battery:.1f}%
  Status: {a2.status.value}
  Current task: {a2.current_task.task_type.value if a2.current_task else "none"}
  Task priority: {a2.current_task.priority if a2.current_task else 0}/5
  Distance traveled today: {a2.total_distance:.1f} units
  Tasks completed: {a2.tasks_completed}

Distance between them: {dist:.2f} grid units (collision radius: {self.COLLISION_RADIUS}).

Decide which AMR should yield. Consider: battery efficiency, task urgency, workload balance.
Respond in EXACTLY this JSON format (no other text):
{{"yield": "<AMR name>", "reason": "<one concise sentence>"}}"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.1,
            )
            self._llm_calls += 1
            text = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            yielder_name = data.get("yield", a1.name)
            reason       = data.get("reason", "LLM decision")
            yielder = a1 if a1.name in yielder_name else a2
            return yielder, f"🤖 LLM: {reason}"
        except Exception as e:
            logger.warning(f"[CoordinatorAgent] LLM collision negotiation failed: {e}")
            yielder = a1 if a1.battery > a2.battery else a2
            return yielder, f"LLM fallback: battery level ({a1.battery:.0f}% vs {a2.battery:.0f}%)"

    # ------------------------------------------------------------------
    # Task Assignment
    # ------------------------------------------------------------------

    async def _assign_pending_tasks(
        self, tasks: List[Task], amrs: List[AMRState]
    ) -> None:
        available = [
            a for a in amrs
            if a.status == AMRStatus.IDLE
            and not a.is_compromised
            and a.battery > self.config.get("energy_critical_threshold", 10)
        ]
        if not available:
            return

        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        assigned_ids = set()

        for task in sorted_tasks:
            if task.status != TaskStatus.PENDING:
                continue

            scored = [
                (self._score_amr_for_task(a, task), a)
                for a in available if a.amr_id not in assigned_ids
            ]
            if not scored:
                continue

            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_amr = scored[0]

            # If top two scores are very close — ask LLM
            if (self._groq_client
                    and len(scored) >= 2
                    and (best_score - scored[1][0]) < self.LLM_SCORE_MARGIN):
                best_amr, llm_reason = await self._llm_pick_assignee(
                    task, scored[0][1], scored[1][1]
                )
                await self.log(
                    f"LLM task assignment override: {llm_reason}", AlertSeverity.INFO
                )

            await self._assign_task(best_amr, task)
            assigned_ids.add(best_amr.amr_id)

    def _score_amr_for_task(self, amr: AMRState, task: Task) -> float:
        dist_score    = max(0, 10 - amr.position.distance_to(task.target)) if task.target else 5.0
        battery_score = amr.battery / 10.0
        idle_bonus    = 5.0 if amr.status == AMRStatus.IDLE else 0.0
        return dist_score + battery_score + idle_bonus

    async def _llm_pick_assignee(
        self, task: Task, a1: AMRState, a2: AMRState
    ) -> Tuple[AMRState, str]:
        """Ask LLaMA to pick the best AMR for a task when scores are close."""
        prompt = f"""You are an AI warehouse fleet manager. Assign a task to the best AMR.

Task:
  Type: {task.task_type.value}
  Priority: {task.priority}/5
  Target location: ({task.target.x:.1f}, {task.target.y:.1f})

Candidate AMR 1: {a1.name}
  Battery: {a1.battery:.1f}%
  Position: ({a1.position.x:.1f}, {a1.position.y:.1f})
  Distance to task: {a1.position.distance_to(task.target):.1f} units
  Tasks completed today: {a1.tasks_completed}
  Status: {a1.status.value}

Candidate AMR 2: {a2.name}
  Battery: {a2.battery:.1f}%
  Position: ({a2.position.x:.1f}, {a2.position.y:.1f})
  Distance to task: {a2.position.distance_to(task.target):.1f} units
  Tasks completed today: {a2.tasks_completed}
  Status: {a2.status.value}

Pick the best AMR. Consider: task urgency, battery efficiency, workload balance, proximity.
Respond in EXACTLY this JSON format (no other text):
{{"assign_to": "<AMR name>", "reason": "<one concise sentence>"}}"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.1,
            )
            self._llm_calls += 1
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            chosen_name = data.get("assign_to", a1.name)
            reason      = data.get("reason", "LLM decision")
            chosen = a1 if a1.name in chosen_name else a2
            return chosen, reason
        except Exception as e:
            logger.warning(f"[CoordinatorAgent] LLM task assignment failed: {e}")
            return a1, "LLM unavailable — used highest score"

    async def _assign_task(self, amr: AMRState, task: Task) -> None:
        await self.store.update_task_status(task.task_id, TaskStatus.ASSIGNED, amr.amr_id)
        await self.store.update_amr_task(amr.amr_id, task)
        await self.store.update_amr_status(amr.amr_id, AMRStatus.MOVING)
        if task.target:
            await self.store.set_amr_target(amr.amr_id, task.target)
        await self.log(
            f"Assigned task [{task.task_type.value}] (priority {task.priority}) "
            f"→ {amr.name} (battery: {amr.battery:.0f}%)",
            AlertSeverity.INFO,
        )
        await self.emit(EventType.TASK_ASSIGNED, {
            "task_id": task.task_id, "amr_id": amr.amr_id,
            "task_type": task.task_type.value,
        })
        self._decisions_made += 1

    # ------------------------------------------------------------------
    # Idle AMR Rebalancing
    # ------------------------------------------------------------------

    async def _rebalance_idle_amrs(
        self, amrs: List[AMRState], snapshot: SystemSnapshot
    ) -> None:
        idle_amrs = [a for a in amrs if a.status == AMRStatus.IDLE]
        if len(idle_amrs) < 2:
            return

        total_active = len([a for a in amrs if a.status != AMRStatus.CHARGING])
        idle_fraction = len(idle_amrs) / max(total_active, 1)

        # If >50% idle AND LLM available → ask for a fleet-level strategy
        if idle_fraction > 0.5 and self._groq_client and len(idle_amrs) >= 3:
            await self._llm_fleet_rebalance(idle_amrs, snapshot)
            return

        # Default rule: spread clustered idle AMRs
        grid = self.config.get("grid_size", 20)
        for i, a1 in enumerate(idle_amrs):
            for a2 in idle_amrs[i+1:]:
                if a1.position.distance_to(a2.position) < 3.0:
                    spread = Position(
                        x=min(grid - 2, a2.position.x + 5),
                        y=min(grid - 2, a2.position.y + 5),
                    )
                    await self.store.set_amr_target(a2.amr_id, spread)
                    await self.store.update_amr_status(a2.amr_id, AMRStatus.MOVING)
                    await self.log(
                        f"Rebalancing: moving {a2.name} away from {a1.name} "
                        f"(clustering at {a1.position.distance_to(a2.position):.1f} units)",
                        AlertSeverity.INFO,
                    )

    async def _llm_fleet_rebalance(
        self, idle_amrs: List[AMRState], snapshot: SystemSnapshot
    ) -> None:
        """Ask LLaMA for a fleet-wide coverage strategy when many AMRs are idle."""
        amr_summaries = "\n".join([
            f"  {a.name}: pos=({a.position.x:.1f},{a.position.y:.1f}) "
            f"battery={a.battery:.0f}% tasks_done={a.tasks_completed}"
            for a in idle_amrs
        ])
        grid = self.config.get("grid_size", 20)

        prompt = f"""You are an AI warehouse fleet manager. Multiple AMRs are idle and need to be repositioned for optimal warehouse coverage.

Warehouse grid: {grid}x{grid} units
Pending tasks in queue: {snapshot.tasks_in_queue}

Idle AMRs:
{amr_summaries}

Suggest target positions for each AMR to maximize warehouse coverage and readiness.
Consider: battery levels (don't send low-battery AMRs far), even grid distribution, task queue size.
Respond in EXACTLY this JSON format (no other text):
[{{"amr": "<name>", "target_x": <float>, "target_y": <float>, "reason": "<short reason>"}}]"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            self._llm_calls += 1
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            assignments = json.loads(text)

            for entry in assignments:
                amr_name = entry.get("amr", "")
                tx = float(entry.get("target_x", grid / 2))
                ty = float(entry.get("target_y", grid / 2))
                reason = entry.get("reason", "LLM rebalance")

                amr = next((a for a in idle_amrs if a.name == amr_name), None)
                if amr:
                    tx = max(0, min(grid - 1, tx))
                    ty = max(0, min(grid - 1, ty))
                    await self.store.set_amr_target(amr.amr_id, Position(tx, ty))
                    await self.store.update_amr_status(amr.amr_id, AMRStatus.MOVING)
                    await self.log(
                        f"🤖 LLM rebalance: {amr_name} → ({tx:.0f},{ty:.0f}) — {reason}",
                        AlertSeverity.INFO,
                    )

        except Exception as e:
            logger.warning(f"[CoordinatorAgent] LLM fleet rebalance failed: {e}")
            # Fall back to basic spread
            grid = self.config.get("grid_size", 20)
            for i, amr in enumerate(idle_amrs):
                spread = Position(
                    x=min(grid - 2, amr.position.x + 4),
                    y=min(grid - 2, amr.position.y + 4),
                )
                await self.store.set_amr_target(amr.amr_id, spread)
                await self.store.update_amr_status(amr.amr_id, AMRStatus.MOVING)

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_task_created(self, event: Event) -> None:
        self._events_received += 1
        await self.log(
            f"New task received: [{event.payload.get('task_type', 'unknown')}] "
            f"priority {event.payload.get('priority', 1)}",
            AlertSeverity.INFO,
        )

    async def _on_amr_status_change(self, event: Event) -> None:
        self._events_received += 1
        if event.payload.get("new_status") == AMRStatus.IDLE.value:
            await self.log(
                f"{event.payload.get('amr_id')} is now idle — checking task queue.",
                AlertSeverity.INFO,
            )

    async def _on_collision_warning(self, event: Event) -> None:
        self._events_received += 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        base = super().get_stats()
        base["llm_calls"] = self._llm_calls
        return base