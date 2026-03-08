"""
agents/energy_agent.py
======================
Energy Agent — Q-Learning + LLM Battery Management

Q-learning handles routine per-tick decisions (fast, local).
LLM layer handles:
  - Explaining Q-learning decisions in natural language
  - Fleet-level charging scheduling when multiple AMRs need charge
  - Generating battery health summaries for the dashboard
  - Advising on adaptive drain/charge thresholds based on task queue

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.base_agent import BaseAgent
from core.models import (
    AlertSeverity, AMRState, AMRStatus, Event, EventType,
    Position, SystemSnapshot, TaskStatus, TaskType, Task
)

logger = logging.getLogger(__name__)

# ── Q-learning dimensions ────────────────────────────────────────────
N_BATTERY_BINS = 4
N_DIST_BINS    = 3
N_QUEUE_BINS   = 2
N_ACTIONS      = 3

ACTION_CONTINUE     = 0
ACTION_GO_CHARGE    = 1
ACTION_REDUCE_SPEED = 2


def _battery_bin(battery: float) -> int:
    if battery <= 10: return 0
    if battery <= 25: return 1
    if battery <= 60: return 2
    return 3

def _dist_bin(dist: float) -> int:
    if dist < 5:  return 0
    if dist < 12: return 1
    return 2

def _queue_bin(n: int) -> int:
    return 0 if n == 0 else 1


class EnergyAgent(BaseAgent):
    """
    Battery management agent combining Q-learning (speed) with LLM (intelligence).

    Decision pipeline per AMR per tick:
    1. If battery critical → force charge (override everything)
    2. Q-table: select action (ε-greedy)
    3. If charging conflict (multiple AMRs need same station) → ask LLM
    4. Apply action, update Q-table (Bellman equation)
    5. Every 10 ticks → LLM summarises fleet battery health
    """

    LOW_THRESHOLD      = 20.0
    CRITICAL_THRESHOLD = 10.0

    ALPHA   = 0.1
    GAMMA   = 0.9
    EPSILON = 0.15

    DRAIN_MOVING  = 0.3
    DRAIN_WORKING = 0.5
    DRAIN_IDLE    = 0.05
    CHARGE_RATE   = 2.0

    LLM_SUMMARY_INTERVAL = 5    # ticks between LLM fleet summaries

    def __init__(self, bus, store, config=None):
        super().__init__("EnergyAgent", bus, store, config)

        self.q_table = np.zeros(
            (N_BATTERY_BINS, N_DIST_BINS, N_QUEUE_BINS, N_ACTIONS), dtype=np.float32
        )
        self._seed_qtable()

        self._last_state:  Dict[str, Tuple[int, int, int]] = {}
        self._last_action: Dict[str, int] = {}
        self._charging_assignments: Dict[str, str] = {}

        self._total_reward:         float = 0.0
        self._charge_events:        int   = 0
        self._prevented_depletions: int   = 0
        self._llm_calls:            int   = 0
        self._last_llm_summary_tick: int  = 0

        self._groq_client = None
        self._llm_enabled = bool(os.getenv("GROQ_API_KEY"))
        self._llm_model   = os.getenv("LLM_MODEL", "llama3-70b-8192")

        # Latest LLM fleet health summary (shown in dashboard)
        self.llm_fleet_summary: str = ""

    def _seed_qtable(self) -> None:
        for b in range(N_BATTERY_BINS):
            for d in range(N_DIST_BINS):
                for q in range(N_QUEUE_BINS):
                    if b == 0:   self.q_table[b, d, q] = [-20,  15,  -5]
                    elif b == 1: self.q_table[b, d, q] = [ -5,   8,   2]
                    elif b == 2: self.q_table[b, d, q] = [  8,  -3,   3]
                    else:        self.q_table[b, d, q] = [ 10,  -8,   0]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        self.bus.subscribe(EventType.CHARGING_COMPLETE,     self._on_charging_complete)
        self.bus.subscribe(EventType.CHARGE_STATION_ASSIGN, self._on_station_assigned)

        if self._llm_enabled:
            self._init_groq()
            await self.log("LLM battery advisor ENABLED.", AlertSeverity.INFO)

        await self.log(
            f"Energy Agent online. Q-learning initialized with {self.q_table.size} state-action pairs.",
            AlertSeverity.INFO,
        )

    def _init_groq(self) -> None:
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except ImportError:
            logger.warning("groq package not installed.")
            self._groq_client = None
            self._llm_enabled = False

    # ------------------------------------------------------------------
    # Main Tick
    # ------------------------------------------------------------------

    async def on_tick(self, snapshot: SystemSnapshot) -> None:
        # Check if multiple AMRs need charging simultaneously → LLM scheduling
        needs_charge = [
            a for a in snapshot.amrs.values()
            if a.battery <= self.LOW_THRESHOLD
            and a.status not in (AMRStatus.CHARGING, AMRStatus.QUARANTINED)
            and not a.is_compromised
        ]
        free_stations = [s for s in self.store.get_all_stations() if not s.is_occupied]

        if (len(needs_charge) > len(free_stations)
                and self._groq_client
                and needs_charge):
            await self._llm_schedule_charging(needs_charge, free_stations, snapshot)

        for amr in snapshot.amrs.values():
            await self._process_amr_energy(amr, snapshot)

        await self._simulate_battery_dynamics(snapshot)

        # Periodic LLM fleet health summary
        tick = snapshot.tick
        if (self._groq_client
                and tick > 0
                and tick - self._last_llm_summary_tick >= self.LLM_SUMMARY_INTERVAL):
            await self._llm_generate_fleet_summary(snapshot)
            self._last_llm_summary_tick = tick

    # ------------------------------------------------------------------
    # Per-AMR Energy Processing
    # ------------------------------------------------------------------

    async def _process_amr_energy(self, amr: AMRState, snapshot: SystemSnapshot) -> None:
        if amr.status in (AMRStatus.CHARGING, AMRStatus.QUARANTINED, AMRStatus.ERROR):
            return
        if amr.is_compromised:
            return

        nearest    = self.store.get_nearest_free_station(amr.position)
        dist_to_st = amr.position.distance_to(nearest.position) if nearest else 99.0

        state = (
            _battery_bin(amr.battery),
            _dist_bin(dist_to_st),
            _queue_bin(snapshot.tasks_in_queue),
        )

        if amr.battery <= self.CRITICAL_THRESHOLD:
            await self._force_charge(amr, nearest)
            await self._update_qtable(amr.amr_id, state, ACTION_GO_CHARGE, -20)
            return

        action = self._select_action(state)
        reward = await self._apply_action(amr, action, nearest, snapshot)
        await self._update_qtable(amr.amr_id, state, action, reward)
        self._last_state[amr.amr_id]  = state
        self._last_action[amr.amr_id] = action

    def _select_action(self, state) -> int:
        if random.random() < self.EPSILON:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    async def _apply_action(self, amr, action, station, snapshot) -> float:
        if action == ACTION_GO_CHARGE:
            if station:
                await self._send_to_charge(amr, station)
                self._charge_events += 1
                reward = 5.0 if amr.battery <= self.LOW_THRESHOLD else -3.0
            else:
                await self.log(f"{amr.name} needs charge but NO FREE STATION!", AlertSeverity.CRITICAL)
                reward = -10.0
        elif action == ACTION_REDUCE_SPEED:
            await self.log(f"{amr.name} reducing speed to conserve battery ({amr.battery:.1f}%)", AlertSeverity.INFO)
            reward = 1.0
        else:
            if amr.battery <= self.LOW_THRESHOLD:
                await self.log(f"⚠️ {amr.name} battery LOW ({amr.battery:.1f}%) — continuing.", AlertSeverity.WARNING)
                await self.emit(EventType.BATTERY_LOW, {"amr_id": amr.amr_id, "battery": amr.battery}, AlertSeverity.WARNING)
                reward = -5.0
            else:
                reward = 2.0

        self._total_reward += reward
        return reward

    async def _update_qtable(self, amr_id, state, action, reward) -> None:
        current_q  = self.q_table[state][action]
        max_next_q = float(np.max(self.q_table[state]))
        self.q_table[state][action] = current_q + self.ALPHA * (
            reward + self.GAMMA * max_next_q - current_q
        )
        self._decisions_made += 1

    # ------------------------------------------------------------------
    # LLM: Charging Scheduler
    # ------------------------------------------------------------------

    async def _llm_schedule_charging(
        self,
        amrs_needing_charge: List[AMRState],
        free_stations: list,
        snapshot: SystemSnapshot,
    ) -> None:
        """
        When more AMRs need charging than stations available,
        ask LLM to decide which AMRs should charge first.
        """
        amr_info = "\n".join([
            f"  {a.name}: battery={a.battery:.1f}%, "
            f"task_priority={a.current_task.priority if a.current_task else 0}, "
            f"status={a.status.value}"
            for a in amrs_needing_charge
        ])

        prompt = f"""You are an AI energy manager for a warehouse robot fleet.

Multiple AMRs need charging but there are fewer stations available.

AMRs needing charge (sorted by urgency):
{amr_info}

Available charging stations: {len(free_stations)}
AMRs needing charge: {len(amrs_needing_charge)}

Decide which {len(free_stations)} AMR(s) should charge first.
Prioritize: critically low battery > active high-priority tasks > workload balance.
Respond in EXACTLY this JSON format (no other text):
{{"charge_first": ["<name1>", "<name2>"], "reasoning": "<one sentence>"}}"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.1,
            )
            self._llm_calls += 1
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            charge_names = data.get("charge_first", [])
            reasoning    = data.get("reasoning", "")

            await self.log(
                f"🤖 LLM charging schedule: charge {charge_names} first — {reasoning}",
                AlertSeverity.INFO,
            )

            # Set priority flag on selected AMRs (simulator will prefer them)
            for amr in amrs_needing_charge:
                if amr.name in charge_names:
                    await self.store.set_amr_charging_priority(amr.amr_id, True)

        except Exception as e:
            logger.warning(f"[EnergyAgent] LLM charging schedule failed: {e}")

    # ------------------------------------------------------------------
    # LLM: Fleet Battery Health Summary
    # ------------------------------------------------------------------

    async def _llm_generate_fleet_summary(self, snapshot: SystemSnapshot) -> None:
        """Generate a natural language fleet battery health summary every N ticks."""
        amrs = list(snapshot.amrs.values())
        amr_info = "\n".join([
            f"  {a.name}: {a.battery:.0f}% battery, {a.status.value}, "
            f"{a.tasks_completed} tasks done"
            for a in amrs
        ])

        prompt = f"""You are an AI fleet health analyst for a warehouse robot system.

Current fleet status at tick {snapshot.tick}:
{amr_info}

Average battery: {snapshot.fleet_avg_battery:.1f}%
Tasks in queue: {snapshot.tasks_in_queue}
Total alerts: {snapshot.total_alerts}

Write a brief 2-sentence fleet battery health summary for the operations dashboard.
Be specific about which AMRs need attention. Use plain English, no bullet points.
Respond with ONLY the summary text, nothing else."""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3,
            )
            self._llm_calls += 1
            self.llm_fleet_summary = response.choices[0].message.content.strip()
            await self.log(f"🤖 LLM Fleet Summary: {self.llm_fleet_summary}", AlertSeverity.INFO)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[EnergyAgent] LLM fleet summary failed: {error_msg}")
            await self.log(f"⚠️ LLM call failed: {error_msg[:120]}", AlertSeverity.WARNING)
            # Set a fallback so the dashboard panel always shows something
            if not self.llm_fleet_summary:
                self.llm_fleet_summary = (
                    f"LLM unavailable (tick {snapshot.tick}): {error_msg[:80]}"
                )

    # ------------------------------------------------------------------
    # Charging Actions (unchanged from original)
    # ------------------------------------------------------------------

    async def _force_charge(self, amr: AMRState, station) -> None:
        if station:
            await self.log(
                f"🔴 CRITICAL: {amr.name} battery at {amr.battery:.1f}%! "
                f"Force-routing to {station.station_id}.",
                AlertSeverity.CRITICAL,
            )
            await self._send_to_charge(amr, station)
            self._prevented_depletions += 1
        else:
            await self.log(
                f"🔴 CRITICAL: {amr.name} at {amr.battery:.1f}% — NO STATION FREE!",
                AlertSeverity.CRITICAL,
            )
        await self.emit(EventType.BATTERY_CRITICAL,
                        {"amr_id": amr.amr_id, "battery": amr.battery},
                        AlertSeverity.CRITICAL)

    async def _send_to_charge(self, amr: AMRState, station) -> None:
        if amr.current_task and amr.current_task.status not in (
            TaskStatus.COMPLETED, TaskStatus.CANCELLED
        ):
            await self.store.update_task_status(amr.current_task.task_id, TaskStatus.CANCELLED)
            await self.emit(EventType.TASK_DELEGATION, {
                "task_id": amr.current_task.task_id,
                "from_amr_id": amr.amr_id, "reason": "low_battery",
            })

        await self.store.assign_station(station.station_id, amr.amr_id)
        await self.store.update_amr_status(amr.amr_id, AMRStatus.LOW_BATTERY)
        await self.store.set_amr_target(amr.amr_id, station.position)
        await self.store.update_amr_task(amr.amr_id, None)
        self._charging_assignments[amr.amr_id] = station.station_id
        await self.emit(EventType.CHARGE_STATION_ASSIGN, {
            "amr_id": amr.amr_id, "station_id": station.station_id,
            "battery": amr.battery,
        })

    async def _simulate_battery_dynamics(self, snapshot: SystemSnapshot) -> None:
        for amr in snapshot.amrs.values():
            new_battery = amr.battery

            if amr.status == AMRStatus.CHARGING:
                new_battery = min(100.0, amr.battery + self.CHARGE_RATE)
                if new_battery >= 98.0:
                    new_battery = 100.0
                    await self.store.update_amr_status(amr.amr_id, AMRStatus.IDLE)
                    station_id = self._charging_assignments.pop(amr.amr_id, None)
                    if station_id:
                        await self.store.release_station(station_id)
                    await self.log(f"✅ {amr.name} fully charged. Returning to fleet.", AlertSeverity.INFO)
                    await self.emit(EventType.CHARGING_COMPLETE, {"amr_id": amr.amr_id})
            elif amr.status == AMRStatus.MOVING:
                new_battery -= self.DRAIN_MOVING
            elif amr.status == AMRStatus.WORKING:
                new_battery -= self.DRAIN_WORKING
            elif amr.status in (AMRStatus.IDLE, AMRStatus.LOW_BATTERY):
                new_battery -= self.DRAIN_IDLE

            await self.store.update_amr_battery(amr.amr_id, new_battery)

            if amr.status == AMRStatus.LOW_BATTERY and amr.amr_id in self._charging_assignments:
                sid = self._charging_assignments[amr.amr_id]
                station = next((s for s in self.store.get_all_stations()
                                if s.station_id == sid), None)
                if station and amr.position.distance_to(station.position) < 1.5:
                    await self.store.update_amr_status(amr.amr_id, AMRStatus.CHARGING)
                    await self.log(f"🔋 {amr.name} docked at station {sid}. Charging...", AlertSeverity.INFO)

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_charging_complete(self, event: Event) -> None:
        self._events_received += 1

    async def _on_station_assigned(self, event: Event) -> None:
        self._events_received += 1

    # ------------------------------------------------------------------
    # Dashboard Data
    # ------------------------------------------------------------------

    def get_qtable_stats(self) -> dict:
        return {
            "total_reward":          round(self._total_reward, 2),
            "charge_events":         self._charge_events,
            "prevented_depletions":  self._prevented_depletions,
            "avg_q_value":           round(float(np.mean(self.q_table)), 4),
            "exploration_rate":      self.EPSILON,
            "llm_calls":             self._llm_calls,
            "llm_fleet_summary":     self.llm_fleet_summary,
        }

    def get_qtable_heatmap_data(self) -> dict:
        collapsed = np.mean(self.q_table, axis=2)
        return {
            "battery_labels": ["Critical(0-10%)", "Low(10-25%)", "Med(25-60%)", "High(60-100%)"],
            "dist_labels":    ["Close(<5)", "Medium(5-12)", "Far(>12)"],
            "action_labels":  ["Continue", "Go Charge", "Reduce Speed"],
            "q_values":       collapsed.tolist(),
        }

    def get_stats(self) -> dict:
        base = super().get_stats()
        base["llm_calls"] = self._llm_calls
        return base