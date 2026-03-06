"""
core/orchestrator.py
=====================
System Orchestrator — The Entry Point

Wires together:
  - ConfigManager   (typed, validated config)
  - EventBus        (async pub/sub)
  - StateStore      (shared state + SQLite)
  - 3 Agents        (Coordinator, Energy, Sentinel)
  - AMRSimulator    (virtual warehouse floor)
  - ScenarioEngine  (demo scenarios)
  - MetricsExporter (CSV/JSON exports)
  - Logger          (colored console + rotating file)

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from core.event_bus import EventBus
from core.state_store import StateStore
from core.logger import setup_logging
from core.metrics_exporter import MetricsExporter
from config.config_manager import load_config, SystemConfig
from agents.coordinator_agent import CoordinatorAgent
from agents.energy_agent import EnergyAgent
from agents.sentinel_agent import SentinelAgent
from simulation.simulator import AMRSimulator
from simulation.scenarios import ScenarioEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level system coordinator.

    Lifecycle:
        orch = Orchestrator()
        await orch.start()
        await orch.run_forever()   # blocks
        await orch.stop()
    """

    def __init__(self, config_overrides: dict = None):
        load_dotenv()

        # Config
        self._sys_config: SystemConfig = load_config(config_overrides)
        self.config: dict = self._sys_config.to_flat_dict()

        # Logging
        Path("logs").mkdir(exist_ok=True)
        setup_logging(
            log_level=self._sys_config.log_level,
            log_file=self._sys_config.log_path,
        )

        # Core infrastructure
        self.bus   = EventBus(max_queue_size=2000)
        self.store = StateStore(db_path=self._sys_config.db_path)

        # Agents
        self.coordinator = CoordinatorAgent(self.bus, self.store, self.config)
        self.energy      = EnergyAgent(self.bus, self.store, self.config)
        self.sentinel    = SentinelAgent(self.bus, self.store, self.config)

        # Simulator
        self.simulator = AMRSimulator(self.bus, self.store, self.config)

        # Scenario engine (initialized after simulator.initialize())
        self.scenario_engine: Optional[ScenarioEngine] = None

        # Metrics exporter
        self.exporter = MetricsExporter(
            store=self.store,
            sentinel=self.sentinel,
            energy=self.energy,
        )

        self._tasks:   list = []
        self._running: bool = False

    async def start(self) -> None:
        logger.info("=" * 60)
        logger.info("  AMR Multi-Agent Framework — Starting")
        logger.info("=" * 60)

        self.simulator.initialize()

        self.scenario_engine = ScenarioEngine(
            simulator=self.simulator,
            store=self.store,
            bus=self.bus,
        )

        await self.coordinator.start()
        await self.energy.start()
        await self.sentinel.start()

        self._running = True

        self._tasks = [
            asyncio.create_task(self.bus.run(),         name="event_bus"),
            asyncio.create_task(self.simulator.run(),   name="simulator"),
            asyncio.create_task(self.coordinator.run(), name="coordinator"),
            asyncio.create_task(self.energy.run(),      name="energy"),
            asyncio.create_task(self.sentinel.run(),    name="sentinel"),
        ]

        cfg = self._sys_config
        logger.info(f"Fleet:     {cfg.simulation.num_amrs} AMRs on "
                    f"{cfg.simulation.grid_size}x{cfg.simulation.grid_size} grid")
        logger.info(f"Tick rate: {cfg.simulation.tick_rate}s/tick")
        logger.info(f"LLM:       {'enabled' if cfg.llm.enabled else 'disabled (set GROQ_API_KEY)'}")
        logger.info("All systems GO.")

    async def run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator: received cancel.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        logger.info("Orchestrator: shutting down...")
        self._running = False
        await self.simulator.stop()
        await self.coordinator.teardown()
        await self.energy.teardown()
        await self.sentinel.teardown()
        await self.bus.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        logger.info("Orchestrator: stopped.")

    def get_status(self) -> dict:
        return {
            "running":           self._running,
            "tick":              self.simulator.tick,
            "bus_stats":         self.bus.get_stats(),
            "coordinator_stats": self.coordinator.get_stats(),
            "energy_stats":      self.energy.get_stats(),
            "sentinel_stats":    self.sentinel.get_stats(),
        }

    def get_full_config(self) -> dict:
        return self._sys_config.to_dict()