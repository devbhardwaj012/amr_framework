"""
core/base_agent.py
==================
Abstract base class for all agents in the framework.

Every agent (Coordinator, Energy, Sentinel) inherits from BaseAgent.
This enforces a consistent interface and makes adding new agents trivial:
  1. Create a new file in /agents/
  2. Subclass BaseAgent
  3. Override setup() and the handler methods
  4. Register in the orchestrator

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional

from core.models import (
    AlertSeverity, Event, EventType, SystemSnapshot
)

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state_store import StateStore

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Lifecycle:
        1. __init__()  — store references, set name
        2. setup()     — subscribe to events, init ML models, etc.
        3. run()       — optional background loop (override if needed)
        4. teardown()  — cleanup on shutdown

    Every agent communicates ONLY via the event bus.
    Agents access AMR state ONLY via the state store.
    """

    def __init__(
        self,
        name: str,
        bus: "EventBus",
        store: "StateStore",
        config: Dict[str, Any] = None,
    ):
        self.name    = name
        self.bus     = bus
        self.store   = store
        self.config  = config or {}
        self.logger  = logging.getLogger(f"agent.{name}")

        self._running:    bool  = False
        self._tick_count: int   = 0
        self._started_at: float = 0.0

        # Stats tracked for every agent
        self._events_received: int = 0
        self._events_published: int = 0
        self._decisions_made: int = 0

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every agent
    # ------------------------------------------------------------------

    @abstractmethod
    async def setup(self) -> None:
        """
        Called once at startup.
        Subscribe to events here. Initialize models here.
        DO NOT do heavy work in __init__ — do it here.
        """
        ...

    @abstractmethod
    async def on_tick(self, snapshot: SystemSnapshot) -> None:
        """
        Called on every simulation tick with the latest system snapshot.
        This is where most agent logic lives.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle methods (can be overridden)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Optional background loop for agents that need periodic work
        beyond tick-driven logic. Override if needed.
        Default: no-op.
        """
        pass

    async def teardown(self) -> None:
        """Called on graceful shutdown. Override to clean up resources."""
        self._running = False
        self.logger.info(f"[{self.name}] Torn down.")

    # ------------------------------------------------------------------
    # Helper methods available to all agents
    # ------------------------------------------------------------------

    async def emit(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        severity: AlertSeverity = AlertSeverity.INFO,
    ) -> None:
        """Publish an event to the bus. Tracks stats automatically."""
        event = Event(
            event_type=event_type,
            source=self.name,
            payload=payload,
            severity=severity,
        )
        await self.bus.publish(event)
        self._events_published += 1

    async def log(
        self,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Log a message via the bus (shows in dashboard)."""
        await self.bus.log(self.name, message, severity, metadata)

        # Also log to Python logger for file logging
        level_map = {
            AlertSeverity.INFO:     logging.INFO,
            AlertSeverity.WARNING:  logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
        self.logger.log(level_map[severity], message)

    def get_stats(self) -> Dict[str, Any]:
        """Return agent performance stats for the dashboard."""
        uptime = time.time() - self._started_at if self._started_at else 0
        return {
            "agent_name":        self.name,
            "events_received":   self._events_received,
            "events_published":  self._events_published,
            "decisions_made":    self._decisions_made,
            "uptime_seconds":    round(uptime, 1),
            "ticks_processed":   self._tick_count,
        }

    async def _handle_tick(self, event: Event) -> None:
        """
        Internal handler subscribed to TICK events.
        Extracts snapshot and calls on_tick().
        """
        self._events_received += 1
        self._tick_count += 1
        snapshot = self.store.get_snapshot(event.payload.get("tick", 0))
        await self.on_tick(snapshot)

    async def start(self) -> None:
        """Called by orchestrator to activate the agent."""
        self._running    = True
        self._started_at = time.time()
        await self.setup()
        # Subscribe to tick events
        self.bus.subscribe(EventType.TICK, self._handle_tick)
        self.logger.info(f"[{self.name}] Started and listening.")