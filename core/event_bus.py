"""
core/event_bus.py
=================
Asynchronous publish/subscribe event bus.

This is the ONLY way agents communicate. No agent ever calls another agent
directly. This design means:
  - Adding a new agent = just subscribe to relevant events
  - Removing an agent = unsubscribe, nothing else breaks
  - Testing an agent = inject events, observe published events

Pattern: Agents subscribe to EventTypes. When an event is published,
all subscribers for that type receive a copy asynchronously.

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Coroutine, Dict, List, Optional

from core.models import Event, EventType, AgentLog, AlertSeverity

logger = logging.getLogger(__name__)


# Type alias for async event handler callbacks
EventHandler = Callable[[Event], Coroutine]


class EventBus:
    """
    Central async publish/subscribe message bus.

    Usage:
        bus = EventBus()

        # Subscribe (in agent __init__ or setup)
        bus.subscribe(EventType.BATTERY_LOW, my_handler)

        # Publish (from anywhere)
        await bus.publish(Event(event_type=EventType.BATTERY_LOW, ...))

        # Start processing
        await bus.run()
    """

    def __init__(self, max_queue_size: int = 1000):
        # Map of EventType -> list of async handlers
        self._subscribers: Dict[EventType, List[EventHandler]] = defaultdict(list)

        # Wildcard subscribers receive ALL events (used by logger, dashboard)
        self._wildcard_subscribers: List[EventHandler] = []

        # Internal async queue
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)

        # History of recent events (capped) for dashboard display
        self._event_history: List[Event] = []
        self._max_history: int = 500

        # Agent log history
        self._agent_logs: List[AgentLog] = []
        self._max_logs: int = 200

        self._running: bool = False
        self._stats: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Subscription API
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe an async handler to a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"[EventBus] Subscribed {handler.__qualname__} to {event_type.value}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to ALL events (useful for logging and monitoring)."""
        self._wildcard_subscribers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a subscription."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    # ------------------------------------------------------------------
    # Publishing API
    # ------------------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        Non-blocking — event is queued for async dispatch.
        """
        try:
            await self._queue.put(event)
            self._stats["published"] += 1
        except asyncio.QueueFull:
            logger.warning(f"[EventBus] Queue full! Dropping event: {event.event_type.value}")
            self._stats["dropped"] += 1

    def publish_sync(self, event: Event) -> None:
        """
        Synchronous publish (use from non-async contexts only).
        Creates a task on the running event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.publish(event))
            else:
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            # No event loop running — put directly (simulation mode)
            self._queue.put_nowait(event)

    # ------------------------------------------------------------------
    # Log helper — agents use this instead of print()
    # ------------------------------------------------------------------

    async def log(
        self,
        agent_name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: dict = None,
    ) -> None:
        """Convenience: publish a log event AND store in log history."""
        log_entry = AgentLog(
            agent_name=agent_name,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )
        self._agent_logs.append(log_entry)
        if len(self._agent_logs) > self._max_logs:
            self._agent_logs = self._agent_logs[-self._max_logs:]

        # Also publish as an event so dashboard can reactively update
        await self.publish(Event(
            event_type=EventType.AGENT_LOG,
            source=agent_name,
            payload=log_entry.to_dict(),
            severity=severity,
        ))

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def _dispatch(self, event: Event) -> None:
        """Dispatch one event to all relevant subscribers."""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Gather all handlers: specific + wildcard
        handlers = (
            self._subscribers.get(event.event_type, [])
            + self._wildcard_subscribers
        )

        if not handlers:
            return

        # Run all handlers concurrently
        tasks = [asyncio.create_task(h(event)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any handler errors without crashing the bus
        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.error(
                    f"[EventBus] Handler {handler.__qualname__} raised: {result}",
                    exc_info=result,
                )
                self._stats["handler_errors"] += 1

        self._stats["dispatched"] += 1

    async def run(self) -> None:
        """
        Main processing loop. Run this as an asyncio task.
        Processes events from the queue one by one.
        """
        self._running = True
        logger.info("[EventBus] Started.")

        while self._running:
            try:
                # Wait for next event with timeout so we can check _running
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue  # No event, loop again
            except asyncio.CancelledError:
                logger.info("[EventBus] Cancelled.")
                break
            except Exception as e:
                logger.error(f"[EventBus] Unexpected error: {e}", exc_info=True)

    async def stop(self) -> None:
        """Gracefully stop the bus after draining the queue."""
        logger.info("[EventBus] Stopping...")
        self._running = False
        # Drain remaining items
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Read-only accessors (for dashboard)
    # ------------------------------------------------------------------

    def get_recent_events(self, n: int = 50, event_type: Optional[EventType] = None) -> List[Event]:
        """Get the N most recent events, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-n:]

    def get_agent_logs(self, n: int = 50, agent_name: Optional[str] = None) -> List[AgentLog]:
        """Get recent agent logs, optionally filtered by agent."""
        logs = self._agent_logs
        if agent_name:
            logs = [l for l in logs if l.agent_name == agent_name]
        return logs[-n:]

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)