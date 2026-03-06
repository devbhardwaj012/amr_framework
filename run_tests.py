"""
run_tests.py
============
Standalone test runner — no pytest required.

Usage:
    python run_tests.py           # run all tests
    python run_tests.py models    # run only model tests
    python run_tests.py store     # run only state store tests
    python run_tests.py agents    # run only agent tests
    python run_tests.py config    # run only config tests

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import traceback
from typing import Callable, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Tiny test framework
# ---------------------------------------------------------------------------

_tests: List[Tuple[str, str, Callable]] = []   # (suite, name, fn)
_results: List[Tuple[str, str, bool, str]] = []


def test(suite: str, name: str):
    """Decorator to register a test function."""
    def decorator(fn):
        _tests.append((suite, name, fn))
        return fn
    return decorator


def run_all(filter_suite: str = None) -> bool:
    """Run registered tests, return True if all pass."""
    import numpy as np
    loop = asyncio.new_event_loop()

    to_run = _tests if not filter_suite else [
        t for t in _tests if filter_suite.lower() in t[0].lower()
    ]

    if not to_run:
        print(f"No tests found for filter: '{filter_suite}'")
        return False

    current_suite = None
    passed = failed = 0

    for suite, name, fn in to_run:
        if suite != current_suite:
            print(f"\n{'─'*52}")
            print(f"  {suite}")
            print(f"{'─'*52}")
            current_suite = suite

        try:
            result = fn(loop)
            print(f"  ✅  {name}")
            passed += 1
            _results.append((suite, name, True, ""))
        except AssertionError as e:
            msg = str(e) or "assertion failed"
            print(f"  ❌  {name}")
            print(f"       → {msg}")
            failed += 1
            _results.append((suite, name, False, msg))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  💥  {name}")
            print(f"       → {msg}")
            if os.getenv("VERBOSE"):
                traceback.print_exc()
            failed += 1
            _results.append((suite, name, False, msg))

    print(f"\n{'═'*52}")
    print(f"  {'ALL PASSED ✅' if failed == 0 else 'SOME FAILED ❌'}")
    print(f"  {passed} passed  |  {failed} failed  |  {passed+failed} total")
    print(f"{'═'*52}\n")
    return failed == 0


def r(loop, coro):
    """Run an async coroutine from a sync test."""
    return loop.run_until_complete(coro)


def ms(loop=None):
    """Make a fresh StateStore with a temp DB."""
    from core.state_store import StateStore
    d = tempfile.mkdtemp()
    return StateStore(db_path=os.path.join(d, "test.db"))


def ma(amr_id="A1", x=5.0, y=5.0, battery=100.0):
    """Make a fresh AMRState."""
    from core.models import AMRState, Position
    a = AMRState(amr_id=amr_id, name=amr_id, position=Position(x, y))
    a.battery = battery
    return a


# ===========================================================================
# SUITE 1: Models
# ===========================================================================

@test("Models", "Position.distance_to — 3-4-5 triangle")
def _(loop):
    from core.models import Position
    assert abs(Position(0,0).distance_to(Position(3,4)) - 5.0) < 1e-9

@test("Models", "Position.distance_to self == 0")
def _(loop):
    from core.models import Position
    p = Position(7, 3)
    assert p.distance_to(p) == 0.0

@test("Models", "Position.distance_to is symmetric")
def _(loop):
    from core.models import Position
    p1, p2 = Position(1, 2), Position(5, 6)
    assert abs(p1.distance_to(p2) - p2.distance_to(p1)) < 1e-9

@test("Models", "Position.to_dict")
def _(loop):
    from core.models import Position
    assert Position(1.5, 2.5).to_dict() == {"x": 1.5, "y": 2.5}

@test("Models", "AMRState default status = IDLE")
def _(loop):
    from core.models import AMRState, AMRStatus, Position
    a = AMRState(amr_id="A", name="A", position=Position(0, 0))
    assert a.status == AMRStatus.IDLE

@test("Models", "AMRState default battery = 100")
def _(loop):
    from core.models import AMRState, Position
    a = AMRState(amr_id="A", name="A", position=Position(0, 0))
    assert a.battery == 100.0

@test("Models", "AMRState not compromised by default")
def _(loop):
    from core.models import AMRState, Position
    a = AMRState(amr_id="A", name="A", position=Position(0, 0))
    assert a.is_compromised is False

@test("Models", "AMRState.to_dict has required keys")
def _(loop):
    from core.models import AMRState, Position
    a = AMRState(amr_id="A", name="A", position=Position(0, 0))
    d = a.to_dict()
    for k in ["amr_id", "name", "status", "battery", "is_compromised", "tasks_completed"]:
        assert k in d, f"Missing key: {k}"

@test("Models", "Task IDs are unique")
def _(loop):
    from core.models import Task
    assert Task().task_id != Task().task_id

@test("Models", "Task default status = PENDING")
def _(loop):
    from core.models import Task, TaskStatus
    assert Task().status == TaskStatus.PENDING

@test("Models", "Task.to_dict correct task_type")
def _(loop):
    from core.models import Task, TaskType
    assert Task(task_type=TaskType.PICKUP).to_dict()["task_type"] == "pickup"

@test("Models", "NetworkPacket.to_feature_vector length = 4")
def _(loop):
    from core.models import NetworkPacket
    assert len(NetworkPacket(amr_id="A").to_feature_vector()) == 4

@test("Models", "NetworkPacket not anomalous by default")
def _(loop):
    from core.models import NetworkPacket
    assert NetworkPacket(amr_id="A").is_anomalous is False

@test("Models", "Event IDs are unique")
def _(loop):
    from core.models import Event, EventType
    assert Event(event_type=EventType.TICK).event_id != Event(event_type=EventType.TICK).event_id

@test("Models", "Event timestamp is recent")
def _(loop):
    import time
    from core.models import Event, EventType
    before = time.time()
    e = Event(event_type=EventType.TICK)
    after = time.time()
    assert before <= e.timestamp <= after

@test("Models", "ChargingStation not occupied by default")
def _(loop):
    from core.models import ChargingStation, Position
    s = ChargingStation(station_id="CS1", position=Position(0, 0))
    assert s.is_occupied is False and s.occupant_id is None

@test("Models", "AMRStatus enum values")
def _(loop):
    from core.models import AMRStatus
    assert AMRStatus.IDLE.value == "idle"
    assert AMRStatus.QUARANTINED.value == "quarantined"

@test("Models", "EventType enum values")
def _(loop):
    from core.models import EventType
    assert EventType.TICK.value == "tick"
    assert EventType.INTRUSION_ALERT.value == "intrusion_alert"
    assert EventType.BATTERY_LOW.value == "battery_low"


# ===========================================================================
# SUITE 2: StateStore
# ===========================================================================

@test("StateStore", "register and retrieve AMR")
def _(loop):
    store = ms()
    store.register_amr(ma())
    assert store.get_amr("A1") is not None

@test("StateStore", "get unknown AMR returns None")
def _(loop):
    assert ms().get_amr("GHOST") is None

@test("StateStore", "get_all_amrs returns all registered")
def _(loop):
    store = ms()
    store.register_amr(ma("A1"))
    store.register_amr(ma("A2"))
    assert len(store.get_all_amrs()) == 2

@test("StateStore", "update_amr_battery")
def _(loop):
    store = ms()
    store.register_amr(ma())
    r(loop, store.update_amr_battery("A1", 42.5))
    assert store.get_amr("A1").battery == 42.5

@test("StateStore", "battery clamps to 100 max")
def _(loop):
    store = ms()
    store.register_amr(ma())
    r(loop, store.update_amr_battery("A1", 200.0))
    assert store.get_amr("A1").battery == 100.0

@test("StateStore", "battery clamps to 0 min")
def _(loop):
    store = ms()
    store.register_amr(ma())
    r(loop, store.update_amr_battery("A1", -10.0))
    assert store.get_amr("A1").battery == 0.0

@test("StateStore", "update_amr_status")
def _(loop):
    from core.models import AMRStatus
    store = ms()
    store.register_amr(ma())
    r(loop, store.update_amr_status("A1", AMRStatus.MOVING))
    assert store.get_amr("A1").status == AMRStatus.MOVING

@test("StateStore", "update_amr_position tracks distance")
def _(loop):
    from core.models import Position
    store = ms()
    store.register_amr(ma(x=0.0, y=0.0))
    r(loop, store.update_amr_position("A1", Position(3.0, 4.0)))
    amr = store.get_amr("A1")
    assert abs(amr.total_distance - 5.0) < 0.001
    assert amr.position.x == 3.0 and amr.position.y == 4.0

@test("StateStore", "set_amr_compromised marks quarantined")
def _(loop):
    from core.models import AMRStatus
    store = ms()
    store.register_amr(ma())
    r(loop, store.set_amr_compromised("A1", True))
    amr = store.get_amr("A1")
    assert amr.is_compromised is True
    assert amr.status == AMRStatus.QUARANTINED
    assert amr.alerts_triggered == 1

@test("StateStore", "increment_tasks_completed")
def _(loop):
    store = ms()
    store.register_amr(ma())
    r(loop, store.increment_tasks_completed("A1"))
    r(loop, store.increment_tasks_completed("A1"))
    assert store.get_amr("A1").tasks_completed == 2

@test("StateStore", "update unknown AMR does not raise")
def _(loop):
    r(loop, ms().update_amr_battery("GHOST", 50.0))  # Must not raise

@test("StateStore", "charging station register and retrieve")
def _(loop):
    from core.models import ChargingStation, Position
    store = ms()
    store.register_station(ChargingStation(station_id="CS1", position=Position(0, 0)))
    assert len(store.get_all_stations()) == 1

@test("StateStore", "assign and release charging station")
def _(loop):
    from core.models import ChargingStation, Position
    store = ms()
    store.register_station(ChargingStation(station_id="CS1", position=Position(0, 0)))
    r(loop, store.assign_station("CS1", "A1"))
    s = store.get_all_stations()[0]
    assert s.is_occupied is True and s.occupant_id == "A1"
    r(loop, store.release_station("CS1"))
    assert s.is_occupied is False and s.occupant_id is None

@test("StateStore", "get_free_station skips occupied")
def _(loop):
    from core.models import ChargingStation, Position
    store = ms()
    store.register_station(ChargingStation(station_id="CS1", position=Position(0, 0)))
    store.register_station(ChargingStation(station_id="CS2", position=Position(5, 5)))
    r(loop, store.assign_station("CS1", "A1"))
    assert store.get_free_station().station_id == "CS2"

@test("StateStore", "get_free_station returns None when all occupied")
def _(loop):
    from core.models import ChargingStation, Position
    store = ms()
    store.register_station(ChargingStation(station_id="CS1", position=Position(0, 0)))
    r(loop, store.assign_station("CS1", "A1"))
    assert store.get_free_station() is None

@test("StateStore", "get_nearest_free_station")
def _(loop):
    from core.models import ChargingStation, Position
    store = ms()
    store.register_station(ChargingStation(station_id="CS1", position=Position(0, 0)))
    store.register_station(ChargingStation(station_id="CS2", position=Position(18, 18)))
    assert store.get_nearest_free_station(Position(1, 1)).station_id == "CS1"

@test("StateStore", "add_task and get_pending_tasks")
def _(loop):
    from core.models import Task, TaskType
    store = ms()
    task = Task(task_type=TaskType.PICKUP)
    r(loop, store.add_task(task))
    pending = r(loop, store.get_pending_tasks())
    assert len(pending) == 1 and pending[0].task_id == task.task_id

@test("StateStore", "update_task_status removes from pending")
def _(loop):
    from core.models import Task, TaskStatus
    store = ms()
    task = Task()
    r(loop, store.add_task(task))
    r(loop, store.update_task_status(task.task_id, TaskStatus.ASSIGNED, "A1"))
    assert len(r(loop, store.get_pending_tasks())) == 0

@test("StateStore", "snapshot has correct tick")
def _(loop):
    store = ms()
    store.register_amr(ma())
    assert store.get_snapshot(tick=42).tick == 42

@test("StateStore", "snapshot average battery")
def _(loop):
    store = ms()
    store.register_amr(ma("A1"))
    store.register_amr(ma("A2"))
    r(loop, store.update_amr_battery("A1", 60.0))
    r(loop, store.update_amr_battery("A2", 40.0))
    snap = store.get_snapshot(tick=1)
    assert abs(snap.fleet_avg_battery - 50.0) < 0.1


# ===========================================================================
# SUITE 3: EventBus
# ===========================================================================

@test("EventBus", "subscribe and receive event")
def _(loop):
    from core.event_bus import EventBus
    from core.models import Event, EventType
    bus = EventBus()
    received = []
    async def handler(e): received.append(e)
    bus.subscribe(EventType.TICK, handler)
    async def _run():
        task = asyncio.create_task(bus.run())
        await bus.publish(Event(event_type=EventType.TICK))
        await asyncio.sleep(0.05)
        await bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    assert len(received) == 1

@test("EventBus", "wildcard subscriber receives all events")
def _(loop):
    from core.event_bus import EventBus
    from core.models import Event, EventType
    bus = EventBus()
    received = []
    async def handler(e): received.append(e.event_type)
    bus.subscribe_all(handler)
    async def _run():
        task = asyncio.create_task(bus.run())
        await bus.publish(Event(event_type=EventType.TICK))
        await bus.publish(Event(event_type=EventType.BATTERY_LOW))
        await asyncio.sleep(0.05)
        await bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    assert EventType.TICK in received and EventType.BATTERY_LOW in received

@test("EventBus", "unsubscribe stops delivery")
def _(loop):
    from core.event_bus import EventBus
    from core.models import Event, EventType
    bus = EventBus()
    received = []
    async def handler(e): received.append(e)
    bus.subscribe(EventType.TICK, handler)
    bus.unsubscribe(EventType.TICK, handler)
    async def _run():
        task = asyncio.create_task(bus.run())
        await bus.publish(Event(event_type=EventType.TICK))
        await asyncio.sleep(0.05)
        await bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    assert len(received) == 0

@test("EventBus", "broken handler does not crash bus")
def _(loop):
    from core.event_bus import EventBus
    from core.models import Event, EventType
    bus = EventBus()
    good = []
    async def bad(e): raise ValueError("boom")
    async def good_h(e): good.append(e)
    bus.subscribe(EventType.TICK, bad)
    bus.subscribe(EventType.TICK, good_h)
    async def _run():
        task = asyncio.create_task(bus.run())
        await bus.publish(Event(event_type=EventType.TICK))
        await asyncio.sleep(0.05)
        await bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    assert len(good) == 1
    assert bus.get_stats().get("handler_errors", 0) >= 1

@test("EventBus", "stats.published increments")
def _(loop):
    from core.event_bus import EventBus
    from core.models import Event, EventType
    bus = EventBus()
    async def _run():
        task = asyncio.create_task(bus.run())
        await bus.publish(Event(event_type=EventType.TICK))
        await asyncio.sleep(0.05)
        await bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    assert bus.get_stats()["published"] >= 1

@test("EventBus", "log stores entry in history")
def _(loop):
    from core.event_bus import EventBus
    from core.models import AlertSeverity
    bus = EventBus()
    async def _run():
        task = asyncio.create_task(bus.run())
        await bus.log("TestAgent", "hello test", AlertSeverity.INFO)
        await asyncio.sleep(0.05)
        await bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    logs = bus.get_agent_logs(10)
    assert any(l.message == "hello test" for l in logs)


# ===========================================================================
# SUITE 4: Energy Agent & Q-Learning
# ===========================================================================

@test("EnergyAgent", "_battery_bin: critical (≤10)")
def _(loop):
    from agents.energy_agent import _battery_bin
    assert _battery_bin(0) == 0 and _battery_bin(5) == 0 and _battery_bin(10) == 0

@test("EnergyAgent", "_battery_bin: low (10-25)")
def _(loop):
    from agents.energy_agent import _battery_bin
    assert _battery_bin(11) == 1 and _battery_bin(15) == 1 and _battery_bin(25) == 1

@test("EnergyAgent", "_battery_bin: medium (25-60)")
def _(loop):
    from agents.energy_agent import _battery_bin
    assert _battery_bin(26) == 2 and _battery_bin(50) == 2 and _battery_bin(60) == 2

@test("EnergyAgent", "_battery_bin: high (>60)")
def _(loop):
    from agents.energy_agent import _battery_bin
    assert _battery_bin(61) == 3 and _battery_bin(90) == 3 and _battery_bin(100) == 3

@test("EnergyAgent", "_dist_bin correct bins")
def _(loop):
    from agents.energy_agent import _dist_bin
    assert _dist_bin(1) == 0 and _dist_bin(8) == 1 and _dist_bin(15) == 2

@test("EnergyAgent", "_queue_bin correct bins")
def _(loop):
    from agents.energy_agent import _queue_bin
    assert _queue_bin(0) == 0 and _queue_bin(1) == 1 and _queue_bin(10) == 1

@test("EnergyAgent", "Q-table shape is (4, 3, 2, 3)")
def _(loop):
    from core.event_bus import EventBus
    from agents.energy_agent import EnergyAgent
    agent = EnergyAgent(EventBus(), ms(), {})
    assert agent.q_table.shape == (4, 3, 2, 3)

@test("EnergyAgent", "Q-table seeded: critical battery → GO_CHARGE")
def _(loop):
    import numpy as np
    from core.event_bus import EventBus
    from agents.energy_agent import EnergyAgent, ACTION_GO_CHARGE
    agent = EnergyAgent(EventBus(), ms(), {})
    for d in range(3):
        for q in range(2):
            best = int(np.argmax(agent.q_table[0, d, q]))
            assert best == ACTION_GO_CHARGE, f"d={d},q={q}: got {best}"

@test("EnergyAgent", "Q-table seeded: high battery → CONTINUE")
def _(loop):
    import numpy as np
    from core.event_bus import EventBus
    from agents.energy_agent import EnergyAgent, ACTION_CONTINUE
    agent = EnergyAgent(EventBus(), ms(), {})
    for d in range(3):
        for q in range(2):
            best = int(np.argmax(agent.q_table[3, d, q]))
            assert best == ACTION_CONTINUE, f"d={d},q={q}: got {best}"

@test("EnergyAgent", "Bellman update changes Q-value")
def _(loop):
    from core.event_bus import EventBus
    from agents.energy_agent import EnergyAgent, ACTION_GO_CHARGE
    agent = EnergyAgent(EventBus(), ms(), {})
    before = float(agent.q_table[1, 1, 0, ACTION_GO_CHARGE])
    r(loop, agent._update_qtable("X", (1, 1, 0), ACTION_GO_CHARGE, reward=10.0))
    assert float(agent.q_table[1, 1, 0, ACTION_GO_CHARGE]) != before

@test("EnergyAgent", "get_qtable_stats has required keys")
def _(loop):
    from core.event_bus import EventBus
    from agents.energy_agent import EnergyAgent
    stats = EnergyAgent(EventBus(), ms(), {}).get_qtable_stats()
    for k in ["total_reward", "charge_events", "prevented_depletions", "avg_q_value"]:
        assert k in stats, f"Missing: {k}"

@test("EnergyAgent", "get_qtable_heatmap_data shape")
def _(loop):
    from core.event_bus import EventBus
    from agents.energy_agent import EnergyAgent
    data = EnergyAgent(EventBus(), ms(), {}).get_qtable_heatmap_data()
    assert len(data["q_values"]) == 4
    assert len(data["battery_labels"]) == 4
    assert len(data["dist_labels"]) == 3


# ===========================================================================
# SUITE 5: Coordinator Agent
# ===========================================================================

@test("CoordinatorAgent", "closer AMR scores higher for task")
def _(loop):
    from core.models import AMRState, Position, Task
    from core.event_bus import EventBus
    from agents.coordinator_agent import CoordinatorAgent
    coord = CoordinatorAgent(EventBus(), ms(), {"energy_critical_threshold": 10})
    close = ma("C", x=2.0, y=2.0, battery=80)
    far   = ma("F", x=18.0, y=18.0, battery=80)
    task  = Task(target=Position(2.5, 2.5))
    assert coord._score_amr_for_task(close, task) > coord._score_amr_for_task(far, task)

@test("CoordinatorAgent", "higher battery scores higher (equidistant)")
def _(loop):
    from core.models import Task, Position
    from core.event_bus import EventBus
    from agents.coordinator_agent import CoordinatorAgent
    coord = CoordinatorAgent(EventBus(), ms(), {"energy_critical_threshold": 10})
    hi = ma("H", x=5.0, y=5.0, battery=90)
    lo = ma("L", x=5.0, y=5.0, battery=20)
    task = Task(target=Position(5.0, 5.0))
    assert coord._score_amr_for_task(hi, task) > coord._score_amr_for_task(lo, task)

@test("CoordinatorAgent", "idle AMR gets bonus over moving")
def _(loop):
    from core.models import AMRStatus, Task, Position
    from core.event_bus import EventBus
    from agents.coordinator_agent import CoordinatorAgent
    coord = CoordinatorAgent(EventBus(), ms(), {"energy_critical_threshold": 10})
    idle   = ma("I", x=5.0, y=5.0, battery=80); idle.status = AMRStatus.IDLE
    moving = ma("M", x=5.0, y=5.0, battery=80); moving.status = AMRStatus.MOVING
    task = Task(target=Position(5.0, 5.0))
    assert coord._score_amr_for_task(idle, task) > coord._score_amr_for_task(moving, task)

@test("CoordinatorAgent", "_assign_task updates store")
def _(loop):
    from core.models import Task, TaskType, TaskStatus, Position
    from core.event_bus import EventBus
    from agents.coordinator_agent import CoordinatorAgent
    store = ms()
    store.register_amr(ma("AMR_01"))
    coord = CoordinatorAgent(EventBus(), store, {"energy_critical_threshold": 10})
    task = Task(task_type=TaskType.PICKUP, target=Position(8, 8))
    r(loop, store.add_task(task))
    amr = store.get_amr("AMR_01")

    async def _run():
        bus_task = asyncio.create_task(coord.bus.run())
        await coord._assign_task(amr, task)
        await asyncio.sleep(0.05)
        await coord.bus.stop()
        bus_task.cancel()
    loop.run_until_complete(_run())

    updated = store.get_all_tasks()[0]
    assert updated.status == TaskStatus.ASSIGNED
    assert updated.assigned_to == "AMR_01"


# ===========================================================================
# SUITE 6: Sentinel Agent
# ===========================================================================

@test("SentinelAgent", "generates normal packet (not anomalous)")
def _(loop):
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    sent = SentinelAgent(EventBus(), ms(), {})
    amr  = ma()
    pkt  = sent._generate_packet(amr, tick=1)
    assert pkt.amr_id == "A1"
    assert not pkt.is_anomalous
    assert pkt.packet_size > 0 and pkt.latency_ms > 0

@test("SentinelAgent", "attack packet is_anomalous=True")
def _(loop):
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    sent = SentinelAgent(EventBus(), ms(), {})
    atk  = {"type": "PACKET_FLOOD", "remaining": 5, "start_tick": 0, "duration": 5}
    pkt  = sent._generate_attack_packet("A1", atk, 512, 5.0, -70.0, tick=1)
    assert pkt.is_anomalous is True

@test("SentinelAgent", "model trains after MIN_SAMPLES packets")
def _(loop):
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    sent = SentinelAgent(EventBus(), ms(), {})
    amr  = ma("AMR_01")
    for i in range(sent.MIN_SAMPLES + 10):
        sent._packet_buffers["AMR_01"].append(sent._generate_packet(amr, tick=i))
    r(loop, sent._train_model("AMR_01"))
    assert sent._model_trained["AMR_01"] is True
    assert "AMR_01" in sent._models

@test("SentinelAgent", "PACKET_FLOOD detected (scores more negative)")
def _(loop):
    import numpy as np
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    sent = SentinelAgent(EventBus(), ms(), {})
    amr  = ma("AMR_01")
    for i in range(80): sent._packet_buffers["AMR_01"].append(sent._generate_packet(amr, i))
    r(loop, sent._train_model("AMR_01"))
    normal_mean = np.mean([sent._score_packet("AMR_01", sent._generate_packet(amr, i)) for i in range(15)])
    atk  = {"type": "PACKET_FLOOD", "remaining": 5, "start_tick": 0, "duration": 5}
    atk_score = sent._score_packet("AMR_01", sent._generate_attack_packet("AMR_01", dict(atk), 512, 5.0, -70.0, 1))
    assert atk_score < normal_mean, f"FLOOD {atk_score:.3f} >= normal {normal_mean:.3f}"

@test("SentinelAgent", "JAMMING detected (scores more negative)")
def _(loop):
    import numpy as np
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    sent = SentinelAgent(EventBus(), ms(), {})
    amr  = ma("AMR_01")
    for i in range(80): sent._packet_buffers["AMR_01"].append(sent._generate_packet(amr, i))
    r(loop, sent._train_model("AMR_01"))
    normal_mean = np.mean([sent._score_packet("AMR_01", sent._generate_packet(amr, i)) for i in range(15)])
    atk  = {"type": "JAMMING", "remaining": 5, "start_tick": 0, "duration": 5}
    atk_score = sent._score_packet("AMR_01", sent._generate_attack_packet("AMR_01", dict(atk), 512, 5.0, -70.0, 1))
    assert atk_score < normal_mean

@test("SentinelAgent", "DATA_EXFIL detected (scores more negative)")
def _(loop):
    import numpy as np
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    sent = SentinelAgent(EventBus(), ms(), {})
    amr  = ma("AMR_01")
    for i in range(80): sent._packet_buffers["AMR_01"].append(sent._generate_packet(amr, i))
    r(loop, sent._train_model("AMR_01"))
    normal_mean = np.mean([sent._score_packet("AMR_01", sent._generate_packet(amr, i)) for i in range(15)])
    atk  = {"type": "DATA_EXFIL", "remaining": 5, "start_tick": 0, "duration": 5}
    atk_score = sent._score_packet("AMR_01", sent._generate_attack_packet("AMR_01", dict(atk), 512, 5.0, -70.0, 1))
    assert atk_score < normal_mean

@test("SentinelAgent", "quarantine marks AMR compromised in store")
def _(loop):
    from core.event_bus import EventBus
    from core.models import NetworkPacket
    from agents.sentinel_agent import SentinelAgent
    store = ms()
    store.register_amr(ma("AMR_01"))
    sent = SentinelAgent(EventBus(), store, {})
    amr  = store.get_amr("AMR_01")
    pkt  = NetworkPacket(amr_id="AMR_01", is_anomalous=True)
    async def _run():
        task = asyncio.create_task(sent.bus.run())
        await sent._quarantine_amr(amr, pkt, -0.7, "PACKET_FLOOD", tick=1)
        await asyncio.sleep(0.05)
        await sent.bus.stop()
        task.cancel()
    loop.run_until_complete(_run())
    assert store.get_amr("AMR_01").is_compromised is True
    assert sent._quarantines_issued == 1

@test("SentinelAgent", "get_security_stats has required keys")
def _(loop):
    from core.event_bus import EventBus
    from agents.sentinel_agent import SentinelAgent
    stats = SentinelAgent(EventBus(), ms(), {}).get_security_stats()
    for k in ["packets_analyzed", "anomalies_detected", "quarantines_issued", "models_trained"]:
        assert k in stats, f"Missing: {k}"


# ===========================================================================
# SUITE 7: Config Manager
# ===========================================================================

@test("ConfigManager", "loads with defaults")
def _(loop):
    from config.config_manager import ConfigManager
    cfg = ConfigManager().load()
    assert cfg.simulation.num_amrs  == 5
    assert cfg.simulation.grid_size == 20
    assert cfg.energy.low_threshold == 20.0

@test("ConfigManager", "override num_amrs")
def _(loop):
    from config.config_manager import ConfigManager
    cfg = ConfigManager().load(overrides={"num_amrs": 3})
    assert cfg.simulation.num_amrs == 3

@test("ConfigManager", "override grid_size")
def _(loop):
    from config.config_manager import ConfigManager
    cfg = ConfigManager().load(overrides={"grid_size": 15})
    assert cfg.simulation.grid_size == 15

@test("ConfigManager", "to_flat_dict has required keys")
def _(loop):
    from config.config_manager import ConfigManager
    flat = ConfigManager().load().to_flat_dict()
    for k in ["num_amrs", "grid_size", "tick_rate", "energy_low_threshold", "llm_model"]:
        assert k in flat, f"Missing: {k}"

@test("ConfigManager", "validation rejects num_amrs=0")
def _(loop):
    from config.config_manager import ConfigManager
    try:
        ConfigManager().load(overrides={"num_amrs": 0})
        assert False, "Should have raised"
    except AssertionError:
        pass

@test("ConfigManager", "validation rejects grid_size=5")
def _(loop):
    from config.config_manager import ConfigManager
    try:
        ConfigManager().load(overrides={"grid_size": 5})
        assert False, "Should have raised"
    except AssertionError:
        pass

@test("ConfigManager", "critical_threshold < low_threshold enforced")
def _(loop):
    from config.config_manager import ConfigManager
    try:
        ConfigManager().load(overrides={"energy_critical_threshold": 30})  # > low=20
        assert False, "Should have raised"
    except AssertionError:
        pass


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    filter_suite = sys.argv[1] if len(sys.argv) > 1 else None

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     AMR Multi-Agent Framework — Test Suite          ║")
    print("╚══════════════════════════════════════════════════════╝")

    start = time.time()
    success = run_all(filter_suite)
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.2f}s")
    sys.exit(0 if success else 1)