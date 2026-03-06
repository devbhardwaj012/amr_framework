"""
tests/test_state_store.py
=========================
Unit tests for the state store.

Author: AMR Multi-Agent Framework
"""

import sys
import os
import asyncio
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state_store import StateStore
from core.models import (
    AMRState, AMRStatus, ChargingStation, Position, Task, TaskStatus, TaskType
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_store() -> StateStore:
    """Create a fresh store with a temp DB."""
    tmpdir = tempfile.mkdtemp()
    return StateStore(db_path=os.path.join(tmpdir, "test.db"))


def make_amr(amr_id="AMR_01", name="Atlas", x=5.0, y=5.0) -> AMRState:
    return AMRState(amr_id=amr_id, name=name, position=Position(x, y))


def make_station(station_id="CS_01", x=0.0, y=0.0) -> ChargingStation:
    return ChargingStation(station_id=station_id, position=Position(x, y))


class TestStateStoreAMR:

    def test_register_and_retrieve(self):
        store = make_store()
        amr   = make_amr()
        store.register_amr(amr)
        assert store.get_amr("AMR_01") is amr

    def test_get_unknown_amr_returns_none(self):
        store = make_store()
        assert store.get_amr("NONEXISTENT") is None

    def test_get_all_amrs(self):
        store = make_store()
        store.register_amr(make_amr("AMR_01"))
        store.register_amr(make_amr("AMR_02", "Bolt"))
        assert len(store.get_all_amrs()) == 2

    def test_update_battery(self):
        store = make_store()
        store.register_amr(make_amr())
        run(store.update_amr_battery("AMR_01", 42.5))
        assert store.get_amr("AMR_01").battery == 42.5

    def test_battery_clamps_to_0_100(self):
        store = make_store()
        store.register_amr(make_amr())
        run(store.update_amr_battery("AMR_01", 150.0))
        assert store.get_amr("AMR_01").battery == 100.0
        run(store.update_amr_battery("AMR_01", -10.0))
        assert store.get_amr("AMR_01").battery == 0.0

    def test_update_status(self):
        store = make_store()
        store.register_amr(make_amr())
        run(store.update_amr_status("AMR_01", AMRStatus.MOVING))
        assert store.get_amr("AMR_01").status == AMRStatus.MOVING

    def test_update_position_tracks_distance(self):
        store = make_store()
        store.register_amr(make_amr(x=0.0, y=0.0))
        run(store.update_amr_position("AMR_01", Position(3.0, 4.0)))
        amr = store.get_amr("AMR_01")
        assert abs(amr.total_distance - 5.0) < 0.001
        assert amr.position.x == 3.0
        assert amr.position.y == 4.0

    def test_set_amr_compromised(self):
        store = make_store()
        store.register_amr(make_amr())
        run(store.set_amr_compromised("AMR_01", True))
        amr = store.get_amr("AMR_01")
        assert amr.is_compromised is True
        assert amr.status == AMRStatus.QUARANTINED
        assert amr.alerts_triggered == 1

    def test_increment_tasks_completed(self):
        store = make_store()
        store.register_amr(make_amr())
        run(store.increment_tasks_completed("AMR_01"))
        run(store.increment_tasks_completed("AMR_01"))
        assert store.get_amr("AMR_01").tasks_completed == 2

    def test_update_ignores_unknown_amr(self):
        """Updating a non-existent AMR should not raise."""
        store = make_store()
        run(store.update_amr_battery("GHOST", 50.0))  # Should not raise


class TestStateStoreChargingStation:

    def test_register_station(self):
        store   = make_store()
        station = make_station()
        store.register_station(station)
        assert station in store.get_all_stations()

    def test_assign_and_release_station(self):
        store = make_store()
        store.register_station(make_station())

        run(store.assign_station("CS_01", "AMR_01"))
        station = store.get_all_stations()[0]
        assert station.is_occupied is True
        assert station.occupant_id == "AMR_01"

        run(store.release_station("CS_01"))
        assert station.is_occupied is False
        assert station.occupant_id is None

    def test_get_free_station(self):
        store = make_store()
        store.register_station(make_station("CS_01", 0, 0))
        store.register_station(make_station("CS_02", 5, 5))

        run(store.assign_station("CS_01", "AMR_01"))
        free = store.get_free_station()
        assert free is not None
        assert free.station_id == "CS_02"

    def test_get_free_station_none_when_all_occupied(self):
        store = make_store()
        store.register_station(make_station())
        run(store.assign_station("CS_01", "AMR_01"))
        assert store.get_free_station() is None

    def test_get_nearest_free_station(self):
        store = make_store()
        store.register_station(make_station("CS_01", 0, 0))
        store.register_station(make_station("CS_02", 18, 18))
        # AMR at (1,1) — CS_01 is closer
        nearest = store.get_nearest_free_station(Position(1, 1))
        assert nearest.station_id == "CS_01"


class TestStateStoreTasks:

    def test_add_and_get_pending(self):
        store = make_store()
        task  = Task(task_type=TaskType.PICKUP)
        run(store.add_task(task))

        pending = run(store.get_pending_tasks())
        assert len(pending) == 1
        assert pending[0].task_id == task.task_id

    def test_update_task_status(self):
        store = make_store()
        task  = Task()
        run(store.add_task(task))
        run(store.update_task_status(task.task_id, TaskStatus.ASSIGNED, "AMR_01"))

        # Should no longer appear in pending
        pending = run(store.get_pending_tasks())
        assert all(t.task_id != task.task_id for t in pending)

    def test_multiple_pending_tasks(self):
        store = make_store()
        for _ in range(5):
            run(store.add_task(Task()))
        pending = run(store.get_pending_tasks())
        assert len(pending) == 5


class TestStateStoreSnapshot:

    def test_snapshot_structure(self):
        store = make_store()
        store.register_amr(make_amr("AMR_01"))
        store.register_amr(make_amr("AMR_02", "Bolt"))
        store.register_station(make_station())

        snap = store.get_snapshot(tick=42)
        assert snap.tick == 42
        assert "AMR_01" in snap.amrs
        assert "AMR_02" in snap.amrs
        assert snap.fleet_avg_battery == 100.0
        assert snap.active_amrs == 2

    def test_snapshot_battery_average(self):
        store = make_store()
        store.register_amr(make_amr("A1"))
        store.register_amr(make_amr("A2"))
        run(store.update_amr_battery("A1", 60.0))
        run(store.update_amr_battery("A2", 40.0))

        snap = store.get_snapshot(tick=1)
        assert abs(snap.fleet_avg_battery - 50.0) < 0.1