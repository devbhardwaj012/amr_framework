"""
tests/test_models.py
====================
Unit tests for core data models.

Run: python -m pytest tests/ -v

Author: AMR Multi-Agent Framework
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import (
    Position, AMRState, AMRStatus, Task, TaskType, TaskStatus,
    NetworkPacket, Event, EventType, AlertSeverity, ChargingStation
)


class TestPosition:
    def test_distance_same_point(self):
        p = Position(3.0, 4.0)
        assert p.distance_to(p) == 0.0

    def test_distance_known(self):
        p1 = Position(0, 0)
        p2 = Position(3, 4)
        assert abs(p1.distance_to(p2) - 5.0) < 1e-9

    def test_distance_symmetry(self):
        p1 = Position(1, 2)
        p2 = Position(5, 6)
        assert abs(p1.distance_to(p2) - p2.distance_to(p1)) < 1e-9

    def test_to_dict(self):
        p = Position(1.5, 2.5)
        d = p.to_dict()
        assert d == {"x": 1.5, "y": 2.5}

    def test_repr(self):
        p = Position(3.14159, 2.71828)
        r = repr(p)
        assert "3.1" in r and "2.7" in r


class TestAMRState:
    def _make_amr(self) -> AMRState:
        return AMRState(
            amr_id="AMR_01",
            name="Atlas",
            position=Position(5.0, 5.0),
        )

    def test_default_status(self):
        amr = self._make_amr()
        assert amr.status == AMRStatus.IDLE

    def test_default_battery(self):
        amr = self._make_amr()
        assert amr.battery == 100.0

    def test_to_dict_keys(self):
        amr = self._make_amr()
        d = amr.to_dict()
        required_keys = [
            "amr_id", "name", "position", "status", "battery",
            "tasks_completed", "total_distance", "is_compromised"
        ]
        for k in required_keys:
            assert k in d, f"Missing key: {k}"

    def test_to_dict_values(self):
        amr = self._make_amr()
        d   = amr.to_dict()
        assert d["amr_id"] == "AMR_01"
        assert d["name"]   == "Atlas"
        assert d["status"] == "idle"
        assert d["battery"] == 100.0

    def test_not_compromised_by_default(self):
        amr = self._make_amr()
        assert amr.is_compromised is False


class TestTask:
    def test_unique_ids(self):
        t1 = Task()
        t2 = Task()
        assert t1.task_id != t2.task_id

    def test_default_status(self):
        t = Task()
        assert t.status == TaskStatus.PENDING

    def test_to_dict(self):
        t = Task(task_type=TaskType.PICKUP, priority=3)
        d = t.to_dict()
        assert d["task_type"] == "pickup"
        assert d["priority"]  == 3
        assert d["status"]    == "pending"

    def test_with_target(self):
        pos = Position(10, 15)
        t   = Task(task_type=TaskType.NAVIGATE, target=pos)
        d   = t.to_dict()
        assert d["target"] == {"x": 10, "y": 15}


class TestNetworkPacket:
    def test_feature_vector_length(self):
        p = NetworkPacket(amr_id="AMR_01")
        fv = p.to_feature_vector()
        assert len(fv) == 4

    def test_feature_vector_types(self):
        p  = NetworkPacket(amr_id="AMR_01")
        fv = p.to_feature_vector()
        for v in fv:
            assert isinstance(v, (int, float))

    def test_normal_not_anomalous(self):
        p = NetworkPacket(amr_id="AMR_01")
        assert p.is_anomalous is False


class TestEvent:
    def test_unique_ids(self):
        e1 = Event(event_type=EventType.TICK)
        e2 = Event(event_type=EventType.TICK)
        assert e1.event_id != e2.event_id

    def test_to_dict(self):
        e = Event(
            event_type=EventType.BATTERY_LOW,
            source="EnergyAgent",
            payload={"amr_id": "AMR_01", "battery": 18.5},
            severity=AlertSeverity.WARNING,
        )
        d = e.to_dict()
        assert d["event_type"] == "battery_low"
        assert d["source"]     == "EnergyAgent"
        assert d["severity"]   == "warning"
        assert d["payload"]["battery"] == 18.5

    def test_timestamp_recent(self):
        before = time.time()
        e      = Event(event_type=EventType.TICK)
        after  = time.time()
        assert before <= e.timestamp <= after


class TestChargingStation:
    def test_not_occupied_by_default(self):
        s = ChargingStation(station_id="CS_01", position=Position(0, 0))
        assert s.is_occupied is False
        assert s.occupant_id is None

    def test_to_dict(self):
        s = ChargingStation(station_id="CS_01", position=Position(1, 1))
        d = s.to_dict()
        assert d["station_id"]  == "CS_01"
        assert d["is_occupied"] is False


class TestEnums:
    def test_amr_status_values(self):
        assert AMRStatus.IDLE.value       == "idle"
        assert AMRStatus.MOVING.value     == "moving"
        assert AMRStatus.QUARANTINED.value == "quarantined"

    def test_event_type_values(self):
        assert EventType.TICK.value             == "tick"
        assert EventType.BATTERY_LOW.value      == "battery_low"
        assert EventType.INTRUSION_ALERT.value  == "intrusion_alert"

    def test_task_type_values(self):
        assert TaskType.PICKUP.value    == "pickup"
        assert TaskType.NAVIGATE.value  == "navigate"