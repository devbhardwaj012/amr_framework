"""
tests/test_agents.py
====================
Integration-level tests for the 3 agents.

These tests spin up a real EventBus + StateStore and verify agent
behavior by:
  1. Setting up a scenario (AMR states, tasks)
  2. Publishing events
  3. Asserting on state changes and published events

Author: AMR Multi-Agent Framework
"""

import sys
import os
import asyncio
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus
from core.state_store import StateStore
from core.models import (
    AMRState, AMRStatus, ChargingStation, Event, EventType,
    AlertSeverity, Position, Task, TaskStatus, TaskType, SystemSnapshot
)
from agents.coordinator_agent import CoordinatorAgent
from agents.energy_agent import (
    EnergyAgent, _battery_bin, _dist_bin, _queue_bin,
    ACTION_CONTINUE, ACTION_GO_CHARGE, ACTION_REDUCE_SPEED,
    N_BATTERY_BINS, N_DIST_BINS, N_QUEUE_BINS
)
from agents.sentinel_agent import SentinelAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_infrastructure():
    """Create fresh bus + store for each test."""
    tmpdir = tempfile.mkdtemp()
    bus   = EventBus()
    store = StateStore(db_path=os.path.join(tmpdir, "test.db"))
    return bus, store


def make_amr(amr_id="AMR_01", name="Atlas", x=5.0, y=5.0, battery=100.0) -> AMRState:
    amr = AMRState(amr_id=amr_id, name=name, position=Position(x, y))
    amr.battery = battery
    return amr


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Q-learning helper tests
# ---------------------------------------------------------------------------

class TestQLearningHelpers:
    def test_battery_bins(self):
        assert _battery_bin(5)   == 0   # critical
        assert _battery_bin(15)  == 1   # low
        assert _battery_bin(50)  == 2   # medium
        assert _battery_bin(90)  == 3   # high

    def test_battery_bin_boundaries(self):
        assert _battery_bin(10)  == 0   # exactly at critical boundary → critical
        assert _battery_bin(10.1) == 1  # just above critical → low
        assert _battery_bin(25)  == 1   # at low boundary → low
        assert _battery_bin(25.1) == 2  # just above → medium

    def test_dist_bins(self):
        assert _dist_bin(3)  == 0  # close
        assert _dist_bin(8)  == 1  # medium
        assert _dist_bin(15) == 2  # far

    def test_queue_bins(self):
        assert _queue_bin(0) == 0
        assert _queue_bin(1) == 1
        assert _queue_bin(9) == 1


class TestEnergyAgentQTable:
    def setup_method(self):
        bus, store = make_infrastructure()
        self.agent = EnergyAgent(bus, store, {})

    def test_qtable_shape(self):
        import numpy as np
        assert self.agent.q_table.shape == (
            N_BATTERY_BINS, N_DIST_BINS, N_QUEUE_BINS, 3
        )

    def test_qtable_seeded_correctly(self):
        """Critical battery state should strongly prefer GO_CHARGE."""
        import numpy as np
        # State: critical battery, any distance, any queue
        for d in range(N_DIST_BINS):
            for q in range(N_QUEUE_BINS):
                best_action = int(np.argmax(self.agent.q_table[0, d, q]))
                assert best_action == ACTION_GO_CHARGE, (
                    f"Critical battery should prefer GO_CHARGE, got {best_action}"
                )

    def test_qtable_high_battery_prefers_continue(self):
        """High battery state should prefer CONTINUE."""
        import numpy as np
        for d in range(N_DIST_BINS):
            for q in range(N_QUEUE_BINS):
                best_action = int(np.argmax(self.agent.q_table[3, d, q]))
                assert best_action == ACTION_CONTINUE, (
                    f"High battery should prefer CONTINUE, got {best_action}"
                )

    def test_qtable_update_changes_value(self):
        """Q-table update should change the value at the given state."""
        import numpy as np
        state  = (1, 1, 0)
        action = ACTION_GO_CHARGE
        before = float(self.agent.q_table[state][action])
        run(self.agent._update_qtable("AMR_01", state, action, reward=10.0))
        after  = float(self.agent.q_table[state][action])
        assert after != before

    def test_get_qtable_stats(self):
        stats = self.agent.get_qtable_stats()
        assert "total_reward"         in stats
        assert "charge_events"        in stats
        assert "prevented_depletions" in stats

    def test_get_qtable_heatmap_data(self):
        data = self.agent.get_qtable_heatmap_data()
        assert "battery_labels" in data
        assert "dist_labels"    in data
        assert "action_labels"  in data
        assert "q_values"       in data
        assert len(data["q_values"]) == N_BATTERY_BINS


# ---------------------------------------------------------------------------
# Coordinator Agent tests
# ---------------------------------------------------------------------------

class TestCoordinatorAgent:
    def setup_method(self):
        self.bus, self.store = make_infrastructure()
        self.agent = CoordinatorAgent(self.bus, self.store, {
            "energy_critical_threshold": 10,
            "grid_size": 20,
        })

    def test_score_amr_for_task_closer_wins(self):
        """AMR closer to task target should get higher score."""
        amr_close = make_amr("A1", x=2.0, y=2.0, battery=80)
        amr_far   = make_amr("A2", x=18.0, y=18.0, battery=80)
        task = Task(task_type=TaskType.NAVIGATE, target=Position(2.5, 2.5))

        score_close = self.agent._score_amr_for_task(amr_close, task)
        score_far   = self.agent._score_amr_for_task(amr_far, task)
        assert score_close > score_far

    def test_score_amr_higher_battery_wins(self):
        """Among equidistant AMRs, higher battery should score better."""
        amr_hi  = make_amr("A1", x=5.0, y=5.0, battery=90)
        amr_lo  = make_amr("A2", x=5.0, y=5.0, battery=20)
        task    = Task(task_type=TaskType.NAVIGATE, target=Position(5.0, 5.0))

        score_hi = self.agent._score_amr_for_task(amr_hi, task)
        score_lo = self.agent._score_amr_for_task(amr_lo, task)
        assert score_hi > score_lo

    def test_score_idle_amr_gets_bonus(self):
        """IDLE AMR should score higher than MOVING AMR all else equal."""
        amr_idle   = make_amr("A1", x=5.0, y=5.0, battery=80)
        amr_moving = make_amr("A2", x=5.0, y=5.0, battery=80)
        amr_idle.status   = AMRStatus.IDLE
        amr_moving.status = AMRStatus.MOVING
        task = Task(target=Position(5.0, 5.0))

        s_idle   = self.agent._score_amr_for_task(amr_idle, task)
        s_moving = self.agent._score_amr_for_task(amr_moving, task)
        assert s_idle > s_moving

    def test_assign_task_updates_store(self):
        """After _assign_task(), store should reflect assignment."""
        self.store.register_amr(make_amr("AMR_01"))
        task = Task(task_type=TaskType.PICKUP, target=Position(8, 8))
        run(self.store.add_task(task))
        amr = self.store.get_amr("AMR_01")

        async def _test():
            await self.agent._assign_task(amr, task)

        run(_test())

        updated_task = self.store.get_all_tasks()[0]
        assert updated_task.status      == TaskStatus.ASSIGNED
        assert updated_task.assigned_to == "AMR_01"


# ---------------------------------------------------------------------------
# Sentinel Agent tests
# ---------------------------------------------------------------------------

class TestSentinelAgent:
    def setup_method(self):
        self.bus, self.store = make_infrastructure()
        self.agent = SentinelAgent(self.bus, self.store, {})

    def test_generate_normal_packet(self):
        amr    = make_amr()
        packet = self.agent._generate_packet(amr, tick=1)
        assert packet.amr_id == amr.amr_id
        assert not packet.is_anomalous
        assert packet.packet_size > 0
        assert packet.latency_ms > 0

    def test_generate_attack_packet_is_anomalous(self):
        """Force an attack and verify generated packet is marked anomalous."""
        amr    = make_amr("AMR_01")
        attack = self.agent._start_attack("AMR_01")
        packet = self.agent._generate_attack_packet(
            "AMR_01", attack, 512, 5.0, -70.0, tick=1
        )
        assert packet.is_anomalous is True

    def test_model_trains_after_min_samples(self):
        """After MIN_SAMPLES packets, model should be trained."""
        amr = make_amr("AMR_01")
        for i in range(self.agent.MIN_SAMPLES):
            pkt = self.agent._generate_packet(amr, tick=i)
            self.agent._packet_buffers["AMR_01"].append(pkt)

        run(self.agent._train_model("AMR_01"))
        assert self.agent._model_trained["AMR_01"] is True
        assert "AMR_01" in self.agent._models

    def test_score_normal_packet_not_critical(self):
        """Normal packets should score above the critical threshold."""
        amr = make_amr("AMR_01")
        # Train model with normal packets
        for i in range(self.agent.MIN_SAMPLES):
            pkt = self.agent._generate_packet(amr, tick=i)
            self.agent._packet_buffers["AMR_01"].append(pkt)
        run(self.agent._train_model("AMR_01"))

        # Score a new normal packet
        normal_pkt = self.agent._generate_packet(amr, tick=999)
        score = self.agent._score_packet("AMR_01", normal_pkt)
        # Normal packet should score above warn threshold most of the time
        assert score > self.agent.ANOMALY_SCORE_CRIT

    def test_quarantine_marks_amr_compromised(self):
        """_quarantine_amr should mark AMR as compromised in state store."""
        self.store.register_amr(make_amr("AMR_01"))
        amr = self.store.get_amr("AMR_01")

        from core.models import NetworkPacket
        packet = NetworkPacket(amr_id="AMR_01", is_anomalous=True)

        async def _test():
            # Start bus briefly so publish works
            task = asyncio.create_task(self.bus.run())
            await self.agent._quarantine_amr(amr, packet, -0.45, "PACKET_FLOOD", tick=1)
            await asyncio.sleep(0.05)
            await self.bus.stop()
            task.cancel()

        run(_test())
        assert self.store.get_amr("AMR_01").is_compromised is True
        assert self.agent._quarantines_issued == 1

    def test_security_stats(self):
        stats = self.agent.get_security_stats()
        required = [
            "packets_analyzed", "anomalies_detected",
            "quarantines_issued", "models_trained", "active_attacks"
        ]
        for k in required:
            assert k in stats

    def test_get_recent_alerts_empty(self):
        assert self.agent.get_recent_alerts() == []


# ---------------------------------------------------------------------------
# Config manager tests
# ---------------------------------------------------------------------------

class TestConfigManager:
    def test_load_defaults(self):
        from config.config_manager import ConfigManager
        mgr = ConfigManager()
        cfg = mgr.load()
        assert cfg.simulation.num_amrs    == 5
        assert cfg.simulation.grid_size   == 20
        assert cfg.energy.low_threshold   == 20.0
        assert cfg.security.contamination == 0.05

    def test_load_with_overrides(self):
        from config.config_manager import ConfigManager
        mgr = ConfigManager()
        cfg = mgr.load(overrides={"num_amrs": 3, "grid_size": 15})
        assert cfg.simulation.num_amrs  == 3
        assert cfg.simulation.grid_size == 15

    def test_to_flat_dict_has_required_keys(self):
        from config.config_manager import ConfigManager
        mgr = ConfigManager()
        cfg = mgr.load()
        flat = cfg.to_flat_dict()
        for key in ["num_amrs", "grid_size", "tick_rate",
                    "energy_low_threshold", "llm_model"]:
            assert key in flat, f"Missing flat key: {key}"

    def test_validation_raises_on_invalid(self):
        from config.config_manager import ConfigManager
        mgr = ConfigManager()
        with pytest.raises(AssertionError):
            mgr.load(overrides={"num_amrs": 0})  # Must be >= 1