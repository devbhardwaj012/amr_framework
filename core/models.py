"""
core/models.py
==============
Shared data models for the entire AMR Multi-Agent Framework.
All agents, the simulator, and the dashboard use these models.
Adding new fields here automatically propagates everywhere.

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AMRStatus(str, Enum):
    """All possible AMR operational states."""
    IDLE        = "idle"         # Powered on, waiting for tasks
    MOVING      = "moving"       # Navigating to a waypoint
    WORKING     = "working"      # Executing a task at current location
    CHARGING    = "charging"     # Docked at a charging station
    LOW_BATTERY = "low_battery"  # Below threshold, heading to charge
    ERROR       = "error"        # Hardware or software fault
    QUARANTINED = "quarantined"  # Isolated by Sentinel (security breach)


class TaskType(str, Enum):
    """Types of tasks an AMR can be assigned."""
    NAVIGATE   = "navigate"    # Move to a position
    PICKUP     = "pickup"      # Pick up an item
    DROPOFF    = "dropoff"     # Drop off an item
    INSPECT    = "inspect"     # Inspect an area
    CHARGE     = "charge"      # Go to charging station
    PATROL     = "patrol"      # Patrol a route


class TaskStatus(str, Enum):
    """Lifecycle of a task."""
    PENDING    = "pending"
    ASSIGNED   = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class EventType(str, Enum):
    """All events that flow through the message bus."""
    # AMR lifecycle
    AMR_STATUS_CHANGE     = "amr_status_change"
    AMR_POSITION_UPDATE   = "amr_position_update"
    AMR_BATTERY_UPDATE    = "amr_battery_update"

    # Task lifecycle
    TASK_CREATED          = "task_created"
    TASK_ASSIGNED         = "task_assigned"
    TASK_COMPLETED        = "task_completed"
    TASK_FAILED           = "task_failed"

    # Coordination
    COLLISION_WARNING     = "collision_warning"
    PATH_NEGOTIATION      = "path_negotiation"
    TASK_DELEGATION       = "task_delegation"

    # Energy
    BATTERY_LOW           = "battery_low"
    BATTERY_CRITICAL      = "battery_critical"
    CHARGING_COMPLETE     = "charging_complete"
    CHARGE_STATION_ASSIGN = "charge_station_assign"

    # Security
    ANOMALY_DETECTED      = "anomaly_detected"
    INTRUSION_ALERT       = "intrusion_alert"
    AMR_QUARANTINED       = "amr_quarantined"
    AMR_RESTORED          = "amr_restored"
    NETWORK_SCAN          = "network_scan"

    # System
    TICK                  = "tick"             # Simulation heartbeat
    AGENT_LOG             = "agent_log"        # Agent published a log message


class AlertSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Core Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """2D grid position."""
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass
class Task:
    """A unit of work to be assigned and executed by an AMR."""
    task_id:     str       = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type:   TaskType  = TaskType.NAVIGATE
    target:      Optional[Position] = None
    priority:    int       = 1                  # 1 (low) to 5 (critical)
    assigned_to: Optional[str] = None           # AMR ID
    status:      TaskStatus = TaskStatus.PENDING
    created_at:  float     = field(default_factory=time.time)
    started_at:  Optional[float] = None
    completed_at: Optional[float] = None
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":     self.task_id,
            "task_type":   self.task_type.value,
            "target":      self.target.to_dict() if self.target else None,
            "priority":    self.priority,
            "assigned_to": self.assigned_to,
            "status":      self.status.value,
            "created_at":  self.created_at,
        }


@dataclass
class NetworkPacket:
    """
    Simulated 5G network packet for an AMR.
    Sentinel Agent uses streams of these to detect anomalies.
    """
    amr_id:          str
    timestamp:       float = field(default_factory=time.time)
    packet_size:     int   = 512         # bytes
    latency_ms:      float = 5.0         # milliseconds
    signal_strength: float = -70.0       # dBm  (-50 excellent, -90 poor)
    packet_loss:     float = 0.0         # 0.0 to 1.0
    src_ip:          str   = "10.0.0.1"
    dst_ip:          str   = "10.0.0.2"
    protocol:        str   = "UDP"
    is_anomalous:    bool  = False       # ground truth for evaluation

    def to_feature_vector(self) -> List[float]:
        """Convert to ML feature vector for anomaly detection."""
        return [
            self.packet_size,
            self.latency_ms,
            self.signal_strength,
            self.packet_loss,
        ]


@dataclass
class AMRState:
    """
    Complete state of a single AMR at a point in time.
    This is the central shared object — every agent reads from this.
    """
    amr_id:          str
    name:            str
    position:        Position
    status:          AMRStatus      = AMRStatus.IDLE
    battery:         float          = 100.0        # percentage 0-100
    current_task:    Optional[Task] = None
    target_position: Optional[Position] = None
    speed:           float          = 1.0          # grid units per tick
    task_history:    List[str]      = field(default_factory=list)   # task_ids
    is_compromised:  bool           = False
    last_seen:       float          = field(default_factory=time.time)

    # Metrics tracked per AMR
    total_distance:    float = 0.0
    tasks_completed:   int   = 0
    tasks_failed:      int   = 0
    energy_consumed:   float = 0.0
    alerts_triggered:  int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amr_id":          self.amr_id,
            "name":            self.name,
            "position":        self.position.to_dict(),
            "status":          self.status.value,
            "battery":         round(self.battery, 1),
            "current_task":    self.current_task.to_dict() if self.current_task else None,
            "target_position": self.target_position.to_dict() if self.target_position else None,
            "is_compromised":  self.is_compromised,
            "tasks_completed": self.tasks_completed,
            "tasks_failed":    self.tasks_failed,
            "total_distance":  round(self.total_distance, 2),
            "energy_consumed": round(self.energy_consumed, 2),
            "alerts_triggered": self.alerts_triggered,
        }


@dataclass
class ChargingStation:
    """A charging station on the warehouse floor."""
    station_id:  str
    position:    Position
    is_occupied: bool  = False
    occupant_id: Optional[str] = None  # AMR ID currently charging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id":  self.station_id,
            "position":    self.position.to_dict(),
            "is_occupied": self.is_occupied,
            "occupant_id": self.occupant_id,
        }


@dataclass
class Event:
    """
    A message on the central event bus.
    Every agent communicates exclusively via events — no direct calls.
    """
    event_id:   str            = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType      = EventType.TICK
    source:     str            = "system"       # Who published this event
    payload:    Dict[str, Any] = field(default_factory=dict)
    timestamp:  float          = field(default_factory=time.time)
    severity:   AlertSeverity  = AlertSeverity.INFO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":   self.event_id,
            "event_type": self.event_type.value,
            "source":     self.source,
            "payload":    self.payload,
            "timestamp":  self.timestamp,
            "severity":   self.severity.value,
        }


@dataclass
class AgentLog:
    """Structured log entry from any agent, shown in the dashboard."""
    agent_name:  str
    message:     str
    severity:    AlertSeverity = AlertSeverity.INFO
    timestamp:   float = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "message":    self.message,
            "severity":   self.severity.value,
            "timestamp":  self.timestamp,
        }


@dataclass
class SystemSnapshot:
    """
    Complete state of the entire system at one moment in time.
    Used by the dashboard and stored to SQLite for history.
    """
    tick:              int
    timestamp:         float = field(default_factory=time.time)
    amrs:              Dict[str, AMRState]       = field(default_factory=dict)
    charging_stations: Dict[str, ChargingStation] = field(default_factory=dict)
    pending_tasks:     List[Task]                 = field(default_factory=list)
    recent_events:     List[Event]                = field(default_factory=list)
    agent_logs:        List[AgentLog]             = field(default_factory=list)

    # Aggregate metrics
    fleet_avg_battery:   float = 0.0
    active_amrs:         int   = 0
    tasks_in_queue:      int   = 0
    total_alerts:        int   = 0