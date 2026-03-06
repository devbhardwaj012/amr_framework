"""
config/config_manager.py
========================
Centralized configuration management for the AMR framework.

Loads from (in priority order):
  1. Runtime overrides (dict passed to Orchestrator)
  2. .env file / environment variables
  3. config/defaults.yaml (if present)
  4. Hardcoded defaults (always available)

All config values are validated and typed here.
Every module imports config via: from config.config_manager import cfg

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass — every setting with its type and default value
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Simulation engine settings."""
    num_amrs:           int   = 5
    grid_size:          int   = 20
    tick_rate:          float = 1.0       # seconds per tick
    task_work_duration: float = 3.0       # seconds an AMR spends "working" on a task
    task_spawn_prob:    float = 0.15      # probability of new task each tick
    amr_speed:          float = 1.0       # grid units per tick


@dataclass
class EnergyConfig:
    """Energy Agent settings."""
    low_threshold:      float = 20.0   # % battery — warn
    critical_threshold: float = 10.0   # % battery — force charge
    drain_moving:       float = 0.30   # % per tick
    drain_working:      float = 0.50   # % per tick
    drain_idle:         float = 0.05   # % per tick
    charge_rate:        float = 2.00   # % per tick
    # Q-learning
    ql_alpha:           float = 0.10   # learning rate
    ql_gamma:           float = 0.90   # discount factor
    ql_epsilon:         float = 0.15   # exploration rate


@dataclass
class SecurityConfig:
    """Sentinel Agent settings."""
    contamination:          float = 0.05   # IsolationForest expected anomaly rate
    min_train_samples:      int   = 30     # packets before model trains
    window_size:            int   = 100    # rolling buffer per AMR
    anomaly_warn_threshold: float = -0.10  # score below this → warning
    anomaly_crit_threshold: float = -0.30  # score below this → quarantine
    attack_base_prob:       float = 0.005  # attack start probability per tick
    quarantine_duration:    float = 15.0   # seconds before auto-restore


@dataclass
class CoordinatorConfig:
    """Coordinator Agent settings."""
    collision_radius:   float = 2.0   # grid units — warn distance
    collision_critical: float = 1.0   # grid units — emergency stop
    rebalance_thresh:   float = 3.0   # idle AMR clustering distance


@dataclass
class LLMConfig:
    """LLM (Groq) settings."""
    enabled:    bool  = False          # set to True when API key found
    model:      str   = "llama3-70b-8192"
    max_tokens: int   = 512
    temperature: float = 0.2


@dataclass
class SystemConfig:
    """Top-level system config — contains all sub-configs."""
    simulation:  SimulationConfig  = field(default_factory=SimulationConfig)
    energy:      EnergyConfig      = field(default_factory=EnergyConfig)
    security:    SecurityConfig    = field(default_factory=SecurityConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    llm:         LLMConfig         = field(default_factory=LLMConfig)

    # Paths
    db_path:  str = "data/amr_history.db"
    log_path: str = "logs/amr_system.log"
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten to single-level dict for backward compatibility
        with old config dict style.
        """
        return {
            # Simulation
            "num_amrs":                  self.simulation.num_amrs,
            "grid_size":                 self.simulation.grid_size,
            "tick_rate":                 self.simulation.tick_rate,
            "task_work_duration":        self.simulation.task_work_duration,
            "task_spawn_prob":           self.simulation.task_spawn_prob,
            "amr_speed":                 self.simulation.amr_speed,
            # Energy
            "energy_low_threshold":      self.energy.low_threshold,
            "energy_critical_threshold": self.energy.critical_threshold,
            "drain_moving":              self.energy.drain_moving,
            "drain_working":             self.energy.drain_working,
            "drain_idle":                self.energy.drain_idle,
            "charge_rate":               self.energy.charge_rate,
            "ql_alpha":                  self.energy.ql_alpha,
            "ql_gamma":                  self.energy.ql_gamma,
            "ql_epsilon":                self.energy.ql_epsilon,
            # Security
            "anomaly_contamination":     self.security.contamination,
            "anomaly_sensitivity":       self.security.anomaly_crit_threshold,
            "attack_base_prob":          self.security.attack_base_prob,
            "quarantine_duration":       self.security.quarantine_duration,
            # Coordinator
            "collision_radius":          self.coordinator.collision_radius,
            "collision_critical":        self.coordinator.collision_critical,
            # LLM
            "llm_model":                 self.llm.model,
            "llm_max_tokens":            self.llm.max_tokens,
            # System
            "db_path":                   self.db_path,
            "log_path":                  self.log_path,
        }


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class ConfigManager:
    """
    Loads, validates, and provides access to system configuration.

    Usage:
        from config.config_manager import load_config
        cfg = load_config()                    # loads from env
        cfg = load_config(overrides={"num_amrs": 3})  # with overrides
    """

    def __init__(self):
        self._config: Optional[SystemConfig] = None

    def load(self, overrides: Dict[str, Any] = None) -> SystemConfig:
        """
        Load config from environment variables + optional overrides.
        Returns a fully populated SystemConfig.
        """
        cfg = SystemConfig()

        # --- Apply environment variables ---
        cfg.simulation.num_amrs           = int(os.getenv("NUM_AMRS", cfg.simulation.num_amrs))
        cfg.simulation.grid_size          = int(os.getenv("GRID_SIZE", cfg.simulation.grid_size))
        cfg.simulation.tick_rate          = float(os.getenv("SIMULATION_TICK_RATE", cfg.simulation.tick_rate))
        cfg.simulation.task_spawn_prob    = float(os.getenv("TASK_SPAWN_PROB", cfg.simulation.task_spawn_prob))

        cfg.energy.low_threshold          = float(os.getenv("ENERGY_LOW_THRESHOLD", cfg.energy.low_threshold))
        cfg.energy.critical_threshold     = float(os.getenv("ENERGY_CRITICAL_THRESHOLD", cfg.energy.critical_threshold))

        cfg.security.contamination        = float(os.getenv("ANOMALY_SENSITIVITY", cfg.security.contamination))
        cfg.security.attack_base_prob     = float(os.getenv("ATTACK_BASE_PROB", cfg.security.attack_base_prob))

        cfg.llm.model                     = os.getenv("LLM_MODEL", cfg.llm.model)
        cfg.llm.enabled                   = bool(os.getenv("GROQ_API_KEY", ""))

        cfg.log_level                     = os.getenv("LOG_LEVEL", cfg.log_level)
        cfg.db_path                       = os.getenv("DB_PATH", cfg.db_path)

        # --- Apply runtime overrides (highest priority) ---
        if overrides:
            self._apply_overrides(cfg, overrides)

        # --- Validate ---
        self._validate(cfg)

        self._config = cfg
        logger.info(f"[Config] Loaded: {cfg.simulation.num_amrs} AMRs, "
                    f"grid={cfg.simulation.grid_size}, "
                    f"LLM={'on' if cfg.llm.enabled else 'off'}")
        return cfg

    def _apply_overrides(self, cfg: SystemConfig, overrides: Dict[str, Any]) -> None:
        """Apply flat override dict to nested config."""
        mapping = {
            "num_amrs":                  ("simulation", "num_amrs"),
            "grid_size":                 ("simulation", "grid_size"),
            "tick_rate":                 ("simulation", "tick_rate"),
            "task_work_duration":        ("simulation", "task_work_duration"),
            "energy_low_threshold":      ("energy", "low_threshold"),
            "energy_critical_threshold": ("energy", "critical_threshold"),
            "ql_alpha":                  ("energy", "ql_alpha"),
            "ql_epsilon":                ("energy", "ql_epsilon"),
            "attack_base_prob":          ("security", "attack_base_prob"),
            "quarantine_duration":       ("security", "quarantine_duration"),
            "collision_radius":          ("coordinator", "collision_radius"),
            "llm_model":                 ("llm", "model"),
            "db_path":                   (None, "db_path"),
        }
        for key, value in overrides.items():
            if key in mapping:
                section, attr = mapping[key]
                if section:
                    setattr(getattr(cfg, section), attr, value)
                else:
                    setattr(cfg, attr, value)

    def _validate(self, cfg: SystemConfig) -> None:
        """Validate config values and clamp to sane ranges."""
        assert 1 <= cfg.simulation.num_amrs <= 20,   "num_amrs must be 1-20"
        assert 10 <= cfg.simulation.grid_size <= 100, "grid_size must be 10-100"
        assert 0.1 <= cfg.simulation.tick_rate <= 10, "tick_rate must be 0.1-10s"
        assert 0 < cfg.energy.low_threshold <= 50,   "low_threshold must be 1-50%"
        assert 0 < cfg.energy.critical_threshold < cfg.energy.low_threshold, \
            "critical_threshold must be less than low_threshold"
        assert 0 < cfg.security.contamination < 0.5, "contamination must be 0-0.5"

        # Clamp Q-learning params
        cfg.energy.ql_alpha   = max(0.01, min(1.0, cfg.energy.ql_alpha))
        cfg.energy.ql_gamma   = max(0.1,  min(0.99, cfg.energy.ql_gamma))
        cfg.energy.ql_epsilon = max(0.01, min(0.5,  cfg.energy.ql_epsilon))

    @property
    def config(self) -> Optional[SystemConfig]:
        return self._config


# Module-level singleton
_manager = ConfigManager()


def load_config(overrides: Dict[str, Any] = None) -> SystemConfig:
    """Load and return the system config. Call once at startup."""
    from dotenv import load_dotenv
    load_dotenv()
    return _manager.load(overrides)


def get_config() -> SystemConfig:
    """Get the already-loaded config. Raises if load_config() not called first."""
    if _manager.config is None:
        return load_config()
    return _manager.config