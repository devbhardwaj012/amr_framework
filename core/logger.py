"""
core/logger.py
==============
Centralized logging setup for the AMR framework.

Features:
  - Colored console output per log level
  - Rotating file handler (keeps last 5 logs, 5MB each)
  - Per-agent named loggers
  - Structured JSON log option for production

Call setup_logging() once at startup (done automatically by Orchestrator).

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
COLORS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "DIM":      "\033[2m",
}

# Agent name color map (so each agent has its own color)
AGENT_COLORS = {
    "CoordinatorAgent": "\033[34m",   # Blue
    "EnergyAgent":      "\033[33m",   # Yellow
    "SentinelAgent":    "\033[31m",   # Red
    "simulator":        "\033[36m",   # Cyan
    "EventBus":         "\033[35m",   # Magenta
    "StateStore":       "\033[32m",   # Green
}


class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI colors per log level and agent name."""

    FORMAT = (
        "{dim}%(asctime)s{reset} "
        "{agent_color}[%(name)-20s]{reset} "
        "{level_color}{bold}%(levelname)-8s{reset} "
        "%(message)s"
    )

    def format(self, record: logging.LogRecord) -> str:
        level_color = COLORS.get(record.levelname, "")
        dim         = COLORS["DIM"]
        bold        = COLORS["BOLD"]
        reset       = COLORS["RESET"]

        # Color by agent name
        agent_color = COLORS["RESET"]
        for agent_name, color in AGENT_COLORS.items():
            if agent_name in record.name:
                agent_color = color
                break

        log_fmt = self.FORMAT.format(
            dim=dim, reset=reset,
            agent_color=agent_color,
            level_color=level_color,
            bold=bold,
        )
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging (useful in production)."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "line":      record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(
    log_level: str = "INFO",
    log_file:  str = "logs/amr_system.log",
    json_logs: bool = False,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 5,
) -> None:
    """
    Configure the root logger for the entire application.
    Call once at startup in main.py or orchestrator.py.

    Args:
        log_level:    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        log_file:     Path to rotating log file
        json_logs:    If True, use JSON format for file logs
        max_bytes:    Max file size before rotation
        backup_count: Number of backup log files to keep
    """
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger   = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # ── Console handler (colored) ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # ── Rotating file handler ──
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    if json_logs:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx", "asyncio", "sklearn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info(
        f"Logging initialized: level={log_level}, file={log_file}, "
        f"json={json_logs}"
    )


def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get a named logger for an agent."""
    return logging.getLogger(f"agent.{agent_name}")