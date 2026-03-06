"""
core/metrics_exporter.py
=========================
Metrics Exporter — Export simulation data for analysis and reporting.

Exports to:
  - CSV: per-tick fleet metrics (battery, tasks, alerts)
  - CSV: per-AMR performance summary
  - CSV: security alert log
  - JSON: full system snapshot history

Usage:
    from core.metrics_exporter import MetricsExporter
    exporter = MetricsExporter(store, sentinel_agent)
    exporter.export_all("exports/run_001")

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from core.state_store import StateStore
    from agents.sentinel_agent import SentinelAgent
    from agents.energy_agent import EnergyAgent

logger = logging.getLogger(__name__)


class MetricsExporter:
    """
    Exports simulation metrics to CSV and JSON files for:
    - Post-run analysis
    - Report generation
    - Comparison between runs
    """

    def __init__(
        self,
        store:    "StateStore",
        sentinel: Optional["SentinelAgent"] = None,
        energy:   Optional["EnergyAgent"]   = None,
    ):
        self.store    = store
        self.sentinel = sentinel
        self.energy   = energy

    def export_all(self, output_dir: str = "exports") -> dict:
        """
        Export all metrics. Returns dict of {filename: path}.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base      = os.path.join(output_dir, timestamp)

        exported = {}

        exported["fleet_metrics"]  = self.export_fleet_metrics(f"{base}_fleet.csv")
        exported["amr_summary"]    = self.export_amr_summary(f"{base}_amrs.csv")
        exported["battery_history"] = self.export_battery_history(f"{base}_battery.csv")

        if self.sentinel:
            exported["security_alerts"] = self.export_security_alerts(f"{base}_alerts.csv")

        if self.energy:
            exported["qtable"] = self.export_qtable(f"{base}_qtable.json")

        logger.info(f"[MetricsExporter] Exported {len(exported)} files to {output_dir}")
        return exported

    # ------------------------------------------------------------------
    # Individual export methods
    # ------------------------------------------------------------------

    def export_fleet_metrics(self, path: str) -> str:
        """Export per-tick fleet metrics."""
        battery_hist = self.store.get_battery_history()
        task_hist    = self.store.get_task_history()

        # Merge by tick
        bat_by_tick  = {entry["tick"]: entry["avg"]     for entry in battery_hist}
        task_by_tick = {entry["tick"]: entry["pending"] for entry in task_hist}
        all_ticks    = sorted(set(bat_by_tick) | set(task_by_tick))

        rows = [
            {
                "tick":            tick,
                "avg_battery":     round(bat_by_tick.get(tick, 0), 2),
                "pending_tasks":   task_by_tick.get(tick, 0),
            }
            for tick in all_ticks
        ]
        self._write_csv(path, rows, ["tick", "avg_battery", "pending_tasks"])
        return path

    def export_amr_summary(self, path: str) -> str:
        """Export per-AMR performance summary."""
        rows = []
        for amr in self.store.get_all_amrs():
            total_tasks = amr.tasks_completed + amr.tasks_failed
            success_rate = (
                round(amr.tasks_completed / total_tasks * 100, 1)
                if total_tasks > 0 else 0.0
            )
            rows.append({
                "amr_id":         amr.amr_id,
                "name":           amr.name,
                "final_status":   amr.status.value,
                "final_battery":  round(amr.battery, 1),
                "tasks_completed": amr.tasks_completed,
                "tasks_failed":   amr.tasks_failed,
                "success_rate_%": success_rate,
                "total_distance": round(amr.total_distance, 2),
                "energy_consumed": round(amr.energy_consumed, 2),
                "alerts_triggered": amr.alerts_triggered,
                "is_compromised": amr.is_compromised,
            })

        self._write_csv(path, rows, list(rows[0].keys()) if rows else [])
        return path

    def export_battery_history(self, path: str) -> str:
        """Export battery history as CSV."""
        history = self.store.get_battery_history()
        self._write_csv(path, history, ["tick", "avg"])
        return path

    def export_security_alerts(self, path: str) -> str:
        """Export security alert log."""
        if not self.sentinel:
            return ""
        alerts = self.sentinel.get_recent_alerts(1000)
        if not alerts:
            return ""
        keys = ["tick", "amr_id", "amr_name", "attack_type", "severity",
                "score", "packet_size", "latency_ms", "packet_loss"]
        self._write_csv(path, alerts, keys)
        return path

    def export_qtable(self, path: str) -> str:
        """Export Q-table state as JSON."""
        if not self.energy:
            return ""
        data = {
            "qtable_stats":    self.energy.get_qtable_stats(),
            "qtable_heatmap":  self.energy.get_qtable_heatmap_data(),
            "raw_qtable":      self.energy.q_table.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def get_summary_report(self) -> str:
        """
        Generate a human-readable text summary for the dashboard.
        """
        amrs      = self.store.get_all_amrs()
        snap      = self.store.get_snapshot(0)
        total_tasks = sum(a.tasks_completed for a in amrs)
        total_dist  = sum(a.total_distance  for a in amrs)
        total_energy = sum(a.energy_consumed for a in amrs)

        lines = [
            "╔══════════════════════════════════╗",
            "║    SIMULATION SUMMARY REPORT     ║",
            "╚══════════════════════════════════╝",
            "",
            f"Fleet Size:        {len(amrs)} AMRs",
            f"Fleet Avg Battery: {snap.fleet_avg_battery:.1f}%",
            f"Tasks Completed:   {total_tasks}",
            f"Total Distance:    {total_dist:.1f} grid units",
            f"Energy Consumed:   {total_energy:.1f}%",
            f"Total Alerts:      {snap.total_alerts}",
            "",
            "── Per-AMR ─────────────────────────",
        ]
        for a in amrs:
            lines.append(
                f"  {a.name:10s}: battery={a.battery:.0f}%  "
                f"tasks={a.tasks_completed}  "
                f"dist={a.total_distance:.1f}  "
                f"{'🚨QUARANTINED' if a.is_compromised else '✅OK'}"
            )

        if self.sentinel:
            sec = self.sentinel.get_security_stats()
            lines += [
                "",
                "── Security ────────────────────────",
                f"  Packets Analyzed:   {sec['packets_analyzed']}",
                f"  Anomalies Detected: {sec['anomalies_detected']}",
                f"  Quarantines Issued: {sec['quarantines_issued']}",
            ]

        if self.energy:
            q = self.energy.get_qtable_stats()
            lines += [
                "",
                "── Energy / Q-Learning ─────────────",
                f"  Charge Events:      {q['charge_events']}",
                f"  Prevented Depletions: {q['prevented_depletions']}",
                f"  Total QL Reward:    {q['total_reward']}",
            ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_csv(self, path: str, rows: list, fieldnames: list) -> None:
        if not rows:
            logger.warning(f"[MetricsExporter] No data for {path}")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"[MetricsExporter] Wrote {len(rows)} rows → {path}")