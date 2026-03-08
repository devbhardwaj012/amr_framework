"""
agents/sentinel_agent.py
=========================
Sentinel Agent — Isolation Forest + LLM Threat Intelligence

Isolation Forest (ML) handles:
  - Unsupervised anomaly detection on 5G network packets
  - Per-AMR model trained on its own traffic baseline

LLM (Groq/LLaMA3-70B) handles:
  - Natural language incident reports for each quarantine
  - Threat classification and severity assessment
  - Fleet-wide security posture summary
  - Restoration recommendations after quarantine period

Author: AMR Multi-Agent Framework
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.base_agent import BaseAgent
from core.models import (
    AlertSeverity, AMRState, AMRStatus, Event, EventType,
    NetworkPacket, SystemSnapshot
)

logger = logging.getLogger(__name__)


class SentinelAgent(BaseAgent):
    """
    5G network security agent combining Isolation Forest ML with LLM threat intelligence.

    Per-tick pipeline:
    1. Simulate network packet for each active AMR
    2. Buffer packets in rolling window (per AMR)
    3. Train Isolation Forest once MIN_SAMPLES collected
    4. Score each new packet → flag anomalies
    5. If anomaly score crosses CRIT threshold:
       a. Quarantine AMR (rule-based, instant)
       b. Ask LLM to generate incident report (async)
    6. Every N ticks → LLM security posture summary
    """

    CONTAMINATION      = 0.05
    MIN_SAMPLES        = 30
    WINDOW_SIZE        = 100
    ANOMALY_SCORE_WARN = -0.55
    ANOMALY_SCORE_CRIT = -0.65
    ATTACK_BASE_PROB   = 0.005
    ATTACK_BURST_PROB  = 0.05
    LLM_SUMMARY_INTERVAL = 8    # ticks between LLM security summaries

    def __init__(self, bus, store, config=None):
        super().__init__("SentinelAgent", bus, store, config)

        self._packet_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.WINDOW_SIZE)
        )
        self._models:        Dict[str, IsolationForest] = {}
        self._scalers:       Dict[str, StandardScaler]  = {}
        self._model_trained: Dict[str, bool] = defaultdict(bool)
        self._active_attacks: Dict[str, dict] = {}
        self._quarantined:   Dict[str, float] = {}
        self._quarantine_duration = 15.0
        self._alerts:   List[dict] = []
        self._max_alerts = 200

        self._groq_client   = None
        self._llm_enabled   = bool(os.getenv("GROQ_API_KEY"))
        self._llm_model     = os.getenv("LLM_MODEL", "llama3-70b-8192")
        self._llm_calls:    int = 0
        self._last_llm_summary_tick: int = 0

        self._packets_analyzed:  int = 0
        self._anomalies_detected: int = 0
        self._quarantines_issued: int = 0

        # Latest LLM security summary (shown in dashboard)
        self.llm_security_summary: str = ""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        self.bus.subscribe(EventType.AMR_QUARANTINED, self._on_amr_quarantined)
        self.bus.subscribe(EventType.AMR_RESTORED,    self._on_amr_restored)

        if self._llm_enabled:
            self._init_groq()
            await self.log("LLM threat intelligence ENABLED.", AlertSeverity.INFO)

        await self.log(
            f"Sentinel Agent online. Monitoring 5G network traffic. "
            f"Isolation Forest contamination={self.CONTAMINATION}.",
            AlertSeverity.INFO,
        )

    def _init_groq(self) -> None:
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except ImportError:
            logger.warning("groq not installed.")
            self._groq_client = None
            self._llm_enabled = False

    # ------------------------------------------------------------------
    # Main Tick
    # ------------------------------------------------------------------

    async def on_tick(self, snapshot: SystemSnapshot) -> None:
        current_time = time.time()

        for amr in snapshot.amrs.values():
            if amr.status == AMRStatus.QUARANTINED:
                await self._check_quarantine_expiry(amr, current_time)
                continue

            packet = self._generate_packet(amr, snapshot.tick)
            self._packet_buffers[amr.amr_id].append(packet)
            self._packets_analyzed += 1

            if len(self._packet_buffers[amr.amr_id]) >= self.MIN_SAMPLES:
                if not self._model_trained[amr.amr_id]:
                    await self._train_model(amr.amr_id)

                if self._model_trained[amr.amr_id]:
                    score = self._score_packet(amr.amr_id, packet)
                    await self._evaluate_score(amr, packet, score, snapshot.tick)

        # Periodic LLM security posture summary
        tick = snapshot.tick
        if (self._groq_client
                and tick > 0
                and tick - self._last_llm_summary_tick >= self.LLM_SUMMARY_INTERVAL):
            await self._llm_security_posture(snapshot)
            self._last_llm_summary_tick = tick

        # Periodic scan log
        if tick % 10 == 0:
            total = len(snapshot.amrs)
            monitored = sum(1 for aid in snapshot.amrs if self._model_trained[aid])
            q = sum(1 for a in snapshot.amrs.values() if a.status == AMRStatus.QUARANTINED)
            await self.log(
                f"🔍 Network scan: {total} AMRs total, {monitored} monitored, {q} quarantined.",
                AlertSeverity.INFO,
            )

    # ------------------------------------------------------------------
    # Packet Generation (simulation)
    # ------------------------------------------------------------------

    def _generate_packet(self, amr: AMRState, tick: int) -> NetworkPacket:
        """Simulate a normal 5G network packet from an AMR."""
        # Decide if an attack starts this tick
        amr_id = amr.amr_id
        if amr_id in self._active_attacks:
            attack = self._active_attacks[amr_id]
            attack["remaining"] -= 1
            if attack["remaining"] <= 0:
                del self._active_attacks[amr_id]
                logger.info(f"[Sentinel] Attack on {amr_id} ended.")
            else:
                return self._generate_attack_packet(
                    amr_id, attack,
                    normal_size=512, normal_latency=5.0, normal_signal=-70.0, tick=tick
                )
        elif (amr_id not in self._quarantined
              and random.random() < self.ATTACK_BASE_PROB):
            attack_type = random.choice(["PACKET_FLOOD", "JAMMING", "DATA_EXFIL", "REPLAY_ATTACK"])
            duration    = random.randint(5, 12)
            self._active_attacks[amr_id] = {
                "type": attack_type, "remaining": duration,
                "start_tick": tick, "duration": duration,
            }
            logger.info(f"[Sentinel] Simulating {attack_type} attack on {amr_id}")
            return self._generate_attack_packet(
                amr_id, self._active_attacks[amr_id],
                normal_size=512, normal_latency=5.0, normal_signal=-70.0, tick=tick
            )

        # Normal packet
        return NetworkPacket(
            amr_id=amr_id,
            packet_size=random.gauss(512, 30),
            latency_ms=random.gauss(5.0, 0.5),
            signal_strength=random.gauss(-70.0, 2.0),
            packet_loss=random.uniform(0.0, 0.03),
            is_anomalous=False,
        )

    def _generate_attack_packet(
        self, amr_id: str, attack: dict,
        normal_size: float, normal_latency: float, normal_signal: float, tick: int
    ) -> NetworkPacket:
        attack_type = attack["type"]
        if attack_type == "PACKET_FLOOD":
            return NetworkPacket(amr_id=amr_id,
                packet_size=random.uniform(40, 80),
                latency_ms=normal_latency * random.uniform(18, 22),
                signal_strength=normal_signal + random.uniform(-3, 3),
                packet_loss=random.uniform(0.0, 0.05),
                is_anomalous=True)
        elif attack_type == "JAMMING":
            return NetworkPacket(amr_id=amr_id,
                packet_size=random.gauss(normal_size, 20),
                latency_ms=normal_latency * random.uniform(2, 4),
                signal_strength=random.uniform(-115, -108),
                packet_loss=random.uniform(0.70, 0.99),
                is_anomalous=True)
        elif attack_type == "DATA_EXFIL":
            return NetworkPacket(amr_id=amr_id,
                packet_size=random.uniform(40000, 65000),
                latency_ms=random.uniform(2.0, 3.5),
                signal_strength=normal_signal + random.uniform(-2, 2),
                packet_loss=random.uniform(0.0, 0.02),
                is_anomalous=True)
        else:  # REPLAY_ATTACK
            return NetworkPacket(amr_id=amr_id,
                packet_size=random.gauss(normal_size, 25),
                latency_ms=normal_latency * random.uniform(1.5, 2.5),
                signal_strength=random.uniform(-105, -98),
                packet_loss=random.uniform(0.50, 0.90),
                is_anomalous=True)

    # ------------------------------------------------------------------
    # ML Model
    # ------------------------------------------------------------------

    async def _train_model(self, amr_id: str) -> None:
        buffer = self._packet_buffers[amr_id]
        normal = [p for p in buffer if not p.is_anomalous]
        if len(normal) < self.MIN_SAMPLES:
            return

        X = np.array([p.to_feature_vector() for p in normal])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            contamination=self.CONTAMINATION,
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_scaled)

        self._models[amr_id]       = model
        self._scalers[amr_id]      = scaler
        self._model_trained[amr_id] = True
        await self.log(
            f"📡 Isolation Forest trained for {amr_id} on {len(normal)} normal packets.",
            AlertSeverity.INFO,
        )

    def _score_packet(self, amr_id: str, packet: NetworkPacket) -> float:
        model  = self._models[amr_id]
        scaler = self._scalers[amr_id]
        X = np.array([packet.to_feature_vector()])
        return float(model.score_samples(scaler.transform(X))[0])

    # ------------------------------------------------------------------
    # Anomaly Evaluation
    # ------------------------------------------------------------------

    async def _evaluate_score(
        self, amr: AMRState, packet: NetworkPacket, score: float, tick: int
    ) -> None:
        if score >= self.ANOMALY_SCORE_WARN:
            return  # Normal traffic

        self._anomalies_detected += 1
        attack_type = self._active_attacks.get(amr.amr_id, {}).get("type", "UNKNOWN")
        severity    = "critical" if score < self.ANOMALY_SCORE_CRIT else "warning"

        alert = {
            "tick":        tick,
            "amr_id":      amr.amr_id,
            "amr_name":    amr.name,
            "attack_type": attack_type,
            "severity":    severity,
            "score":       round(score, 4),
            "packet_size": round(packet.packet_size, 1),
            "latency_ms":  round(packet.latency_ms, 2),
            "packet_loss": round(packet.packet_loss, 3),
            "timestamp":   time.time(),
        }
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts.pop(0)

        if score < self.ANOMALY_SCORE_CRIT and amr.amr_id not in self._quarantined:
            await self._quarantine_amr(amr, packet, score, attack_type, tick)
        else:
            await self.log(
                f"⚠️ Suspicious traffic on {amr.name}: score={score:.3f}, type={attack_type}",
                AlertSeverity.WARNING,
            )
            await self.emit(EventType.INTRUSION_ALERT, {
                "amr_id": amr.amr_id, "score": score,
                "severity": "warning", "attack_type": attack_type,
            }, AlertSeverity.WARNING)

    async def _quarantine_amr(
        self, amr: AMRState, packet: NetworkPacket,
        score: float, attack_type: str, tick: int
    ) -> None:
        print(f"🚨 INTRUSION DETECTED on {amr.amr_id}! Score={score:.3f}, Type={attack_type}. AMR QUARANTINED.")
        await self.store.set_amr_compromised(amr.amr_id, True)
        self._quarantined[amr.amr_id] = time.time()
        self._quarantines_issued += 1

        await self.log(
            f"🚨 AMR {amr.name} QUARANTINED. Score={score:.3f}, Attack={attack_type}",
            AlertSeverity.CRITICAL,
        )
        await self.emit(EventType.AMR_QUARANTINED, {
            "amr_id": amr.amr_id, "score": score,
            "attack_type": attack_type, "tick": tick,
        }, AlertSeverity.CRITICAL)
        await self.emit(EventType.INTRUSION_ALERT, {
            "amr_id": amr.amr_id, "score": score,
            "severity": "critical", "attack_type": attack_type,
        }, AlertSeverity.CRITICAL)

        self._decisions_made += 1

        # Ask LLM to write incident report (non-blocking)
        if self._groq_client:
            asyncio.create_task(
                self._llm_incident_report(amr, packet, score, attack_type, tick)
            )

    # ------------------------------------------------------------------
    # LLM: Incident Report
    # ------------------------------------------------------------------

    async def _llm_incident_report(
        self, amr: AMRState, packet: NetworkPacket,
        score: float, attack_type: str, tick: int
    ) -> None:
        """Generate a structured security incident report using LLaMA."""
        prompt = f"""You are a cybersecurity analyst for an autonomous warehouse robot fleet.
Generate a concise incident report for a detected network attack.

INCIDENT DATA:
  AMR ID: {amr.amr_id}
  AMR Name: {amr.name}
  Attack Type: {attack_type}
  Anomaly Score: {score:.4f} (threshold: {self.ANOMALY_SCORE_CRIT})
  Simulation Tick: {tick}
  AMR Battery: {amr.battery:.1f}%
  AMR Status at time of detection: {amr.status.value}

NETWORK PACKET EVIDENCE:
  Packet size: {packet.packet_size:.0f} bytes (normal: ~512 bytes)
  Latency: {packet.latency_ms:.1f} ms (normal: ~5 ms)
  Signal strength: {packet.signal_strength:.1f} dBm (normal: ~-70 dBm)
  Packet loss rate: {packet.packet_loss:.1%} (normal: <3%)

Write a JSON incident report with these exact fields:
{{
  "severity": "CRITICAL" or "HIGH",
  "attack_summary": "<2 sentences: what happened and how it was detected>",
  "impact": "<1 sentence: potential impact if not contained>",
  "action_taken": "<1 sentence: what automated response was triggered>",
  "recommendation": "<1 sentence: what the operator should do next>"
}}
Respond with ONLY the JSON, no other text."""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.2,
            )
            self._llm_calls += 1
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            report = json.loads(text)

            # Attach report to the latest alert for this AMR
            for alert in reversed(self._alerts):
                if alert["amr_id"] == amr.amr_id:
                    alert["llm_report"] = report
                    break

            await self.log(
                f"🤖 LLM Incident Report for {amr.name}:\n"
                f"  Summary: {report.get('attack_summary', '')}\n"
                f"  Impact:  {report.get('impact', '')}\n"
                f"  Action:  {report.get('action_taken', '')}\n"
                f"  Recommend: {report.get('recommendation', '')}",
                AlertSeverity.CRITICAL,
            )
        except Exception as e:
            logger.warning(f"[SentinelAgent] LLM incident report failed: {e}")

    # ------------------------------------------------------------------
    # LLM: Security Posture Summary
    # ------------------------------------------------------------------

    async def _llm_security_posture(self, snapshot: SystemSnapshot) -> None:
        """Periodic LLM-generated fleet security health summary."""
        amrs     = list(snapshot.amrs.values())
        q_count  = sum(1 for a in amrs if a.is_compromised)
        ok_count = len(amrs) - q_count

        recent_attacks = [a for a in self._alerts[-20:] if a.get("severity") == "critical"]
        attack_types   = list({a["attack_type"] for a in recent_attacks})

        prompt = f"""You are a security operations analyst for a warehouse robot fleet.

Current security status at tick {snapshot.tick}:
  Total AMRs: {len(amrs)}
  Healthy AMRs: {ok_count}
  Quarantined AMRs: {q_count}
  Total incidents this session: {self._quarantines_issued}
  Recent attack types detected: {attack_types if attack_types else ['none']}
  Anomalies detected this session: {self._anomalies_detected}
  Packets analyzed: {self._packets_analyzed}

Write a 2-sentence security posture summary for the operations dashboard.
Be specific and actionable. No bullet points. Plain English only.
Respond with ONLY the summary text."""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3,
            )
            self._llm_calls += 1
            self.llm_security_summary = response.choices[0].message.content.strip()
            await self.log(f"🤖 Security posture: {self.llm_security_summary}", AlertSeverity.INFO)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[SentinelAgent] LLM security posture failed: {error_msg}")
            await self.log(f"⚠️ LLM security call failed: {error_msg[:120]}", AlertSeverity.WARNING)
            if not self.llm_security_summary:
                self.llm_security_summary = (
                    f"LLM unavailable (tick {snapshot.tick}): {error_msg[:80]}"
                )

    # ------------------------------------------------------------------
    # Quarantine Management
    # ------------------------------------------------------------------

    async def _check_quarantine_expiry(self, amr: AMRState, current_time: float) -> None:
        quarantine_start = self._quarantined.get(amr.amr_id)
        if not quarantine_start:
            return

        if current_time - quarantine_start >= self._quarantine_duration:
            del self._quarantined[amr.amr_id]
            if amr.amr_id in self._active_attacks:
                del self._active_attacks[amr.amr_id]

            await self.store.set_amr_compromised(amr.amr_id, False)
            await self.store.update_amr_status(amr.amr_id, AMRStatus.IDLE)

            # Reset ML model for this AMR (fresh start after restoration)
            self._model_trained[amr.amr_id] = False
            self._packet_buffers[amr.amr_id].clear()

            await self.log(
                f"✅ {amr.name} RESTORED after quarantine. Network model reset.",
                AlertSeverity.INFO,
            )
            await self.emit(EventType.AMR_RESTORED, {"amr_id": amr.amr_id})

            # LLM restoration advisory
            if self._groq_client:
                asyncio.create_task(self._llm_restoration_advisory(amr))

    async def _llm_restoration_advisory(self, amr: AMRState) -> None:
        """Ask LLM for post-quarantine monitoring recommendations."""
        prompt = f"""A quarantined warehouse robot has been restored to service after {self._quarantine_duration:.0f} seconds.

Robot: {amr.name}
Previous incidents: {amr.alerts_triggered}
Battery level at restoration: {amr.battery:.1f}%

Write ONE sentence recommending how operators should monitor this robot going forward.
Respond with ONLY the sentence, no other text."""
        try:
            response = self._groq_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.3,
            )
            self._llm_calls += 1
            advisory = response.choices[0].message.content.strip()
            await self.log(f"🤖 Restoration advisory for {amr.name}: {advisory}", AlertSeverity.INFO)
        except Exception as e:
            logger.warning(f"[SentinelAgent] LLM restoration advisory failed: {e}")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_amr_quarantined(self, event: Event) -> None:
        self._events_received += 1

    async def _on_amr_restored(self, event: Event) -> None:
        self._events_received += 1

    # ------------------------------------------------------------------
    # Dashboard Data
    # ------------------------------------------------------------------

    def get_security_stats(self) -> dict:
        return {
            "packets_analyzed":   self._packets_analyzed,
            "anomalies_detected": self._anomalies_detected,
            "quarantines_issued": self._quarantines_issued,
            "models_trained":     sum(1 for v in self._model_trained.values() if v),
            "active_attacks":     len(self._active_attacks),
            "llm_calls":          self._llm_calls,
            "llm_security_summary": self.llm_security_summary,
        }

    def get_recent_alerts(self, n: int = 50) -> List[dict]:
        return list(reversed(self._alerts[-n:]))

    def get_stats(self) -> dict:
        base = super().get_stats()
        base["llm_calls"] = self._llm_calls
        return base


# Fix missing asyncio import in methods
import asyncio