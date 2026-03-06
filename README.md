# AMR Multi-Agent Framework

> **Autonomous Mobile Robot Fleet Management** powered by Q-Learning, Isolation Forest, and LLM (Groq/LLaMA3-70B)

A production-grade, research-quality multi-agent system that simulates the intelligent management of an autonomous robot fleet in a warehouse environment. Three specialized AI agents cooperate over an async event bus, each with a distinct responsibility — and each enhanced by a large language model that explains, arbitrates, and narrates every decision.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Architecture Overview](#2-architecture-overview)
3. [Folder Structure](#3-folder-structure)
4. [File Reference](#4-file-reference)
5. [The AMR Fleet](#5-the-amr-fleet)
6. [The Three AI Agents](#6-the-three-ai-agents)
7. [How the LLM Is Used](#7-how-the-llm-is-used)
8. [Setup & Installation](#8-setup--installation)
9. [Running the System](#9-running-the-system)
10. [The Streamlit Dashboard — Complete Guide](#10-the-streamlit-dashboard--complete-guide)
11. [Demo Scenarios](#11-demo-scenarios)
12. [Testing Guide](#12-testing-guide)
13. [Configuration Reference](#13-configuration-reference)
14. [Adding a New AI Agent](#14-adding-a-new-ai-agent)
15. [Tech Stack](#15-tech-stack)

---

## 1. What Is This Project?

Modern warehouses use fleets of **Autonomous Mobile Robots (AMRs)** to move goods across large spaces. Managing them is a hard multi-agent coordination problem:

- Robots must pick up tasks (pickup, delivery, inspection) efficiently
- They must not collide with each other
- They must recharge before their batteries die
- Their 5G network traffic must be monitored for cyberattacks
- All of this must happen simultaneously, in real time, at scale

This project simulates that exact problem and solves it using three cooperating AI agents:

| Agent | Technology | Responsibility |
|---|---|---|
| **Coordinator** | Rule engine + LLM | Collision avoidance, task assignment, fleet rebalancing |
| **Energy** | Q-Learning + LLM | Battery monitoring, charging scheduling, depletion prevention |
| **Sentinel** | Isolation Forest + LLM | Network anomaly detection, threat response, incident reporting |

Every agent has two layers: a **fast rule/ML layer** for real-time decisions, and an **LLM layer** (Groq/LLaMA3-70B) that handles ambiguous situations, generates natural language explanations, and produces structured reports.

The entire system runs on an **async event bus** — agents communicate by publishing and subscribing to events, never calling each other directly. This makes it loosely coupled, extensible, and easy to reason about.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│         Fleet Map │ Analytics │ Scenarios │ Export           │
└──────────────────────────┬──────────────────────────────────┘
                           │ reads state
┌──────────────────────────▼──────────────────────────────────┐
│                      Orchestrator                            │
│    boots all components, wires dependencies, runs forever    │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌───▼────────┐ ┌──▼─────────┐
│  Simulator  │ │Coordinator │ │  Energy    │ │  Sentinel  │
│  (tick      │ │  Agent     │ │  Agent     │ │  Agent     │
│  engine)    │ │ Rules+LLM  │ │ QL+LLM     │ │ IF+LLM     │
└──────┬──────┘ └─────┬──────┘ └───┬────────┘ └──┬─────────┘
       │              │            │              │
       └──────────────┴────────────┴──────────────┘
                                │
                    ┌───────────▼────────────┐
                    │      Event Bus          │
                    │  (async pub/sub)         │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │      State Store        │
                    │  (in-memory + SQLite)   │
                    └────────────────────────┘
```

**Data flow on every tick:**
1. Simulator fires a `TICK` event with a full `SystemSnapshot`
2. All three agents receive the snapshot simultaneously
3. Each agent makes decisions and publishes result events
4. State Store is updated (battery levels, positions, task status)
5. Dashboard reads the latest State Store snapshot and re-renders

---

## 3. Folder Structure

```
amr_framework/
│
├── .env.example              ← Template for your environment variables
├── .env                      ← Your actual config (never commit this)
├── requirements.txt          ← Python dependencies
├── main.py                   ← Headless entry point (no dashboard)
├── run_tests.py              ← Standalone test runner (no pytest needed)
├── README.md                 ← This file
│
├── venv/                     ← Python virtual environment (auto-created)
│
├── config/                   ← Centralized configuration
│   ├── __init__.py
│   └── config_manager.py     ← Typed config with validation, .env loading
│
├── core/                     ← Framework infrastructure
│   ├── __init__.py
│   ├── models.py             ← All data types (AMRState, Task, Event, etc.)
│   ├── event_bus.py          ← Async pub/sub event system
│   ├── state_store.py        ← Shared memory + SQLite persistence
│   ├── base_agent.py         ← Abstract base class all agents inherit
│   ├── logger.py             ← Colored console + rotating file logging
│   ├── metrics_exporter.py   ← CSV/JSON export for analysis
│   └── orchestrator.py       ← System entry point — wires everything
│
├── agents/                   ← The three AI agents
│   ├── __init__.py
│   ├── coordinator_agent.py  ← Fleet coordination (452 lines)
│   ├── energy_agent.py       ← Battery management (456 lines)
│   └── sentinel_agent.py     ← Network security (544 lines)
│
├── simulation/               ← The virtual warehouse
│   ├── __init__.py
│   ├── simulator.py          ← Tick engine, AMR movement, task spawning
│   └── scenarios.py          ← 6 pre-built demo scenarios
│
├── dashboard/                ← Streamlit web interface
│   ├── __init__.py
│   ├── app.py                ← Page 1: Live Fleet Map (617 lines)
│   └── pages/
│       ├── __init__.py
│       ├── 1_Analytics.py    ← Page 2: Deep analytics + Q-table heatmap
│       ├── 2_Scenarios.py    ← Page 3: One-click demo scenarios
│       └── 3_Export.py       ← Page 4: Download reports and data
│
├── tests/                    ← Test suite (75 tests)
│   ├── __init__.py
│   ├── conftest.py           ← Pytest async config
│   ├── test_models.py        ← 18 tests for all data models
│   ├── test_state_store.py   ← 20 tests for state management
│   ├── test_event_bus.py     ← 6 tests for event system
│   └── test_agents.py        ← 31 tests for agent logic + ML
│
├── data/                     ← Auto-created at runtime
│   └── amr_history.db        ← SQLite database (battery/task history)
│
└── logs/                     ← Auto-created at runtime
    └── amr_system.log        ← Rotating log (5MB max, 5 backups)
```

---

## 4. File Reference

### `config/config_manager.py`
Single source of truth for every configurable value. Structured as nested dataclasses (`SimulationConfig`, `EnergyConfig`, `SecurityConfig`, `CoordinatorConfig`, `LLMConfig`). Loads from `.env` first, then applies runtime overrides. Validates all values on load — raises `AssertionError` with a clear message if anything is out of range.

### `core/models.py`
All shared data structures. Key types:
- `AMRState` — everything about a robot: id, name, position, battery, status, tasks, alerts
- `Task` — a work order: type (pickup/dropoff/inspect/charge), priority (1-5), target position
- `NetworkPacket` — simulated 5G packet: size, latency, signal strength, loss rate, anomaly flag
- `Event` — pub/sub message: event_type, payload dict, timestamp
- `SystemSnapshot` — point-in-time view of the entire fleet, published each tick
- `ChargingStation` — a docking station: position, occupant, availability

### `core/event_bus.py`
Async priority queue-based event bus. Agents call `bus.subscribe(EventType.X, handler)` and `bus.publish(event)`. Handlers run concurrently. Errors in one handler never crash others. Supports wildcard subscriptions (`subscribe_all`) and maintains a rolling history of recent events for the dashboard.

### `core/state_store.py`
Thread-safe in-memory store backed by SQLite for persistence. All agents read/write here. Key operations: `update_amr_battery`, `update_amr_status`, `update_amr_position`, `set_amr_compromised`, `assign_station`, `release_station`, `add_task`, `get_pending_tasks`. Records battery and task history for dashboard charts.

### `core/base_agent.py`
Abstract base class every agent inherits. Provides: `start()`, `teardown()`, `run()` (the main async loop), `on_tick()` (override this), `emit()` (publish events), `log()` (structured logging), `get_stats()` (performance counters).

### `core/orchestrator.py`
The system entry point. Instantiates everything in the right order, wires dependencies, launches all async tasks. Also exposes `get_status()` for the dashboard and `exporter` / `scenario_engine` for dashboard pages.

### `simulation/simulator.py`
The virtual warehouse floor. Every `tick_rate` seconds it:
1. Moves all MOVING AMRs toward their target (by `amr_speed` grid units)
2. Advances WORKING AMRs (completes task after `task_work_duration` seconds)
3. Spawns new tasks with probability `task_spawn_prob`
4. Publishes a `TICK` event with a fresh `SystemSnapshot`

AMR names come from a fixed list: **Atlas, Bolt, Cygnus, Delta, Echo, Falcon, Gemini** (up to 7 named AMRs; beyond that they get generic IDs).

### `core/metrics_exporter.py`
Exports simulation data at any point. Generates: fleet metrics CSV (per-tick battery/tasks), AMR performance CSV (per-robot summary), battery history CSV, security alerts CSV, Q-table JSON. Also generates a plain-text summary report suitable for project submissions.

---

## 5. The AMR Fleet

### Default Configuration (out of the box)

| Parameter | Default | Description |
|---|---|---|
| **Number of AMRs** | **5** | Atlas, Bolt, Cygnus, Delta, Echo |
| **Grid Size** | **20 × 20** | Warehouse floor in grid units |
| **Charging Stations** | **2** (= `num_amrs // 2`) | CS_01, CS_02 placed at fixed positions |
| **Tick Rate** | **1.0 second** | Time between simulation steps |
| **Task Spawn Probability** | **15% per tick** | Chance a new task appears each tick |
| **AMR Speed** | **1.0 unit/tick** | Movement speed across the grid |

### Battery Dynamics

| Status | Drain Rate |
|---|---|
| Moving | 0.3% per tick |
| Working | 0.5% per tick |
| Idle | 0.05% per tick |
| Charging | +2.0% per tick (gains) |

Every AMR starts at 100% battery. The Energy Agent monitors them all and intervenes before any reaches 0%.

### Scaling Up

To run more AMRs, either edit `.env`:
```bash
NUM_AMRS=7    # Up to 7 named, beyond that generic IDs
GRID_SIZE=30  # Bigger grid for more AMRs
```
Or pass overrides in code:
```python
orch = Orchestrator(config_overrides={"num_amrs": 7, "grid_size": 30})
```

---

## 6. The Three AI Agents

### Coordinator Agent (`coordinator_agent.py`)

**Responsibility:** Keeping the fleet productive and collision-free.

**Collision Detection:** Every tick, runs an O(n²) pairwise proximity check across all active AMRs.
- Distance < 2.0 units → **Warning**: negotiate which AMR yields
- Distance < 1.0 unit → **Emergency**: both AMRs stop immediately

**Task Assignment:** When a task is created (`TASK_CREATED` event), scores every available AMR:
```
score = (10 - distance_to_task) + (battery / 10) + (5 if IDLE else 0)
```
The highest-scoring available AMR gets the task.

**Fleet Rebalancing:** When idle AMRs cluster together (< 3.0 units apart), spreads them to improve warehouse coverage. When > 50% of the fleet is idle, asks the LLM for an optimal coverage strategy.

**Events published:** `TASK_ASSIGNED`, `COLLISION_WARNING`, `PATH_NEGOTIATION`, `TASK_DELEGATION`

---

### Energy Agent (`energy_agent.py`)

**Responsibility:** Preventing battery depletion using Q-learning.

**Q-Learning Setup:**
- **State space:** `(battery_bin, dist_to_station_bin, task_queue_bin)` → 4 × 3 × 2 = 24 states
- **Actions:** `CONTINUE_WORKING (0)`, `GO_CHARGE (1)`, `REDUCE_SPEED (2)`
- **Q-table shape:** (4, 3, 2, 3) → 72 state-action values
- **Algorithm:** ε-greedy (ε=0.15) with Bellman update (α=0.1, γ=0.9)

**Reward Structure:**

| Situation | Reward |
|---|---|
| Reached critical battery (≤10%) | −20 |
| Proactive charge when battery low | +5 |
| Premature charge with high battery | −3 |
| Continue working at healthy battery | +2 |
| Low battery and chose to keep working | −5 |
| Reduced speed to conserve energy | +1 |

**Pre-seeded Q-table:** Critical battery states are pre-biased toward GO_CHARGE, and high battery states toward CONTINUE, so the agent starts making good decisions immediately rather than exploring randomly.

**Events published:** `BATTERY_LOW`, `BATTERY_CRITICAL`, `CHARGE_STATION_ASSIGN`, `CHARGING_COMPLETE`, `TASK_DELEGATION`

---

### Sentinel Agent (`sentinel_agent.py`)

**Responsibility:** Detecting and responding to 5G network attacks.

**Attack Types Simulated:**

| Attack | Packet Size | Latency | Signal | Packet Loss |
|---|---|---|---|---|
| Normal | ~512 bytes | ~5 ms | ~−70 dBm | < 3% |
| PACKET_FLOOD | 40–80 bytes | ~100 ms | Normal | < 5% |
| JAMMING | Normal | 10–20 ms | −108 to −115 dBm | 70–99% |
| DATA_EXFIL | 40–65 KB | ~2.5 ms | Normal | < 2% |
| REPLAY_ATTACK | Normal | 7.5–12 ms | −98 to −105 dBm | 50–90% |

**Isolation Forest ML Pipeline:**
1. Collect 30+ normal packets per AMR (rolling window of 100)
2. Fit `IsolationForest(contamination=0.05)` on normal traffic
3. Score each new packet: normal traffic scores −0.40 to −0.50, attacks score −0.60 to −0.80
4. Warning threshold: score < −0.55 → alert
5. Critical threshold: score < −0.65 → quarantine AMR

**Quarantine logic:** Marks AMR as compromised in State Store, publishes `AMR_QUARANTINED`, resets the ML model. After 15 seconds, auto-restores the AMR and rebuilds its traffic baseline from scratch.

**Events published:** `INTRUSION_ALERT`, `AMR_QUARANTINED`, `AMR_RESTORED`

---

## 7. How the LLM Is Used

The LLM (Groq free tier, LLaMA3-70B-8192) is called at **8 specific decision points** across all three agents. Every call uses structured JSON prompts and parses the response back into Python objects — the LLM output directly drives system behavior, not just narration.

### Coordinator: 3 LLM Functions

**`_llm_negotiate_collision(a1, a2, dist)`**
Called when two AMRs have equal-priority tasks and are on a collision course. Sends both AMRs' full status (battery, task type, distance traveled, tasks completed) to LLaMA and asks which should yield. Returns `{"yield": "<name>", "reason": "<sentence>"}`.

**`_llm_pick_assignee(task, a1, a2)`**
Called when the two top-scored AMRs for a task are within 2 points of each other. Sends task details + both AMR profiles and asks for the best assignment. Returns `{"assign_to": "<name>", "reason": "<sentence>"}`.

**`_llm_fleet_rebalance(idle_amrs, snapshot)`**
Called when > 50% of the fleet is idle. Sends all idle AMRs' positions and battery levels, asks for optimal warehouse coverage positions. Returns a list of `{"amr": "<name>", "target_x": float, "target_y": float, "reason": "<sentence>"}` — each entry directly updates the AMR's movement target.

### Energy: 2 LLM Functions

**`_llm_schedule_charging(needs_charge, free_stations, snapshot)`**
Called when more AMRs need charging than stations are available. Sends the queue of AMRs with their battery levels and task priorities, asks which ones should charge first. Returns `{"charge_first": ["name1", "name2"], "reasoning": "<sentence>"}`.

**`_llm_generate_fleet_summary(snapshot)`**
Called every 15 ticks. Sends full fleet status and asks for a 2-sentence battery health narrative for the dashboard. Output is stored in `energy.llm_fleet_summary` and displayed on the Analytics page.

### Sentinel: 3 LLM Functions

**`_llm_incident_report(amr, packet, score, attack_type, tick)`**
Called immediately when an AMR is quarantined. Sends the anomaly evidence (score, packet features, attack type) and requests a structured incident report. Returns `{"severity", "attack_summary", "impact", "action_taken", "recommendation"}` — stored in the alert record and shown in the dashboard.

**`_llm_security_posture(snapshot)`**
Called every 20 ticks if any quarantines have occurred. Asks for a 2-sentence security posture summary based on incident history. Output stored in `sentinel.llm_security_summary`.

**`_llm_restoration_advisory(amr)`**
Called when a quarantined AMR is restored. Asks for a one-sentence monitoring recommendation for operators. Logged and shown in agent logs.

---

## 8. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A free Groq API key (get one in 30 seconds at [console.groq.com](https://console.groq.com))

### Step 1 — Clone and create virtual environment
```bash
git clone <your-repo-url>
cd amr_framework

python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment
```bash
cp .env.example .env
```
Open `.env` and set your Groq API key:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
All other settings have sensible defaults and can be left as-is for a first run.

### Step 4 — Verify everything works
```bash
python run_tests.py
# Expected: 75 passed | 0 failed
```

---

## 9. Running the System

### Option A — Full Dashboard (recommended)
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`. The Orchestrator starts automatically when the dashboard loads.

### Option B — Headless mode (no browser)
```bash
python main.py
```
Runs all agents in the terminal with colored log output. Good for testing on a server.

### Option C — Custom config
```bash
NUM_AMRS=7 GRID_SIZE=25 SIMULATION_TICK_RATE=0.5 streamlit run dashboard/app.py
```
Or edit `.env` permanently.

---

## 10. The Streamlit Dashboard — Complete Guide

The dashboard has **4 pages** accessible from the sidebar.

---

### Page 1: Fleet Map (`dashboard/app.py`)

The main live operations view. Auto-refreshes every 2 seconds.

**Top Header Strip:**
Shows 5 live metrics: Current Tick, Active AMRs, Fleet Battery %, Tasks in Queue, Total Security Alerts. These update in real time.

**Fleet Map (left panel):**
A `plotly` scatter plot of the 20×20 warehouse grid showing every AMR as a colored dot. Charging stations appear as star markers.
- Green dot = healthy AMR
- Yellow dot = low battery (< 20%)
- Red dot = quarantined / compromised
- Hover over any dot to see the AMR's full status

**AMR Status Cards (right panel):**
One card per AMR showing: name, battery percentage with color bar (green > 40%, yellow 20–40%, red < 20%), current status badge, tasks completed, and distance traveled.

**Live Agent Log (bottom):**
Scrolling feed of all agent decisions in real time. Color-coded by severity: green = INFO, yellow = WARNING, red = CRITICAL. This is where you see the LLM decisions appear with a 🤖 prefix.

**LLM Insights Panel:**
Shows the latest `llm_fleet_summary` from the Energy Agent and `llm_security_summary` from the Sentinel Agent — plain English AI-generated status reports.

---

### Page 2: Analytics (`dashboard/pages/1_Analytics.py`)

Deep-dive performance analysis. Auto-refreshes every 3 seconds.

**Fleet Performance Table:**
Full tabular view of all AMRs with: battery %, tasks completed/failed, success rate %, total distance, energy consumed, alerts triggered, and compromised status.

**AMR Capability Radar Chart:**
A radar/spider chart showing 5 dimensions per AMR — Battery, Tasks Done, Distance, Reliability, Security — all normalized to 0–100. Lets you instantly see which AMRs are overworked or underperforming.

**Q-Learning Analysis:**
- Three heatmaps (one per action: Continue, Go Charge, Reduce Speed) showing the Q-table values across (battery_level × distance_to_station) state space. Watch these change as the agent learns over time.
- Battery history line chart with a dashed critical threshold line at 20%
- Key stats: Total Reward accumulated, Charge Events triggered, Prevented Depletions

**Security Analysis:**
- Bar chart of alert counts by attack type (PACKET_FLOOD, JAMMING, DATA_EXFIL, REPLAY_ATTACK)
- Pie chart of alert severity distribution (critical vs warning)
- Scatter plot of anomaly scores over time with threshold lines marked

**Agent Performance Comparison:**
Two bar charts comparing all three agents on: decisions made per tick and events published per tick. Tells you which agent is the most active.

**Event Bus Statistics:**
Total events published, dispatched, handler errors, and dropped. Breakdown of the top 12 event types by frequency.

---

### Page 3: Scenarios (`dashboard/pages/2_Scenarios.py`)

One-click demo control panel for presentations.

**System Status Strip:**
Live tick, active AMRs, battery, task queue, and alerts — same as Fleet Map header.

**Scenario Cards:**
Six cards laid out in a 2-column grid. Each card shows the scenario name, a description of what it does, and which agent(s) it demonstrates. Click the button to launch:

| Scenario | What It Does | Best Demonstrates |
|---|---|---|
| **Normal Operations** | Baseline healthy fleet, tasks flowing normally | Overall system harmony |
| **Battery Crisis** | Forces 3 AMRs to 8–12% battery simultaneously | Energy Agent Q-learning + forced charging |
| **Collision Stress** | Moves all AMRs toward the center point | Coordinator collision avoidance + emergency stop |
| **Network Attack** | Triggers PACKET_FLOOD + JAMMING on 2 AMRs | Sentinel Isolation Forest + quarantine |
| **Task Surge** | Injects 10 high-priority tasks at once | Coordinator task assignment at scale |
| **Full Stress Test** | All of the above simultaneously | Everything at once — maximum chaos |

After launching a scenario, the result log appears below the button showing exactly what state changes were injected and what events were fired. Then switch to the Fleet Map tab to watch the agents react.

**Manual Controls:**
- **Inject Single Task** — choose task type (pickup/dropoff/inspect) and priority (1–5), fire it into the queue
- **Force AMR to Low Battery** — select any AMR and drop its battery to a custom % to watch the Energy Agent respond
- **Simulation Speed** — drag the tick rate slider (0.1s to 5.0s per tick) to speed up or slow down the simulation. Toggle "Task Burst Mode" to spawn tasks 3× faster.

---

### Page 4: Export (`dashboard/pages/3_Export.py`)

Download all simulation data for your project report.

**Summary Report (text):**
Auto-generated plain text report with fleet size, final battery averages, tasks completed, distance traveled, energy consumed, per-AMR breakdown, and security/Q-learning statistics. Click "Download Report (.txt)".

**AMR Performance CSV:**
One row per AMR with all performance metrics. Click "Download AMR Data (.csv)".

**Battery History CSV:**
Per-tick average battery level. Useful for plotting in Excel/Matplotlib. Click "Download Battery History (.csv)".

**Security Alerts CSV:**
Every security alert with tick, AMR name, attack type, severity, anomaly score, and packet features. Click "Download Alerts (.csv)".

**Q-Table JSON:**
Full Q-table state (raw values + heatmap data + stats) as JSON. Useful for analyzing learning progress. Click "Download Q-Table (.json)".

---

## 11. Demo Scenarios

For a project presentation, run through these in order:

**1. Start the dashboard and let it run for 30 seconds** — let the fleet find a rhythm, models start training.

**2. Launch Battery Crisis** — watch 3 AMRs turn yellow/red on the fleet map. The Energy Agent will route them to charging stations. If LLM is enabled, you'll see `🤖 LLM charging schedule: charge [X, Y] first` in the live log.

**3. Launch Network Attack** — watch one or two AMRs turn red and get quarantined. The Sentinel Agent fires an incident report. After 15 seconds, the AMRs restore automatically.

**4. Launch Task Surge** — watch the Coordinator assign 10 tasks in rapid succession. If LLM is enabled, some assignments will show `🤖 LLM task assignment override`.

**5. Launch Full Stress Test** — everything breaks at once. Show how all three agents cooperate under pressure without any central controller.

**6. Go to Analytics** — show the Q-table heatmaps (they will have evolved from the pre-seeded values), the radar chart showing which AMR is most efficient, and the security alert distribution.

**7. Go to Export** — download the summary report and show it as part of your project documentation.

---

## 12. Testing Guide

### Run all 75 tests
```bash
python run_tests.py
```

### Run a specific test suite
```bash
python run_tests.py models      # 18 tests — data models
python run_tests.py store       # 20 tests — state management
python run_tests.py eventbus    # 6 tests  — event bus
python run_tests.py energy      # 12 tests — Q-learning
python run_tests.py coordinator # 4 tests  — scoring + assignment
python run_tests.py sentinel    # 8 tests  — Isolation Forest + detection
python run_tests.py config      # 7 tests  — config validation
```

### What is tested

**Models (18 tests)**
- `Position.distance_to` — 3-4-5 triangle, self-distance, symmetry
- `AMRState` — default values, serialization
- `Task` — unique ID generation, status defaults
- `NetworkPacket` — feature vector length, anomaly flag
- `Event` — unique IDs, timestamp accuracy
- `ChargingStation` — occupancy defaults
- All enums — correct string values

**StateStore (20 tests)**
- AMR registration and retrieval
- Battery clamping (0–100%)
- Position tracking + distance accumulation
- Quarantine marking
- Charging station assign/release/find-nearest
- Task queue add/assign/complete lifecycle
- Snapshot generation

**EventBus (6 tests)**
- Subscribe and receive
- Wildcard subscriptions
- Unsubscribe removes handler
- Handler errors don't crash bus
- Published event counter
- Log history storage

**EnergyAgent (12 tests)**
- `_battery_bin` — all four ranges
- `_dist_bin` — all three ranges
- `_queue_bin` — empty vs non-empty
- Q-table shape: (4, 3, 2, 3)
- Q-table seeding: critical battery → GO_CHARGE bias
- Q-table seeding: high battery → CONTINUE bias
- Bellman update changes Q-value
- Stats dict has required keys
- Heatmap data shape

**CoordinatorAgent (4 tests)**
- Closer AMR scores higher
- Higher battery scores higher (equidistant)
- IDLE status gets bonus over MOVING
- `_assign_task` updates StateStore correctly

**SentinelAgent (8 tests)**
- Normal packet not anomalous
- Attack packet flagged anomalous
- Model trains after MIN_SAMPLES
- PACKET_FLOOD scores more negative than normal
- JAMMING scores more negative than normal
- DATA_EXFIL scores more negative than normal
- Quarantine marks AMR compromised in StateStore
- Security stats dict has required keys

**ConfigManager (7 tests)**
- Defaults load correctly
- Overrides apply to correct fields
- Flat dict export has all keys
- Validation rejects num_amrs=0
- Validation rejects grid_size=5
- Critical threshold must be < low threshold

### Using pytest (optional)
```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 13. Configuration Reference

All settings can be changed in `.env` or passed as `config_overrides` to `Orchestrator`.

### Simulation Settings
| Variable | Default | Description |
|---|---|---|
| `NUM_AMRS` | `5` | Number of robots (1–20) |
| `GRID_SIZE` | `20` | Warehouse grid dimensions (10–100) |
| `SIMULATION_TICK_RATE` | `1.0` | Seconds per simulation tick (0.1–10) |
| `TASK_SPAWN_PROB` | `0.15` | Probability of new task per tick (0–1) |

### Energy Agent Settings
| Variable | Default | Description |
|---|---|---|
| `ENERGY_LOW_THRESHOLD` | `20.0` | % battery that triggers low battery warning |
| `ENERGY_CRITICAL_THRESHOLD` | `10.0` | % battery that forces charging |

### Security Settings
| Variable | Default | Description |
|---|---|---|
| `ANOMALY_SENSITIVITY` | `0.05` | Isolation Forest contamination rate. Lower = more sensitive to attacks but more false positives |
| `ATTACK_BASE_PROB` | `0.005` | Probability of attack per AMR per tick. Raise to 0.05 for a more attack-heavy demo |

### LLM Settings
| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key (required for LLM features) |
| `LLM_MODEL` | `llama3-70b-8192` | Groq model. Alternatives: `llama3-8b-8192` (faster/less smart), `mixtral-8x7b-32768` (longer context) |
| `LLM_MAX_TOKENS` | `512` | Max response length |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## 14. Adding a New AI Agent

The framework is designed for extension. Here is the exact process for adding a new agent, step by step.

### Step 1 — Create the file

Create `agents/your_agent.py`. Copy this skeleton:

```python
from core.base_agent import BaseAgent
from core.models import AlertSeverity, Event, EventType, SystemSnapshot

class YourAgent(BaseAgent):
    def __init__(self, bus, store, config=None):
        super().__init__("YourAgent", bus, store, config)
        # your state here

    async def setup(self) -> None:
        # Subscribe to events you care about
        self.bus.subscribe(EventType.TICK, self._on_something)
        await self.log("YourAgent online.", AlertSeverity.INFO)

    async def on_tick(self, snapshot: SystemSnapshot) -> None:
        # This is called every tick automatically by BaseAgent.run()
        # Put your main logic here
        for amr in snapshot.amrs.values():
            pass  # do something with each AMR

    async def _on_something(self, event: Event) -> None:
        self._events_received += 1
        # react to an event

    def get_stats(self) -> dict:
        base = super().get_stats()
        base["my_custom_metric"] = 42
        return base
```

### Step 2 — Register in the Orchestrator

Open `core/orchestrator.py` and add three lines:

```python
# In imports
from agents.your_agent import YourAgent

# In __init__
self.your_agent = YourAgent(self.bus, self.store, self.config)

# In start()
await self.your_agent.start()
self._tasks.append(asyncio.create_task(self.your_agent.run(), name="your_agent"))

# In stop()
await self.your_agent.teardown()
```

### Step 3 — Define any new event types

If your agent needs to publish events that don't exist yet, add them to `core/models.py`:

```python
class EventType(str, Enum):
    # ... existing events ...
    YOUR_NEW_EVENT = "your_new_event"
```

### Step 4 — Add your LLM function (optional)

```python
async def _llm_your_decision(self, context_data) -> dict:
    prompt = f"""You are an AI ... 
    
    Context: {context_data}
    
    Respond in EXACTLY this JSON format:
    {{"decision": "<value>", "reason": "<sentence>"}}"""
    
    try:
        response = self._groq_client.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        self._llm_calls += 1
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return {"decision": "default", "reason": "LLM unavailable"}
```

### Step 5 — Write tests

Add a new test section to `run_tests.py`:

```python
@test("YourAgent", "does something correctly")
def _(loop):
    from core.event_bus import EventBus
    from agents.your_agent import YourAgent
    agent = YourAgent(EventBus(), ms(), {})
    assert agent is not None
    # assert your logic here
```

Run `python run_tests.py youragent` to test just your new suite.

---

## 15. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq API + LLaMA3-70B-8192 | Natural language decisions, reports |
| **ML / Anomaly Detection** | scikit-learn IsolationForest | Unsupervised network traffic analysis |
| **Reinforcement Learning** | NumPy Q-table (custom) | Battery-aware charging decisions |
| **Async Framework** | Python asyncio | Concurrent agent execution |
| **Persistence** | SQLite3 (stdlib) | Battery history, task history |
| **Dashboard** | Streamlit | Multi-page web UI |
| **Charts** | Plotly | Interactive fleet map, heatmaps, radar |
| **Data** | Pandas | Tables and CSV export |
| **Config** | python-dotenv + dataclasses | Typed, validated configuration |
| **Logging** | Python logging + ANSI colors | Colored console + rotating log file |
| **Testing** | Custom runner (no pytest needed) | 75 tests, 100% passing |

---

## Project Statistics

| Metric | Value |
|---|---|
| Total Python files | 31 |
| Total lines of code | ~4,300 |
| Test count | 75 tests |
| Test pass rate | 100% |
| LLM call sites | 8 (across 3 agents) |
| Event types | 20 |
| Dashboard pages | 4 |
| Demo scenarios | 6 |
| Charging stations (default) | 2 |
| AMR fleet size (default) | 5 |
| AMR names | Atlas, Bolt, Cygnus, Delta, Echo |

---

*Built as a B.Tech final year project demonstrating multi-agent systems, reinforcement learning, unsupervised ML, and LLM integration in a real-world robotics context.*