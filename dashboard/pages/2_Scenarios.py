"""
dashboard/pages/2_Scenarios.py
================================
Scenario Control Panel — Demo Scenarios for Presentations

One-click scenario execution with live result logging.
Perfect for demonstrating agent capabilities in your B.Tech presentation.

Author: AMR Multi-Agent Framework
"""

import sys
import os
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AMR Scenarios",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
  .stApp { background: #0a0e1a; }
  .scenario-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
  }
  .scenario-card:hover { border-color: #38bdf8; }
  .scenario-title {
    font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 6px;
  }
  .scenario-desc {
    font-size: 0.8rem; color: #94a3b8; line-height: 1.5;
  }
  .scenario-tag {
    display: inline-block; font-size: 0.65rem; padding: 2px 8px;
    border-radius: 20px; margin-right: 6px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .tag-coordinator { background: rgba(56,189,248,0.15); color: #38bdf8; }
  .tag-energy      { background: rgba(250,204,21,0.15);  color: #facc15; }
  .tag-sentinel    { background: rgba(239,68,68,0.15);   color: #ef4444; }
  .tag-all         { background: rgba(167,139,250,0.15); color: #a78bfa; }
  .result-box {
    background: rgba(17,24,39,0.8); border: 1px solid #1e2d3d;
    border-radius: 8px; padding: 12px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #94a3b8; white-space: pre-wrap; max-height: 200px;
    overflow-y: auto;
  }
  div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_orch():
    try:
        from dashboard.app import get_orchestrator
        orch, loop = get_orchestrator()
        return orch, loop
    except Exception:
        st.error("Main dashboard not running. Please start: `streamlit run dashboard/app.py`")
        st.stop()


def run_async(coro, loop):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=10)
    except Exception as e:
        return f"Error: {e}"


orch, loop  = get_orch()
store       = orch.store

# Lazy-init scenario engine
if "scenario_engine" not in st.session_state:
    from simulation.scenarios import ScenarioEngine
    st.session_state.scenario_engine = ScenarioEngine(
        simulator=orch.simulator,
        store=orch.store,
        bus=orch.bus,
    )

engine = st.session_state.scenario_engine

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("🎬 Demo Scenario Control Panel")
st.markdown(
    "<div style='color:#64748b; font-size:0.85rem; margin-bottom:24px;'>"
    "Pre-built scenarios to demonstrate agent capabilities. "
    "Trigger any scenario and watch the agents react in real time on the Fleet Map.</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# System status strip
# ------------------------------------------------------------------

snap = store.get_snapshot(orch.simulator.tick)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Tick",       orch.simulator.tick)
c2.metric("Active AMRs",        snap.active_amrs)
c3.metric("Fleet Battery",      f"{snap.fleet_avg_battery:.1f}%")
c4.metric("Tasks in Queue",     snap.tasks_in_queue)
c5.metric("Total Alerts",       snap.total_alerts)

st.markdown("---")

# ------------------------------------------------------------------
# Scenario cards
# ------------------------------------------------------------------

from simulation.scenarios import Scenario, SCENARIO_DESCRIPTIONS

SCENARIO_META = {
    Scenario.NORMAL_OPERATIONS: {
        "icon":  "🏭",
        "tags":  [("Coordinator", "coordinator"), ("Energy", "energy")],
        "color": "#38bdf8",
    },
    Scenario.BATTERY_CRISIS: {
        "icon":  "🪫",
        "tags":  [("Energy Agent", "energy"), ("Q-Learning", "energy")],
        "color": "#f59e0b",
    },
    Scenario.COLLISION_STRESS: {
        "icon":  "💥",
        "tags":  [("Coordinator", "coordinator"), ("Path Negotiation", "coordinator")],
        "color": "#a78bfa",
    },
    Scenario.NETWORK_ATTACK: {
        "icon":  "🚨",
        "tags":  [("Sentinel", "sentinel"), ("Isolation Forest", "sentinel")],
        "color": "#ef4444",
    },
    Scenario.TASK_SURGE: {
        "icon":  "📦",
        "tags":  [("Coordinator", "coordinator"), ("Task Assignment", "coordinator")],
        "color": "#4ade80",
    },
    Scenario.FULL_STRESS_TEST: {
        "icon":  "⚡",
        "tags":  [("All Agents", "all"), ("Max Chaos", "all")],
        "color": "#fb923c",
    },
}

# Session state for results
if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = {}

# 2-column grid of scenario cards
scenarios = list(Scenario)
col_left, col_right = st.columns(2)

for i, scenario in enumerate(scenarios):
    meta = SCENARIO_META[scenario]
    col  = col_left if i % 2 == 0 else col_right

    with col:
        tags_html = " ".join(
            f'<span class="scenario-tag tag-{tag_class}">{tag_label}</span>'
            for tag_label, tag_class in meta["tags"]
        )
        st.markdown(
            f"""
            <div class="scenario-card">
              <div class="scenario-title">{meta['icon']} {scenario.value.replace('_', ' ').title()}</div>
              <div class="scenario-desc">{SCENARIO_DESCRIPTIONS[scenario]}</div>
              <div style="margin-top:10px;">{tags_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_key = f"btn_{scenario.value}"
        if st.button(
            f"▶  Launch Scenario",
            key=btn_key,
            use_container_width=True,
            type="primary" if scenario == Scenario.FULL_STRESS_TEST else "secondary",
        ):
            with st.spinner(f"Executing {scenario.value}..."):
                result = run_async(engine.run(scenario), loop)
            st.session_state.scenario_result[scenario.value] = result
            st.success(f"✅ Scenario launched! Watch the Fleet Map tab for agent reactions.")

        # Show result if available
        if scenario.value in st.session_state.scenario_result:
            result_text = st.session_state.scenario_result[scenario.value]
            st.markdown(
                f'<div class="result-box">{result_text}</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")

# ------------------------------------------------------------------
# Manual controls section
# ------------------------------------------------------------------

st.markdown("### 🕹️ Manual Controls")
st.markdown(
    "<div style='color:#64748b; font-size:0.8rem; margin-bottom:16px;'>"
    "Fine-grained controls for live experimentation.</div>",
    unsafe_allow_html=True,
)

mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown("**Inject Single Task**")
    from core.models import TaskType
    task_type_str = st.selectbox(
        "Type", [t.value for t in TaskType if t != TaskType.CHARGE],
        key="manual_task_type",
    )
    priority = st.slider("Priority", 1, 5, 3, key="manual_priority")
    if st.button("🚀 Inject Task", use_container_width=True, key="manual_inject"):
        result = run_async(
            orch.simulator.inject_task(
                task_type=TaskType(task_type_str),
                priority=priority,
            ),
            loop,
        )
        st.success(f"Task injected: {task_type_str} (priority {priority})")

with mc2:
    st.markdown("**Force AMR to Low Battery**")
    amrs_list = store.get_all_amrs()
    amr_names = [a.name for a in amrs_list]
    selected_amr = st.selectbox("AMR", amr_names, key="force_battery_amr")
    target_battery = st.slider("Battery %", 1, 30, 8, key="force_battery_val")
    if st.button("🪫 Set Battery", use_container_width=True, key="force_battery_btn"):
        amr = next((a for a in amrs_list if a.name == selected_amr), None)
        if amr:
            result = run_async(
                store.update_amr_battery(amr.amr_id, float(target_battery)),
                loop,
            )
            st.warning(f"{selected_amr} battery → {target_battery}%")

with mc3:
    st.markdown("**Simulation Speed**")
    speed = st.slider(
        "Tick Rate (s)", 0.1, 5.0,
        value=orch.simulator.tick_rate,
        step=0.1,
        key="manual_speed",
    )
    if speed != orch.simulator.tick_rate:
        orch.simulator.set_tick_rate(speed)
    burst = st.toggle("Task Burst Mode", key="manual_burst")
    orch.simulator.set_task_burst(burst)
    if burst:
        st.info("Tasks spawning 3× faster")

# ------------------------------------------------------------------
# Active scenario status
# ------------------------------------------------------------------

if engine.active_scenario:
    st.markdown("---")
    st.markdown(
        f"<div style='color:#38bdf8; font-size:0.85rem;'>"
        f"🔄 Last scenario: <b>{engine.active_scenario.value}</b></div>",
        unsafe_allow_html=True,
    )