"""
dashboard/pages/2_Scenarios.py
================================
Scenario Control Panel — Demo Scenarios for Presentations
"""

import sys, os, asyncio, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="AMR Scenarios", page_icon="🎬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
  .stApp { background: #0a0e1a; }
  .scenario-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 12px;
  }
  .scenario-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
  .scenario-desc  { font-size: 0.8rem; color: #94a3b8; line-height: 1.5; }
  .scenario-tag {
    display: inline-block; font-size: 0.63rem; padding: 2px 8px;
    border-radius: 20px; margin-right: 5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .tag-coordinator { background: rgba(56,189,248,0.15);  color: #38bdf8; }
  .tag-energy      { background: rgba(250,204,21,0.15);  color: #facc15; }
  .tag-sentinel    { background: rgba(239,68,68,0.15);   color: #ef4444; }
  .tag-all         { background: rgba(167,139,250,0.15); color: #a78bfa; }
  .result-box {
    background: rgba(17,24,39,0.8); border: 1px solid #1e2d3d;
    border-radius: 8px; padding: 12px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #94a3b8; white-space: pre-wrap;
    max-height: 200px; overflow-y: auto;
  }
  .legend-box {
    background: rgba(17,24,39,0.7); border: 1px solid #1e2d3d;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.76rem; color: #94a3b8; line-height: 1.7; margin-bottom: 12px;
  }
  .legend-title {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #38bdf8; margin-bottom: 6px; font-weight: 600;
  }
  div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)


def get_orch():
    try:
        from dashboard.app import get_orchestrator
        orch, loop = get_orchestrator()
        return orch, loop
    except Exception:
        st.error("Main dashboard not running. Open it first: `streamlit run dashboard/app.py`")
        st.stop()


def run_async(coro, loop):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=10)
    except Exception as e:
        return f"Error: {e}"


orch, loop = get_orch()
store      = orch.store

if "scenario_engine" not in st.session_state:
    from simulation.scenarios import ScenarioEngine
    st.session_state.scenario_engine = ScenarioEngine(
        simulator=orch.simulator, store=orch.store, bus=orch.bus,
    )
engine = st.session_state.scenario_engine

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 Demo Scenario Control Panel")

st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ What are scenarios?</div>
  Scenarios are pre-built situations that stress-test the AI agents.
  Click <b>Launch</b> on any card, then switch to the <b>Fleet Map</b> page to watch the agents react in real time.
  Each scenario is designed to demonstrate a specific agent's capabilities.<br><br>
  <b>Tip for presentations:</b> Launch <i>Battery Crisis</i> first (watch the Energy Agent),
  then <i>Network Attack</i> (watch the Sentinel), then <i>Full Stress Test</i> (all three at once).
</div>
""", unsafe_allow_html=True)

# ── Status strip ──────────────────────────────────────────────────────────────
snap = store.get_snapshot(orch.simulator.tick)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⏱️ Current Tick",    orch.simulator.tick)
c2.metric("🤖 Active AMRs",     snap.active_amrs)
c3.metric("🔋 Fleet Battery",   f"{snap.fleet_avg_battery:.1f}%")
c4.metric("📦 Tasks in Queue",  snap.tasks_in_queue)
c5.metric("🚨 Total Alerts",    snap.total_alerts)

st.markdown("---")

# ── Scenario cards ────────────────────────────────────────────────────────────
from simulation.scenarios import Scenario, SCENARIO_DESCRIPTIONS

SCENARIO_META = {
    Scenario.NORMAL_OPERATIONS: {
        "icon": "🏭", "color": "#38bdf8",
        "what_happens": "Healthy baseline. Robots pick up and drop off goods normally. Good as a starting point before launching stress scenarios.",
        "watch_for":    "Coordinator assigns tasks. Energy Agent monitors battery. Sentinel trains its ML model on clean traffic.",
        "tags": [("Coordinator", "coordinator"), ("Energy", "energy")],
    },
    Scenario.BATTERY_CRISIS: {
        "icon": "🪫", "color": "#f59e0b",
        "what_happens": "Forces 3 robots to 8–12% battery simultaneously, simulating a power failure situation.",
        "watch_for":    "Energy Agent sends them to charging stations. Watch the Q-learning heatmap shift. LLM arbitrates who charges first if both stations are occupied.",
        "tags": [("Energy Agent", "energy"), ("Q-Learning", "energy")],
    },
    Scenario.COLLISION_STRESS: {
        "icon": "💥", "color": "#a78bfa",
        "what_happens": "Moves all robots toward the grid centre simultaneously, forcing near-collisions.",
        "watch_for":    "Coordinator's O(n²) proximity check fires. Robots within 2 units trigger PATH_NEGOTIATION (one yields). Within 1 unit = emergency stop.",
        "tags": [("Coordinator", "coordinator"), ("Path Negotiation", "coordinator")],
    },
    Scenario.NETWORK_ATTACK: {
        "icon": "🚨", "color": "#ef4444",
        "what_happens": "Injects PACKET_FLOOD and JAMMING attacks on 2 robots simultaneously.",
        "watch_for":    "Sentinel's Isolation Forest scores drop. Score below −0.55 → WARNING. Below −0.65 → robot QUARANTINED (red X on map). LLM generates incident report. After 15 s, robot auto-restores.",
        "tags": [("Sentinel", "sentinel"), ("Isolation Forest", "sentinel")],
    },
    Scenario.TASK_SURGE: {
        "icon": "📦", "color": "#4ade80",
        "what_happens": "Injects 10 high-priority tasks (priority 4–5) into the queue at once.",
        "watch_for":    "Coordinator scores all available robots and assigns tasks rapidly. LLM breaks ties when two robots score within 2 points of each other.",
        "tags": [("Coordinator", "coordinator"), ("Task Assignment", "coordinator")],
    },
    Scenario.FULL_STRESS_TEST: {
        "icon": "⚡", "color": "#fb923c",
        "what_happens": "All of the above at the same time — battery crisis + collisions + network attack + task surge.",
        "watch_for":    "All three agents firing simultaneously. Tests inter-agent coordination under maximum load. Best used after individual scenarios to show composure.",
        "tags": [("All Agents", "all"), ("Max Chaos", "all")],
    },
}

if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = {}

scenarios    = list(Scenario)
col_left, col_right = st.columns(2)

for i, scenario in enumerate(scenarios):
    meta = SCENARIO_META[scenario]
    col  = col_left if i % 2 == 0 else col_right

    with col:
        tags_html = " ".join(
            f'<span class="scenario-tag tag-{tc}">{tl}</span>'
            for tl, tc in meta["tags"]
        )
        st.markdown(f"""
<div class="scenario-card">
  <div class="scenario-title">{meta['icon']} {scenario.value.replace('_',' ').title()}</div>
  <div class="scenario-desc">
    <b>What happens:</b> {meta['what_happens']}<br>
    <b>Watch for:</b> <span style="color:#38bdf8">{meta['watch_for']}</span>
  </div>
  <div style="margin-top:10px;">{tags_html}</div>
</div>
""", unsafe_allow_html=True)

        if st.button(
            f"▶  Launch  {meta['icon']}",
            key=f"btn_{scenario.value}",
            use_container_width=True,
            type="primary" if scenario == Scenario.FULL_STRESS_TEST else "secondary",
        ):
            with st.spinner(f"Running {scenario.value}..."):
                result = run_async(engine.run(scenario), loop)
            st.session_state.scenario_result[scenario.value] = result
            st.success("✅ Scenario launched! Switch to Fleet Map to watch the agents react.")

        if scenario.value in st.session_state.scenario_result:
            res = st.session_state.scenario_result[scenario.value]
            st.markdown(f'<div class="result-box">{res}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Manual controls ───────────────────────────────────────────────────────────
st.markdown("### 🕹️ Manual Controls")
st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ What these controls do</div>
  These let you poke the simulation by hand — useful for live demos where you want to trigger a specific reaction
  without running a full scenario.
</div>
""", unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown("**📦 Inject Single Task**")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:8px;'>"
        "Adds one task directly to the queue. The Coordinator assigns it to the highest-scoring available robot.</div>",
        unsafe_allow_html=True,
    )
    from core.models import TaskType
    task_type_str = st.selectbox("Type", [t.value for t in TaskType if t != TaskType.CHARGE],
                                 key="manual_task_type")
    priority = st.slider("Priority (1=low, 5=critical)", 1, 5, 3, key="manual_priority")
    if st.button("🚀 Inject Task", use_container_width=True, key="manual_inject"):
        run_async(orch.simulator.inject_task(task_type=TaskType(task_type_str), priority=priority), loop)
        st.success(f"Injected {task_type_str} (priority {priority})")

with mc2:
    st.markdown("**🪫 Force Robot to Low Battery**")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:8px;'>"
        "Drops a robot's battery to a specific % — watch the Energy Agent react and send it to charge.</div>",
        unsafe_allow_html=True,
    )
    amrs_list    = store.get_all_amrs()
    amr_names    = [a.name for a in amrs_list]
    selected_amr = st.selectbox("Robot", amr_names, key="force_battery_amr")
    target_batt  = st.slider("Set battery to (%)", 1, 30, 8, key="force_battery_val")
    if st.button("🪫 Set Battery", use_container_width=True, key="force_battery_btn"):
        amr = next((a for a in amrs_list if a.name == selected_amr), None)
        if amr:
            run_async(store.update_amr_battery(amr.amr_id, float(target_batt)), loop)
            st.warning(f"{selected_amr} battery → {target_batt}% — watch Fleet Map!")

with mc3:
    st.markdown("**⚙️ Simulation Speed**")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:8px;'>"
        "Tick rate = seconds between simulation steps. Slower = easier to watch individual decisions.</div>",
        unsafe_allow_html=True,
    )
    speed = st.slider("Tick rate (s)", 0.1, 5.0, value=orch.simulator.tick_rate, step=0.1,
                      key="manual_speed")
    if speed != orch.simulator.tick_rate:
        orch.simulator.set_tick_rate(speed)
    burst = st.toggle("⚡ Task Burst Mode (3× spawn rate)", key="manual_burst")
    orch.simulator.set_task_burst(burst)
    if burst:
        st.info("Tasks spawning 3× faster — great for stressing the Coordinator")

if engine.active_scenario:
    st.markdown("---")
    st.markdown(
        f"<div style='color:#38bdf8;font-size:0.82rem;'>"
        f"🔄 Last scenario run: <b>{engine.active_scenario.value}</b></div>",
        unsafe_allow_html=True,
    )