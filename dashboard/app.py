"""
dashboard/app.py
================
Page 1 — Live Fleet Map

Sections:
  - Sidebar: system status, simulation controls, quick agent stats
  - Top KPIs: battery, active AMRs, tasks, alerts, tick
  - Fleet Map: 2D grid with AMR positions and charging stations
  - Battery Levels: horizontal bar chart per AMR
  - Battery History: fleet average over time
  - Agent Activity Log: live scrolling log
  - Q-Learning Table: best-action heatmap
  - Security Monitor: live alerts
  - Fleet Status Table: full per-AMR data

Run: streamlit run dashboard/app.py
"""

import sys, os, time, asyncio, threading, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Logging (file must exist before basicConfig)
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/amr_system.log"),
        logging.StreamHandler(),
    ],
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Fleet Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
  .stApp { background: #0a0e1a; }

  .alert-critical {
    background: rgba(239,68,68,0.1); border-left: 3px solid #ef4444;
    border-radius: 4px; padding: 8px 12px; margin: 4px 0; font-size: 0.78rem;
  }
  .alert-warning {
    background: rgba(245,158,11,0.1); border-left: 3px solid #f59e0b;
    border-radius: 4px; padding: 8px 12px; margin: 4px 0; font-size: 0.78rem;
  }
  .alert-info {
    background: rgba(56,189,248,0.08); border-left: 3px solid #38bdf8;
    border-radius: 4px; padding: 8px 12px; margin: 4px 0; font-size: 0.78rem;
  }
  .legend-box {
    background: rgba(17,24,39,0.7); border: 1px solid #1e2d3d;
    border-radius: 8px; padding: 12px 14px;
    font-size: 0.76rem; color: #94a3b8; line-height: 1.7; margin-bottom: 10px;
  }
  .legend-title {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #38bdf8; margin-bottom: 6px; font-weight: 600;
  }
  .section-header {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.15em;
    color: #64748b; margin: 14px 0 6px 0;
    border-bottom: 1px solid #1e2d3d; padding-bottom: 4px;
  }
  div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8 !important; }
  div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Orchestrator (one instance, cached for the session) ───────────────────────
@st.cache_resource
def get_orchestrator():
    from core.orchestrator import Orchestrator
    orch = Orchestrator()
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(orch.start())
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True, name="orchestrator")
    thread.start()
    time.sleep(2.0)          # give agents time to start
    return orch, loop


def run_async(coro, loop):
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=5)
    except Exception as e:
        st.warning(f"Async call failed: {e}")
        return None


# ── Chart builders ────────────────────────────────────────────────────────────

def build_fleet_map(store, grid_size: int = 20) -> go.Figure:
    """2D warehouse grid with AMR positions and charging stations."""
    fig = go.Figure()

    # Grid background
    fig.add_shape(type="rect", x0=0, y0=0, x1=grid_size, y1=grid_size,
                  fillcolor="rgba(17,24,39,0.8)", line=dict(color="#1e2d3d", width=1))
    for i in range(0, grid_size + 1, 5):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=grid_size,
                      line=dict(color="#1e2d3d", width=0.5, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=i, x1=grid_size, y1=i,
                      line=dict(color="#1e2d3d", width=0.5, dash="dot"))

    # Charging stations  ⚡
    for s in store.get_all_stations():
        color = "#f59e0b" if s.is_occupied else "#4ade80"
        fig.add_trace(go.Scatter(
            x=[s.position.x], y=[s.position.y],
            mode="markers+text",
            marker=dict(symbol="square", size=18, color=color,
                        line=dict(color="white", width=1)),
            text=[f"⚡{s.station_id}"],
            textposition="top center",
            textfont=dict(size=9, color=color),
            name=f"Station {s.station_id}",
            showlegend=False,
            hovertemplate=(
                f"<b>{s.station_id}</b><br>"
                f"{'Occupied by ' + str(s.occupant_id) if s.is_occupied else '✅ Free'}"
                "<extra></extra>"
            ),
        ))

    # AMRs — colour and symbol encode status
    STATUS_COLOR = {
        "idle":        "#38bdf8",   # sky blue
        "moving":      "#a78bfa",   # purple
        "working":     "#4ade80",   # green
        "charging":    "#f59e0b",   # amber
        "low_battery": "#fb923c",   # orange
        "error":       "#ef4444",   # red
        "quarantined": "#ef4444",   # red
    }
    STATUS_SYMBOL = {
        "idle":        "circle",
        "moving":      "triangle-up",
        "working":     "star",
        "charging":    "square",
        "low_battery": "diamond",
        "error":       "x",
        "quarantined": "x",
    }

    for a in store.get_all_amrs():
        st_val  = a.status.value
        color   = STATUS_COLOR.get(st_val, "#94a3b8")
        symbol  = STATUS_SYMBOL.get(st_val, "circle")
        size    = 22 if a.is_compromised else 16

        fig.add_trace(go.Scatter(
            x=[a.position.x], y=[a.position.y],
            mode="markers+text",
            marker=dict(symbol=symbol, size=size, color=color,
                        line=dict(color="white", width=1.5 if a.is_compromised else 0.8)),
            text=[a.name],
            textposition="top center",
            textfont=dict(size=9, color=color),
            name=a.name,
            showlegend=False,
            hovertemplate=(
                f"<b>{a.name}</b> ({a.amr_id})<br>"
                f"Status: {st_val}<br>"
                f"Battery: {a.battery:.1f}%<br>"
                f"Position: ({a.position.x:.1f}, {a.position.y:.1f})<br>"
                f"Tasks done: {a.tasks_completed}<br>"
                f"Security: {'🚨 QUARANTINED' if a.is_compromised else '✅ OK'}"
                "<extra></extra>"
            ),
        ))

        # Draw line to target if moving
        if a.target_position and st_val == "moving":
            fig.add_shape(type="line",
                x0=a.position.x, y0=a.position.y,
                x1=a.target_position.x, y1=a.target_position.y,
                line=dict(color=color, width=1, dash="dot"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(17,24,39,0.8)",
        xaxis=dict(range=[-0.5, grid_size + 0.5], showgrid=False,
                   tickfont=dict(color="#4b5563", size=8)),
        yaxis=dict(range=[-0.5, grid_size + 0.5], showgrid=False,
                   tickfont=dict(color="#4b5563", size=8), scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        dragmode=False,
    )
    return fig


def build_battery_chart(store) -> go.Figure:
    """Horizontal battery bars per AMR, colour-coded by level."""
    amrs = store.get_all_amrs()
    if not amrs:
        return go.Figure()

    names   = [a.name for a in amrs]
    bats    = [a.battery for a in amrs]
    colors  = [
        "#ef4444" if b < 15 else "#f59e0b" if b < 35 else "#4ade80"
        for b in bats
    ]

    fig = go.Figure(go.Bar(
        y=names, x=bats, orientation="h",
        marker_color=colors,
        text=[f"{b:.1f}%" for b in bats], textposition="inside",
        textfont=dict(color="white", size=10),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=20, line_dash="dash", line_color="#f59e0b",
                  annotation_text="⚠️ Low", annotation_font_color="#f59e0b",
                  annotation_font_size=9)
    fig.add_vline(x=10, line_dash="dot",  line_color="#ef4444",
                  annotation_text="🚨 Critical", annotation_font_color="#ef4444",
                  annotation_font_size=9)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(range=[0, 105], tickfont=dict(color="#64748b", size=8),
                   gridcolor="#1e2d3d"),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=0, b=0), height=160,
    )
    return fig


def build_battery_history_chart(store) -> go.Figure:
    """Fleet average battery over time."""
    hist = store.get_battery_history()
    if len(hist) < 2:
        return go.Figure()
    df = pd.DataFrame(hist)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["tick"], y=df["avg"],
        mode="lines", line=dict(color="#38bdf8", width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        name="Fleet avg",
    ))
    fig.add_hline(y=20, line_dash="dash", line_color="#f59e0b")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#4b5563", size=8), gridcolor="#1e2d3d"),
        yaxis=dict(tickfont=dict(color="#4b5563", size=8), range=[0, 105],
                   gridcolor="#1e2d3d"),
        margin=dict(l=0, r=0, t=0, b=0), height=130, showlegend=False,
    )
    return fig


def build_qtable_heatmap(energy_agent) -> go.Figure:
    """Best-action Q-table heatmap (collapsed across queue dimension)."""
    try:
        q_data = energy_agent.get_qtable_heatmap_data()
        q_arr  = np.array(q_data["q_values"])          # (bat, dist, actions)
        best   = np.argmax(q_arr, axis=2)               # (bat, dist)
        labels = [["Continue", "Go Charge", "Reduce Speed"][int(v)] for row in best for v in row]
        labels = np.array(labels).reshape(best.shape)

        CMAP = [[0, "#38bdf8"], [0.5, "#4ade80"], [1.0, "#f59e0b"]]
        fig = go.Figure(go.Heatmap(
            z=best,
            x=q_data["dist_labels"],
            y=q_data["battery_labels"],
            colorscale=CMAP,
            showscale=False,
            text=labels,
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            hovertemplate="Battery: %{y}<br>Distance: %{x}<br>Best Action: %{text}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
            xaxis=dict(title="Dist to Station", tickfont=dict(color="#64748b", size=8)),
            yaxis=dict(title="Battery Level",   tickfont=dict(color="#64748b", size=8)),
            margin=dict(l=0, r=0, t=0, b=0), height=180,
        )
        return fig
    except Exception:
        return go.Figure()


def severity_html(severity: str, message: str, agent: str, ts: float) -> str:
    icons = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
    icon  = icons.get(severity, "•")
    t     = time.strftime("%H:%M:%S", time.localtime(ts))
    css   = f"alert-{severity}"
    return (
        f'<div class="{css}">'
        f'<span style="opacity:0.55;font-size:0.68rem;">{t} [{agent}]</span><br>'
        f'{icon} {message}'
        f'</div>'
    )


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    orch, loop = get_orchestrator()
    store      = orch.store
    simulator  = orch.simulator

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 18px 0;">
          <div style="font-size:2rem;">🤖</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.95rem;
                      color:#38bdf8;font-weight:700;">AMR FLEET CONTROL</div>
          <div style="font-size:0.6rem;color:#4b5563;letter-spacing:0.1em;">
            PRIVATE 5G · MULTI-AGENT AI</div>
        </div>
        """, unsafe_allow_html=True)

        status = orch.get_status()
        snap   = store.get_snapshot(status["tick"])

        # ── System status ──
        st.markdown('<div class="section-header">📡 System Status</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Tick",        status["tick"],     help="Simulation heartbeat counter")
        c2.metric("Active AMRs", snap.active_amrs,   help="Robots not quarantined or in error")

        llm_on = bool(os.getenv("GROQ_API_KEY"))
        st.markdown(
            f"<div style='font-size:0.72rem;color:{'#4ade80' if llm_on else '#64748b'};margin:4px 0 8px 0;'>"
            f"{'🤖 LLM: Groq/LLaMA3 ENABLED' if llm_on else '⚪ LLM: disabled (no GROQ_API_KEY)'}</div>",
            unsafe_allow_html=True,
        )

        # ── Simulation controls ──
        st.markdown('<div class="section-header">⚙️ Simulation Controls</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.7rem;color:#4b5563;margin-bottom:6px;'>"
            "Tick speed = how fast the simulation runs. Lower = faster robots, quicker decisions.</div>",
            unsafe_allow_html=True,
        )
        speed = st.slider("Tick Speed (s)", 0.2, 3.0, simulator.tick_rate, 0.1)
        if speed != simulator.tick_rate:
            simulator.set_tick_rate(speed)

        task_burst = st.toggle("⚡ Task Burst Mode",
                               help="Spawns tasks 3× faster — great for stress testing the Coordinator")
        simulator.set_task_burst(task_burst)

        # ── Inject task ──
        st.markdown('<div class="section-header">📦 Inject Task</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.7rem;color:#4b5563;margin-bottom:6px;'>"
            "Manually add a task to the queue. The Coordinator will assign it to the best available robot.</div>",
            unsafe_allow_html=True,
        )
        from core.models import TaskType
        task_type_str = st.selectbox(
            "Task Type",
            [t.value for t in TaskType if t != TaskType.CHARGE],
            help="PICKUP/DROPOFF/INSPECT/NAVIGATE/PATROL",
        )
        priority = st.slider("Priority", 1, 5, 3,
                             help="1 = lowest, 5 = critical — higher priority tasks get assigned first")
        if st.button("🚀 Inject Task", use_container_width=True):
            run_async(simulator.inject_task(task_type=TaskType(task_type_str), priority=priority), loop)
            st.success("Task injected!")

        # ── Agent quick stats ──
        st.markdown('<div class="section-header">🧠 Agent Stats</div>', unsafe_allow_html=True)
        for name, stats in [
            ("🗺️ Coordinator", orch.coordinator.get_stats()),
            ("⚡ Energy",       orch.energy.get_stats()),
            ("🛡️ Sentinel",     orch.sentinel.get_stats()),
        ]:
            st.markdown(
                f"<div style='font-size:0.72rem;color:#94a3b8;margin:2px 0;'>"
                f"<b style='color:#38bdf8'>{name}</b> — "
                f"{stats['decisions_made']} decisions · {stats['events_published']} events</div>",
                unsafe_allow_html=True,
            )

        # ── Bus stats ──
        st.markdown('<div class="section-header">📡 Event Bus</div>', unsafe_allow_html=True)
        bs = status["bus_stats"]
        st.markdown(
            f"<div style='font-size:0.72rem;color:#94a3b8;'>"
            f"Published: {bs.get('published',0)} · Dispatched: {bs.get('dispatched',0)} · "
            f"Dropped: {bs.get('dropped',0)}</div>",
            unsafe_allow_html=True,
        )

        refresh = st.slider("🔄 Auto-refresh (s)", 1, 10, 2,
                            help="How often this page reloads to show latest data")

    # ── Main layout ───────────────────────────────────────────────────────────
    snapshot = store.get_snapshot(status["tick"])

    # ── KPI strip ──
    st.markdown("#### 📊 Live Fleet Metrics")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:8px;'>"
        "These 5 numbers summarise the entire fleet right now. They update every refresh cycle.</div>",
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🔋 Fleet Avg Battery",  f"{snapshot.fleet_avg_battery:.1f}%",
              help="Average battery % across all active robots")
    k2.metric("🤖 Active AMRs",        f"{snapshot.active_amrs}/{len(snapshot.amrs)}",
              help="Active / total. A robot is 'inactive' only if quarantined or in error")
    k3.metric("📦 Tasks in Queue",     snapshot.tasks_in_queue,
              help="Pending tasks waiting to be assigned to a robot")
    k4.metric("🚨 Total Alerts",       snapshot.total_alerts,
              help="Cumulative security alerts raised by the Sentinel agent")
    k5.metric("⏱️ Simulation Tick",    status["tick"],
              help="Each tick = one simulation step (default 1 second)")

    st.markdown("---")

    # ── LLM Insights — always visible, right at the top ──────────────────────
    st.markdown("#### 🤖 LLM Insights — Groq / LLaMA3-70B")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:8px;'>"
        "Natural language summaries generated live by the AI agents. "
        "Energy Agent updates every 5 ticks · Sentinel updates every 8 ticks.</div>",
        unsafe_allow_html=True,
    )
    llm_fleet    = getattr(orch.energy,   "llm_fleet_summary",    "")
    llm_security = getattr(orch.sentinel, "llm_security_summary", "")

    ins1, ins2 = st.columns(2)
    with ins1:
        if llm_fleet:
            st.info(f"⚡ **Energy Agent says:**\n\n{llm_fleet}")
        else:
            st.info(
                "⚡ **Energy Agent**\n\n"
                "⏳ Waiting for first LLM summary… appears after ~5 simulation ticks."
            )
    with ins2:
        if llm_security:
            st.warning(f"🛡️ **Sentinel Agent says:**\n\n{llm_security}")
        else:
            st.warning(
                "🛡️ **Sentinel Agent**\n\n"
                "⏳ Waiting for first security report… appears after ~8 simulation ticks."
            )

    st.markdown("---")

    # ── Row 1: Fleet map + battery charts ──
    col_map, col_bat = st.columns([3, 2])

    with col_map:
        st.markdown("#### 🗺️ Fleet Map")
        st.markdown("""
<div class="legend-box">
  <div class="legend-title">Map Legend</div>
  <b>Robots</b> — shape and colour show current status:<br>
  <span style="color:#38bdf8">● Circle</span> = Idle (waiting for a task) &nbsp;
  <span style="color:#a78bfa">▲ Triangle</span> = Moving (navigating) &nbsp;
  <span style="color:#4ade80">★ Star</span> = Working (executing task)<br>
  <span style="color:#f59e0b">■ Square</span> = Charging &nbsp;
  <span style="color:#fb923c">◆ Diamond</span> = Low battery (&lt;20%) &nbsp;
  <span style="color:#ef4444">✕ X</span> = Quarantined / Error<br><br>
  <b>Stations</b> — <span style="color:#4ade80">■ Green square</span> = free &nbsp;
  <span style="color:#f59e0b">■ Amber square</span> = occupied<br>
  Dotted lines show a moving robot's planned path to its target.
  <b>Hover over any robot or station for details.</b>
</div>
""", unsafe_allow_html=True)
        st.plotly_chart(
            build_fleet_map(store, orch.config.get("grid_size", 20)),
            use_container_width=True, config={"displayModeBar": False},
        )

    with col_bat:
        st.markdown("#### 🔋 Battery Levels")
        st.markdown("""
<div class="legend-box">
  <div class="legend-title">Battery Bar Colours</div>
  <span style="color:#4ade80">■ Green</span> = healthy (&gt;35%) &nbsp;
  <span style="color:#f59e0b">■ Amber</span> = low (15–35%) &nbsp;
  <span style="color:#ef4444">■ Red</span> = critical (&lt;15%)<br>
  The Energy Agent sends robots to charge before they hit the dashed lines.
</div>
""", unsafe_allow_html=True)
        st.plotly_chart(build_battery_chart(store), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("#### 📈 Fleet Avg Battery History")
        st.markdown(
            "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:4px;'>"
            "Average battery % across all robots over time. "
            "Dip near 20% = Energy Agent activated charging.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_battery_history_chart(store), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Row 2: Log | Q-table | Security ──
    col_log, col_q, col_sec = st.columns(3)

    with col_log:
        st.markdown("#### 📡 Agent Activity Log")
        st.markdown("""
<div class="legend-box">
  <div class="legend-title">Log Entry Legend</div>
  <span style="color:#ef4444">🚨 Red</span> = Critical (quarantine, collision, depletion)<br>
  <span style="color:#f59e0b">⚠️ Amber</span> = Warning (low battery, path conflict)<br>
  <span style="color:#38bdf8">ℹ️ Blue</span> = Info (task assigned, charged, restored)<br>
  🤖 prefix = decision made by the LLM (Groq/LLaMA3)
</div>
""", unsafe_allow_html=True)
        logs     = orch.bus.get_agent_logs(n=20)
        log_html = "".join([
            severity_html(l.severity.value, l.message, l.agent_name, l.timestamp)
            for l in reversed(logs)
        ])
        st.markdown(
            f'<div style="height:280px;overflow-y:auto;background:rgba(17,24,39,0.6);'
            f'border-radius:8px;padding:8px;">{log_html}</div>',
            unsafe_allow_html=True,
        )

    with col_q:
        st.markdown("#### 🧠 Q-Learning: Best Action")
        st.markdown("""
<div class="legend-box">
  <div class="legend-title">How to read this heatmap</div>
  Each cell = what the Energy Agent has learned is the best action in that situation.<br>
  <span style="color:#38bdf8">■ Blue</span> = Continue Working &nbsp;
  <span style="color:#4ade80">■ Green</span> = Go Charge &nbsp;
  <span style="color:#f59e0b">■ Amber</span> = Reduce Speed<br>
  Rows = battery level · Columns = distance to nearest free station.
  Cells change as the agent learns from reward signals.
</div>
""", unsafe_allow_html=True)
        st.plotly_chart(build_qtable_heatmap(orch.energy), use_container_width=True,
                        config={"displayModeBar": False})
        q_stats = orch.energy.get_qtable_stats()
        mc1, mc2 = st.columns(2)
        mc1.metric("Charge Events",       q_stats["charge_events"],
                   help="Times a robot was sent to charge by the Q-learning agent")
        mc2.metric("Prevented Depletions", q_stats["prevented_depletions"],
                   help="Robots saved from 0% battery")

    with col_sec:
        st.markdown("#### 🛡️ Security Monitor")
        st.markdown("""
<div class="legend-box">
  <div class="legend-title">How Sentinel works</div>
  Monitors 5G packets using <b>Isolation Forest</b> ML.<br>
  Normal score ≈ −0.45 · Warning &lt; −0.55 · Quarantine &lt; −0.65<br>
  Quarantined robots are isolated for 15 s then auto-restored.
</div>
""", unsafe_allow_html=True)
        sec_stats = orch.sentinel.get_security_stats()
        sc1, sc2 = st.columns(2)
        sc1.metric("📦 Packets",      sec_stats["packets_analyzed"],
                   help="Total network packets inspected by Isolation Forest")
        sc2.metric("⚠️ Anomalies",    sec_stats["anomalies_detected"],
                   help="Packets flagged as suspicious (score below −0.55)")
        sc1.metric("🔒 Quarantines",  sec_stats["quarantines_issued"],
                   help="Robots fully isolated (score below −0.65)")
        sc2.metric("💥 Active Attacks", sec_stats["active_attacks"],
                   help="Attack simulations running right now")

        st.markdown("**Recent Alerts**")
        alerts = orch.sentinel.get_recent_alerts(8)
        if alerts:
            alert_html = "".join([
                severity_html(
                    a["severity"],
                    f"[{a['attack_type']}] {a['amr_name']} — score: {a['score']:.3f}",
                    "Sentinel", time.time(),
                )
                for a in reversed(alerts)
            ])
            st.markdown(
                f'<div style="height:170px;overflow-y:auto;background:rgba(17,24,39,0.6);'
                f'border-radius:8px;padding:8px;">{alert_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#4ade80;font-size:0.82rem;">✅ Network is clean — no threats detected</div>',
                unsafe_allow_html=True,
            )

    # ── Row 3: Fleet status table ──
    st.markdown("#### 🤖 Full Fleet Status Table")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;margin-bottom:6px;'>"
        "Complete live snapshot of every robot. "
        "<b>Done</b> = tasks completed · <b>Dist</b> = grid units travelled · "
        "<b>Alerts</b> = security alerts triggered · <b>Security</b> = network status.</div>",
        unsafe_allow_html=True,
    )
    amrs = store.get_all_amrs()
    if amrs:
        rows = []
        for a in amrs:
            rows.append({
                "🤖 Name":      a.name,
                "ID":           a.amr_id,
                "Status":       a.status.value.upper(),
                "🔋 Battery":   f"{a.battery:.1f}%",
                "📍 Position":  f"({a.position.x:.1f}, {a.position.y:.1f})",
                "📦 Task":      a.current_task.task_type.value if a.current_task else "—",
                "✅ Done":      a.tasks_completed,
                "📏 Dist":      f"{a.total_distance:.1f}",
                "🚨 Alerts":    a.alerts_triggered,
                "🔒 Security":  "🚨 QUARANTINED" if a.is_compromised else "✅ OK",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Auto-refresh ──
    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()