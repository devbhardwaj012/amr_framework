"""
dashboard/app.py
================
Streamlit Dashboard — AMR Multi-Agent Framework

Real-time visualization of:
  - Fleet map (AMR positions, charging stations, paths)
  - Battery levels per AMR
  - Agent activity logs
  - Security alerts (Sentinel)
  - Q-learning heatmap (Energy Agent)
  - System metrics over time

Run: streamlit run dashboard/app.py

Author: AMR Multi-Agent Framework
"""

import sys
import os
import time
import asyncio
import threading
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/amr_system.log"),
        logging.StreamHandler(),
    ]
)

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AMR Fleet Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
  }
  .stApp { background: #0a0e1a; }
  
  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
  }
  .alert-critical {
    background: rgba(239, 68, 68, 0.1);
    border-left: 3px solid #ef4444;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.8rem;
  }
  .alert-warning {
    background: rgba(245, 158, 11, 0.1);
    border-left: 3px solid #f59e0b;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.8rem;
  }
  .alert-info {
    background: rgba(56, 189, 248, 0.08);
    border-left: 3px solid #38bdf8;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.8rem;
  }
  .status-online  { color: #4ade80; }
  .status-warning { color: #f59e0b; }
  .status-danger  { color: #ef4444; }
  
  div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8 !important;
  }
  .section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #64748b;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #1e2d3d;
    padding-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state & Orchestrator startup
# ------------------------------------------------------------------

@st.cache_resource
def get_orchestrator():
    """
    Create and start the orchestrator in a background thread.
    Cached globally so only one instance runs.
    """
    from core.orchestrator import Orchestrator

    orch = Orchestrator()
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(orch.start())
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True, name="orchestrator")
    thread.start()

    # Give it a moment to initialize
    time.sleep(2.0)
    return orch, loop


# ------------------------------------------------------------------
# Helper: run async from sync context
# ------------------------------------------------------------------

def run_async(coro, loop):
    """Submit a coroutine to the background event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=5)
    except Exception as e:
        st.warning(f"Async call failed: {e}")
        return None


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def build_fleet_map(store, grid_size: int = 20) -> go.Figure:
    """Build the 2D fleet map with AMR positions and stations."""
    fig = go.Figure()

    # Background grid
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=grid_size, y1=grid_size,
        fillcolor="rgba(17,24,39,0.8)", line=dict(color="#1e2d3d", width=1),
    )
    # Grid lines
    for i in range(0, grid_size + 1, 5):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=grid_size,
                      line=dict(color="#1e2d3d", width=0.5, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=i, x1=grid_size, y1=i,
                      line=dict(color="#1e2d3d", width=0.5, dash="dot"))

    # Charging stations
    stations = store.get_all_stations()
    for s in stations:
        color = "#f59e0b" if s.is_occupied else "#4ade80"
        fig.add_trace(go.Scatter(
            x=[s.position.x], y=[s.position.y],
            mode="markers+text",
            marker=dict(symbol="square", size=18, color=color,
                        line=dict(color="white", width=1)),
            text=[f"⚡{s.station_id}"],
            textposition="top center",
            textfont=dict(size=9, color=color),
            name="Charging Station",
            showlegend=False,
            hovertemplate=(
                f"<b>{s.station_id}</b><br>"
                f"{'Occupied by ' + s.occupant_id if s.is_occupied else 'Free'}"
                "<extra></extra>"
            ),
        ))

    # AMRs
    STATUS_COLORS = {
        "idle":        "#38bdf8",
        "moving":      "#a78bfa",
        "working":     "#4ade80",
        "charging":    "#f59e0b",
        "low_battery": "#fb923c",
        "error":       "#ef4444",
        "quarantined": "#ef4444",
    }
    STATUS_SYMBOLS = {
        "idle":        "circle",
        "moving":      "arrow-right",
        "working":     "star",
        "charging":    "circle",
        "low_battery": "triangle-up",
        "error":       "x",
        "quarantined": "x",
    }

    amrs = store.get_all_amrs()
    for amr in amrs:
        color  = STATUS_COLORS.get(amr.status.value, "#94a3b8")
        symbol = STATUS_SYMBOLS.get(amr.status.value, "circle")

        # Draw target line if moving
        if amr.target_position and amr.status.value == "moving":
            fig.add_trace(go.Scatter(
                x=[amr.position.x, amr.target_position.x],
                y=[amr.position.y, amr.target_position.y],
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.4,
                showlegend=False,
                hoverinfo="skip",
            ))

        # AMR marker
        battery_icon = "🔋" if amr.battery > 20 else "🪫"
        fig.add_trace(go.Scatter(
            x=[amr.position.x], y=[amr.position.y],
            mode="markers+text",
            marker=dict(
                symbol=symbol, size=20, color=color,
                line=dict(color="white", width=1.5),
                opacity=0.3 if amr.is_compromised else 1.0,
            ),
            text=[f"{amr.name}"],
            textposition="bottom center",
            textfont=dict(size=9, color=color),
            name=amr.name,
            showlegend=False,
            hovertemplate=(
                f"<b>{amr.name} ({amr.amr_id})</b><br>"
                f"Status: {amr.status.value}<br>"
                f"Battery: {amr.battery:.1f}%<br>"
                f"Position: ({amr.position.x:.1f}, {amr.position.y:.1f})<br>"
                f"Tasks done: {amr.tasks_completed}<br>"
                f"{'🚨 QUARANTINED' if amr.is_compromised else ''}"
                "<extra></extra>"
            ),
        ))

        # Battery ring overlay (colored border shows battery level)
        if amr.battery < 25:
            fig.add_trace(go.Scatter(
                x=[amr.position.x], y=[amr.position.y],
                mode="markers",
                marker=dict(symbol="circle-open", size=30, color="#ef4444", line=dict(width=2)),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.9)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-0.5, grid_size + 0.5], showgrid=False, zeroline=False,
                   tickfont=dict(color="#4b5563", size=9)),
        yaxis=dict(range=[-0.5, grid_size + 0.5], showgrid=False, zeroline=False,
                   tickfont=dict(color="#4b5563", size=9), scaleanchor="x"),
        height=460,
        hoverlabel=dict(bgcolor="#1f2937", font_size=12, font_color="white"),
    )
    return fig


def build_battery_chart(store) -> go.Figure:
    """Bar chart of all AMR battery levels."""
    amrs = store.get_all_amrs()
    names    = [a.name for a in amrs]
    batteries = [a.battery for a in amrs]
    colors   = ["#ef4444" if b <= 20 else "#f59e0b" if b <= 40 else "#4ade80"
                for b in batteries]

    fig = go.Figure(go.Bar(
        x=names, y=batteries,
        marker_color=colors,
        text=[f"{b:.0f}%" for b in batteries],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.add_hline(y=20, line_dash="dash", line_color="#ef4444", opacity=0.5,
                  annotation_text="Critical", annotation_font_color="#ef4444")
    fig.add_hline(y=40, line_dash="dash", line_color="#f59e0b", opacity=0.5,
                  annotation_text="Low", annotation_font_color="#f59e0b")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        yaxis=dict(range=[0, 110], gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=10, b=0),
        height=220,
        showlegend=False,
    )
    return fig


def build_battery_history_chart(store) -> go.Figure:
    """Line chart of fleet average battery over time."""
    history = store.get_battery_history()
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)
    fig = go.Figure(go.Scatter(
        x=df["tick"], y=df["avg"],
        mode="lines", fill="tozeroy",
        line=dict(color="#38bdf8", width=2),
        fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        yaxis=dict(range=[0, 105], gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        xaxis=dict(title="Tick", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=5, b=0),
        height=180,
    )
    return fig


def build_qtable_heatmap(energy_agent) -> go.Figure:
    """Heatmap of Q-learning table values."""
    data = energy_agent.get_qtable_heatmap_data()
    # Show best action per state
    q_arr = np.array(data["q_values"])   # (battery, dist, actions)
    best_actions = np.argmax(q_arr, axis=2)  # (battery, dist)
    best_q = np.max(q_arr, axis=2)

    fig = go.Figure(go.Heatmap(
        z=best_q,
        x=data["dist_labels"],
        y=data["battery_labels"],
        colorscale="RdYlGn",
        text=[[data["action_labels"][a] for a in row] for row in best_actions],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="Battery: %{y}<br>Distance: %{x}<br>Best action: %{text}<br>Q-value: %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#94a3b8"), title=dict(text="Q", font=dict(color="#94a3b8"))),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=5, b=0),
        height=220,
    )
    return fig


def severity_html(severity: str, message: str, agent: str, ts: float) -> str:
    """Render a log entry as styled HTML."""
    icons = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
    icon  = icons.get(severity, "•")
    t     = time.strftime("%H:%M:%S", time.localtime(ts))
    css   = f"alert-{severity}"
    return (
        f'<div class="{css}">'
        f'<span style="opacity:0.6;font-size:0.7rem;">{t} [{agent}]</span><br>'
        f'{icon} {message}'
        f'</div>'
    )


# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------

def main():
    orch, loop = get_orchestrator()
    store      = orch.store
    simulator  = orch.simulator

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 10px 0 20px 0;">
          <div style="font-size:2rem;">🤖</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; color:#38bdf8; font-weight:700;">
            AMR FLEET CONTROL
          </div>
          <div style="font-size:0.65rem; color:#4b5563; letter-spacing:0.1em;">
            PRIVATE 5G MULTI-AGENT
          </div>
        </div>
        """, unsafe_allow_html=True)

        status = orch.get_status()

        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tick", status["tick"])
        with col2:
            snap = store.get_snapshot(status["tick"])
            st.metric("Active AMRs", snap.active_amrs)

        st.markdown('<div class="section-header">Simulation Controls</div>', unsafe_allow_html=True)
        speed = st.slider("Tick Speed (s)", 0.2, 3.0, simulator.tick_rate, 0.1,
                          help="Seconds between simulation ticks")
        if speed != simulator.tick_rate:
            simulator.set_tick_rate(speed)

        task_burst = st.toggle("Task Burst Mode", value=False,
                               help="Spawn tasks 3x faster")
        simulator.set_task_burst(task_burst)

        st.markdown('<div class="section-header">Inject Task</div>', unsafe_allow_html=True)
        from core.models import TaskType
        task_type_str = st.selectbox("Task Type", [t.value for t in TaskType
                                                    if t != TaskType.CHARGE])
        priority = st.slider("Priority", 1, 5, 3)
        if st.button("🚀 Inject Task", use_container_width=True):
            task_type = TaskType(task_type_str)
            run_async(simulator.inject_task(task_type=task_type, priority=priority), loop)
            st.success("Task injected!")

        st.markdown('<div class="section-header">Agent Stats</div>', unsafe_allow_html=True)
        c_stats = orch.coordinator.get_stats()
        e_stats = orch.energy.get_stats()
        s_stats = orch.sentinel.get_stats()

        for name, stats in [("Coordinator", c_stats), ("Energy", e_stats), ("Sentinel", s_stats)]:
            st.markdown(
                f'<div style="font-size:0.75rem; color:#94a3b8; margin:2px 0;">'
                f'<b style="color:#38bdf8">{name}</b> — '
                f'{stats["decisions_made"]} decisions, {stats["events_published"]} events'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Bus Stats</div>', unsafe_allow_html=True)
        bus_stats = status["bus_stats"]
        st.markdown(
            f'<div style="font-size:0.75rem; color:#94a3b8;">'
            f'Published: {bus_stats.get("published",0)} | '
            f'Dispatched: {bus_stats.get("dispatched",0)} | '
            f'Dropped: {bus_stats.get("dropped",0)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        refresh = st.slider("Auto-refresh (s)", 1, 10, 2)

    # ------------------------------------------------------------------
    # Main layout — 3 columns
    # ------------------------------------------------------------------
    snapshot = store.get_snapshot(status["tick"])

    # ── Top KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Fleet Avg Battery", f"{snapshot.fleet_avg_battery:.1f}%")
    k2.metric("Active AMRs",        f"{snapshot.active_amrs}/{len(snapshot.amrs)}")
    k3.metric("Tasks in Queue",     snapshot.tasks_in_queue)
    k4.metric("Total Alerts",       snapshot.total_alerts)
    k5.metric("Simulation Tick",    status["tick"])

    st.markdown("---")

    # ── Row 1: Fleet map | Battery levels ──
    col_map, col_bat = st.columns([3, 2])

    with col_map:
        st.markdown("#### 🗺️ Fleet Map")
        st.plotly_chart(
            build_fleet_map(store, orch.config.get("grid_size", 20)),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_bat:
        st.markdown("#### 🔋 Battery Levels")
        st.plotly_chart(build_battery_chart(store), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("#### 📈 Fleet Avg Battery History")
        st.plotly_chart(build_battery_history_chart(store), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Row 2: Agent logs | Q-table | Security ──
    col_log, col_q, col_sec = st.columns(3)

    with col_log:
        st.markdown("#### 📡 Agent Activity")
        logs = orch.bus.get_agent_logs(n=20)
        log_html = "".join([
            severity_html(l.severity.value, l.message, l.agent_name, l.timestamp)
            for l in reversed(logs)
        ])
        st.markdown(
            f'<div style="height:300px; overflow-y:auto; '
            f'background:rgba(17,24,39,0.6); border-radius:8px; padding:8px;">'
            f'{log_html}</div>',
            unsafe_allow_html=True,
        )

    with col_q:
        st.markdown("#### 🧠 Q-Learning Table")
        st.markdown(
            "<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px;'>"
            "Best action per (battery, distance) state</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_qtable_heatmap(orch.energy), use_container_width=True,
                        config={"displayModeBar": False})
        q_stats = orch.energy.get_qtable_stats()
        mc1, mc2 = st.columns(2)
        mc1.metric("Charge Events",    q_stats["charge_events"])
        mc2.metric("Prevented Depletions", q_stats["prevented_depletions"])

    with col_sec:
        st.markdown("#### 🛡️ Security Monitor")
        sec_stats = orch.sentinel.get_security_stats()
        sc1, sc2 = st.columns(2)
        sc1.metric("Packets Analyzed",  sec_stats["packets_analyzed"])
        sc2.metric("Anomalies Found",   sec_stats["anomalies_detected"])
        sc1.metric("Quarantines",       sec_stats["quarantines_issued"])
        sc2.metric("Active Attacks",    sec_stats["active_attacks"])

        st.markdown("**Recent Alerts**")
        alerts = orch.sentinel.get_recent_alerts(8)
        if alerts:
            alert_html = "".join([
                severity_html(
                    a["severity"],
                    f"[{a['attack_type']}] {a['amr_name']} — score: {a['score']:.3f}",
                    "Sentinel",
                    time.time(),
                )
                for a in reversed(alerts)
            ])
            st.markdown(
                f'<div style="height:180px; overflow-y:auto; '
                f'background:rgba(17,24,39,0.6); border-radius:8px; padding:8px;">'
                f'{alert_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#4ade80; font-size:0.85rem;">✅ No threats detected</div>',
                unsafe_allow_html=True,
            )

    # ── Row 3: AMR detail table ──
    st.markdown("#### 🤖 Fleet Status Table")
    amrs = store.get_all_amrs()
    if amrs:
        rows = []
        for a in amrs:
            rows.append({
                "Name":     a.name,
                "ID":       a.amr_id,
                "Status":   a.status.value.upper(),
                "Battery":  f"{a.battery:.1f}%",
                "Position": f"({a.position.x:.1f}, {a.position.y:.1f})",
                "Task":     a.current_task.task_type.value if a.current_task else "—",
                "Done":     a.tasks_completed,
                "Dist":     f"{a.total_distance:.1f}",
                "Alerts":   a.alerts_triggered,
                "Security": "🚨 QUARANTINED" if a.is_compromised else "✅ OK",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------
    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()