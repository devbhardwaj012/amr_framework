"""
dashboard/pages/1_Analytics.py
================================
Analytics Deep-Dive Page — Streamlit multi-page app

Shows:
  - Per-AMR performance breakdown (tasks, distance, energy consumed)
  - Q-learning evolution over time (reward curve, policy heatmap)
  - Security incident timeline
  - Event bus throughput
  - Agent decision counts
  - Coordinator collision/negotiation log

Accessed via the sidebar navigation in Streamlit.

Author: AMR Multi-Agent Framework
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AMR Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
  .stApp { background: #0a0e1a; }
  .section-title {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em;
    color: #64748b; border-bottom: 1px solid #1e2d3d; padding-bottom: 6px; margin: 20px 0 12px 0;
  }
  div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Get orchestrator from cache
# ------------------------------------------------------------------

def get_orch():
    try:
        from dashboard.app import get_orchestrator
        orch, loop = get_orchestrator()
        return orch, loop
    except Exception:
        st.error("Main dashboard not running. Please start from `streamlit run dashboard/app.py`")
        st.stop()


orch, loop = get_orch()
store  = orch.store
status = orch.get_status()
tick   = status["tick"]

st.title("📊 AMR Fleet Analytics")
st.caption(f"Tick {tick} — refreshes every 3 seconds")

# ------------------------------------------------------------------
# Row 1: Fleet performance table
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Fleet Performance</div>', unsafe_allow_html=True)

amrs = store.get_all_amrs()
if amrs:
    rows = []
    for a in amrs:
        efficiency = (
            round(a.tasks_completed / max(a.energy_consumed, 0.1), 3)
            if a.energy_consumed > 0 else 0
        )
        rows.append({
            "AMR":            a.name,
            "Status":         a.status.value,
            "Battery %":      round(a.battery, 1),
            "Tasks Done":     a.tasks_completed,
            "Tasks Failed":   a.tasks_failed,
            "Distance (u)":   round(a.total_distance, 1),
            "Energy Used %":  round(a.energy_consumed, 1),
            "Tasks/Energy":   efficiency,
            "Alerts":         a.alerts_triggered,
            "Compromised":    "🚨" if a.is_compromised else "✅",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# Row 2: Radar charts — AMR comparison
# ------------------------------------------------------------------
st.markdown('<div class="section-title">AMR Capability Radar</div>', unsafe_allow_html=True)

if amrs:
    max_tasks = max(a.tasks_completed for a in amrs) or 1
    max_dist  = max(a.total_distance  for a in amrs) or 1
    max_batt  = 100.0

    fig_radar = go.Figure()
    categories = ["Battery", "Tasks Done", "Distance", "Reliability", "Security"]
    colors = ["#38bdf8", "#4ade80", "#a78bfa", "#fb923c", "#f43f5e"]

    for i, a in enumerate(amrs):
        reliability = 1 - (a.tasks_failed / max(a.tasks_completed + a.tasks_failed, 1))
        security    = 1 - (a.alerts_triggered / max(sum(x.alerts_triggered for x in amrs), 1))
        values = [
            a.battery / max_batt,
            a.tasks_completed / max_tasks,
            a.total_distance  / max_dist,
            reliability,
            security,
        ]
        values_norm = [round(v * 100, 1) for v in values]
        fig_radar.add_trace(go.Scatterpolar(
            r=values_norm + [values_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=a.name,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)].replace("#", "rgba(") + ",0.1)",
            opacity=0.8,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e2d3d",
                           tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
            bgcolor="rgba(17,24,39,0.8)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------
# Row 3: Energy Agent deep dive
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Energy Agent — Q-Learning Analysis</div>', unsafe_allow_html=True)

col_q1, col_q2, col_q3 = st.columns(3)

q_stats = orch.energy.get_qtable_stats()
col_q1.metric("Total Reward",          q_stats["total_reward"])
col_q2.metric("Charge Events",         q_stats["charge_events"])
col_q3.metric("Prevented Depletions",  q_stats["prevented_depletions"])

# Q-table heatmap per action
q_data = orch.energy.get_qtable_heatmap_data()
q_arr  = np.array(q_data["q_values"])  # (battery, dist, actions)

col_h1, col_h2, col_h3 = st.columns(3)
action_names = q_data["action_labels"]

for col, action_idx, action_name in zip(
    [col_h1, col_h2, col_h3],
    [0, 1, 2],
    action_names,
):
    z = q_arr[:, :, action_idx]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=q_data["dist_labels"],
        y=q_data["battery_labels"],
        colorscale="RdYlGn",
        showscale=False,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate=f"Action: {action_name}<br>Battery: %{{y}}<br>Dist: %{{x}}<br>Q: %{{z:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=action_name, font=dict(color="#94a3b8", size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#64748b", size=8)),
        yaxis=dict(tickfont=dict(color="#64748b", size=8)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,
    )
    col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Battery history chart
st.markdown("**Fleet Average Battery Over Time**")
battery_hist = store.get_battery_history()
if len(battery_hist) > 2:
    df_bat = pd.DataFrame(battery_hist)
    fig_bh = go.Figure()
    fig_bh.add_trace(go.Scatter(
        x=df_bat["tick"], y=df_bat["avg"],
        mode="lines", fill="tozeroy",
        line=dict(color="#38bdf8", width=2),
        fillcolor="rgba(56,189,248,0.07)",
        name="Avg Battery",
    ))
    fig_bh.add_hline(y=20, line_dash="dash", line_color="#ef4444",
                     annotation_text="Critical 20%", annotation_font_color="#ef4444")
    fig_bh.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        yaxis=dict(range=[0, 105], gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        xaxis=dict(title="Simulation Tick", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=5, b=0),
        height=200,
        showlegend=False,
    )
    st.plotly_chart(fig_bh, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------
# Row 4: Security deep dive
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Sentinel Agent — Security Analysis</div>', unsafe_allow_html=True)

sec_stats = orch.sentinel.get_security_stats()
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Packets Analyzed",   sec_stats["packets_analyzed"])
s2.metric("Anomalies Detected", sec_stats["anomalies_detected"])
s3.metric("Quarantines",        sec_stats["quarantines_issued"])
s4.metric("Models Trained",     sec_stats["models_trained"])
s5.metric("Active Attacks",     sec_stats["active_attacks"])

alerts = orch.sentinel.get_recent_alerts(50)
if alerts:
    col_tl, col_dist = st.columns([3, 2])

    with col_tl:
        st.markdown("**Alert Timeline**")
        df_alerts = pd.DataFrame(alerts)
        # Group by attack type
        if "attack_type" in df_alerts.columns:
            type_counts = df_alerts["attack_type"].value_counts().reset_index()
            type_counts.columns = ["Attack Type", "Count"]
            fig_bar = go.Figure(go.Bar(
                x=type_counts["Attack Type"],
                y=type_counts["Count"],
                marker_color=["#ef4444", "#f59e0b", "#a78bfa", "#38bdf8"],
                text=type_counts["Count"],
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.8)",
                xaxis=dict(tickfont=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
                margin=dict(l=0, r=0, t=5, b=0),
                height=220,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_dist:
        st.markdown("**Severity Distribution**")
        if "severity" in df_alerts.columns:
            sev_counts = df_alerts["severity"].value_counts()
            colors_pie = {"critical": "#ef4444", "warning": "#f59e0b", "info": "#38bdf8"}
            fig_pie = go.Figure(go.Pie(
                labels=sev_counts.index.tolist(),
                values=sev_counts.values.tolist(),
                marker_colors=[colors_pie.get(s, "#94a3b8") for s in sev_counts.index],
                hole=0.5,
                textfont=dict(color="white", size=11),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=5, b=0),
                height=220,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # Alert score scatter
    st.markdown("**Anomaly Score Distribution**")
    if "score" in df_alerts.columns and "amr_name" in df_alerts.columns:
        df_scatter = df_alerts.copy()
        fig_sc = go.Figure()
        for sev, color in [("critical", "#ef4444"), ("warning", "#f59e0b")]:
            subset = df_scatter[df_scatter["severity"] == sev] if "severity" in df_scatter else df_scatter
            if not subset.empty:
                fig_sc.add_trace(go.Scatter(
                    x=list(range(len(subset))),
                    y=subset["score"].tolist(),
                    mode="markers",
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=sev.title(),
                    hovertemplate="AMR: %{text}<br>Score: %{y:.3f}<extra></extra>",
                    text=subset.get("amr_name", ["?"] * len(subset)).tolist(),
                ))
        if orch.sentinel.ANOMALY_SCORE_WARN:
            fig_sc.add_hline(y=orch.sentinel.ANOMALY_SCORE_WARN, line_dash="dash",
                             line_color="#f59e0b", annotation_text="Warn threshold")
            fig_sc.add_hline(y=orch.sentinel.ANOMALY_SCORE_CRIT, line_dash="dash",
                             line_color="#ef4444", annotation_text="Critical threshold")
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.8)",
            xaxis=dict(title="Alert #", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Anomaly Score", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=5, b=0),
            height=220,
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
else:
    st.success("✅ No security alerts recorded yet.")

# ------------------------------------------------------------------
# Row 5: Agent performance comparison
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Agent Performance Comparison</div>', unsafe_allow_html=True)

agent_data = [
    {"Agent": "Coordinator", **orch.coordinator.get_stats()},
    {"Agent": "Energy",      **orch.energy.get_stats()},
    {"Agent": "Sentinel",    **orch.sentinel.get_stats()},
]
df_agents = pd.DataFrame(agent_data)

col_a1, col_a2 = st.columns(2)

with col_a1:
    fig_dec = go.Figure(go.Bar(
        x=df_agents["Agent"],
        y=df_agents["decisions_made"],
        marker_color=["#38bdf8", "#4ade80", "#ef4444"],
        text=df_agents["decisions_made"],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig_dec.update_layout(
        title=dict(text="Decisions Made per Agent", font=dict(color="#94a3b8", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=35, b=0),
        height=220,
        showlegend=False,
    )
    st.plotly_chart(fig_dec, use_container_width=True, config={"displayModeBar": False})

with col_a2:
    fig_ev = go.Figure(go.Bar(
        x=df_agents["Agent"],
        y=df_agents["events_published"],
        marker_color=["#a78bfa", "#fb923c", "#f43f5e"],
        text=df_agents["events_published"],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig_ev.update_layout(
        title=dict(text="Events Published per Agent", font=dict(color="#94a3b8", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=35, b=0),
        height=220,
        showlegend=False,
    )
    st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------
# Row 6: Event bus stats
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Event Bus Statistics</div>', unsafe_allow_html=True)

bus_stats = status["bus_stats"]
b1, b2, b3, b4 = st.columns(4)
b1.metric("Total Published",   bus_stats.get("published", 0))
b2.metric("Total Dispatched",  bus_stats.get("dispatched", 0))
b3.metric("Handler Errors",    bus_stats.get("handler_errors", 0))
b4.metric("Dropped (overflow)", bus_stats.get("dropped", 0))

recent_events = orch.bus.get_recent_events(200)
if recent_events:
    from collections import Counter
    type_counts = Counter(e.event_type.value for e in recent_events)
    df_ev = pd.DataFrame(
        [(k, v) for k, v in type_counts.most_common(12)],
        columns=["Event Type", "Count"]
    )
    fig_ev_types = go.Figure(go.Bar(
        y=df_ev["Event Type"],
        x=df_ev["Count"],
        orientation="h",
        marker_color="#38bdf8",
        text=df_ev["Count"],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig_ev_types.update_layout(
        title=dict(text="Top Event Types (last 200)", font=dict(color="#94a3b8", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=35, b=0),
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_ev_types, use_container_width=True, config={"displayModeBar": False})

# Auto-refresh
time.sleep(3)
st.rerun()