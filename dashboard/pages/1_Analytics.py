"""
dashboard/pages/1_Analytics.py
================================
Analytics Deep-Dive Page

Sections:
  1. Fleet Performance Table    — per-AMR metrics
  2. AMR Capability Radar       — spider chart comparison
  3. Q-Learning Analysis        — heatmaps + battery chart
  4. Security Analysis          — attack charts + anomaly scatter
  5. Agent Performance          — decisions / events bar charts
  6. Event Bus Stats            — event type frequency
"""

import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="AMR Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
  .stApp { background: #0a0e1a; }

  .section-title {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em;
    color: #64748b; border-bottom: 1px solid #1e2d3d;
    padding-bottom: 6px; margin: 24px 0 12px 0;
  }
  .legend-box {
    background: rgba(17,24,39,0.7);
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-bottom: 12px;
  }
  .legend-title {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #38bdf8; margin-bottom: 6px; font-weight: 600;
  }
  .legend-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .sq  { width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex-shrink: 0; }
  div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #38bdf8 !important; }
  div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Get orchestrator ──────────────────────────────────────────────────────────
def get_orch():
    try:
        from dashboard.app import get_orchestrator
        orch, loop = get_orchestrator()
        return orch, loop
    except Exception:
        st.error("Main dashboard not running. Open it first: `streamlit run dashboard/app.py`")
        st.stop()

orch, loop = get_orch()
store  = orch.store
status = orch.get_status()
tick   = status["tick"]

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 AMR Fleet Analytics")
st.caption(
    f"Tick **{tick}** · Auto-refreshes every 3 s · "
    "All charts update live as agents make decisions"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Fleet Performance Table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Section 1 — Fleet Performance Table</div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ How to read this table</div>
  Each row is one robot. <b>Tasks/Energy</b> = how many tasks completed per 1% battery spent — higher is more efficient.
  <b>Compromised</b> = 🚨 means the Sentinel agent has quarantined this robot due to a detected network attack;
  ✅ means network traffic is clean. <b>Status</b> shows what the robot is doing right now.
</div>
""", unsafe_allow_html=True)

amrs = store.get_all_amrs()
if amrs:
    rows = []
    for a in amrs:
        total      = a.tasks_completed + getattr(a, "tasks_failed", 0)
        failed     = getattr(a, "tasks_failed", 0)
        efficiency = round(a.tasks_completed / max(a.energy_consumed, 0.1), 3) if a.energy_consumed > 0 else 0
        rows.append({
            "🤖 Robot":       a.name,
            "Status":         a.status.value,
            "🔋 Battery %":   round(a.battery, 1),
            "✅ Tasks Done":  a.tasks_completed,
            "❌ Tasks Failed": failed,
            "📏 Distance (u)": round(a.total_distance, 1),
            "⚡ Energy Used %": round(a.energy_consumed, 1),
            "📈 Tasks/Energy": efficiency,
            "🚨 Alerts":       a.alerts_triggered,
            "🔒 Security":     "🚨 QUARANTINED" if a.is_compromised else "✅ Safe",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Waiting for simulation to start...")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Radar Chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🕸️ Section 2 — AMR Capability Radar (Spider Chart)</div>', unsafe_allow_html=True)

col_radar, col_radar_legend = st.columns([3, 1])

with col_radar_legend:
    st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ How to read this</div>
  Each axis is a capability normalised to <b>0–100</b>. Bigger area = better overall robot.<br><br>
  <b>🔋 Battery</b> — current charge level<br>
  <b>📦 Tasks Done</b> — tasks completed vs best robot<br>
  <b>📏 Distance</b> — total distance travelled<br>
  <b>✅ Reliability</b> — task success rate (100% = no failures)<br>
  <b>🔒 Security</b> — 100% = zero alerts; drops when attacked<br><br>
  A robot with a <i>small area</i> is underperforming — check the Fleet Map to see why.
</div>
""", unsafe_allow_html=True)

with col_radar:
    if amrs:
        max_tasks = max(a.tasks_completed for a in amrs) or 1
        max_dist  = max(a.total_distance  for a in amrs) or 1

        # Convert hex colours to proper rgba strings for fill
        HEX_COLORS = ["#38bdf8", "#4ade80", "#a78bfa", "#fb923c", "#f43f5e"]
        RGBA_FILLS  = [
            "rgba(56,189,248,0.12)",
            "rgba(74,222,128,0.12)",
            "rgba(167,139,250,0.12)",
            "rgba(251,146,60,0.12)",
            "rgba(244,63,94,0.12)",
        ]

        fig_radar = go.Figure()
        categories = ["Battery", "Tasks Done", "Distance", "Reliability", "Security"]

        for i, a in enumerate(amrs):
            failed      = getattr(a, "tasks_failed", 0)
            reliability = 1 - (failed / max(a.tasks_completed + failed, 1))
            total_alerts = sum(x.alerts_triggered for x in amrs) or 1
            security     = 1 - (a.alerts_triggered / total_alerts)

            values = [
                a.battery / 100.0,
                a.tasks_completed / max_tasks,
                a.total_distance  / max_dist,
                reliability,
                security,
            ]
            values_norm = [round(v * 100, 1) for v in values]

            fig_radar.add_trace(go.Scatterpolar(
                r     = values_norm + [values_norm[0]],
                theta = categories  + [categories[0]],
                fill      = "toself",
                name      = a.name,
                line_color = HEX_COLORS[i % len(HEX_COLORS)],
                fillcolor  = RGBA_FILLS[i % len(RGBA_FILLS)],   # ← BUG FIX
                opacity    = 0.9,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="#1e2d3d",
                    tickfont=dict(color="#64748b", size=9),
                ),
                angularaxis=dict(
                    gridcolor="#1e2d3d",
                    tickfont=dict(color="#94a3b8", size=11),
                ),
                bgcolor="rgba(17,24,39,0.8)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=40, t=20, b=20),
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Q-Learning Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 Section 3 — Energy Agent · Q-Learning Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ What is Q-Learning?</div>
  The Energy Agent uses <b>Reinforcement Learning</b> to decide whether each robot should
  <i>keep working</i>, <i>go charge</i>, or <i>slow down</i>.
  It maintains a table (the <b>Q-table</b>) of 72 values — each cell is the learned "goodness"
  of an action in a given situation. <b>Green = good choice, Red = bad choice.</b><br><br>
  The three heatmaps below show how good each of the three actions is across all battery/distance combinations.
  Watch them change over time as the agent learns from reward signals.
</div>
""", unsafe_allow_html=True)

# KPI row
q_stats = orch.energy.get_qtable_stats()
qc1, qc2, qc3, qc4 = st.columns(4)
qc1.metric("Total Reward Earned",   q_stats["total_reward"],
           help="Sum of all rewards the Q-learning agent has received. Higher = better decisions over time.")
qc2.metric("Charge Events Triggered", q_stats["charge_events"],
           help="How many times the agent sent a robot to a charging station.")
qc3.metric("Prevented Depletions",  q_stats["prevented_depletions"],
           help="Robots that were routed to charge before hitting 0% battery.")
qc4.metric("LLM Overrides",         q_stats.get("llm_calls", 0),
           help="Times the LLM (Groq/LLaMA3) stepped in to arbitrate a charging conflict.")

# Q-table heatmaps
st.markdown("##### Q-Table Heatmaps — one per action")
st.markdown(
    "<div style='font-size:0.78rem;color:#64748b;margin-bottom:8px;'>"
    "Rows = battery level (Critical/Low/Medium/High). "
    "Columns = distance to nearest free charging station (Close/Medium/Far). "
    "Cell value = Q-value: how good this action is in this state. "
    "<b>Green = high reward, Red = bad idea.</b>"
    "</div>",
    unsafe_allow_html=True,
)

q_data    = orch.energy.get_qtable_heatmap_data()
q_arr     = np.array(q_data["q_values"])   # shape (battery, dist, actions)
col_h1, col_h2, col_h3 = st.columns(3)
action_names   = q_data["action_labels"]
action_icons   = ["⚙️ Continue Working", "⚡ Go Charge", "🐢 Reduce Speed"]
action_descs   = [
    "Robot keeps its current task. Good when battery is healthy.",
    "Robot heads to nearest free charging station. Essential at low battery.",
    "Robot slows down to save battery. Buys time when no station is free.",
]

for col, action_idx in zip([col_h1, col_h2, col_h3], [0, 1, 2]):
    z   = q_arr[:, :, action_idx]
    fig = go.Figure(go.Heatmap(
        z            = z,
        x            = q_data["dist_labels"],
        y            = q_data["battery_labels"],
        colorscale   = "RdYlGn",
        showscale    = False,
        text         = [[f"{v:.2f}" for v in row] for row in z],
        texttemplate = "%{text}",
        textfont     = dict(size=9, color="white"),
        hovertemplate = (
            f"Action: {action_names[action_idx]}<br>"
            "Battery: %{y}<br>Distance: %{x}<br>Q-value: %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text=action_icons[action_idx], font=dict(color="#94a3b8", size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(17,24,39,0.8)",
        xaxis=dict(title="Distance to Station", tickfont=dict(color="#64748b", size=8)),
        yaxis=dict(title="Battery Level",       tickfont=dict(color="#64748b", size=8)),
        margin=dict(l=0, r=0, t=36, b=0),
        height=210,
    )
    col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    col.markdown(
        f"<div style='font-size:0.72rem;color:#64748b;text-align:center;'>"
        f"{action_descs[action_idx]}</div>",
        unsafe_allow_html=True,
    )

# Battery history chart
st.markdown("##### Fleet Average Battery Over Time")
st.markdown(
    "<div style='font-size:0.78rem;color:#64748b;margin-bottom:6px;'>"
    "Average battery % across the whole fleet per tick. "
    "The dashed line at <b>20%</b> is the low-battery warning threshold — "
    "the Energy Agent intervenes when the fleet average approaches it."
    "</div>",
    unsafe_allow_html=True,
)
battery_hist = store.get_battery_history()
if len(battery_hist) > 2:
    df_bat = pd.DataFrame(battery_hist)
    fig_bh = go.Figure()
    fig_bh.add_trace(go.Scatter(
        x=df_bat["tick"], y=df_bat["avg"],
        mode="lines",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.07)",
        name="Avg Battery %",
    ))
    fig_bh.add_hline(
        y=20, line_dash="dash", line_color="#f59e0b",
        annotation_text="⚠️ Low Battery Threshold (20%)",
        annotation_font_color="#f59e0b",
    )
    fig_bh.add_hline(
        y=10, line_dash="dot", line_color="#ef4444",
        annotation_text="🚨 Critical Threshold (10%)",
        annotation_font_color="#ef4444",
    )
    fig_bh.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(17,24,39,0.8)",
        xaxis=dict(title="Simulation Tick", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
        yaxis=dict(title="Battery %", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8"),
                   range=[0, 105]),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=5, b=0),
        height=220,
        showlegend=False,
    )
    st.plotly_chart(fig_bh, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Security Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🛡️ Section 4 — Sentinel Agent · Security Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ How the Sentinel Agent works</div>
  Every robot communicates over a simulated 5G network. The Sentinel watches all network packets
  using an <b>Isolation Forest</b> ML model (unsupervised anomaly detection).
  Normal packets get a score near <b>−0.45</b>. When traffic looks suspicious the score drops —
  below <b>−0.55</b> triggers a WARNING, below <b>−0.65</b> QUARANTINES the robot and calls the LLM
  to write an incident report.<br><br>
  <b>Attack types:</b>
  <span style="color:#ef4444">PACKET_FLOOD</span> (tiny bursts, high latency) ·
  <span style="color:#f59e0b">JAMMING</span> (weak signal, huge packet loss) ·
  <span style="color:#a78bfa">DATA_EXFIL</span> (massive packets, ultra-low latency) ·
  <span style="color:#38bdf8">REPLAY_ATTACK</span> (duplicate packets, degraded signal)
</div>
""", unsafe_allow_html=True)

sec_stats = orch.sentinel.get_security_stats()
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("📦 Packets Analyzed",   sec_stats["packets_analyzed"],
          help="Total 5G packets the Sentinel has inspected using Isolation Forest.")
s2.metric("⚠️ Anomalies Detected", sec_stats["anomalies_detected"],
          help="Packets that scored below the warning threshold (−0.55).")
s3.metric("🔒 Quarantines Issued", sec_stats["quarantines_issued"],
          help="Times a robot was isolated because anomaly score dropped below −0.65.")
s4.metric("🧠 Models Trained",     sec_stats["models_trained"],
          help="Each robot gets its own Isolation Forest model. Trains after 30 clean packets.")
s5.metric("💥 Active Attacks",     sec_stats["active_attacks"],
          help="Attack simulations currently running on the fleet.")

alerts = orch.sentinel.get_recent_alerts(50)
if alerts:
    col_tl, col_dist = st.columns([3, 2])

    with col_tl:
        st.markdown("##### 📊 Alerts by Attack Type")
        st.markdown(
            "<div style='font-size:0.75rem;color:#64748b;margin-bottom:6px;'>"
            "Bar height = how many times that attack type was detected. "
            "More bars = more varied attack surface."
            "</div>",
            unsafe_allow_html=True,
        )
        df_alerts = pd.DataFrame(alerts)
        if "attack_type" in df_alerts.columns:
            type_counts = df_alerts["attack_type"].value_counts().reset_index()
            type_counts.columns = ["Attack Type", "Count"]
            ATTACK_COLORS = {
                "PACKET_FLOOD":  "#ef4444",
                "JAMMING":       "#f59e0b",
                "DATA_EXFIL":    "#a78bfa",
                "REPLAY_ATTACK": "#38bdf8",
            }
            bar_colors = [ATTACK_COLORS.get(t, "#94a3b8") for t in type_counts["Attack Type"]]
            fig_bar = go.Figure(go.Bar(
                x=type_counts["Attack Type"], y=type_counts["Count"],
                marker_color=bar_colors,
                text=type_counts["Count"], textposition="outside",
                textfont=dict(color="white"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
                xaxis=dict(tickfont=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8"), title="Alert Count"),
                margin=dict(l=0, r=0, t=5, b=0), height=240, showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_dist:
        st.markdown("##### 🥧 Severity Breakdown")
        st.markdown(
            "<div style='font-size:0.75rem;color:#64748b;margin-bottom:6px;'>"
            "<b>Critical</b> = robot quarantined. <b>Warning</b> = suspicious but still active."
            "</div>",
            unsafe_allow_html=True,
        )
        if "severity" in df_alerts.columns:
            sev_counts  = df_alerts["severity"].value_counts()
            colors_pie  = {"critical": "#ef4444", "warning": "#f59e0b", "info": "#38bdf8"}
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
                margin=dict(l=0, r=0, t=5, b=0), height=240,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # Anomaly score scatter
    st.markdown("##### 📉 Anomaly Score Distribution")
    st.markdown(
        "<div style='font-size:0.75rem;color:#64748b;margin-bottom:6px;'>"
        "Each dot is one security alert. X-axis = alert sequence number. "
        "Y-axis = Isolation Forest anomaly score. "
        "<b>Lower score = more suspicious.</b> "
        "The dashed lines show where the agent triggers WARNING vs QUARANTINE."
        "</div>",
        unsafe_allow_html=True,
    )
    if "score" in df_alerts.columns and "amr_name" in df_alerts.columns:
        fig_sc = go.Figure()
        for sev, color, label in [
            ("critical", "#ef4444", "🔴 Critical (quarantined)"),
            ("warning",  "#f59e0b", "🟡 Warning (suspicious)"),
        ]:
            subset = df_alerts[df_alerts.get("severity", pd.Series()) == sev] \
                     if "severity" in df_alerts else df_alerts
            if not subset.empty:
                fig_sc.add_trace(go.Scatter(
                    x=list(range(len(subset))),
                    y=subset["score"].tolist(),
                    mode="markers",
                    marker=dict(color=color, size=9, opacity=0.85),
                    name=label,
                    hovertemplate="Robot: %{text}<br>Score: %{y:.3f}<extra></extra>",
                    text=subset["amr_name"].tolist() if "amr_name" in subset else [],
                ))

        warn_thresh = getattr(orch.sentinel, "ANOMALY_SCORE_WARN", -0.55)
        crit_thresh = getattr(orch.sentinel, "ANOMALY_SCORE_CRIT", -0.65)
        fig_sc.add_hline(y=warn_thresh, line_dash="dash",  line_color="#f59e0b",
                         annotation_text="⚠️ Warning threshold")
        fig_sc.add_hline(y=crit_thresh, line_dash="dot",   line_color="#ef4444",
                         annotation_text="🚨 Quarantine threshold")
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
            xaxis=dict(title="Alert #", gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8")),
            yaxis=dict(title="Anomaly Score (lower = worse)", gridcolor="#1e2d3d",
                       tickfont=dict(color="#94a3b8")),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=5, b=0), height=240,
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
else:
    st.success("✅ No security alerts recorded yet — the network is clean.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Agent Performance Comparison
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚙️ Section 5 — Agent Performance Comparison</div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ What these bars mean</div>
  <b>Decisions Made</b> — every time an agent actively changed something (assigned a task,
  sent a robot to charge, issued an alert, etc.). Higher = more active agent.<br>
  <b>Events Published</b> — messages the agent sent on the event bus for other agents to react to.
  Agents only talk through events — never directly. More events = more coordination happening.
</div>
""", unsafe_allow_html=True)

agent_data = [
    {"Agent": "🗺️ Coordinator", **orch.coordinator.get_stats()},
    {"Agent": "⚡ Energy",       **orch.energy.get_stats()},
    {"Agent": "🛡️ Sentinel",     **orch.sentinel.get_stats()},
]
df_agents = pd.DataFrame(agent_data)

col_a1, col_a2 = st.columns(2)

with col_a1:
    fig_dec = go.Figure(go.Bar(
        x=df_agents["Agent"], y=df_agents["decisions_made"],
        marker_color=["#38bdf8", "#4ade80", "#ef4444"],
        text=df_agents["decisions_made"], textposition="outside",
        textfont=dict(color="white"),
    ))
    fig_dec.update_layout(
        title=dict(text="Decisions Made per Agent", font=dict(color="#94a3b8", size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8"),
                   title="# Decisions"),
        margin=dict(l=0, r=0, t=35, b=0), height=240, showlegend=False,
    )
    st.plotly_chart(fig_dec, use_container_width=True, config={"displayModeBar": False})

with col_a2:
    fig_ev = go.Figure(go.Bar(
        x=df_agents["Agent"], y=df_agents["events_published"],
        marker_color=["#a78bfa", "#fb923c", "#f43f5e"],
        text=df_agents["events_published"], textposition="outside",
        textfont=dict(color="white"),
    ))
    fig_ev.update_layout(
        title=dict(text="Events Published per Agent", font=dict(color="#94a3b8", size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8"),
                   title="# Events"),
        margin=dict(l=0, r=0, t=35, b=0), height=240, showlegend=False,
    )
    st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Event Bus Statistics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📡 Section 6 — Event Bus Statistics</div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend-box">
  <div class="legend-title">ℹ️ What is the Event Bus?</div>
  The event bus is how agents communicate. Instead of calling each other directly,
  every agent <i>publishes</i> events (e.g. "TASK_ASSIGNED") and <i>subscribes</i> to events
  it cares about. This makes the system loosely coupled — adding a new agent doesn't require
  touching existing code.<br><br>
  <b>Published</b> = total events created · <b>Dispatched</b> = events delivered to handlers ·
  <b>Handler Errors</b> = events that caused an exception (should be 0) ·
  <b>Dropped</b> = events lost because queue was full (should be 0)
</div>
""", unsafe_allow_html=True)

bus_stats = status["bus_stats"]
b1, b2, b3, b4 = st.columns(4)
b1.metric("📤 Total Published",    bus_stats.get("published", 0))
b2.metric("📬 Total Dispatched",   bus_stats.get("dispatched", 0))
b3.metric("❗ Handler Errors",     bus_stats.get("handler_errors", 0),
          delta=None if bus_stats.get("handler_errors", 0) == 0 else "⚠️ Check logs",
          delta_color="inverse")
b4.metric("🗑️ Dropped (overflow)", bus_stats.get("dropped", 0))

recent_events = orch.bus.get_recent_events(200)
if recent_events:
    from collections import Counter
    type_counts = Counter(e.event_type.value for e in recent_events)
    df_ev = pd.DataFrame(
        [(k, v) for k, v in type_counts.most_common(12)],
        columns=["Event Type", "Count"]
    )
    fig_ev_types = go.Figure(go.Bar(
        y=df_ev["Event Type"], x=df_ev["Count"],
        orientation="h",
        marker_color="#38bdf8",
        text=df_ev["Count"], textposition="outside",
        textfont=dict(color="white"),
    ))
    fig_ev_types.update_layout(
        title=dict(text="Top 12 Event Types (last 200 events)", font=dict(color="#94a3b8", size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        xaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#94a3b8"), title="Count"),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        margin=dict(l=0, r=0, t=35, b=0), height=370, showlegend=False,
    )
    st.plotly_chart(fig_ev_types, use_container_width=True, config={"displayModeBar": False})

# Auto-refresh
time.sleep(3)
st.rerun()