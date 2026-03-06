"""
dashboard/pages/3_Export.py
============================
Export & Report Page

Download simulation data as CSV, Q-table as JSON,
and generate a summary report for your project submission.

Author: AMR Multi-Agent Framework
"""

import sys
import os
import io
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="AMR Export", page_icon="📥", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
  .stApp { background: #0a0e1a; }
  .report-box {
    background: rgba(17,24,39,0.9); border: 1px solid #1e2d3d;
    border-radius: 8px; padding: 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    color: #94a3b8; white-space: pre; overflow-x: auto;
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
        st.error("Main dashboard not running. Please start: `streamlit run dashboard/app.py`")
        st.stop()


orch, loop = get_orch()
store      = orch.store

st.title("📥 Export & Reports")
st.markdown(
    "<div style='color:#64748b; font-size:0.85rem;'>"
    "Download simulation data for analysis or include in your project report.</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ------------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------------
st.markdown("### 📋 Summary Report")

from core.metrics_exporter import MetricsExporter
exporter = MetricsExporter(
    store=store,
    sentinel=orch.sentinel,
    energy=orch.energy,
)
report = exporter.get_summary_report()
st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

report_bytes = report.encode("utf-8")
st.download_button(
    "📄 Download Report (.txt)",
    data=report_bytes,
    file_name=f"amr_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain",
    use_container_width=True,
)

st.markdown("---")

# ------------------------------------------------------------------
# Fleet CSV
# ------------------------------------------------------------------
st.markdown("### 🤖 AMR Performance CSV")

amrs = store.get_all_amrs()
rows = []
for a in amrs:
    total = a.tasks_completed + a.tasks_failed
    rows.append({
        "AMR ID":           a.amr_id,
        "Name":             a.name,
        "Status":           a.status.value,
        "Battery %":        round(a.battery, 1),
        "Tasks Completed":  a.tasks_completed,
        "Tasks Failed":     a.tasks_failed,
        "Success Rate %":   round(a.tasks_completed / total * 100, 1) if total else 0,
        "Distance":         round(a.total_distance, 2),
        "Energy Consumed":  round(a.energy_consumed, 2),
        "Alerts":           a.alerts_triggered,
        "Compromised":      a.is_compromised,
    })

if rows:
    df_amr = pd.DataFrame(rows)
    st.dataframe(df_amr, use_container_width=True, hide_index=True)
    csv_buf = io.StringIO()
    df_amr.to_csv(csv_buf, index=False)
    st.download_button(
        "📊 Download AMR Data (.csv)",
        data=csv_buf.getvalue().encode(),
        file_name=f"amr_performance_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")

# ------------------------------------------------------------------
# Battery history CSV
# ------------------------------------------------------------------
st.markdown("### 🔋 Battery History CSV")

bat_hist = store.get_battery_history()
if bat_hist:
    df_bat = pd.DataFrame(bat_hist)
    st.dataframe(df_bat.tail(50), use_container_width=True, hide_index=True)
    csv_buf2 = io.StringIO()
    df_bat.to_csv(csv_buf2, index=False)
    st.download_button(
        "📈 Download Battery History (.csv)",
        data=csv_buf2.getvalue().encode(),
        file_name=f"battery_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")

# ------------------------------------------------------------------
# Security alerts CSV
# ------------------------------------------------------------------
st.markdown("### 🛡️ Security Alerts CSV")

alerts = orch.sentinel.get_recent_alerts(200)
if alerts:
    df_alerts = pd.DataFrame(alerts)
    st.dataframe(df_alerts, use_container_width=True, hide_index=True)
    csv_buf3 = io.StringIO()
    df_alerts.to_csv(csv_buf3, index=False)
    st.download_button(
        "🚨 Download Alerts (.csv)",
        data=csv_buf3.getvalue().encode(),
        file_name=f"security_alerts_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("No security alerts recorded yet.")

st.markdown("---")

# ------------------------------------------------------------------
# Q-Table JSON
# ------------------------------------------------------------------
st.markdown("### 🧠 Q-Table Export (JSON)")

q_data = {
    "qtable_stats":    orch.energy.get_qtable_stats(),
    "qtable_heatmap":  orch.energy.get_qtable_heatmap_data(),
    "raw_qtable":      orch.energy.q_table.tolist(),
    "exported_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
    "tick":            orch.simulator.tick,
}

st.json(q_data["qtable_stats"])
st.download_button(
    "🧠 Download Q-Table (.json)",
    data=json.dumps(q_data, indent=2).encode(),
    file_name=f"qtable_{time.strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
    use_container_width=True,
)

st.markdown("---")
st.caption(f"Last refreshed: {time.strftime('%H:%M:%S')} — Tick {orch.simulator.tick}")