import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import time
from src.ingestion import EvidenceIngestion
from src.email_analysis import EmailAnalyzer
from src.behavior_analysis import BehaviorAnalyzer
from src.fusion import FusionEngine
from src.reporting import ReportManager

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Crime Pattern Signature Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Severity badge colours ---------- */
    .severity-low     { background:#22c55e; color:#fff; padding:4px 14px; border-radius:6px; font-weight:700; }
    .severity-medium  { background:#f59e0b; color:#fff; padding:4px 14px; border-radius:6px; font-weight:700; }
    .severity-high    { background:#ef4444; color:#fff; padding:4px 14px; border-radius:6px; font-weight:700; }
    .severity-critical{ background:#7f1d1d; color:#fff; padding:4px 14px; border-radius:6px; font-weight:700; animation:pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

    /* ---------- Timeline ---------- */
    .tl-container  { position:relative; padding-left:28px; margin:12px 0; }
    .tl-container::before { content:''; position:absolute; left:10px; top:0; bottom:0; width:3px; background:#334155; border-radius:2px; }
    .tl-item       { position:relative; margin-bottom:18px; padding:12px 16px; border-radius:10px; background:#1e293b; border:1px solid #334155; }
    .tl-item::before { content:''; position:absolute; left:-23px; top:16px; width:12px; height:12px; border-radius:50%; border:2px solid #fff; }
    .tl-email::before      { background:#ef4444; }
    .tl-behavior::before   { background:#f59e0b; }
    .tl-fusion::before     { background:#8b5cf6; }
    .tl-time       { font-size:.78rem; color:#94a3b8; margin-bottom:4px; }
    .tl-title      { font-weight:600; font-size:.95rem; }
    .tl-desc       { font-size:.82rem; color:#cbd5e1; margin-top:4px; }

    /* ---------- Rule card ---------- */
    .rule-card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
    .rule-id   { font-weight:700; color:#38bdf8; font-size:.9rem; }
    .rule-desc { font-size:.82rem; color:#e2e8f0; margin-top:2px; }
    .rule-evi  { font-size:.78rem; color:#94a3b8; margin-top:4px; }

    /* ---------- Escalation step ---------- */
    .esc-step { display:inline-block; font-size:.82rem; padding:3px 10px; margin:2px 4px; border-radius:6px; }
    .esc-arrow { display:inline-block; font-size:1rem; padding:0 4px; color:#64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Helper: Severity gauge (Plotly)
# ──────────────────────────────────────────────────────────────
SEVERITY_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
    "CRITICAL": "#7f1d1d",
}


def render_severity_gauge(severity: str, confidence: float):
    """Renders a Plotly gauge for the given severity & confidence."""
    value = SEVERITY_MAP.get(severity, 0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"suffix": f"  ({severity})", "font": {"size": 22, "color": "#e2e8f0"}},
            delta={"reference": 0, "increasing": {"color": "#ef4444"}},
            gauge={
                "axis": {"range": [0, 3], "tickvals": [0, 1, 2, 3], "ticktext": ["LOW", "MED", "HIGH", "CRIT"], "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": SEVERITY_COLORS.get(severity, "#22c55e"), "thickness": 0.75},
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 1], "color": "rgba(34,197,94,0.15)"},
                    {"range": [1, 2], "color": "rgba(245,158,11,0.15)"},
                    {"range": [2, 3], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.8,
                    "value": value,
                },
            },
            title={"text": f"Confidence: {confidence:.0%}", "font": {"size": 14, "color": "#94a3b8"}},
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Helper: Severity badge
# ──────────────────────────────────────────────────────────────
def severity_badge(sev: str) -> str:
    cls = f"severity-{sev.lower()}"
    return f'<span class="{cls}">{sev}</span>'


# ──────────────────────────────────────────────────────────────
# Helper: Event-type CSS class for timeline dot
# ──────────────────────────────────────────────────────────────
EVENT_TYPE_CLASS = {
    "Email threat": "tl-email",
    "Behavior anomaly": "tl-behavior",
    "Fusion event": "tl-fusion",
}


# ══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════
st.title("Digital Crime Pattern Signature Generator")
st.markdown("### AI-Driven Cyber Forensic System")

# Sidebar ──────────────────────────────────────
st.sidebar.header("Evidence Upload")
use_test_data = st.sidebar.checkbox("Use Synthetic Test Data", value=False)
email_file = st.sidebar.file_uploader("Upload Email Evidence (CSV)", type=["csv"])
log_file = st.sidebar.file_uploader("Upload User Activity Logs (CSV)", type=["csv"])

if st.sidebar.button("Generate Crime Signature") or st.session_state.get("analyzed"):
    st.session_state["analyzed"] = True

    with st.spinner("Initializing Forensic Engine..."):
        # ── 1. Ingestion ──
        ingestion = EvidenceIngestion()

        if use_test_data:
            if os.path.exists("data/test_emails.csv") and os.path.exists("data/test_logs.csv"):
                import shutil
                shutil.copy("data/test_emails.csv", "temp_emails.csv")
                shutil.copy("data/test_logs.csv", "temp_logs.csv")
                st.sidebar.info("Loaded Synthetic Test Data.")
            else:
                st.error("Test data not found. Please generate it first.")
                st.stop()
        elif email_file and log_file:
            with open("temp_emails.csv", "wb") as f:
                f.write(email_file.getbuffer())
            with open("temp_logs.csv", "wb") as f:
                f.write(log_file.getbuffer())
        else:
            st.error("Please upload evidence files or select 'Use Synthetic Test Data'.")
            st.stop()

        emails_df = ingestion.load_emails("temp_emails.csv")
        logs_df = ingestion.load_logs("temp_logs.csv")

        if emails_df is None:
            st.error("Failed to load Email Evidence. Check format & columns: Subject, Body, Sender, Receiver, Timestamp.")
            st.stop()
        if logs_df is None:
            st.error("Failed to load User Logs. Check format & columns: User ID, Action, Resource, Timestamp.")
            st.stop()

        st.info("Layer 1: Evidence Ingestion Complete")

        # ── 2. Analysis ──
        progress_bar = st.progress(0)

        st.text("Running Transformer-based Email Analysis...")
        email_analyzer = EmailAnalyzer()
        email_results = email_analyzer.analyze_emails(emails_df)
        progress_bar.progress(40)

        st.text("Running Autoencoder + LSTM Behavior Analysis...")
        behavior_analyzer = BehaviorAnalyzer()
        if not os.path.exists("models/behavior_model/ae.pth"):
            st.warning("No pre-trained behavior model found. Training on current dataset...")
            behavior_analyzer.train(logs_df)

        behavior_results = behavior_analyzer.analyze_behavior(logs_df)
        progress_bar.progress(70)

        # ── 3. Fusion ──
        st.text("Fusing Evidence & Generating Signatures...")
        fusion_engine = FusionEngine()
        signatures = fusion_engine.fuse_evidence(email_results, behavior_results)
        progress_bar.progress(90)

        # ── 4. Reporting ──
        report_manager = ReportManager()
        email_stats = {"suspicious_count": len(email_results[email_results["Is_Suspicious"]])}
        behavior_stats = {"anomaly_count": len(behavior_results)}
        report_path = report_manager.generate_report(signatures, email_stats, behavior_stats)
        progress_bar.progress(100)

        st.success("Analysis Complete!")

    # ═══════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════

    # ── Executive Summary Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Suspicious Emails", email_stats["suspicious_count"])
    col2.metric("Behavioral Anomalies", behavior_stats["anomaly_count"])
    critical_count = len(signatures[signatures["Severity"] == "CRITICAL"]) if not signatures.empty else 0
    col3.metric("Critical Patterns", critical_count)
    high_count = len(signatures[signatures["Severity"] == "HIGH"]) if not signatures.empty else 0
    col4.metric("High Patterns", high_count)

    st.divider()

    # ══════════════════════════════════
    # 2.1  SEVERITY GAUGE  (NEW)
    # ══════════════════════════════════
    if not signatures.empty:
        st.subheader("🎯 Risk Visualization — Severity Gauge")

        # Determine overall (worst) severity
        sev_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        worst_idx = signatures["Severity"].map(sev_order).idxmax()
        worst_sev = signatures.loc[worst_idx, "Severity"]
        avg_conf = signatures["Confidence_Score"].mean()

        gcol1, gcol2 = st.columns([1, 1])
        with gcol1:
            fig = render_severity_gauge(worst_sev, avg_conf)
            st.plotly_chart(fig, use_container_width=True)
        with gcol2:
            st.markdown(f"**Overall Risk Level:** {severity_badge(worst_sev)}", unsafe_allow_html=True)
            st.markdown(f"**Average Confidence:** `{avg_conf:.2%}`")
            st.markdown(f"**Total Signatures:** `{len(signatures)}`")
            # Colour legend
            st.markdown(
                '<span class="severity-low">LOW</span> '
                '<span class="severity-medium">MEDIUM</span> '
                '<span class="severity-high">HIGH</span> '
                '<span class="severity-critical">CRITICAL</span>',
                unsafe_allow_html=True,
            )

        st.divider()

    # ══════════════════════════════════
    # Crime Signatures Table
    # ══════════════════════════════════
    st.subheader("Detected Digital Crime Pattern Signatures")
    if not signatures.empty:
        display_cols = ["Timestamp", "Crime_Type", "Severity", "Confidence_Score", "Explanation"]
        st.dataframe(signatures[display_cols], use_container_width=True)

        critical_sigs = signatures[signatures["Severity"] == "CRITICAL"]
        if not critical_sigs.empty:
            st.error(f"🚨 CRITICAL ALERT: {len(critical_sigs)} Coordinated Attacks Detected!")
    else:
        st.success("No significant crime patterns detected.")

    st.divider()

    # ══════════════════════════════════
    # 2.2  FORENSIC TIMELINE  (NEW)
    # ══════════════════════════════════
    if not signatures.empty:
        st.subheader("🕵️ Forensic Timeline")
        st.caption("Chronological reconstruction of events – click any expander to view raw evidence.")

        # Colour legend
        st.markdown(
            "🔴 **Email threat** &nbsp;&nbsp; 🟡 **Behavior anomaly** &nbsp;&nbsp; 🟣 **Fusion event**"
        )

        sorted_sigs = signatures.sort_values("Timestamp")
        timeline_html = '<div class="tl-container">'
        for idx, row in sorted_sigs.iterrows():
            evt = row.get("event_type", "Fusion event")
            css_cls = EVENT_TYPE_CLASS.get(evt, "tl-fusion")
            sev_html = severity_badge(row["Severity"])
            timeline_html += (
                f'<div class="tl-item {css_cls}">'
                f'  <div class="tl-time">{row["Timestamp"]}</div>'
                f'  <div class="tl-title">{row["Crime_Type"]} &nbsp;{sev_html}</div>'
                f'  <div class="tl-desc">{row["Explanation"][:200]}</div>'
                f"</div>"
            )
        timeline_html += "</div>"
        st.markdown(timeline_html, unsafe_allow_html=True)

        # Expandable raw evidence per event
        for idx, row in sorted_sigs.iterrows():
            evt = row.get("event_type", "Fusion event")
            with st.expander(f"📄 Raw Evidence — {row['Crime_Type']} @ {row['Timestamp']}"):
                st.json(
                    {
                        "timestamp": str(row["Timestamp"]),
                        "event_type": evt,
                        "evidence_id": f"SIG-{idx+1:04d}",
                        "crime_type": row["Crime_Type"],
                        "severity": row["Severity"],
                        "confidence": row["Confidence_Score"],
                        "explanation": row["Explanation"],
                        "involved_user": row["Involved_User"],
                        "severity_at_time": row["Severity"],
                    }
                )

        st.divider()

    # ══════════════════════════════════
    # TRIGGERED RULES PANEL (NEW)
    # ══════════════════════════════════
    if not signatures.empty:
        st.subheader("📋 Triggered Fusion Rules")
        st.caption("Structured rules output — NOT hidden inside explanation paragraphs.")

        for idx, row in signatures.iterrows():
            rules = row.get("triggered_rules", [])
            if not rules:
                continue

            with st.expander(
                f"Signature #{idx+1}: {row['Crime_Type']} — {len(rules)} rule(s) triggered",
                expanded=True,
            ):
                for rule in rules:
                    st.markdown(
                        f'<div class="rule-card">'
                        f'  <div class="rule-id">{rule["rule_id"]}</div>'
                        f'  <div class="rule-desc">{rule["rule_description"]}</div>'
                        f'  <div class="rule-evi">Evidence: {rule["evidence_source"]}  |  Weight: {rule["rule_weight"]:.2f}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Escalation path
                steps = row.get("severity_escalation_steps", [])
                if steps:
                    st.markdown("**Escalation Path:**")
                    path_html = ""
                    for i, s in enumerate(steps):
                        color = SEVERITY_COLORS.get(s["new_severity"], "#64748b")
                        if i > 0:
                            path_html += '<span class="esc-arrow">→</span>'
                        path_html += (
                            f'<span class="esc-step" style="background:{color};color:#fff;">'
                            f'{s["new_severity"]} ({s["rule_triggered"]})</span>'
                        )
                    st.markdown(path_html, unsafe_allow_html=True)

        st.divider()

    # ══════════════════════════════════
    # Detailed Views (Tabs) — original
    # ══════════════════════════════════
    tab1, tab2, tab3 = st.tabs(["Email Evidence", "Behavior Analysis", "Explainability"])

    with tab1:
        st.subheader("Email Analysis (Transformer)")
        st.dataframe(email_results, use_container_width=True)

    with tab2:
        st.subheader("User Behavior Analysis (Autoencoder + LSTM)")
        if not behavior_results.empty:
            st.dataframe(behavior_results, use_container_width=True)
            st.line_chart(behavior_results["Anomaly_Score"])
        else:
            st.write("No anomalies detected.")

    with tab3:
        st.subheader("Forensic Reasoning")
        if not signatures.empty:
            for i, row in signatures.iterrows():
                with st.expander(f"{row['Crime_Type']} ({row['Severity']})"):
                    st.write(f"**Explanation:** {row['Explanation']}")
                    st.progress(
                        row["Email_Contribution_Pct"] / 100,
                        text=f"Email Contribution: {row['Email_Contribution_Pct']}%",
                    )
                    st.progress(
                        row["Behavior_Contribution_Pct"] / 100,
                        text=f"Behavior Contribution: {row['Behavior_Contribution_Pct']}%",
                    )

    # ── Download Report ──
    with open(report_path, "rb") as pdf_file:
        st.download_button(
            label="📥 Download Forensic PDF Report",
            data=pdf_file,
            file_name="Forensic_Report.pdf",
            mime="application/pdf",
        )

else:
    st.info("Please upload both Email and Log datasets to begin analysis.")
    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.markdown("1. **Ingestion**: Cleans and normalizes CSV data.")
    st.markdown("2. **Email Analysis**: Fine-tuned DistilBERT Transformer for phishing detection.")
    st.markdown("3. **Behavior Analysis**: Autoencoder + LSTM for anomaly detection.")
    st.markdown("4. **Fusion Engine**: Correlates events to generate explainable crime signatures.")
    st.markdown("5. **Risk Gauge**: Visual severity indicator with confidence scoring.")
    st.markdown("6. **Forensic Timeline**: Chronological event reconstruction with raw evidence.")
    st.markdown("7. **Triggered Rules**: Structured, traceable rule-based escalation output.")
