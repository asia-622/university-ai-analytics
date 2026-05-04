"""
app.py — University AI Analytics Agent
Streamlit UI with 7 sections:
  Home | Upload & Analyze | Dashboard | Subject Analysis |
  Student Search | Comparison | AI Agent Chat
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="University AI Analytics Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark academic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0f172a;
    --surface:   #1e293b;
    --surface2:  #334155;
    --border:    #475569;
    --text:      #e2e8f0;
    --muted:     #94a3b8;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --success:   #34d399;
    --warning:   #fbbf24;
    --danger:    #f87171;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="stMetricValue"]  { color: var(--accent) !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.1em; }

/* Headers */
h1 { color: var(--accent) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { color: var(--text) !important; font-weight: 600 !important; }
h3 { color: var(--accent2) !important; font-weight: 500 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: transform 0.1s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.35) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* DataFrames */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Chat messages */
.chat-bubble-user {
    background: linear-gradient(135deg, var(--accent2), #6366f1);
    color: #fff;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0 0.4rem 15%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-bubble-ai {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.4rem 15% 0.4rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-label { font-size: 0.7rem; color: var(--muted); margin-bottom: 0.2rem; text-transform: uppercase; letter-spacing: 0.1em; }

/* Stat cards */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    color: var(--accent);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    margin: 0.1rem;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'IBM Plex Sans', sans-serif !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }

/* Expanders */
.streamlit-expanderHeader { background: var(--surface) !important; color: var(--text) !important; border-radius: 8px !important; }
.streamlit-expanderContent { background: var(--surface) !important; border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Module imports
# ─────────────────────────────────────────────────────────────────────────────
from file_handler import load_file
from data_preprocessing import preprocess, get_student_row
from rag_engine import RAGEngine, build_chunks
from chatbot import UniversityAgent
from model import train_model, predict_batch
import dashboard as dash
from tools import (
    get_dataset_summary, get_department_stats,
    get_top_students, get_attendance_analysis, get_subject_analysis,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load API key from Streamlit secrets (hidden from UI)
# ─────────────────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "meta": None,
        "agent": None,
        "rag": None,
        "ml_model": None,
        "chat_history": [],   # list of (role, content)
        "api_key": _get_api_key(),
        "rag_built": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 University Agent")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📂 Upload & Analyze", "📊 Dashboard",
         "📚 Subject Analysis", "🔍 Student Search",
         "⚖️ Comparison", "🤖 AI Agent Chat"],
        label_visibility="collapsed",
    )

    # ── API key is loaded silently from st.secrets — no input shown ──

    if st.session_state["meta"]:
        meta = st.session_state["meta"]
        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"- **Rows:** {meta['n_students']:,}")
        st.markdown(f"- **Departments:** {meta['n_departments']}")
        st.markdown(f"- **Subjects:** {len(meta['subject_cols'])}")
        if meta["has_attendance"]:
            st.markdown("- ✅ Attendance column")
        if st.session_state["rag_built"]:
            st.markdown("- ✅ RAG index built")
        if st.session_state["ml_model"]:
            m = st.session_state["ml_model"]
            st.markdown(f"- ✅ ML model R²={m['metrics']['r2']}")

    st.markdown("---")
    st.caption("University AI Analytics Agent v1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _require_data():
    if st.session_state["meta"] is None:
        st.warning("⚠️ Please upload a dataset first on the **Upload & Analyze** page.")
        st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎓 University AI Analytics Agent")
    st.markdown("### Intelligent academic data analysis powered by LLM + RAG + Tools")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-card'>
        <h3>🧠 AI Agent</h3>
        <p style='color:#94a3b8;font-size:0.9rem'>
        GPT-4 powered agent with tool calling, RAG retrieval, and conversation memory.
        </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
        <h3>📡 RAG Engine</h3>
        <p style='color:#94a3b8;font-size:0.9rem'>
        FAISS vector index over your dataset for semantic search and context retrieval.
        </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
        <h3>🔧 8 Tools</h3>
        <p style='color:#94a3b8;font-size:0.9rem'>
        Dataset summary, department stats, top students, attendance analysis, ML prediction & more.
        </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    steps = [
        ("1", "📂 Upload & Analyze", "Upload CSV / Excel / JSON dataset"),
        ("2", "📊 Dashboard", "Explore auto-generated charts"),
        ("3", "🔍 Student Search", "Search and view student profiles"),
        ("4", "🤖 AI Agent Chat", "Ask questions in natural language"),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='stat-card' style='text-align:center'>
            <div style='font-size:2rem;font-weight:700;color:#38bdf8'>{num}</div>
            <div style='font-weight:600;margin:0.4rem 0'>{title}</div>
            <div style='color:#94a3b8;font-size:0.8rem'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 Sample Questions")
    samples = [
        "Which department has the highest average marks?",
        "Show me students with attendance below 75%",
        "Who are the top 5 students overall?",
        "Predict performance for student John",
        "What is the average score in Mathematics?",
        "Compare department-wise attendance",
    ]
    cols = st.columns(3)
    for i, q in enumerate(samples):
        with cols[i % 3]:
            st.markdown(f"<span class='tag'>💬 {q}</span>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Upload & Analyze
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📂 Upload & Analyze":
    st.markdown("# 📂 Upload & Analyze Dataset")
    st.markdown("Supports CSV, Excel (.xlsx/.xls), and JSON. Files up to 200 MB+.")

    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("📥 Loading file…"):
            df_raw = load_file(uploaded)

        if df_raw is not None:
            with st.spinner("⚙️ Preprocessing…"):
                meta = preprocess(df_raw)
            st.session_state["meta"] = meta

            # Build RAG
            with st.spinner("🧠 Building RAG index…"):
                rag = RAGEngine(api_key=st.session_state.get("api_key"))
                chunks = build_chunks(meta)
                rag.build(chunks)
                st.session_state["rag"] = rag
                st.session_state["rag_built"] = True

            # Train ML model
            with st.spinner("📈 Training ML model…"):
                ml = train_model(meta)
                st.session_state["ml_model"] = ml

            # Create / update agent
            agent = UniversityAgent(api_key=st.session_state.get("api_key"))
            agent.attach_data(meta, rag, ml)
            st.session_state["agent"] = agent

            st.success(f"✅ Dataset loaded! {meta['n_students']:,} students, {len(meta['subject_cols'])} subjects detected.")

            # Long format notice
            if meta.get("is_long_format"):
                st.info(
                    f"🔄 **Long format detected** — Data had one row per subject "
                    f"(Subject column: `{meta['subject_col_long']}`, Marks column: `{meta['marks_col_long']}`). "
                    f"Auto-converted to wide format: **{len(meta['subject_cols'])} subjects** as separate columns."
                )

            # File info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Students", f"{meta['n_students']:,}")
            col2.metric("Columns", len(meta["df"].columns))
            col3.metric("Departments", meta["n_departments"])
            col4.metric("Subjects", len(meta["subject_cols"]))

            st.markdown("---")

            # Detected columns
            with st.expander("🔍 Detected Column Mapping", expanded=True):
                cols_info = {
                    "Student Name": meta.get("name_col") or "Not detected",
                    "Department": meta.get("dept_col") or "Not detected",
                    "Attendance": meta.get("attend_col") or "Not detected",
                    "Roll No": meta.get("roll_col") or "Not detected",
                    "Year/Semester": meta.get("year_col") or "Not detected",
                    "Subject Columns": ", ".join(meta["subject_cols"]) or "None",
                }
                for k, v in cols_info.items():
                    color = "#34d399" if "Not detected" not in str(v) and "None" not in str(v) else "#f87171"
                    st.markdown(f"**{k}:** <span style='color:{color}'>{v}</span>", unsafe_allow_html=True)

            # Data preview
            st.markdown("### 👀 Data Preview (first 20 rows)")
            st.dataframe(meta["df"].head(20), use_container_width=True)

            # RAG status
            if st.session_state["rag_built"]:
                st.info(f"🧠 RAG index: {len(chunks)} chunks indexed")

            # ML model status
            if ml:
                metrics = ml["metrics"]
                st.success(f"📈 ML model trained — R²: {metrics['r2']}  MAE: {metrics['mae']}  "
                           f"Target: `{ml['target_col']}`  Features: {ml['feature_cols']}")

            # Downloads
            st.markdown("---")
            st.markdown("### ⬇️ Download")
            c1, c2 = st.columns(2)
            with c1:
                csv_buf = meta["df"].to_csv(index=False).encode()
                st.download_button("📥 Download Cleaned CSV", csv_buf,
                                   file_name="cleaned_data.csv", mime="text/csv")
            with c2:
                xls_buf = io.BytesIO()
                with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
                    meta["df"].to_excel(writer, index=False, sheet_name="Cleaned")
                st.download_button("📥 Download Excel", xls_buf.getvalue(),
                                   file_name="cleaned_data.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# 📊 Analytics Dashboard")

    summary = get_dataset_summary(meta)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Students",      f"{summary['total_students']:,}")
    c2.metric("🏛️ Departments",   summary["n_departments"])
    c3.metric("📚 Subjects",      len(summary["subject_columns"]))
    if "class_average" in summary:
        c4.metric("📈 Class Avg", f"{summary['class_average']:.1f}")
    if "average_attendance" in summary:
        c5.metric("✅ Avg Attendance", f"{summary['average_attendance']:.1f}%")

    st.markdown("---")

    # Charts row 1 — pie + attendance only (no subject bar here)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(dash.department_pie(meta),        use_container_width=True)
    with col2:
        st.plotly_chart(dash.attendance_histogram(meta),  use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(dash.dept_marks_bar(meta),        use_container_width=True)
    with col4:
        st.plotly_chart(dash.grade_distribution(meta),    use_container_width=True)

    # Department summary table
    if meta.get("dept_col"):
        st.markdown("### 🏛️ Department Summary Table")
        dept_stats = get_department_stats(meta)
        if "departments" in dept_stats:
            rows = []
            for dept, info in dept_stats["departments"].items():
                row = {"Department": dept, "Students": info["count"]}
                if "avg_marks"      in info: row["Avg Marks"]      = info["avg_marks"]
                if "avg_attendance" in info: row["Avg Attendance"] = info["avg_attendance"]
                rows.append(row)
            st.dataframe(
                pd.DataFrame(rows).sort_values("Students", ascending=False),
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Department-wise Analysis  (renamed from Subject Analysis)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Subject Analysis":
    _require_data()
    meta     = st.session_state["meta"]
    df       = meta["df"]
    scols    = meta["subject_cols"]
    dept_col = meta.get("dept_col")
    name_col = meta.get("name_col")
    dept_subject_map = meta.get("dept_subject_map", {})

    st.markdown("# 🏛️ Department-wise Analysis")

    if not scols:
        st.error("❌ No subject/marks columns detected.")
        st.stop()

    import plotly.express as px
    import plotly.graph_objects as go

    def _score_color(v, max_v):
        p = (v / max_v) if max_v else 0
        if p >= 0.80: return "#34d399"
        if p >= 0.65: return "#fbbf24"
        if p >= 0.50: return "#fb923c"
        return "#f87171"

    BASE = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#e2e8f0"),
        margin=dict(l=10, r=60, t=50, b=30),
        xaxis=dict(color="#94a3b8", showgrid=True,
                   gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(color="#e2e8f0", automargin=True, showgrid=False),
        showlegend=False,
    )

    # ── Department buttons ────────────────────────────────────────────────────
    if not dept_col:
        st.error("❌ No department column detected.")
        st.stop()

    dept_list = sorted(df[dept_col].dropna().unique().tolist())

    st.markdown("### 🏛️ Select a Department to view its Subject Performance")
    st.caption("Click any department button below:")

    # Use session state to track selected dept
    if "sel_dept" not in st.session_state:
        st.session_state["sel_dept"] = dept_list[0]

    # Render department buttons in a row
    btn_cols = st.columns(len(dept_list))
    for i, dept in enumerate(dept_list):
        is_active = st.session_state["sel_dept"] == dept
        if btn_cols[i].button(
            dept,
            key=f"dept_btn_{i}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["sel_dept"] = dept

    selected_dept = st.session_state["sel_dept"]
    dept_df = df[df[dept_col] == selected_dept].reset_index(drop=True)

    st.markdown(f"---\n### 📊 **{selected_dept}** — Subject Performance")

    # KPI metrics for this dept
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students",    len(dept_df))
    if "Average" in dept_df.columns:
        k2.metric("Dept Avg", f"{dept_df['Average'].mean():.1f}")
    if meta.get("attend_col") and meta["attend_col"] in dept_df.columns:
        k3.metric("Avg Attendance", f"{dept_df[meta['attend_col']].mean():.1f}%")
    if "Grade" in dept_df.columns:
        k4.metric("Top Grade", dept_df["Grade"].value_counts().index[0])

    # Get subjects for this department only
    if dept_subject_map and selected_dept in dept_subject_map:
        dept_scols = [s for s in dept_subject_map[selected_dept] if s in scols]
    else:
        dept_scols = [s for s in scols if dept_df[s].notna().sum() > 0 and dept_df[s].mean() > 0]

    if not dept_scols:
        st.warning("No subjects found for this department.")
    else:
        dept_avgs = dept_df[dept_scols].mean().sort_values(ascending=True)
        max_m     = float(dept_avgs.max()) or 100
        colors    = [_score_color(v, max_m) for v in dept_avgs.values]

        # Horizontal bar chart — clean & readable
        fig_dept = go.Figure(go.Bar(
            x=dept_avgs.values,
            y=dept_avgs.index.tolist(),
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"  {v:.1f}" for v in dept_avgs.values],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
        ))
        fig_dept.update_layout(
            title=f"{selected_dept} — Average Marks per Subject",
            height=max(350, len(dept_scols) * 38 + 80),
            xaxis=dict(color="#94a3b8", range=[0, max_m * 1.2],
                       showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
            yaxis=dict(color="#e2e8f0", automargin=True, showgrid=False,
                       tickfont=dict(size=13)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans", color="#e2e8f0"),
            margin=dict(l=10, r=80, t=50, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_dept, use_container_width=True)

        # Summary table
        with st.expander("📋 View Detailed Stats Table"):
            tbl = dept_df[dept_scols].agg(["mean","max","min","std"]).T.round(1)
            tbl.columns = ["Average","Max","Min","Std Dev"]
            tbl.index.name = "Subject"
            st.dataframe(tbl.sort_values("Average", ascending=False), use_container_width=True)

        # Student list for this dept
        with st.expander(f"👥 All {len(dept_df)} Students in {selected_dept}"):
            show_cols = [c for c in [name_col, meta.get("roll_col"),
                         meta.get("year_col"), "Average", "Grade"] if c and c in dept_df.columns]
            st.dataframe(
                dept_df[show_cols].sort_values("Average", ascending=False)
                if "Average" in dept_df.columns else dept_df[show_cols],
                use_container_width=True, height=350,
            )
            csv_d = dept_df.to_csv(index=False).encode()
            st.download_button(f"📥 Download {selected_dept} Data", csv_d,
                               file_name=f"{selected_dept}_data.csv", mime="text/csv")




# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Student Search
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Student Search":
    _require_data()
    meta = st.session_state["meta"]
    df   = meta["df"]
    st.markdown("# 🔍 Student Search & Profile")

    name_col = meta.get("name_col")
    dept_col = meta.get("dept_col")
    scols    = meta["subject_cols"]

    # ── Search mode toggle ────────────────────────────────────────────────────
    search_mode = st.radio(
        "Search by:", ["👤 Student Name", "🏛️ Department"],
        horizontal=True, key="search_mode_radio"
    )

    # ── Grade badge helper ────────────────────────────────────────────────────
    def _gbadge(avg):
        if   avg >= 90: g, c = "A+", "#22d3ee"
        elif avg >= 80: g, c = "A",  "#34d399"
        elif avg >= 70: g, c = "B",  "#a3e635"
        elif avg >= 60: g, c = "C",  "#fbbf24"
        elif avg >= 50: g, c = "D",  "#fb923c"
        else:           g, c = "F",  "#f87171"
        return g, c

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1 — STUDENT NAME SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    if search_mode == "👤 Student Name":
        if name_col is None:
            st.error("❌ No student name column detected.")
            st.stop()

        query = st.text_input("🔎 Enter student name", placeholder="e.g. Ayesha, Ali, Sara…")

        if query:
            rows = get_student_row(meta, query)
            if rows.empty:
                st.warning(f"No student found matching **'{query}'**")
                st.stop()

            st.success(f"Found **{len(rows)}** student(s) matching '{query}'")

            # If multiple — let user pick ONE by ID/dept
            if len(rows) > 1:
                def _label(r):
                    lbl = str(r[name_col])
                    if meta.get("roll_col") and meta["roll_col"] in r.index:
                        lbl += f"  |  ID: {r[meta['roll_col']]}"
                    if dept_col and dept_col in r.index:
                        lbl += f"  |  {r[dept_col]}"
                    if meta.get("year_col") and meta["year_col"] in r.index:
                        lbl += f"  |  Sem {r[meta['year_col']]}"
                    return lbl
                labels = [_label(rows.iloc[i]) for i in range(len(rows))]
                sel = st.selectbox(
                    "Multiple students found — select one:",
                    range(len(rows)), format_func=lambda i: labels[i],
                    key="stu_sel"
                )
                row = rows.iloc[sel]
            else:
                row = rows.iloc[0]

            sname = str(row.get(name_col, "?"))

            # ── Info metrics ──────────────────────────────────────────────────
            st.markdown(f"### 👤 {sname}")
            c1, c2, c3, c4 = st.columns(4)
            if dept_col and dept_col in row.index:
                c1.metric("Department", str(row[dept_col]))
            if meta.get("roll_col") and meta["roll_col"] in row.index:
                c2.metric("Student ID", str(row[meta["roll_col"]]))
            if meta.get("year_col") and meta["year_col"] in row.index:
                c3.metric("Semester", str(row[meta["year_col"]]))
            if "Average" in row.index:
                c4.metric("Average", f"{row['Average']:.2f}")
            if "Grade" in row.index:
                g, gc = _gbadge(float(row.get("Average", 0)))
                st.markdown(
                    f"**GRADE** &nbsp; <span style='color:{gc};font-size:1.4rem;"
                    f"font-weight:700'>{row['Grade']}</span>",
                    unsafe_allow_html=True
                )

            # ── Subject charts — dept subjects ONLY ──────────────────────────
            if scols:
                import plotly.express as px
                import plotly.graph_objects as go

                dept_subject_map = meta.get("dept_subject_map", {})
                student_dept = str(row.get(dept_col, "")) if dept_col and dept_col in row.index else ""

                if dept_subject_map and student_dept and student_dept in dept_subject_map:
                    student_scols = [sc for sc in dept_subject_map[student_dept] if sc in scols]
                else:
                    student_scols = [
                        sc for sc in scols
                        if sc in row.index and pd.notna(row[sc]) and float(row[sc]) > 0
                    ]

                if student_scols:
                    marks     = [float(row[sc]) for sc in student_scols]
                    chart_df  = pd.DataFrame({"Subject": student_scols, "Marks": marks})
                    chart_sorted = chart_df.sort_values("Marks", ascending=True)

                    max_m  = max(marks) or 100
                    colors = []
                    for v in chart_sorted["Marks"]:
                        pct = v / max_m
                        if   pct >= 0.80: colors.append("#34d399")
                        elif pct >= 0.65: colors.append("#fbbf24")
                        elif pct >= 0.50: colors.append("#fb923c")
                        else:             colors.append("#f87171")

                    CHART_LAYOUT = dict(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="IBM Plex Sans", color="#e2e8f0", size=13),
                    )

                    # ── Bar chart (horizontal, readable) ─────────────────────
                    fig_bar = go.Figure(go.Bar(
                        x=chart_sorted["Marks"],
                        y=chart_sorted["Subject"],
                        orientation="h",
                        marker_color=colors,
                        marker_line_width=0,
                        text=[f"  {v:.0f}" for v in chart_sorted["Marks"]],
                        textposition="outside",
                        textfont=dict(color="#e2e8f0", size=13),
                    ))
                    fig_bar.update_layout(
                        title=f"📊 {sname} — Marks per Subject ({len(student_scols)} subjects)",
                        height=max(350, len(student_scols) * 38 + 80),
                        xaxis=dict(color="#94a3b8", range=[0, max_m * 1.2],
                                   showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
                        yaxis=dict(color="#e2e8f0", automargin=True, showgrid=False,
                                   tickfont=dict(size=13)),
                        margin=dict(l=10, r=80, t=50, b=20),
                        showlegend=False,
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # ── Pie chart (share of each subject's marks) ─────────────
                    fig_pie = go.Figure(go.Pie(
                        labels=chart_df["Subject"],
                        values=chart_df["Marks"],
                        hole=0.38,
                        textinfo="label+value",
                        textfont=dict(size=13, color="#e2e8f0"),
                        marker=dict(
                            colors=px.colors.qualitative.Bold[:len(student_scols)],
                            line=dict(color="#0f172a", width=2),
                        ),
                        hovertemplate="<b>%{label}</b><br>Marks: %{value:.0f}<extra></extra>",
                    ))
                    fig_pie.update_layout(
                        title=f"🥧 {sname} — Subject Marks Distribution",
                        height=420,
                        legend=dict(
                            font=dict(color="#cbd5e1", size=12),
                            bgcolor="rgba(0,0,0,0)",
                            orientation="v",
                        ),
                        margin=dict(l=10, r=10, t=50, b=20),
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                else:
                    st.info("No subject marks found for this student.")

            # ML prediction
            ml = st.session_state.get("ml_model")
            if ml:
                try:
                    row_df = pd.DataFrame([row])
                    pred   = predict_batch(ml, row_df).iloc[0]
                    st.info(f"🤖 ML Predicted **{ml['target_col']}**: **{pred:.2f}** "
                            f"*(R²={ml['metrics']['r2']})*")
                except Exception:
                    pass

            # Download
            csv_bytes = pd.DataFrame([row]).to_csv(index=False).encode()
            st.download_button("📥 Download Profile", csv_bytes,
                               file_name=f"{sname}_profile.csv", mime="text/csv")

    # ── Department search moved to "Subject Analysis" page ───────────────────
    else:
        st.info("💡 For department-wise analysis, please visit the **📚 Subject Analysis** page from the sidebar — it has interactive department buttons with subject performance charts.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Comparison
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Comparison":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# ⚖️ Student Comparison")

    name_col = meta.get("name_col")
    scols = meta["subject_cols"]

    if name_col is None:
        st.error("❌ No student name column detected.")
        st.stop()
    if not scols:
        st.error("❌ No subject columns detected.")
        st.stop()

    all_names = meta["df"][name_col].dropna().astype(str).unique().tolist()
    selected = st.multiselect(
        "Select 2–5 students to compare",
        options=all_names,
        max_selections=5,
    )

    if len(selected) < 2:
        st.info("Please select at least 2 students.")
    else:
        rows = []
        for name in selected:
            match = get_student_row(meta, name)
            if not match.empty:
                r = match.iloc[0].copy()
                r["__name__"] = name
                rows.append(r)

        if len(rows) < 2:
            st.warning("Could not find data for the selected students.")
        else:
            comp_df = pd.DataFrame(rows).reset_index(drop=True)

            # Bar chart
            st.plotly_chart(dash.comparison_bar(comp_df, scols), use_container_width=True)

            # Radar chart (need ≥3 subjects)
            if len(scols) >= 3:
                st.plotly_chart(dash.comparison_radar(comp_df, scols), use_container_width=True)

            # Summary table
            st.markdown("### 📋 Comparison Table")
            display_cols = ["__name__"] + scols
            if "Average" in comp_df.columns: display_cols.append("Average")
            if meta.get("attend_col") and meta["attend_col"] in comp_df.columns:
                display_cols.append(meta["attend_col"])
            tbl = comp_df[[c for c in display_cols if c in comp_df.columns]].copy()
            tbl = tbl.rename(columns={"__name__": "Student"})
            st.dataframe(tbl, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: AI Agent Chat
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent Chat":
    st.markdown("# 🤖 AI Agent Chat")
    st.markdown("Ask anything about the dataset. The agent uses **RAG + Tools + Memory**.")

    if st.session_state["meta"] is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    # Update agent with latest API key
    if st.session_state.get("agent"):
        agent: UniversityAgent = st.session_state["agent"]
        if st.session_state.get("api_key") and agent.client is None:
            try:
                from groq import Groq
                agent.client = Groq(api_key=st.session_state["api_key"])
            except Exception:
                pass
    else:
        agent = UniversityAgent(api_key=st.session_state.get("api_key"))
        agent.attach_data(
            st.session_state["meta"],
            st.session_state.get("rag"),
            st.session_state.get("ml_model"),
        )
        st.session_state["agent"] = agent

    # Agent status bar
    has_key = bool(st.session_state.get("api_key"))
    status_color = "#34d399" if has_key else "#fbbf24"
    status_text = "🟢 Groq LLaMA + RAG + Tools Active" if has_key else "🟡 RAG+Tools only (add Groq API key in Streamlit secrets)"
    st.markdown(
        f"<div style='background:rgba(30,41,59,0.8);border:1px solid #334155;"
        f"border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;"
        f"color:{status_color};font-size:0.85rem'>{status_text}</div>",
        unsafe_allow_html=True,
    )

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for role, content in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(
                    f"<div class='chat-label'>You</div>"
                    f"<div class='chat-bubble-user'>{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-label'>🤖 Agent</div>"
                    f"<div class='chat-bubble-ai'>{content}</div>",
                    unsafe_allow_html=True,
                )

    # Suggestion chips
    suggestions = [
        "Summarise the dataset",
        "Which department has highest marks?",
        "Show students with low attendance",
        "Who are the top 5 students?",
        "Analyse subject performance",
    ]
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}"):
            st.session_state["_pending_msg"] = sug
            st.rerun()

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder="Ask about students, departments, subjects, attendance…",
            height=80,
            label_visibility="collapsed",
            value=st.session_state.pop("_pending_msg", ""),
        )
        c_send, c_clear = st.columns([3, 1])
        send = c_send.form_submit_button("📨 Send", use_container_width=True)
        clear = c_clear.form_submit_button("🗑 Clear", use_container_width=True)

    if clear:
        st.session_state["chat_history"] = []
        agent.reset()
        st.rerun()

    if send and user_input.strip():
        st.session_state["chat_history"].append(("user", user_input.strip()))

        with st.spinner("🤖 Thinking…"):
            reply = agent.chat(user_input.strip())

        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()

    # Export chat
    if st.session_state["chat_history"]:
        chat_text = "\n\n".join(
            f"{'User' if r == 'user' else 'Agent'}: {c}"
            for r, c in st.session_state["chat_history"]
        )
        st.download_button(
            "📥 Export Chat",
            chat_text.encode(),
            file_name="chat_history.txt",
            mime="text/plain",
        )
