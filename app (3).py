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

    # KPI row
    summary = get_dataset_summary(meta)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Students", f"{summary['total_students']:,}")
    c2.metric("🏛️ Departments", summary["n_departments"])
    c3.metric("📚 Subjects", len(summary["subject_columns"]))
    if "class_average" in summary:
        c4.metric("📈 Class Avg", f"{summary['class_average']:.1f}")
    if "average_attendance" in summary:
        c5.metric("✅ Avg Attendance", f"{summary['average_attendance']:.1f}%")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(dash.marks_bar_chart(meta), use_container_width=True)
    with col2:
        st.plotly_chart(dash.department_pie(meta), use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(dash.attendance_histogram(meta), use_container_width=True)
    with col4:
        st.plotly_chart(dash.grade_distribution(meta), use_container_width=True)

    # Full-width department chart
    st.plotly_chart(dash.dept_marks_bar(meta), use_container_width=True)

    # Department table
    if meta.get("dept_col"):
        st.markdown("### 🏛️ Department Summary Table")
        dept_stats = get_department_stats(meta)
        if "departments" in dept_stats:
            rows = []
            for dept, info in dept_stats["departments"].items():
                row = {"Department": dept, "Count": info["count"]}
                if "avg_marks" in info: row["Avg Marks"] = info["avg_marks"]
                if "avg_attendance" in info: row["Avg Attendance"] = info["avg_attendance"]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).sort_values("Count", ascending=False),
                         use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Subject Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Subject Analysis":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# 📚 Subject-wise Analysis")

    scols = meta["subject_cols"]
    if not scols:
        st.error("❌ No subject/marks columns detected.")
        if meta.get("is_long_format") is False:
            st.info("💡 Tip: Your data might be in long format (one row per subject). "
                    "Make sure your Subject column is named 'Subject' and marks column is named 'Marks'.")
        st.stop()

    df = meta["df"]
    name_col = meta.get("name_col")

    import plotly.express as px

    # ── Two tabs ──────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Class Overview", "👤 Per-Student Subject Performance"])

    # ════ TAB 1: Class Overview ════════════════════════════════════════════════
    with tab1:
        n_subjects = len(scols)

        # ── Top metrics ───────────────────────────────────────────────────────
        avgs_series = df[scols].mean().sort_values(ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Subjects", n_subjects)
        c2.metric("Highest Avg Subject", f"{avgs_series.index[0]} ({avgs_series.iloc[0]:.1f})")
        c3.metric("Lowest Avg Subject",  f"{avgs_series.index[-1]} ({avgs_series.iloc[-1]:.1f})")
        c4.metric("Overall Average", f"{df[scols].mean().mean():.1f}")

        st.markdown("---")

        # ── Chart selection based on subject count ────────────────────────────
        if n_subjects <= 12:
            # Clean horizontal bar — fully readable
            st.plotly_chart(dash.marks_bar_chart(meta), use_container_width=True)

        else:
            # Too many subjects — offer 3 smart views
            view = st.radio(
                "📊 Chart style choose karein:",
                ["🔥 Heatmap (Dept × Subject)", "🗂️ Treemap (Overview)", "📊 Top/Bottom Bar"],
                horizontal=True, key="subj_chart_view"
            )

            if view == "🔥 Heatmap (Dept × Subject)":
                st.caption("💡 Har department ka har subject mein average — color green = acha, red = kamzor")
                st.plotly_chart(dash.subject_heatmap_by_dept(meta), use_container_width=True)

            elif view == "🗂️ Treemap (Overview)":
                st.caption("💡 Bada box = zyada students, green = high marks, red = low marks")
                st.plotly_chart(dash.subject_treemap(meta), use_container_width=True)

            else:
                st.caption("💡 Top 10 highest + Bottom 5 lowest average subjects")
                st.plotly_chart(dash.marks_bar_chart(meta), use_container_width=True)

        st.markdown("---")

        # ── Per-Subject dropdown stats ────────────────────────────────────────
        subject = st.selectbox("🔍 Kisi ek subject ki detail dekhein", scols, key="subj_class")
        col_data = df[subject].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average",  f"{col_data.mean():.2f}")
        c2.metric("Highest",  f"{col_data.max():.2f}")
        c3.metric("Lowest",   f"{col_data.min():.2f}")
        c4.metric("Std Dev",  f"{col_data.std():.2f}")

        if name_col:
            st.plotly_chart(
                dash.subject_top_students(df, subject, name_col, n=10),
                use_container_width=True,
            )

        st.markdown("### 📋 All Subjects Summary Table")
        subj_data = get_subject_analysis(meta)
        if "subjects" in subj_data:
            sum_rows = [
                {"Subject": s, "Average": info["average"], "Max": info["max"],
                 "Min": info["min"], "Std Dev": info["std"]}
                for s, info in subj_data["subjects"].items()
            ]
            st.dataframe(
                pd.DataFrame(sum_rows).sort_values("Average", ascending=False),
                use_container_width=True,
            )

    # ════ TAB 2: Per-Student Subject Performance ═══════════════════════════════
    with tab2:
        st.markdown("### 👤 Har Student ki Alag Subject Performance")

        if not name_col:
            st.error("❌ Student name column detect nahi hua.")
            st.stop()

        # ── Search ────────────────────────────────────────────────────────────
        search_q = st.text_input(
            "🔎 Student naam search karein",
            placeholder="e.g. Ayesha, Bilal, Ahmed…",
            key="per_student_subj_search"
        )

        # ── Color helpers ─────────────────────────────────────────────────────
        def _bar_color(score: float, max_s: float) -> str:
            pct = (score / max_s * 100) if max_s else 0
            if pct >= 80: return "#34d399"
            if pct >= 60: return "#fbbf24"
            if pct >= 40: return "#fb923c"
            return "#f87171"

        def _grade_badge(avg: float) -> str:
            if   avg >= 90: g, c = "A+", "#22d3ee"
            elif avg >= 80: g, c = "A",  "#34d399"
            elif avg >= 70: g, c = "B",  "#a3e635"
            elif avg >= 60: g, c = "C",  "#fbbf24"
            elif avg >= 50: g, c = "D",  "#fb923c"
            else:           g, c = "F",  "#f87171"
            return (f"<span style='background:{c}22;color:{c};border:1px solid {c}55;"
                    f"border-radius:20px;padding:0.15rem 0.65rem;font-size:0.82rem;"
                    f"font-weight:700'>{g}</span>")

        # Max per subject for relative bar
        max_per_subj = {sc: float(df[sc].max()) if df[sc].max() > 0 else 100 for sc in scols}

        # ── Filter & Select ONE student ───────────────────────────────────────
        import re as _re

        roll_col = meta.get("roll_col")   # Student_ID column

        if search_q.strip():
            mask = df[name_col].astype(str).str.lower().str.contains(
                _re.escape(search_q.strip().lower()), na=False
            )
            matched_df = df[mask].reset_index(drop=True)
            if matched_df.empty:
                st.warning(f"**'{search_q}'** naam ka koi student nahi mila.")
                st.stop()

            # ── If multiple rows found, let user pick ONE by ID ───────────────
            if len(matched_df) > 1:
                # Build display labels: "Ayesha (ID: 1001 | CS)" 
                def _row_label(r):
                    lbl = str(r[name_col])
                    if roll_col and roll_col in r.index:
                        lbl += f"  |  ID: {r[roll_col]}"
                    if meta.get("dept_col") and meta["dept_col"] in r.index:
                        lbl += f"  |  {r[meta['dept_col']]}"
                    if meta.get("year_col") and meta["year_col"] in r.index:
                        lbl += f"  |  Sem {r[meta['year_col']]}"
                    return lbl

                labels   = [_row_label(matched_df.iloc[i]) for i in range(len(matched_df))]
                selected = st.selectbox(
                    f"✅ {len(matched_df)} students mili — kisi ek ko select karein:",
                    options=range(len(matched_df)),
                    format_func=lambda i: labels[i],
                    key="student_selector"
                )
                show_df = matched_df.iloc[[selected]].reset_index(drop=True)
            else:
                show_df = matched_df

        else:
            # ── Pagination for all students ───────────────────────────────────
            page_size   = 10
            total_stu   = len(df)
            total_pages = max(1, (total_stu + page_size - 1) // page_size)
            pg = st.number_input(
                f"Page (1–{total_pages})", min_value=1, max_value=total_pages,
                value=1, step=1, key="stu_pg"
            )
            start   = (pg - 1) * page_size
            show_df = df.iloc[start: start + page_size].reset_index(drop=True)
            st.caption(f"Students {start+1}–{min(start+page_size, total_stu)} of {total_stu}")

        # ── Render ONE card per row ───────────────────────────────────────────
        for _, row in show_df.iterrows():
            sname = str(row.get(name_col, "Unknown"))
            avg   = float(row["Average"]) if "Average" in row.index else float(
                np.nanmean([row[sc] for sc in scols if sc in row.index and not np.isnan(row[sc])])
            )

            # Subject progress bars — only subjects that have a value
            bars_html = ""
            valid_scols = [sc for sc in scols if sc in row.index and not np.isnan(float(row[sc] if row[sc] == row[sc] else 0))]
            for sc in valid_scols:
                score = float(row[sc])
                max_s = max_per_subj[sc]
                color = _bar_color(score, max_s)
                bar_w = min(100, round((score / max_s) * 100)) if max_s else 0
                bars_html += f"""
                <div style='margin:0.35rem 0'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:0.1rem'>
                    <span style='font-size:0.82rem;color:#cbd5e1;font-weight:500'>{sc}</span>
                    <span style='font-size:0.85rem;font-weight:700;color:{color}'>{score:.0f}</span>
                  </div>
                  <div style='background:#334155;border-radius:6px;height:8px'>
                    <div style='background:{color};width:{bar_w}%;height:8px;border-radius:6px'></div>
                  </div>
                </div>"""

            # Info badges
            extra = ""
            for key, label in [("dept_col","Dept"),("roll_col","ID"),
                                ("year_col","Semester"),("attend_col","Attendance")]:
                cname = meta.get(key)
                if cname and cname in row.index:
                    extra += (f"<span style='background:#1e3a5f;color:#93c5fd;border-radius:12px;"
                              f"padding:0.1rem 0.6rem;font-size:0.76rem;margin-right:0.4rem'>"
                              f"{label}: {row[cname]}</span>")

            grade_html = _grade_badge(avg)

            st.markdown(f"""
            <div style='background:#1e293b;border:1px solid #334155;border-radius:14px;
                        padding:1.3rem 1.5rem;margin-bottom:1rem'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start;
                          margin-bottom:0.5rem'>
                <div>
                  <span style='font-size:1.1rem;font-weight:700;color:#e2e8f0'>👤 {sname}</span>
                  &nbsp;&nbsp;{grade_html}
                </div>
                <div style='text-align:right'>
                  <div style='font-size:0.72rem;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:0.08em'>Average</div>
                  <div style='font-size:1.4rem;font-weight:700;color:#38bdf8'>{avg:.1f}</div>
                </div>
              </div>
              <div style='margin-bottom:0.8rem'>{extra}</div>
              <hr style='border-color:#334155;margin:0.6rem 0'>
              <div style='columns:2;column-gap:1.5rem'>{bars_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Individual bar chart (sirf is student ke valid subjects) ──────
            if valid_scols:
                chart_data = pd.DataFrame({
                    "Subject": valid_scols,
                    "Marks":   [float(row[sc]) for sc in valid_scols],
                })
                fig_one = px.bar(
                    chart_data, x="Subject", y="Marks",
                    color="Subject",
                    text_auto=".0f",
                    title=f"📊 {sname} — Subject-wise Marks",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_one.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="IBM Plex Sans", color="#e2e8f0"),
                    margin=dict(l=30, r=30, t=50, b=60),
                    xaxis=dict(color="#94a3b8", tickangle=-30),
                    yaxis=dict(color="#94a3b8", range=[0, 105]),
                    showlegend=False,
                )
                st.plotly_chart(fig_one, use_container_width=True)

        # ── Full table ────────────────────────────────────────────────────────
        with st.expander("📋 Sab Students — Table View"):
            disp = [name_col] + scols
            if "Average" in df.columns: disp.append("Average")
            if "Grade"   in df.columns: disp.append("Grade")
            if meta.get("dept_col"):    disp.append(meta["dept_col"])
            st.dataframe(df[[c for c in disp if c in df.columns]],
                         use_container_width=True, height=400)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Student Search
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Student Search":
    _require_data()
    meta = st.session_state["meta"]
    st.markdown("# 🔍 Student Search & Profile")

    name_col = meta.get("name_col")
    if name_col is None:
        st.error("❌ No student name column detected.")
        st.stop()

    query = st.text_input("🔎 Enter student name", placeholder="e.g. Alice, John…")

    if query:
        rows = get_student_row(meta, query)
        if rows.empty:
            st.warning(f"No students found matching **'{query}'**")
        else:
            st.success(f"Found **{len(rows)}** student(s)")
            for idx in range(min(len(rows), 5)):
                row = rows.iloc[idx]
                name = str(row.get(name_col, "?"))
                st.markdown(f"### 👤 {name}")

                info_cols = st.columns(4)
                i = 0
                for col_name in [meta.get("dept_col"), meta.get("roll_col"),
                                  meta.get("year_col"), meta.get("attend_col")]:
                    if col_name and col_name in row.index:
                        val = row[col_name]
                        label = col_name.replace("_", " ").title()
                        info_cols[i % 4].metric(label, f"{val}")
                        i += 1

                if "Average" in row.index:
                    info_cols[i % 4].metric("Average", f"{row['Average']:.2f}")
                    i += 1
                if "Grade" in row.index:
                    info_cols[i % 4].metric("Grade", str(row["Grade"]))

                # Subject chart
                scols = meta["subject_cols"]
                if scols:
                    fig = dash.student_subject_bar(row, scols, name)
                    st.plotly_chart(fig, use_container_width=True)

                # Full record
                with st.expander("📋 Full Record"):
                    record = {k: v for k, v in row.items()
                              if not str(k).startswith("_")}
                    st.json({k: (float(v) if isinstance(v, (np.floating, np.integer)) else str(v))
                             for k, v in record.items()})

                # ML prediction
                ml = st.session_state.get("ml_model")
                if ml:
                    try:
                        row_df = pd.DataFrame([row])
                        pred = predict_batch(ml, row_df).iloc[0]
                        st.info(f"🤖 Predicted **{ml['target_col']}**: **{pred:.2f}**  "
                                f"*(ML model — R²={ml['metrics']['r2']})*")
                    except Exception:
                        pass

                st.markdown("---")

    # Download search result
    if query and not get_student_row(meta, query).empty:
        result_df = get_student_row(meta, query)
        csv_bytes = result_df.to_csv(index=False).encode()
        st.download_button("📥 Download Results", csv_bytes,
                           file_name=f"search_{query}.csv", mime="text/csv")


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
