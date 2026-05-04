"""
AI University Analytics Agent
FYP-Level Production Application
Dark Navy Theme | Professional UI | Correct Data Logic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import io
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UniAnalytics AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — DARK NAVY PROFESSIONAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:   #0B1F3A;
    --bg-secondary: #0F2850;
    --bg-card:      #132d58;
    --bg-hover:     #1a3a6e;
    --accent-blue:  #3A8DFF;
    --accent-cyan:  #00D4FF;
    --accent-teal:  #00C9A7;
    --accent-purple:#A855F7;
    --accent-gold:  #F59E0B;
    --text-primary: #EAEAEA;
    --text-secondary:#CFCFCF;
    --text-muted:   #8BA3C7;
    --border:       #1E3F70;
    --success:      #10B981;
    --warning:      #F59E0B;
    --danger:       #EF4444;
}

/* ── App background ── */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091729 0%, #0B1F3A 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* ── Main content area ── */
[data-testid="stMain"] {
    background-color: var(--bg-primary) !important;
}
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px;
}

/* ── Headers ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.8rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--accent-teal) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s ease;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Select boxes & inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox label, .stTextInput label,
.stTextArea label, .stMultiSelect label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* ── Dataframes / tables ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}
.dvn-scroller { background: var(--bg-card) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
}

/* ── Expander ── */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
summary { color: var(--text-primary) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
.stChatInput > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stChatInput textarea { color: var(--text-primary) !important; }

/* ── Custom card HTML ── */
.uni-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.uni-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.uni-card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-cyan);
}
.uni-card-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-blue   { background: #1a3a6e; color: #3A8DFF; }
.badge-cyan   { background: #0e3040; color: #00D4FF; }
.badge-teal   { background: #0c3030; color: #00C9A7; }
.badge-purple { background: #2e1a4e; color: #A855F7; }
.badge-gold   { background: #3d2800; color: #F59E0B; }

/* Streamlit default text overrides */
p, span, div, li { color: var(--text-primary); }
.stMarkdown p { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────
PALETTE = ["#3A8DFF", "#00D4FF", "#00C9A7", "#A855F7", "#F59E0B",
           "#EF4444", "#10B981", "#F97316", "#EC4899", "#6366F1"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#CFCFCF", size=13),
    title_font=dict(family="Space Grotesk, sans-serif", color="#EAEAEA", size=16),
    legend=dict(
        bgcolor="rgba(15,40,80,0.7)",
        bordercolor="#1E3F70",
        borderwidth=1,
        font=dict(color="#CFCFCF"),
    ),
    xaxis=dict(
        gridcolor="#1E3F70",
        linecolor="#1E3F70",
        tickfont=dict(color="#CFCFCF"),
        title_font=dict(color="#8BA3C7"),
        zerolinecolor="#1E3F70",
    ),
    yaxis=dict(
        gridcolor="#1E3F70",
        linecolor="#1E3F70",
        tickfont=dict(color="#CFCFCF"),
        title_font=dict(color="#8BA3C7"),
        zerolinecolor="#1E3F70",
    ),
    margin=dict(l=40, r=30, t=50, b=40),
)


def styled_chart(fig, height=420):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


# ─────────────────────────────────────────────
# SAMPLE DATA GENERATOR
# ─────────────────────────────────────────────
def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    departments = {
        "Computer Science": {
            1: ["Programming Fundamentals", "Calculus I", "English Composition", "Digital Logic"],
            2: ["Object Oriented Programming", "Calculus II", "Discrete Mathematics", "Linear Algebra"],
            3: ["Data Structures", "Algorithms", "Database Systems", "Statistics"],
            4: ["Operating Systems", "Computer Networks", "Software Engineering", "AI Fundamentals"],
        },
        "Electrical Engineering": {
            1: ["Circuit Analysis", "Engineering Mathematics", "Physics I", "Workshop Practice"],
            2: ["Electronics I", "Circuit Theory", "Signals & Systems", "Physics II"],
            3: ["Digital Electronics", "Electromagnetic Fields", "Control Systems", "Measurements"],
            4: ["Power Systems", "Microprocessors", "Communication Systems", "VLSI Design"],
        },
        "Business Administration": {
            1: ["Principles of Management", "Business Communication", "Microeconomics", "Accounting I"],
            2: ["Marketing Fundamentals", "Macroeconomics", "Accounting II", "Business Statistics"],
            3: ["Financial Management", "Human Resource Management", "Operations Management", "Business Law"],
            4: ["Strategic Management", "International Business", "Entrepreneurship", "Business Ethics"],
        },
        "Mathematics": {
            1: ["Calculus I", "Linear Algebra I", "Introduction to Proofs", "Statistics I"],
            2: ["Calculus II", "Linear Algebra II", "Discrete Mathematics", "Statistics II"],
            3: ["Real Analysis", "Abstract Algebra", "Numerical Methods", "Probability Theory"],
            4: ["Complex Analysis", "Topology", "Differential Equations", "Mathematical Modeling"],
        },
    }

    student_names = [
        "Ayesha Khan", "Ali Hassan", "Fatima Malik", "Usman Ahmed", "Sana Raza",
        "Bilal Qureshi", "Zara Hussain", "Omar Sheikh", "Hira Baig", "Tariq Mehmood",
        "Amna Siddiqui", "Kamran Ali", "Nadia Iqbal", "Asad Butt", "Rabia Tahir",
        "Hassan Mirza", "Sara Chaudhry", "Imran Javed", "Maria Abbasi", "Faisal Nawaz",
    ]

    records = []
    dept_students = {dept: [] for dept in departments}
    for i, name in enumerate(student_names):
        dept = list(departments.keys())[i % len(departments)]
        dept_students[dept].append(name)

    for dept, students in dept_students.items():
        for student in students:
            semester = np.random.choice([1, 2, 3, 4])
            subjects = departments[dept][semester]
            base_performance = np.random.randint(50, 85)
            for subject in subjects:
                marks = int(np.clip(base_performance + np.random.randint(-15, 20), 30, 100))
                records.append({
                    "Student_Name": student,
                    "Department": dept,
                    "Semester": semester,
                    "Subject": subject,
                    "Marks": marks,
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "Home"


# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────
def get_df() -> pd.DataFrame | None:
    return st.session_state.df


def grade_label(marks: float) -> str:
    if marks >= 90: return "A+"
    elif marks >= 80: return "A"
    elif marks >= 70: return "B"
    elif marks >= 60: return "C"
    elif marks >= 50: return "D"
    return "F"


def grade_color(marks: float) -> str:
    if marks >= 80: return "badge-teal"
    elif marks >= 60: return "badge-blue"
    elif marks >= 50: return "badge-gold"
    return "badge-purple"


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:1.4rem">
        <h2 style="
            font-family:'Space Grotesk',sans-serif;
            font-size:1.55rem;
            font-weight:700;
            color:#EAEAEA;
            margin:0 0 4px 0;
            letter-spacing:-0.01em;
        ">{title}</h2>
        {'<p style="color:#8BA3C7;font-size:0.9rem;margin:0">'+subtitle+'</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value, sub: str = "", col=None):
    html = f"""
    <div class="uni-card">
        <div class="uni-card-title">{label}</div>
        <div class="uni-card-value">{value}</div>
        {'<div class="uni-card-sub">'+sub+'</div>' if sub else ''}
    </div>
    """
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.2rem 0.5rem 1.6rem;border-bottom:1px solid #1E3F70;margin-bottom:1rem">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:700;color:#EAEAEA">
                🎓 UniAnalytics
            </div>
            <div style="font-size:0.75rem;color:#8BA3C7;margin-top:3px">AI University Analytics Agent</div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "🏠  Home": "Home",
            "📤  Upload & Analyze": "Upload",
            "📊  Dashboard": "Dashboard",
            "🏢  Department-wise Analysis": "Department",
            "🔍  Student Search": "Student",
            "⚖️  Comparison": "Comparison",
            "🤖  AI Agent Chat": "Chat",
        }

        for label, key in pages.items():
            active = st.session_state.page == key
            btn_style = "background:linear-gradient(90deg,#1a3a6e,#132d58);color:#3A8DFF;" if active else ""
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
            ):
                st.session_state.page = key
                st.rerun()

        # Dataset status indicator
        st.markdown("<br>", unsafe_allow_html=True)
        df = get_df()
        if df is not None:
            st.markdown(f"""
            <div style="background:#0c3030;border:1px solid #00C9A7;border-radius:9px;padding:0.8rem 1rem">
                <div style="font-size:0.7rem;color:#00C9A7;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Dataset Loaded</div>
                <div style="color:#EAEAEA;font-size:0.85rem;margin-top:3px">{len(df):,} records</div>
                <div style="color:#8BA3C7;font-size:0.75rem">{df['Student_Name'].nunique()} students · {df['Department'].nunique()} departments</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#2e1a0e;border:1px solid #F59E0B;border-radius:9px;padding:0.8rem 1rem">
                <div style="font-size:0.7rem;color:#F59E0B;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">No Dataset</div>
                <div style="color:#CFCFCF;font-size:0.8rem;margin-top:3px">Upload a file or load sample data</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0F2850 0%,#132d58 100%);
        border:1px solid #1E3F70;
        border-radius:18px;
        padding:2.5rem 2.8rem;
        margin-bottom:2rem;
    ">
        <div style="
            font-family:'Space Grotesk',sans-serif;
            font-size:2.2rem;
            font-weight:700;
            color:#EAEAEA;
            line-height:1.2;
            margin-bottom:0.6rem;
        ">AI University Analytics Agent</div>
        <div style="color:#8BA3C7;font-size:1rem;max-width:600px;line-height:1.6">
            A final year project delivering intelligent academic performance insights — 
            powered by AI, designed for clarity, built for decision-makers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("📤", "Upload & Analyze", "Import your CSV dataset and instantly validate structure and quality.", "badge-blue"),
        ("📊", "Dashboard", "High-level KPIs and institution-wide performance overview.", "badge-cyan"),
        ("🏢", "Department-wise Analysis", "Filter by department, explore top-performing subjects.", "badge-teal"),
        ("🔍", "Student Search", "Search any student and view their personal subject-wise performance.", "badge-purple"),
        ("⚖️", "Comparison", "Compare multiple students or departments side by side.", "badge-gold"),
        ("🤖", "AI Agent Chat", "Ask natural language questions about your academic data.", "badge-blue"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc, badge) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="uni-card" style="height:160px">
                <div style="font-size:1.6rem;margin-bottom:0.5rem">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:0.95rem;font-weight:600;color:#EAEAEA;margin-bottom:0.4rem">{title}</div>
                <div style="font-size:0.8rem;color:#8BA3C7;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Load Sample Dataset", use_container_width=True):
            st.session_state.df = generate_sample_data()
            st.session_state.page = "Dashboard"
            st.rerun()
    with col2:
        if st.button("📤 Go to Upload Page", use_container_width=True):
            st.session_state.page = "Upload"
            st.rerun()


# ─────────────────────────────────────────────
# PAGE: UPLOAD & ANALYZE
# ─────────────────────────────────────────────
def page_upload():
    section_header("Upload & Analyze", "Import your CSV dataset and validate its structure")

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Required columns: Student_Name, Department, Semester, Subject, Marks",
        )

        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                required = {"Student_Name", "Department", "Semester", "Subject", "Marks"}
                missing = required - set(df.columns)

                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                else:
                    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
                    df.dropna(subset=["Marks"], inplace=True)
                    df["Marks"] = df["Marks"].astype(int).clip(0, 100)
                    st.session_state.df = df
                    st.success(f"✅ Dataset loaded successfully — {len(df):,} records")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col2:
        st.markdown("""
        <div class="uni-card">
            <div class="uni-card-title">Required Columns</div>
            <div style="margin-top:0.8rem">
        """, unsafe_allow_html=True)
        for col, dtype in [
            ("Student_Name", "Text"),
            ("Department", "Text"),
            ("Semester", "Integer (1–8)"),
            ("Subject", "Text"),
            ("Marks", "Integer (0–100)"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid #1E3F70">
                <span style="color:#EAEAEA;font-size:0.85rem"><code style="color:#00D4FF;background:#0e2040;padding:1px 6px;border-radius:4px">{col}</code></span>
                <span style="color:#8BA3C7;font-size:0.78rem">{dtype}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Sample data
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("📋 Load Sample Data", use_container_width=True):
            st.session_state.df = generate_sample_data()
            st.rerun()

    df = get_df()
    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Data Preview", f"{len(df):,} records loaded")

        c1, c2, c3, c4 = st.columns(4)
        stat_card("Total Records", f"{len(df):,}", col=c1)
        stat_card("Students", df["Student_Name"].nunique(), col=c2)
        stat_card("Departments", df["Department"].nunique(), col=c3)
        stat_card("Avg. Marks", f"{df['Marks'].mean():.1f}", col=c4)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            df.head(50).style.background_gradient(subset=["Marks"], cmap="Blues"),
            use_container_width=True,
            height=320,
        )

        # Data quality
        with st.expander("📋 Data Quality Report"):
            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Null Values", df.isnull().sum().sum())
            with q2:
                st.metric("Duplicate Rows", df.duplicated().sum())
            with q3:
                st.metric("Marks Range", f"{df['Marks'].min()} – {df['Marks'].max()}")

            marks_dist = go.Figure(go.Histogram(
                x=df["Marks"], nbinsx=20,
                marker_color="#3A8DFF",
                marker_line=dict(color="#0B1F3A", width=1),
            ))
            marks_dist.update_layout(
                title="Marks Distribution — Full Dataset",
                xaxis_title="Marks", yaxis_title="Frequency",
                **PLOTLY_LAYOUT, height=280,
            )
            st.plotly_chart(marks_dist, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard():
    df = get_df()
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload or load sample data first.")
        return

    section_header("Analytics Dashboard", "Institution-wide performance overview")

    # KPI row
    avg_marks = df["Marks"].mean()
    pass_rate = (df["Marks"] >= 50).mean() * 100
    top_dept = df.groupby("Department")["Marks"].mean().idxmax()
    total_students = df["Student_Name"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Average", f"{avg_marks:.1f}", f"out of 100")
    c2.metric("Pass Rate", f"{pass_rate:.1f}%", f"+{pass_rate-50:.1f}% above threshold")
    c3.metric("Total Students", total_students)
    c4.metric("Top Department", top_dept)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Dept avg + Semester trend
    col1, col2 = st.columns(2)

    with col1:
        dept_avg = df.groupby("Department")["Marks"].mean().sort_values(ascending=True).reset_index()
        fig = go.Figure(go.Bar(
            x=dept_avg["Marks"],
            y=dept_avg["Department"],
            orientation="h",
            marker=dict(
                color=dept_avg["Marks"],
                colorscale=[[0, "#3A8DFF"], [0.5, "#00D4FF"], [1, "#00C9A7"]],
                showscale=False,
            ),
            text=dept_avg["Marks"].round(1),
            textposition="outside",
            textfont=dict(color="#CFCFCF", size=12),
        ))
        fig.update_layout(title="Average Marks by Department", xaxis_title="Average Marks", **PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sem_avg = df.groupby("Semester")["Marks"].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sem_avg["Semester"], y=sem_avg["Marks"],
            mode="lines+markers",
            line=dict(color="#00D4FF", width=3),
            marker=dict(color="#00D4FF", size=10, line=dict(color="#0B1F3A", width=2)),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)",
            name="Avg Marks",
        ))
        fig2.update_layout(
            title="Average Marks by Semester",
            xaxis_title="Semester", yaxis_title="Average Marks",
            **PLOTLY_LAYOUT, height=340,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Grade distribution + Dept student count
    col3, col4 = st.columns(2)

    with col3:
        df["Grade"] = df["Marks"].apply(grade_label)
        grade_counts = df["Grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        order = ["A+", "A", "B", "C", "D", "F"]
        grade_counts["Grade"] = pd.Categorical(grade_counts["Grade"], categories=order, ordered=True)
        grade_counts = grade_counts.sort_values("Grade")
        fig3 = go.Figure(go.Bar(
            x=grade_counts["Grade"],
            y=grade_counts["Count"],
            marker_color=PALETTE[:len(grade_counts)],
            text=grade_counts["Count"],
            textposition="outside",
            textfont=dict(color="#CFCFCF"),
        ))
        fig3.update_layout(title="Grade Distribution (All Records)", xaxis_title="Grade", yaxis_title="Count", **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        dept_students = df.groupby("Department")["Student_Name"].nunique().reset_index()
        dept_students.columns = ["Department", "Students"]
        fig4 = go.Figure(go.Pie(
            labels=dept_students["Department"],
            values=dept_students["Students"],
            hole=0.5,
            marker=dict(colors=PALETTE[:len(dept_students)], line=dict(color="#0B1F3A", width=2)),
            textfont=dict(color="#EAEAEA"),
        ))
        fig4.update_layout(title="Students per Department", **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig4, use_container_width=True)

    # Top & Bottom students
    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.9rem;font-weight:600;color:#00C9A7;margin-bottom:0.5rem">🏆 Top 5 Students (by Average Marks)</div>', unsafe_allow_html=True)
        top5 = df.groupby("Student_Name")["Marks"].mean().nlargest(5).reset_index()
        top5.columns = ["Student", "Avg Marks"]
        top5["Avg Marks"] = top5["Avg Marks"].round(1)
        top5["Grade"] = top5["Avg Marks"].apply(grade_label)
        st.dataframe(top5, use_container_width=True, hide_index=True)

    with col6:
        st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.9rem;font-weight:600;color:#EF4444;margin-bottom:0.5rem">⚠️ Bottom 5 Students (Need Attention)</div>', unsafe_allow_html=True)
        bot5 = df.groupby("Student_Name")["Marks"].mean().nsmallest(5).reset_index()
        bot5.columns = ["Student", "Avg Marks"]
        bot5["Avg Marks"] = bot5["Avg Marks"].round(1)
        bot5["Grade"] = bot5["Avg Marks"].apply(grade_label)
        st.dataframe(bot5, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE: DEPARTMENT-WISE ANALYSIS
# ─────────────────────────────────────────────
def page_department():
    df = get_df()
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        return

    section_header("Department-wise Analysis", "Explore subject performance filtered by department")

    departments = sorted(df["Department"].unique().tolist())

    col_sel, col_sem = st.columns([2, 2])
    with col_sel:
        selected_dept = st.selectbox(
            "Select Department",
            departments,
            index=0,
            key="dept_select",
        )
    with col_sem:
        semesters = ["All Semesters"] + sorted(df[df["Department"] == selected_dept]["Semester"].unique().tolist())
        selected_sem = st.selectbox("Filter by Semester", semesters, key="dept_sem")

    # Filter
    dept_df = df[df["Department"] == selected_dept].copy()
    if selected_sem != "All Semesters":
        dept_df = dept_df[dept_df["Semester"] == selected_sem]

    if dept_df.empty:
        st.info("No data found for the selected filters.")
        return

    # KPIs
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", dept_df["Student_Name"].nunique())
    c2.metric("Subjects Offered", dept_df["Subject"].nunique())
    c3.metric("Avg Marks", f"{dept_df['Marks'].mean():.1f}")
    c4.metric("Pass Rate", f"{(dept_df['Marks'] >= 50).mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 5 subjects
    subj_avg = dept_df.groupby("Subject")["Marks"].mean().sort_values(ascending=False).reset_index()
    subj_avg.columns = ["Subject", "Average Marks"]
    subj_avg["Average Marks"] = subj_avg["Average Marks"].round(1)

    top5 = subj_avg.head(5)
    show_all = st.toggle("Show All Subjects", value=False, key="show_all_subjects")
    display_df = subj_avg if show_all else top5

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure(go.Bar(
            x=display_df["Average Marks"],
            y=display_df["Subject"],
            orientation="h",
            marker=dict(
                color=display_df["Average Marks"],
                colorscale=[[0, "#3A8DFF"], [0.5, "#00D4FF"], [1, "#00C9A7"]],
                showscale=False,
            ),
            text=display_df["Average Marks"],
            textposition="outside",
            textfont=dict(color="#CFCFCF", size=12),
        ))
        title_str = f"{'All' if show_all else 'Top 5'} Subjects — {selected_dept}"
        if selected_sem != "All Semesters":
            title_str += f" (Semester {selected_sem})"
        fig.update_layout(
            title=title_str,
            xaxis_title="Average Marks",
            height=max(320, len(display_df) * 44 + 100),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.9rem;font-weight:600;color:#EAEAEA;margin-bottom:0.6rem">Subject Rankings</div>', unsafe_allow_html=True)
        ranked = display_df.copy()
        ranked.index = range(1, len(ranked) + 1)
        st.dataframe(ranked, use_container_width=True, height=380)

    # Per-student subject breakdown within dept
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Student Performance in This Department", "Individual subject-wise marks")

    students_in_dept = sorted(dept_df["Student_Name"].unique().tolist())
    sel_student = st.selectbox("Select Student", students_in_dept, key="dept_student_select")

    student_data = dept_df[dept_df["Student_Name"] == sel_student][["Subject", "Marks", "Semester"]].sort_values("Marks", ascending=False)

    c_tbl, c_bar = st.columns(2)
    with c_tbl:
        st.dataframe(student_data, use_container_width=True, hide_index=True)
    with c_bar:
        fig_s = go.Figure(go.Bar(
            x=student_data["Marks"],
            y=student_data["Subject"],
            orientation="h",
            marker_color=PALETTE[:len(student_data)],
            text=student_data["Marks"],
            textposition="outside",
            textfont=dict(color="#CFCFCF"),
        ))
        fig_s.update_layout(title=f"{sel_student} — Subject Marks", xaxis_range=[0, 110], **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_s, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: STUDENT SEARCH  ← CRITICAL FIX
# ─────────────────────────────────────────────
def page_student_search():
    df = get_df()
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        return

    section_header("Student Search", "Search for a student to view their personal subject-wise performance")

    all_students = sorted(df["Student_Name"].unique().tolist())

    col_search, col_or = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("Search Student Name", placeholder="e.g. Ayesha", key="student_search_input")
    with col_or:
        st.markdown("<br>", unsafe_allow_html=True)

    # Filter matching names
    if search_query.strip():
        matches = [s for s in all_students if search_query.strip().lower() in s.lower()]
    else:
        matches = all_students

    if not matches:
        st.info("No students match your search query.")
        return

    selected_student = st.selectbox("Select Student", matches, key="student_select_dropdown")

    # ── CRITICAL: filter ONLY this student's data ──
    student_df = df[df["Student_Name"] == selected_student].copy()

    if student_df.empty:
        st.info("No records found for this student.")
        return

    # Student overview banner
    dept = student_df["Department"].iloc[0]
    semester = student_df["Semester"].iloc[0]
    avg = student_df["Marks"].mean()
    grade = grade_label(avg)

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#0F2850,#132d58);
        border:1px solid #1E3F70;
        border-radius:14px;
        padding:1.4rem 1.8rem;
        margin:1rem 0;
        display:flex;
        align-items:center;
        gap:2rem;
    ">
        <div style="
            width:56px;height:56px;border-radius:50%;
            background:linear-gradient(135deg,#3A8DFF,#00D4FF);
            display:flex;align-items:center;justify-content:center;
            font-size:1.5rem;font-weight:700;color:#0B1F3A;
            font-family:'Space Grotesk',sans-serif;flex-shrink:0;
        ">{selected_student[0]}</div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;color:#EAEAEA">{selected_student}</div>
            <div style="color:#8BA3C7;font-size:0.82rem;margin-top:3px">{dept} &nbsp;·&nbsp; Semester {semester}</div>
        </div>
        <div style="margin-left:auto;text-align:right">
            <div style="font-size:2rem;font-weight:700;font-family:'Space Grotesk',sans-serif;color:#00D4FF">{avg:.1f}</div>
            <div style="color:#8BA3C7;font-size:0.78rem">Overall Average</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:2rem;font-weight:700;font-family:'Space Grotesk',sans-serif;color:#00C9A7">{grade}</div>
            <div style="color:#8BA3C7;font-size:0.78rem">Grade</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ONLY this student's subjects ──
    # Sort for display
    subj_df = student_df[["Subject", "Marks"]].sort_values("Marks", ascending=False).reset_index(drop=True)
    subj_df.index = range(1, len(subj_df) + 1)
    subj_df["Grade"] = subj_df["Marks"].apply(grade_label)
    subj_df["Status"] = subj_df["Marks"].apply(lambda m: "Pass ✅" if m >= 50 else "Fail ❌")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Subject Table", "📊 Bar Chart", "🥧 Pie Chart"])

    with tab1:
        st.markdown(f'<div style="color:#8BA3C7;font-size:0.82rem;margin-bottom:0.5rem">{selected_student} — {len(subj_df)} subject(s) in Semester {semester}</div>', unsafe_allow_html=True)
        st.dataframe(subj_df, use_container_width=True)

    with tab2:
        fig_bar = go.Figure(go.Bar(
            x=subj_df["Subject"],
            y=subj_df["Marks"],
            marker=dict(
                color=subj_df["Marks"],
                colorscale=[[0, "#3A8DFF"], [0.5, "#00D4FF"], [1, "#00C9A7"]],
                showscale=False,
                line=dict(color="#0B1F3A", width=1),
            ),
            text=subj_df["Marks"],
            textposition="outside",
            textfont=dict(color="#CFCFCF", size=12),
            hovertemplate="<b>%{x}</b><br>Marks: %{y}<extra></extra>",
        ))
        # Pass line
        fig_bar.add_hline(y=50, line=dict(color="#F59E0B", dash="dot", width=1.5),
                          annotation_text="Pass Line (50)", annotation_font=dict(color="#F59E0B", size=11))
        fig_bar.update_layout(
            title=f"{selected_student} — Subject-wise Marks",
            xaxis_title="Subject",
            yaxis_title="Marks",
            yaxis_range=[0, 115],
            **PLOTLY_LAYOUT, height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        fig_pie = go.Figure(go.Pie(
            labels=subj_df["Subject"],
            values=subj_df["Marks"],
            hole=0.45,
            marker=dict(
                colors=PALETTE[:len(subj_df)],
                line=dict(color="#0B1F3A", width=2),
            ),
            textfont=dict(color="#EAEAEA", size=12),
            hovertemplate="<b>%{label}</b><br>Marks: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            title=f"{selected_student} — Marks Distribution",
            **PLOTLY_LAYOUT, height=400,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Strongest / weakest
    st.markdown("<br>", unsafe_allow_html=True)
    col_s, col_w = st.columns(2)
    strongest = subj_df.iloc[0]
    weakest = subj_df.iloc[-1]
    with col_s:
        st.markdown(f"""
        <div class="uni-card">
            <div class="uni-card-title">💪 Strongest Subject</div>
            <div style="font-size:1.1rem;font-weight:600;color:#EAEAEA;margin:6px 0">{strongest['Subject']}</div>
            <div class="uni-card-value">{strongest['Marks']}</div>
            <div class="uni-card-sub">Grade: {strongest['Grade']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_w:
        st.markdown(f"""
        <div class="uni-card">
            <div class="uni-card-title">⚠️ Weakest Subject</div>
            <div style="font-size:1.1rem;font-weight:600;color:#EAEAEA;margin:6px 0">{weakest['Subject']}</div>
            <div class="uni-card-value" style="color:{'#EF4444' if weakest['Marks'] < 50 else '#F59E0B'}">{weakest['Marks']}</div>
            <div class="uni-card-sub">Grade: {weakest['Grade']}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: COMPARISON
# ─────────────────────────────────────────────
def page_comparison():
    df = get_df()
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        return

    section_header("Comparison", "Compare students or departments side by side")

    mode = st.radio("Comparison Mode", ["Students", "Departments"], horizontal=True)

    if mode == "Students":
        all_students = sorted(df["Student_Name"].unique().tolist())
        selected = st.multiselect(
            "Select Students to Compare (2–5)",
            all_students,
            default=all_students[:2],
            max_selections=5,
        )

        if len(selected) < 2:
            st.info("Please select at least 2 students.")
            return

        # Build comparison — per-student averages and subject marks
        st.markdown("<br>", unsafe_allow_html=True)

        # Summary table
        summary_rows = []
        for name in selected:
            s_df = df[df["Student_Name"] == name]
            summary_rows.append({
                "Student": name,
                "Department": s_df["Department"].iloc[0],
                "Semester": s_df["Semester"].iloc[0],
                "Subjects": s_df["Subject"].nunique(),
                "Avg Marks": round(s_df["Marks"].mean(), 1),
                "Highest": s_df["Marks"].max(),
                "Lowest": s_df["Marks"].min(),
                "Grade": grade_label(s_df["Marks"].mean()),
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Radar / grouped bar
        with col1:
            fig = go.Figure()
            for i, name in enumerate(selected):
                s_df = df[df["Student_Name"] == name][["Subject", "Marks"]].sort_values("Subject")
                fig.add_trace(go.Bar(
                    name=name,
                    x=s_df["Subject"],
                    y=s_df["Marks"],
                    marker_color=PALETTE[i],
                ))
            fig.update_layout(
                title="Subject-wise Marks Comparison",
                barmode="group",
                xaxis_title="Subject", yaxis_title="Marks",
                **PLOTLY_LAYOUT, height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            for i, name in enumerate(selected):
                avg = df[df["Student_Name"] == name]["Marks"].mean()
                fig2.add_trace(go.Bar(
                    name=name,
                    x=[name],
                    y=[round(avg, 1)],
                    marker_color=PALETTE[i],
                    text=[round(avg, 1)],
                    textposition="outside",
                    textfont=dict(color="#CFCFCF"),
                ))
            fig2.update_layout(
                title="Overall Average Comparison",
                yaxis_range=[0, 110],
                **PLOTLY_LAYOUT, height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:  # Departments
        all_depts = sorted(df["Department"].unique().tolist())
        selected_depts = st.multiselect("Select Departments to Compare", all_depts, default=all_depts[:2])

        if len(selected_depts) < 2:
            st.info("Please select at least 2 departments.")
            return

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary
        rows = []
        for dept in selected_depts:
            d = df[df["Department"] == dept]
            rows.append({
                "Department": dept,
                "Students": d["Student_Name"].nunique(),
                "Subjects": d["Subject"].nunique(),
                "Avg Marks": round(d["Marks"].mean(), 1),
                "Pass Rate": f"{(d['Marks']>=50).mean()*100:.1f}%",
                "Top Subject": d.groupby("Subject")["Marks"].mean().idxmax(),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            for i, dept in enumerate(selected_depts):
                d = df[df["Department"] == dept]
                sem_avg = d.groupby("Semester")["Marks"].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=sem_avg["Semester"], y=sem_avg["Marks"],
                    mode="lines+markers",
                    name=dept,
                    line=dict(color=PALETTE[i], width=2.5),
                    marker=dict(size=8),
                ))
            fig.update_layout(title="Semester-wise Performance by Department",
                              xaxis_title="Semester", yaxis_title="Average Marks",
                              **PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            for i, dept in enumerate(selected_depts):
                d = df[df["Department"] == dept]
                fig2.add_trace(go.Box(
                    y=d["Marks"], name=dept,
                    marker_color=PALETTE[i],
                    line_color=PALETTE[i],
                    fillcolor=f"rgba({','.join(str(int(PALETTE[i].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.15)",
                ))
            fig2.update_layout(title="Marks Distribution by Department",
                               yaxis_title="Marks",
                               **PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: AI AGENT CHAT
# ─────────────────────────────────────────────
def page_chat():
    df = get_df()

    section_header("AI Agent Chat", "Ask natural language questions about your academic data")

    # Build context summary for the AI
    if df is not None:
        data_context = f"""
You are an AI analytics assistant for a university dataset.

Dataset Summary:
- Total records: {len(df)}
- Students: {df['Student_Name'].nunique()}
- Departments: {', '.join(df['Department'].unique())}
- Semesters: {sorted(df['Semester'].unique())}
- Subjects offered: {df['Subject'].nunique()} unique across departments
- Overall average marks: {df['Marks'].mean():.2f}
- Pass rate (>=50): {(df['Marks']>=50).mean()*100:.1f}%

Department averages:
{df.groupby('Department')['Marks'].mean().round(2).to_string()}

Top 5 students by average:
{df.groupby('Student_Name')['Marks'].mean().nlargest(5).round(2).to_string()}

Answer questions clearly and professionally. Use data from the context above.
If asked about a specific student, only refer to that student's data.
"""
    else:
        data_context = (
            "You are an AI analytics assistant for a university. "
            "No dataset is currently loaded. Advise the user to upload or load sample data first."
        )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    prompt = st.chat_input("Ask about student performance, department rankings, grades...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    client = anthropic.Anthropic()
                    messages_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ]
                    response = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=1024,
                        system=data_context,
                        messages=messages_payload,
                    )
                    reply = response.content[0].text
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e:
                    err = f"⚠️ AI Agent error: {str(e)}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})

    # Suggested prompts
    if not st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="color:#8BA3C7;font-size:0.82rem;margin-bottom:0.6rem">💡 Suggested Questions</div>', unsafe_allow_html=True)
        suggestions = [
            "Which department has the highest average marks?",
            "Who are the top 3 performing students?",
            "What is the overall pass rate?",
            "Which semester shows the lowest performance?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": s})
                    st.rerun()

    # Clear button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
def main():
    sidebar_nav()
    page = st.session_state.page

    if page == "Home":
        page_home()
    elif page == "Upload":
        page_upload()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Department":
        page_department()
    elif page == "Student":
        page_student_search()
    elif page == "Comparison":
        page_comparison()
    elif page == "Chat":
        page_chat()


if __name__ == "__main__":
    main()
