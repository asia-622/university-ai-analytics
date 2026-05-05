"""
app.py — University AI Analytics Agent  (clean rewrite)
"""
from __future__ import annotations
import io, json, os, re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="University AI Analytics", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{--bg:#0f172a;--surface:#1e293b;--surface2:#334155;--border:#475569;
      --text:#e2e8f0;--muted:#94a3b8;--accent:#38bdf8;--accent2:#818cf8;
      --success:#34d399;--warning:#fbbf24;--danger:#f87171;}
html,body,[data-testid="stApp"]{background:var(--bg)!important;color:var(--text)!important;
  font-family:'IBM Plex Sans',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-size:2rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.8rem!important;text-transform:uppercase;letter-spacing:.1em;}
h1{color:var(--accent)!important;font-weight:700!important;}
h2{color:var(--text)!important;font-weight:600!important;}
h3{color:var(--accent2)!important;font-weight:500!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  color:#0f172a!important;border:none!important;border-radius:8px!important;
  font-weight:600!important;padding:.5rem 1.5rem!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(56,189,248,.35)!important;}
.stTextInput>div>div>input,.stSelectbox>div>div,.stMultiSelect>div>div{
  background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;}
.chat-bubble-user{background:linear-gradient(135deg,var(--accent2),#6366f1);color:#fff;
  padding:.85rem 1.2rem;border-radius:18px 18px 4px 18px;margin:.4rem 0 .4rem 15%;font-size:.9rem;}
.chat-bubble-ai{background:var(--surface);border:1px solid var(--border);color:var(--text);
  padding:.85rem 1.2rem;border-radius:18px 18px 18px 4px;margin:.4rem 15% .4rem 0;font-size:.9rem;}
.chat-label{font-size:.7rem;color:var(--muted);margin-bottom:.2rem;text-transform:uppercase;letter-spacing:.1em;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from file_handler       import load_file
from data_preprocessing import preprocess, get_student_row
from rag_engine         import RAGEngine, build_chunks
from chatbot            import UniversityAgent
from model              import train_model, predict_batch
import dashboard as dash
from tools import (get_dataset_summary, get_department_stats,
                   get_top_students, get_attendance_analysis, get_subject_analysis)

# ── API key ───────────────────────────────────────────────────────────────────
def _api_key():
    try:    return st.secrets["GROQ_API_KEY"]
    except: return os.environ.get("GROQ_API_KEY","")

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in {"meta":None,"agent":None,"rag":None,"ml_model":None,
            "chat_history":[],"api_key":_api_key(),"rag_built":False}.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 University Agent")
    st.markdown("---")
    page = st.radio("Navigate",
        ["🏠 Home","📂 Upload & Analyze","📊 Dashboard",
         "🏛️ Dept Analysis","🔍 Student Search","⚖️ Comparison","🤖 AI Agent Chat"],
        label_visibility="collapsed")
    if st.session_state["meta"]:
        m = st.session_state["meta"]
        st.markdown("---\n### 📋 Dataset Info")
        st.markdown(f"- **Students:** {m['n_students']:,}")
        st.markdown(f"- **Departments:** {m['n_departments']}")
        st.markdown(f"- **Subjects:** {len(m['subject_cols'])}")
        if st.session_state["rag_built"]: st.markdown("- ✅ RAG ready")
        if st.session_state["ml_model"]:
            st.markdown(f"- ✅ ML R²={st.session_state['ml_model']['metrics']['r2']}")
    st.markdown("---")
    st.caption("University AI Analytics v2.0")

def _need_data():
    if st.session_state["meta"] is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

# ── Chart layout helper ───────────────────────────────────────────────────────
def _layout(**kw):
    base = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans",color="#e2e8f0",size=13),
                margin=dict(l=10,r=80,t=55,b=20))
    base.update(kw); return base

def _bar_color(v,mx):
    p=v/mx if mx else 0
    return "#34d399" if p>=.80 else "#fbbf24" if p>=.65 else "#fb923c" if p>=.50 else "#f87171"

def _grade_str(avg):
    if avg>=90: return "A+","#22d3ee"
    if avg>=80: return "A","#34d399"
    if avg>=70: return "B","#a3e635"
    if avg>=60: return "C","#fbbf24"
    if avg>=50: return "D","#fb923c"
    return "F","#f87171"

def _horiz_bar(subjects, values, title, height=None):
    df_b = pd.DataFrame({"Subject":subjects,"Marks":values}).sort_values("Marks",ascending=True)
    mx   = max(values) or 100
    cols = [_bar_color(v,mx) for v in df_b["Marks"]]
    fig  = go.Figure(go.Bar(
        x=df_b["Marks"], y=df_b["Subject"], orientation="h",
        marker_color=cols, marker_line_width=0,
        text=[f"  {v:.1f}" for v in df_b["Marks"]],
        textposition="outside", textfont=dict(color="#e2e8f0",size=13)))
    fig.update_layout(title=title,
        height=height or max(350,len(subjects)*38+80),
        xaxis=dict(color="#94a3b8",range=[0,mx*1.22],showgrid=True,
                   gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(color="#e2e8f0",automargin=True,showgrid=False,tickfont=dict(size=13)),
        showlegend=False,**_layout())
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎓 University AI Analytics Agent")
    st.markdown("### Intelligent academic data analysis — LLM + RAG + Tools")
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.markdown("**📂 Upload**\n\nCSV / Excel / JSON up to 200MB")
    c2.markdown("**📊 Analyze**\n\nAuto-detect departments, subjects, attendance")
    c3.markdown("**🤖 Ask AI**\n\nGroq LLaMA + RAG for natural language Q&A")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Upload & Analyze":
    st.markdown("# 📂 Upload & Analyze")
    uploaded = st.file_uploader("Upload your university dataset",
                                type=["csv","xlsx","xls","json"])
    if uploaded:
        with st.spinner("Loading & preprocessing…"):
            raw_df = load_file(uploaded)
            if raw_df is None: st.stop()
            meta   = preprocess(raw_df)
            chunks = build_chunks(meta)
            rag    = RAGEngine(api_key=st.session_state["api_key"])
            rag.build(chunks)
            ml     = train_model(meta)
            agent  = UniversityAgent(api_key=st.session_state["api_key"])
            agent.attach_data(meta, rag, ml)
            st.session_state.update({"meta":meta,"rag":rag,"ml_model":ml,
                                      "agent":agent,"rag_built":True,"chat_history":[]})

        st.success(f"✅ {meta['n_students']:,} students loaded · {len(meta['subject_cols'])} subjects detected")
        if meta.get("is_long_format"):
            st.info(f"🔄 Long format detected — auto-pivoted · Subject col: `{meta['subject_col_long']}` · Marks col: `{meta['marks_col_long']}`")

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Students",   f"{meta['n_students']:,}")
        k2.metric("Columns",    len(meta["df"].columns))
        k3.metric("Departments",meta["n_departments"])
        k4.metric("Subjects",   len(meta["subject_cols"]))

        with st.expander("📋 Detected Column Mapping"):
            for label,key in [("Name","name_col"),("Department","dept_col"),
                               ("Attendance","attend_col"),("Roll No","roll_col"),
                               ("Year/Sem","year_col")]:
                v   = meta.get(key,"—") or "—"
                clr = "#34d399" if v!="—" else "#f87171"
                st.markdown(f"**{label}:** <span style='color:{clr}'>{v}</span>",
                            unsafe_allow_html=True)

        st.markdown("### 👀 Data Preview")
        st.dataframe(meta["df"].head(20), use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download Cleaned CSV",
                               meta["df"].to_csv(index=False).encode(),
                               "cleaned.csv","text/csv")
        with c2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as w:
                meta["df"].to_excel(w,index=False,sheet_name="Cleaned")
            st.download_button("📥 Download Excel", buf.getvalue(),
                               "cleaned.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif st.session_state["meta"]:
        st.info("✅ Dataset already loaded. Upload a new file to replace it.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    _need_data()
    meta    = st.session_state["meta"]
    summary = get_dataset_summary(meta)
    st.markdown("# 📊 Analytics Dashboard")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("👥 Students",     f"{summary['total_students']:,}")
    k2.metric("🏛️ Departments",  summary["n_departments"])
    k3.metric("📚 Subjects",     len(summary["subject_columns"]))
    if "class_average"    in summary: k4.metric("📈 Class Avg",    f"{summary['class_average']:.1f}")
    if "average_attendance" in summary: k5.metric("✅ Attendance", f"{summary['average_attendance']:.1f}%")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(dash.department_pie(meta),       use_container_width=True)
    with c2: st.plotly_chart(dash.attendance_histogram(meta), use_container_width=True)
    c3,c4 = st.columns(2)
    with c3: st.plotly_chart(dash.dept_marks_bar(meta),       use_container_width=True)
    with c4: st.plotly_chart(dash.grade_distribution(meta),   use_container_width=True)

    if meta.get("dept_col"):
        st.markdown("### 🏛️ Department Summary")
        ds = get_department_stats(meta)
        if "departments" in ds:
            rows=[{"Department":d,"Students":i["count"],
                   **({"Avg Marks":i["avg_marks"]} if "avg_marks" in i else {}),
                   **({"Avg Attendance":i["avg_attendance"]} if "avg_attendance" in i else {})}
                  for d,i in ds["departments"].items()]
            st.dataframe(pd.DataFrame(rows).sort_values("Students",ascending=False),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEPT ANALYSIS  (was Subject Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏛️ Dept Analysis":
    _need_data()
    meta     = st.session_state["meta"]
    df       = meta["df"]
    scols    = meta["subject_cols"]
    dept_col = meta.get("dept_col")
    dmap     = meta.get("dept_subject_map",{})

    st.markdown("# 🏛️ Department-wise Subject Analysis")

    if not scols:
        st.error("❌ No subject columns detected."); st.stop()
    if not dept_col:
        st.error("❌ No department column detected."); st.stop()

    dept_list = sorted(df[dept_col].dropna().unique().tolist())

    # ── Department buttons ────────────────────────────────────────────────────
    st.markdown("### Click a Department:")
    if "sel_dept" not in st.session_state or st.session_state["sel_dept"] not in dept_list:
        st.session_state["sel_dept"] = dept_list[0]

    btn_cols = st.columns(len(dept_list))
    for i,dept in enumerate(dept_list):
        active = st.session_state["sel_dept"]==dept
        if btn_cols[i].button(dept, key=f"db_{i}",
                              type="primary" if active else "secondary",
                              use_container_width=True):
            st.session_state["sel_dept"]=dept

    dept      = st.session_state["sel_dept"]
    dept_df   = df[df[dept_col]==dept].reset_index(drop=True)

    # Get ONLY this dept's subjects from long-format mapping
    df_long       = meta.get("df_long")
    subj_col_long = meta.get("subject_col_long")

    if df_long is not None and subj_col_long and dept_col:
        try:
            # Get subjects actually taught in this department from original data
            dept_long = df_long[df_long[dept_col].astype(str)==dept]
            actual    = dept_long[subj_col_long].dropna().unique().tolist()
            dscols    = [s for s in actual if s in scols]
        except Exception:
            dscols = []
    elif dmap and dept in dmap:
        dscols = [s for s in dmap[dept] if s in scols]
    else:
        dscols = [s for s in scols if dept_df[s].notna().sum()>0 and dept_df[s].mean()>0]

    st.markdown(f"---\n### 📊 **{dept}** — {len(dept_df)} students · {len(dscols)} subjects")

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Students", len(dept_df))
    if "Average" in dept_df.columns:
        k2.metric("Avg Marks", f"{dept_df['Average'].mean():.1f}")
    if meta.get("attend_col") and meta["attend_col"] in dept_df.columns:
        k3.metric("Avg Attendance", f"{dept_df[meta['attend_col']].mean():.1f}%")
    if "Grade" in dept_df.columns:
        k4.metric("Top Grade", dept_df["Grade"].value_counts().index[0])

    if dscols:
        avgs = dept_df[dscols].mean()
        st.plotly_chart(_horiz_bar(avgs.index.tolist(), avgs.values.tolist(),
                                   f"{dept} — Average Marks per Subject"),
                        use_container_width=True)

        with st.expander("📋 Subject Stats Table"):
            tbl=dept_df[dscols].agg(["mean","max","min","std"]).T.round(1)
            tbl.columns=["Average","Max","Min","Std Dev"]
            st.dataframe(tbl.sort_values("Average",ascending=False),use_container_width=True)

        with st.expander(f"👥 All {len(dept_df)} students in {dept}"):
            scols_show=[c for c in [meta.get("name_col"),meta.get("roll_col"),
                        meta.get("year_col"),"Average","Grade"] if c and c in dept_df.columns]
            st.dataframe(
                dept_df[scols_show].sort_values("Average",ascending=False)
                if "Average" in dept_df.columns else dept_df[scols_show],
                use_container_width=True, height=350)
            st.download_button(f"📥 Download {dept}",
                               dept_df.to_csv(index=False).encode(),
                               f"{dept}.csv","text/csv")
    else:
        st.warning("No subjects found for this department.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDENT SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Student Search":
    _need_data()
    meta     = st.session_state["meta"]
    df       = meta["df"]
    scols    = meta["subject_cols"]
    name_col = meta.get("name_col")
    dept_col = meta.get("dept_col")
    dmap     = meta.get("dept_subject_map",{})

    st.markdown("# 🔍 Student Search & Profile")

    if not name_col:
        st.error("❌ No student name column detected."); st.stop()

    query = st.text_input("🔎 Enter student name",
                          placeholder="e.g. Ayesha, Ali, Sara…")
    if not query:
        st.info("Type a student name above to see their subject-wise performance.")
        st.stop()

    rows = get_student_row(meta, query)
    if rows.empty:
        st.warning(f"No student found matching **'{query}'**"); st.stop()

    st.success(f"✅ Found **{len(rows)}** student(s) matching '{query}'")

    # ── Show ALL matched students ─────────────────────────────────────────────
    for idx in range(len(rows)):
        row   = rows.iloc[idx]
        sname = str(row.get(name_col,"?"))
        avg   = float(row["Average"]) if "Average" in row.index else 0.0
        g,gc  = _grade_str(avg)

        # ── Profile card ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:14px;
                    padding:1.2rem 1.5rem;margin:1rem 0 0.5rem 0'>
          <span style='font-size:1.15rem;font-weight:700;color:#e2e8f0'>👤 {sname}</span>
          &nbsp;&nbsp;<span style='background:{gc}22;color:{gc};border:1px solid {gc}55;
          border-radius:20px;padding:0.15rem 0.7rem;font-size:0.85rem;font-weight:700'>{g}</span>
        </div>""", unsafe_allow_html=True)

        k1,k2,k3,k4 = st.columns(4)
        if dept_col and dept_col in row.index:
            k1.metric("Department", str(row[dept_col]))
        if meta.get("roll_col") and meta["roll_col"] in row.index:
            k2.metric("Student ID", str(row[meta["roll_col"]]))
        if meta.get("year_col") and meta["year_col"] in row.index:
            k3.metric("Semester", str(row[meta["year_col"]]))
        k4.metric("Average", f"{avg:.1f}")

        # ── Get this student's actual subjects from long-format data ──────────
        df_long        = meta.get("df_long")
        subj_col_long  = meta.get("subject_col_long")
        marks_col_long = meta.get("marks_col_long")
        roll_col       = meta.get("roll_col")
        student_id     = row.get(roll_col) if roll_col and roll_col in row.index else None

        stu_scols = []

        # Method 1: original long data — most accurate
        if df_long is not None and subj_col_long and marks_col_long and student_id is not None:
            try:
                id_col_long = None
                for c in df_long.columns:
                    if roll_col and (c == roll_col or roll_col.lower() in c.lower()):
                        id_col_long = c; break
                if id_col_long:
                    stu_long = df_long[
                        df_long[id_col_long].astype(str) == str(int(float(str(student_id))))
                    ]
                    actual   = stu_long[subj_col_long].dropna().unique().tolist()
                    stu_scols = [s for s in actual if s in scols]
            except Exception:
                stu_scols = []

        # Method 2: fallback — non-NaN non-zero values
        if not stu_scols:
            stu_scols = [
                s for s in scols
                if s in row.index and pd.notna(row[s]) and float(row[s]) > 0
            ]

        if not stu_scols:
            st.info("No subject marks found."); st.markdown("---"); continue

        marks = [float(row[s]) for s in stu_scols]

        # ── Bar chart ─────────────────────────────────────────────────────────
        st.plotly_chart(
            _horiz_bar(stu_scols, marks,
                       f"📊 {sname} (ID:{student_id}) — {len(stu_scols)} Subjects"),
            use_container_width=True)

        # ── Pie chart ─────────────────────────────────────────────────────────
        fig_pie = go.Figure(go.Pie(
            labels=stu_scols, values=marks, hole=0.38,
            textinfo="label+value",
            textfont=dict(size=12, color="#e2e8f0"),
            marker=dict(
                colors=px.colors.qualitative.Bold[:len(stu_scols)],
                line=dict(color="#0f172a", width=2)),
            hovertemplate="<b>%{label}</b><br>Marks: %{value:.0f}<extra></extra>",
        ))
        fig_pie.update_layout(
            title=f"🥧 {sname} — Marks Distribution",
            height=400,
            legend=dict(font=dict(color="#cbd5e1", size=11),
                        bgcolor="rgba(0,0,0,0)", orientation="v"),
            **_layout(margin=dict(l=10, r=10, t=50, b=20)))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.download_button(
            f"📥 Download {sname} (ID:{student_id}) Profile",
            pd.DataFrame([row]).to_csv(index=False).encode(),
            f"{sname}_{student_id}_profile.csv", "text/csv",
            key=f"dl_{idx}")

        st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Comparison":
    _need_data()
    meta     = st.session_state["meta"]
    name_col = meta.get("name_col")
    scols    = meta["subject_cols"]
    st.markdown("# ⚖️ Student Comparison")

    if not name_col: st.error("❌ No name column."); st.stop()
    if not scols:    st.error("❌ No subject columns."); st.stop()

    all_names=meta["df"][name_col].dropna().astype(str).unique().tolist()
    selected=st.multiselect("Select 2–5 students",all_names,max_selections=5)
    if len(selected)<2:
        st.info("Please select at least 2 students."); st.stop()

    rows=[]
    for n in selected:
        m=get_student_row(meta,n)
        if not m.empty:
            r=m.iloc[0].copy(); r["__name__"]=n; rows.append(r)
    if len(rows)<2:
        st.warning("Could not find enough student data."); st.stop()

    comp=pd.DataFrame(rows).reset_index(drop=True)
    st.plotly_chart(dash.comparison_bar(comp,scols),  use_container_width=True)
    if len(scols)>=3:
        st.plotly_chart(dash.comparison_radar(comp,scols),use_container_width=True)
    tbl=comp[["__name__"]+[c for c in scols+["Average"] if c in comp.columns]].rename(columns={"__name__":"Student"})
    st.dataframe(tbl,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI AGENT CHAT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent Chat":
    st.markdown("# 🤖 AI Agent Chat")
    if st.session_state["meta"] is None:
        st.warning("⚠️ Please upload a dataset first."); st.stop()

    if st.session_state.get("agent"):
        agent=st.session_state["agent"]
        if st.session_state.get("api_key") and agent.client is None:
            try:
                from groq import Groq
                agent.client=Groq(api_key=st.session_state["api_key"])
            except: pass
    else:
        agent=UniversityAgent(api_key=st.session_state.get("api_key"))
        agent.attach_data(st.session_state["meta"],
                          st.session_state.get("rag"),
                          st.session_state.get("ml_model"))
        st.session_state["agent"]=agent

    has_key=bool(st.session_state.get("api_key"))
    clr="#34d399" if has_key else "#fbbf24"
    txt="🟢 Groq LLaMA Active" if has_key else "🟡 No API key — rule-based mode"
    st.markdown(f"<div style='background:rgba(30,41,59,.8);border:1px solid #334155;"
                f"border-radius:8px;padding:.6rem 1rem;color:{clr};font-size:.85rem'>{txt}</div>",
                unsafe_allow_html=True)

    for role,content in st.session_state["chat_history"]:
        if role=="user":
            st.markdown(f"<div class='chat-label'>You</div>"
                        f"<div class='chat-bubble-user'>{content}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-label'>🤖 Agent</div>"
                        f"<div class='chat-bubble-ai'>{content}</div>",
                        unsafe_allow_html=True)

    suggs=["Summarise the dataset","Which dept has highest marks?",
           "Students with low attendance","Top 5 students","Analyse subjects"]
    sc=st.columns(len(suggs))
    for i,s in enumerate(suggs):
        if sc[i].button(s,key=f"s{i}"):
            st.session_state["_pm"]=s; st.rerun()

    with st.form("cf",clear_on_submit=True):
        ui=st.text_area("Message",placeholder="Ask about students, marks, attendance…",
                        height=80,label_visibility="collapsed",
                        value=st.session_state.pop("_pm",""))
        cs,cc=st.columns([3,1])
        send=cs.form_submit_button("📨 Send",use_container_width=True)
        clear=cc.form_submit_button("🗑 Clear",use_container_width=True)

    if clear:
        st.session_state["chat_history"]=[]; agent.reset(); st.rerun()
    if send and ui.strip():
        st.session_state["chat_history"].append(("user",ui.strip()))
        with st.spinner("🤖 Thinking…"):
            reply=agent.chat(ui.strip())
        st.session_state["chat_history"].append(("assistant",reply))
        st.rerun()

    if st.session_state["chat_history"]:
        txt="\n\n".join(f"{'User' if r=='user' else 'Agent'}: {c}"
                        for r,c in st.session_state["chat_history"])
        st.download_button("📥 Export Chat",txt.encode(),"chat.txt","text/plain")
