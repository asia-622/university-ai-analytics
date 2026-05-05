"""
chatbot.py — Core AI Agent loop.
"""
from __future__ import annotations

import json
import logging

from memory import ConversationMemory
from rag_engine import RAGEngine
from tools import TOOL_SCHEMAS, call_tool
from utils import get_groq_key, truncate

logger = logging.getLogger("university_agent.chatbot")

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False


SYSTEM_PROMPT = """You are a University AI Analytics Agent — a smart, friendly academic data analyst.

LANGUAGE: Always respond in clear ENGLISH only. Never use Roman Urdu or any other language.

RESPONSE FORMAT RULES (very important):
- Always give clean, readable, well-structured answers.
- Use bullet points or numbered lists for multiple items.
- Use bold for important values: **72.5%**, **Computer Science**.
- Never dump raw data or long pipe-separated strings.
- Keep responses concise — max 3-4 sentences per section.
- Round all numbers to 1-2 decimal places.
- Always add a brief insight or conclusion after data.

CAPABILITIES:
1. TOOLS — call tools to get accurate data from the dataset.
2. RAG — use retrieved context only as background reference, NOT as direct output.
3. MEMORY — you remember the conversation.

GUIDELINES:
- Use tools for all data questions. Do NOT output raw tool results directly.
- Summarise RAG context into clean sentences — never paste it as-is.
- If asked about a department: give student count, avg marks, top subject.
- If asked about a student: give name, dept, avg, grade, top/weak subject.
- If asked about attendance: give average %, low-attendance count, threshold.
- Always end with a 1-line insight like "📌 CS department leads in OOP performance."
"""


class UniversityAgent:
    def __init__(self, api_key: str | None = None):
        key = api_key or get_groq_key()
        self.client = Groq(api_key=key) if (key and GROQ_OK) else None
        self.memory  = ConversationMemory(max_messages=20)
        self.rag: RAGEngine | None = None
        self.meta: dict | None = None
        self.model = None
        self._setup_memory()

    def _setup_memory(self) -> None:
        self.memory.set_system(SYSTEM_PROMPT)

    def attach_data(self, meta: dict, rag: RAGEngine, ml_model=None) -> None:
        self.meta  = meta
        self.rag   = rag
        self.model = ml_model

    def chat(self, user_message: str) -> str:
        if not self.client:
            return self._no_llm_response(user_message)

        # Build clean RAG context — summarised, not raw
        rag_note = ""
        if self.rag and self.rag.is_ready:
            raw_ctx = self.rag.format_context(user_message, top_k=4)
            if raw_ctx and raw_ctx != "No relevant data found.":
                # Strip pipe-heavy lines, keep short ones
                lines = [l.strip() for l in raw_ctx.split("\n") if l.strip()]
                clean_lines = []
                for l in lines:
                    # Skip lines with >8 pipe chars (raw dept dumps)
                    if l.count("|") > 8:
                        continue
                    clean_lines.append(l)
                if clean_lines:
                    rag_note = "\n[Background context from dataset — use as reference only]:\n"
                    rag_note += "\n".join(clean_lines[:6])  # max 6 clean lines

        augmented = user_message + rag_note
        self.memory.add_user(augmented)

        try:
            response = self._agent_loop(augmented)
        except Exception as exc:
            logger.exception("Agent loop error: %s", exc)
            response = f"⚠️ Something went wrong: {exc}"

        self.memory.add_assistant(response)
        return response

    def _agent_loop(self, user_msg: str, max_rounds: int = 5) -> str:
        messages = self.memory.get_messages()

        for round_n in range(max_rounds):
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOL_SCHEMAS if self.meta else None,
                tool_choice="auto" if self.meta else None,
                temperature=0.2,
                max_tokens=1000,
            )
            choice = resp.choices[0]

            if choice.finish_reason != "tool_calls":
                return choice.message.content or ""

            tool_calls = choice.message.tool_calls
            messages.append(choice.message)

            for tc in tool_calls:
                fn_name = tc.function.name
                args    = json.loads(tc.function.arguments or "{}")
                logger.info("Tool call: %s(%s)", fn_name, args)
                result  = call_tool(fn_name, args, self.meta, self.model)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        # Final forced answer
        messages.append({
            "role": "user",
            "content": "Based on the tool results above, give a clean, well-formatted answer in English."
        })
        final = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        return final.choices[0].message.content or ""

    def _no_llm_response(self, query: str) -> str:
        """Clean formatted response without LLM."""
        if self.meta is None:
            return "⚠️ Please upload a dataset first."

        q = query.lower()

        try:
            if any(w in q for w in ["summary", "overview", "about", "tell me"]):
                r = json.loads(call_tool("get_dataset_summary", {}, self.meta))
                return (
                    f"📊 **Dataset Overview**\n\n"
                    f"- **Total Students:** {r.get('total_students', '?'):,}\n"
                    f"- **Departments:** {r.get('n_departments', '?')} — {', '.join(r.get('departments', []))}\n"
                    f"- **Subjects detected:** {len(r.get('subject_columns', []))}\n"
                    f"- **Class Average:** {r.get('class_average', r.get('average_marks', '?'))}\n"
                    f"- **Avg Attendance:** {r.get('average_attendance', 'N/A')}\n\n"
                    f"*(Add a Groq API key for AI-powered Q&A)*"
                )

            elif any(w in q for w in ["total", "how many", "count", "students"]):
                r = json.loads(call_tool("get_total_students", {}, self.meta))
                return f"👥 Total students in the dataset: **{r.get('total_students', '?'):,}**"

            elif any(w in q for w in ["department", "dept", "branch"]):
                r = json.loads(call_tool("get_department_stats", {}, self.meta))
                if "departments" in r:
                    lines = ["🏛️ **Department Statistics**\n"]
                    for dept, info in r["departments"].items():
                        line = f"- **{dept}**: {info['count']} students"
                        if "avg_marks" in info:
                            line += f" | Avg Marks: **{info['avg_marks']}**"
                        if "avg_attendance" in info:
                            line += f" | Attendance: **{info['avg_attendance']}%**"
                        lines.append(line)
                    return "\n".join(lines)

            elif any(w in q for w in ["top", "best", "highest", "topper", "rank"]):
                r = json.loads(call_tool("get_top_students", {}, self.meta))
                if "top_students" in r:
                    lines = ["🏆 **Top Students**\n"]
                    for i, s in enumerate(r["top_students"], 1):
                        line = f"{i}. **{s.get('name', '?')}**"
                        for k, v in s.items():
                            if k != "name":
                                line += f" | {k}: **{v}**"
                        lines.append(line)
                    return "\n".join(lines)

            elif any(w in q for w in ["attendance", "absent", "present"]):
                r = json.loads(call_tool("get_attendance_analysis", {}, self.meta))
                dist = r.get("distribution", {})
                return (
                    f"✅ **Attendance Analysis**\n\n"
                    f"- **Average Attendance:** {r.get('average_attendance', '?')}%\n"
                    f"- **Students below {r.get('threshold_used', 75)}%:** {r.get('students_below_threshold', '?')}\n"
                    f"- 🟢 Excellent (≥90%): {dist.get('excellent (>=90%)', 0)} students\n"
                    f"- 🟡 Good (75-90%): {dist.get('good (75-90%)', 0)} students\n"
                    f"- 🔴 Low (<75%): {dist.get('low (<75%)', 0)} students"
                )

            elif any(w in q for w in ["subject", "marks", "score", "performance"]):
                r = json.loads(call_tool("get_subject_analysis", {}, self.meta))
                if "subjects" in r:
                    sorted_subj = sorted(r["subjects"].items(),
                                         key=lambda x: x[1]["average"], reverse=True)
                    lines = ["📚 **Subject Performance**\n"]
                    for subj, info in sorted_subj[:10]:
                        lines.append(f"- **{subj}**: Avg **{info['average']}** | Max: {info['max']} | Min: {info['min']}")
                    if len(sorted_subj) > 10:
                        lines.append(f"_...and {len(sorted_subj)-10} more subjects_")
                    return "\n".join(lines)

        except Exception as e:
            logger.error("No-LLM response error: %s", e)

        return (
            "🔑 **Groq API key not set.**\n\n"
            "Add your Groq API key in Streamlit secrets to enable AI-powered responses.\n"
            "Without it, I can answer basic questions about: summary, departments, top students, attendance, subjects."
        )

    def reset(self) -> None:
        self.memory.clear()
        self._setup_memory()

    @property
    def history(self) -> list[dict]:
        return [m for m in self.memory.get_messages() if m["role"] != "system"]
