"""
app.py — Streamlit conversational UI for FinAgent CIO.
Launch: double-click launch.bat  or  streamlit run app.py
"""

import os
import sys
import threading
import queue
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Community Cloud there is no .env file — secrets live in the
# dashboard (st.secrets). Copy them into os.environ so every tool that calls
# os.getenv() works identically in both local and cloud environments.
try:
    for _k, _v in st.secrets.items():
        if _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:
    pass  # Not on Streamlit Cloud, or secrets not yet configured — that's fine


# ── Live stdout capture ────────────────────────────────────────────────────────
class _StdoutCapture:
    """
    Context-manager that tees every stdout line into a Queue,
    while still printing to the original terminal.
    """
    def __init__(self, q: queue.Queue):
        self._q = q
        self._orig = sys.stdout
        self._buf = ""

    def write(self, text: str):
        self._orig.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._q.put(stripped)

    def flush(self):
        self._orig.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig


# ── Thinking-display helpers ────────────────────────────────────────────────────
# Maps agent name → (icon, step label shown in the live panel)
_STEP_MAP: dict[str, tuple[str, str]] = {
    "QueryAnalyst":  ("🔍", "Step 1 — Understanding your query"),
    "MacroAgent":    ("🌐", "Step 2 — Macro & Sector Context"),
    "CompanyAgent":  ("🏢", "Step 3 — Company Fundamentals"),
    "NewsAgent":     ("📰", "Step 4 — News & Sentiment"),
    "ReportManager": ("📄", "Generating Report"),
    "CIO":           ("🧠", "Synthesizing CIO Recommendation"),
}

# Maps tool function name (substring) → friendly description
_TOOL_MAP: dict[str, str] = {
    "understand_query":      "Classifying query intent",
    "web_search":            "Searching the web",
    "finance_search":        "Searching financial databases",
    "news_search":           "Fetching latest news",
    "get_financial_summary": "Pulling price & valuation metrics",
    "get_income_statement":  "Loading 3-year income statement",
    "get_cash_flow":         "Loading cash flow & FCF",
    "get_key_metrics":       "Computing key financial metrics",
    "get_balance_sheet":     "Loading balance sheet",
    "get_analyst_ratings":   "Fetching analyst consensus",
    "get_earnings_surprise": "Checking EPS beat/miss history",
    "get_company_news":      "Fetching company news",
    "get_news_sentiment":    "Running AI sentiment scoring",
    "get_company_overview":  "Loading company overview",
    "get_earnings_history":  "Loading EPS history",
    "get_macro_indicator":   "Fetching macro economic data",
    "get_stock_news":        "Searching stock news",
    "get_market_sentiment":  "Gauging market sentiment",
    "save_full_report":      "Generating investment report",
    "save_to_excel":         "Saving Excel report",
    "save_to_pdf":           "Saving PDF report",
}

# Lines containing any of these are always dropped — no exceptions
_SKIP_ALWAYS: list[str] = [
    "WARNING", "ERROR", "CRITICAL", "DEBUG",
    "UserWarning", "DeprecationWarning", "FutureWarning",
    "site-packages", "warnings.warn", "stacklevel",
    "chrome_", "impersonate", "curl_cffi",
    "use_container_width", "will be removed",
    "pip install", "subprocess", "NotImplemented",
    "NativeCommandError", 'Traceback', '  File "',
    "raise_for_status", "requests.exceptions",
    "HTTPError", "ConnectionError", "ReadTimeout",
    "403 Client", "404 Not", "500 Internal",
    "token=", "api_key=", "Authorization",
]


def _parse_log(raw: str, seen_agents: set) -> str | None:
    """
    Whitelist-only log parser — only structured step/tool lines are emitted.
    All framework internals, warnings, JSON blobs, and API noise are dropped.

    Args:
        raw:         Raw stdout line from the agent thread.
        seen_agents: Mutable set tracking which phase headers have been shown
                     in the current query; prevents duplicate step banners.
    """
    stripped = raw.strip()
    if len(stripped) < 4:
        return None

    # Hard-skip: errors, warnings, framework internals, security-sensitive tokens
    for fragment in _SKIP_ALWAYS:
        if fragment in raw:
            return None

    # Skip pure box-drawing / separator lines (Agno's pretty-print borders)
    if all(c in "─│┌┐└┘├┤┬┴┼╔╗╚╝═╠╣╦╩╬━ \t" for c in stripped):
        return None

    # ── WHITELIST 1: Agent phase transitions ─────────────────────────────────
    for name, (icon, label) in _STEP_MAP.items():
        if name in stripped and name not in seen_agents:
            seen_agents.add(name)
            return f"\n{icon} **{label}**"

    # ── WHITELIST 2: Tool invocations ────────────────────────────────────────
    for kw, friendly in _TOOL_MAP.items():
        if kw in stripped:
            # Try to surface the ticker / query argument for context
            arg_hint = ""
            if "(" in stripped and ")" in stripped:
                try:
                    s = stripped.index("(") + 1
                    e = stripped.index(")", s)
                    cand = stripped[s:e].strip().strip("'\"")
                    if cand and len(cand) <= 30 and " " not in cand:
                        arg_hint = f"  `{cand}`"
                except ValueError:
                    pass
            return f"  ↳ {friendly}{arg_hint}"

    # Drop everything else — raw API responses, JSON, model output fragments
    return None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAgent CIO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: nuclear black + white — no exceptions ────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   GLOBAL: force every element black bg / white text
═══════════════════════════════════════════════ */
:root {
    --background-color: #000000;
    --secondary-background-color: #0d0d0d;
    --text-color: #ffffff;
}

*, *::before, *::after {
    color: #ffffff !important;
    box-sizing: border-box;
}

html, body {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Streamlit wrappers */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"],
.main, .block-container,
.element-container,
[data-testid="column"] {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #222 !important;
}

/* ═══════════════════════════════════════════════
   TEXT
═══════════════════════════════════════════════ */
p, span, div, li, td, th, label, a, pre, code,
h1, h2, h3, h4, h5, h6, small, strong, em, b, i,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"], [data-testid="stCaptionContainer"],
.stMarkdown, .stMarkdown * {
    color: #ffffff !important;
    background-color: transparent !important;
}

h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.4rem !important; color: #60a5fa !important; }
h3 { font-size: 1.15rem !important; color: #34d399 !important; }
h4 { color: #a78bfa !important; }

/* Inline code */
code {
    background-color: #1a1a1a !important;
    color: #86efac !important;
    padding: 2px 5px;
    border-radius: 4px;
}
pre { background-color: #111 !important; padding: 12px; border-radius: 8px; }
pre code { background-color: transparent !important; }

/* Links */
a { color: #60a5fa !important; text-decoration: none; }
a:hover { text-decoration: underline; }

/* ═══════════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════════ */
input, textarea, select,
.stTextInput input,
.stTextArea textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: #111 !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}
input::placeholder, textarea::placeholder { color: #555 !important; }
input:focus, textarea:focus { border-color: #3b82f6 !important; outline: none !important; }

/* ═══════════════════════════════════════════════
   SELECT / DROPDOWN
═══════════════════════════════════════════════ */
[data-baseweb="select"] > div,
[data-baseweb="select"] * {
    background-color: #111 !important;
    color: #ffffff !important;
    border-color: #333 !important;
}
[data-baseweb="menu"],
[data-baseweb="menu"] * {
    background-color: #111 !important;
    color: #ffffff !important;
}
[data-baseweb="option"]:hover {
    background-color: #1d4ed8 !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] * {
    background-color: #111 !important;
}

/* ═══════════════════════════════════════════════
   CHAT INPUT
═══════════════════════════════════════════════ */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background-color: #111 !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] button {
    background-color: #1d4ed8 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
[data-testid="stChatInput"] button:hover {
    background-color: #2563eb !important;
}

/* ═══════════════════════════════════════════════
   CHAT MESSAGES
═══════════════════════════════════════════════ */
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] > div {
    background-color: #0d0d0d !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 12px !important;
}
[data-testid="stChatMessage"] *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] span {
    color: #ffffff !important;
    background-color: transparent !important;
}
[data-testid="stChatMessage"] h2 { color: #60a5fa !important; }
[data-testid="stChatMessage"] h3 { color: #34d399 !important; }
[data-testid="stChatMessage"] code {
    background-color: #1a1a1a !important;
    color: #86efac !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.stButton > button {
    background-color: #111 !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: #3b82f6 !important;
}

.stDownloadButton > button {
    background-color: #052e16 !important;
    color: #86efac !important;
    border: 1px solid #166534 !important;
    border-radius: 8px !important;
}
.stDownloadButton > button:hover {
    background-color: #14532d !important;
}

/* ═══════════════════════════════════════════════
   TABLES / DATAFRAMES
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] *,
[data-testid="stTable"],
[data-testid="stTable"] * {
    background-color: #0d0d0d !important;
    color: #ffffff !important;
}
thead th {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
}
tbody tr:nth-child(odd) td  { background-color: #0d0d0d !important; }
tbody tr:nth-child(even) td { background-color: #111 !important; }

/* ═══════════════════════════════════════════════
   EXPANDER
═══════════════════════════════════════════════ */
[data-testid="stExpander"],
[data-testid="stExpander"] > div,
details, summary {
    background-color: #0d0d0d !important;
    color: #ffffff !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 8px !important;
}
summary { cursor: pointer; padding: 8px 12px; }
summary:hover { background-color: #111 !important; }
[data-testid="stExpander"] * { color: #ffffff !important; }

/* ═══════════════════════════════════════════════
   ALERTS / NOTIFICATIONS
═══════════════════════════════════════════════ */
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    background-color: #111 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* ═══════════════════════════════════════════════
   MISC
═══════════════════════════════════════════════ */
hr { border-color: #1f1f1f !important; margin: 16px 0; }
[data-testid="stSpinner"] * { color: #60a5fa !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* Streamlit top bar */
header[data-testid="stHeader"] {
    background-color: #000000 !important;
    border-bottom: 1px solid #111 !important;
}
header[data-testid="stHeader"] * { color: #ffffff !important; }

/* Tool menu / hamburger */
[data-testid="stToolbar"] { display: none !important; }

/* iframe overlays that can carry white bg */
iframe { background-color: #000 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Lazy agent loader ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agents():
    from agents.team_config import cio_team
    from tools import get_tool_registry
    return cio_team, get_tool_registry()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.divider()

    model_options = {
        "DeepSeek Chat (fast)":        "deepseek/deepseek-chat",
        "DeepSeek Reasoner R1 (deep)": "deepseek/deepseek-reasoner",
        "OpenAI GPT-4o":               "openai/gpt-4o",
        "OpenAI GPT-4o Mini":          "openai/gpt-4o-mini",
    }
    chosen_label = st.selectbox("CIO Brain Model", list(model_options.keys()), index=0)
    chosen_model = model_options[chosen_label]

    api_key_input = st.text_input(
        "API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        help="DeepSeek: platform.deepseek.com  |  OpenAI: platform.openai.com",
    )

    output_dir = st.text_input("Reports Folder", value="reports")

    st.divider()
    st.markdown("### Active Tools")
    try:
        from tools import get_tool_names
        for t in get_tool_names():
            st.markdown(f"- `{t}`")
    except Exception:
        st.markdown("_(loading...)_")

    st.divider()
    st.markdown("### Quick Examples")
    examples = [
        "Analyze Tesla TSLA",
        "Analyze Apple AAPL",
        "Is NVDA expensive on P/E?",
        "Compare MSFT and GOOGL",
        "How was HK market in 2015?",
        "What is FCF yield?",
        "Generate a full report for CRM",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", width="stretch"):
            st.session_state["prefill_query"] = ex

    st.divider()
    if st.button("Clear Conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ── Apply config ───────────────────────────────────────────────────────────────
def _apply_config():
    if api_key_input:
        if "deepseek" in chosen_model:
            os.environ["DEEPSEEK_API_KEY"] = api_key_input
        else:
            os.environ["OPENAI_API_KEY"] = api_key_input
    os.environ["MODEL_ID"] = chosen_model
    os.environ["REASONING_MODEL_ID"] = chosen_model


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# FinAgent CIO")
st.markdown(
    "AI Investment Research Assistant &nbsp;·&nbsp; "
    "DeepSeek · LiteLLM · Agno &nbsp;·&nbsp; "
    "yfinance · Finnhub · Alpha Vantage · Tavily"
)
st.divider()

# ── Render conversation history ────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
prompt = st.chat_input(
    "Ask about any stock, market, or financial concept...",
    key="main_input",
) or prefill

if prompt:
    _apply_config()

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build conversation context (last 4 exchanges)
    history = st.session_state.messages[:-1]
    ctx_lines = [
        f"{'User' if m['role']=='user' else 'CIO'}: {m['content'][:400]}"
        for m in history[-8:]
    ]
    context_block = "\n".join(ctx_lines)
    full_query = (
        f"[Conversation history]\n{context_block}\n\n[Current question]\n{prompt}"
        if context_block else prompt
    )

    # ── Run agent with live activity log ──────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):

        stdout_q: queue.Queue = queue.Queue()
        result_q: queue.Queue = queue.Queue()

        def _run():
            with _StdoutCapture(stdout_q):
                try:
                    team, _ = load_agents()
                    resp = team.run(full_query)
                    content = (
                        resp.content
                        if hasattr(resp, "content") and resp.content
                        else str(resp)
                    )
                    result_q.put(("ok", content))
                except Exception as exc:
                    result_q.put(("err", str(exc)))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # ── Live thinking panel ────────────────────────────────────────────
        # seen_agents is reset per query so phase banners show fresh each time
        seen_agents: set[str] = set()

        with st.status("Thinking...", expanded=True) as status_widget:
            activity_placeholder = st.empty()
            activity_lines: list[str] = []

            while thread.is_alive() or not stdout_q.empty():
                changed = False
                while not stdout_q.empty():
                    raw = stdout_q.get_nowait()
                    parsed = _parse_log(raw, seen_agents)
                    if parsed:
                        activity_lines.append(parsed)
                        changed = True

                if changed:
                    display = "\n\n".join(activity_lines[-40:])
                    activity_placeholder.markdown(display)

                time.sleep(0.12)

            thread.join(timeout=10)

            # Drain anything remaining after thread exits
            while not stdout_q.empty():
                raw = stdout_q.get_nowait()
                parsed = _parse_log(raw, seen_agents)
                if parsed:
                    activity_lines.append(parsed)
            if activity_lines:
                activity_placeholder.markdown("\n\n".join(activity_lines[-40:]))

            total_steps = len([l for l in activity_lines if "**Step" in l or "**Synthesizing" in l])
            status_widget.update(
                label=f"Analysis complete  ({total_steps} steps)",
                state="complete",
                expanded=False,
            )

        # ── Render final response below the collapsed log ─────────────────
        response_text = ""
        if not result_q.empty():
            status, content = result_q.get()
            if status == "ok":
                response_text = content
                st.markdown(response_text)
            else:
                st.error(f"**Error:** {content}")
                response_text = f"Error: {content}"
        else:
            st.error("Request timed out. Please try again.")
            response_text = "Error: timed out."

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Show download buttons only if report files were just created
    if os.path.isdir(output_dir):
        recent = sorted(
            [f for f in os.listdir(output_dir) if f.endswith((".xlsx", ".pdf", ".txt"))],
            key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
            reverse=True,
        )[:4]
        if recent:
            with st.expander("Download Generated Reports", expanded=True):
                cols = st.columns(min(len(recent), 3))
                for i, fname in enumerate(recent):
                    ext = fname.rsplit(".", 1)[-1].upper()
                    icon = "Excel" if ext == "XLSX" else "PDF" if ext == "PDF" else "TXT"
                    with cols[i % 3]:
                        with open(os.path.join(output_dir, fname), "rb") as f:
                            st.download_button(
                                label=f"[{icon}] {fname}",
                                data=f,
                                file_name=fname,
                                mime="application/octet-stream",
                                width="stretch",
                            )

# ── Report history ─────────────────────────────────────────────────────────────
if os.path.isdir(output_dir):
    files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith((".xlsx", ".pdf", ".txt"))],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
    if files:
        with st.expander(f"Report History  ({len(files)} files)", expanded=False):
            import pandas as pd
            rows = [
                {
                    "File": f,
                    "Type": f.rsplit(".", 1)[-1].upper(),
                    "Size (KB)": round(os.path.getsize(os.path.join(output_dir, f)) / 1024, 1),
                    "Saved": datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(output_dir, f))
                    ).strftime("%Y-%m-%d %H:%M"),
                }
                for f in files
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
