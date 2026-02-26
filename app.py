"""
app.py — Streamlit conversational UI for FinAgent CIO.
Launch: double-click launch.bat  or  streamlit run app.py
"""

import os
import sys
import json
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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


# ── Conversation persistence ────────────────────────────────────────────────────
_CONV_DIR = Path("conversations")


def _new_conv_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_conv(conv_id: str, title: str, messages: list) -> None:
    """Persist a conversation to disk as JSON. Silently no-ops if not writable."""
    try:
        _CONV_DIR.mkdir(exist_ok=True)
        data = {
            "id": conv_id,
            "title": title or "Untitled",
            "updated_at": datetime.now().isoformat(),
            "messages": messages,
        }
        (_CONV_DIR / f"{conv_id}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass  # Read-only filesystem (some cloud environments) — silently skip


def _load_all_convs() -> list[dict]:
    """Return all saved conversations sorted newest-first."""
    if not _CONV_DIR.exists():
        return []
    convs = []
    for p in _CONV_DIR.glob("*.json"):
        try:
            convs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(convs, key=lambda c: c.get("updated_at", ""), reverse=True)


def _delete_conv(conv_id: str) -> None:
    try:
        (_CONV_DIR / f"{conv_id}.json").unlink(missing_ok=True)
    except Exception:
        pass


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
    "QueryAnalyst":    ("🔍", "Step 1 — Understanding your query"),
    "MacroAgent":      ("🌐", "Step 2 — Macro & Sector Context"),
    "CompanyAgent":    ("🏢", "Step 3 — Company Fundamentals"),
    "WallStreetAgent": ("🏦", "Step 4 — Wall Street Analyst Intelligence"),
    "NewsAgent":       ("📰", "Step 5 — News & Sentiment"),
    "ReportManager":   ("📄", "Generating Report"),
    "CIO":             ("🧠", "Synthesizing CIO Recommendation"),
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
    "get_analyst_ratings":       "Fetching Wall Street consensus & targets",
    "get_analyst_news":          "Scanning broker research mentions",
    "get_wall_street_breakdown": "Running full Wall Street intelligence",
    "wall_street_search":        "Deep-diving analyst reports (Tavily)",
    "save_full_report":          "Generating investment report",
    "save_to_excel":             "Saving Excel report",
    "save_to_pdf":               "Saving PDF report",
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
    Two-tier parser:
      • Structured markers  (__agent__, __tool__, __text__) emitted by the
        streaming _run() function — always handled first.
      • Legacy stdout lines (fallback non-streaming path) — whitelist only.
    Returns a markdown string to display, or None to suppress.
    """
    # ── Tier 1: Structured stream markers ────────────────────────────────────
    if raw.startswith("__agent__"):
        name = raw[9:]
        if name in _STEP_MAP and name not in seen_agents:
            seen_agents.add(name)
            icon, label = _STEP_MAP[name]
            return f"\n{icon} **{label}**"
        return None

    if raw.startswith("__tool__"):
        rest = raw[8:]
        for kw, friendly in _TOOL_MAP.items():
            if rest.startswith(kw):
                arg = rest[len(kw):]          # e.g. "  `TSLA`"
                return f"  ↳ {friendly}{arg}"
        return None

    if raw.startswith("__text__"):
        text = raw[8:].strip()
        if (len(text) >= 25
                and not any(s in text for s in _SKIP_ALWAYS)
                and not text.startswith(("{", "[", "http", "def ", "class ", "import "))
                and not text.startswith("#")
                and text.count("|") < 4):    # skip table rows
            return f"  › {text[:150]}"
        return None

    # ── Tier 2: Legacy stdout (non-streaming fallback) ───────────────────────
    stripped = raw.strip()
    if len(stripped) < 4:
        return None
    for fragment in _SKIP_ALWAYS:
        if fragment in raw:
            return None
    if all(c in "─│┌┐└┘├┤┬┴┼╔╗╚╝═╠╣╦╩╬━ \t" for c in stripped):
        return None
    for name, (icon, label) in _STEP_MAP.items():
        if name in stripped and name not in seen_agents:
            seen_agents.add(name)
            return f"\n{icon} **{label}**"
    for kw, friendly in _TOOL_MAP.items():
        if kw in stripped:
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
    return None

# ── CIO output post-processor ─────────────────────────────────────────────────
# Safety net: strips any system-thinking lines that leak into the final response
# even after show_members_responses=False.  Applied after collecting all stream chunks.
_SYSTEM_LINE_PREFIXES = (
    "analysis complete",
    "i'll analyze", "let me ", "i will ", "i'm going to ", "i need to ",
    "based on my analysis", "now i'll", "now let me", "let's start",
    "i've gathered", "i've analyzed", "i'll now", "i'll provide",
    "i'll coordinate", "i'll delegate",
    "understand_query(", "finance_search(", "news_search(", "web_search(",
    "get_financial_summary(", "get_income_statement(", "get_macro_indicator(",
    "get_company_news(", "wall_street_search(", "get_wall_street_breakdown(",
    "completed in ", ".0s.", ".1s.", ".2s.", ".3s.", ".4s.", ".5s.",
    "query analysis report", "1. query type", "2. identified ticker",
    "3. user intent", "4. key context", "5. recommended", "6. priority",
    "7. suggested analysis",
    "query analysis results",
    "🧠 query understanding",
)

def _escape_currency(text: str) -> str:
    """
    Prevent Streamlit / KaTeX from mis-parsing currency dollar signs as LaTeX
    math delimiters.  e.g. 'HK$518.5' would otherwise render as 'HK' followed
    by a broken inline-math expression, causing the character-by-character
    display bug.  Replace $ with the Unicode full-width ＄ (U+FF04) which
    looks identical in the UI but is not a KaTeX trigger.
    """
    if not text:
        return text
    # Only escape $ that is followed by a digit, decimal, or comma (currency use).
    # Leave $ alone when it's part of a Markdown code block (```) or HTML.
    result = _re.sub(r'\$(?=[\d,.])', '＄', text)
    return result


def _clean_cio_output(text: str) -> str:
    """
    Remove any residual system-thinking lines or verbose QueryAnalyst reports
    from the CIO's final response.  Lines that start with known system-thinking
    phrases are dropped; the rest is reassembled.
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned: list[str] = []
    skip_block = False   # True while inside a numbered-list "report" block

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Detect start of a numbered report block (e.g. "1. Query Type Classification")
        if _re.match(r'^[1-9]\d?\.\s+[A-Z]', stripped) and len(stripped) < 80:
            # Only skip if it looks like internal report boilerplate
            if any(kw in lower for kw in ["query type", "ticker", "user intent",
                                           "key context", "recommended analysis",
                                           "priority area", "suggested analysis"]):
                skip_block = True
        # A markdown header or blank line ends a skip block
        if stripped.startswith("##") or stripped.startswith("**") or stripped == "":
            skip_block = False

        if skip_block:
            continue

        # Drop individual system-thinking lines
        if any(lower.startswith(p) for p in _SYSTEM_LINE_PREFIXES):
            continue
        # Drop lines that look like raw tool call completions
        if _re.search(r'completed in \d+\.\d+s', line):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # Collapse runs of 3+ blank lines into 2
    result = _re.sub(r'\n{3,}', '\n\n', result)
    return result


# ── Pre-flight depth hint ──────────────────────────────────────────────────────
import re as _re

# Numeric exchange ticker patterns — e.g. "3690 HK", "8306 T", "2015.HK"
# These must NEVER be answered from the CIO's training-data knowledge alone
# because the CIO can misidentify companies (e.g. 3690.HK is Meituan, not Geely).
_NUMERIC_TICKER_RE = _re.compile(
    r'(?<!\w)\d{1,6}(?:\s+|\.)(?:HK|T|L|SS|SZ|KS|TW|JP|NS|BO)(?!\w)',
    _re.IGNORECASE,
)

# Detect standard US/global alpha tickers (2-5 uppercase letters).
# Exclude common non-ticker abbreviations so "AI" or "HK" alone don't trigger.
_NON_TICKER_WORDS = {
    "AI", "US", "UK", "HK", "EU", "PE", "EV", "IPO", "ETF", "GDP", "CPI",
    "FED", "EPS", "DCF", "FCF", "ROE", "ROA", "QE", "FX", "VC", "CF",
    "TV", "HI", "LO", "MA", "NA", "OK", "OR", "BY", "AT", "TO", "IN",
    "IS", "IT", "MY", "GO", "NO", "SO", "DO", "BE",
}
_US_TICKER_RE = _re.compile(r'\b([A-Z]{2,5})\b')


def _has_numeric_ticker(text: str) -> bool:
    return bool(_NUMERIC_TICKER_RE.search(text))


def _has_alpha_ticker(text: str) -> bool:
    """Return True if the query contains what looks like a stock ticker symbol."""
    for m in _US_TICKER_RE.finditer(text):
        word = m.group(1)
        if word not in _NON_TICKER_WORDS:
            return True
    return False


def _detect_depth_hint(text: str) -> int:
    """
    Fast client-side depth heuristic (runs in microseconds, no API call).
    Returns 1, 2, 4 to tell the CIO a depth level, or 0 meaning
    'let the CIO decide via its own depth-level assessment'.

    KEY RULES:
    • Queries containing ANY stock ticker (numeric or alpha) → minimum Level 2.
      The CIO's training data is months/years stale — live data is required.
    • "comprehensive/deep/analysis" keywords → Level 4.
    • Level 1 is ONLY for pure concept questions (no ticker, no company name).
    """
    t = text.lower().strip()
    words = t.split()
    wc = len(words)

    deep_kw = ("analyz", "analysis", "research", "deep dive", "comprehensive",
               "detailed", "全面", "详细", "深度", "research note", "thesis",
               "should i invest", "worth investing", "值得投资", "帮我写",
               "full analysis", "in-depth", "indepth")

    # ── Deep/comprehensive keywords → Level 4 (always, regardless of length) ─
    if any(k in t for k in deep_kw):
        return 4

    # ── Guard: numeric exchange tickers must call CompanyAgent (never L1) ──
    if _has_numeric_ticker(text):
        return 2

    # ── Guard: alpha tickers (NVDA, TSLA, AAPL…) also need live data ────────
    # "hows NVDA", "TSLA price", "what's AAPL doing" → live CompanyAgent call.
    if _has_alpha_ticker(text):
        return 2

    # ── Explicit fast-path: user explicitly wants a one-liner ─────────────────
    if any(w in t for w in ["quick", "brief", "tldr", "tl;dr", "just tell",
                              "one sentence", "一句话", "简单说", "简短"]):
        return 1

    # ── Pure concept question: no ticker present, explains a term ─────────────
    concept_starts = ("what is", "what are", "what's a", "explain ", "define ",
                      "how does", "how do ", "what does", "什么是", "如何理解")
    if wc <= 10 and any(t.startswith(s) or s in t for s in concept_starts):
        return 1

    # ── Short query without ticker/company → Level 1 ─────────────────────────
    # Only safe if NO ticker detected above (both guards already returned 2).
    if wc <= 4:
        return 1

    # ── Medium query → Level 2 ────────────────────────────────────────────────
    if wc <= 10:
        return 2

    return 0  # let the CIO assess


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

/* Sidebar — outer border on the section only; NO border on inner wrappers
   (inner border-right creates phantom vertical lines next to columns/buttons) */
section[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #1e1e1e !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] [data-testid="column"],
section[data-testid="stSidebar"] .element-container {
    background-color: #0d0d0d !important;
    border: none !important;   /* no inner borders — prevents vertical lines */
}
/* Sidebar form controls — subtle, no heavy box */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background-color: #161616 !important;
    border: 1px solid #272727 !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: #3b82f6 !important;
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
   CHAT INPUT  — single outer border, zero inner frames
   The global [data-baseweb="textarea"] textarea rule gives
   the inner textarea its own border+border-radius — we nuke
   everything inside stChatInput with the * selector so no
   inner frame can bleed through regardless of source.
═══════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background-color: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #444 !important;
    box-shadow: none !important;
}
/* Nuclear: strip every border/shadow/radius from every child
   (excludes the send button so it keeps its own bg) */
[data-testid="stChatInput"] *:not(button):not(svg):not(path) {
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
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
   BUTTONS  (global)
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
    color: #ffffff !important;
}

/* ── Sidebar conversation history buttons — need higher specificity
   because Streamlit Cloud may render these with a white default bg ── */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #1a1a1a !important;
    color: #d4d4d4 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-weight: 400 !important;
    padding: 6px 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}

/* Caption text under each conversation entry */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p {
    color: #666 !important;
    font-size: 10px !important;
    margin-top: -6px !important;
    margin-bottom: 4px !important;
    padding-left: 2px !important;
    line-height: 1.2 !important;
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

/* Streamlit top bar — background only; do NOT set color on * to avoid
   bleeding white into the sidebar toggle icons */
header[data-testid="stHeader"] {
    background-color: #000000 !important;
    border-bottom: 1px solid #111 !important;
}

/* Hide only the branding / share toolbar, not the sidebar toggle */
[data-testid="stToolbar"] { display: none !important; }

/* iframe overlays that can carry white bg */
iframe { background-color: #000 !important; }

/* ═══════════════════════════════════════════════
   SIDEBAR — COLLAPSE BUTTON  ("«" inside the open sidebar)
   Style ONLY the <button> element, NOT the wrapper div —
   styling the wrapper div creates phantom dark boxes inside
   the sidebar content area.
═══════════════════════════════════════════════ */
[data-testid="stSidebarCollapseButton"] button {
    background-color: #1f1f1f !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background-color: #2563eb !important;
    border-color: #3b82f6 !important;
}

/* ═══════════════════════════════════════════════
   SIDEBAR — EXPAND BUTTON  (appears when sidebar is collapsed)
   Convert it into a prominent fixed blue ☰ button in the
   top-left corner.  Covers all known data-testid variants:
     collapsedControl               (legacy Streamlit)
     stSidebarCollapsedControl      (Streamlit 1.x)
     stSidebarUserCollapsedControl  (Streamlit 1.3x+)
═══════════════════════════════════════════════ */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarUserCollapsedControl"] {
    position: fixed !important;
    top: 6px !important;
    left: 6px !important;
    z-index: 9999998 !important;
    width: 38px !important;
    height: 38px !important;
    background: #1d4ed8 !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
}
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarUserCollapsedControl"] button {
    background: transparent !important;
    border: none !important;
    width: 38px !important;
    height: 38px !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* Hide native SVG chevron, replace with ☰ text via ::after */
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarUserCollapsedControl"] svg { display: none !important; }
[data-testid="collapsedControl"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="stSidebarUserCollapsedControl"] button::after {
    content: "☰";
    font-size: 18px;
    color: #ffffff;
    line-height: 1;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="stSidebarUserCollapsedControl"]:hover {
    background: #2563eb !important;
}

/* ═══════════════════════════════════════════════
   MOBILE RESPONSIVE  (≤ 768 px)
═══════════════════════════════════════════════ */
@media screen and (max-width: 768px) {

    /* Reduce content padding so text isn't clipped */
    .block-container,
    [data-testid="block-container"] {
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        padding-top: 3.5rem !important;   /* room for floating ☰ */
        max-width: 100% !important;
    }

    /* Smaller headings */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.05rem !important; }
    h3 { font-size: 0.95rem !important; }

    /* Touch-friendly buttons (Apple HIG: 44 px min) */
    .stButton > button {
        min-height: 44px !important;
        font-size: 13px !important;
        padding: 6px 10px !important;
    }

    /* Sidebar: full-width drawer on mobile */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        width: 82vw !important;
        max-width: 300px !important;
    }

    /* Chat messages — tighter padding */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] > div {
        padding: 8px 10px !important;
        border-radius: 8px !important;
    }

    /* Chat input — stick to bottom, no extra margin */
    [data-testid="stChatInput"] {
        border-radius: 10px !important;
    }

    /* Status / thinking panel */
    [data-testid="stStatus"] {
        font-size: 12px !important;
    }

    /* Divider spacing */
    hr { margin: 8px 0 !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar toggle (JavaScript) ────────────────────────────────────────────────
# Two-layer approach for maximum reliability across Streamlit versions:
#  1. CSS (above) styles the native expand button as a blue ☰ — works when
#     Streamlit's testid matches one of the three known variants.
#  2. JS (here) injects a guaranteed-present floating ☰ button as fallback.
#     When clicked it uses p.MouseEvent (cross-frame reliable) to click the
#     native button, falling back to a page reload if nothing is found.
#     The reload always opens the sidebar because localStorage is cleared first.
components.html("""
<script>
(function () {
    var p = window.parent;
    if (!p || !p.document) return;
    var doc = p.document;

    /* ── 1. Clear localStorage sidebar state (sidebar opens expanded on reload) ── */
    try {
        var ls = p.localStorage;
        if (ls) {
            Object.keys(ls).forEach(function (k) {
                if (/sidebar/i.test(k)) ls.removeItem(k);
            });
        }
    } catch (e) {}

    /* ── 2. Sidebar state helper ── */
    function isSidebarOpen() {
        var sb = doc.querySelector('section[data-testid="stSidebar"]');
        if (!sb) return true;
        var aria = sb.getAttribute('aria-expanded');
        if (aria === 'false') return false;
        if (aria === 'true')  return true;
        return sb.getBoundingClientRect().width > 50;
    }

    /* ── 3. Try every known method to open the sidebar ── */
    function tryOpen() {
        var selectors = [
            '[data-testid="collapsedControl"] button',
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarUserCollapsedControl"] button',
            '[data-testid*="CollapsedControl"] button',
        ];
        for (var i = 0; i < selectors.length; i++) {
            var el = doc.querySelector(selectors[i]);
            if (el && el.id !== 'fa-sb-open') {
                try {
                    /* Use parent window's MouseEvent — cross-frame reliable */
                    el.dispatchEvent(new p.MouseEvent('click', {
                        bubbles: true, cancelable: true, view: p
                    }));
                } catch (_) { try { el.click(); } catch (_2) {} }
                return;
            }
        }
        /* Hard fallback: reload — sidebar opens expanded because localStorage is clear */
        p.location.reload();
    }

    /* ── 4. Inject / sync the floating ☰ button ── */
    function syncBtn(btn) {
        btn.style.display = isSidebarOpen() ? 'none' : 'flex';
    }

    function injectBtn() {
        if (doc.getElementById('fa-sb-open')) return;
        var btn = doc.createElement('button');
        btn.id = 'fa-sb-open';
        btn.innerHTML = '&#9776;';  /* ☰ */
        btn.title = 'Open sidebar';
        btn.setAttribute('style', [
            'position:fixed',
            'top:8px', 'left:8px',
            'z-index:2147483647',
            'width:38px', 'height:38px',
            'background:#1d4ed8',
            'color:#fff',
            'border:none',
            'border-radius:8px',
            'font-size:18px',
            'cursor:pointer',
            'display:none',
            'align-items:center',
            'justify-content:center',
            'box-shadow:0 2px 10px rgba(0,0,0,0.6)',
            'transition:background 0.15s',
        ].join(';'));
        btn.addEventListener('click', function () {
            tryOpen();
            setTimeout(function () { syncBtn(btn); }, 500);
        });
        btn.addEventListener('mouseenter', function () { btn.style.background = '#2563eb'; });
        btn.addEventListener('mouseleave', function () { btn.style.background = '#1d4ed8'; });
        doc.body.appendChild(btn);

        /* Watch ONLY sidebar aria-expanded — targeted, never the whole DOM */
        var sb = doc.querySelector('section[data-testid="stSidebar"]');
        if (sb) {
            try {
                new MutationObserver(function () { syncBtn(btn); })
                    .observe(sb, { attributes: true, attributeFilter: ['aria-expanded'] });
            } catch (_) {}
        }
        setInterval(function () { syncBtn(btn); }, 1000);
        syncBtn(btn);
    }

    if (doc.readyState !== 'loading') injectBtn();
    else doc.addEventListener('DOMContentLoaded', injectBtn);

    var n = 0, t = setInterval(function () {
        injectBtn();
        if (++n >= 5) clearInterval(t);
    }, 1000);
})();
</script>
""", height=0, scrolling=False)

# Hide the 0-height component iframe from the layout
st.markdown(
    "<style>iframe[height='0']{display:none!important}</style>",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_id" not in st.session_state:
    st.session_state.conv_id = _new_conv_id()
if "conv_title" not in st.session_state:
    st.session_state.conv_title = ""


# ── Lazy agent loader ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agents():
    from agents.team_config import cio_team
    from tools import get_tool_registry
    return cio_team, get_tool_registry()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FinAgent CIO")

    # ── New Chat ────────────────────────────────────────────────────────────────
    if st.button("＋  New Chat", width="stretch", key="new_chat_btn"):
        # Auto-save current conversation before starting fresh
        if st.session_state.messages:
            _save_conv(
                st.session_state.conv_id,
                st.session_state.conv_title,
                st.session_state.messages,
            )
        st.session_state.messages = []
        st.session_state.conv_id = _new_conv_id()
        st.session_state.conv_title = ""
        st.rerun()

    # Thin separator — raw HTML avoids Streamlit's wrapper divs that render as boxes
    st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #222'>",
                unsafe_allow_html=True)

    # ── Model selector ──────────────────────────────────────────────────────────
    model_options = {
        "DeepSeek Chat (fast)":        "deepseek/deepseek-chat",
        "DeepSeek Reasoner R1 (deep)": "deepseek/deepseek-reasoner",
    }
    chosen_label = st.selectbox("AI Model", list(model_options.keys()), index=0)
    chosen_model = model_options[chosen_label]

    # Reports output folder (local only — cloud uses ephemeral storage)
    output_dir = st.text_input("Reports Folder", value="reports")

    st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #222'>",
                unsafe_allow_html=True)

    # ── Conversation history ────────────────────────────────────────────────────
    all_convs = _load_all_convs()
    if all_convs:
        st.markdown("### Recent Chats")
        for conv in all_convs[:20]:
            c_id    = conv.get("id", "")
            c_title = conv.get("title", "Untitled")
            c_date  = conv.get("updated_at", "")[:10]
            c_msgs  = conv.get("messages", [])
            label   = c_title[:32] + ("…" if len(c_title) > 32 else "")

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                active = c_id == st.session_state.conv_id
                btn_label = f"**{label}**" if active else label
                if st.button(btn_label, key=f"conv_{c_id}", width="stretch"):
                    st.session_state.messages   = c_msgs
                    st.session_state.conv_id    = c_id
                    st.session_state.conv_title = conv.get("title", "")
                    st.rerun()
                n_ex = len(c_msgs) // 2
                st.caption(f"{c_date} · {n_ex} msg{'s' if n_ex != 1 else ''}")
            with col_del:
                if st.button("🗑", key=f"del_{c_id}"):
                    _delete_conv(c_id)
                    if c_id == st.session_state.conv_id:
                        st.session_state.messages   = []
                        st.session_state.conv_id    = _new_conv_id()
                        st.session_state.conv_title = ""
                    st.rerun()
        st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #222'>",
                    unsafe_allow_html=True)

    # ── Quick Examples ──────────────────────────────────────────────────────────
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


# ── Apply config ───────────────────────────────────────────────────────────────
def _apply_config():
    # API keys are read exclusively from .env (local) or st.secrets (Streamlit Cloud).
    # They are never accepted from the UI to prevent accidental exposure.
    os.environ["MODEL_ID"] = chosen_model
    os.environ["REASONING_MODEL_ID"] = chosen_model


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# FinAgent CIO")
st.markdown("AI-Powered Investment Research Assistant")
st.divider()

# ── Render conversation history ────────────────────────────────────────────────
# Only render the most recent 12 messages to keep browser memory low.
# All messages are still stored in session_state and saved to disk for history.
_render_msgs = st.session_state.messages[-12:]
if len(st.session_state.messages) > 12:
    st.caption(f"↑ {len(st.session_state.messages) - 12} earlier messages hidden — load from sidebar to view full history")
for msg in _render_msgs:
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

    # Pre-flight depth hint — injected as first line so the CIO reads it before
    # anything else, preventing unnecessary sub-agent calls for simple queries.
    depth_hint = _detect_depth_hint(prompt)
    _numeric = _has_numeric_ticker(prompt)
    _HINT_LABELS: dict[int, str] = {
        1: (
            "[FAST-PATH Level 1: answer INSTANTLY from knowledge. "
            "ZERO tool calls. ≤100 words.]"
        ),
        2: (
            # Numeric ticker variant — must call CompanyAgent to resolve company name
            "[FAST-PATH Level 2 — NUMERIC TICKER: "
            "Skip QueryAnalyst BUT MUST call CompanyAgent(ticker) FIRST. "
            "NEVER guess company name from training data — "
            "Asian numeric codes are frequently mis-identified "
            "(e.g. 3690.HK=Meituan NOT Geely; 0175.HK=Geely; "
            "1211.HK=BYD; 2015.HK=Li Auto; 0700.HK=Tencent; "
            "9988.HK=Alibaba; 9866.HK=NIO; 9868.HK=XPeng; "
            "0941.HK=China Mobile; 0005.HK=HSBC). "
            "≤250 words.]"
            if _numeric else
            "[FAST-PATH Level 2: skip QueryAnalyst. "
            "AT MOST 1 sub-agent call. ≤250 words.]"
        ),
    }
    hint_prefix = _HINT_LABELS.get(depth_hint, "")

    # Build conversation context (last 4 exchanges)
    history = st.session_state.messages[:-1]
    ctx_lines = [
        f"{'User' if m['role']=='user' else 'CIO'}: {m['content'][:400]}"
        for m in history[-8:]
    ]
    context_block = "\n".join(ctx_lines)
    base_query = (
        f"[Conversation history]\n{context_block}\n\n[Current question]\n{prompt}"
        if context_block else prompt
    )
    full_query = f"{hint_prefix}\n\n{base_query}" if hint_prefix else base_query

    # ── Run agent with live activity log ──────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):

        stdout_q: queue.Queue = queue.Queue()
        result_q: queue.Queue = queue.Queue()

        def _run():
            """
            Preferred path: Agno stream=True yields RunResponse chunks so we
            can emit live step/tool/content markers to stdout_q.
            Fallback: non-streaming with stdout capture (if streaming unsupported).
            """
            def _emit(msg: str) -> None:
                stdout_q.put(msg)

            def _scan_chunk(text: str, seen: set) -> None:
                """Emit markers for any agent names / tool calls found in chunk."""
                for name in _STEP_MAP:
                    if name in text and name not in seen:
                        seen.add(name)
                        _emit(f"__agent__{name}")
                for kw in _TOOL_MAP:
                    if kw in text:
                        arg = ""
                        if "(" in text:
                            try:
                                s = text.index("(") + 1
                                e = text.index(")", s)
                                cand = text[s:e].strip().strip("'\"")
                                if cand and len(cand) <= 30 and " " not in cand:
                                    arg = f"  `{cand}`"
                            except ValueError:
                                pass
                        _emit(f"__tool__{kw}{arg}")
                        break   # one tool emit per chunk is enough

            try:
                team, _ = load_agents()
                chunks: list[str] = []
                stream_seen: set[str] = set()

                # ── Streaming path ──────────────────────────────────────────
                try:
                    for chunk in team.run(full_query, stream=True):
                        c = ""
                        if hasattr(chunk, "content") and chunk.content:
                            c = str(chunk.content)
                            chunks.append(c)

                        _scan_chunk(c, stream_seen)

                        # Emit readable content lines for the thinking panel
                        for line in c.split("\n"):
                            line = line.strip()
                            if (30 <= len(line) <= 160
                                    and not any(s in line for s in _SKIP_ALWAYS)
                                    and not line.startswith(("{", "[", "http"))
                                    and line.count("|") < 4):
                                _emit(f"__text__{line}")

                    raw_final = "".join(chunks).strip()
                    final = _clean_cio_output(raw_final)
                    result_q.put(("ok", final or "[No response generated]"))

                except Exception:
                    # ── Non-streaming fallback ──────────────────────────────
                    with _StdoutCapture(stdout_q):
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

            n_steps = len([l for l in activity_lines if "**Step" in l or "**Synthesizing" in l or "**Generating" in l])
            n_tools = len([l for l in activity_lines if l.strip().startswith("↳")])
            summary = f"Analysis complete — {n_steps} agents · {n_tools} tool calls"
            status_widget.update(
                label=summary,
                state="complete",
                expanded=False,
            )

        # ── Render final response below the collapsed log ─────────────────
        response_text = ""
        if not result_q.empty():
            status, content = result_q.get()
            if status == "ok":
                response_text = content
                st.markdown(_escape_currency(response_text))
            else:
                st.error(f"**Error:** {content}")
                response_text = f"Error: {content}"
        else:
            st.error("Request timed out. Please try again.")
            response_text = "Error: timed out."

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Auto-save conversation after every exchange
    if not st.session_state.conv_title and st.session_state.messages:
        # Use the first user message (truncated) as the conversation title
        first_user = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"), ""
        )
        st.session_state.conv_title = first_user[:60]
    _save_conv(
        st.session_state.conv_id,
        st.session_state.conv_title,
        st.session_state.messages,
    )

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
