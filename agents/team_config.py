"""
agents/team_config.py — Multi-agent CIO team.

Data-source matrix:
─────────────────────────────────────────────────────────────────────
Agent           Tools
─────────────────────────────────────────────────────────────────────
QueryAnalyst    TavilyEngine          ← intent classification
MacroAgent      TavilyEngine          ← macro / sector web search
                AlphaVantageEngine    ← GDP, CPI, FedRate macro data
CompanyAgent    FinanceEngine         ← yfinance price/financials
                FinnhubEngine         ← analyst ratings, earnings, insiders
                AlphaVantageEngine    ← EPS history, overview, EV/EBITDA
NewsAgent       TavilyEngine          ← grounded news search
                NewsEngine            ← DuckDuckGo (fallback)
                FinnhubEngine         ← company news (reliable non-US)
                AlphaVantageEngine    ← AI-scored news sentiment
ReportManager   ReportEngine          ← Excel + PDF (only on user request)
─────────────────────────────────────────────────────────────────────

Key change: ReportManager is ONLY called when the user explicitly
asks for a report ("生成报告", "save report", "export", etc.).
════════════════════════════════════════════════════════════════════
在此处添加新的 Agent:
  1. 创建 tools/my_tool.py (继承 BaseTool)
  2. 在下方仿照现有 Agent 定义新 Agent
  3. 在 cio_team members 列表追加即可
════════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.litellm import LiteLLM
from agno.team import Team

from tools.tavily_engine import TavilyEngine
from tools.finance_engine import FinanceEngine
from tools.finnhub_engine import FinnhubEngine
from tools.alpha_vantage_engine import AlphaVantageEngine
from tools.news_engine import NewsEngine
from tools.report_engine import ReportEngine

load_dotenv()


# ── Model factories ───────────────────────────────────────────────────────────

def _chat_model() -> LiteLLM:
    model_id = os.getenv("MODEL_ID", "deepseek/deepseek-chat")
    return LiteLLM(id=model_id, api_key=_resolve_key(model_id))


def _reasoner_model() -> LiteLLM:
    model_id = os.getenv("REASONING_MODEL_ID", "deepseek/deepseek-chat")
    return LiteLLM(id=model_id, api_key=_resolve_key(model_id))


def _resolve_key(model_id: str) -> str:
    if model_id.startswith("deepseek/"):
        return os.getenv("DEEPSEEK_API_KEY", "")
    if model_id.startswith("openai/"):
        return os.getenv("OPENAI_API_KEY", "")
    if model_id.startswith("anthropic/"):
        return os.getenv("ANTHROPIC_API_KEY", "")
    # ════════════════════════════════════════════════════════════════════
    # 在此处添加新的模型提供商
    # if model_id.startswith("mistral/"): return os.getenv("MISTRAL_API_KEY","")
    # ════════════════════════════════════════════════════════════════════
    return os.getenv("DEEPSEEK_API_KEY", "")


# ── Agent 0: QueryAnalyst ─────────────────────────────────────────────────────

query_analyst = Agent(
    name="QueryAnalyst",
    role="Query Intent Classifier & Context Researcher",
    model=_chat_model(),
    tools=[TavilyEngine()],
    instructions=[
        "You are the team's first line of intelligence.",
        "ALWAYS call understand_query(user_input) first — no exceptions.",
        "This tells the CIO exactly what kind of question it is and which",
        "agents/tools to call. Return the full output verbatim.",
    ],
    markdown=True,
)


# ── Agent 1: MacroAgent ───────────────────────────────────────────────────────

macro_agent = Agent(
    name="MacroAgent",
    role="Macroeconomic & Sector Analyst",
    model=_chat_model(),
    tools=[TavilyEngine(), AlphaVantageEngine()],
    instructions=[
        "You are a senior macroeconomic analyst. Be CONCISE — 150-200 words max.",
        "",
        "Call at most 2 tools:",
        "  US stocks / current:  get_macro_indicator('GDP') + get_macro_indicator('FED_RATE')",
        "  Non-US / sectors:     finance_search('<country or sector> macro outlook 2026')",
        "  Historical queries:   web_search('<topic> <year> economic conditions')",
        "",
        "Output format (strict, no padding):",
        "  ## Macro Context",
        "  - [3 data-backed bullets: interest rates, growth, inflation relevant to the query]",
        "  ## Sector Impact",
        "  - [1-2 bullets: tailwind or headwind for the specific sector/company]",
    ],
    markdown=True,
)


# ── Agent 2: CompanyAgent ─────────────────────────────────────────────────────

company_agent = Agent(
    name="CompanyAgent",
    role="Fundamental Research Analyst",
    model=_chat_model(),
    tools=[FinanceEngine(), FinnhubEngine(), AlphaVantageEngine()],
    instructions=[
        "You are a fundamental equity research analyst. Be CONCISE — 200-250 words max.",
        "Only run for STOCK_ANALYSIS queries.",
        "",
        "Call EXACTLY these 4 tools in order — no more:",
        "  1. get_financial_summary(ticker)   — current price, P/E, market cap (yfinance)",
        "  2. get_income_statement(ticker)    — 3-year revenue & profit trend (yfinance)",
        "  3. get_analyst_ratings(ticker)     — analyst Buy/Hold/Sell consensus (Finnhub, US only)",
        "  4. get_earnings_surprise(ticker)   — last 4 quarters EPS beat/miss (Finnhub, US only)",
        "",
        "Note: Finnhub returns a skip message for non-US tickers (e.g. 8306.T, HSBA.L) — expected.",
        "For non-US tickers, tools 3 & 4 will gracefully return N/A — that is fine.",
        "",
        "Output format (strict):",
        "  ## Fundamental Analysis",
        "  - Valuation: [P/E, P/S, Market Cap vs peers]",
        "  - Revenue trend: [3-year direction and key driver]",
        "  - Profitability: [margin & FCF summary]",
        "  - Analyst view: [consensus + EPS beat pattern]",
        "  - Key risk flag: [1 specific concern from the data]",
        "",
        "Do NOT call report tools.",
    ],
    markdown=True,
)


# ── Agent 3: NewsAgent ────────────────────────────────────────────────────────

news_agent = Agent(
    name="NewsAgent",
    role="Market Intelligence & Sentiment Analyst",
    model=_chat_model(),
    tools=[TavilyEngine(), FinnhubEngine(), AlphaVantageEngine(), NewsEngine()],
    instructions=[
        "You are a market intelligence and news analyst. Be CONCISE — 150-200 words max.",
        "",
        "Call EXACTLY 1 primary tool (2 if the first returns no results):",
        "  US ticker:       get_company_news(ticker)  — Finnhub, most reliable",
        "  Non-US ticker:   news_search('<company name> stock news latest 2026')",
        "  Historical:      web_search('<company or market> <year> key events')",
        "  If no results:   get_stock_news(ticker)  — DuckDuckGo fallback",
        "",
        "Output format (strict):",
        "  ## News & Sentiment",
        "  Top 3-5 headlines — format: [Date] Source — Headline",
        "  Sentiment verdict: Positive / Neutral / Negative",
        "  Rationale: [1 sentence explaining the verdict]",
    ],
    markdown=True,
)


# ── Agent 4: ReportManager ────────────────────────────────────────────────────

report_manager = Agent(
    name="ReportManager",
    role="Chief Report Officer",
    model=_chat_model(),
    tools=[ReportEngine(output_dir="reports")],
    instructions=[
        "You generate investment reports ONLY when explicitly asked by the user.",
        "Trigger phrases: '生成报告', '保存报告', 'generate report', 'save report',",
        "  'export', '导出', 'PDF', 'Excel', 'download'.",
        "When triggered: call save_full_report() and confirm the file paths.",
        "Do NOT run automatically at the end of every analysis.",
    ],
    markdown=True,
)


# ── CIO Team ──────────────────────────────────────────────────────────────────

cio_team = Team(
    name="CIO_FinancialAnalysisTeam",
    mode="coordinate",
    model=_reasoner_model(),
    members=[query_analyst, macro_agent, company_agent, news_agent, report_manager],
    instructions=[
        "You are the Chief Investment Officer of a quantitative research firm.",
        "You are smart about routing — you do NOT call every agent every time.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "STEP 0 — MANDATORY: Ask QueryAnalyst first.",
        "  Call: understand_query('<user input>')",
        "  Read the Query Type, tickers, time period, and approach.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "STEP 0.5 — DEPTH LEVEL (assess from user tone BEFORE routing):",
        "",
        "  LEVEL 1 — Quick Hit  (output ≤ 120 words)",
        "    Signals: 'quick', 'brief', 'tldr', 'just tell me', '一句话', '简单说',",
        "             message ≤ 4 words, very casual phrasing",
        "    Action:  Answer DIRECTLY from knowledge. Skip ALL sub-agents.",
        "    Output:  2-3 sentences: key fact + 1 data point + verdict.",
        "",
        "  LEVEL 2 — Snapshot  (output ≤ 280 words)",
        "    Signals: Short casual question, 'how is X?', 'what's X like?',",
        "             'worth buying?', '怎么样', no explicit depth request",
        "    Action:  Run at most 1 sub-agent (NewsAgent or CompanyAgent).",
        "    Output:  2-3 short bullet sections + one-line verdict.",
        "",
        "  LEVEL 3 — Standard  (output ≤ 500 words)  ← DEFAULT",
        "    Signals: Normal clear question, no depth modifiers.",
        "    Action:  Route per the routing table below.",
        "    Output:  Standard sections, each 3-5 bullets.",
        "",
        "  LEVEL 4 — Deep Dive  (output ≤ 850 words)",
        "    Signals: 'analyze in detail', 'deep dive', 'comprehensive',",
        "             'detailed analysis', '详细分析', '全面分析',",
        "             decision-framing ('should I invest', 'is it worth it')",
        "    Action:  All relevant sub-agents; instruct each to be thorough.",
        "    Output:  Full sections, include data tables, explicit data conflicts.",
        "",
        "  LEVEL 5 — Full Thesis  (output ≤ 1300 words)",
        "    Signals: 'full analysis', 'research note', 'investment thesis',",
        "             'institutional quality', '完整报告', multi-stock comparison,",
        "             '帮我写一份研究报告'",
        "    Action:  All sub-agents with maximum detail; ReportManager if asked.",
        "    Output:  Report-quality, all data, full synthesis & scenario analysis.",
        "",
        "  IMPORTANT: Even Level 1 must include at least ONE cited data point",
        "  (price, P/E, growth rate, etc.) to stay professional and credible.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "ROUTING BY QUERY TYPE:",
        "",
        "  STOCK_ANALYSIS ('分析TSLA', 'Should I buy NVDA', '8306.T怎么样'):",
        "    1. MacroAgent  — sector + macro context",
        "    2. CompanyAgent — fundamentals (yfinance + Finnhub + AlphaVantage)",
        "    3. NewsAgent    — headlines + AI sentiment (Finnhub first, then others)",
        "    4. YOUR synthesis → BUY / HOLD / SELL",
        "    ✘ Do NOT call ReportManager (user hasn't asked for a report)",
        "",
        "  MARKET_ANALYSIS ('港股今年', 'US market outlook'):",
        "    1. MacroAgent  — broad market + macro indicators",
        "    2. NewsAgent    — market-level news via Tavily",
        "    3. YOUR synthesis",
        "    ✘ Skip CompanyAgent and ReportManager",
        "",
        "  HISTORICAL_ANALYSIS ('how's 2015 HK', '2020年美股崩盘'):",
        "    1. MacroAgent  — Tavily web_search for historical macro",
        "    2. NewsAgent    — Tavily web_search for historical events",
        "    3. YOUR synthesis — clearly state the year being discussed",
        "    ✘ Skip CompanyAgent (yfinance has no historical market data)",
        "    ✘ Skip ReportManager",
        "",
        "  COMPARISON ('TSLA vs AAPL'):",
        "    Run STOCK_ANALYSIS for each ticker, then compare side-by-side.",
        "",
        "  CONCEPT_EXPLANATION ('什么是P/E', 'explain DCF'):",
        "    Answer directly from knowledge. Skip ALL sub-agents.",
        "",
        "  REPORT_REQUEST ('生成报告', 'save report', 'export PDF'):",
        "    Call ReportManager.save_full_report() only.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "SYNTHESIS (mandatory reflection before output):",
        "  Before writing the final answer, verify:",
        "    1. Do macro conditions support or contradict the fundamentals?",
        "    2. Does recent news confirm or challenge the valuation thesis?",
        "    3. Are there data conflicts? State them explicitly — do not hide them.",
        "  Then commit to a clear verdict: BUY / HOLD / SELL",
        "    - Conviction level: High / Medium / Low",
        "    - Target price range and time horizon (e.g. 12-month)",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "OUTPUT FORMAT — adapt sections and depth to DEPTH LEVEL above:",
        "  Level 1: 2-3 sentences only.",
        "  Level 2: ## Snapshot + ## Verdict  (no full section structure).",
        "  Level 3-5:",
        "    ## 🧠 Query Understanding   ← 1-2 lines: what was asked + query type",
        "    ## 📊 [Relevant agent sections — summarise, don't repeat verbatim]",
        "    ## 🎯 Recommendation        ← BUY/HOLD/SELL | Conviction | Target | Horizon",
        "    ## ⚠️  Key Risks             ← bullets (3 for L3, 5 for L4-5)",
        "    ## 🚀 Catalysts             ← bullets (2 for L3, 3-4 for L4-5)",
        "    Level 4-5 add: ## 📐 Data Snapshot  (key metrics table)",
        "    Level 5 add:   ## 🔭 Scenario Analysis  (bull / base / bear cases)",
        "  (No report section unless user explicitly asked for one)",
        "",
        "CONVERSATION: For follow-ups, use prior context — skip QueryAnalyst.",
        "",
        "> ⚠️ Disclaimer: AI-generated analysis. Not financial advice.",
    ],
    markdown=True,
    show_members_responses=True,
    share_member_interactions=True,
    # NOTE: add_history_to_context intentionally omitted — no DB backend configured.
    # Conversation history is injected manually via full_query in app.py.
)
