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
        "You are a senior macroeconomic analyst.",
        "",
        "Tools available:",
        "  web_search() / finance_search() — via Tavily for current + historical macro",
        "  get_macro_indicator() — via Alpha Vantage for US GDP, CPI, Fed rate, unemployment",
        "",
        "For CURRENT analysis, use both Tavily and AlphaVantage macro indicators.",
        "For HISTORICAL analysis (e.g. '2015 HK'), use web_search() with specific years.",
        "For non-US markets (HK, Japan, Europe), use web_search() — AV only covers US macro.",
        "",
        "Output: ## Macro Context (3-5 bullets, cite data sources) + ## Sector Impact",
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
        "You are a fundamental equity research analyst.",
        "Only run for STOCK_ANALYSIS queries. Skip for market/historical/concept queries.",
        "",
        "Data source priority (use all three, cross-reference):",
        "  1. FinanceEngine    — yfinance: works for ALL tickers (US + international)",
        "  2. FinnhubEngine    — US stocks ONLY (TSLA, AAPL, MSFT…). If ticker has a dot",
        "     suffix like .L .T .HK .PA, it will return a skip message — that is expected.",
        "  3. AlphaVantageEngine — EPS history, overview (EV/EBITDA, PEG) — US stocks best",
        "",
        "Recommended call order:",
        "  get_financial_summary(ticker)       — price, P/E, market cap",
        "  get_key_metrics(ticker)              — writes to SharedState for reports",
        "  get_income_statement(ticker)         — 3-year revenue/profit",
        "  get_cash_flow(ticker)                — FCF and capex",
        "  get_analyst_ratings(ticker)          — Finnhub consensus",
        "  get_price_targets(ticker)            — Finnhub price targets",
        "  get_earnings_surprise(ticker)        — Finnhub EPS beat/miss",
        "  get_earnings_history(ticker)         — AlphaVantage EPS history",
        "  get_insider_transactions(ticker)     — Finnhub insiders (optional)",
        "",
        "Output: ## Fundamental Analysis (valuation, growth, quality, red flags)",
        "",
        "NOTE: Do NOT trigger report generation — that is the user's choice.",
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
        "You are a market intelligence and news analyst.",
        "",
        "Data source strategy:",
        "  Decision tree based on ticker type:",
        "  US tickers (no dot, or .A/.B class only like BRK.A):",
        "    1. FinnhubEngine.get_company_news(ticker)      — primary, most reliable",
        "    2. AlphaVantageEngine.get_news_sentiment(ticker) — AI sentiment scores",
        "    3. NewsEngine.get_stock_news(ticker)           — supplement",
        "  Non-US tickers (HSBA.L, 8306.T, 0700.HK, AIR.PA …):",
        "    1. TavilyEngine.news_search('<company name> stock news') — primary",
        "    2. NewsEngine.get_stock_news(ticker)           — tries company name lookup",
        "    NOTE: Finnhub will return a skip message for non-US — ignore it, use Tavily.",
        "",
        "For HISTORICAL queries:",
        "  Use TavilyEngine.web_search('market event YEAR') — NOT real-time news tools.",
        "",
        "Output: ## News & Sentiment (top 5 headlines with date/source) + sentiment verdict",
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
        "SYNTHESIS (for analysis queries):",
        "  Use chain-of-thought to:",
        "    1. Identify any conflicts between macro / fundamentals / news",
        "    2. Weigh multi-source evidence (yfinance + Finnhub + AlphaVantage)",
        "    3. State BUY / HOLD / SELL with conviction (High/Med/Low)",
        "    4. Give target price range and time horizon",
        "    5. List 3-5 risks and 2-3 catalysts",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "OUTPUT FORMAT:",
        "  ## 🧠 Query Understanding   ← 1-2 lines from QueryAnalyst",
        "  ## 📊 [Relevant sections]",
        "  ## 🎯 Recommendation        ← BUY/HOLD/SELL + conviction + target",
        "  ## ⚠️  Key Risks             ← 3-5 bullets",
        "  ## 🚀 Catalysts             ← 2-3 bullets",
        "  (No report file section unless user asked for one)",
        "",
        "CONVERSATION: Maintain context. For follow-ups ('那风险呢?', 'what about Apple?'),",
        "use prior context without re-running QueryAnalyst.",
        "",
        "> ⚠️ Disclaimer: AI-generated analysis. Not financial advice.",
    ],
    markdown=True,
    show_members_responses=True,
    share_member_interactions=True,
    # NOTE: add_history_to_context intentionally omitted — no DB backend configured.
    # Conversation history is injected manually via full_query in app.py.
)
