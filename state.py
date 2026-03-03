"""
state.py — Global SharedState singleton.

Every agent and tool imports get_state() to read or write analysis data.
This ensures a single source of truth that flows from data-collection
agents → CIO synthesis → ReportManager export.

No imports from tools/ or agents/ here to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SharedState:
    # ── Query context ────────────────────────────────────────────────────
    ticker: str = ""
    company_name: str = ""
    query: str = ""
    timestamp: str = ""

    # ── Agent outputs (populated by sub-agents during the run) ───────────
    macro_analysis: str = ""        # MacroAgent findings
    sector_analysis: str = ""       # SectorAgent findings
    company_summary: str = ""       # CompanyAgent fundamentals narrative
    news_headlines: list = field(default_factory=list)   # list of str
    news_sentiment: str = ""        # "Bullish" | "Neutral" | "Bearish"

    # ── Raw tabular data (used by ReportManager for Excel/PDF) ───────────
    raw_metrics: dict = field(default_factory=dict)
    income_data: dict = field(default_factory=dict)
    cashflow_data: dict = field(default_factory=dict)
    balance_sheet_data: dict = field(default_factory=dict)

    # ── Risk metrics (computed by FinanceEngine.get_risk_metrics) ────────
    # Keys: MA_20, MA_50, MA_200, Vol_30d, Vol_60d, Vol_252d,
    #        Max_Drawdown, Max_Drawdown_Period, Sharpe_Approx, ticker, period
    # Note for Derivatives: these feed into Delta/Gamma/Vega calculations
    # when Black-Scholes pricing is added (see finance_engine.py).
    risk_metrics: dict = field(default_factory=dict)

    # ── CIO synthesis (written by the CIO / Team coordinator) ───────────
    cio_reasoning: str = ""         # R1 chain-of-thought text
    recommendation: str = ""        # "BUY" | "HOLD" | "SELL"
    target_price: str = ""
    risk_factors: list = field(default_factory=list)
    catalysts: list = field(default_factory=list)
    time_horizon: str = ""          # "Short-term" | "Medium-term" | "Long-term"
    conviction: str = ""            # "High" | "Medium" | "Low"

    # ── Saved report paths ───────────────────────────────────────────────
    excel_path: str = ""
    pdf_path: str = ""

    # ────────────────────────────────────────────────────────────────────
    # 在此处添加新的数据字段（对应新 API 数据源）
    # 例如:
    #   fred_macro_data: dict = field(default_factory=dict)
    #   alpha_vantage_earnings: dict = field(default_factory=dict)
    #   bloomberg_estimates: dict = field(default_factory=dict)
    # ────────────────────────────────────────────────────────────────────

    def reset(self, ticker: str = "", query: str = "") -> None:
        """Clear all fields and start a fresh analysis session."""
        self.__init__()
        self.ticker = ticker.upper().strip() if ticker else ""
        self.query = query
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def is_populated(self) -> bool:
        """True once at least one agent has written data."""
        return bool(self.ticker and (self.raw_metrics or self.company_summary))

    def to_excel_sheets(self) -> dict:
        """Returns a dict of {sheet_name: data} for Excel export."""
        sheets: dict[str, Any] = {}
        if self.raw_metrics:
            sheets["Key Metrics"] = self.raw_metrics
        if self.income_data:
            sheets["Income Statement"] = self.income_data
        if self.cashflow_data:
            sheets["Cash Flow"] = self.cashflow_data
        if self.balance_sheet_data:
            sheets["Balance Sheet"] = self.balance_sheet_data
        if self.risk_metrics:
            sheets["Risk Metrics"] = self.risk_metrics
        if self.news_headlines:
            sheets["News"] = [{"Headline": h} for h in self.news_headlines]
        return sheets

    def summary_text(self) -> str:
        """One-line summary for logging."""
        return (
            f"[{self.timestamp}] {self.ticker} | "
            f"Rec={self.recommendation} | "
            f"Conviction={self.conviction}"
        )


# ── Global singleton ─────────────────────────────────────────────────────────
_state = SharedState()


def get_state() -> SharedState:
    """Return the global SharedState instance."""
    return _state
