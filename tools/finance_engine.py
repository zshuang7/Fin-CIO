"""
tools/finance_engine.py — yfinance wrapper that extends BaseTool.

Data written to SharedState so every agent and the ReportManager
can access it without redundant API calls.

════════════════════════════════════════════════════════════════════
在此处添加新的 API Key 和工具函数（例如 Alpha Vantage / FRED）:

  import os
  AV_KEY  = os.getenv("ALPHA_VANTAGE_API_KEY", "")
  FRED_KEY = os.getenv("FRED_API_KEY", "")

  def get_earnings_surprise(self, ticker: str) -> str:
      # 在此调用 Alpha Vantage earnings API
      url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={AV_KEY}"
      ...
════════════════════════════════════════════════════════════════════
"""

from agno.utils.log import logger

from tools.base_tool import BaseTool
from state import get_state


def _yf():
    """Lazy-import yfinance so it is only loaded when a finance tool is actually called."""
    import yfinance as yf  # noqa: PLC0415
    return yf


class FinanceEngine(BaseTool):
    tool_name = "finance_engine"
    tool_description = (
        "Fetches real-time and historical financial data via yfinance: "
        "key metrics, income statements, balance sheets, and cash flows."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        self.register(self.get_financial_summary)
        self.register(self.get_income_statement)
        self.register(self.get_balance_sheet)
        self.register(self.get_cash_flow)
        self.register(self.get_key_metrics)

        # ════════════════════════════════════════════════════════════════
        # 在此处添加新的 API Key（yfinance 无需 key，其他数据源在此配置）
        # 例如: self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        # ════════════════════════════════════════════════════════════════

    # ── Public tool functions ────────────────────────────────────────────

    @staticmethod
    def _freshness_tag(ts_unix: int | float | None = None,
                       date_str: str | None = None) -> str:
        """
        Build a human-readable data-freshness note.
        Pass either a Unix timestamp (from yfinance) or an ISO date string.
        Returns a coloured tag the CIO can quote in its output.
        """
        from datetime import datetime as _dt
        try:
            if ts_unix:
                data_date = _dt.fromtimestamp(int(ts_unix))
            elif date_str:
                data_date = _dt.fromisoformat(str(date_str)[:10])
            else:
                return "[DATA DATE: Unknown — verify from official filings]"

            days_ago = (_dt.now() - data_date).days
            label = data_date.strftime("%Y-%m-%d")
            if days_ago > 365:
                flag = "⚠️ STALE >1yr"
            elif days_ago > 180:
                flag = "⚠️ STALE >6mo"
            else:
                flag = "✓ Recent"
            return f"[DATA AS OF: {label}  ({days_ago}d ago)  {flag}]"
        except Exception:
            return "[DATA DATE: Unknown]"

    def get_financial_summary(self, ticker: str) -> str:
        """
        Returns a formatted text summary of the stock's key valuation
        and fundamental statistics.  Results are also written to SharedState.

        Args:
            ticker: Stock ticker symbol, e.g. 'TSLA'.
        """
        try:
            info = _yf().Ticker(ticker).info
            state = get_state()
            state.ticker = ticker.upper()
            state.company_name = info.get("longName", ticker.upper())

            div_yield = info.get("dividendYield")
            div_str = f"{round(div_yield * 100, 2)}%" if div_yield else "N/A"

            # ── Data freshness ───────────────────────────────────────────────
            freshness = self._freshness_tag(
                ts_unix=info.get("mostRecentQuarter") or info.get("earningsTimestamp"),
            )

            lines = [
                f"{'=' * 55}",
                f"  {state.company_name} ({ticker.upper()})",
                f"  {freshness}",
                f"{'=' * 55}",
                f"  Current Price   : {info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
                f"  Market Cap      : {self._fmt_large(info.get('marketCap'))}",
                f"  P/E (TTM)       : {info.get('trailingPE', 'N/A')}",
                f"  Forward P/E     : {info.get('forwardPE', 'N/A')}",
                f"  EPS (TTM)       : {info.get('trailingEps', 'N/A')}",
                f"  Dividend Yield  : {div_str}",
                f"  52-Week High    : {info.get('fiftyTwoWeekHigh', 'N/A')}",
                f"  52-Week Low     : {info.get('fiftyTwoWeekLow', 'N/A')}",
                f"  Beta            : {info.get('beta', 'N/A')}",
                f"  Analyst Target  : {info.get('targetMeanPrice', 'N/A')}",
                f"  Recommendation  : {str(info.get('recommendationKey', 'N/A')).upper()}",
                f"  Sector          : {info.get('sector', 'N/A')}",
                f"  Industry        : {info.get('industry', 'N/A')}",
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[finance_engine] get_financial_summary({ticker}): {e}")
            return f"Error fetching summary for {ticker}: {e}"

    def get_income_statement(self, ticker: str, years: int = 3) -> str:
        """
        Returns the last N years of annual income statement data.

        Args:
            ticker: Stock ticker symbol.
            years:  Number of fiscal years (default 3).
        """
        try:
            fin = _yf().Ticker(ticker).financials
            if fin is None or fin.empty:
                return f"No income statement data for {ticker}."
            subset = fin.iloc[:, :years]
            want = ["Total Revenue", "Gross Profit", "Operating Income",
                    "Net Income", "EBITDA"]
            rows = subset.loc[[r for r in want if r in subset.index]]

            # ── Data freshness: most recent column = most recent fiscal year ──
            most_recent = str(subset.columns[0])[:10] if not subset.empty else None
            freshness = self._freshness_tag(date_str=most_recent)

            # Write to SharedState
            get_state().income_data = rows.to_dict()
            header = f"Income Statement — {ticker.upper()} (last {years} yrs)  {freshness}"
            return f"{header}\n{rows.to_string()}"
        except Exception as e:
            logger.error(f"[finance_engine] get_income_statement({ticker}): {e}")
            return f"Error: {e}"

    def get_balance_sheet(self, ticker: str, years: int = 3) -> str:
        """
        Returns the last N years of annual balance sheet data.

        Args:
            ticker: Stock ticker symbol.
            years:  Number of fiscal years (default 3).
        """
        try:
            bs = _yf().Ticker(ticker).balance_sheet
            if bs is None or bs.empty:
                return f"No balance sheet data for {ticker}."
            subset = bs.iloc[:, :years]
            want = ["Total Assets", "Total Liabilities Net Minority Interest",
                    "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents"]
            rows = subset.loc[[r for r in want if r in subset.index]]

            get_state().balance_sheet_data = rows.to_dict()
            return f"Balance Sheet — {ticker.upper()} (last {years} yrs):\n{rows.to_string()}"
        except Exception as e:
            logger.error(f"[finance_engine] get_balance_sheet({ticker}): {e}")
            return f"Error: {e}"

    def get_cash_flow(self, ticker: str, years: int = 3) -> str:
        """
        Returns the last N years of annual cash flow statement data.

        Args:
            ticker: Stock ticker symbol.
            years:  Number of fiscal years (default 3).
        """
        try:
            cf = _yf().Ticker(ticker).cashflow
            if cf is None or cf.empty:
                return f"No cash flow data for {ticker}."
            subset = cf.iloc[:, :years]
            want = ["Operating Cash Flow", "Free Cash Flow",
                    "Capital Expenditure", "Investing Cash Flow"]
            rows = subset.loc[[r for r in want if r in subset.index]]

            get_state().cashflow_data = rows.to_dict()
            return f"Cash Flow — {ticker.upper()} (last {years} yrs):\n{rows.to_string()}"
        except Exception as e:
            logger.error(f"[finance_engine] get_cash_flow({ticker}): {e}")
            return f"Error: {e}"

    def get_key_metrics(self, ticker: str) -> dict:
        """
        Returns a flat dict of key metrics and writes them to SharedState.
        Used by ReportManager when building the Excel/PDF report.

        Args:
            ticker: Stock ticker symbol.
        """
        try:
            info = _yf().Ticker(ticker).info
            div_yield = info.get("dividendYield")
            profit_margin = info.get("profitMargins")

            metrics = {
                "Ticker":           ticker.upper(),
                "Company":          info.get("longName", "N/A"),
                "Price":            info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "Market Cap":       self._fmt_large(info.get("marketCap")),
                "P/E (TTM)":        info.get("trailingPE", "N/A"),
                "Forward P/E":      info.get("forwardPE", "N/A"),
                "EPS (TTM)":        info.get("trailingEps", "N/A"),
                "Dividend Yield %": round(div_yield * 100, 2) if div_yield else "N/A",
                "52W High":         info.get("fiftyTwoWeekHigh", "N/A"),
                "52W Low":          info.get("fiftyTwoWeekLow", "N/A"),
                "Beta":             info.get("beta", "N/A"),
                "Revenue (TTM)":    self._fmt_large(info.get("totalRevenue")),
                "Profit Margin %":  round(profit_margin * 100, 2) if profit_margin else "N/A",
                "Analyst Target":   info.get("targetMeanPrice", "N/A"),
                "Consensus":        str(info.get("recommendationKey", "N/A")).upper(),
                "Sector":           info.get("sector", "N/A"),
                "Industry":         info.get("industry", "N/A"),
            }

            # ════════════════════════════════════════════════════════════
            # 在此处添加更多指标（来自其他 API，如 Alpha Vantage / FRED）
            # 例如:
            #   metrics["EV/EBITDA"] = self._fetch_ev_ebitda(ticker)
            #   metrics["FCF Yield"] = self._fetch_fcf_yield(ticker)
            # ════════════════════════════════════════════════════════════

            get_state().raw_metrics = metrics
            return metrics
        except Exception as e:
            logger.error(f"[finance_engine] get_key_metrics({ticker}): {e}")
            return {"error": str(e)}

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _fmt_large(value) -> str:
        """Format large numbers as $1.23T, $456.7B, $12.3M, etc."""
        if value is None:
            return "N/A"
        try:
            n = float(value)
            if abs(n) >= 1e12:
                return f"${n/1e12:.2f}T"
            if abs(n) >= 1e9:
                return f"${n/1e9:.2f}B"
            if abs(n) >= 1e6:
                return f"${n/1e6:.2f}M"
            return f"${n:,.0f}"
        except (TypeError, ValueError):
            return str(value)
