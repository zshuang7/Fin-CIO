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
        self.register(self.get_risk_metrics)

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

    def get_risk_metrics(self, ticker: str, period: str = "1y") -> str:
        """Compute quantitative risk metrics using pandas/numpy.

        Calculates moving averages, annualized volatility at multiple windows,
        max drawdown, and an approximate Sharpe ratio. Results are written to
        SharedState.risk_metrics for downstream use by CIO and ReportManager.

        Args:
            ticker: Stock ticker symbol, e.g. 'TSLA', '0700.HK'.
            period: yfinance period string (default '1y'). Use '2y' for
                    200-day MA to have enough history.

        Returns:
            Formatted text summary of risk metrics.

        Note for Derivatives (Black-Scholes ready):
            The annualized volatility (Vol_252d) computed here is the key input
            for Black-Scholes option pricing (sigma). When derivatives pricing
            is added, Delta / Gamma / Vega can be computed from:
              - S = current price (from get_key_metrics)
              - sigma = Vol_252d (from this function)
              - r = risk-free rate (from Alpha Vantage / FRED)
              - T = time to expiry
              - K = strike price
            See: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
        """
        import numpy as np
        import pandas as pd

        try:
            tk = _yf().Ticker(ticker)
            hist: pd.DataFrame = tk.history(period=period, auto_adjust=True)

            if hist.empty or len(hist) < 20:
                return f"Insufficient price history for {ticker} (need ≥20 trading days)."

            close: pd.Series = hist["Close"]
            daily_returns: pd.Series = close.pct_change().dropna()

            # ── Moving Averages ──────────────────────────────────────────
            ma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
            ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # ── Annualized Volatility (σ√252) at different windows ───────
            sqrt_252 = float(np.sqrt(252))
            vol_30d = float(daily_returns.tail(30).std() * sqrt_252) if len(daily_returns) >= 30 else None
            vol_60d = float(daily_returns.tail(60).std() * sqrt_252) if len(daily_returns) >= 60 else None
            vol_252d = float(daily_returns.std() * sqrt_252)

            # ── Max Drawdown ─────────────────────────────────────────────
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.cummax()
            drawdowns = (cumulative - running_max) / running_max
            max_dd = float(drawdowns.min())
            max_dd_date = str(drawdowns.idxmin().date()) if not drawdowns.empty else "N/A"

            # ── Approximate Sharpe (assuming risk-free ≈ 4.5% for USD) ───
            # Note: replace with actual risk-free rate from FRED / Alpha Vantage
            rf_annual = 0.045
            mean_annual = float(daily_returns.mean() * 252)
            sharpe = (mean_annual - rf_annual) / vol_252d if vol_252d > 0 else None

            current_price = float(close.iloc[-1])

            metrics = {
                "ticker": ticker.upper(),
                "period": period,
                "current_price": round(current_price, 2),
                "MA_20": round(float(ma_20), 2) if ma_20 is not None else "N/A (< 20 days)",
                "MA_50": round(float(ma_50), 2) if ma_50 is not None else "N/A (< 50 days)",
                "MA_200": round(float(ma_200), 2) if ma_200 is not None else "N/A (< 200 days)",
                "Vol_30d": f"{vol_30d:.1%}" if vol_30d is not None else "N/A",
                "Vol_60d": f"{vol_60d:.1%}" if vol_60d is not None else "N/A",
                "Vol_252d": f"{vol_252d:.1%}",
                "Max_Drawdown": f"{max_dd:.1%}",
                "Max_Drawdown_Date": max_dd_date,
                "Sharpe_Approx": round(sharpe, 2) if sharpe is not None else "N/A",
                "trading_days": len(daily_returns),
            }

            # Write to SharedState for Excel export / CIO synthesis
            get_state().risk_metrics = metrics

            # ── Format output ────────────────────────────────────────────
            price_vs_ma = ""
            if ma_50 is not None:
                pct_vs_50 = (current_price / float(ma_50) - 1) * 100
                price_vs_ma = f"  Price vs 50-MA  : {pct_vs_50:+.1f}%\n"
            if ma_200 is not None:
                pct_vs_200 = (current_price / float(ma_200) - 1) * 100
                price_vs_ma += f"  Price vs 200-MA : {pct_vs_200:+.1f}%"

            lines = [
                f"{'=' * 55}",
                f"  Risk Metrics — {ticker.upper()} ({period})",
                f"{'=' * 55}",
                f"  Current Price   : {current_price:.2f}",
                f"  20-day MA       : {metrics['MA_20']}",
                f"  50-day MA       : {metrics['MA_50']}",
                f"  200-day MA      : {metrics['MA_200']}",
                price_vs_ma,
                f"  Vol (30d ann.)  : {metrics['Vol_30d']}",
                f"  Vol (60d ann.)  : {metrics['Vol_60d']}",
                f"  Vol (252d ann.) : {metrics['Vol_252d']}  ← Black-Scholes σ input",
                f"  Max Drawdown    : {metrics['Max_Drawdown']} (on {max_dd_date})",
                f"  Sharpe (approx) : {metrics['Sharpe_Approx']}  (rf=4.5%)",
                f"  Trading Days    : {metrics['trading_days']}",
            ]
            return "\n".join(l for l in lines if l)

        except Exception as e:
            logger.error(f"[finance_engine] get_risk_metrics({ticker}): {e}")
            return f"Error computing risk metrics for {ticker}: {e}"

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
