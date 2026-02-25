"""
tools/alpha_vantage_engine.py — Alpha Vantage API wrapper.

Provides AI-scored news sentiment, detailed earnings history,
comprehensive fundamental overviews, and macro economic indicators
(GDP, CPI, inflation, unemployment) — data not available in yfinance.

════════════════════════════════════════════════════════════════════
在此处添加新的 API Key 和工具函数:
  ALPHA_VANTAGE_API_KEY 已在 .env 配置。
  免费版限速: 25次/天 (premium: 75次/分钟)。
  升级: https://www.alphavantage.co/premium/

  可扩展功能:
  - get_technical_indicator(): RSI, MACD, Bollinger Bands
  - get_forex_rate(): 汇率数据
  - get_commodity(): 大宗商品 (油价, 黄金等)
════════════════════════════════════════════════════════════════════
"""

import os
import time

import requests
from agno.utils.log import logger

from tools.base_tool import BaseTool
from state import get_state

_BASE = "https://www.alphavantage.co/query"
_RATE_DELAY = 12  # seconds between calls (free tier: 5 calls/min)


class AlphaVantageEngine(BaseTool):
    tool_name = "alpha_vantage_engine"
    tool_description = (
        "Fetches AI-scored news sentiment, earnings history, company overview, "
        "and macro economic indicators (GDP, CPI, unemployment) from Alpha Vantage."
    )

    def __init__(self):
        super().__init__(name=self.tool_name)
        # ════════════════════════════════════════════════════════════════
        # 在此处添加新的 API Key
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        # self.polygon_key  = os.getenv("POLYGON_API_KEY", "")  # 未来扩展
        # self.fred_key     = os.getenv("FRED_API_KEY", "")     # 未来扩展
        # ════════════════════════════════════════════════════════════════
        if not self.api_key:
            logger.warning("[alpha_vantage_engine] ALPHA_VANTAGE_API_KEY not set in .env")

        self._last_call = 0  # track rate limiting
        self.register(self.get_news_sentiment)
        self.register(self.get_company_overview)
        self.register(self.get_earnings_history)
        self.register(self.get_macro_indicator)

    def _get(self, params: dict) -> dict | None:
        """Rate-limited GET request to Alpha Vantage."""
        if not self.api_key:
            return None
        # Respect free tier rate limit
        elapsed = time.time() - self._last_call
        if elapsed < _RATE_DELAY:
            time.sleep(_RATE_DELAY - elapsed)
        try:
            params["apikey"] = self.api_key
            resp = requests.get(_BASE, params=params, timeout=15)
            resp.raise_for_status()
            self._last_call = time.time()
            data = resp.json()
            # AV returns {"Information": "..."} on rate limit
            if "Information" in data or "Note" in data:
                msg = data.get("Information") or data.get("Note", "")
                logger.warning(f"[alpha_vantage_engine] API limit: {msg[:120]}")
                return None
            return data
        except requests.RequestException as e:
            logger.error(f"[alpha_vantage_engine] {params.get('function')}: {e}")
            return None

    # ── News Sentiment (with NLP scores) ─────────────────────────────────────

    def get_news_sentiment(self, ticker: str, limit: int = 8) -> str:
        """
        Fetches news articles with AI-generated sentiment scores for a stock.
        Each article gets an Overall Sentiment (Bullish/Bearish/Neutral)
        and a relevance score to the specific ticker.

        Args:
            ticker: Stock ticker, e.g. 'TSLA'.
            limit:  Number of articles to return (default 8).
        """
        data = self._get({
            "function":    "NEWS_SENTIMENT",
            "tickers":     ticker,
            "limit":       limit,
            "sort":        "LATEST",
        })
        if not data or "feed" not in data:
            return f"[AlphaVantage] No sentiment data for {ticker}."

        articles = data["feed"][:limit]
        overall_scores = []
        lines = [f"News Sentiment (Alpha Vantage AI) — {ticker.upper()}:", ""]

        for a in articles:
            title       = a.get("title", "")[:100]
            source      = a.get("source", "N/A")
            time_pub    = a.get("time_published", "")[:8]
            date_fmt    = f"{time_pub[:4]}-{time_pub[4:6]}-{time_pub[6:]}" if len(time_pub) >= 8 else "N/A"
            overall_sent= a.get("overall_sentiment_label", "Neutral")
            overall_score = a.get("overall_sentiment_score", 0)

            # Find this ticker's specific sentiment in the article
            ticker_sentiment = "N/A"
            ticker_relevance = 0
            for ts in a.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    ticker_sentiment = ts.get("ticker_sentiment_label", "Neutral")
                    ticker_relevance = float(ts.get("relevance_score", 0))
                    overall_scores.append(float(ts.get("ticker_sentiment_score", 0)))
                    break

            rel_bar = "█" * int(ticker_relevance * 10)
            lines.append(f"  [{date_fmt}] {title}")
            lines.append(
                f"    Source: {source} | Sentiment: {ticker_sentiment} | "
                f"Relevance: {rel_bar} {ticker_relevance:.2f}"
            )

        # Aggregate sentiment
        if overall_scores:
            avg = sum(overall_scores) / len(overall_scores)
            if avg > 0.15:
                agg = f"BULLISH (score: +{avg:.2f})"
            elif avg < -0.15:
                agg = f"BEARISH (score: {avg:.2f})"
            else:
                agg = f"NEUTRAL (score: {avg:.2f})"
            lines.insert(1, f"  Aggregated Sentiment: {agg}")
            lines.insert(2, "")

            # Update SharedState
            get_state().news_sentiment = agg.split()[0]

        return "\n".join(lines)

    # ── Company Overview ──────────────────────────────────────────────────────

    def get_company_overview(self, ticker: str) -> str:
        """
        Returns a comprehensive fundamental overview including description,
        sector, valuation multiples, profitability, and dividend data.
        Complements yfinance with additional fields like PEG ratio, EV/EBITDA.

        Args:
            ticker: Stock ticker.
        """
        data = self._get({"function": "OVERVIEW", "symbol": ticker})
        if not data or not data.get("Symbol"):
            return f"[AlphaVantage] No overview data for {ticker}."

        def _v(key: str) -> str:
            return data.get(key, "N/A") or "N/A"

        lines = [
            f"Company Overview (Alpha Vantage) — {_v('Name')} ({ticker.upper()})",
            f"  Description : {_v('Description')[:200]}...",
            f"  Sector      : {_v('Sector')}  |  Industry: {_v('Industry')}",
            f"  Exchange    : {_v('Exchange')}  |  Currency: {_v('Currency')}",
            "",
            "  Valuation:",
            f"    P/E (TTM)    : {_v('TrailingPE')}",
            f"    Forward P/E  : {_v('ForwardPE')}",
            f"    PEG Ratio    : {_v('PEGRatio')}",
            f"    P/B Ratio    : {_v('PriceToBookRatio')}",
            f"    P/S (TTM)    : {_v('PriceToSalesRatioTTM')}",
            f"    EV/EBITDA    : {_v('EVToEBITDA')}",
            f"    EV/Revenue   : {_v('EVToRevenue')}",
            "",
            "  Profitability:",
            f"    Profit Margin: {_v('ProfitMargin')}",
            f"    Op. Margin   : {_v('OperatingMarginTTM')}",
            f"    ROE          : {_v('ReturnOnEquityTTM')}",
            f"    ROA          : {_v('ReturnOnAssetsTTM')}",
            "",
            "  Growth & Dividends:",
            f"    Revenue Growth (YoY): {_v('RevenueGrowthYOY') if 'RevenueGrowthYOY' in data else _v('QuarterlyRevenueGrowthYOY')}",
            f"    EPS Growth (YoY)    : {_v('QuarterlyEarningsGrowthYOY')}",
            f"    Dividend Yield      : {_v('DividendYield')}",
            f"    Dividend/Share      : {_v('DividendPerShare')}",
            "",
            "  Size & Debt:",
            f"    Market Cap      : {_v('MarketCapitalization')}",
            f"    Book Value/Share: {_v('BookValue')}",
            f"    Beta            : {_v('Beta')}",
            f"    52W High / Low  : {_v('52WeekHigh')} / {_v('52WeekLow')}",
            f"    Analyst Target  : {_v('AnalystTargetPrice')}",
        ]
        return "\n".join(lines)

    # ── Earnings History ──────────────────────────────────────────────────────

    def get_earnings_history(self, ticker: str, quarters: int = 8) -> str:
        """
        Returns the last N quarters of EPS (actual vs estimate) with
        beat/miss labels and annual EPS trend.

        Args:
            ticker:   Stock ticker.
            quarters: Number of quarterly periods (default 8).
        """
        data = self._get({"function": "EARNINGS", "symbol": ticker})
        if not data:
            return f"[AlphaVantage] No earnings data for {ticker}."

        quarterly = data.get("quarterlyEarnings", [])[:quarters]
        annual    = data.get("annualEarnings", [])[:4]

        lines = [f"Earnings History (Alpha Vantage) — {ticker.upper()}:", ""]

        if quarterly:
            lines.append("  Quarterly EPS:")
            beats = 0
            for q in quarterly:
                date    = q.get("fiscalDateEnding", "N/A")[:7]
                actual  = q.get("reportedEPS", "N/A")
                est     = q.get("estimatedEPS", "N/A")
                surprise= q.get("surprisePercentage", "N/A")
                try:
                    surprise_f = float(surprise)
                    emoji = "✅" if surprise_f > 0 else "❌"
                    surprise_str = f"{surprise_f:+.1f}%"
                    if surprise_f > 0:
                        beats += 1
                except (TypeError, ValueError):
                    emoji = "—"
                    surprise_str = "N/A"
                lines.append(
                    f"    {emoji} {date}  Actual:{actual}  Est:{est}  "
                    f"Surprise:{surprise_str}"
                )
            if quarterly:
                lines.append(f"  => Beat rate: {beats}/{len(quarterly)} quarters ({round(beats/len(quarterly)*100)}%)")

        if annual:
            lines.append("\n  Annual EPS Trend:")
            for a in annual:
                lines.append(f"    {a.get('fiscalDateEnding','')[:4]}  EPS: {a.get('reportedEPS','N/A')}")

        return "\n".join(lines)

    # ── Macro Economic Indicators ─────────────────────────────────────────────

    def get_macro_indicator(self, indicator: str = "GDP") -> str:
        """
        Fetches US macro economic data. Useful for macro context in analysis.
        Available indicators:
          GDP, REAL_GDP, CPI, INFLATION, RETAIL_SALES,
          UNEMPLOYMENT, NONFARM_PAYROLL, FEDERAL_FUNDS_RATE,
          TREASURY_YIELD, CONSUMER_SENTIMENT

        Args:
            indicator: Alpha Vantage function name (default 'REAL_GDP').
        """
        # Map friendly names to AV function names
        name_map = {
            "GDP":          "REAL_GDP",
            "REAL_GDP":     "REAL_GDP",
            "CPI":          "CPI",
            "INFLATION":    "INFLATION",
            "UNEMPLOYMENT": "UNEMPLOYMENT",
            "PAYROLL":      "NONFARM_PAYROLL",
            "FED_RATE":     "FEDERAL_FUNDS_RATE",
            "TREASURY":     "TREASURY_YIELD",
            "SENTIMENT":    "CONSUMER_SENTIMENT",
            "RETAIL":       "RETAIL_SALES",
        }
        fn = name_map.get(indicator.upper(), indicator.upper())
        data = self._get({"function": fn})
        if not data or "data" not in data:
            return f"[AlphaVantage] No data for indicator '{indicator}'."

        points = data["data"][:6]
        unit   = data.get("unit", "")
        lines  = [f"Macro Indicator: {fn} ({unit})"]
        for p in points:
            lines.append(f"  {p.get('date','')[:7]}  {p.get('value','N/A')}")

        # Simple trend
        try:
            vals = [float(p["value"]) for p in points if p.get("value")]
            if len(vals) >= 2:
                trend = "Rising" if vals[0] > vals[1] else "Falling"
                lines.append(f"  => Recent trend: {trend}")
        except (ValueError, TypeError):
            pass

        return "\n".join(lines)
