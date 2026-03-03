"""
tests/conftest.py — Shared pytest fixtures and mock data for tool engine tests.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_FMP_RESPONSE = {
    "symbol": "AAPL",
    "retrieved_at": "2026-02-25T12:00:00",
    "source": "FMP grades-consensus",
    "raw_counts": {"strongBuy": 15, "buy": 8, "hold": 5, "sell": 1, "strongSell": 0},
    "raw_latest": {"symbol": "AAPL", "grade": "Buy"},
    "vote": {"bullish_pct": 79, "neutral_pct": 17, "bearish_pct": 4},
    "market_sentiment": "Strongly Bullish",
}

SAMPLE_POLYGON_RESPONSE = {
    "ticker": "NVDA",
    "retrieved_at": "2026-02-25T12:00:00",
    "source": "polygon.io",
    "window_days": 30,
    "items": [
        {
            "date": "2026-02-24",
            "source": "Bloomberg",
            "title": "NVIDIA Earnings Beat Expectations",
            "url": "https://example.com/1",
            "summary": "NVIDIA reported strong Q4 results.",
        },
        {
            "date": "2026-02-23",
            "source": "Reuters",
            "title": "AI Chip Demand Continues to Surge",
            "url": "https://example.com/2",
            "summary": "Data center demand remains robust.",
        },
    ],
}

SAMPLE_NEWSDATA_RESPONSE = {
    "query": "TSLA Tesla",
    "retrieved_at": "2026-02-25T12:00:00",
    "source": "newsdata.io",
    "window_days": 30,
    "items": [
        {
            "date": "2026-02-24",
            "source": "CNBC",
            "title": "Tesla Expands Robotaxi Fleet",
            "url": "https://example.com/3",
            "summary": "Tesla's autonomous driving program scales.",
        },
    ],
}

SAMPLE_TAVILY_SNIPPETS = {
    "symbol": "MSFT",
    "retrieved_at": "2026-02-25T12:00:00",
    "window_days": 30,
    "bank_snippets": [
        {
            "entity": "Goldman Sachs",
            "date": "Feb 20, 2026",
            "url": "https://example.com/gs",
            "takeaway": "Maintain Buy rating, AI cloud growth accelerating.",
            "tier": 1,
        },
    ],
    "media_snippets": [
        {
            "entity": "Bloomberg",
            "date": "Feb 22, 2026",
            "url": "https://example.com/bb",
            "takeaway": "Microsoft Azure gains market share.",
            "tier": 1,
        },
    ],
}

SAMPLE_EODHD_SIGNALS = {
    "ticker": "GOOG",
    "retrieved_at": "2026-02-25T12:00:00",
    "consensus": {
        "rating": "Buy",
        "target_price": 210.0,
        "distribution": {"strong_buy": 10, "buy": 12, "hold": 5, "sell": 1, "strong_sell": 0},
        "analyst_count": 28,
    },
    "bank_votes": [
        {"bank": "JPMorgan", "stance": "Overweight", "date": "2026-02-15",
         "source": "analyst_note", "title": "AI search monetization", "url": ""},
    ],
    "vote_tally": {"bullish": 22, "neutral": 5, "bearish": 1},
    "news_max_date": "2026-02-24",
    "confidence": "high",
}

SAMPLE_FINANCE_METRICS = {
    "Ticker": "AAPL",
    "Company": "Apple Inc.",
    "Price": 227.5,
    "Market Cap": "3.48T",
    "P/E (TTM)": 37.2,
    "Forward P/E": 29.1,
    "52-Week High": 245.0,
    "52-Week Low": 164.1,
}


@pytest.fixture
def fmp_json():
    return json.dumps(SAMPLE_FMP_RESPONSE)


@pytest.fixture
def polygon_json():
    return json.dumps(SAMPLE_POLYGON_RESPONSE)


@pytest.fixture
def newsdata_json():
    return json.dumps(SAMPLE_NEWSDATA_RESPONSE)


@pytest.fixture
def tavily_snippets_json():
    return json.dumps(SAMPLE_TAVILY_SNIPPETS)


@pytest.fixture
def eodhd_signals_json():
    return json.dumps(SAMPLE_EODHD_SIGNALS)


@pytest.fixture
def finance_metrics():
    return SAMPLE_FINANCE_METRICS.copy()


# ── CIO output samples for audit tests ──────────────────────────────────────

GOOD_CIO_OUTPUT = """\
**NVIDIA looks like the best-positioned AI compounder, but the valuation asks you to pay up front for tomorrow's growth.**

## Business & Position

NVIDIA's data center revenue grew 150% YoY to ＄22B in Q4. The Blackwell platform is
shipping and demand exceeds supply. Gross margins at 74% reflect pricing power.

## 🏛️ Institutional & Expert Consensus

**Consensus Vote:** 🟢 Bullish: 85% | 🟡 Neutral: 12% | 🔴 Bearish: 3%

**Wall Street Research:**
- **Goldman Sachs** (Feb 20): Maintain Buy, PT ＄180 — "Blackwell cycle extends growth runway"
- **JPMorgan** (Feb 18): Overweight, PT ＄175 — "Data center capex cycle still early innings"

## 📰 Latest Media News & Sentiment

Bloomberg reports NVIDIA's upcoming GTC conference is expected to showcase next-gen
Rubin architecture. WSJ notes hyperscaler capex budgets remain elevated through 2027.
Sentiment: Strongly positive.

---

This tends to fit growth investors who can tolerate 35x forward P/E and believe
the AI infrastructure cycle has 3-5 more years to run. If you need value, look elsewhere.

> ⚠️ Disclaimer: AI-generated analysis for educational purposes. Not financial advice.
"""

BAD_CIO_OUTPUT = """\
Certainly! Here is my analysis of NVIDIA.

NVIDIA is a great company. You should buy it right now for guaranteed returns.
The stock will definitely go to $500.

I recommend buying 100 shares immediately.
"""


@pytest.fixture
def good_cio_output():
    return GOOD_CIO_OUTPUT


@pytest.fixture
def bad_cio_output():
    return BAD_CIO_OUTPUT
