"""
tests/test_tools.py — Unit tests for tool engines and audit system.

Covers:
  1. JSON schema conformance for structured tool outputs (Pydantic validation)
  2. Ticker normalization edge cases
  3. Structural audit pre-check (no LLM needed)
  4. Error handling for missing API keys / invalid tickers
  5. DSPy router classification logic
"""

import json
import os
import sys
import re

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.schemas import (
    AuditVerdict,
    DimensionScore,
    FmpConsensusOutput,
    PolygonNewsOutput,
    NewsDataOutput,
    TavilySnippetsOutput,
    EodhdSignalsOutput,
    FinanceKeyMetrics,
    run_structural_check,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: JSON schema validation for tool outputs
# ═══════════════════════════════════════════════════════════════════════════════


class TestFmpSchema:
    def test_valid_fmp_consensus(self, fmp_json):
        data = json.loads(fmp_json)
        result = FmpConsensusOutput(**data)
        assert result.symbol == "AAPL"
        assert result.source == "FMP grades-consensus"
        assert result.vote["bullish_pct"] + result.vote["neutral_pct"] + result.vote["bearish_pct"] == 100

    def test_fmp_missing_symbol_fails(self):
        with pytest.raises(Exception):
            FmpConsensusOutput(retrieved_at="2026-01-01", source="FMP")

    def test_fmp_extra_fields_allowed(self, fmp_json):
        data = json.loads(fmp_json)
        data["extra_field"] = "should not fail"
        result = FmpConsensusOutput(**data)
        assert result.symbol == "AAPL"


class TestPolygonSchema:
    def test_valid_polygon_news(self, polygon_json):
        data = json.loads(polygon_json)
        result = PolygonNewsOutput(**data)
        assert result.ticker == "NVDA"
        assert len(result.items) == 2
        assert result.items[0].source == "Bloomberg"

    def test_empty_items_ok(self):
        data = {
            "ticker": "XYZ",
            "retrieved_at": "2026-01-01",
            "source": "polygon.io",
            "window_days": 30,
            "items": [],
        }
        result = PolygonNewsOutput(**data)
        assert result.ticker == "XYZ"
        assert len(result.items) == 0


class TestNewsDataSchema:
    def test_valid_newsdata(self, newsdata_json):
        data = json.loads(newsdata_json)
        result = NewsDataOutput(**data)
        assert result.query == "TSLA Tesla"
        assert len(result.items) == 1
        assert result.items[0].source == "CNBC"


class TestTavilySnippetsSchema:
    def test_valid_snippets(self, tavily_snippets_json):
        data = json.loads(tavily_snippets_json)
        result = TavilySnippetsOutput(**data)
        assert result.symbol == "MSFT"
        assert len(result.bank_snippets) == 1
        assert result.bank_snippets[0].entity == "Goldman Sachs"
        assert result.bank_snippets[0].tier == 1

    def test_tier_range_enforced(self):
        with pytest.raises(Exception):
            from audit.schemas import InstitutionSnippet
            InstitutionSnippet(entity="Bad", tier=5)


class TestEodhdSignalsSchema:
    def test_valid_signals(self, eodhd_signals_json):
        data = json.loads(eodhd_signals_json)
        result = EodhdSignalsOutput(**data)
        assert result.ticker == "GOOG"
        assert result.consensus["rating"] == "Buy"
        assert len(result.bank_votes) == 1


class TestFinanceMetricsSchema:
    def test_valid_metrics(self, finance_metrics):
        result = FinanceKeyMetrics(**finance_metrics)
        assert result.Ticker == "AAPL"
        assert result.Price == 227.5

    def test_string_price_ok(self):
        result = FinanceKeyMetrics(Ticker="BAD", Company="Test", Price="N/A")
        assert result.Ticker == "BAD"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Structural audit pre-check
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuralCheck:
    def test_good_output_passes(self, good_cio_output):
        result = run_structural_check(good_cio_output, "standard")
        assert result.has_disclaimer is True
        assert result.has_forbidden_opener is False
        assert len(result.missing_sections) == 0
        assert result.passed is True
        assert result.word_count > 50

    def test_bad_output_fails(self, bad_cio_output):
        result = run_structural_check(bad_cio_output, "standard")
        assert result.has_disclaimer is False
        assert result.has_forbidden_opener is True
        assert result.passed is False
        assert len(result.issues) >= 2

    def test_fast_level_no_sections_required(self):
        text = "Quick take: NVDA is strong. Buy the dip.\n> Not financial advice."
        result = run_structural_check(text, "fast")
        assert len(result.missing_sections) == 0

    def test_master_requires_macro(self):
        text = (
            "Opening take.\n"
            "## 🏛️ Institutional & Expert Consensus\ndata\n"
            "## 📰 Latest Media News\ndata\n"
            "> Not financial advice."
        )
        result = run_structural_check(text, "master")
        assert "Macro" in result.missing_sections

    def test_deep_dive_requires_scenario(self):
        text = (
            "Opening take.\n"
            "## 🏛️ Institutional & Expert Consensus\ndata\n"
            "## 📰 Latest Media News\ndata\n"
            "## Macro Context\ndata\n"
            "> Not financial advice."
        )
        result = run_structural_check(text, "deep_dive")
        assert "Scenario" in result.missing_sections

    def test_empty_input(self):
        result = run_structural_check("", "standard")
        assert result.passed is False
        assert result.word_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: AuditVerdict model validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditVerdictModel:
    def test_valid_verdict(self):
        v = AuditVerdict(
            overall_grade="A",
            overall_score=92,
            factuality=DimensionScore(score=23, flag="pass", reason="All facts verified."),
            compliance=DimensionScore(score=25, flag="pass", reason="Disclaimer present."),
            logic=DimensionScore(score=22, flag="pass", reason="Consistent reasoning."),
            completeness=DimensionScore(score=22, flag="pass", reason="All sections present."),
        )
        assert v.overall_grade == "A"
        assert v.overall_score == 92

    def test_score_out_of_range_fails(self):
        with pytest.raises(Exception):
            DimensionScore(score=30, flag="pass", reason="Over max")

    def test_negative_score_fails(self):
        with pytest.raises(Exception):
            DimensionScore(score=-1, flag="fail", reason="Negative")

    def test_invalid_grade_fails(self):
        with pytest.raises(Exception):
            AuditVerdict(
                overall_grade="Z",
                overall_score=50,
                factuality=DimensionScore(score=10, flag="warn", reason="t"),
                compliance=DimensionScore(score=10, flag="warn", reason="t"),
                logic=DimensionScore(score=10, flag="warn", reason="t"),
                completeness=DimensionScore(score=10, flag="warn", reason="t"),
            )

    def test_grade_normalization(self):
        v = AuditVerdict(
            overall_grade="  b ",
            overall_score=75,
            factuality=DimensionScore(score=20, flag="pass", reason="ok"),
            compliance=DimensionScore(score=20, flag="pass", reason="ok"),
            logic=DimensionScore(score=18, flag="pass", reason="ok"),
            completeness=DimensionScore(score=17, flag="warn", reason="ok"),
        )
        assert v.overall_grade == "B"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Ticker normalization
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickerNormalization:
    """Test the regex patterns used in app.py for ticker detection."""

    _NUMERIC_TICKER_RE = re.compile(
        r"(?<!\w)(\d{1,6})\s*[\.\s]\s*(HK|T|L|SS|SZ|KS|TW|JP)\b", re.I
    )

    def test_hkex_numeric_basic(self):
        assert self._NUMERIC_TICKER_RE.search("2015 HK")

    def test_hkex_dotted(self):
        assert self._NUMERIC_TICKER_RE.search("0700.HK")

    def test_japan_ticker(self):
        assert self._NUMERIC_TICKER_RE.search("8306.T")

    def test_london_ticker(self):
        assert self._NUMERIC_TICKER_RE.search("HSBA.L") is None  # alpha prefix

    def test_plain_number_no_match(self):
        assert self._NUMERIC_TICKER_RE.search("the year 2015 was") is None

    def test_us_alpha_pattern(self):
        us_re = re.compile(r"\b([A-Z]{2,5})\b")
        non_tickers = {"AI", "US", "UK", "HK", "EU", "PE", "EV", "IPO"}
        matches = [m.group(1) for m in us_re.finditer("NVDA is better than TSLA")]
        filtered = [t for t in matches if t not in non_tickers]
        assert "NVDA" in filtered
        assert "TSLA" in filtered

    def test_non_ticker_words_excluded(self):
        us_re = re.compile(r"\b([A-Z]{2,5})\b")
        non_tickers = {"AI", "US", "UK", "HK", "EU", "PE", "EV", "IPO"}
        matches = [m.group(1) for m in us_re.finditer("AI is hot in the US")]
        filtered = [t for t in matches if t not in non_tickers]
        assert len(filtered) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: DSPy router classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestDspyRouter:
    def test_stock_analysis(self):
        from dspy_router import classify_query
        assert classify_query("how's NVDA doing?", has_ticker=True) == "STOCK_ANALYSIS"

    def test_comparison(self):
        from dspy_router import classify_query
        assert classify_query("compare AAPL vs MSFT", has_ticker=True) == "COMPARISON"

    def test_concept(self):
        from dspy_router import classify_query
        assert classify_query("what is a P/E ratio?", has_ticker=False) == "CONCEPT"

    def test_report_request(self):
        from dspy_router import classify_query
        assert classify_query("generate report for AAPL", has_ticker=True) == "REPORT_REQUEST"

    def test_market_analysis(self):
        from dspy_router import classify_query
        assert classify_query("how are emerging markets performing this quarter?", has_ticker=False) == "MARKET_ANALYSIS"

    def test_route_fast_with_ticker(self):
        from dspy_router import route
        agents = route("fast", "STOCK_ANALYSIS", has_ticker=True)
        assert agents == ["CompanyAgent"]

    def test_route_standard_stock(self):
        from dspy_router import route
        agents = route("standard", "STOCK_ANALYSIS", has_ticker=True)
        assert "CompanyAgent" in agents
        assert "WallStreetAgent" in agents
        assert "NewsAgent" in agents

    def test_route_master_includes_macro(self):
        from dspy_router import route
        agents = route("master", "STOCK_ANALYSIS", has_ticker=True)
        assert "MacroAgent" in agents

    def test_route_concept_no_agents(self):
        from dspy_router import route
        agents = route("standard", "CONCEPT", has_ticker=False)
        assert agents == []


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Audit rubric prompt construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditRubric:
    def test_system_prompt_built(self):
        from audit.rubric import build_judge_system_prompt
        prompt = build_judge_system_prompt()
        assert "Factuality" in prompt
        assert "Compliance" in prompt
        assert "Logic" in prompt
        assert "Completeness" in prompt
        assert "JSON" in prompt

    def test_user_message_built(self):
        from audit.rubric import build_judge_user_message
        msg = build_judge_user_message("CIO answer here", "agent data here", "standard")
        assert "standard" in msg
        assert "CIO answer here" in msg
        assert "agent data here" in msg

    def test_user_message_truncation(self):
        from audit.rubric import build_judge_user_message
        long_text = "x" * 10000
        msg = build_judge_user_message(long_text, long_text, "standard")
        assert len(msg) < 20000


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Judge fallback (structural-only, no OpenAI key)
# ═══════════════════════════════════════════════════════════════════════════════


class TestJudgeFallback:
    def test_structural_only_audit(self, good_cio_output):
        from audit.judge import run_audit
        verdict = run_audit(good_cio_output, agent_data="", level="standard", use_llm=False)
        assert isinstance(verdict, AuditVerdict)
        assert verdict.overall_score > 0
        assert verdict.overall_grade in ("A", "B", "C", "D", "F")

    def test_bad_output_lower_score(self, bad_cio_output):
        from audit.judge import run_audit
        verdict = run_audit(bad_cio_output, agent_data="", level="standard", use_llm=False)
        assert verdict.overall_score < 80
        assert verdict.compliance.flag in ("warn", "fail")

    def test_structural_fast_level(self):
        from audit.judge import run_audit
        text = "Quick take on AAPL. Looks good.\n> Not financial advice."
        verdict = run_audit(text, agent_data="", level="fast", use_llm=False)
        assert verdict.completeness.flag == "pass"
