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
    SFCAuditResult,
    SFCDimensionScore,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: SFC Audit Schemas (HK SFC compliance models)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSFCSchemas:
    def test_sfc_dimension_score_valid(self):
        s = SFCDimensionScore(score=8, reason="Good SFC tone")
        assert s.score == 8
        assert s.reason == "Good SFC tone"

    def test_sfc_dimension_score_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SFCDimensionScore(score=11, reason="Over max")
        with pytest.raises(ValidationError):
            SFCDimensionScore(score=-1, reason="Under min")

    def test_sfc_audit_result_full(self):
        result = SFCAuditResult(
            sfc_tone=SFCDimensionScore(score=8, reason="Appropriate language"),
            explainability=SFCDimensionScore(score=9, reason="Clear reasoning"),
            risk_disclosure=SFCDimensionScore(score=7, reason="Missing one risk"),
            total_score=24,
            verdict="PASS",
            remediation="None required",
        )
        assert result.verdict == "PASS"
        assert result.total_score == 24

    def test_sfc_audit_result_defaults(self):
        result = SFCAuditResult()
        assert result.verdict == "REVIEW"
        assert result.total_score == 0

    def test_audit_verdict_with_sfc(self):
        sfc = SFCAuditResult(
            sfc_tone=SFCDimensionScore(score=6, reason="Some violations"),
            explainability=SFCDimensionScore(score=7, reason="OK"),
            risk_disclosure=SFCDimensionScore(score=5, reason="Missing disclaimers"),
            total_score=18,
            verdict="REVIEW",
            remediation="Add risk warnings",
        )
        verdict = AuditVerdict(
            overall_grade="B",
            overall_score=72,
            factuality=DimensionScore(score=20, flag="pass", reason="OK"),
            compliance=DimensionScore(score=18, flag="pass", reason="OK"),
            logic=DimensionScore(score=16, flag="warn", reason="Minor issues"),
            completeness=DimensionScore(score=18, flag="pass", reason="OK"),
            sfc_audit=sfc,
        )
        assert verdict.sfc_audit is not None
        assert verdict.sfc_audit.verdict == "REVIEW"

    def test_audit_verdict_without_sfc(self):
        verdict = AuditVerdict(
            overall_grade="A",
            overall_score=90,
            factuality=DimensionScore(score=23, flag="pass", reason="Good"),
            compliance=DimensionScore(score=22, flag="pass", reason="Good"),
            logic=DimensionScore(score=22, flag="pass", reason="Good"),
            completeness=DimensionScore(score=23, flag="pass", reason="Good"),
        )
        assert verdict.sfc_audit is None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: DSPy Report module (structural tests, no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDspyReportHelpers:
    def test_normalize_recommendation(self):
        from dspy_report import _normalize_recommendation
        assert _normalize_recommendation("BUY") == "BUY"
        assert _normalize_recommendation("Strong Buy") == "BUY"
        assert _normalize_recommendation("SELL") == "SELL"
        assert _normalize_recommendation("Underweight / Sell") == "SELL"
        assert _normalize_recommendation("HOLD") == "HOLD"
        assert _normalize_recommendation("Neutral") == "HOLD"
        assert _normalize_recommendation("something else") == "HOLD"

    def test_normalize_conviction(self):
        from dspy_report import _normalize_conviction
        assert _normalize_conviction("High") == "High"
        assert _normalize_conviction("high conviction") == "High"
        assert _normalize_conviction("Low") == "Low"
        assert _normalize_conviction("very low") == "Low"
        assert _normalize_conviction("Medium") == "Medium"
        assert _normalize_conviction("moderate") == "Medium"

    def test_safe_parse_json_list_valid(self):
        from dspy_report import _safe_parse_json_list
        result = _safe_parse_json_list('["risk 1", "risk 2", "risk 3"]')
        assert result == ["risk 1", "risk 2", "risk 3"]

    def test_safe_parse_json_list_invalid(self):
        from dspy_report import _safe_parse_json_list
        result = _safe_parse_json_list("- risk 1\n- risk 2")
        assert len(result) == 2
        assert "risk 1" in result[0]

    def test_safe_parse_json_list_single(self):
        from dspy_report import _safe_parse_json_list
        result = _safe_parse_json_list("just a single risk")
        assert result == ["just a single risk"]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: SFC metric in optimize.py
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: SQLite Audit DB
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditDB:
    def test_log_and_retrieve(self, tmp_path):
        from tools.audit_db import AuditDB
        db = AuditDB(db_path=str(tmp_path / "test.db"))
        row_id = db.log_agent_call(
            agent_name="CompanyAgent",
            query_text="Analyze TSLA",
            output_text="P/E is 72x",
            ticker="TSLA",
        )
        assert row_id > 0
        entries = db.get_recent_entries(ticker="TSLA")
        assert len(entries) == 1
        assert entries[0]["agent_name"] == "CompanyAgent"

    def test_cio_decision_log(self, tmp_path):
        from tools.audit_db import AuditDB
        db = AuditDB(db_path=str(tmp_path / "test2.db"))
        row_id = db.log_cio_decision(
            query="Analyze AAPL",
            cio_output="AAPL looks strong...",
            ticker="AAPL",
            compliance_score=90,
            sfc_verdict="PASS",
            sfc_score=28,
            recommendation="BUY",
            rec_json={"recommendation": "BUY", "target_price": "$200"},
        )
        assert row_id > 0
        history = db.get_compliance_history("AAPL")
        assert len(history) == 1
        assert history[0]["sfc_verdict"] == "PASS"

    def test_count_entries(self, tmp_path):
        from tools.audit_db import AuditDB
        db = AuditDB(db_path=str(tmp_path / "test3.db"))
        assert db.count_entries() == 0
        db.log_agent_call(agent_name="Test", query_text="q1")
        db.log_agent_call(agent_name="Test", query_text="q2")
        assert db.count_entries() == 2
        assert db.count_entries(agent_name="Test") == 2
        assert db.count_entries(agent_name="Other") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Knowledge Engine (chunking, graceful fallback)
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeEngine:
    def test_chunk_short_text(self):
        from tools.knowledge_engine import _chunk_text
        chunks = _chunk_text("Short text", max_chars=1500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_long_text(self):
        from tools.knowledge_engine import _chunk_text
        text = "A " * 1000
        chunks = _chunk_text(text, max_chars=500)
        assert len(chunks) > 1

    def test_chunk_paragraphs(self):
        from tools.knowledge_engine import _chunk_text
        text = ("First paragraph with enough content to exceed limit. " * 10
                + "\n\n"
                + "Second paragraph with enough content to exceed limit. " * 10)
        chunks = _chunk_text(text, max_chars=300)
        assert len(chunks) >= 2

    def test_embed_graceful_fallback(self):
        from tools.knowledge_engine import embed_report
        result = embed_report(ticker="TEST", analysis_text="test")
        assert isinstance(result, str)

    def test_search_graceful_fallback(self):
        from tools.knowledge_engine import search_similar
        results = search_similar("test query")
        assert isinstance(results, list)

    def test_stats_graceful_fallback(self):
        from tools.knowledge_engine import get_store_stats
        stats = get_store_stats()
        assert "reports_count" in stats
        assert "chroma_path" in stats


class TestSFCMetric:
    def test_sfc_metric_penalizes_guarantees(self):
        """CIO output with 'guaranteed returns' should score lower."""
        import dspy
        from optimize import cio_metric

        class FakePrediction:
            opening_take = "AAPL is guaranteed to rise."
            analysis = "This stock will definitely double. You should buy immediately."
            decision_frame = "Buy now for risk-free returns."

        ex = dspy.Example(query="AAPL analysis", level="fast").with_inputs("query", "level")
        pred = FakePrediction()
        score = cio_metric(ex, pred)
        assert score < 0.7  # should be penalized

    def test_sfc_metric_rewards_hedging(self):
        """CIO output with SFC-appropriate hedging should score better."""
        import dspy
        from optimize import cio_metric

        class FakePrediction:
            opening_take = "AAPL appears to offer a reasonable risk-reward at current levels."
            analysis = ("Based on available data, the company may benefit from AI trends. "
                       "We believe the margin trajectory suggests moderate upside potential. "
                       "This could appeal to growth-oriented investors.")
            decision_frame = ("This tends to fit investors who can live with tech volatility "
                            "and care about long-term compounding.")

        ex = dspy.Example(query="AAPL analysis", level="fast").with_inputs("query", "level")
        pred = FakePrediction()
        score = cio_metric(ex, pred)
        assert score > 0.5  # hedged language should score better
