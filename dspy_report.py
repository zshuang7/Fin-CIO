"""
dspy_report.py — DSPy-powered structured recommendation + SFC compliance judge.

Converts the ReportManager's free-text recommendation into a compilable DSPy
pipeline that produces:
  1. Structured JSON recommendation (auditable, typed fields)
  2. SFC compliance check via GPT-4o (independent judge — NOT DeepSeek)

IMPORTANT: The SFC judge uses GPT-4o exclusively. DeepSeek (the CIO model)
must NOT judge its own output — that would be self-evaluation.

The structured output feeds into:
  - SharedState.recommendation_json (for Excel/PDF export via ReportEngine)
  - audit/ module (for compliance scoring)

Note for Derivatives (ISDA ready):
  The recommendation_json includes a 'derivatives_note' field placeholder.
  When options/structured product pricing is added, this field will contain
  Greeks, hedging ratios, and ISDA documentation references.

Architecture:
  Agno agents (data) → dspy_cio.py (CIO synthesis) → dspy_report.py (structured rec + SFC judge)
                                                       ↓
                                                  state.recommendation_json
                                                       ↓
                                              report_engine.py (Excel/PDF)
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import dspy

logger = logging.getLogger(__name__)

# ── LM configuration (reuse or create) ──────────────────────────────────────

_lm_instance: Optional[dspy.LM] = None


def _get_lm() -> dspy.LM:
    """Get or create the DeepSeek LM instance for DSPy."""
    global _lm_instance
    if _lm_instance is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        model_id = os.getenv("MODEL_ID", "deepseek/deepseek-chat")
        _lm_instance = dspy.LM(model_id, api_key=api_key, max_tokens=2048)
        dspy.configure(lm=_lm_instance)
    return _lm_instance


# ══════════════════════════════════════════════════════════════════════════════
# DSPy Signatures
# ══════════════════════════════════════════════════════════════════════════════


class StructuredRecommendation(dspy.Signature):
    """You are a senior investment analyst producing a structured recommendation
    from a CIO analysis. Extract the key fields into clean, typed JSON.

    Rules:
    - recommendation MUST be exactly one of: BUY, HOLD, SELL
    - target_price must include currency symbol and be a specific number or range
    - conviction must be exactly one of: High, Medium, Low
    - time_horizon must be one of: Short-term (0-6mo), Medium-term (6-18mo), Long-term (18mo+)
    - risk_factors: list of 3-5 specific, actionable risk descriptions
    - catalysts: list of 2-4 specific near-term catalysts
    - reasoning_summary: 2-3 sentence thesis (the "why" behind the recommendation)
    - Do NOT hallucinate data. If the CIO analysis lacks a field, use "N/A".
    """

    cio_analysis: str = dspy.InputField(
        desc="The full CIO analysis text (markdown) from dspy_cio synthesis"
    )
    ticker: str = dspy.InputField(desc="The stock ticker symbol, e.g. 'TSLA', '0700.HK'")

    recommendation: str = dspy.OutputField(
        desc="Exactly one of: BUY, HOLD, SELL"
    )
    target_price: str = dspy.OutputField(
        desc="Target price with currency, e.g. '$250-280' or 'HK$650-760'. 'N/A' if not available."
    )
    conviction: str = dspy.OutputField(
        desc="Exactly one of: High, Medium, Low"
    )
    time_horizon: str = dspy.OutputField(
        desc="One of: Short-term (0-6mo), Medium-term (6-18mo), Long-term (18mo+)"
    )
    reasoning_summary: str = dspy.OutputField(
        desc="2-3 sentence investment thesis summarizing the key 'why'"
    )
    risk_factors: str = dspy.OutputField(
        desc="JSON array of 3-5 risk strings, e.g. [\"Regulatory risk from SFC\", \"Margin compression\"]"
    )
    catalysts: str = dspy.OutputField(
        desc="JSON array of 2-4 catalyst strings, e.g. [\"Q3 earnings beat\", \"New product launch\"]"
    )
    # Note for Derivatives: placeholder for future Greeks / ISDA fields
    derivatives_note: str = dspy.OutputField(
        desc="Derivatives relevance note: implied vol regime, hedging suggestion, or 'N/A - no derivatives context'. "
             "When Black-Scholes pricing is integrated, this field will contain Delta/Gamma/Vega recommendations."
    )


class SFCComplianceCheck(dspy.Signature):
    """You are a Hong Kong SFC (Securities and Futures Commission) compliance officer.
    Evaluate the following investment recommendation for regulatory compliance.

    SFC Code of Conduct requirements (Chapter 6 — Suitability):
    - Must not guarantee or promise investment returns
    - Must include appropriate risk warnings for the recommendation type
    - Must not use misleading or deceptive language
    - Must clearly distinguish between factual information and opinion/projection
    - Must include a disclaimer that this is not personalized financial advice
    - Must not make unsuitable recommendations without knowing client's risk profile

    SFC Guidelines on Online Distribution and Advisory Platforms (Oct 2019):
    - Automated advice must be transparent about its AI/algorithmic nature
    - Risk disclosures must be prominent, not buried in footnotes
    - Suitability assessment must be referenced or explicitly waived

    Score each dimension 0-10 and provide a brief reason.
    """

    recommendation_json: str = dspy.InputField(
        desc="The structured recommendation JSON to audit"
    )
    cio_analysis: str = dspy.InputField(
        desc="The full CIO analysis text that produced this recommendation"
    )

    sfc_tone_score: str = dspy.OutputField(
        desc="0-10 score for SFC-appropriate tone. Deduct for: guarantees, 'you should buy', "
             "aggressive language without risk context. Format: '<score>/10 — <reason>'"
    )
    explainability_score: str = dspy.OutputField(
        desc="0-10 score for reasoning traceability. Deduct for: unsupported claims, "
             "missing data citations, circular reasoning. Format: '<score>/10 — <reason>'"
    )
    risk_disclosure_score: str = dspy.OutputField(
        desc="0-10 score for risk warnings. Deduct for: missing disclaimer, no risk factors, "
             "downplaying material risks. Format: '<score>/10 — <reason>'"
    )
    overall_sfc_verdict: str = dspy.OutputField(
        desc="PASS (>=24/30), REVIEW (18-23/30), or FAIL (<18/30) with 1-sentence summary"
    )
    remediation: str = dspy.OutputField(
        desc="Specific fixes needed for compliance. 'None required' if PASS."
    )


# ══════════════════════════════════════════════════════════════════════════════
# DSPy Module — chains recommendation extraction + SFC compliance check
# ══════════════════════════════════════════════════════════════════════════════


class ReportSynthesizer(dspy.Module):
    """Produces structured recommendation JSON and runs SFC compliance audit.

    Pipeline:
      CIO analysis text → StructuredRecommendation → SFCComplianceCheck
                           ↓                          ↓
                      recommendation_json         sfc_verdict
    """

    def __init__(self):
        super().__init__()
        self.recommend = dspy.ChainOfThought(StructuredRecommendation)
        # SFC judge deliberately NOT using DSPy/DeepSeek — uses GPT-4o via audit/judge.py

    def forward(
        self,
        cio_analysis: str,
        ticker: str,
        run_sfc_check: bool = True,
    ) -> dict:
        """Run the full pipeline.

        Args:
            cio_analysis: Full CIO markdown output from dspy_cio.
            ticker: Stock ticker symbol.
            run_sfc_check: Whether to run the SFC compliance judge (default True).

        Returns:
            Dict with keys:
              - recommendation_json: dict with all structured fields
              - sfc_verdict: dict with compliance scores (or None if skipped)
              - timestamp: ISO timestamp
        """
        _get_lm()

        # Step 1: Extract structured recommendation
        rec_pred = self.recommend(cio_analysis=cio_analysis, ticker=ticker)

        # Parse JSON arrays from DSPy output
        risk_factors = _safe_parse_json_list(rec_pred.risk_factors)
        catalysts = _safe_parse_json_list(rec_pred.catalysts)

        recommendation_json = {
            "ticker": ticker.upper(),
            "recommendation": _normalize_recommendation(rec_pred.recommendation),
            "target_price": rec_pred.target_price.strip(),
            "conviction": _normalize_conviction(rec_pred.conviction),
            "time_horizon": rec_pred.time_horizon.strip(),
            "reasoning_summary": rec_pred.reasoning_summary.strip(),
            "risk_factors": risk_factors,
            "catalysts": catalysts,
            "derivatives_note": rec_pred.derivatives_note.strip(),
            "generated_at": datetime.now().isoformat(),
            "model": os.getenv("MODEL_ID", "deepseek/deepseek-chat"),
        }

        result = {
            "recommendation_json": recommendation_json,
            "sfc_verdict": None,
            "timestamp": datetime.now().isoformat(),
        }

        # Step 2: SFC compliance check via GPT-4o (independent judge)
        # IMPORTANT: Uses GPT-4o, NOT DeepSeek — the CIO model must not
        # judge its own output.
        if run_sfc_check:
            try:
                from audit.judge import run_sfc_judge
                rec_json_str = json.dumps(recommendation_json, ensure_ascii=False, indent=2)
                truncated = cio_analysis[:6000] if len(cio_analysis) > 6000 else cio_analysis

                sfc_result = run_sfc_judge(truncated, rec_json_str)

                if sfc_result is not None:
                    sfc_verdict = {
                        "sfc_tone": sfc_result.get("sfc_tone", {}),
                        "explainability": sfc_result.get("explainability", {}),
                        "risk_disclosure": sfc_result.get("risk_disclosure", {}),
                        "total_score": sfc_result.get("total_score", 0),
                        "verdict": sfc_result.get("verdict", "REVIEW"),
                        "remediation": sfc_result.get("remediation", ""),
                        "judged_at": datetime.now().isoformat(),
                        "judge_model": "gpt-4o",
                    }
                    result["sfc_verdict"] = sfc_verdict

                    logger.info(
                        "SFC audit (GPT-4o) for %s: %s (%d/30)",
                        ticker,
                        sfc_verdict["verdict"],
                        sfc_verdict["total_score"],
                    )
            except Exception as exc:
                logger.error("SFC compliance check failed for %s: %s", ticker, exc)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _safe_parse_json_list(raw: str) -> list[str]:
    """Parse a DSPy output string that should be a JSON array of strings."""
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    # Fallback: split by newlines or commas
    items = [line.strip().lstrip("- •*") for line in raw.split("\n") if line.strip()]
    return items or [raw]


def _normalize_recommendation(raw: str) -> str:
    """Normalize recommendation to exactly BUY/HOLD/SELL."""
    upper = raw.strip().upper()
    if "BUY" in upper:
        return "BUY"
    if "SELL" in upper:
        return "SELL"
    return "HOLD"


def _normalize_conviction(raw: str) -> str:
    """Normalize conviction to exactly High/Medium/Low."""
    lower = raw.strip().lower()
    if "high" in lower:
        return "High"
    if "low" in lower:
        return "Low"
    return "Medium"


# ── Singleton ────────────────────────────────────────────────────────────────

_synthesizer: Optional[ReportSynthesizer] = None


def get_report_synthesizer() -> ReportSynthesizer:
    """Return a singleton ReportSynthesizer."""
    global _synthesizer
    if _synthesizer is None:
        _get_lm()
        _synthesizer = ReportSynthesizer()
    return _synthesizer
