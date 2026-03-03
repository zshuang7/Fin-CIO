"""
audit/judge.py — GPT-4o compliance judge engine.

Chains a fast structural pre-check with a GPT-4o LLM evaluation to produce
an AuditVerdict for each CIO response.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .schemas import AuditVerdict, DimensionScore, StructuralCheckResult, run_structural_check
from .rubric import build_judge_system_prompt, build_judge_user_message

logger = logging.getLogger(__name__)

# ── Grade from score ─────────────────────────────────────────────────────────

def _grade_from_score(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _flag_from_score(score: int) -> str:
    if score >= 18:
        return "pass"
    if score >= 12:
        return "warn"
    return "fail"


# ── Structural-only fallback verdict ─────────────────────────────────────────

def _build_structural_verdict(check: StructuralCheckResult, level: str) -> AuditVerdict:
    """Build an AuditVerdict from structural checks alone (no LLM)."""
    factuality_score = 20  # structural check can't deeply verify facts
    compliance_score = 25 if check.has_disclaimer else 10
    if check.has_forbidden_opener:
        compliance_score = max(compliance_score - 8, 0)
    logic_score = 18  # structural check can't verify logic
    completeness_score = 25 if not check.missing_sections else max(5, 25 - 6 * len(check.missing_sections))

    total = factuality_score + compliance_score + logic_score + completeness_score

    return AuditVerdict(
        overall_grade=_grade_from_score(total),
        overall_score=total,
        factuality=DimensionScore(
            score=factuality_score,
            flag=_flag_from_score(factuality_score),
            reason="Structural check only — factuality requires LLM judge for deep verification.",
        ),
        compliance=DimensionScore(
            score=compliance_score,
            flag=_flag_from_score(compliance_score),
            reason="Disclaimer present." if check.has_disclaimer else "Missing disclaimer statement.",
        ),
        logic=DimensionScore(
            score=logic_score,
            flag=_flag_from_score(logic_score),
            reason="Structural check only — logic consistency requires LLM judge.",
        ),
        completeness=DimensionScore(
            score=completeness_score,
            flag=_flag_from_score(completeness_score),
            reason="All required sections present." if not check.missing_sections
            else f"Missing: {', '.join(check.missing_sections)}",
        ),
    )


# ── GPT-4o judge call ───────────────────────────────────────────────────────

def _call_gpt4o_judge(
    cio_answer: str,
    agent_data: str,
    level: str,
) -> Optional[AuditVerdict]:
    """Call OpenAI GPT-4o to evaluate the CIO answer. Returns None on failure."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM judge, using structural only")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        system_prompt = build_judge_system_prompt()
        user_message = build_judge_user_message(cio_answer, agent_data, level)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        verdict = AuditVerdict(**data)
        logger.info(
            "GPT-4o audit: %s (%d/100) — F:%d C:%d L:%d Q:%d",
            verdict.overall_grade,
            verdict.overall_score,
            verdict.factuality.score,
            verdict.compliance.score,
            verdict.logic.score,
            verdict.completeness.score,
        )
        return verdict

    except ImportError:
        logger.warning("openai package not installed — falling back to structural audit")
        return None
    except Exception as exc:
        logger.error("GPT-4o judge failed: %s", exc)
        return None


# ── Main audit function ─────────────────────────────────────────────────────

def run_audit(
    cio_answer: str,
    agent_data: str = "",
    level: str = "standard",
    use_llm: bool = True,
) -> AuditVerdict:
    """Run the full compliance audit on a CIO answer.

    1. Structural pre-check (instant)
    2. GPT-4o judge (if use_llm=True and OPENAI_API_KEY is set)
    3. Falls back to structural-only verdict if LLM fails

    Args:
        cio_answer: The final markdown text shown to the user.
        agent_data: Concatenated raw outputs from agents (for cross-checking).
        level: Analysis level (fast/standard/master/deep_dive/comparison).
        use_llm: Whether to attempt the GPT-4o judge call.
    """
    structural = run_structural_check(cio_answer, level)

    if not use_llm:
        return _build_structural_verdict(structural, level)

    llm_verdict = _call_gpt4o_judge(cio_answer, agent_data, level)
    if llm_verdict is not None:
        return llm_verdict

    return _build_structural_verdict(structural, level)
