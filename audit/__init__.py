"""
audit — Independent compliance audit system for Fin-CIO.

IMPORTANT: All LLM-as-a-judge calls use GPT-4o exclusively.
DeepSeek is the CIO brain — it MUST NOT judge its own output.

Layer 1: Structural Pydantic validation (instant, no LLM)
Layer 2: GPT-4o unified judge (general compliance + HK SFC regulatory)
Layer 3: DSPy metric integration (in optimize.py)

Usage:
    from audit import run_audit_async, run_audit_sync

    # Async (for background thread in Streamlit)
    import queue
    result_q = queue.Queue()
    run_audit_async(cio_answer, agent_data, level, result_q)
    verdict = result_q.get()

    # Sync (for testing / offline batch)
    verdict = run_audit_sync(cio_answer, agent_data, level)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .schemas import (
    AuditVerdict,
    DimensionScore,
    SFCAuditResult,
    SFCDimensionScore,
    StructuralCheckResult,
    run_structural_check,
)
from .judge import run_audit, run_sfc_judge

logger = logging.getLogger(__name__)

__all__ = [
    "run_audit_async",
    "run_audit_sync",
    "run_sfc_judge",
    "run_structural_check",
    "AuditVerdict",
    "DimensionScore",
    "SFCAuditResult",
    "SFCDimensionScore",
    "StructuralCheckResult",
]


def run_audit_sync(
    cio_answer: str,
    agent_data: str = "",
    level: str = "standard",
    use_llm: bool = True,
) -> AuditVerdict:
    """Run the full audit synchronously. Returns an AuditVerdict."""
    return run_audit(cio_answer, agent_data, level, use_llm)


def run_audit_async(
    cio_answer: str,
    agent_data: str = "",
    level: str = "standard",
    result_queue: Optional[queue.Queue] = None,
    use_llm: bool = True,
) -> threading.Thread:
    """Launch the audit in a background thread.

    The AuditVerdict is placed into result_queue when done.
    Returns the thread object (already started).
    """
    q = result_queue or queue.Queue()

    def _worker():
        try:
            verdict = run_audit(cio_answer, agent_data, level, use_llm)
            q.put(("ok", verdict))
        except Exception as exc:
            logger.error("Audit thread failed: %s", exc)
            q.put(("err", str(exc)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
