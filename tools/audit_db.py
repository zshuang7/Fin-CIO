"""
tools/audit_db.py — SQLite audit trail for SFC compliance record-keeping.

Every agent call is logged: timestamp, model, input query, output text,
compliance score, and SFC audit verdict. This creates the paper trail
required by HK SFC Code of Conduct for automated advisory systems.

Schema:
  agent_audit_log (
      id            INTEGER PRIMARY KEY,
      timestamp     TEXT NOT NULL,        -- ISO-8601
      agent_name    TEXT NOT NULL,        -- e.g. 'CompanyAgent', 'CIO'
      model_id      TEXT,                 -- e.g. 'deepseek/deepseek-chat'
      ticker        TEXT,                 -- e.g. 'TSLA', '0700.HK'
      query_text    TEXT,                 -- user query or agent input
      output_text   TEXT,                 -- agent response (truncated to 10K chars)
      level         TEXT,                 -- fast/standard/master/deep_dive
      compliance_score INTEGER,           -- 0-100 from audit/judge.py
      sfc_verdict   TEXT,                 -- PASS/REVIEW/FAIL from SFC judge
      sfc_score     INTEGER,             -- 0-30 from SFC judge
      recommendation TEXT,               -- BUY/HOLD/SELL (if applicable)
      rec_json      TEXT,                 -- full recommendation_json (JSON string)
      session_id    TEXT                  -- links entries from same conversation
  )

Note for Derivatives:
  When derivatives pricing is added, additional columns will be appended:
    product_type TEXT,         -- 'equity' / 'option' / 'structured_product'
    isda_ref     TEXT,         -- ISDA documentation reference
    greeks_json  TEXT          -- {delta, gamma, vega, theta} snapshot

Usage:
    from tools.audit_db import get_audit_db
    db = get_audit_db()
    db.log_agent_call(agent_name="CompanyAgent", query="Analyze TSLA", ...)
    recent = db.get_recent_entries(ticker="TSLA", limit=10)
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ── ORM Model ────────────────────────────────────────────────────────────────

_DB_PATH = os.getenv("AUDIT_DB_PATH", "data/audit_trail.db")


class Base(DeclarativeBase):
    pass


class AgentAuditLog(Base):
    """SFC-compliant audit trail for every agent decision."""
    __tablename__ = "agent_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False, index=True)
    agent_name = Column(String, nullable=False, index=True)
    model_id = Column(String, default="")
    ticker = Column(String, default="", index=True)
    query_text = Column(Text, default="")
    output_text = Column(Text, default="")
    level = Column(String, default="standard")
    compliance_score = Column(Integer, default=-1)
    sfc_verdict = Column(String, default="")
    sfc_score = Column(Integer, default=-1)
    recommendation = Column(String, default="")
    rec_json = Column(Text, default="")
    session_id = Column(String, default="", index=True)


# ── Database Manager ─────────────────────────────────────────────────────────

class AuditDB:
    """SQLite-backed audit trail manager.

    Thread-safe: each call creates its own session.
    """

    def __init__(self, db_path: str = _DB_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine)
        logger.info("Audit DB initialized at %s", db_path)

    def _session(self) -> Session:
        return self._SessionFactory()

    # ── Write operations ─────────────────────────────────────────────

    def log_agent_call(
        self,
        agent_name: str,
        query_text: str = "",
        output_text: str = "",
        model_id: str = "",
        ticker: str = "",
        level: str = "standard",
        compliance_score: int = -1,
        sfc_verdict: str = "",
        sfc_score: int = -1,
        recommendation: str = "",
        rec_json: Optional[dict] = None,
        session_id: str = "",
    ) -> int:
        """Insert one audit record. Returns the new row ID."""
        # Truncate large outputs to keep DB manageable
        output_truncated = output_text[:10000] if output_text else ""
        query_truncated = query_text[:2000] if query_text else ""
        rec_json_str = json.dumps(rec_json, ensure_ascii=False) if rec_json else ""

        entry = AgentAuditLog(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            model_id=model_id or os.getenv("MODEL_ID", "deepseek/deepseek-chat"),
            ticker=ticker.upper() if ticker else "",
            query_text=query_truncated,
            output_text=output_truncated,
            level=level,
            compliance_score=compliance_score,
            sfc_verdict=sfc_verdict,
            sfc_score=sfc_score,
            recommendation=recommendation,
            rec_json=rec_json_str,
            session_id=session_id or str(uuid.uuid4())[:8],
        )

        with self._session() as session:
            session.add(entry)
            session.commit()
            row_id = entry.id
            logger.debug(
                "Audit log #%d: %s | %s | %s",
                row_id, agent_name, ticker, sfc_verdict or "no-sfc",
            )
            return row_id

    def log_cio_decision(
        self,
        query: str,
        cio_output: str,
        ticker: str = "",
        level: str = "standard",
        compliance_score: int = -1,
        sfc_verdict: str = "",
        sfc_score: int = -1,
        recommendation: str = "",
        rec_json: Optional[dict] = None,
        session_id: str = "",
    ) -> int:
        """Convenience wrapper for logging the CIO's final decision."""
        return self.log_agent_call(
            agent_name="CIO",
            query_text=query,
            output_text=cio_output,
            ticker=ticker,
            level=level,
            compliance_score=compliance_score,
            sfc_verdict=sfc_verdict,
            sfc_score=sfc_score,
            recommendation=recommendation,
            rec_json=rec_json,
            session_id=session_id,
        )

    # ── Read operations ──────────────────────────────────────────────

    def get_recent_entries(
        self,
        ticker: str = "",
        agent_name: str = "",
        limit: int = 20,
    ) -> list[dict]:
        """Fetch recent audit entries, optionally filtered by ticker or agent."""
        with self._session() as session:
            q = session.query(AgentAuditLog)
            if ticker:
                q = q.filter(AgentAuditLog.ticker == ticker.upper())
            if agent_name:
                q = q.filter(AgentAuditLog.agent_name == agent_name)
            q = q.order_by(desc(AgentAuditLog.timestamp)).limit(limit)

            return [
                {
                    "id": row.id,
                    "timestamp": row.timestamp,
                    "agent_name": row.agent_name,
                    "ticker": row.ticker,
                    "level": row.level,
                    "compliance_score": row.compliance_score,
                    "sfc_verdict": row.sfc_verdict,
                    "recommendation": row.recommendation,
                    "query_text": row.query_text[:200],
                }
                for row in q.all()
            ]

    def get_compliance_history(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get compliance score history for a specific ticker (for trending)."""
        with self._session() as session:
            rows = (
                session.query(AgentAuditLog)
                .filter(AgentAuditLog.ticker == ticker.upper())
                .filter(AgentAuditLog.agent_name == "CIO")
                .filter(AgentAuditLog.compliance_score >= 0)
                .order_by(desc(AgentAuditLog.timestamp))
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp": r.timestamp,
                    "compliance_score": r.compliance_score,
                    "sfc_verdict": r.sfc_verdict,
                    "sfc_score": r.sfc_score,
                    "recommendation": r.recommendation,
                }
                for r in rows
            ]

    def count_entries(self, agent_name: str = "") -> int:
        """Count total audit entries, optionally filtered by agent."""
        with self._session() as session:
            q = session.query(AgentAuditLog)
            if agent_name:
                q = q.filter(AgentAuditLog.agent_name == agent_name)
            return q.count()

    def get_session_log(self, session_id: str) -> list[dict]:
        """Retrieve all entries for a given conversation session."""
        with self._session() as session:
            rows = (
                session.query(AgentAuditLog)
                .filter(AgentAuditLog.session_id == session_id)
                .order_by(AgentAuditLog.timestamp)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "agent_name": r.agent_name,
                    "ticker": r.ticker,
                    "query_text": r.query_text[:200],
                    "output_text": r.output_text[:500],
                    "compliance_score": r.compliance_score,
                    "sfc_verdict": r.sfc_verdict,
                }
                for r in rows
            ]


# ── Singleton ────────────────────────────────────────────────────────────────

_db_instance: Optional[AuditDB] = None


def get_audit_db() -> AuditDB:
    """Return the global AuditDB singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AuditDB()
    return _db_instance
