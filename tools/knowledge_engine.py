"""
tools/knowledge_engine.py — ChromaDB vector store for RAG-retrievable knowledge.

Stores and retrieves:
  1. Historical CIO analysis reports (auto-embedded after generation)
  2. SFC compliance documents (PDF upload via insert_document)
  3. ISDA master agreements and term sheets (future)
  4. Past audit verdicts and remediation notes

Uses Agno's Knowledge + ChromaDb integration for seamless agent access.

Architecture:
  report_engine.py (after save) → knowledge_engine.embed_report() → ChromaDB
  team_config.py (ReportManager) → Knowledge(vector_db=ChromaDb) → RAG search

Note for Derivatives:
  When ISDA documentation is added, a separate 'isda_docs' collection
  will store term sheets, CSA agreements, and netting opinions for
  RAG retrieval during structured product pricing.

Usage:
    from tools.knowledge_engine import get_knowledge_store, embed_report, search_similar

    # Embed a report after generation
    embed_report(ticker="TSLA", analysis_text="...", metadata={...})

    # Search for similar past analyses
    results = search_similar("TSLA earnings risk analysis", n_results=3)

    # Get Agno Knowledge instance for agent integration
    knowledge = get_knowledge_store()
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── ChromaDB client (lazy init) ──────────────────────────────────────────────

_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "data/chromadb")
_COLLECTION_REPORTS = "cio_reports"
_COLLECTION_DOCS = "compliance_docs"

_chroma_client = None
_reports_collection = None
_docs_collection = None


_CHROMA_AVAILABLE = True


def _get_chroma():
    """Lazy-initialize ChromaDB persistent client.

    Returns (client, reports_collection, docs_collection).
    Raises RuntimeError if ChromaDB is not available (e.g. Python 3.14
    compatibility issue with chromadb's Pydantic v1 dependency).
    """
    global _chroma_client, _reports_collection, _docs_collection, _CHROMA_AVAILABLE
    if not _CHROMA_AVAILABLE:
        raise RuntimeError("ChromaDB not available in this environment")

    if _chroma_client is None:
        try:
            import chromadb
        except Exception as e:
            _CHROMA_AVAILABLE = False
            logger.warning("ChromaDB import failed (non-critical): %s", e)
            raise RuntimeError(f"ChromaDB not available: {e}") from e

        os.makedirs(_CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)

        _reports_collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_REPORTS,
            metadata={"description": "Historical CIO analysis reports for RAG retrieval"},
        )
        _docs_collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_DOCS,
            metadata={"description": "SFC compliance docs, ISDA references"},
        )
        logger.info(
            "ChromaDB initialized: %s (reports=%d, docs=%d)",
            _CHROMA_PATH,
            _reports_collection.count(),
            _docs_collection.count(),
        )
    return _chroma_client, _reports_collection, _docs_collection


# ══════════════════════════════════════════════════════════════════════════════
# Embedding operations
# ══════════════════════════════════════════════════════════════════════════════


def embed_report(
    ticker: str,
    analysis_text: str,
    metadata: Optional[dict] = None,
    report_id: Optional[str] = None,
) -> str:
    """Embed a CIO analysis report into ChromaDB for future RAG retrieval.

    Called automatically by report_engine.py after saving Excel/PDF.
    The analysis_text is chunked into segments for better retrieval.
    Gracefully skips if ChromaDB is not available.

    Args:
        ticker: Stock ticker (e.g. 'TSLA', '0700.HK').
        analysis_text: Full CIO analysis markdown text.
        metadata: Optional dict with extra metadata (recommendation, level, etc.)
        report_id: Optional unique ID (auto-generated if not provided).

    Returns:
        Confirmation message with the document ID.
    """
    try:
        _, reports_col, _ = _get_chroma()
    except RuntimeError as e:
        return f"ChromaDB skipped: {e}"

    ts = datetime.now().isoformat()
    doc_id = report_id or f"{ticker}_{ts}".replace(":", "-")

    base_meta = {
        "ticker": ticker.upper(),
        "timestamp": ts,
        "type": "cio_report",
    }
    if metadata:
        base_meta.update({k: str(v) for k, v in metadata.items()})

    # Chunk long text for better retrieval granularity
    chunks = _chunk_text(analysis_text, max_chars=1500, overlap=200)

    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        chunk_meta = {**base_meta, "chunk_index": str(i), "total_chunks": str(len(chunks))}
        metadatas.append(chunk_meta)

    reports_col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(
        "Embedded report: %s (%d chunks, %d chars total)",
        doc_id, len(chunks), len(analysis_text),
    )
    return f"Report embedded: {doc_id} ({len(chunks)} chunks)"


def embed_document(
    doc_name: str,
    text: str,
    doc_type: str = "sfc_regulation",
    metadata: Optional[dict] = None,
) -> str:
    """Embed a compliance document (SFC PDF, ISDA agreement) into ChromaDB.

    Args:
        doc_name: Human-readable document name.
        text: Full document text (extracted from PDF/DOCX).
        doc_type: Category — 'sfc_regulation', 'isda_agreement', 'research_report'.
        metadata: Optional extra metadata.

    Returns:
        Confirmation message.

    Note for Derivatives:
        Use doc_type='isda_agreement' for ISDA Master Agreements,
        CSA documents, and netting opinions.
    """
    try:
        _, _, docs_col = _get_chroma()
    except RuntimeError as e:
        return f"ChromaDB skipped: {e}"

    ts = datetime.now().isoformat()
    doc_id = f"{doc_type}_{doc_name}_{ts}".replace(" ", "_").replace(":", "-")

    base_meta = {
        "doc_name": doc_name,
        "doc_type": doc_type,
        "timestamp": ts,
    }
    if metadata:
        base_meta.update({k: str(v) for k, v in metadata.items()})

    chunks = _chunk_text(text, max_chars=1500, overlap=200)

    ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {**base_meta, "chunk_index": str(i), "total_chunks": str(len(chunks))}
        for i in range(len(chunks))
    ]

    docs_col.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    logger.info("Embedded document: %s (%d chunks)", doc_name, len(chunks))
    return f"Document embedded: {doc_name} ({len(chunks)} chunks)"


# ══════════════════════════════════════════════════════════════════════════════
# Search / retrieval operations
# ══════════════════════════════════════════════════════════════════════════════


def search_similar(
    query: str,
    n_results: int = 3,
    ticker: Optional[str] = None,
    collection: str = "reports",
) -> list[dict]:
    """Semantic search across stored reports or documents.

    Args:
        query: Natural language search query.
        n_results: Max results to return.
        ticker: Optional ticker filter (only for reports collection).
        collection: 'reports' or 'docs'.

    Returns:
        List of dicts with keys: text, metadata, distance.
    """
    try:
        _, reports_col, docs_col = _get_chroma()
    except RuntimeError:
        return []

    col = reports_col if collection == "reports" else docs_col

    where_filter = None
    if ticker and collection == "reports":
        where_filter = {"ticker": ticker.upper()}

    try:
        results = col.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
    except Exception as e:
        logger.error("ChromaDB search failed: %s", e)
        return []

    items = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })

    return items


def get_past_analyses(ticker: str, limit: int = 5) -> list[dict]:
    """Retrieve past CIO analyses for a specific ticker (for RAG context).

    The ReportManager can use this to reference prior recommendations
    before generating a new report.
    """
    return search_similar(
        query=f"{ticker} investment analysis recommendation",
        n_results=limit,
        ticker=ticker,
        collection="reports",
    )


def get_compliance_references(query: str, limit: int = 3) -> list[dict]:
    """Search SFC/ISDA compliance documents for relevant guidance.

    Note for Derivatives:
        When ISDA docs are loaded, queries like 'netting opinion Hong Kong'
        will retrieve relevant ISDA Master Agreement sections.
    """
    return search_similar(
        query=query,
        n_results=limit,
        collection="docs",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stats / management
# ══════════════════════════════════════════════════════════════════════════════


def get_store_stats() -> dict:
    """Return basic stats about the vector stores."""
    try:
        _, reports_col, docs_col = _get_chroma()
        return {
            "reports_count": reports_col.count(),
            "docs_count": docs_col.count(),
            "chroma_path": _CHROMA_PATH,
            "available": True,
        }
    except RuntimeError:
        return {"reports_count": 0, "docs_count": 0, "chroma_path": _CHROMA_PATH, "available": False}


def delete_ticker_reports(ticker: str) -> str:
    """Delete all stored reports for a specific ticker."""
    try:
        _, reports_col, _ = _get_chroma()
    except RuntimeError as e:
        return f"ChromaDB not available: {e}"
    try:
        results = reports_col.get(where={"ticker": ticker.upper()})
        if results["ids"]:
            reports_col.delete(ids=results["ids"])
            return f"Deleted {len(results['ids'])} chunks for {ticker.upper()}"
        return f"No reports found for {ticker.upper()}"
    except Exception as e:
        logger.error("Delete failed: %s", e)
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Agno Knowledge integration
# ══════════════════════════════════════════════════════════════════════════════

_knowledge_instance = None


def get_knowledge_store():
    """Return an Agno Knowledge instance backed by ChromaDB.

    Use this in team_config.py to give the ReportManager RAG access:
        from tools.knowledge_engine import get_knowledge_store
        knowledge = get_knowledge_store()
        report_manager = Agent(knowledge=knowledge, ...)
    """
    global _knowledge_instance
    if _knowledge_instance is not None:
        return _knowledge_instance

    try:
        from agno.knowledge.knowledge import Knowledge
        from agno.vectordb.chroma import ChromaDb

        vector_db = ChromaDb(
            collection=_COLLECTION_REPORTS,
            path=_CHROMA_PATH,
            persistent_client=True,
        )
        _knowledge_instance = Knowledge(
            name="Fin-CIO Historical Reports",
            description=(
                "Historical CIO investment analyses, SFC compliance documents, "
                "and ISDA references. Use this to find similar past analyses "
                "and compliance guidance before generating new reports."
            ),
            vector_db=vector_db,
        )
        logger.info("Agno Knowledge store initialized with ChromaDB")
        return _knowledge_instance

    except ImportError as e:
        logger.warning("Agno Knowledge not available: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for better embedding retrieval.

    Uses paragraph boundaries when possible, falls back to character split.
    """
    if len(text) <= max_chars:
        return [text]

    # Try splitting on double newlines (paragraphs) first
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                # Force-split very long paragraphs
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i:i + max_chars].strip())
                current = ""
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())

    # Add overlap between chunks for better context
    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else ""
            overlapped.append(f"{prev_tail} {chunks[i]}".strip())
        return overlapped

    return chunks if chunks else [text[:max_chars]]
