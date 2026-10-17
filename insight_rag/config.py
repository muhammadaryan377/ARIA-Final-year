"""Configuration for ARIA Insight Agent's conversational PDF-RAG capability."""

import os
from pathlib import Path

from core.config import DATA_DIR

RAG_PIPELINE_VERSION = "2.0-enterprise"  # P1: surfaced in query diagnostics (scope_layer + routes)

RAG_ROOT = DATA_DIR / "insight_rag"
RAG_USERS_DIR = RAG_ROOT / "users"
RAG_USERS_DIR.mkdir(parents=True, exist_ok=True)

RAG_DATABASE_URL = (
    os.getenv("ARIA_RAG_DATABASE_URL")
    or os.getenv("RAG_DATABASE_URL")
    or ""
).strip()

EMBEDDING_MODEL = os.getenv("ARIA_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# During the Groq -> DeepSeek migration an existing local .env may still contain
# ARIA_RAG_LLM_MODEL=openai/gpt-oss-20b. Do not send that legacy Groq model name
# to DeepSeek. Explicit DeepSeek model overrides remain supported.
_requested_rag_model = os.getenv("ARIA_RAG_LLM_MODEL", "").strip()
RAG_LLM_MODEL = (
    _requested_rag_model
    if _requested_rag_model.startswith("deepseek-")
    else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"
)
RAG_COLLECTION_PREFIX = os.getenv("ARIA_RAG_COLLECTION_PREFIX", "aria_insight_pdf")

# Semantic routing is model-driven. User utterances are not mapped through phrase
# lists or regex intents; the router returns a validated schema.
ROUTER_MIN_CONFIDENCE = float(os.getenv("ARIA_RAG_ROUTER_MIN_CONFIDENCE", "0.62"))
ROUTER_TIMEOUT_SECONDS = int(os.getenv("ARIA_RAG_ROUTER_TIMEOUT_SECONDS", "10"))

# Local cross-encoder reranking improves precision after hybrid retrieval while
# preserving a graceful fallback when the model is unavailable.
RERANKER_ENABLED = os.getenv("ARIA_RAG_RERANKER_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RERANKER_MODEL = os.getenv(
    "ARIA_RAG_RERANKER_MODEL",
    "Xenova/ms-marco-MiniLM-L-6-v2",
)
RERANKER_TOP_K = int(os.getenv("ARIA_RAG_RERANKER_TOP_K", "10"))

MAX_PDF_MB = int(os.getenv("ARIA_RAG_MAX_PDF_MB", "50"))
MAX_SELECTED_DOCUMENTS = int(os.getenv("ARIA_RAG_MAX_SELECTED_DOCUMENTS", "8"))
TEXT_CHUNK_SIZE = int(os.getenv("ARIA_RAG_TEXT_CHUNK_SIZE", "1500"))
TEXT_CHUNK_OVERLAP = int(os.getenv("ARIA_RAG_TEXT_CHUNK_OVERLAP", "220"))
TABLE_ROWS_PER_CHUNK = int(os.getenv("ARIA_RAG_TABLE_ROWS_PER_CHUNK", "25"))
# Generic tabular (CSV/DB-row) ingestion limits. Large row sets are represented
# by a full-coverage summary chunk plus a bounded stratified sample of row
# chunks so retrieval context stays within budget for any table size.
TABULAR_MAX_ROW_CHUNKS = int(os.getenv("ARIA_RAG_TABULAR_MAX_ROW_CHUNKS", "12"))
TABULAR_MAX_ROWS = int(os.getenv("ARIA_RAG_TABULAR_MAX_ROWS", "50000"))
# Sibling expansion restores split-table neighbours for deterministic table
# math. The cap keeps very large tables (any source) from flooding context;
# full-table aggregates remain available via tabular summary chunks.
TABLE_SIBLING_MAX_CHUNKS = int(os.getenv("ARIA_RAG_TABLE_SIBLING_MAX_CHUNKS", "16"))
RETRIEVAL_FETCH_K = int(os.getenv("ARIA_RAG_RETRIEVAL_FETCH_K", "18"))
RETRIEVAL_TOP_K = int(os.getenv("ARIA_RAG_RETRIEVAL_TOP_K", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("ARIA_RAG_MAX_CONTEXT_CHARS", "24000"))
RECENT_HISTORY_TURNS = int(os.getenv("ARIA_RAG_RECENT_HISTORY_TURNS", "8"))

# Per-query LLM call budget / timeouts / circuit-breaker (reliability, no rewrite).
# Bounded so one question can never loop the hosted provider indefinitely.
RAG_MAX_LLM_CALLS_PER_QUERY = int(os.getenv("ARIA_RAG_MAX_LLM_CALLS_PER_QUERY", "8"))
RAG_VERIFY_TIMEOUT_S = int(os.getenv("ARIA_RAG_VERIFY_TIMEOUT_S", "12"))
RAG_QUERY_TIMEOUT_S = int(os.getenv("ARIA_RAG_QUERY_TIMEOUT_S", "120"))
RAG_CIRCUIT_BREAKER_FAILURES = int(os.getenv("ARIA_RAG_CIRCUIT_BREAKER_FAILURES", "5"))
RAG_CIRCUIT_BREAKER_WINDOW_S = int(os.getenv("ARIA_RAG_CIRCUIT_BREAKER_WINDOW_S", "300"))
# Per-user vector-store cache bound (LRU, storage.py). Each entry owns a
# PGVector + SQLAlchemy pool, so the cache is capped instead of growing one
# pool per user for the lifetime of the process.
RAG_VECTOR_STORE_CACHE_MAX = int(os.getenv("ARIA_RAG_VECTOR_STORE_CACHE_MAX", "64"))
# Lexical/BM25 corpus bounds: never score an unbounded whole corpus in memory.
# 3000 docs max; per-doc text is truncated for scoring only (evidence text is
# never mutated); paging + early-exit keep peak memory flat for any library.
RAG_BM25_MAX_DOCS = int(os.getenv("ARIA_RAG_BM25_MAX_DOCS", "3000"))
RAG_BM25_MAX_CHARS_PER_DOC = int(os.getenv("ARIA_RAG_BM25_MAX_CHARS_PER_DOC", "8000"))
RAG_LEXICAL_CORPUS_MAX_CHUNKS = int(os.getenv("ARIA_RAG_LEXICAL_CORPUS_MAX_CHUNKS", "3000"))
RAG_BM25_PAGE_SIZE = int(os.getenv("ARIA_RAG_BM25_PAGE_SIZE", "1000"))
RAG_BM25_EARLY_EXIT = os.getenv("ARIA_RAG_BM25_EARLY_EXIT", "true").strip().lower() not in {
    "0", "false", "no", "off",
}


def bm25_scan_bounds(
    max_docs: int | None = None,
    max_chars_per_doc: int | None = None,
    page_size: int | None = None,
) -> tuple[int, int, int]:
    """Return enforced (max_docs, max_chars_per_doc, page_size) BM25 bounds.

    Caps: max_docs <= 3000 (hard ceiling), per-doc chars >= 1, page size
    bounded to [100, max_docs] so one scan can never OOM the request.
    """
    docs = RAG_BM25_MAX_DOCS if max_docs is None else int(max_docs)
    chars = RAG_BM25_MAX_CHARS_PER_DOC if max_chars_per_doc is None else int(max_chars_per_doc)
    page = RAG_BM25_PAGE_SIZE if page_size is None else int(page_size)
    docs = max(1, min(docs, 3000))
    chars = max(1, chars)
    page = max(100, min(page, docs))
    return docs, chars, page


def truncate_for_bm25(text: str, max_chars: int | None = None) -> str:
    """Enforce the per-doc char cap for BM25 term counting (scoring only)."""
    cap = RAG_BM25_MAX_CHARS_PER_DOC if max_chars is None else int(max_chars)
    value = text or ""
    return value[:max(1, cap)]


def validate_rag_config() -> None:
    """Fail fast on unsafe or internally inconsistent RAG settings."""
    errors: list[str] = []
    if not (0.0 <= ROUTER_MIN_CONFIDENCE <= 1.0):
        errors.append("ARIA_RAG_ROUTER_MIN_CONFIDENCE must be between 0 and 1")
    if ROUTER_TIMEOUT_SECONDS < 1:
        errors.append("ARIA_RAG_ROUTER_TIMEOUT_SECONDS must be >= 1")
    if TEXT_CHUNK_SIZE < 256:
        errors.append("ARIA_RAG_TEXT_CHUNK_SIZE must be >= 256")
    if TEXT_CHUNK_OVERLAP < 0 or TEXT_CHUNK_OVERLAP >= TEXT_CHUNK_SIZE:
        errors.append("ARIA_RAG_TEXT_CHUNK_OVERLAP must be >= 0 and smaller than chunk size")
    if TABLE_ROWS_PER_CHUNK < 1:
        errors.append("ARIA_RAG_TABLE_ROWS_PER_CHUNK must be >= 1")
    if TABULAR_MAX_ROW_CHUNKS < 1:
        errors.append("ARIA_RAG_TABULAR_MAX_ROW_CHUNKS must be >= 1")
    if TABULAR_MAX_ROWS < 1:
        errors.append("ARIA_RAG_TABULAR_MAX_ROWS must be >= 1")
    if TABLE_SIBLING_MAX_CHUNKS < 1:
        errors.append("ARIA_RAG_TABLE_SIBLING_MAX_CHUNKS must be >= 1")
    if RETRIEVAL_TOP_K < 1 or RETRIEVAL_FETCH_K < RETRIEVAL_TOP_K:
        errors.append("retrieval fetch_k must be >= top_k >= 1")
    if MAX_CONTEXT_CHARS < 4000:
        errors.append("ARIA_RAG_MAX_CONTEXT_CHARS must be >= 4000")
    if MAX_PDF_MB < 1:
        errors.append("ARIA_RAG_MAX_PDF_MB must be >= 1")
    if not (1 <= MAX_SELECTED_DOCUMENTS <= 20):
        errors.append("ARIA_RAG_MAX_SELECTED_DOCUMENTS must be between 1 and 20")
    if RECENT_HISTORY_TURNS < 1:
        errors.append("ARIA_RAG_RECENT_HISTORY_TURNS must be >= 1")
    if RAG_MAX_LLM_CALLS_PER_QUERY < 1:
        errors.append("ARIA_RAG_MAX_LLM_CALLS_PER_QUERY must be >= 1")
    if RAG_VERIFY_TIMEOUT_S < 1:
        errors.append("ARIA_RAG_VERIFY_TIMEOUT_S must be >= 1")
    if RAG_QUERY_TIMEOUT_S < 10:
        errors.append("ARIA_RAG_QUERY_TIMEOUT_S must be >= 10")
    if RAG_CIRCUIT_BREAKER_FAILURES < 1:
        errors.append("ARIA_RAG_CIRCUIT_BREAKER_FAILURES must be >= 1")
    if RAG_CIRCUIT_BREAKER_WINDOW_S < 10:
        errors.append("ARIA_RAG_CIRCUIT_BREAKER_WINDOW_S must be >= 10")
    if RAG_VECTOR_STORE_CACHE_MAX < 1:
        errors.append("ARIA_RAG_VECTOR_STORE_CACHE_MAX must be >= 1")
    if RAG_BM25_MAX_DOCS < 100 or RAG_BM25_MAX_DOCS > 3000:
        errors.append("ARIA_RAG_BM25_MAX_DOCS must be between 100 and 3000")
    if RAG_BM25_MAX_CHARS_PER_DOC < 1000:
        errors.append("ARIA_RAG_BM25_MAX_CHARS_PER_DOC must be >= 1000")
    if RAG_BM25_PAGE_SIZE < 100:
        errors.append("ARIA_RAG_BM25_PAGE_SIZE must be >= 100")
    if RAG_LEXICAL_CORPUS_MAX_CHUNKS < 100:
        errors.append("ARIA_RAG_LEXICAL_CORPUS_MAX_CHUNKS must be >= 100")
    if errors:
        raise RuntimeError("Invalid ARIA RAG configuration: " + "; ".join(errors))


def user_key(user_id: str | int) -> str:
    """Return a filesystem/collection-safe stable key for one authenticated user."""
    import hashlib

    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:20]


def user_dir(user_id: str | int) -> Path:
    path = RAG_USERS_DIR / user_key(user_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "documents").mkdir(parents=True, exist_ok=True)
    (path / "conversations").mkdir(parents=True, exist_ok=True)
    (path / "chunks").mkdir(parents=True, exist_ok=True)
    return path
