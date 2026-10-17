"""Dependency-free lexical retrieval, RRF fusion and provenance signals."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from langchain_core.documents import Document


def chunk_key(doc: Document) -> str:
    meta = doc.metadata
    identity = (meta.get("document_id"), meta.get("chunk_id"))
    if identity[1]:
        return str(identity)
    raw = repr((meta.get("document_id"), meta.get("page"),
                meta.get("table_index"), doc.page_content))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deduplicate_chunks(documents: list[Document]) -> list[Document]:
    unique = {}
    for doc in documents:
        unique.setdefault(chunk_key(doc), doc)
    return list(unique.values())


def _annotate_retrieval(doc: Document, **values) -> Document:
    """Attach request-local provenance without replacing the Document object.

    Existing orchestration and tests rely on stable chunk object identity. The
    retrieval Documents are request-local/load-local objects, so metadata can be
    safely enriched in place while preserving that identity contract.
    """
    doc.metadata.update(values)
    return doc


def fuse_ranked_lists(rankings: list[list[Document]]) -> list[Document]:
    """RRF across subqueries with auditable consensus metadata.

    ``retrieval_votes`` counts independent ranked lists that returned a chunk and
    ``retrieval_query_count`` is the denominator. This allows monitoring to
    compare single-query and decomposed-query requests without inflating scores.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    ranks: dict[str, list[int]] = {}
    active_rankings = [ranking for ranking in rankings if ranking]
    query_count = max(1, len(active_rankings))
    for ranking in active_rankings:
        seen_in_ranking: set[str] = set()
        for rank, doc in enumerate(deduplicate_chunks(ranking), start=1):
            key = chunk_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            if key not in seen_in_ranking:
                ranks.setdefault(key, []).append(rank)
                seen_in_ranking.add(key)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    result: list[Document] = []
    for key in ordered:
        result.append(
            _annotate_retrieval(
                documents[key],
                retrieval_votes=len(ranks.get(key, [])),
                retrieval_query_count=query_count,
                retrieval_rrf_score=round(scores[key], 8),
                retrieval_subquery_ranks=list(ranks.get(key, [])),
            )
        )
    return result


def _split_identifier(token: str) -> list[str]:
    """Split snake/camel/dotted identifiers into generic sub-tokens.

    Tabular headers are single tokens (``weekly_sales``) while questions use
    plain words (``sales``). Expanding both sides with the same rule keeps
    lexical matching working for any column naming convention without
    referencing concrete dataset, table, or column names.
    """
    parts = re.split(r"[^0-9a-zA-Z]+", token)
    expanded: list[str] = []
    for part in parts:
        expanded.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", part) or ([part] if part else []))
    return [item.casefold() for item in expanded if item]


def terms(text: str) -> list[str]:
    # Unicode letters/digits retain Roman Urdu, Arabic and exact identifiers.
    raw = re.findall(r"\w+(?:[.-]\w+)*", text.casefold(), flags=re.UNICODE)
    output: list[str] = []
    for token in raw:
        output.append(token)
        for sub in _split_identifier(token):
            if sub != token and len(sub) > 1 and sub not in output:
                output.append(sub)
    return output


def _bm25_caps() -> tuple[int, int, int]:
    """Return (max_docs, max_chars_per_doc, page_size) bounds for BM25."""
    try:
        from .config import bm25_scan_bounds
        return bm25_scan_bounds()
    except Exception:
        return 3000, 8000, 1000


def bm25_search(query: str, documents: list[Document], *, limit: int,
                max_docs: int | None = None, max_chars_per_doc: int | None = None) -> list[Document]:
    """Okapi BM25 with document frequency and length normalisation.

    The corpus must already be restricted to the authenticated user's active
    documents. Index construction is request-local; no stale cross-user cache.
    Returned chunks include their BM25 score/rank for diagnostics and source
    provenance; the score is not treated as a calibrated relevance probability.

    Memory safety: the corpus is bounded (default from config). Oversized
    corpora are scored in pages and only the top-``limit`` survive, so a
    very large library can never OOM the request by loading everything at
    once. Per-document text is truncated for term counting only.
    """
    documents = deduplicate_chunks(documents)
    query_terms = terms(query)
    if not documents or not query_terms:
        return []
    cap_docs, cap_chars, cap_page = _bm25_caps()
    if max_docs is None:
        max_docs = cap_docs
    if max_chars_per_doc is None:
        max_chars_per_doc = cap_chars
    try:
        from .config import bm25_scan_bounds
        max_docs, max_chars_per_doc, page_size = bm25_scan_bounds(
            max_docs, max_chars_per_doc, cap_page)
    except Exception:
        max_docs = max(1, min(int(max_docs), 3000))
        max_chars_per_doc = max(1, int(max_chars_per_doc))
        page_size = min(max_docs, 1000)
    # Hard early-exit: never score beyond the doc cap; the flag is recorded
    # for diagnostics so truncation is observable, never silent.
    truncated_corpus = len(documents) > max_docs
    if truncated_corpus:
        documents = documents[:max_docs]
    try:
        from .config import truncate_for_bm25 as _trunc
    except Exception:
        _trunc = lambda t, c=None: (t or "")[:int(c or max_chars_per_doc)]  # noqa: E731
    # Deterministic bound: score pages of the (already relevance-unordered)
    # corpus and keep the global top-``limit``. Page size keeps peak memory
    # flat regardless of library size.
    limit = max(0, int(limit))
    scored: list[tuple[float, int]] = []
    offset = 0
    total = len(documents)
    while offset < total:
        page = documents[offset:offset + page_size]
        counts = [Counter(terms(_trunc(doc.page_content or "", max_chars_per_doc))) for doc in page]
        lengths = [sum(count.values()) for count in counts]
        average = sum(lengths) / max(1, len(lengths)) or 1.0
        frequency = Counter(term for count in counts for term in count)
        for local, count in enumerate(counts):
            score = 0.0
            for term in set(query_terms):
                tf = count.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (len(counts) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[local] / average))
            if score > 0:
                scored.append((score, offset + local))
        offset += page_size
        # Never hold more than the bounded corpus: drop to top candidates.
        if len(scored) > max(limit * 4, limit + 50):
            scored.sort(key=lambda item: (-item[0], item[1]))
            del scored[max(limit * 2, limit + 25):]
    scored.sort(key=lambda item: (-item[0], item[1]))
    result: list[Document] = []
    for rank, (score, index) in enumerate(scored[:limit], start=1):
        result.append(
            _annotate_retrieval(
                documents[index],
                retrieval_bm25_score=round(score, 8),
                retrieval_bm25_rank=rank,
            )
        )
    if truncated_corpus:
        try:
            from .diagnostics import record as _record
            _record("bm25_corpus_truncated", True)
        except Exception:
            pass
    return result
