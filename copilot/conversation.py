"""Conversation memory for Copilot follow-ups.

Stores a bounded, per-user transcript of (question, resolved context, answer)
turns and provides deterministic coreference resolution so short follow-ups
("what about last year?", "break that down by region", "show me top 5 instead")
reuse the previous entity/measure/grouping without an LLM round-trip.

Provider-independent: no LLM is involved here.
"""

from __future__ import annotations

import re
import threading
import time

_MAX_TURNS = 12
_TTL_SECONDS = 6 * 60 * 60


def _row_bucket(count) -> str:
    """Coarse row-count bucket (never a value leak, only a magnitude)."""
    try:
        n = int(count) if count is not None else None
    except (TypeError, ValueError):
        return "unknown"
    if n is None:
        return "unknown"
    if n <= 0:
        return "empty"
    if n < 10:
        return "tiny"
    if n < 100:
        return "small"
    if n < 10000:
        return "medium"
    if n < 1000000:
        return "large"
    return "xlarge"


def _table_signature(name: str, info: dict) -> str:
    """Secret-free per-table shape: column names+types + row-count bucket."""
    info = info if isinstance(info, dict) else {}
    cols = info.get("columns", [])
    if isinstance(cols, dict):
        items = sorted((str(k), str((v or {}).get("data_type", "")))
                       for k, v in cols.items())
    else:
        items = []
        for c in cols or []:
            if isinstance(c, dict):
                items.append((str(c.get("column") or c.get("name") or ""),
                              str(c.get("data_type", "") or "")))
            else:
                items.append((str(c), ""))
        items = sorted(items)
    col_part = ";".join(f"{n}:{d}".strip().lower() for n, d in items if n)
    count = info.get("row_count")
    if count is None:
        for stats in ((info.get("null_stats") or {}).values()):
            if isinstance(stats, dict) and stats.get("total_rows") is not None:
                count = stats.get("total_rows")
                break
    return f"{name}[{col_part}]({_row_bucket(count)})"


def _edge_signature(mapping: dict) -> str:
    """Secret-free relationship shape: sorted structural edge descriptors."""
    mapping = mapping if isinstance(mapping, dict) else {}
    edges: set[str] = set()
    for key in ("declared_relationships", "inferred_relationships",
                "relationship_edges"):
        for e in (mapping.get(key) or []):
            if isinstance(e, dict):
                parts = []
                for k in sorted(e.keys()):
                    v = e[k]
                    if isinstance(v, (str, int, float, bool)):
                        parts.append(f"{k}={v}")
                    elif isinstance(v, (list, tuple)) and all(
                            isinstance(x, (str, int, float)) for x in v):
                        parts.append(f"{k}={','.join(str(x) for x in v)}")
                if parts:
                    edges.add("|".join(parts))
            elif e is not None:
                edges.add(str(e))
    return ";".join(sorted(edges))


def _doc_ids(document_id=None, linked: dict | None = None,
             mapping: dict | None = None) -> str:
    """Sorted document identity (ids only, never content)."""
    ids: set[str] = set()

    def _add(value) -> None:
        if isinstance(value, (list, tuple, set)):
            for v in value:
                _add(v)
        elif value:
            ids.add(str(value))

    _add(document_id)
    if isinstance(linked, dict):
        _add(linked.get("document_id"))
        _add(linked.get("document_ids"))
    if isinstance(mapping, dict):
        _add(mapping.get("document_id"))
        _add(mapping.get("document_ids"))
    return ",".join(sorted(ids))


def dataset_fingerprint(*, mapping=None, document_id=None, linked=None) -> str:
    """Stable, secret-free identity of the dataset(s) a turn belongs to.

    Combines the primary schema (database name + per-table column names/types
    + row-count buckets + relationship edge set) with the multi-source
    pointers (document ids, linked database name + its own table shapes).
    Same dataset -> same fingerprint; a dataset switch (even one that keeps
    the database/table names but changes columns, edges, docs or magnitude)
    -> a different fingerprint, so stale turns are ignored instead of
    reused. Never includes credentials, URIs or values: only names, types,
    edge structure, document ids and coarse buckets.
    """
    mapping = mapping if isinstance(mapping, dict) else {}
    tables = mapping.get("tables") or {}
    db = str(mapping.get("database") or mapping.get("db_name") or "")
    sigs = ",".join(_table_signature(t, tables.get(t))
                     for t in sorted(tables))
    edges = _edge_signature(mapping)
    docs = _doc_ids(document_id, linked if isinstance(linked, dict) else None,
                    mapping)
    linked = linked if isinstance(linked, dict) else {}
    linked_mapping = linked.get("mapping") if isinstance(
        linked.get("mapping"), dict) else {}
    linked_tables = linked_mapping.get("tables") or {}
    linked_db = str(linked.get("db_name")
                    or linked_mapping.get("database") or "")
    linked_sigs = ",".join(_table_signature(t, linked_tables.get(t))
                           for t in sorted(linked_tables))
    linked_edges = _edge_signature(linked_mapping)
    return "|".join([db, sigs, edges, docs, linked_db, linked_sigs,
                      linked_edges])


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


class ConversationMemory:
    """Per-user bounded conversation memory (process-local, TTL'd)."""

    _lock = threading.RLock()
    _store: dict[str, list[dict]] = {}
    _touched: dict[str, float] = {}

    def __init__(self, user_id):
        self.user_id = str(user_id)

    # -- storage ---------------------------------------------------------

    def turns(self, dataset: str | None = None) -> list[dict]:
        with self._lock:
            self._evict()
            all_turns = list(self._store.get(self.user_id, []))
        if dataset is None:
            return all_turns
        # Dataset switch: only turns from the current dataset are visible,
        # so stale schema/tables/SQL from another source can never leak in.
        # Turns stored before dataset tagging (dataset=None) are treated as
        # belonging to an unknown dataset and are NOT reused across switches.
        return [t for t in all_turns if t.get("dataset") == dataset]

    def last(self, dataset: str | None = None) -> dict | None:
        turns = self.turns(dataset=dataset)
        return turns[-1] if turns else None

    def append(self, *, question: str, context: dict | None = None,
               answer: str = "", sql: str | None = None,
               dataset: str | None = None) -> None:
        with self._lock:
            self._evict()
            bucket = self._store.setdefault(self.user_id, [])
            bucket.append({
                "question": question,
                "context": context or {},
                "answer": answer,
                "sql": sql,
                "dataset": dataset,
                "ts": time.time(),
            })
            del bucket[:-_MAX_TURNS]
            self._touched[self.user_id] = time.time()

    def clear(self) -> None:
        with self._lock:
            self._store.pop(self.user_id, None)
            self._touched.pop(self.user_id, None)

    def _evict(self) -> None:
        now = time.time()
        stale = [u for u, t in self._touched.items() if now - t > _TTL_SECONDS]
        for u in stale:
            self._store.pop(u, None)
            self._touched.pop(u, None)

    # -- coreference -----------------------------------------------------

    # Strong reference phrases that only make sense with prior context.
    _FOLLOWUP_MARKERS = (
        "what about", "how about", "and last", "and the same", "the same",
        "break that", "break it down", "those", "that one", "why", "instead",
        "drill down", "go deeper", "expand on", "for those", "for that",
    )

    @classmethod
    def is_followup(cls, question: str) -> bool:
        """Heuristic: short, reference-heavy questions that need prior context.

        Deliberately conservative — a fully self-contained question (even a
        long one that contains a marker word like 'that') must NOT be rewritten
        with stale context.
        """
        q = _norm(question)
        if not q:
            return False
        if len(q.split()) <= 4:
            return True
        words = set(re.findall(r"[a-z0-9]+", q))
        for marker in cls._FOLLOWUP_MARKERS:
            if " " in marker:
                if marker in q:
                    return True
            elif marker in words:
                return True
        return False

    def resolve(self, question: str, dataset: str | None = None) -> str:
        """Return `question` enriched with prior-turn references when needed.

        Deterministic and conservative: only prepends context that actually
        exists, and never invents entities. A standalone question is returned
        unchanged. When `dataset` is given, only turns from that dataset are
        consulted, so a follow-up after a dataset switch is never rewritten
        with stale context.
        """
        if not self.is_followup(question):
            return question
        prev = self.last(dataset=dataset)
        if not prev:
            return question
        ctx = prev.get("context") or {}
        bits = []
        prev_q = prev.get("question")
        if prev_q:
            bits.append(f"Previous question: {prev_q}")
        for key, label in (("entity", "entity"), ("measure", "measure"),
                           ("group_by", "grouping"), ("tables", "tables")):
            val = ctx.get(key)
            if isinstance(val, (list, tuple)) and val:
                bits.append(f"Previous {label}: {', '.join(str(v) for v in val[:4])}")
            elif isinstance(val, str) and val:
                bits.append(f"Previous {label}: {val}")
        if not bits:
            return question
        return "\n".join(bits) + f"\nCurrent question: {question}"