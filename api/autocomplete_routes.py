"""Data-aware autocomplete routes (pro.docx sections 55-58, 60).

Returns suggestions generated from the actual connected database —
not hardcoded static lists.  Builds a lightweight in-memory ``data_index``
of: table names, metric column names, date column names, categorical
column names, and the most common distinct values for each categorical
column.  Suggestions are then derived by matching the user's partial
input against this index with typo tolerance (difflib SequenceMatcher
ratio), so the user can type ``ali`` and ARIA suggests ``Ali Traders``
if that value is actually present in the data.
"""

import difflib
import json
import re
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from core.config import get_session, save_session
from core.data_intelligence import (
    hint_hit as _hint_hit,
    is_id_like_name as _is_id_like_name,
    name_tokens as _name_tokens,
)
from core.deps import get_current_user

router = APIRouter()


class AutocompleteRequest(BaseModel):
    partial: str = ""
    limit: int = 8

# In-memory data_index cache: keyed by absolute schema_path, with file
# mtime as a freshness check.  The index is small (distinct values per
# categorical column, capped), so per-user memory pressure is minimal.
_INDEX_CACHE: dict[str, tuple[float, dict]] = {}
_INDEX_LOCK = threading.RLock()

# Cap how many distinct values we index per column — keeps the index
# bounded for wide cardinalities.  Top-N by frequency is what we store.
_MAX_DISTINCT_PER_COLUMN = 25
# Hard cap on the total cached entries.
_MAX_INDEX_ENTRIES = 100

# Words that signal an analytical intent — used to build template-style
# completions ("Top {cat} by {metric}", etc.) when the user types
# an incomplete phrase like "top" or "top p".  All derived from the
# actual data model — never hardcoded business content.
_INTENT_WORDS = {
    "top": "top",
    "best": "top",
    "highest": "top",
    "worst": "bottom",
    "lowest": "bottom",
    "bottom": "bottom",
    "sum": "sum",
    "total": "total",
    "average": "average",
    "avg": "average",
    "mean": "average",
    "count": "count",
    "how": "count",
    "many": "count",
    "revenue": "revenue",
    "sales": "revenue",
    "income": "revenue",
    "turnover": "revenue",
    "quantity": "quantity",
    "qty": "quantity",
    "units": "quantity",
    "trend": "trend",
    "over": "trend",
    "compare": "compare",
    "vs": "compare",
    "versus": "compare",
    "against": "compare",
    "difference": "compare",
    "by": "group_by",
    "per": "group_by",
}

_NUMERIC_HINT = {"amount", "price", "cost", "revenue", "sales", "total", "sum",
                 "count", "quantity", "qty", "value", "profit", "income",
                 "balance", "pay", "discount", "tax", "fee", "rate",
                 "percent", "number", "num", "stock", "weight", "score"}


# ---------------------------------------------------------------------------
# DB engine construction (mirrors api/kpi_routes._build_engine so the
# autocomplete endpoint handles both relational + SQLite/CSV sources
# consistently).  Duplicated rather than imported to avoid coupling
# unrelated route modules.
# ---------------------------------------------------------------------------
def _build_engine(session):
    db_config = dict(session.get("db") or {})
    if not db_config or session.get("source_type") != "relational":
        sqlite_path = session.get("db_uri") or session.get("csv_db_path")
        if sqlite_path:
            if sqlite_path.startswith("sqlite:///"):
                uri = sqlite_path
            elif sqlite_path.startswith("sqlite://"):
                uri = sqlite_path
            else:
                uri = f"sqlite:///{sqlite_path}"
            return create_engine(uri), "sqlite", "sqlite"
        return None, "unknown", "unknown"
    try:
        from core.secret_box import decrypt as _decrypt_secret
    except Exception:
        _decrypt_secret = None
    if db_config.get("password") and _decrypt_secret:
        try:
            db_config["password"] = _decrypt_secret(db_config["password"])
        except Exception:
            pass
    db_type = session.get("db_type", "postgresql")
    user = quote_plus(str(db_config.get("user", "")))
    password = quote_plus(str(db_config.get("password", "")))
    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    db = db_config.get("db", "")
    if not db:
        raise HTTPException(status_code=400, detail="No database configured.")
    _cargs: dict = {}
    try:
        from core.db import search_path_connect_args as _spa_a
        _sch_a = [str(s).strip() for s in (session.get("schemas") or [])
                  if str(s or "").strip()]
        _db_schema = db_config.get("db_schema", "")
        if _db_schema:
            for _s in _db_schema.split(","):
                _s = _s.strip()
                if _s and _s not in _sch_a:
                    _sch_a.append(_s)
        _cargs = _spa_a(db_type, _sch_a)
    except Exception:
        _cargs = {}
    if db_type == "mysql":
        uri = (f"mysql+pymysql://{user}:{password}"
               f"@{host}:{port}/{db}")
    else:
        uri = (f"postgresql://{user}:{password}"
               f"@{host}:{port}/{db}")
    return create_engine(uri, pool_pre_ping=True,
                         **({"connect_args": _cargs} if _cargs else {})), session.get("dialect", "postgresql"), db_type


def _quote(name, dialect):
    # Dotted keys (sales.employee) quote per segment, never as one name.
    try:
        segs = [s for s in str(name or "").split(".") if s]
        if len(segs) > 1:
            if dialect == "mysql":
                return ".".join(f"`{s.replace('`', '``')}`" for s in segs)
            return ".".join(
                f'"{s.replace(chr(34), chr(34) + chr(34))}"' for s in segs)
    except Exception:
        pass
    if dialect == "mysql":
        return f"`{name.replace('`', '``')}`"
    return f'"{name.replace(chr(34), chr(34)+chr(34))}"'


def _column_iter(table_info):
    raw = table_info.get("columns") or {}
    if isinstance(raw, dict):
        yield from raw.items()
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item.get("column", ""), item
            else:
                yield str(item), {}


def _looks_like_date(col_name, col_info):
    low = col_name.lower()
    if _hint_hit(col_name, ("date", "time", "created", "updated", "day",
                            "month", "year", "period", "quarter", "week")):
        return True
    dtype = str(col_info.get("dtype", col_info.get("data_type", ""))).lower()
    return any(t in dtype for t in ("date", "time", "timestamp"))


def _looks_like_numeric(col_name, col_info):
    if _looks_like_date(col_name, col_info):
        return False
    if _is_id_like(col_name):
        return False
    if _hint_hit(col_name, _NUMERIC_HINT):
        return True
    dtype = str(col_info.get("dtype", col_info.get("data_type", ""))).lower()
    return any(t in dtype for t in ("int", "float", "decimal", "numeric",
                                    "bigint", "double", "real"))


def _looks_like_categorical(col_name, col_info):
    """Heuristic: text/string column with short, label-like values."""
    if _looks_like_numeric(col_name, col_info) or _looks_like_date(col_name, col_info):
        return False
    if _hint_hit(col_name, ("name", "type", "category", "region", "status",
                            "group", "kind", "label", "tag", "class")):
        return True
    dtype = str(col_info.get("dtype", col_info.get("data_type", ""))).lower()
    return any(t in dtype for t in ("char", "text", "varchar", "string"))


# ---------------------------------------------------------------------------
# Data index construction
# ---------------------------------------------------------------------------
def _build_data_index(session, schema_path: Path):
    """Read schema mapping + sample distinct values to build a small
    in-memory index describing the actual database.  Cached per
    schema_path + mtime to avoid re-querying the database on every
    keystroke.  Returns a dict suitable for matching against partial
    user input."""
    if not schema_path.exists():
        return {"tables": [], "metrics": [], "categories": [],
                "dates": [], "values": [], "ready": False}

    try:
        mapping = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        return {"tables": [], "metrics": [], "categories": [],
                "dates": [], "values": [], "ready": False}

    tables = mapping.get("tables", {})
    if not tables:
        return {"tables": [], "metrics": [], "categories": [],
                "dates": [], "values": [], "ready": False}

    engine, dialect, _ = _build_engine(session)
    if engine is None:
        return {"tables": [], "metrics": [], "categories": [],
                "dates": [], "values": [], "ready": False}

    metric_cols: list[tuple[str, str]] = []     # (table, col)
    cat_cols: list[tuple[str, str]] = []        # (table, col)
    date_cols: list[tuple[str, str]] = []       # (table, col)
    table_names: list[str] = []

    for tname, tinfo in tables.items():
        table_names.append(tname)
        for col_name, col_info in _column_iter(tinfo):
            if not col_name:
                continue
            if _looks_like_date(col_name, col_info):
                date_cols.append((tname, col_name))
            elif _looks_like_numeric(col_name, col_info):
                metric_cols.append((tname, col_name))
            elif _looks_like_categorical(col_name, col_info):
                cat_cols.append((tname, col_name))

    # Sample distinct values from categorical columns — this is the
    # "actual data" feed that makes autocomplete data-aware (section 56).
    distinct_values: list[dict] = []
    if cat_cols:
        try:
            with engine.connect() as conn:
                # Cap total samples to keep the index small.
                for tname, col_name in cat_cols[:20]:
                    q = (f"SELECT {_quote(col_name, dialect)} AS v, "
                         f"COUNT(*) AS c FROM {_quote(tname, dialect)} "
                         f"WHERE {_quote(col_name, dialect)} IS NOT NULL "
                         f"GROUP BY {_quote(col_name, dialect)} "
                         f"ORDER BY c DESC LIMIT {_MAX_DISTINCT_PER_COLUMN}")
                    try:
                        rows = conn.execute(text(q)).fetchall()
                    except Exception:
                        continue
                    for r in rows:
                        v = r[0]
                        if v is None:
                            continue
                        s = str(v).strip()
                        if not s:
                            continue
                        distinct_values.append({
                            "value": s,
                            "table": tname,
                            "column": col_name,
                            "count": int(r[1] or 0),
                        })
        except Exception:
            pass
        finally:
            try:
                engine.dispose()
            except Exception:
                pass

    return {
        "tables": table_names,
        "metrics": [{"table": t, "column": c, "name": c} for t, c in metric_cols],
        "categories": [{"table": t, "column": c, "name": c} for t, c in cat_cols],
        "dates": [{"table": t, "column": c, "name": c} for t, c in date_cols],
        "values": distinct_values,
        "ready": True,
    }


def _get_data_index(session, user_id=None):
    """Return cached data_index for the session's schema, rebuilding if
    the schema file mtime changed."""
    schema_path: Path = session.get("schema_path")
    if not schema_path or not isinstance(schema_path, Path):
        try:
            schema_path = Path(str(schema_path)) if schema_path else None
        except Exception:
            schema_path = None
    if not schema_path:        return {"tables": [], "metrics": [], "categories": [],
                "dates": [], "values": [], "ready": False}

    try:
        mtime = schema_path.stat().st_mtime if schema_path.exists() else 0.0
    except Exception:
        mtime = 0.0
    # Key includes the user: two users must never share an index even if
    # paths ever collide (linked/shared DBs, tmp paths in tests).
    try:
        _uid = int(str(user_id)) if user_id is not None else 0
    except (TypeError, ValueError):
        _uid = 0
    key = (_uid, str(schema_path))
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached and cached[0] == mtime:
            # LRU touch: reinsert so eviction drops least-recently-used.
            _INDEX_CACHE.pop(key, None)
            _INDEX_CACHE[key] = cached
            return cached[1]
        # Evict least-recently-used if at capacity
        if len(_INDEX_CACHE) >= _MAX_INDEX_ENTRIES and key not in _INDEX_CACHE:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
        idx = _build_data_index(session, schema_path)
        _INDEX_CACHE[key] = (mtime, idx)
        return idx


# ---------------------------------------------------------------------------
# Suggestion ranking
# ---------------------------------------------------------------------------
def _normalize(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _match_quality(query, candidate):
    """Match quality tier: 3 = prefix, 2 = substring, 1 = fuzzy, 0 = none.

    Prefix matches carry the strongest signal (typing "sal" almost surely
    means the Sales column, not a fuzzy value hit), so ranking scores
    follow the tier instead of rating every hit equally.
    """
    if not query:
        return 0
    q = query.lower()
    c = candidate.lower()
    if c.startswith(q) or any(w.startswith(q) for w in c.split()):
        return 3
    if q in c:
        return 2
    # Fuzzy: allow genuine typos (section 59).  The bar sits well above
    # coincidental 3-letter overlaps ("what" must not match "Art") while
    # real typos ("salse" -> "sales") still clear it.  Comparison is
    # two-sided so it stays fair regardless of string length.
    if len(q) >= 3:
        # Cap the comparison window to the longer of |query| and |candidate|
        window = c[:max(len(q), len(c))]
        if _fuzzy_ratio(q, window) >= 0.65:
            return 1
    return 0


def _partial_match(query, candidate):
    """True if ``query`` is a prefix or fuzzy substring of ``candidate``."""
    if not query:
        return True
    return _match_quality(query, candidate) > 0


def _is_id_like(name: str) -> bool:
    # Separator-agnostic (shared helper): "order_id", "Order ID" and
    # "postal_code" are identifiers; "country" is not.
    return bool(_is_id_like_name(name))


def _rank_suggestions(index, partial, limit):
    """Build ranked list of suggestion dicts based on the partial input.

    Each suggestion has the shape:
        {"text": "<full question>", "kind": "value"|"template"|"metric", "score": float}

    The score is in [0, 1] for client-side ranking transparency.  All
    suggestions must come from the actual data — never static strings.
    """
    partial_n = _normalize(partial)
    if len(partial_n) < 1:
        return []

    suggestions: list[dict] = []
    seen: set[str] = set()

    def _add(text, kind, score):
        key = text.lower().strip()
        if not key or key in seen:
            return
        seen.add(key)
        suggestions.append({"text": text, "kind": kind, "score": round(score, 3)})

    # 1) Value-level matches: actual distinct values from the database.
    #    This is the highest signal — typing "ali" should surface
    #    "Ali Traders" if it exists in the data (section 58).
    #    Scores follow match quality so a prefix hit outranks a typo-level
    #    fuzzy hit, and pairs are capped so intent/metric templates below
    #    always have room in the limit window.
    _value_pairs = 0
    _best_value_q = 0
    if partial_n:
        for v in index.get("values", []):
            if _value_pairs >= 3:
                break
            val = v["value"]
            q = _match_quality(partial_n, val)
            if q == 0:
                # Also try matching against the *first* word of multi-word
                # values (e.g. typing "ali" -> "Ali Traders").
                q = _match_quality(partial_n, val.split(" ")[0])
                if q == 0:
                    continue
            _best_value_q = max(_best_value_q, q)
            col = v.get("column", "value")
            # Dimension columns (text values) get contextual suggestions.
            # Wording is neutral (no hardcoded business nouns like "Sales").
            if q >= 3:
                _add(f"Filter {col} to {val}", "value", 0.95)
                _add(f"Show details for {val}", "value", 0.85)
            elif q == 2:
                _add(f"Filter {col} to {val}", "value", 0.82)
                _add(f"Show details for {val}", "value", 0.76)
            else:
                _add(f"Filter {col} to {val}", "value", 0.68)
                _add(f"Show details for {val}", "value", 0.62)
            _value_pairs += 1

    # 2) Template matches: build "Top {category} by {metric}" only if
    #    we actually have a category + a metric.  Triggered by typing
    #    intent words like "top", "best", "highest" (section 57).
    tokens = partial_n.split()
    intent_tokens = [t for t in tokens if t in _INTENT_WORDS]
    # Prefix expansion: incomplete words still signal intent ("tot"→total,
    # "ave"→average, "tre"→trend) so mid-typing always offers something.
    for _tok in tokens:
        if _tok in _INTENT_WORDS or len(_tok) < 3:
            continue
        for _word, _canon in _INTENT_WORDS.items():
            if _word.startswith(_tok):
                if _canon not in intent_tokens:
                    intent_tokens.append(_canon)
                break
    # Prefer non-ID dimensions for breakdowns (grouping by an ID column
    # is analytically weak). Fall back to all categories if needed.
    _cats = [c for c in index.get("categories", []) if not _is_id_like(c.get("name", ""))]
    _cats = _cats or index.get("categories", [])
    if intent_tokens and (_cats or index["metrics"]):
        for cat in _cats[:5]:
            for met in index.get("metrics", [])[:5]:
                if "top" in intent_tokens or "best" in intent_tokens or "highest" in intent_tokens:
                    _add(f"Top {cat['name'].replace('_', ' ')} by {met['name'].replace('_', ' ')}",
                         "template", 0.78)
                if "total" in intent_tokens or "sum" in intent_tokens:
                    _add(f"Total {met['name'].replace('_', ' ')} by {cat['name'].replace('_', ' ')}",
                         "template", 0.76)
                if "average" in intent_tokens or "avg" in intent_tokens:
                    _add(f"Average {met['name'].replace('_', ' ')} by {cat['name'].replace('_', ' ')}",
                         "template", 0.74)
                if "count" in intent_tokens or "how" in intent_tokens:
                    _add(f"Count of {cat['name'].replace('_', ' ')}", "template", 0.72)

    # 3) Column-level matches: typing the start of a column name.
    #    A prefix of a real column name is a strong signal (typing "sal"
    #    means the Sales column), so it outranks fuzzy value hits.
    if partial_n:
        for met in index.get("metrics", []):
            name = met["name"].replace("_", " ")
            qm = _match_quality(partial_n, name)
            if qm == 0:
                continue
            _first_cat = _cats[0]["name"].replace("_", " ") if _cats else "category"
            _add(f"Show {name} by {_first_cat}",
                 "metric", 0.90 if qm >= 3 else 0.68)
        for cat in _cats:
            name = cat["name"].replace("_", " ")
            qc = _match_quality(partial_n, name)
            if qc == 0:
                continue
            if index["metrics"]:
                m = index["metrics"][0]["name"].replace("_", " ")
                _add(f"Top {name} by {m}", "metric", 0.88 if qc >= 3 else 0.66)
        # Date columns: typing "order"/"month" should surface the trend the
        # schema can actually plot (only offered when a metric exists).
        for dt in index.get("dates", [])[:3]:
            name = dt["name"].replace("_", " ")
            qd = _match_quality(partial_n, name)
            if qd == 0:
                continue
            if index["metrics"]:
                m = index["metrics"][0]["name"].replace("_", " ")
                _add(f"{m.title()} by {name}", "metric", 0.88 if qd >= 3 else 0.66)

    # 4) Time-aware fallback (section 74).  If a date column exists and
    #    the user types a time word not already covered, offer a trend
    #    template.  This is data-driven (we only suggest it when we
    #    actually have a date column to plot).
    if any(w in partial_n for w in ("trend", "over", "by month", "by day", "over time")) \
            or "trend" in intent_tokens:
        if index["dates"] and index["metrics"]:
            m = index["metrics"][0]["name"].replace("_", " ")
            _add(f"{m.title()} trend over time", "template", 0.70)

    # 5) Question-starter fallback: "what", "which", "show…" carry no intent
    #    word, yet the dropdown must still respond with grounded starters
    #    from THIS schema rather than going blank. Also fires when the only
    #    hits so far are typo-level fuzzy matches — the user is clearly
    #    starting a fresh question, not filtering to that value.
    if not suggestions or _best_value_q < 2:
        _starts = ("what", "which", "show", "tell", "give", "list", "is",
                   "are", "how", "compare", "find", "get")
        _pn = partial_n.strip()
        _is_starter = any(_pn == w or _pn.startswith(w + " ") or (len(_pn) >= 2 and w.startswith(_pn))
                          for w in _starts)
        if _is_starter:
            _m0 = index["metrics"][0]["name"].replace("_", " ") if index.get("metrics") else None
            _c0 = _cats[0]["name"].replace("_", " ") if _cats else None
            if _m0 and _c0:
                _add(f"Top {_c0} by {_m0}", "template", 0.60)
                _add(f"Total {_m0} by {_c0}", "template", 0.59)
            elif _m0:
                _add(f"What is the total {_m0}?", "template", 0.60)
            if _c0:
                _add(f"Which {_c0} has the most records?", "template", 0.58)

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions[:limit]


# ---------------------------------------------------------------------------
# Follow-up context (section 61)
# ---------------------------------------------------------------------------
def _store_conversation(user_id, session, question, answer_summary):
    """Persist the last Q/A pair on the session so follow-ups can resolve
    context.  Uses save_session to flush to disk for cross-process
    consistency."""
    conv = session.get("conversation") or []
    conv.append({
        "question": question,
        "summary": answer_summary,
        "ts": time.time(),
    })
    # Keep only the last 5 turns to bound the payload.
    session["conversation"] = conv[-5:]
    save_session(user_id, session)


def _build_followup_goal(session, raw_question):
    """If ``raw_question`` looks like a follow-up (no clear entity),
    prepend the previous question's context so the goal agent has
    something to anchor to.  This implements section 61."""
    conv = session.get("conversation") or []
    if not conv:
        return raw_question
    last = conv[-1]
    prev_q = (last.get("question") or "").strip()
    if not prev_q:
        return raw_question
    # Heuristic: very short questions, or questions that start with
    # follow-up words, get the previous question prepended.
    followup_signals = (
        "how much", "what about", "only for", "in 20", "compare with",
        "compare to", "vs ", "and ", "also", "now", "then", "next",
        "why", "show me", "give me", "more", "less", "filter",
    )
    q_lower = raw_question.lower().strip()
    if len(q_lower.split()) <= 6 or any(q_lower.startswith(s) for s in followup_signals):
        return f"Previous question: {prev_q}\nFollow-up: {raw_question}"
    return raw_question


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/api/autocomplete")
def autocomplete_get(
    partial: str = "",
    limit: int = 8,
    user: dict = Depends(get_current_user),
):
    """GET autocomplete — for simple browser fetch with query params."""
    return _autocomplete(partial, limit, user)


@router.post("/api/autocomplete")
def autocomplete_post(
    body: AutocompleteRequest,
    user: dict = Depends(get_current_user),
):
    """POST autocomplete — accepts JSON body."""
    return _autocomplete(body.partial, body.limit, user)


def _autocomplete(partial: str, limit: int, user: dict):
    partial = (partial or "").strip()
    limit = max(1, min(int(limit or 8), 20))

    session = get_session(user["user_id"])

    # Even when not connected, the call should not 500 — return an
    # empty list so the frontend can keep working.
    if not session.get("db") and session.get("source_type") != "semi-structured":
        return {"ok": True, "suggestions": [], "ready": False,
                "index_summary": {"tables": 0, "values": 0}}

    if not session.get("schema_path") or not Path(str(session["schema_path"])).exists():
        return {"ok": True, "suggestions": [], "ready": False,
                "index_summary": {"tables": 0, "values": 0}}

    index = _get_data_index(session, user["user_id"])
    suggestions = _rank_suggestions(index, partial, limit)

    return {
        "ok": True,
        "ready": bool(index.get("ready")),
        "suggestions": suggestions,
        "index_summary": {
            "tables": len(index.get("tables", [])),
            "metrics": len(index.get("metrics", [])),
            "categories": len(index.get("categories", [])),
            "dates": len(index.get("dates", [])),
            "values": len(index.get("values", [])),
        },
    }


@router.post("/api/conversation/store")
def store_conversation(request: Request, payload: dict,
                       user: dict = Depends(get_current_user)):
    """Persist a Q/A pair as the most recent conversation turn.

    The frontend calls this after rendering an Ask ARIA answer so that
    subsequent follow-up questions can resolve against this context
    (section 61).  The endpoint is intentionally lightweight — it only
    stores what the UI tells it, and the goal agent re-derives the
    answer from the database when the follow-up is asked.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    question = (payload.get("question") or "").strip()
    summary = (payload.get("summary") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    session = get_session(user["user_id"])
    _store_conversation(user["user_id"], session, question, summary)
    return {"ok": True, "turns": len(session.get("conversation") or [])}


@router.post("/api/conversation/resolve_followup")
def resolve_followup(request: Request, payload: dict,
                     user: dict = Depends(get_current_user)):
    """Given a possibly-incomplete follow-up, return the resolved goal
    text.  Returns the original question unchanged if there is no
    prior conversation, or prepends ``Previous question:`` so the goal
    agent can see the context."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    session = get_session(user["user_id"])
    resolved = _build_followup_goal(session, question)
    return {
        "ok": True,
        "original": question,
        "resolved": resolved,
        "has_context": resolved != question,
    }
