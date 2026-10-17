"""Database-aware SQL engine for Copilot.

Generates, validates and executes read-only analytical SQL against PostgreSQL,
MySQL or SQLite using the same code path. Reuses ARIA's existing read-only
guard (core.validation) and the shared SQLAlchemy engine built by the session,
so there is ONE database abstraction — not one per engine.

Dialect differences (date truncation, identifier quoting, LIMIT) are handled
in the prompt and the validator, never with per-schema hardcoding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import create_engine, text

from core.validation import (
    strip_code_fences,
    unknown_sql_columns,
    unknown_sql_tables,
)

log = logging.getLogger(__name__)


# Bounded engine cache: avoids pinning credential-bearing URIs forever
# (previous @lru_cache never closed/expired). TTL + explicit dispose keep
# memory bounded; cache key is a redacted fingerprint, never the raw URI.
_ENGINE_CACHE: dict = {}
_ENGINE_CACHE_TTL_S = 600
_ENGINE_CACHE_MAX = 8
try:
    import threading as _threading
    _ENGINE_LOCK = _threading.Lock()
except Exception:  # pragma: no cover
    _ENGINE_LOCK = None


def _cache_key(db_uri: str) -> str:
    import hashlib as _hl
    try:
        return _hl.sha256(db_uri.encode("utf-8")).hexdigest()[:32]
    except Exception:
        return str(abs(hash(db_uri)))


def _get_engine(db_uri: str, schemas=None):
    import time as _time
    try:
        _sch = ",".join(sorted({str(s).strip() for s in (schemas or []) if str(s or "").strip()}))
    except Exception:
        _sch = ""
    key = _cache_key(str(db_uri) + "|" + _sch)
    now = _time.monotonic()
    lock = _ENGINE_LOCK
    try:
        if lock is not None:
            lock.acquire()
        entry = _ENGINE_CACHE.get(key)
        if entry is not None:
            eng, _stored_uri, ts = entry
            if now - ts < _ENGINE_CACHE_TTL_S:
                return eng
            try:
                eng.dispose()
            except Exception:
                pass
            _ENGINE_CACHE.pop(key, None)
        _cargs: dict = {}
        try:
            from core.db import search_path_connect_args as _spa
            _dt = ("mysql" if str(db_uri).startswith("mysql")
                   else "sqlite" if str(db_uri).startswith("sqlite") else "postgresql")
            _cargs = _spa(_dt, [s for s in _sch.split(",") if s])
        except Exception:
            _cargs = {}
        eng = create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800,
                            **({"connect_args": _cargs} if _cargs else {}))
        if len(_ENGINE_CACHE) >= _ENGINE_CACHE_MAX:
            try:
                oldest = min(_ENGINE_CACHE.items(), key=lambda kv: kv[1][2])[0]
                try:
                    _ENGINE_CACHE[oldest][0].dispose()
                except Exception:
                    pass
                _ENGINE_CACHE.pop(oldest, None)
            except Exception:
                pass
        _ENGINE_CACHE[key] = (eng, db_uri, now)
        return eng
    finally:
        try:
            if lock is not None:
                lock.release()
        except Exception:
            pass


def dispose_engines() -> None:
    """Close all cached engines (tests / shutdown). Never raises."""
    try:
        if _ENGINE_LOCK is not None:
            _ENGINE_LOCK.acquire()
        for _k, (eng, _u, _t) in list(_ENGINE_CACHE.items()):
            try:
                eng.dispose()
            except Exception:
                pass
        _ENGINE_CACHE.clear()
    except Exception:
        pass
    finally:
        try:
            if _ENGINE_LOCK is not None:
                _ENGINE_LOCK.release()
        except Exception:
            pass

# Deterministic read-only guard: reject anything that can mutate data/schema.
_LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bgroup\s+by\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)


@dataclass
class SqlResult:
    ok: bool
    sql: str | None = None
    columns: list[str] = None
    rows: list[dict] = None
    row_count: int = 0
    error: str | None = None
    attempts: int = 1
    dialect: str = "postgresql"

    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.rows is None:
            self.rows = []


def is_read_only(sql: str) -> bool:
    """True only for SELECT / WITH ... SELECT statements. Deterministic.

    Delegates to the single shared guard in core.validation so Copilot and
    the Goal Agent enforce the same contract.
    """
    from core.validation import is_read_only_sql as _shared_is_read_only
    return _shared_is_read_only(sql)


def clean_sql(raw: str) -> str:
    """Extract a single SQL statement from possibly fenced/chatty LLM output."""
    if not raw:
        return ""
    candidate = strip_code_fences(raw) or raw
    # Keep everything from the first SELECT/WITH keyword onward.
    m = re.search(r"\b(select|with)\b", candidate, re.IGNORECASE)
    if not m:
        return candidate.strip()
    stmt = candidate[m.start():].strip()
    # For CTEs (WITH ... ; SELECT ...), find the last semicolon-terminated
    # statement so the full CTE + final SELECT is preserved.
    semi = stmt.rfind(";")
    if semi != -1:
        stmt = stmt[:semi]
    return stmt.strip()


def _dialect_rules(dialect: str) -> str:
    if dialect == "mysql":
        return (
            "MySQL syntax: quote identifiers with backticks (always quote names "
            "containing spaces or special characters, e.g. `Order Date`), use "
            "DATE_FORMAT(col, '%Y-%m') for month grouping and CURDATE()/NOW() "
            "for current time. There is no ILIKE; use LIKE. Use LIMIT for row caps."
        )
    if dialect == "sqlite":
        return (
            "SQLite syntax: use strftime('%Y-%m', col) for month grouping and "
            "date('now') for current date. Column names may contain spaces — "
            "always wrap such names in double quotes (e.g. \"Order Date\"). "
            "Use LIMIT for row caps."
        )
    return (
        "PostgreSQL syntax: use date_trunc('month', col) for month grouping and "
        "CURRENT_DATE for today. You may use ILIKE for case-insensitive match. "
        "Always double-quote identifiers containing spaces or mixed case. Use LIMIT."
    )


def _text_number_rule(dialect: str) -> str:
    """Prompt rule: aggregate TEXT-stored numbers via the safe coercion.

    Generic (no table/column names): unseen databases routinely store
    numbers as TEXT with currency symbols, thousand separators or '%'.
    A bare CAST(col AS NUMERIC) crashes on those values; the coercion
    strips formatting and yields NULL for unparsable values instead.
    """
    try:
        from core.text_numeric import coercion_sql as _coerce
        _expr = _coerce(dialect, '"COLUMN"')
    except Exception:
        _expr = None
    if not _expr:
        return ("- Never use bare CAST(text_col AS NUMERIC): TEXT columns may "
                "hold formatted numbers (currency symbols, commas, %). "
                "Strip formatting to NULL-safe numbers instead.")
    return ("- TEXT columns may store formatted numbers (currency symbols "
            "like \u20b9, thousand separators, %): NEVER use bare "
            "CAST(col AS NUMERIC/INTEGER) on them — it crashes on the first "
            "formatted value. For SUM/AVG/MIN/MAX over a TEXT column use "
            f"this safe coercion instead: {_expr} (replace \"COLUMN\" with "
            "the real column reference).")


_CAST_NUMERIC_RE = None


def _bare_cast_pattern():
    """CAST("col" AS <numeric type>) — the crash pattern on TEXT columns."""
    global _CAST_NUMERIC_RE
    if _CAST_NUMERIC_RE is None:
        import re as _re
        _CAST_NUMERIC_RE = _re.compile(
            r"CAST\s*\(\s*((?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
            r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)"
            r"\s+AS\s+(NUMERIC|INTEGER|INT|BIGINT|SMALLINT|FLOAT|REAL"
            r"|DOUBLE(?:\s+PRECISION)?|DECIMAL(?:\s*\([^)]*\))?|NUMERIC"
            r"(?:\s*\([^)]*\))?)\s*\)",
            _re.IGNORECASE)
    return _CAST_NUMERIC_RE


def _text_columns_of(tables: dict | None) -> set[str]:
    """Lowercased TEXT column names (qualified and bare) from the mapping."""
    out: set[str] = set()
    try:
        for t, info in (tables or {}).items():
            cols = (info or {}).get("columns", [])
            items = cols.values() if isinstance(cols, dict) else cols
            for c in items or []:
                name = (c.get("column") if isinstance(c, dict) else c) or ""
                dtype = (c.get("data_type") if isinstance(c, dict) else "") or ""
                dt = str(dtype).lower()
                if not str(name):
                    continue
                if dt == "text" or "varchar" in dt or "char" in dt:
                    out.add(str(name).strip().lower())
                    out.add(f"{t}.{name}".lower())
    except Exception:
        pass
    return out


def _rewrite_text_casts(sql: str, dialect: str, tables: dict | None) -> str:
    """Rewrite bare CASTs over TEXT columns into the NULL-safe coercion.

    Fully generic: only fires when the mapping declares the column TEXT
    (unknown dtypes are left untouched), so numeric columns and intended
    casts are never rewritten.
    """
    if not sql or not tables:
        return sql
    _texts = _text_columns_of(tables)
    if not _texts:
        return sql
    try:
        from core.text_numeric import coercion_sql as _coerce
    except Exception:
        return sql
    import re as _re

    def _fix(m):
        ref = m.group(1) or ""
        bare = _re.sub(r'["`\[\]]', "", ref).strip().lower()
        if bare in _texts:
            try:
                return _coerce(dialect, ref)
            except Exception:
                return m.group(0)
        return m.group(0)

    try:
        return _bare_cast_pattern().sub(_fix, sql)
    except Exception:
        return sql


def _date_trunc_pattern():
    """date_trunc('<grain>', ref) — crashes when ref is a TEXT date."""
    import re as _re
    return _re.compile(
        r"date_trunc\s*\(\s*'(month|quarter|year|week|day)'\s*,"
        r"\s*((?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
        r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)\s*\)",
        _re.IGNORECASE)


def _as_date_pattern():
    """CAST(ref AS DATE) or ref::date — crashes when ref is a TEXT date."""
    import re as _re
    return _re.compile(
        r"(?:CAST\s*\(\s*((?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
        r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)\s+AS\s+DATE\s*\)"
        r"|((?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
        r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?)::date)",
        _re.IGNORECASE)


def _rewrite_text_dates(sql: str, dialect: str, tables: dict | None) -> str:
    """Parse TEXT dates before date_trunc/CAST-AS-DATE (generic, no names).

    Unseen databases store dates as TEXT in local spellings (DD-MM-YYYY,
    ISO, YYYY-MM). date_trunc()/::date on those values crashes; the shared
    date_parse_expression tries DD-MM-YYYY then ISO and yields NULL (never
    an error) for the rest. Only mapping-declared TEXT columns are
    rewritten; native date columns pass through untouched.
    """
    if not sql or not tables or str(dialect).lower() not in (
            "postgresql", "mysql", "sqlite"):
        return sql
    _texts = _text_columns_of(tables)
    if not _texts:
        return sql
    try:
        from core.data_intelligence import date_parse_expression as _parse
    except Exception:
        return sql
    import re as _re

    _DATE_NAME_TOKENS = ("date", "time", "timestamp", "month", "year", "day",
                         "week", "quarter", "period")

    def _is_text_date(ref: str) -> bool:
        # TEXT storage + date-like name: a genuine date column in disguise.
        # (A date_trunc over a non-date TEXT column is a hallucination and
        # must keep failing loudly so the repair loop fixes the column.)
        bare = _re.sub(r'["`\[\]]', "", ref or "").strip().lower()
        if bare not in _texts:
            return False
        short = bare.split(".")[-1]
        return any(t in short for t in _DATE_NAME_TOKENS)

    def _fix_trunc(m):
        grain, ref = m.group(1), m.group(2)
        if _is_text_date(ref):
            try:
                return f"date_trunc('{grain}', {_parse(ref, dialect)})"
            except Exception:
                return m.group(0)
        return m.group(0)

    def _fix_cast(m):
        ref = m.group(1) or m.group(2) or ""
        if ref and _is_text_date(ref):
            try:
                return _parse(ref, dialect)
            except Exception:
                return m.group(0)
        return m.group(0)

    try:
        # CAST/::date first: the trunc rewrite inserts a parse expression
        # that itself contains ::date, and the trunc pattern only matches
        # plain column refs, so this order never double-wraps.
        sql = _as_date_pattern().sub(_fix_cast, sql)
        sql = _date_trunc_pattern().sub(_fix_trunc, sql)
    except Exception:
        pass
    return sql


def _split_ref(ref: str):
    """Split a dotted reference into segments without quotes/brackets."""
    import re as _re
    try:
        return [p for p in _re.sub(r'["`\[\]]', "", ref or "").split(".") if p]
    except Exception:
        return []


def _sql_table_tokens(sql: str) -> set[str]:
    """Bare table-name tokens appearing in FROM/JOIN (alias-aware base)."""
    import re as _re
    out: set[str] = set()
    try:
        _tref = (r"(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
                 r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))*")
        for _m in re.finditer(
                r"\b(?:FROM|JOIN)\s*(" + _tref + r")"
                r"(?:\s+(?:AS\s+)?(\w+))?", sql, re.IGNORECASE):
            for _seg in _split_ref(_m.group(1)):
                out.add(_seg.lower())
            if _m.group(2):
                out.add(_m.group(2).lower())
    except Exception:
        pass
    return out


def _rank_rels_for_sql(relationships, sql: str, limit: int = 30) -> list[str]:
    """Order validated edges by relevance to the attempted SQL's tables.

    Generic: edges touching the SQL's own tables (or their aliases) come
    first so the repair sees the bridge it needs instead of an arbitrary
    top-N cut that crowds it out.
    """
    try:
        _rels = [str(r) for r in (relationships or [])]
        if not _rels:
            return []
        _toks = _sql_table_tokens(sql)
        if not _toks:
            return _rels[:limit]
        def _score(r: str) -> int:
            _rl = r.lower()
            _hits = sum(1 for _t in _toks if _t and _t in _rl)
            return -_hits
        return sorted(_rels, key=_score)[:limit]
    except Exception:
        try:
            return list(relationships or [])[:limit]
        except Exception:
            return []


def _join_path_hint(relationships, table1: str, table2: str) -> str | None:
    """Shortest validated join path between two bare table names, else None.

    Generic BFS over the mapping's relationship lines so the repair can be
    told the exact bridge ("join A->B->C") instead of a flat edge list the
    model keeps misreading. Composite conditions ride along per pair.
    """
    import re as _re
    from collections import defaultdict as _dd, deque as _dq
    try:
        _adj: dict = _dd(list)
        _conds: dict = {}
        for _r in relationships or []:
            for _part in re.split(r"\bAND\b", str(_r), flags=re.IGNORECASE):
                _m = re.search(
                    r"([\w.\"`\[\]]+)\s*\.\s*([\w\"`\[\]]+)\s*=\s*"
                    r"([\w.\"`\[\]]+)\s*\.\s*([\w\"`\[\]]+)", _part)
                if not _m:
                    continue
                _a = _split_ref(_m.group(1))[-1].lower()
                _ca = _m.group(2).strip('"`[]').lower()
                _b = _split_ref(_m.group(3))[-1].lower()
                _cb = _m.group(4).strip('"`[]').lower()
                if not _a or not _b:
                    continue
                _adj[_a].append(_b)
                _adj[_b].append(_a)
                _conds.setdefault((_a, _b), []).append((_ca, _cb))
                _conds.setdefault((_b, _a), []).append((_cb, _ca))
        _s, _t = str(table1 or "").lower(), str(table2 or "").lower()
        if not _s or not _t or _s == _t:
            return None
        _prev: dict[str, str | None] = {_s: None}
        _dq_ = _dq([_s])
        while _dq_:
            _cur = _dq_.popleft()
            if _cur == _t:
                break
            for _nb in _adj.get(_cur, []):
                if _nb not in _prev:
                    _prev[_nb] = _cur
                    _dq_.append(_nb)
        if _t not in _prev:
            return None
        _path = [_t]
        while _path[-1] != _s:
            _path.append(_prev[_path[-1]])
        _path.reverse()
        _steps = []
        for _x, _y in zip(_path, _path[1:]):
            _cc = _conds.get((_x, _y), [])
            _on = " AND ".join(f"{_x}.{_c1} = {_y}.{_c2}" for _c1, _c2 in _cc)
            _steps.append((_y, _on))
        _txt = _path[0]
        for _y, _on in _steps:
            _txt += f" JOIN {_y} ON {_on}"
        return "FROM " + _txt
    except Exception:
        return None


def _bad_join_tables(sql: str, relationships) -> list[str] | None:
    """Bare table pair of the first join equality in `sql`, else None.

    Used to aim the bridge-path hint at the exact tables the model tried
    to join. Alias- and schema-tolerant like the check itself.
    """
    import re as _re
    try:
        _tref = (r"(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
                 r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))*")
        _aliases: dict[str, list[str]] = {}
        for _m in re.finditer(
                r"\b(?:FROM|JOIN)\s*(" + _tref + r")"
                r"(?:\s+(?:AS\s+)?(\w+))?", sql, re.IGNORECASE):
            _full = _split_ref(_m.group(1))
            _al = (_m.group(2) or "").lower()
            if _full and _al and _al not in (
                    "where", "group", "order", "limit", "join", "on",
                    "left", "right", "inner", "outer", "full", "cross"):
                _aliases[_al] = [s.lower() for s in _full]
        _eq_re = re.compile(
            r"(" + _tref + r"\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))"
            r"\s*=\s*(" + _tref + r"\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))",
            re.IGNORECASE)
        for _eq in _eq_re.finditer(sql):
            _lsegs, _rsegs = _split_ref(_eq.group(1)), _split_ref(_eq.group(2))
            if len(_lsegs) < 2 or len(_rsegs) < 2:
                continue
            _lt, _rt = _lsegs[:-1], _rsegs[:-1]
            _lt = _aliases.get(_lt[0].lower() if len(_lt) == 1 else "", _lt)
            _rt = _aliases.get(_rt[0].lower() if len(_rt) == 1 else "", _rt)
            if _lt and _rt and _lt[-1].lower() != _rt[-1].lower():
                return [_lt[-1], _rt[-1]]
    except Exception:
        pass
    return None


def _fix_unknown_columns(sql: str, tables: dict | None) -> str | None:
    """Deterministically repair hallucinated column names (generic).

    The model often emits a near-miss from memory (qty for orderqty,
    ProductName for name). For each table, map unknown referenced columns
    to the same table's real columns via exact/plural/substring/fuzzy
    match and rewrite. Returns fixed SQL, or None when nothing fixes
    cleanly (then the normal LLM repair loop handles it).
    """
    import re as _re
    import difflib as _dl
    if not sql or not tables:
        return None
    try:
        _cols_by_table: dict[str, set[str]] = {}
        for _t, _info in (tables or {}).items():
            _cols = (_info or {}).get("columns", [])
            _names = set()
            for _c in (_cols.values() if isinstance(_cols, dict) else _cols or []):
                _n = (_c.get("column") if isinstance(_c, dict) else _c) or ""
                if _n:
                    _names.add(str(_n))
            _cols_by_table[str(_t).lower()] = _names
            _cols_by_table[str(_t).split(".")[-1].lower()] = _names
        _refs: dict[str, str] = {}  # (table_lower, col_lower) -> fix or None
        _fixed = sql
        _changed = False
        # Never rewrite inside string literals ('foo.bar' is data, not SQL).
        _lit_spans = []
        try:
            for _lm in re.finditer(r"'(?:[^']|'')*'", sql):
                _lit_spans.append(_lm.span())
        except Exception:
            _lit_spans = []
        def _in_lit(_pos):
            return any(_a <= _pos < _b for _a, _b in _lit_spans)
        for _m in re.finditer(
                r"([A-Za-z_][\w$]*)\s*\.\s*[\"`\[]?([A-Za-z_][\w$]*)[\"`\]]?",
                sql):
            if _in_lit(_m.start()):
                continue
            _tq, _cq = _m.group(1).lower(), _m.group(2)
            if _tq in ("pg_catalog",) or not _cq:
                continue
            _known = _cols_by_table.get(_tq)
            if _known is None:
                # qualifier may be an alias: resolve via FROM/JOIN
                _am = re.search(
                    r"\b(?:FROM|JOIN)\s+((?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
                    r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))*)\s+"
                    + re.escape(_m.group(1)) + r"\b", sql, re.IGNORECASE)
                if _am:
                    _full = _re.sub(r'["`\[\]]', "", _am.group(1)).lower()
                    _known = _cols_by_table.get(_full,
                               _cols_by_table.get(_full.split(".")[-1]))
            if not _known:
                continue
            _cl = _cq.lower()
            if _cl in {c.lower() for c in _known}:
                continue
            _key = (_tq, _cl)
            if _key in _refs:
                _fix = _refs[_key]
            else:
                _fix = None
                _cands = [c for c in _known
                          if c.lower() == _cl or c.lower() == _cl + "s"
                          or (len(_cl) > 3 and c.lower().startswith(_cl))
                          or (len(c) > 3 and _cl.startswith(c.lower()))]
                # Generic business abbreviations (qty->quantity): the model
                # shortens names from memory; expand against same-table
                # columns containing the full word. Not schema-specific.
                if not _cands:
                    _ABBREV = {"qty": "quantity", "quant": "quantity",
                               "amt": "amount", "num": "number",
                               "desc": "description", "addr": "address",
                               "cust": "customer", "prod": "product",
                               "cat": "category", "tot": "total",
                               "val": "value", "rev": "revenue",
                               "avg": "average", "max": "maximum",
                               "min": "minimum"}
                    _exp = _ABBREV.get(_cl)
                    if _exp:
                        _cands = [c for c in _known
                                  if _exp in c.lower() or _cl in c.lower()]
                if not _cands:
                    _best, _score = None, 0.0
                    for _c in _known:
                        _r = _dl.SequenceMatcher(None, _cl, _c.lower()).ratio()
                        if _r > _score:
                            _best, _score = _c, _r
                    if _best and _score >= 0.85:
                        _cands = [_best]
                if len(_cands) == 1:
                    _fix = _cands[0]
                _refs[_key] = _fix
            if _fix:
                _fixed = re.sub(
                    r"(\b" + re.escape(_m.group(1)) + r"\s*\.\s*[\"`\[]?)"
                    + re.escape(_cq) + r"([\"`\]]?\b)", r"\g<1>" + _fix + r"\g<2>",
                    _fixed)
                _changed = True
        return _fixed if _changed else None
    except Exception:
        return None


def _invalid_join(sql: str, relationships) -> str | None:
    """First JOIN equality with no matching validated relationship, else None.

    Fully generic: edges come from the live mapping ("a.b.c = x.y.z");
    SQL aliases resolve via its own FROM/JOIN clauses; comparison is on
    bare segments so schema-qualified and bare SQL both validate. A missing
    edge list disables the check (never blocks). Self-joins always pass.
    """
    import re as _re
    if not sql or not relationships:
        return None
    try:
        _edges = []
        for _r in relationships:
            # A relationship line may carry several ANDed conditions for
            # one composite FK — split them so every condition is enforced.
            for _part in re.split(r"\bAND\b", str(_r), flags=re.IGNORECASE):
                _m = re.search(
                    r"([\w.\"`\[\]]+)\s*\.\s*([\w\"`\[\]]+)\s*=\s*"
                    r"([\w.\"`\[\]]+)\s*\.\s*([\w\"`\[\]]+)", _part)
                if not _m:
                    continue
                _lt = _split_ref(_m.group(1))
                _lc = _m.group(2).strip('"`[]')
                _rt = _split_ref(_m.group(3))
                _rc = _m.group(4).strip('"`[]')
                if _lt and _rt:
                    _edges.append((_lt[-1].lower(), _lc.lower(),
                                   _rt[-1].lower(), _rc.lower()))
        if not _edges:
            return None
        from collections import defaultdict as _dd
        _pair_conds: dict = _dd(set)
        for a, b, c, d in _edges:
            _pair_conds[(a, c)].add((b, d))
            _pair_conds[(c, a)].add((d, b))
        _tref = (r"(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)"
                 r"(?:\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))*")
        _aliases: dict[str, list[str]] = {}
        for _m in re.finditer(
                r"\b(?:FROM|JOIN)\s*(" + _tref + r")"
                r"(?:\s+(?:AS\s+)?(\w+))?", sql, re.IGNORECASE):
            _full = _split_ref(_m.group(1))
            _al = (_m.group(2) or "").lower()
            if _full and _al and _al not in (
                    "where", "group", "order", "limit", "join", "on",
                    "left", "right", "inner", "outer", "full", "cross"):
                _aliases[_al] = [s.lower() for s in _full]
        _eq_re = re.compile(
            r"(" + _tref + r"\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))"
            r"\s*=\s*(" + _tref + r"\s*\.\s*(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))",
            re.IGNORECASE)
        # Per-JOIN ON spans: every condition of a composite FK must live in
        # the SAME join (a partial ON fans out). Split top-level JOIN..ON..
        # (next JOIN/WHERE/GROUP/ORDER/LIMIT) segments; equalities outside
        # any ON (e.g. WHERE filters) never count as join conditions.
        _segs: list[str] = []
        try:
            _bounds = [m.start() for m in re.finditer(
                r"\b(?:JOIN|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
                sql, re.IGNORECASE)]
            _bounds.append(len(sql))
            _prev = 0
            for _b in _bounds:
                _segs.append(sql[_prev:_b])
                _prev = _b
        except Exception:
            _segs = [sql]
        for _seg in _segs:
            if not re.search(r"\bJOIN\b", _seg, re.IGNORECASE):
                continue
            _have = set()
            for _eq in _eq_re.finditer(_seg):
                _lsegs = _split_ref(_eq.group(1))
                _rsegs = _split_ref(_eq.group(2))
                if len(_lsegs) < 2 or len(_rsegs) < 2:
                    continue
                _lt, _lc = _lsegs[:-1], _lsegs[-1].lower()
                _rt, _rc = _rsegs[:-1], _rsegs[-1].lower()
                _lt = _aliases.get(_lt[0].lower() if len(_lt) == 1 else "", _lt)
                _rt = _aliases.get(_rt[0].lower() if len(_rt) == 1 else "", _rt)
                _ltb, _rtb = _lt[-1].lower(), _rt[-1].lower()
                if _ltb == _rtb:
                    continue  # self-join: not verifiable, allow
                _have.add((_ltb, _lc, _rtb, _rc))
                _have.add((_rtb, _rc, _ltb, _lc))
            # Every asserted equality is checked: unrelated table pairs
            # and partial composite joins are rejected. Only equalities in
            # this JOIN's own ON span count (WHERE filters never satisfy a
            # join condition).
            for (_t1, _c1, _t2, _c2) in _have:
                _need = _pair_conds.get((_t1, _t2)) or set()
                if not _need:
                    return (f"{_t1} joined to {_t2} with no declared "
                            f"relationship in this schema")
                for (_ec1, _ec2) in _need:
                    if ((_t1, _ec1, _t2, _ec2) not in _have
                            and (_t2, _ec2, _t1, _ec1) not in _have):
                        return (f"incomplete composite join {_t1} <-> {_t2}: "
                                f"missing {_ec1} = {_ec2}")
    except Exception:
        return None
    return None


_NUMERIC_CAST_FAILURE_RE = None


_REVENUE_WORDS = ("revenue", "revenues", "sales", "profit", "income",
                   "turnover", "gmv")
_COST_LIKE = ("cost", "cogs", "expense", "expenditure", "spend")


def _revenue_cost_mismatch(question: str, sql: str) -> str | None:
    """SUM/AVG over a cost-like column for a revenue question, else None.

    Generic business-vocabulary guard (no schema names): standardcost is
    spending, not income. Returns the offending column or None.
    """
    import re as _re
    try:
        _q = str(question or "").lower()
        if not any(w in _q for w in _REVENUE_WORDS):
            return None
        for _m in re.finditer(
                r"\b(?:SUM|AVG)\s*\(\s*(?:DISTINCT\s+)?"
                r"(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+)(?:\s*\.\s*"
                r"(?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|\w+))?"
                r"\s*\.\s*[\"`\[]?([A-Za-z_][\w$]*)[\"`\]]?", sql,
                re.IGNORECASE):
            _col = _m.group(1).lower()
            if any(k in _col for k in _COST_LIKE) and not any(
                    k in _col for k in ("price", "amount", "total", "value")):
                return _m.group(1)
    except Exception:
        pass
    return None


def _is_numeric_cast_failure(error: str | None) -> bool:
    """True when the DB rejected a bare numeric cast on formatted text."""
    global _NUMERIC_CAST_FAILURE_RE
    if _NUMERIC_CAST_FAILURE_RE is None:
        import re as _re
        _NUMERIC_CAST_FAILURE_RE = _re.compile(
            r"invalid input syntax for type numeric|invalid text representation"
            r"|could not convert|truncated incorrect|incorrect decimal value"
            r"|invalid number|ORA-01722|conversion failed",
            _re.IGNORECASE)
    try:
        return bool(error and _NUMERIC_CAST_FAILURE_RE.search(error))
    except Exception:
        return False


def build_sql_prompt(question: str, ctx, dialect: str, feedback: str | None = None) -> str:
    """Compose the SQL-generation prompt from retrieved context only."""
    schema_block = ctx.to_prompt() if hasattr(ctx, "to_prompt") else str(ctx)
    retry_block = ""
    if feedback:
        retry_block = (
            "\nThe previous attempt failed with this error — fix the SQL:\n"
            f"{feedback}\n"
        )
    return (
        "You are a senior data analyst. Write ONE read-only SQL query that answers "
        "the user's business question using ONLY the tables and relationships below.\n\n"
        f"DATABASE DIALECT: {dialect}\n{_dialect_rules(dialect)}\n\n"
        f"SCHEMA CONTEXT (retrieved, not the full schema):\n{schema_block}\n\n"
        f"QUESTION: {question}\n{retry_block}\n"
        "Rules:\n"
        "- Return aggregation at the granularity the question asks for "
        "(one row per group, month, product, customer, etc.).\n"
        "- Use only SELECT (never modify data).\n"
        "- Reference only the tables/columns provided; join via the relationships shown.\n"
        "- Lines starting with 'Previous ' are conversation context from an earlier "
        "question, never schema: FROM/JOIN only tables named under SCHEMA CONTEXT.\n"
        "- Add a LIMIT (<= 1000) unless the question is a single summary value.\n"
        "- When the question says 'top/best/highest/biggest N' (or 'bottom/lowest N'), "
        "add LIMIT N with ORDER BY the metric DESC (ASC for bottom/lowest).\n"
        "- When the question asks about revenue/sales/profit/money/value, aggregate "
        "money-like columns (amount, total, price, value, linetotal) — never counts, "
        "quantities or ids. 'How many'/'count' questions use COUNT(*) instead.\n"
        "- Revenue/sales/profit is money IN: never aggregate cost/cogs/expense/standardcost "
        "columns for it (those measure spending, not income).\n"
        f"{_text_number_rule(dialect)}\n"
        "- Date columns may be stored as TEXT in local spellings (DD-MM-YYYY, "
        "ISO, YYYY-MM — check the sample values): NEVER date_trunc() or "
        "CAST a TEXT date directly. Parse DD-MM-YYYY with TO_DATE(col, "
        "'DD-MM-YYYY') (postgres), STR_TO_DATE(col, '%d-%m-%Y') (mysql), "
        "and fall back to the native ISO parse, with ELSE NULL for bad "
        "values.\n"
        "- Output ONLY the SQL statement, no explanation, no markdown fences.\n"
    )


class SqlEngine:
    """Runs validated read-only SQL against a SQLAlchemy URI."""

    def __init__(self, db_uri: str, dialect: str = "postgresql",
                 max_rows: int = 1000, max_attempts: int = 3, schemas=None):
        self.db_uri = db_uri
        self.dialect = dialect or "postgresql"
        self.max_rows = int(max_rows)
        self.max_attempts = int(max_attempts)
        try:
            self.schemas = [str(s).strip() for s in (schemas or [])
                            if str(s or "").strip()]
        except Exception:
            self.schemas = []

    # -- validation ------------------------------------------------------

    def validate(self, sql: str, tables: dict | None = None) -> tuple[bool, str | None]:
        if not sql:
            return False, "No SQL generated."
        if not is_read_only(sql):
            return False, "Rejected: only read-only SELECT statements are allowed."
        if tables:
            bad_tables = unknown_sql_tables(sql, list(tables.keys()))
            if bad_tables:
                return False, f"Unknown table(s): {', '.join(bad_tables)}."
            cols_by_table = {}
            for t, info in tables.items():
                cols = info.get("columns", [])
                if isinstance(cols, list):
                    cols_by_table[t] = [c.get("column") if isinstance(c, dict) else c for c in cols]
                elif isinstance(cols, dict):
                    cols_by_table[t] = list(cols.keys())
            bad_cols = unknown_sql_columns(sql, cols_by_table)
            if bad_cols:
                return False, f"Unknown column reference(s): {', '.join(bad_cols[:5])}."
        return True, None

    # -- execution -------------------------------------------------------

    def execute(self, sql: str, tables: dict | None = None) -> SqlResult:
        ok, reason = self.validate(sql, tables)
        if not ok:
            return SqlResult(ok=False, sql=sql, error=reason, dialect=self.dialect)

        # Deterministic safety net (any provider, incl. weak local models):
        # a bare CAST over a mapping-declared TEXT column crashes on the
        # first formatted value ('₹399') — rewrite to the NULL-safe
        # coercion before hitting the database.
        safe_sql = _rewrite_text_casts(sql, self.dialect, tables)
        safe_sql = _rewrite_text_dates(safe_sql, self.dialect, tables)
        # Row cap without rewriting grouped/aggregate semantics: only append
        # SQL LIMIT to plain row queries. Grouped/aggregate queries keep their
        # grain; fetchmany() still bounds returned rows.
        if not _LIMIT_RE.search(sql) and self.dialect in ("postgresql", "mysql", "sqlite"):
            if not (_GROUP_BY_RE.search(sql) or _AGG_RE.search(sql)):
                # Built from safe_sql: the LIMIT must not discard the
                # text-cast/date rewrites applied above.
                safe_sql = f"{safe_sql.rstrip(';')} LIMIT {self.max_rows}"

        try:
            engine = _get_engine(self.db_uri, self.schemas)
            with engine.connect() as conn:
                result = conn.execute(text(safe_sql))
                columns = list(result.keys())
                rows = [dict(zip(columns, r)) for r in result.fetchmany(self.max_rows)]
            return SqlResult(ok=True, sql=safe_sql, columns=columns, rows=rows,
                             row_count=len(rows), dialect=self.dialect)
        except Exception as exc:
            try:
                from core.errors import sanitize_error as _san
                err = _san(exc)
            except Exception:
                err = "Database query failed."
            return SqlResult(ok=False, sql=safe_sql, error=str(err)[:400],
                             dialect=self.dialect)

    def generate_and_run(self, question: str, ctx, llm, tables: dict | None = None,
                         model_role: str = "sql", relationships=None) -> SqlResult:
        """Generate SQL via the (provider-independent) LLM, then execute.

        One bounded repair attempt on failure — no infinite loop, and failures
        are surfaced gracefully rather than crashing the Copilot.
        `relationships` (validated "a.col = b.col" lines) enables a
        deterministic join check: a JOIN the schema never declared executes
        fine but returns garbage/empty, so it is rejected BEFORE execution
        with the valid edges as repair feedback.
        """
        feedback = None
        last = SqlResult(ok=False, error="not started", dialect=self.dialect)
        for attempt in range(1, self.max_attempts + 1):
            prompt = build_sql_prompt(question, ctx, self.dialect, feedback)
            try:
                # SQL answers are short: cap tokens so CPU inference finishes
                # well inside the timeout instead of timing out mid-generation.
                # Repair attempts use light sampling so an empty first pass
                # does not repeat identically.
                raw = llm.complete(model_role, prompt, temperature=0.0 if attempt == 1 else 0.2,
                                   num_predict=300)
            except Exception as exc:
                return SqlResult(ok=False, error=f"LLM SQL generation failed: {exc}",
                                 attempts=attempt, dialect=self.dialect)
            sql = clean_sql(raw)
            # Deterministic column fix first: near-miss names from memory
            # (qty for orderqty) are repaired without spending an LLM call.
            try:
                _csql = _fix_unknown_columns(sql, tables)
                if _csql:
                    _vok, _ = self.validate(_csql, tables)
                    if _vok:
                        sql = _csql
            except Exception:
                pass
            _bad_join = _invalid_join(sql, relationships) if relationships else None
            if _bad_join:
                _hint = ""
                try:
                    _jt = _bad_join_tables(sql, relationships)
                    if _jt and len(_jt) == 2:
                        _ph = _join_path_hint(relationships, _jt[0], _jt[1])
                        if _ph:
                            _hint = " Use exactly this join path: " + _ph + "."
                except Exception:
                    _hint = ""
                result = SqlResult(
                    ok=False, sql=sql, dialect=self.dialect,
                    error=("Join " + _bad_join + " is not a declared "
                           "relationship in this schema." + _hint +
                           " Valid joins: "
                           + "; ".join(_rank_rels_for_sql(relationships, sql))),
                    attempts=attempt)
            else:
                result = self.execute(sql, tables)
                result.attempts = attempt
                # Revenue-measure guard: a successful SUM(cost) for a revenue
                # question is wrong data, not success — force a repair that
                # names income columns instead of spending ones.
                if result.ok:
                    try:
                        _bad_measure = _revenue_cost_mismatch(question, sql)
                    except Exception:
                        _bad_measure = None
                    if _bad_measure:
                        result = SqlResult(
                            ok=False, sql=sql, dialect=self.dialect,
                            error=(f"Aggregated {_bad_measure}, which measures "
                                   "spending (cost), but the question asks about "
                                   "revenue/income. Aggregate price/amount/total "
                                   "columns instead — never cost/cogs/expense."),
                            attempts=attempt)
            if result.ok:
                return result
            last = result
            feedback = result.error
            # Point the repair at the valid tables so a hallucinated table
            # name gets replaced instead of repeated.
            if tables and "Unknown table" in (result.error or ""):
                feedback += f" Valid tables: {', '.join(sorted(tables.keys()))}."
            # Unknown columns: list the tables' REAL columns so the repair
            # picks from them (e.g. orderqty/unitprice instead of a
            # remembered linetotal that this database does not have).
            # Generic (mapping-driven); capped to keep prompts bounded.
            if tables and "Unknown column" in (result.error or ""):
                try:
                    _col_hint = []
                    for _t in sorted(tables.keys())[:8]:
                        _cols = (tables.get(_t) or {}).get("columns", [])
                        _names = [(_c.get("column") if isinstance(_c, dict) else _c)
                                  for _c in (_cols.values() if isinstance(_cols, dict) else _cols or [])]
                        _names = [str(_n) for _n in _names if _n][:20]
                        if _names:
                            _col_hint.append(f"{_t}: {', '.join(_names)}")
                    if _col_hint:
                        feedback += (" Real columns — use ONLY these: "
                                     + " | ".join(_col_hint))
                except Exception:
                    pass
            # A date op just died on a TEXT date (function date_trunc(unknown,
            # text), DatetimeFieldOverflow, date/time field value out of
            # range) — point the repair at explicit parsing.
            if result.error and ("date_trunc(unknown, text)" in result.error
                    or "DatetimeFieldOverflow" in result.error
                    or "date/time field value out of range" in result.error):
                feedback += (" The date column is TEXT in a local spelling, not "
                             "a native date. Wrap it in a CASE that parses "
                             "DD-MM-YYYY first, then ISO, ELSE NULL — never "
                             "date_trunc()/CAST it directly.")
            # A bare numeric cast just died on formatted TEXT — tell the
            # repair exactly which coercion to use (provider-independent).
            if _is_numeric_cast_failure(result.error):
                try:
                    from core.text_numeric import coercion_sql as _rc
                    _coerce_example = _rc(self.dialect, '"COLUMN"')
                    feedback += (" The CAST failed because a TEXT column holds "
                                 "formatted numbers (currency/commas/%). Use "
                                 + _coerce_example + " instead of CAST for "
                                 "every TEXT column (replace \"COLUMN\" with "
                                 "the real column).")
                except Exception:
                    pass
            log.info("Copilot SQL attempt %d failed: %s", attempt, result.error)
            log.info("DBG verbatim-gate attempt=%s max=%s has_declared=%s",
                     attempt, self.max_attempts,
                     bool(result.error and "declared relationship" in result.error))
            # Verbatim last resort: join-check failures mean the model keeps
            # inventing its own joins. One bonus attempt orders it to copy a
            # validated FROM/JOIN block verbatim and only write the
            # SELECT/GROUP BY/ORDER BY around it.
            if (attempt == self.max_attempts and result.error
                    and "declared relationship" in result.error):
                try:
                    import re as _re2
                    _m = re.search(r"Use exactly this join path: (.+?)\."
                                   r"(?: Valid joins:|$)", result.error)
                    _path = _m.group(1).strip() if _m else ""
                    if _path:
                        try:
                            _vprompt = (
                                "You are a senior data analyst. Write ONE read-only SQL query.\n"
                                f"QUESTION: {question}\n\n"
                                "MANDATORY FROM/JOIN BLOCK — copy it VERBATIM, do not alter "
                                "table names, aliases, or ON conditions:\n"
                                f"{_path}\n\n"
                                "Write only the SELECT list, GROUP BY, ORDER BY and LIMIT "
                                "around that block to answer the question. Use the same "
                                "table names/aliases as the block (no new aliases). "
                                "Output ONLY the SQL statement.")
                            _raw2 = llm.complete(model_role, _vprompt, temperature=0.0,
                                                 num_predict=300)
                            _sql2 = clean_sql(_raw2)
                            _bad2 = _invalid_join(_sql2, relationships) \
                                if relationships else None
                            if _bad2:
                                log.info("Copilot verbatim-join repair still off-plan: %s",
                                         _bad2)
                                last = SqlResult(
                                    ok=False, sql=_sql2, dialect=self.dialect,
                                    error=("Join " + _bad_join + " is not a declared "
                                           "relationship in this schema."),
                                    attempts=attempt + 1)
                                continue
                            _r2 = self.execute(_sql2, tables)
                            _r2.attempts = attempt + 1
                            if _r2.ok and (_r2.rows or []):
                                log.info("Copilot verbatim-join repair succeeded.")
                                return _r2
                            last = _r2
                        except Exception as _vx:
                            log.info("Copilot verbatim-join repair skipped (%s).", _vx)
                except Exception as _vx2:
                    log.info("Copilot verbatim-join path skipped (%s).", _vx2)
        return last