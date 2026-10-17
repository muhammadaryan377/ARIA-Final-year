"""ARIA Intelligence — independent validation layer for Explore / What-If results.

Re-checks reported numbers with fresh SQL (never reuses the result's own
aggregates) so past bug classes — cross-table UndefinedColumn masking,
birth/hire dates picked as event dates (absurd 1966 windows), and
"previous always None" — are caught fail-closed.

Only this module + wiring in ``services.intelligence_service`` + the new
tests may change under the validation task.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("aria.intelligence.validation")

_SQLITE_HELPERS_FLAG = "_aria_sqlite_helpers"


def _sqlite_numstrip(v: Any) -> Any:
    """SQLite UDF: strip a TEXT measure to its numeric core (``Rs.1,099`` →
    ``1099``, ``64%`` → ``64``); ``None`` when nothing numeric remains
    (aggregates skip NULL instead of crashing on ``SUM(text)``).

    Shape-based cleaning shared by the concept in all dialects: a stray
    leading dot from currency prefixes (``Rs.1099``) is dropped only when
    3+ digits follow with no other dot (``.5`` stays a fraction); stray
    minus signs collapse to one leading ``-``; extra dots merge
    (``1.2.3`` → ``1.23``) instead of raising.
    """
    if v is None:
        return None
    try:
        import math as _math
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v if not (_math.isnan(v) or _math.isinf(v)) else None
    except Exception:
        pass
    s = re.sub(r"[^0-9.\-]", "", str(v))
    if re.fullmatch(r"\.[0-9]{3,}", s):
        s = s[1:]
    neg = s.startswith("-")
    s = s.replace("-", "")
    parts = s.split(".")
    if len(parts) > 2:
        s = parts[0] + "." + "".join(parts[1:])
    if neg:
        s = "-" + s
    if not s or s in ("-", ".", "-.", ".-"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def ensure_sqlite_helpers(engine) -> None:
    """Register ``aria_numstrip`` on every sqlite DBAPI connection.

    Idempotent (flagged on the engine). The metric cast expression for
    TEXT-stored measures references this UDF, so both the service engine
    and the validator engine must ensure it — hence it lives here, next to
    the other shared helper.
    """
    try:
        if engine is None or getattr(engine, _SQLITE_HELPERS_FLAG, False):
            return
        from sqlalchemy import event as _event

        def _on_connect(dbapi_conn, _rec):
            try:
                dbapi_conn.create_function("aria_numstrip", 1, _sqlite_numstrip)
            except Exception:
                pass

        _event.listen(engine, "connect", _on_connect)
        setattr(engine, _SQLITE_HELPERS_FLAG, True)
    except Exception:
        pass

BIRTH_HIRE_HINTS = ("birth", "dob", "birthday", "hire", "death", "anniversary")

_REL_TOL = 0.005      # 0.5 % relative
_ABS_TOL = 0.01       # absolute floor
_PCT_TOL = 0.06       # change_pct arithmetic tolerance (2dp rounding)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a DB)
# ---------------------------------------------------------------------------

def is_suspicious_date_col(name: Optional[str]) -> bool:
    """True when a date-column name looks like birth/hire/dob/death."""
    s = str(name or "").lower()
    return any(h in s for h in BIRTH_HIRE_HINTS)


def extract_col_name(qualified_ref: Optional[str]) -> Optional[str]:
    """Last identifier of an alias-qualified ref like ``t0."order_date"``."""
    if not qualified_ref:
        return None
    s = str(qualified_ref)
    quoted = re.findall(r'"([^"]+)"|`([^`]+)|\[([^\]]+)\]', s)
    if quoted:
        parts = [a or b or c for a, b, c in quoted]
        return parts[-1] if parts else None
    tail = s.split(".")[-1].strip().strip('"\'`[]')
    return tail or None


_TOLERANT_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y",
    "%m-%d-%Y %H:%M:%S", "%m-%d-%Y",
    "%d-%m-%Y", "%Y/%m/%d", "%Y.%m.%d", "%d/%m/%Y",
    "%d-%b-%Y", "%d-%B-%Y", "%b %d, %Y", "%B %d, %Y",
    "%d %b %Y", "%d %B %Y",
)

_EPOCH_RE = re.compile(r"^-?\d{9,10}(\.\d+)?$")
_EPOCH_MS_RE = re.compile(r"^-?\d{13}(\.\d+)?$")
_YYYYMMDD_RE = re.compile(r"^(19|20)\d{6}$")


def to_date(v: Any) -> Optional[date]:
    """Tolerant ``date`` parser for str / date / datetime / epoch DB values.

    Shape-based (no per-database callouts): covers ISO datetimes, US
    ``MM/DD/YYYY`` and ``MM-DD-YYYY`` (e.g. walmart TEXT dates), ``DD-Mon-YYYY``
    and long month names, epoch seconds / milliseconds, and ``YYYYMMDD``.
    Pure parsing only — range sanity stays in ``check_date_sanity`` (fail
    closed) so absurd years are still reported, never silently dropped.
    This is the ONE shared parser — ``services.intelligence_service`` imports
    it instead of keeping a second copy, so service bucketing and validator
    checks can never disagree on what a stored value means.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            f = float(v)
        except Exception:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        if abs(f) >= 1e11:
            # Millisecond epoch (any plausible seconds value is < 1e11).
            f = f / 1000.0
        return _epoch_to_date(f)
    s = str(v).strip()
    if not s or s.lower() in ("none", "null", "nat", "nan", ""):
        return None
    if _EPOCH_RE.match(s):
        try:
            return _epoch_to_date(float(s))
        except Exception:
            pass
    if _EPOCH_MS_RE.match(s):
        try:
            return _epoch_to_date(float(s) / 1000.0)
        except Exception:
            pass
    if _YYYYMMDD_RE.match(s):
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            pass
    candidates = [s, s[:19].replace("T", " "), s[:10]]
    for cand in candidates:
        for fmt in _TOLERANT_DATE_FORMATS:
            try:
                return datetime.strptime(cand, fmt).date()
            except Exception:
                continue
    # Trailing-time tolerant: date part before first space for slash/dash
    # spellings with a time suffix the slices above may have missed.
    try:
        head = s.split(" ")[0]
        if head != s:
            for fmt in _TOLERANT_DATE_FORMATS:
                try:
                    return datetime.strptime(head, fmt).date()
                except Exception:
                    continue
    except Exception:
        pass
    # Month-grain spellings without a day (common in CSV exports):
    # "2025-01" (YYYY-MM), "01/2026" (MM/YYYY), "Jan 2026"/"January 2026".
    # Anchored regexes AFTER full-date formats so they never shadow a
    # complete date. Day defaults to the 1st (month bucketing only needs it).
    try:
        if re.match(r"^(19|20)\d{2}-(0[1-9]|1[0-2])$", s):
            return date(int(s[0:4]), int(s[5:7]), 1)
        _my = re.match(r"^(0?[1-9]|1[0-2])/(19|20)\d{2}$", s)
        if _my:
            _mm, _yy = s.split("/")
            return date(int(_yy), int(_mm), 1)
        for _fmt in ("%b %Y", "%B %Y"):
            try:
                return datetime.strptime(s, _fmt).date().replace(day=1)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _epoch_to_date(secs: float) -> Optional[date]:
    try:
        return datetime.fromtimestamp(float(secs), tz=timezone.utc).date()
    except Exception:
        return None


# Back-compat alias: the single shared tolerant parser under both names.
parse_date_tolerant = to_date


def close_enough(a: Any, b: Any, rel: float = _REL_TOL,
                 abs_tol: float = _ABS_TOL) -> bool:
    """``True`` when two numbers agree within rel/abs tolerance."""
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return False
    if math.isnan(fa) or math.isnan(fb):
        return False
    return abs(fa - fb) <= max(abs_tol, rel * max(abs(fa), abs(fb)))


def _infer_do_compare(result: dict, do_compare: Optional[bool]) -> bool:
    if do_compare is not None:
        return bool(do_compare)
    cmp_raw = str(result.get("comparison") or "").strip().lower()
    if not cmp_raw:
        return True  # fail-closed default: comparison expected
    if cmp_raw in ("no_comparison", "no", "none") or cmp_raw.startswith("no_"):
        return False
    if "no comparison" in cmp_raw or cmp_raw == "no":
        return False
    return True


def check_date_sanity(date_col: Optional[str],
                       cur_start: Any, cur_end: Any,
                       prev_start: Any, prev_end: Any,
                       min_date: Any, max_date: Any) -> Tuple[bool, str]:
    """Pure date-sanity check over already-parsed window/data bounds.

    Fails closed on birth/hire-like column names, absurd years, unparseable
    bounds, or a current window with no overlap against [min, max].
    The *previous* window may legitimately be empty (no prior data), so it
    is only checked for absurd years — emptiness is previous_check's job.
    """
    if is_suspicious_date_col(date_col or ""):
        return False, f"date column '{date_col}' looks like birth/hire/dob/death, not an event date"
    cs, ce, ps, pe = (to_date(cur_start), to_date(cur_end),
                      to_date(prev_start), to_date(prev_end))
    if cs is None or ce is None:
        return True, "no time windows (snapshot); date sanity skipped"
    for label, d in (("cur_start", cs), ("cur_end", ce),
                     ("prev_start", ps), ("prev_end", pe)):
        if d is not None and (d.year < 1970 or d.year > 2100):
            return False, f"absurd year in {label}: {d.isoformat()}"
    mind, maxd = to_date(min_date), to_date(max_date)
    if mind is None or maxd is None:
        return False, "date column range unparseable (fail-closed)"
    if cs > maxd or ce < mind:
        return False, (f"current window {cs.isoformat()}..{ce.isoformat()} "
                       f"outside data range {mind.isoformat()}..{maxd.isoformat()}")
    return True, (f"windows sane: current {cs.isoformat()}..{ce.isoformat()} "
                  f"within data {mind.isoformat()}..{maxd.isoformat()}")


def _check_truncated_rows(rows: list, total: Any,
                         prev_total: Any) -> Tuple[bool, str]:
    """Pure containment check for a truncated (top-N) breakdown.

    Requires: rows sorted DESC by ``current``; ``sum(current) <= total`` (or
    ``≈ total`` when the top-N already covers everything); top-row share
    ``rows[0].current / total`` computable and within ``[0, 1]``; and the
    same containment for ``previous`` when ``prev_total`` is not None.
    Returns ``(passed, detail)``.
    """
    try:
        n = len(rows)
        tol = max(_ABS_TOL, 0.01 * n)
        curs = [float((r or {}).get("current") or 0) for r in rows]
        for i in range(len(curs) - 1):
            if curs[i] + tol < curs[i + 1]:
                return (False, f"truncated rows not sorted DESC at index {i}: "
                               f"{curs[i]:.2f} < {curs[i + 1]:.2f}")
        if total is None:
            return False, "truncated scope but result.total is missing"
        try:
            tot = float(total)
        except Exception:
            return False, "truncated scope but result.total is non-numeric"
        sum_cur = sum(curs)
        if abs(tot) <= tol:
            if not close_enough(sum_cur, tot, abs_tol=tol):
                return (False, f"truncated sum(rows current)={sum_cur:.2f} "
                               f"!= near-zero total={tot:.2f}")
            share_txt = "total ~ 0"
        else:
            if sum_cur > tot + tol and not close_enough(sum_cur, tot, abs_tol=tol):
                return (False, f"truncated sum(rows current)={sum_cur:.2f} "
                               f"> total={tot:.2f}")
            share = curs[0] / tot if tot else 0.0
            if not (-tol <= share <= 1.0 + tol):
                return (False, f"top-row share out of range: "
                               f"{curs[0]:.2f}/{tot:.2f}={share:.3f}")
            share_txt = f"top share {share * 100:.1f}%"
        if prev_total is None:
            return (True, f"truncated scope ok: {n} top rows, "
                          f"sum={sum_cur:.2f} of total={tot:.2f} ({share_txt}); "
                          f"previous skipped (None)")
        try:
            ptot = float(prev_total)
        except Exception:
            return False, "truncated scope but previous_total is non-numeric"
        sum_prev = sum(float((r or {}).get("previous") or 0) for r in rows)
        if abs(ptot) <= tol:
            if not close_enough(sum_prev, ptot, abs_tol=tol):
                return (False, f"truncated sum(rows previous)={sum_prev:.2f} "
                               f"!= near-zero previous_total={ptot:.2f}")
        elif sum_prev > ptot + tol and not close_enough(sum_prev, ptot, abs_tol=tol):
            return (False, f"truncated sum(rows previous)={sum_prev:.2f} "
                           f"> previous_total={ptot:.2f}")
        return (True, f"truncated scope ok: {n} top rows, "
                      f"sum={sum_cur:.2f} of total={tot:.2f} ({share_txt}), "
                      f"prev sum={sum_prev:.2f} of {ptot:.2f}")
    except Exception as exc:
        return False, f"truncated check failed (fail-closed): {exc}"


# ---------------------------------------------------------------------------
# Fresh-SQL helpers (separate connection per query)
# ---------------------------------------------------------------------------

def _scalar(engine, sql: str, params: Optional[dict] = None):
    from sqlalchemy import text as sql_text
    with engine.connect() as conn:
        row = conn.execute(sql_text(sql), params or {}).fetchone()
        return row[0] if row is not None else None


def _row(engine, sql: str, params: Optional[dict] = None):
    from sqlalchemy import text as sql_text
    with engine.connect() as conn:
        row = conn.execute(sql_text(sql), params or {}).fetchone()
        return tuple(row) if row is not None else (None,)


def _distinct_range(engine, qd: str, qt: str):
    """True (min, max) over DISTINCT stored values via the shared parser.

    For TEXT/odd-typed date columns where lexicographic MIN/MAX is
    meaningless. Returns ``(None, None)`` when nothing parses (caller then
    fails closed via ``check_date_sanity``).
    """
    from sqlalchemy import text as sql_text
    try:
        with engine.connect() as conn:
            vals = conn.execute(
                sql_text(f"SELECT DISTINCT {qd} FROM {qt} LIMIT 10000")).fetchall()
        parsed = [to_date(v[0]) for v in vals]
        parsed = [d for d in parsed if d is not None]
        if not parsed:
            return None, None
        return min(parsed), max(parsed)
    except Exception:
        return None, None


def _trend_points(result: dict) -> list:
    trend = result.get("trend") or {}
    pts = trend.get("points") if isinstance(trend, dict) else None
    if not pts:
        pts = result.get("points") or result.get("time_series") or []
    return list(pts or [])


def _breakdown_rows(result: dict) -> list:
    for key in ("rows", "breakdown_rows", "table", "breakdown", "breakdown_data"):
        rows = result.get(key)
        if isinstance(rows, list) and rows:
            return rows
    return []


# ---------------------------------------------------------------------------
# Explore validator
# ---------------------------------------------------------------------------

def validate_explore(engine, dialect: str, qm: str, qt: str,
                     where_current_sql: Optional[str],
                     where_previous_sql: Optional[str],
                     params: Optional[dict],
                     result: dict, *,
                     date_col: Optional[str] = None,
                     qd: Optional[str] = None,
                     cur_start=None, cur_end=None,
                     prev_start=None, prev_end=None,
                     do_compare: Optional[bool] = None,
                     agg: str = "SUM",
                     where_base_sql: Optional[str] = None,
                     date_storage: str = "native") -> dict:
    """Independently re-check an Explore result with fresh SQL.

    Returns ``{passed, checks:[{name, passed, detail}], corrected}``.
    ``corrected`` is always ``False`` — fail-closed, numbers are never
    silently rewritten. Never raises: unexpected errors become a failed
    ``validator_error`` check.

    ``agg="COUNT"`` validates a ``COUNT(*)`` (record-count fallback) explore;
    ``date_storage="text"`` validates TEXT/odd-typed date columns by fetching
    raw dates and bucketing in Python with the shared tolerant parser
    (SQL string comparison is meaningless across storage spellings).
    ``result["breakdown_scope"]`` (``{returned, total_groups, truncated}``)
    selects the truncated-breakdown rules; non-truncated equality is untouched.
    """
    checks: List[Dict[str, Any]] = []

    def _ok(name: str, passed: bool, detail: str):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    try:
        ensure_sqlite_helpers(engine)
        result = result or {}
        params = dict(params or {})
        wc = where_current_sql or ""
        wp = where_previous_sql if where_previous_sql is not None else wc
        cmp_on = _infer_do_compare(result, do_compare)
        has_windows = cur_start is not None and cur_end is not None and qd
        col_name = date_col or extract_col_name(qd)

        total = result.get("total")
        prev_total = result.get("previous_total")
        pts = _trend_points(result)
        rows = _breakdown_rows(result)
        scope = result.get("breakdown_scope") or {}
        truncated = bool(scope.get("truncated"))
        is_count = str(agg or "SUM").upper() == "COUNT"
        non_native = (str(date_storage or "native").lower() != "native"
                      and has_windows)
        base_params = dict(params or {})
        for _dk in ("_vc_s", "_vc_e", "_vp_s", "_vp_e", "_ps", "_ce"):
            base_params.pop(_dk, None)

        def _window_sum(start, end):
            """Fresh window aggregate for TEXT/odd-typed date storage.

            Fetches raw ``(date, val)`` rows under the filter-free base WHERE
            and sums in Python with the shared tolerant parser — SQL string
            comparison across spellings (``MM-DD-YYYY`` vs ISO bounds) is
            meaningless. Only used when ``date_storage != "native"``; the
            native path keeps the original SQL-window checks below.
            """
            from sqlalchemy import text as _stext
            lo, hi = to_date(start), to_date(end)
            if lo is None or hi is None:
                return None
            _sel = f"{qd} AS _d" if is_count else f"{qd} AS _d, {qm} AS _v"
            with engine.connect() as _conn:
                _rs = _conn.execute(
                    _stext(f"SELECT {_sel} FROM {qt}"
                           f"{where_base_sql or ''}"),
                    base_params)
                _ks = list(_rs.keys())
                _tot = 0.0
                _n = 0
                for _row in _rs.fetchall():
                    _dd = dict(zip(_ks, _row))
                    _dt = to_date(_dd.get("_d"))
                    if _dt is None or _dt < lo or _dt > hi:
                        continue
                    if is_count:
                        _n += 1
                        continue
                    try:
                        _x = float(_dd.get("_v") or 0)
                        if math.isnan(_x) or math.isinf(_x):
                            continue
                    except Exception:
                        continue
                    _tot += _x
                return _n if is_count else _tot

        def _native_total(wsql):
            if is_count:
                return _scalar(engine, f"SELECT COUNT(*) FROM {qt}{wsql}", params)
            return _scalar(engine, f"SELECT SUM({qm}) FROM {qt}{wsql}", params)

        def _window_count(start, end):
            """Row count inside a window for TEXT/odd-typed date storage."""
            from sqlalchemy import text as _stext
            lo, hi = to_date(start), to_date(end)
            if lo is None or hi is None or not qd:
                return None
            with engine.connect() as _conn:
                _rs = _conn.execute(
                    _stext(f"SELECT {qd} AS _d FROM {qt}{where_base_sql or ''}"),
                    base_params)
                _n = 0
                for _row in _rs.fetchall():
                    _dt = to_date(_row[0])
                    if _dt is not None and lo <= _dt <= hi:
                        _n += 1
                return _n

        # ---- total_check ------------------------------------------------
        try:
            if total is None:
                _ok("total_check", False, "result.total is missing")
            else:
                got = (_window_sum(cur_start, cur_end) if non_native
                       else _native_total(wc))
                got_f = float(got or 0.0)
                if close_enough(got_f, total):
                    _ok("total_check", True,
                        f"SUM(current)={got_f:.2f} matches total={float(total):.2f}")
                else:
                    _ok("total_check", False,
                        f"SUM(current)={got_f:.2f} != total={float(total):.2f}")
        except Exception as exc:
            _ok("total_check", False, f"query failed (fail-closed): {exc}")

        # ---- previous_check ----------------------------------------------
        try:
            if not has_windows:
                if prev_total is None:
                    _ok("previous_check", True,
                        "snapshot without time windows; previous None expected")
                else:
                    _ok("previous_check", False,
                        f"snapshot path must not report previous_total={prev_total}")
            elif prev_total is None:
                if not cmp_on:
                    _ok("previous_check", True, "comparison off; previous None expected")
                elif non_native:
                    _cnt = _window_count(prev_start, prev_end)
                    if _cnt is None:
                        _ok("previous_check", False,
                            "previous window unparseable (fail-closed)")
                    elif _cnt > 0:
                        _ok("previous_check", False,
                            f"previous_total is None but prior window holds {_cnt} rows")
                    else:
                        _ok("previous_check", True, "previous None and prior window empty")
                else:
                    cnt = _scalar(engine, f"SELECT COUNT(*) FROM {qt}{wp}", params)
                    cnt_n = int(cnt or 0)
                    if cnt_n > 0:
                        _ok("previous_check", False,
                            f"previous_total is None but prior window holds {cnt_n} rows")
                    else:
                        _ok("previous_check", True, "previous None and prior window empty")
            else:
                if non_native:
                    # SQL COUNT(*) over a TEXT date window is meaningless
                    # (string comparison across spellings) — compare the
                    # Python-bucketed window sum directly.
                    got = _window_sum(prev_start, prev_end)
                    got_f = float(got or 0.0)
                    if close_enough(got_f, prev_total):
                        _ok("previous_check", True,
                            f"SUM(previous)={got_f:.2f} matches previous_total={float(prev_total):.2f}")
                    else:
                        _ok("previous_check", False,
                            f"SUM(previous)={got_f:.2f} != previous_total={float(prev_total):.2f}")
                else:
                    _prev_cnt = _scalar(engine, f"SELECT COUNT(*) FROM {qt}{wp}", params)
                    _prev_cnt_n = int(_prev_cnt or 0)
                    if _prev_cnt_n == 0 and abs(float(prev_total)) > _ABS_TOL:
                        _ok("previous_check", False,
                            f"previous_total={float(prev_total):.2f} reported but prior window empty")
                    else:
                        got = _native_total(wp)
                        got_f = float(got or 0.0)
                        if close_enough(got_f, prev_total):
                            _ok("previous_check", True,
                                f"SUM(previous)={got_f:.2f} matches previous_total={float(prev_total):.2f}")
                        else:
                            _ok("previous_check", False,
                                f"SUM(previous)={got_f:.2f} != previous_total={float(prev_total):.2f}")
                # (handled — skip the native fallthrough below)
        except Exception as exc:
            _ok("previous_check", False, f"query failed (fail-closed): {exc}")

        # ---- buckets_sum (scope-aware) -------------------------------------
        # Non-truncated: bucket sums must EQUAL the totals (unchanged strict
        # rule — time buckets always cover the full scope). Truncated
        # (snapshot category-points capped at top-30): containment —
        # sum(points) <= total, likewise previous.
        try:
            if not pts:
                if total in (None, 0, 0.0):
                    _ok("buckets_sum", True, "no trend points and total is zero/missing")
                else:
                    _ok("buckets_sum", False,
                        f"no trend points but total={total}")
            elif truncated and not has_windows:
                _n = len(pts)
                _tol = max(_ABS_TOL, 0.01 * _n)
                _sum_cur = sum(float((p or {}).get("current") or 0) for p in pts)
                try:
                    _tot = float(total)
                except Exception:
                    _ok("buckets_sum", False, "truncated scope but total is non-numeric")
                    _tot = None
                if _tot is not None:
                    if (_sum_cur > _tot + _tol
                            and not close_enough(_sum_cur, _tot, abs_tol=_tol)):
                        _ok("buckets_sum", False,
                            f"truncated sum(trend current)={_sum_cur:.2f} > total={_tot:.2f}")
                    elif prev_total is None:
                        _ok("buckets_sum", True,
                            f"truncated trend ok: sum={_sum_cur:.2f} of total={_tot:.2f}; previous skipped (None)")
                    else:
                        _sum_prev = sum(float((p or {}).get("previous") or 0) for p in pts)
                        try:
                            _ptot = float(prev_total)
                        except Exception:
                            _ok("buckets_sum", False, "truncated scope but previous_total is non-numeric")
                            _ptot = None
                        if _ptot is not None:
                            if (_sum_prev > _ptot + _tol
                                    and not close_enough(_sum_prev, _ptot, abs_tol=_tol)):
                                _ok("buckets_sum", False,
                                    f"truncated sum(trend previous)={_sum_prev:.2f} > previous_total={_ptot:.2f}")
                            else:
                                _ok("buckets_sum", True,
                                    f"truncated trend ok: sum={_sum_cur:.2f} of total={_tot:.2f}")
            else:
                n = len(pts)
                sum_cur = sum(float((p or {}).get("current") or 0) for p in pts)
                tol = max(_ABS_TOL, 0.01 * n)
                if total is None or not close_enough(sum_cur, total, abs_tol=tol):
                    _ok("buckets_sum", False,
                        f"sum(trend current)={sum_cur:.2f} != total={total}")
                elif prev_total is None:
                    _ok("buckets_sum", True,
                        f"sum(trend current)={sum_cur:.2f} matches total; previous skipped (None)")
                else:
                    sum_prev = sum(float((p or {}).get("previous") or 0) for p in pts)
                    if close_enough(sum_prev, prev_total, abs_tol=tol):
                        _ok("buckets_sum", True,
                            f"bucket sums match total={total} previous_total={prev_total}")
                    else:
                        _ok("buckets_sum", False,
                            f"sum(trend previous)={sum_prev:.2f} != previous_total={prev_total}")
        except Exception as exc:
            _ok("buckets_sum", False, f"check failed (fail-closed): {exc}")

        # ---- breakdown_sum (scope-aware) -----------------------------------
        # Non-truncated: row sums must EQUAL the totals (unchanged strict rule).
        # Truncated (breakdown_scope.truncated): rows are the top-N groups, so
        # require containment instead — sum(rows) <= total, rows sorted DESC,
        # and top-row share math bounded in [0, 1]; likewise for previous.
        try:
            if not rows:
                _ok("breakdown_sum", True, "no breakdown rows to sum")
            elif not truncated:
                n = len(rows)
                sum_cur = sum(float((r or {}).get("current") or 0) for r in rows)
                tol = max(_ABS_TOL, 0.01 * n)
                if total is None or not close_enough(sum_cur, total, abs_tol=tol):
                    _ok("breakdown_sum", False,
                        f"sum(rows current)={sum_cur:.2f} != total={total}")
                elif prev_total is None:
                    _ok("breakdown_sum", True,
                        f"sum(rows current)={sum_cur:.2f} matches total; previous skipped (None)")
                else:
                    sum_prev = sum(float((r or {}).get("previous") or 0) for r in rows)
                    if close_enough(sum_prev, prev_total, abs_tol=tol):
                        _ok("breakdown_sum", True,
                            f"row sums match total={total} previous_total={prev_total}")
                    else:
                        _ok("breakdown_sum", False,
                            f"sum(rows previous)={sum_prev:.2f} != previous_total={prev_total}")
            else:
                _tr = _check_truncated_rows(rows, total, prev_total)
                _ok("breakdown_sum", _tr[0], _tr[1])
        except Exception as exc:
            _ok("breakdown_sum", False, f"check failed (fail-closed): {exc}")
        except Exception as exc:
            _ok("breakdown_sum", False, f"check failed (fail-closed): {exc}")

        # ---- arithmetic ----------------------------------------------------
        try:
            bad: Optional[str] = None
            for r in rows:
                r = r or {}
                cur = r.get("current")
                prv = r.get("previous")
                ch = r.get("change")
                cp = r.get("change_pct")
                label = str(r.get("category", r.get("label", "?")))
                if prv is None:
                    if ch is not None or cp is not None:
                        bad = (f"row '{label}': previous None but "
                               f"change={ch} change_pct={cp}")
                        break
                    continue
                try:
                    exp_ch = round(float(cur or 0) - float(prv), 2)
                except Exception:
                    bad = f"row '{label}': non-numeric current/previous"
                    break
                if ch is None or abs(float(ch) - exp_ch) > _PCT_TOL:
                    bad = f"row '{label}': change={ch} != cur-prev={exp_ch}"
                    break
                try:
                    if float(prv) == 0:
                        if cp is not None:
                            bad = (f"row '{label}': previous is 0 but "
                                   f"change_pct={cp} (must be None)")
                            break
                    else:
                        exp_cp = round((float(cur or 0) - float(prv))
                                       / abs(float(prv)) * 100, 2)
                        if cp is None or abs(float(cp) - exp_cp) > _PCT_TOL:
                            bad = (f"row '{label}': change_pct={cp} != "
                                   f"expected {exp_cp}")
                            break
                except Exception:
                    bad = f"row '{label}': non-numeric change_pct"
                    break
            if bad is None and total is not None and prev_total is not None:
                exp_ch = round(float(total) - float(prev_total), 2)
                ch_all = result.get("change")
                if ch_all is None or abs(float(ch_all) - exp_ch) > _PCT_TOL:
                    bad = f"top-level change={ch_all} != total-prev={exp_ch}"
                elif float(prev_total) != 0:
                    exp_cp = round((float(total) - float(prev_total))
                                   / abs(float(prev_total)) * 100, 2)
                    cp_all = result.get("change_pct")
                    if cp_all is None or abs(float(cp_all) - exp_cp) > _PCT_TOL:
                        bad = (f"top-level change_pct={cp_all} != "
                               f"expected {exp_cp}")
                elif result.get("change_pct") is not None:
                    bad = "top-level change_pct must be None when previous_total is 0"
            if bad is None and has_windows:
                for p in pts:
                    p = p or {}
                    if p.get("previous") is None and cmp_on and prev_total is not None:
                        bad = (f"trend point '{p.get('label', '?')}' has previous None "
                               f"while comparison is on and previous_total={prev_total}")
                        break
            if bad is None:
                _ok("arithmetic", True,
                    f"change/change_pct arithmetic holds over {len(rows)} rows + top level")
            else:
                _ok("arithmetic", False, bad)
        except Exception as exc:
            _ok("arithmetic", False, f"check failed (fail-closed): {exc}")

        # ---- date_sanity ----------------------------------------------------
        try:
            if not qd and not col_name:
                _ok("date_sanity", True, "no date column in play; skipped")
            elif not has_windows:
                if is_suspicious_date_col(col_name or ""):
                    _ok("date_sanity", False,
                        f"date column '{col_name}' looks like birth/hire/dob/death")
                else:
                    _ok("date_sanity", True, "no time windows (snapshot); skipped")
            elif is_suspicious_date_col(col_name or ""):
                _ok("date_sanity", False,
                    f"date column '{col_name}' looks like birth/hire/dob/death, not an event date")
            else:
                lo_hi = _row(engine, f"SELECT MIN({qd}), MAX({qd}) FROM {qt}")
                mind, maxd = to_date(lo_hi[0]), to_date(lo_hi[1] if len(lo_hi) > 1 else None)
                if (mind is None or maxd is None or mind > maxd) and non_native:
                    # Lexicographic MIN/MAX over TEXT/odd storage is
                    # meaningless (e.g. MM-DD-YYYY) — parse DISTINCT values.
                    mind, maxd = _distinct_range(engine, qd, qt)
                passed, detail = check_date_sanity(
                    col_name, cur_start, cur_end, prev_start, prev_end, mind, maxd)
                _ok("date_sanity", passed, detail)
        except Exception as exc:
            _ok("date_sanity", False, f"query failed (fail-closed): {exc}")

        # ---- sparkline_check -------------------------------------------------
        try:
            n = len(pts)
            trends = [(r or {}).get("trend") for r in rows] if rows else []
            if n == 0 or not trends or all(not t for t in trends):
                _ok("sparkline_check", True, "no time buckets (snapshot); skipped")
            else:
                bad = None
                for r in rows:
                    t = (r or {}).get("trend") or []
                    if t and len(t) != n:
                        bad = (f"row '{(r or {}).get('category', '?')}' trend "
                               f"len {len(t)} != {n} trend points")
                        break
                if bad is None:
                    _ok("sparkline_check", True,
                        f"all row sparklines length {n}")
                else:
                    _ok("sparkline_check", False, bad)
        except Exception as exc:
            _ok("sparkline_check", False, f"check failed (fail-closed): {exc}")

        passed_all = all(c["passed"] for c in checks)
        return {"passed": passed_all, "checks": checks, "corrected": False}
    except Exception as exc:  # never raise out of the validator
        log.exception("intelligence validation crashed")
        return {"passed": False,
                "checks": [{"name": "validator_error", "passed": False,
                            "detail": f"validator crashed (fail-closed): {exc}"}],
                "corrected": False}


# ---------------------------------------------------------------------------
# What-If validator (pure arithmetic, warn-only)
# ---------------------------------------------------------------------------

def validate_whatif(result: dict) -> dict:
    """Re-check What-If arithmetic: projected = current*(1±pct) + monthly.

    Pure function (no SQL). Never raises.
    """
    checks: List[Dict[str, Any]] = []

    def _ok(name: str, passed: bool, detail: str):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    try:
        result = result or {}
        try:
            pct = float(result.get("change_value") or 0) / 100.0
        except Exception:
            pct = 0.0
        ct = str(result.get("change_type") or "increase").lower()
        sign = -1 if ("decr" in ct or "reduc" in ct or ct in ("decrease", "down", "-")) else 1

        def _expected(cur):
            try:
                return float(cur) * (1 + sign * pct)
            except Exception:
                return None

        # ---- projected_check -------------------------------------------
        try:
            cur = result.get("current_value")
            proj = result.get("projected_value")
            exp = _expected(cur)
            if cur is None or proj is None or exp is None:
                _ok("projected_check", False, "current/projected value missing")
            elif close_enough(proj, exp, abs_tol=max(_ABS_TOL, 0.01 * max(1.0, abs(exp)))):
                _ok("projected_check", True,
                    f"projected={float(proj):.2f} matches current*(1±pct)={exp:.2f}")
            else:
                _ok("projected_check", False,
                    f"projected={float(proj):.2f} != current*(1±pct)={exp:.2f}")
        except Exception as exc:
            _ok("projected_check", False, f"check failed: {exc}")

        # ---- delta_check -------------------------------------------------
        try:
            cur = result.get("current_value")
            proj = result.get("projected_value")
            delta = result.get("delta")
            if cur is None or proj is None or delta is None:
                _ok("delta_check", False, "current/projected/delta missing")
            else:
                exp_d = float(proj) - float(cur)
                if close_enough(delta, exp_d, abs_tol=max(_ABS_TOL, 0.005 * abs(exp_d) if exp_d else _ABS_TOL)):
                    _ok("delta_check", True, f"delta={float(delta):.2f} matches projected-current")
                else:
                    _ok("delta_check", False,
                        f"delta={float(delta):.2f} != projected-current={exp_d:.2f}")
        except Exception as exc:
            _ok("delta_check", False, f"check failed: {exc}")

        # ---- revenue_check -----------------------------------------------
        try:
            rev = result.get("revenue") or {}
            rc, rp = rev.get("current"), rev.get("projected")
            if rc is None or rp is None:
                _ok("revenue_check", True, "no revenue block; skipped")
            else:
                exp = _expected(rc)
                if exp is not None and close_enough(rp, exp, abs_tol=max(_ABS_TOL, 0.005 * abs(exp) if exp else _ABS_TOL)):
                    _ok("revenue_check", True, "revenue projection matches (1±pct)")
                else:
                    _ok("revenue_check", False,
                        f"revenue projected={rp} != expected {exp}")
        except Exception as exc:
            _ok("revenue_check", False, f"check failed: {exc}")

        # ---- monthly_check ------------------------------------------------
        try:
            monthly = result.get("monthly") or result.get("comparison") or []
            if not monthly:
                _ok("monthly_check", True, "no monthly series; skipped")
            else:
                bad = None
                for m in monthly:
                    m = m or {}
                    if "baseline" not in m or "scenario" not in m:
                        bad = f"monthly entry '{m.get('label', '?')}' missing baseline/scenario"
                        break
                    exp = _expected(m.get("baseline"))
                    if exp is None or not close_enough(
                            m.get("scenario"), exp,
                            abs_tol=max(_ABS_TOL, 0.005 * abs(exp) if exp else _ABS_TOL)):
                        bad = (f"monthly '{m.get('label', '?')}': scenario={m.get('scenario')} "
                               f"!= baseline*(1±pct)={exp}")
                        break
                if bad is None:
                    _ok("monthly_check", True,
                        f"{len(monthly)} monthly entries: baseline/scenario lengths equal, arithmetic holds")
                else:
                    _ok("monthly_check", False, bad)
        except Exception as exc:
            _ok("monthly_check", False, f"check failed: {exc}")

        passed_all = all(c["passed"] for c in checks)
        return {"passed": passed_all, "checks": checks, "corrected": False}
    except Exception as exc:
        log.exception("what-if validation crashed")
        return {"passed": False,
                "checks": [{"name": "validator_error", "passed": False,
                            "detail": f"validator crashed: {exc}"}],
                "corrected": False}
