"""Insights detection engine — fully dynamic, database-agnostic.

Generates real insights from ANY connected database using schema metadata
(data_type, primary_key, declared_relationships) — zero hardcoded assumptions.

Provides GET /api/insights, POST /api/insights/save, DELETE /api/insights/save/{id},
GET /api/insights/saved, POST /api/insights/{id}/ask.
"""

import logging
import sqlite3
import math
import statistics
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.deps import require_writable, reject_document_sql
from core.config import get_session, save_session
from core.schema_intel import SchemaIntelligence, _safe_name, _run_sql, AMOUNT_HINTS
from core.errors import DB_ERRORS, PURE_ERRORS, get_request_id, log_and_sanitize, sanitize_error
from core.semantic import humanize

router = APIRouter(tags=["insights"])
from services.insights_service import (_safe_float, _fmt, _fmt_currency, _fmt_pct, _iqr_bands, _sanitize_spec_rows, _normalize_chart_spec, _confidence_level, _evidence_quality, _capability_flags, _histogram_bins, _build_takeaways, _enrich_insight, _normalize_legacy_insight, _translate_sql_for_sqlite, build_recommendation_detail, build_explains, CHART_SPEC_VOCAB, _MAX_SPEC_ROWS, _MAX_RAW_VALUES)  # noqa: F401,E402  # thin-wrapper re-export

log = logging.getLogger("aria.insights")


# ── Helpers (local to insights engine) ──────────────────────────────────


class _SQLiteCursor:
    """Wrap a sqlite3 cursor so InsightsEngine's PG-style calls work."""

    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=()):
        return self._cur.execute(_translate_sql_for_sqlite(sql), params or ())

    def __getattr__(self, name):
        return getattr(self._cur, name)


class _SQLiteCompatConnection:
    """Wrap a sqlite3 connection with the DB-API surface InsightsEngine uses."""

    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return _SQLiteCursor(self._conn.cursor())

    def close(self):
        try:
            self._conn.close()
        except (OSError, AttributeError, ValueError, sqlite3.Error) as exc:
            log.debug("sqlite compat close skipped: %s", exc)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _fmtm(v, measure=None):
    """Measure-aware money formatting (generic, never per-column).

    Currency symbols render only for money-like measures (price/cost/
    amount/revenue/...); counts, flags and other numerics render as plain
    comma-separated numbers. Prevents "$6K items" nonsense on any dataset.
    """
    try:
        _ml = str(measure or "").lower()
        # NOTE: bare "total" is deliberately excluded — Total_Items and
        # other counts contain it. Money needs a currency-ish token.
        _money = ("price", "cost", "amount", "revenue", "profit", "salary",
                  "payment", "spend", "budget", "fee", "fare",
                  "tuition", "premium", "wage", "income", "expense",
                  "balance", "sales", "margin")
        if any(_k in _ml for _k in _money):
            return _fmt_currency(v)
        import math as _math
        try:
            _f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if _f != _f:
            return str(v)
        if abs(_f) >= 1_000_000:
            return f"{_f / 1_000_000:.1f}M"
        if abs(_f) >= 10_000:
            return f"{_f / 1_000:.1f}K"
        if abs(_f) >= 100:
            return f"{int(round(_f)):,}"
        return f"{round(_f, 1)}"
    except Exception:
        try:
            return _fmt_currency(v)
        except Exception:
            return str(v)


class InsightsEngine:
    """Detect real insights from any connected database. Fully dynamic.

    Uses shared SchemaIntelligence from core.schema_intel — no duplication.
    Accepts either a raw DB-API connection (psycopg2/pymysql) or a
    SQLAlchemy engine for cross-table queries.
    """

    def __init__(self, conn, mapping: dict, llm=None):
        self.conn = conn
        self.mapping = mapping
        self.llm = llm
        self.intel = SchemaIntelligence(mapping)
        # Large-table sampling (generic): tables above _SAMPLE_ROWS_CAP run
        # detectors against a deterministic ~100k-row TEMP sample instead of
        # full scans. Metadata (intel) always uses the real table name; only
        # SQL FROM clauses go through self._t(). TEMP tables die with the
        # connection, so no cleanup bookkeeping is needed.
        self._sqlt: dict = {}
        self.sample_info: dict = {"sampled": False, "tables": {}}

    _SAMPLE_ROWS_CAP = 100_000

    def _dialect(self) -> str:
        """Connection dialect for date math (generic, never per-database)."""
        mod = type(self.conn).__module__.lower()
        if "pymysql" in mod or "mysql" in mod:
            return "mysql"
        if "sqlite" in mod or "compat" in mod:
            return "sqlite"
        # _SQLiteCompatConnection (defined in this module) speaks SQLite
        # underneath while accepting PG surface syntax: date math must be
        # SQLite-native so it runs without translation gaps.
        try:
            if isinstance(self.conn, _SQLiteCompatConnection):
                return "sqlite"
        except Exception:
            pass
        # SQLAlchemy engines expose their dialect directly (a sqlite-backed
        # engine must not be treated as PostgreSQL).
        try:
            _d = getattr(getattr(self.conn, "dialect", None), "name", "")
            if _d in ("sqlite", "mysql", "postgresql"):
                return _d
        except Exception:
            pass
        return "postgresql"

    def _col_dtype(self, table: str, col: str) -> str:
        """Native data_type for a column (either mapping shape)."""
        try:
            tinfo = (self.mapping.get("tables") or {}).get(table) or {}
            cols = tinfo.get("columns", [])
            if isinstance(cols, dict):
                m = cols.get(col) or {}
                return str(m.get("data_type", ""))
            for c in cols:
                if isinstance(c, dict) and c.get("column") == col:
                    return str(c.get("data_type", ""))
        except Exception:
            pass
        return ""

    def _mexpr(self, table: str, col: str) -> str:
        """Month-truncation expression for the live dialect, TEXT-date safe.

        Native DATE columns use the fast native truncation (unchanged
        output). TEXT dates (ISO, DD-MM-YYYY, YYYY-MM month grain) go
        through the shared date parser first — without it strftime/
        DATE_TRUNC yields NULL and every detector collapses to one NULL
        group. Output grain is identical either way (%Y-%m-01 / month
        start) so period labels downstream never change.
        """
        q = _safe_name(col)
        d = self._dialect()
        dtype = self._col_dtype(table, col).lower()
        is_text = (not dtype or any(t in dtype for t in
                    ("char", "text", "varchar", "string", "clob", "uuid")))
        expr = q
        if is_text:
            try:
                from core.data_intelligence import date_parse_expression as _dpe
                expr = _dpe(q, d if d in ("mysql", "sqlite") else "postgresql")
            except Exception:
                expr = q
        if d == "mysql":
            return f"DATE_FORMAT({expr}, '%Y-%m-01')"
        if d == "sqlite":
            return f"strftime('%Y-%m-01', {expr})"
        if is_text:
            return f"DATE_TRUNC('month', {expr})"
        return f"DATE_TRUNC('month', {q})"

    def _t(self, table: str) -> str:
        """SQL table for detectors: TEMP sample when one was built."""
        return self._sqlt.get(table, table)

    def _sexpr(self, table: str, col: str, qualified_table: str | None = None) -> str:
        """SUM-safe expression for a measure: TEXT-storage aware.

        Fully schema-agnostic (no table/column literals): when the declared
        dtype is text-ish (e.g. a relational table holding ``'\\u20b91,299'``
        or ``'15%'`` in TEXT columns), returns the shared
        :func:`core.text_numeric.coercion_sql` expression so aggregates
        yield numbers instead of raising ``function sum(text) does not
        exist``. Unparsable values become NULL (never invented). Native
        numerics and unknown dtypes keep the plain quoted column. The
        qualifier honours TEMP samples; table-qualified refs (correlations)
        resolve dtypes from their own table. Never raises.
        """
        from core.text_numeric import coercion_sql as _cs
        try:
            d = self._dialect()
        except Exception:
            d = "postgresql"
        _qd = d if d in ("mysql", "sqlite", "postgresql") else "postgresql"
        try:
            base = f"{_safe_name(qualified_table, _qd)}.{_safe_name(col, _qd)}" \
                if qualified_table else _safe_name(col, _qd)
        except Exception:
            base = str(col)
        try:
            dtype = self._col_dtype(qualified_table or table, col).lower()
        except Exception:
            dtype = ""
        if any(k in dtype for k in
               ("char", "text", "varchar", "string", "clob", "uuid")):
            try:
                return _cs(_qd, base)
            except Exception:
                return base
        return base

    def _drop_partial_trailing_month(self, table: str, date_col: str, rows: list) -> list:
        """Drop a trailing partial month (generic incomplete-period guard).

        A month holding only part of its days (data ends mid-month) reads
        as a collapse against full months. When the table's MAX(date) falls
        before its month-end AND more than one month exists, the last
        period is excluded from growth/anomaly reads. Returns filtered rows.
        """
        try:
            if len(rows) < 3:
                return rows
            _mr = _run_sql(self.conn, f"SELECT MAX({_safe_name(date_col)}) FROM {_safe_name(self._t(table))} WHERE {_safe_name(date_col)} IS NOT NULL", limit=1)
            _mrows = _mr[1] if isinstance(_mr, tuple) else []
            if not _mrows:
                return rows
            _first = _mrows[0]
            # Column name varies by dialect (MAX("Date") vs max) — take
            # the first value regardless of key.
            _raw = next(iter(_first.values()), None) if isinstance(_first, dict) else _first[0]
            _s = str(_raw)[:10]
            _y, _m, _d = int(_s[:4]), int(_s[5:7]), int(_s[8:10])
            import calendar as _cal
            _month_end = _cal.monthrange(_y, _m)[1]
            if _d >= _month_end:
                return rows
            _last = rows[-1].get("period") if isinstance(rows[-1], dict) else None
            if _last and str(_last)[:7] == f"{_y}-{_m:02d}":
                return rows[:-1]
            return rows
        except Exception:
            return rows

    def _maybe_sample(self, table: str) -> None:
        """Build a deterministic ~100k-row TEMP sample for huge tables."""
        if table in self._sqlt:
            return
        try:
            ti = (self.mapping.get("tables") or {}).get(table) or {}
            rc = ti.get("row_count")
            if not isinstance(rc, int) or rc <= self._SAMPLE_ROWS_CAP:
                return
            import math as _math
            k = max(1, int(_math.ceil(rc / self._SAMPLE_ROWS_CAP)))
            sname = f"aria_smp_{table}"[:60]
            _real = _safe_name(table)
            _method = "deterministic-stride"
            # Detect dialect from the connection object.
            mod = type(self.conn).__module__.lower()
            if "psycopg2" in mod or "postgres" in mod:
                pct = max(0.1, min(100.0, self._SAMPLE_ROWS_CAP * 100.0 / rc))
                # REPEATABLE seed: same sample every run, not run-to-run flap.
                ddl = (f'CREATE TEMP TABLE {_safe_name(sname)} AS SELECT * FROM '
                       f'{_real} TABLESAMPLE SYSTEM ({pct}) REPEATABLE (42)')
                _method = "system-sample-repeatable-42"
            elif "pymysql" in mod or "mysql" in mod:
                # NOTE: MySQL RAND(constant) returns one value per query
                # (all-or-nothing), so a seeded variant is impossible here.
                # Unseeded RAND() gives correct volume, honestly labeled.
                ddl = (f'CREATE TEMPORARY TABLE {_safe_name(sname)} AS SELECT * FROM '
                       f'{_real} WHERE RAND() < {self._SAMPLE_ROWS_CAP / rc}')
                _method = "random"
            else:
                # sqlite (incl. _SQLiteCompatConnection): deterministic rowid stride.
                ddl = (f'CREATE TEMP TABLE {_safe_name(sname)} AS SELECT * FROM '
                       f'{_real} WHERE (rowid % {k}) = 0')
            cur = self.conn.cursor() if hasattr(self.conn, "cursor") else None
            if cur is None and hasattr(self.conn, "execute"):
                self.conn.execute(ddl)
                try:
                    self.conn.commit()
                except Exception:
                    pass
            else:
                cur.execute(ddl)
                try:
                    self.conn.commit()
                except Exception:
                    pass
                cur.close()
            self._sqlt[table] = sname
            self.sample_info["sampled"] = True
            self.sample_info["tables"][table] = {"sample_rows": self._SAMPLE_ROWS_CAP,
                                                 "total_rows": rc, "method": _method}
        except Exception as exc:
            log.debug("insights sampling skipped for %s: %s", table, exc)

    def _rollback(self) -> None:
        """Reset a poisoned transaction so later detectors keep working."""
        try:
            self.conn.rollback()
        except Exception:
            pass

    def generate_all_insights(self, max_insights: int = 20):
        """Run all detectors, return ranked insights."""
        # Centralized global gate (core/evidence.py): rows < 10 ->
        # INSUFFICIENT_EVIDENCE for every substantive conclusion. The
        # detectors below each carry their own per-capability minimums
        # (trends/anomalies/correlations >= 6 periods, distributions >= 20
        # rows, outliers >= 10 rows); this gate runs first and dominates.
        try:
            from core.evidence import assess_evidence as _assess
            _total = 0
            _known = False
            for _t, _ti in (self.mapping.get("tables") or {}).items():
                _rc = (_ti or {}).get("row_count")
                if isinstance(_rc, int):
                    _known = True
                    _total += max(0, _rc)
            if _known and _assess(_total, "insights").get("status") != "supported":
                return []
        except Exception:
            pass
        insights = []

        for tname in self.intel.tables:
            # Skip empty tables
            tinfo = self.intel.tables[tname]
            if tinfo.get("empty"):
                continue

            # Bound huge tables to a deterministic sample before detectors.
            self._maybe_sample(tname)

            # Skip tables with no meaningful numeric measures (dimension-only tables)
            measure = self.intel.find_best_measure(tname)
            if not measure:
                continue
            # Weak mappings can label text columns as measures; only real
            # numerics reach SQL aggregates (a text SUM would poison the
            # shared transaction for every later detector).
            if measure not in self.intel.get_numeric_cols(tname):
                continue
            # Coverage across measures (generic): the best measure plus up
            # to 2 more numeric non-key columns, so wide tables surface
            # more than one story. Identifier-like columns (PK flags, or
            # data-driven: distinct == row count, e.g. Transaction_ID) never
            # qualify — summing an ID is meaningless.
            _pks = {str(c).lower() for c in (
                (tinfo.get("primary_key") or []) + (tinfo.get("inferred_primary_key") or []))}
            _id_name = __import__("re").compile(r"(^|_)(id|key|code|no|number|guid|uuid)($|_)").search
            _col_flags = {}
            for _cc in (tinfo.get("columns") or []):
                if isinstance(_cc, dict) and _cc.get("column"):
                    if _cc.get("is_primary_key"):
                        _pks.add(str(_cc.get("column")).lower())
            def _looks_identifier(_c):
                if str(_c).lower() in _pks:
                    return True
                if not _id_name(str(_c).lower()):
                    return False
                try:
                    _ir = _run_sql(self.conn, f"SELECT COUNT(*) AS n, COUNT(DISTINCT {_safe_name(self._t(tname))}.{_safe_name(_c)}) AS d FROM {_safe_name(self._t(tname))}", limit=1)
                    _rows = _ir[1] if isinstance(_ir, tuple) else []
                    if _rows:
                        _n, _d = _rows[0].get("n"), _rows[0].get("d")
                        return bool(_n and _d == _n)
                except Exception:
                    pass
                return False
            _measures = [measure]
            for _c in self.intel.get_numeric_cols(tname):
                if _c != measure and len(_measures) < 3 and not _looks_identifier(_c):
                    _measures.append(_c)

            def _run(det, *a):
                for _m in _measures:
                    try:
                        insights.extend(det(tname, _m))
                    except DB_ERRORS as exc:
                        log.debug("insights detector failed for %s: %s", tname, exc)
                        self._rollback()

            _run(self._detect_growth_decline)
            _run(self._detect_trends)
            _run(self._detect_trend_leaderboard)
            _run(self._detect_anomalies)
            _run(self._detect_opportunities)
            _run(self._detect_risks)
            # Non-temporal detectors (work without date columns)
            _run(self._detect_value_outliers)
            _run(self._detect_concentration)
            _run(self._detect_distributions)
            _run(self._detect_segment_comparison)

        # Cross-table correlation
        try:
            insights.extend(self._detect_correlations())
        except DB_ERRORS as exc:
            log.debug("correlations detector failed: %s", exc)

        # Normalize every insight (B-shapes: summary->description, viz_type/
        # viz_data->visualization via _normalize_chart_spec) so ALL insights
        # carry visualization:{type,x,y,series,title,interactive,data,
        # insufficient,unsupported} + the generic per-insight contract.
        normalized = []
        for _ins in insights:
            try:
                if isinstance(_ins, dict) and isinstance(_ins.get("visualization"), dict) and _ins.get("description"):
                    normalized.append(_enrich_insight(_ins))
                else:
                    normalized.append(_normalize_legacy_insight(_ins) if isinstance(_ins, dict) else _ins)
            except PURE_ERRORS as exc:
                log.debug("insight normalize skipped: %s", exc)
                normalized.append(_ins)
        insights = self._deduplicate(normalized)
        insights = self._rank(insights)
        return insights[:max_insights]

    # ── Growth / Decline ──────────────────────────────────────────────────

    def _detect_growth_decline(self, table: str, measure: str | None = None):
        insights = []
        date_col = self.intel.find_best_datetime(table)
        if not date_col:
            return []

        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []

        # Get current vs previous period
        sql = f"""
            WITH monthly AS (
                SELECT {self._mexpr(table, date_col)} AS period,
                       SUM({self._sexpr(table, measure)}) AS val
                FROM {_safe_name(self._t(table))}
                WHERE {_safe_name(date_col)} IS NOT NULL
                GROUP BY period
                ORDER BY period DESC
                LIMIT 12
            )
            SELECT period, val FROM monthly ORDER BY period
        """
        _, rows = _run_sql(self.conn, sql, limit=12)
        # Never compare a trailing partial month against full months.
        rows = self._drop_partial_trailing_month(table, date_col, rows)
        if len(rows) < 2:
            return []

        values = [(r["period"], _safe_float(r["val"])) for r in rows if _safe_float(r["val"]) is not None]
        if len(values) < 2:
            return []

        current_val = values[-1][1]
        previous_val = values[-2][1]
        if previous_val == 0 or previous_val is None:
            return []

        change_pct = ((current_val - previous_val) / abs(previous_val)) * 100
        period_label = self._period_label(values[-1][0], values[-2][0])

        # Determine table description for titles
        tdesc = self.intel.tables[table].get("description", table)
        metric_label = f"{measure} ({tdesc})" if tdesc and tdesc != table else measure

        confidence = self._compute_confidence(change_pct, len(values), abs(current_val), abs(previous_val))
        viz_data = [{"period": str(v[0])[:7], "value": v[1]} for v in values]

        observations = [
            f"{humanize(measure)} was {_fmtm(previous_val, measure)} in the previous period.",
            f"{humanize(measure)} is {_fmtm(current_val, measure)} in the current period.",
            f"This is a {_fmt_pct(change_pct)} change.",
        ]
        if len(values) >= 4:
            avg = statistics.mean([v[1] for v in values])
            observations.append(f"Typical level across the last {len(values)} periods: {_fmtm(avg, measure)}.")

        if change_pct > 5:
            insight_type = "Growth"
            severity = self._compute_severity(change_pct, confidence, current_val)
            title = f"{humanize(measure)} grew {_fmt_pct(change_pct)} to {_fmtm(current_val, measure)}"
        elif change_pct < -5:
            insight_type = "Decline"
            severity = self._compute_severity(abs(change_pct), confidence, current_val)
            title = f"{humanize(measure)} fell {_fmt_pct(change_pct)} to {_fmtm(current_val, measure)}"
        else:
            return []

        impact_value = current_val - previous_val
        dims = self.intel.get_text_cols(table)[:3]
        _move_word = "up" if change_pct > 0 else "down"

        insights.append(self._build_insight(
            insight_id=f"gd_{table}_{measure}",
            insight_type=insight_type,
            severity=severity,
            title=title,
            description=f"{humanize(measure)} moved from {_fmtm(previous_val, measure)} to {_fmtm(current_val, measure)} ({_fmt_pct(change_pct)}) between {period_label}. That is {_move_word} {abs(change_pct):.0f}% — a move worth noting.",
            metric=metric_label,
            metric_value=current_val,
            previous_value=previous_val,
            magnitude=_fmt_pct(change_pct),
            confidence=confidence,
            affected_period=period_label,
            tags=[table, measure, insight_type.lower()],
            observations=observations,
            support_metrics=[f"Period-over-period change: {_fmt_pct(change_pct)}"],
            sample_size=len(values),
            baseline=previous_val,
            current_value=current_val,
            change_pct=change_pct,
            impact_value=impact_value,
            impact_direction="positive" if impact_value > 0 else "negative",
            impact_explanation=f"The {_fmt_pct(change_pct)} move in {humanize(measure).lower()} means {'a gain' if impact_value > 0 else 'a drop'} of {_fmtm(abs(impact_value))}.",
            next_question=f"What caused the {_fmt_pct(change_pct)} {'rise' if change_pct > 0 else 'fall'} in {humanize(measure).lower()}?",
            next_dims=dims,
            viz_type="area",
            viz_data=viz_data,
            source_tables=[table],
            source_columns=[date_col, measure],
        ))

        return insights

    # ── Trends ────────────────────────────────────────────────────────────

    def _detect_trends(self, table: str, measure: str | None = None):
        insights = []
        date_col = self.intel.find_best_datetime(table)
        if not date_col:
            return []

        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []

        sql = f"""
            SELECT {self._mexpr(table, date_col)} AS period,
                   SUM({self._sexpr(table, measure)}) AS val
            FROM {_safe_name(self._t(table))}
            WHERE {_safe_name(date_col)} IS NOT NULL
            GROUP BY period
            ORDER BY period
            LIMIT 24
        """
        _, rows = _run_sql(self.conn, sql, limit=24)
        rows = self._drop_partial_trailing_month(table, date_col, rows)
        if len(rows) < 6:
            return []

        values = [(r["period"], _safe_float(r["val"])) for r in rows if _safe_float(r["val"]) is not None]
        if len(values) < 6:
            return []

        nums = [v[1] for v in values]
        n = len(nums)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(nums)
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(nums))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return []

        slope = numerator / denominator
        if y_mean == 0:
            return []
        slope_pct = (slope / abs(y_mean)) * 100

        consecutive = sum(1 for i in range(1, n) if (nums[i] > nums[i-1]) == (slope > 0))
        consistency = consecutive / (n - 1)

        if abs(slope_pct) < 1 or consistency < 0.6:
            return []

        ss_res = sum((y - (y_mean + slope * (i - x_mean))) ** 2 for i, y in enumerate(nums))
        ss_tot = sum((y - y_mean) ** 2 for y in nums)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        confidence = min(95, max(30, int(r_squared * 100 * consistency)))
        direction = "increasing" if slope > 0 else "decreasing"
        period_months = self._period_label(values[-1][0], values[0][0])
        tdesc = self.intel.tables[table].get("description", table)
        severity = self._compute_severity(abs(slope_pct) * len(nums), confidence, nums[-1])
        # Idea 7: flag unusual periods on the trend line itself (same IQR +
        # z-score rules as the anomaly detector, computed on this series).
        # The frontend renders flagged points in red with an "unusual" label.
        # Bands carry expected (median of in-fence values) + fences per row so
        # the frontend can draw the usual-range overlay. Flat/degenerate
        # series (bands None) fall back to z-score only — no fake band.
        _bands = _iqr_bands(nums)
        if _bands is not None:
            _q1, _q3, _iqr, _lo, _hi, _sev_lo, _sev_hi = _bands
            _in_fence = [v for v in nums if _lo <= v <= _hi]
            _trend_expected = float(statistics.median(_in_fence)) if _in_fence else float(statistics.median(nums))
        else:
            _lo = _hi = None
            _trend_expected = float(statistics.median(nums))
        _mean = statistics.mean(nums)
        _std = statistics.stdev(nums) if len(nums) > 1 else 0
        def _is_unusual(v):
            if _lo is not None and _hi is not None and (v < _lo or v > _hi):
                return True
            if _std > 0 and abs((v - _mean) / _std) > 2.0:
                return True
            return False
        if _bands is not None:
            viz_data = [{"period": str(v[0])[:7], "value": v[1], "anomaly": _is_unusual(v[1]),
                         "expected": _trend_expected, "bandLower": _lo, "bandUpper": _hi} for v in values]
        else:
            viz_data = [{"period": str(v[0])[:7], "value": v[1], "anomaly": _is_unusual(v[1])} for v in values]
        dims = self.intel.get_text_cols(table)[:3]

        insights.append(self._build_insight(
            insight_id=f"tr_{table}_{measure}",
            insight_type="Trend",
            severity=severity,
            title=f"{humanize(measure)} is {direction} ({_fmt_pct(slope_pct)} a period)",
            description=f"{humanize(measure)} has been {direction} steadily over {len(values)} periods ({period_months}), changing by {_fmt_pct(slope_pct)} a period on average.",
            metric=f"{humanize(measure)}",
            metric_value=nums[-1],
            previous_value=nums[0],
            magnitude=_fmt_pct(slope_pct * len(nums)),
            confidence=confidence,
            affected_period=period_months,
            tags=[table, measure, "trend", direction],
            observations=[
                f"{humanize(measure)} moved from {_fmtm(nums[0], measure)} to {_fmtm(nums[-1], measure)} over {len(values)} periods.",
                f"It moved in the same direction in {consistency*100:.0f}% of those periods.",
            ] + ([
                f"{sum(1 for v in values if _is_unusual(v[1]))} of those periods sit outside the usual range and are highlighted on the chart."
            ] if any(_is_unusual(v[1]) for v in values) else []),
            support_metrics=[f"Steady {_fmt_pct(slope_pct)} change a period"],
            sample_size=len(values),
            baseline=nums[0],
            current_value=nums[-1],
            change_pct=((nums[-1] - nums[0]) / abs(nums[0])) * 100 if nums[0] != 0 else 0,
            impact_value=nums[-1] - nums[0],
            impact_direction="positive" if slope > 0 else "negative",
            impact_explanation=f"A sustained {direction} pattern in {humanize(measure).lower()} over {len(values)} periods points to a {'healthy' if slope > 0 else 'worrying'} trajectory.",
            next_question=f"What is driving the {direction} pattern in {humanize(measure).lower()}?",
            next_dims=dims,
            viz_type="area",
            viz_data=viz_data,
            source_tables=[table],
            source_columns=[date_col, measure],
        ))

        return insights

    # ── Trend leaderboard (idea 8) ──────────────────────────────────────

    def _detect_trend_leaderboard(self, table: str, measure: str | None = None):
        """Rank every usable measure in one table by trend speed.

        Dynamic: uses whatever numeric measures the schema actually has
        (via find_all_measures — never assumed names). Needs a date column
        and 2+ measures with enough history each. Returns one insight whose
        chart is a diverging bar of per-period % change, or [] when the
        data cannot support a fair ranking.
        """
        date_col = self.intel.find_best_datetime(table)
        if not date_col:
            return []
        measures = self.intel.find_all_measures(table)
        if len(measures) < 2:
            return []

        ranked = []
        period_label = ""
        n_periods = 0
        for measure in measures[:8]:
            sql = f"""
                SELECT {self._mexpr(table, date_col)} AS period,
                       SUM({self._sexpr(table, measure)}) AS val
                FROM {_safe_name(self._t(table))}
                WHERE {_safe_name(date_col)} IS NOT NULL
                GROUP BY period
                ORDER BY period
                LIMIT 24
            """
            _, rows = _run_sql(self.conn, sql, limit=24)
            rows = self._drop_partial_trailing_month(table, date_col, rows)
            if len(rows) < 6:
                continue
            values = [_safe_float(r["val"]) for r in rows if _safe_float(r["val"]) is not None]
            if len(values) < 6:
                continue
            nums = values
            n = len(nums)
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(nums)
            if y_mean == 0:
                continue
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(nums))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            if denominator == 0:
                continue
            slope = numerator / denominator
            slope_pct = (slope / abs(y_mean)) * 100
            consecutive = sum(1 for i in range(1, n) if (nums[i] > nums[i - 1]) == (slope > 0))
            consistency = consecutive / (n - 1)
            if abs(slope_pct) < 1 or consistency < 0.6:
                continue
            ss_res = sum((y - (y_mean + slope * (i - x_mean))) ** 2 for i, y in enumerate(nums))
            ss_tot = sum((y - y_mean) ** 2 for y in nums)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            ranked.append({
                "measure": measure,
                "slope_pct": round(slope_pct, 1),
                "consistency": round(consistency, 2),
                "r_squared": round(r_squared, 2),
                "score": abs(slope_pct) * consistency,
                "periods": n,
            })
            period_label = self._period_label(rows[-1]["period"], rows[0]["period"])
            n_periods = max(n_periods, n)

        if len(ranked) < 2:
            return []
        ranked.sort(key=lambda r: r["score"], reverse=True)
        top = ranked[0]
        bottom = ranked[-1]
        tdesc = self.intel.tables[table].get("description", table)
        top_dir = "rising" if top["slope_pct"] > 0 else "falling"
        confidence = min(90, max(35, int(top["r_squared"] * 100 * top["consistency"])))
        severity = self._compute_severity(abs(top["slope_pct"]) * top["periods"], confidence, abs(top["slope_pct"]))
        board_lines = ", ".join(
            f"{humanize(r['measure'])} {r['slope_pct']:+.1f}%/period" for r in ranked
        )
        viz_data = [{"name": humanize(r["measure"]), "value": r["slope_pct"]} for r in ranked]
        dims = self.intel.get_text_cols(table)[:3]

        return [self._build_insight(
            insight_id=f"tl_{table}",
            insight_type="Trend",
            severity=severity,
            title=f"Fastest {top_dir} measure: {humanize(top['measure'])} ({top['slope_pct']:+.1f}% a period)",
            description=f"Across {len(ranked)} comparable measures in {period_label}, {humanize(top['measure'])} moves fastest ({top['slope_pct']:+.1f}% a period), while {humanize(bottom['measure'])} moves slowest ({bottom['slope_pct']:+.1f}% a period). Ranking: {board_lines}.",
            metric=f"{tdesc or table} · measure ranking",
            magnitude=f"{top['slope_pct']:+.1f}%/period",
            confidence=confidence,
            affected_period=period_label or f"{n_periods} periods",
            tags=[table, "trend", "leaderboard", top_dir],
            observations=[
                f"Compared {len(ranked)} measures by their sustained move each period over {n_periods} periods." ,
                f"Fastest: {humanize(top['measure'])} at {top['slope_pct']:+.1f}% a period "
                f"(same direction in {top['consistency']*100:.0f}% of periods).",
                f"Slowest: {humanize(bottom['measure'])} at {bottom['slope_pct']:+.1f}% a period.",
            ],
            support_metrics=[f"{humanize(r['measure'])}: {r['slope_pct']:+.1f}%/period" for r in ranked[:5]],
            sample_size=n_periods,
            impact_direction="positive" if top["slope_pct"] > 0 else "negative",
            impact_explanation=(
                f"{humanize(top['measure'])} is pulling hardest in "
                f"{'an upward' if top['slope_pct'] > 0 else 'a downward'} direction; "
                f"the available data does not contain enough information to estimate its financial impact."
            ),
            next_question=f"What is driving the {top_dir} movement in {humanize(top['measure']).lower()}?",
            next_dims=dims,
            viz_type="leaderboard",
            viz_data=viz_data,
            source_tables=[table],
            source_columns=[date_col] + [r["measure"] for r in ranked],
        )]

    # ── Anomalies ─────────────────────────────────────────────────────────

    def _detect_anomalies(self, table: str, measure: str | None = None):
        insights = []
        date_col = self.intel.find_best_datetime(table)
        if not date_col:
            return []

        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []

        # Latest 24 months in chronological order: anomalies must be
        # judged on recent history — earliest-24 cannot see recent breaks.
        sql = f"""
            SELECT period, val FROM (
                SELECT {self._mexpr(table, date_col)} AS period,
                       SUM({self._sexpr(table, measure)}) AS val
                FROM {_safe_name(self._t(table))}
                WHERE {_safe_name(date_col)} IS NOT NULL
                GROUP BY period
                ORDER BY period DESC
                LIMIT 24
            ) ORDER BY period
        """
        _, rows = _run_sql(self.conn, sql, limit=24)
        rows = self._drop_partial_trailing_month(table, date_col, rows)
        if len(rows) < 6:
            return []

        values = [(r["period"], _safe_float(r["val"])) for r in rows if _safe_float(r["val"]) is not None]
        if len(values) < 6:
            return []

        nums = [v[1] for v in values]
        bands = _iqr_bands(nums)
        if bands is None:
            return []
        q1, q3, iqr, lower_bound, upper_bound, severe_lower, severe_upper = bands
        mean_val = statistics.mean(nums)
        std_val = statistics.stdev(nums) if len(nums) > 1 else 0
        in_fence = [v for v in nums if lower_bound <= v <= upper_bound]
        expected_val = float(statistics.median(in_fence)) if in_fence else float(statistics.median(nums))
        tdesc = self.intel.tables[table].get("description", table)

        for period, val in values:
            if val is None:
                continue
            if val < lower_bound or val > upper_bound:
                z_score = (val - mean_val) / std_val if std_val > 0 else 0
                direction = "above" if val > upper_bound else "below"
                severe = val < severe_lower or val > severe_upper
                confidence = min(95, max(40, int(min(abs(z_score) * 25, 95))))
                if severe:
                    severity = "high"
                else:
                    severity = "high" if abs(z_score) > 2.5 else ("medium" if abs(z_score) > 1.5 else "low")
                period_str = period.strftime('%b %Y') if hasattr(period, 'strftime') else str(period)
                viz_data = [{"period": str(v[0])[:7], "value": v[1], "anomaly": v[0] == period,
                             "expected": expected_val, "bandLower": lower_bound, "bandUpper": upper_bound} for v in values]
                dims = self.intel.get_text_cols(table)[:3]

                insights.append(self._build_insight(
                    insight_id=f"an_{table}_{measure}_{period_str.replace(' ', '_')}",
                    insight_type="Anomaly",
                    severity=severity,
                    title=f"{humanize(measure)} was unusually {direction} normal in {period_str}",
                    description=f"{humanize(measure)} reached {_fmtm(val, measure)} in {period_str}, which is {direction} the usual range of {_fmtm(lower_bound, measure)} to {_fmtm(upper_bound, measure)} seen in past periods.",
                    metric=f"{humanize(measure)}",
                    metric_value=val,
                    previous_value=mean_val,
                    magnitude=f"{_fmtm(val, measure)} ({direction} usual)",
                    confidence=confidence,
                    affected_period=period_str,
                    tags=[table, measure, "anomaly", direction],
                    observations=[
                        f"Observed value: {_fmtm(val, measure)} in {period_str}.",
                        f"Usual range based on past periods: {_fmtm(lower_bound, measure)} to {_fmtm(upper_bound, measure)}.",
                        f"Typical level: around {_fmtm(mean_val, measure)}.",
                    ],
                    support_metrics=[f"Outside the usual range by {_fmtm(abs(val - mean_val))}"],
                    sample_size=len(values),
                    baseline=mean_val,
                    current_value=val,
                    change_pct=((val - mean_val) / abs(mean_val) * 100) if mean_val != 0 else 0,
                    impact_value=val - mean_val,
                    impact_direction="positive" if val > mean_val else "negative",
                    impact_explanation=f"The {direction} normal {humanize(measure).lower()} of {_fmtm(val, measure)} in {period_str} may indicate {'an exceptional opportunity' if val > mean_val else 'a potential issue worth investigating'}.",
                    next_question=f"What caused {humanize(measure).lower()} to be {direction} normal in {period_str}?",
                    next_dims=dims,
                    viz_type="area",
                    viz_data=viz_data,
                    source_tables=[table],
                    source_columns=[date_col, measure],
                ))

        return insights

    # ── Opportunities ─────────────────────────────────────────────────────

    def _usable_dimension(self, table: str) -> str | None:
        """First text dimension that isn't near-unique (§13).

        Segmenting by an ID-like column (one value per row) produces
        meaningless 'opportunities'. Prefers compact breakdowns.
        """
        candidates = []
        best = self.intel.find_best_dimension(table)
        if best:
            candidates.append(best)
        for col in self.intel.get_text_cols(table):
            if col not in candidates:
                candidates.append(col)
        ranked = []
        for dim in candidates[:5]:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    f"SELECT COUNT(*), COUNT(DISTINCT {_safe_name(dim)}) FROM {_safe_name(self._t(table))}"
                )
                row = cur.fetchone()
                cur.close()
                if not row or not row[0]:
                    continue
                if float(row[1] or 0) / float(row[0]) > 0.5:
                    continue
                ranked.append((row[1], dim))
            except DB_ERRORS as exc:
                log.debug("dimension cardinality check skipped: %s", exc)
                continue
        ranked.sort(key=lambda x: x[0])
        return ranked[0][1] if ranked else None

    def _detect_opportunities(self, table: str, measure: str | None = None):
        insights = []
        dim = self._usable_dimension(table)
        if not dim:
            return []

        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []

        tdesc = self.intel.tables[table].get("description", table)

        sql = f"""
            SELECT {_safe_name(dim)} AS dim_val, SUM({self._sexpr(table, measure)}) AS val
            FROM {_safe_name(self._t(table))}
            WHERE {_safe_name(dim)} IS NOT NULL
            GROUP BY {_safe_name(dim)}
            ORDER BY val DESC
            LIMIT 20
        """
        _, rows = _run_sql(self.conn, sql, limit=20)
        if len(rows) < 3:
            return []

        values = [(r["dim_val"], _safe_float(r["val"])) for r in rows if _safe_float(r["val"]) is not None]
        if len(values) < 3:
            return []

        total = sum(v[1] for v in values)
        if total == 0:
            return []

        top = values[0]
        top_share = (top[1] / total) * 100
        mid = len(values) // 2
        top_half = values[:mid] if mid > 0 else values[:1]
        bottom_half = values[mid:] if mid < len(values) else values[-1:]

        if top_half and bottom_half:
            top_avg = statistics.mean([v[1] for v in top_half])
            bottom_avg = statistics.mean([v[1] for v in bottom_half])
            if top_avg > 0:
                gap_ratio = (top_avg - bottom_avg) / top_avg * 100
                if gap_ratio > 30:
                    confidence = min(90, max(40, int(gap_ratio * 0.8)))
                    severity = "medium" if gap_ratio > 50 else "low"
                    viz_data = [{"name": str(v[0]), "value": v[1]} for v in values[:10]]

                    insights.append(self._build_insight(
                        insight_id=f"op_{table}_{dim}",
                        insight_type="Opportunity",
                        severity=severity,
                        title=f"{humanize(tdesc or table)}: {humanize(dim)} segments with room to grow",
                        description=f"Your top {humanize(dim).lower()} ({top[0]}) brings in {_fmtm(top[1], measure)} ({top_share:.1f}% of everything). Weaker segments average {_fmtm(bottom_avg, measure)} against {_fmtm(top_avg, measure)} at the top — a {gap_ratio:.0f}% gap.",
                        metric=f"{measure} ({tdesc or table})",
                        metric_value=total,
                        magnitude=f"{gap_ratio:.0f}% gap",
                        confidence=confidence,
                        affected_period="Current period",
                        tags=[table, dim, "opportunity", "underpenetrated"],
                        observations=[
                            f"Top {dim}: {top[0]} at {_fmtm(top[1], measure)} ({top_share:.1f}% share).",
                            f"Top half average: {_fmtm(top_avg, measure)}. Bottom half: {_fmtm(bottom_avg, measure)}. Gap: {gap_ratio:.0f}%.",
                        ],
                        support_metrics=[f"Concentration: {top_share:.1f}%", f"Gap: {gap_ratio:.0f}%"],
                        sample_size=len(values),
                        impact_value=top_avg - bottom_avg,
                        impact_direction="positive",
                        impact_explanation=f"Closing the gap could unlock up to {_fmtm(top_avg - bottom_avg, measure)} in additional {measure}.",
                        next_question=f"What is behind the gap across {humanize(dim).lower()}?",
                        next_dims=[dim],
                        viz_type="bar",
                        viz_data=viz_data,
                        source_tables=[table],
                        source_columns=[dim, measure],
                    ))

        # High-growth dimension
        date_col = self.intel.find_best_datetime(table)
        if date_col and len(values) >= 3:
            # DBAPI placeholders differ by driver (sqlite uses ?, pg/mysql
            # use %s) — never hardcode one and let the other dialect fail.
            _ph = "?" if self._dialect() == "sqlite" else "%s"
            for dv in values[:3]:
                sql2 = f"""
                    SELECT {self._mexpr(table, date_col)} AS period,
                           SUM({self._sexpr(table, measure)}) AS val
                    FROM {_safe_name(self._t(table))}
                    WHERE {_safe_name(dim)} = {_ph} AND {_safe_name(date_col)} IS NOT NULL
                    GROUP BY period ORDER BY period LIMIT 12
                """
                try:
                    cur = self.conn.cursor()
                    cur.execute(sql2, (str(dv[0]),))
                    cols = [d[0] for d in cur.description]
                    period_rows = cur.fetchmany(12)
                    cur.close()
                    if len(period_rows) < 3:
                        continue
                    pvals = [(r[0], _safe_float(r[1])) for r in period_rows if _safe_float(r[1]) is not None]
                    if len(pvals) < 3:
                        continue
                    fv, lv = pvals[0][1], pvals[-1][1]
                    if fv and lv and fv > 0:
                        growth_pct = ((lv - fv) / abs(fv)) * 100
                        if growth_pct > 20:
                            confidence = min(85, max(35, int(growth_pct * 0.5)))
                            viz_data_g = [{"period": "start", "value": fv}, {"period": "end", "value": lv}]
                            insights.append(self._build_insight(
                                insight_id=f"og_{table}_{dim}_{dv[0]}",
                                insight_type="Opportunity",
                                severity="medium",
                                title=f"{humanize(dv[0])} is growing fast ({_fmt_pct(growth_pct)})",
                                description=f"{humanize(dv[0])} grew from {_fmtm(fv, measure)} to {_fmtm(lv, measure)} ({_fmt_pct(growth_pct)}). Up {abs(growth_pct):.0f}% — a run worth backing.",
                                metric=f"{measure} ({tdesc or table})",
                                metric_value=lv,
                                previous_value=fv,
                                magnitude=_fmt_pct(growth_pct),
                                confidence=confidence,
                                affected_period="Period-over-period",
                                tags=[table, dim, "opportunity", "high-growth", str(dv[0])],
                                observations=[f"{dim} '{dv[0]}' grew from {_fmtm(fv, measure)} to {_fmtm(lv, measure)} ({_fmt_pct(growth_pct)})."],
                                support_metrics=[f"Growth: {_fmt_pct(growth_pct)}"],
                                baseline=fv,
                                current_value=lv,
                                change_pct=growth_pct,
                                impact_value=lv - fv,
                                impact_direction="positive",
                                impact_explanation=f"Continued growth in '{dv[0]}' represents an expansion opportunity.",
                                next_question=f"What is driving growth in {dim} '{dv[0]}'?",
                                next_dims=[dim],
                                viz_type="bar",
                                viz_data=viz_data_g,
                                source_tables=[table],
                                source_columns=[dim, date_col, measure],
                            ))
                except Exception:
                    continue

        return insights

    # ── Risks ─────────────────────────────────────────────────────────────

    def _detect_risks(self, table: str, measure: str | None = None):
        insights = []
        date_col = self.intel.find_best_datetime(table)
        if not date_col:
            return []

        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []

        # Latest 24 months in chronological order: anomalies must be
        # judged on recent history, not the earliest window.
        sql = f"""
            SELECT period, val FROM (
                SELECT {self._mexpr(table, date_col)} AS period,
                       SUM({self._sexpr(table, measure)}) AS val
                FROM {_safe_name(self._t(table))}
                WHERE {_safe_name(date_col)} IS NOT NULL
                GROUP BY period
                ORDER BY period DESC
                LIMIT 24
            ) ORDER BY period
        """
        # NOTE (anomaly window): earliest-24 cannot see recent breaks.
        _, rows = _run_sql(self.conn, sql, limit=24)
        rows = self._drop_partial_trailing_month(table, date_col, rows)
        if len(rows) < 6:
            return []

        values = [(r["period"], _safe_float(r["val"])) for r in rows if _safe_float(r["val"]) is not None]
        if len(values) < 4:
            return []

        nums = [v[1] for v in values]
        tdesc = self.intel.tables[table].get("description", table)

        # Consecutive declines
        declining_streak = 0
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] < nums[i - 1]:
                declining_streak += 1
            else:
                break

        if declining_streak >= 3:
            total_decline = ((nums[-1] - nums[0]) / abs(nums[0])) * 100 if nums[0] != 0 else 0
            confidence = min(90, max(40, declining_streak * 20))
            severity = "high" if declining_streak >= 4 else "medium"
            period_label = self._period_label(values[-1][0], values[0][0])
            viz_data = [{"period": str(v[0])[:7], "value": v[1]} for v in values]
            dims = self.intel.get_text_cols(table)[:3]

            insights.append(self._build_insight(
                insight_id=f"rk_{table}_{measure}",
                insight_type="Risk",
                severity=severity,
                title=f"{humanize(measure)} has fallen {declining_streak} periods in a row",
                description=f"{humanize(measure)} fell {declining_streak} periods in a row — down from {_fmtm(nums[0], measure)} to {_fmtm(nums[-1], measure)} ({_fmt_pct(total_decline)}). Down {abs(total_decline):.0f}% overall.",
                metric=f"{measure} ({tdesc or table})",
                metric_value=nums[-1],
                previous_value=nums[0],
                magnitude=_fmt_pct(total_decline),
                confidence=confidence,
                affected_period=period_label,
                tags=[table, measure, "risk", "decline"],
                observations=[
                    f"{humanize(measure)} fell for {declining_streak} periods in a row.",
                    f"Down from {_fmtm(nums[0], measure)} to {_fmtm(nums[-1], measure)} ({_fmt_pct(total_decline)})." ,
                ],
                support_metrics=[f"Declining streak: {declining_streak}", f"Total decline: {_fmt_pct(total_decline)}"],
                sample_size=len(values),
                baseline=nums[0],
                current_value=nums[-1],
                change_pct=total_decline,
                impact_value=nums[-1] - nums[0],
                impact_direction="negative",
                impact_explanation=f"Continued decline could result in further losses of {_fmtm(abs(nums[-1] - nums[0]))} per period.",
                next_question=f"What is behind the continued fall in {humanize(measure).lower()}?",
                next_dims=dims,
                viz_type="area",
                viz_data=viz_data,
                source_tables=[table],
                source_columns=[date_col, measure],
            ))

        # Stagnation
        if len(nums) >= 6:
            mean_val = statistics.mean(nums)
            cv = (statistics.stdev(nums) / mean_val * 100) if mean_val != 0 else 0
            if cv < 5:
                recent_trend = nums[-1] - nums[-3] if len(nums) >= 3 else 0
                if abs(recent_trend) < mean_val * 0.02:
                    confidence = min(75, max(30, int(80 - cv * 10)))
                    period_label = self._period_label(values[-1][0], values[0][0])
                    viz_data = [{"period": str(v[0])[:7], "value": v[1]} for v in values]
                    dims = self.intel.get_text_cols(table)[:3]

                    insights.append(self._build_insight(
                        insight_id=f"rks_{table}_{measure}",
                        insight_type="Risk",
                        severity="low",
                        title=f"{humanize(measure)} is standing still for {len(values)} periods",
                        description=f"{humanize(measure)} has barely moved for {len(values)} periods, holding near {_fmtm(mean_val, measure)}.",
                        metric=f"{humanize(measure)}",
                        metric_value=nums[-1],
                        magnitude="Flat",
                        confidence=confidence,
                        affected_period=period_label,
                        tags=[table, measure, "risk", "stagnation"],
                        observations=[
                            f"{humanize(measure)} averaged {_fmtm(mean_val, measure)} over {len(values)} periods with almost no movement.",
                        ],
                        support_metrics=[f"Nearly flat for {len(values)} periods"],
                        sample_size=len(values),
                        impact_value=0,
                        impact_direction="neutral",
                        impact_explanation=f"Flat {humanize(measure).lower()} may point to a saturated market or missing growth effort.",
                        next_question=f"What could get {humanize(measure).lower()} moving again?",
                        next_dims=dims,
                        viz_type="line",
                        viz_data=viz_data,
                        source_tables=[table],
                        source_columns=[date_col, measure],
                    ))

        return insights

    # ── Correlations ──────────────────────────────────────────────────────

    def _detect_correlations(self):
        """Detect correlations between numeric measures across tables."""
        insights = []
        fact_tables = [t for t in self.intel.tables if self.intel.is_fact_table(t)]

        for i, t1 in enumerate(fact_tables):
            for t2 in fact_tables[i+1:]:
                m1 = self.intel.find_best_measure(t1)
                m2 = self.intel.find_best_measure(t2)
                dt1 = self.intel.find_best_datetime(t1)
                dt2 = self.intel.find_best_datetime(t2)
                if not m1 or not m2 or not dt1 or not dt2:
                    continue

                # Find join path
                join_path = self.intel.find_join_path(t1, t2)
                if not join_path:
                    continue

                join_clause = self.intel.build_join_clause(join_path)
                join_on = join_path[-1]  # last join step
                _dtq = f"{_safe_name(t1)}.{_safe_name(dt1)}"
                _dd = self._dialect()
                if _dd == "mysql":
                    _mline = f"DATE_FORMAT({_dtq}, '%Y-%m-01')"
                elif _dd == "sqlite":
                    _mline = f"strftime('%Y-%m-01', {_dtq})"
                else:
                    _mline = f"DATE_TRUNC('month', {_dtq})"
                sql = f"""
                    SELECT {_mline} AS period,
                           SUM({self._sexpr(t1, m1, qualified_table=t1)}) AS v1,
                           SUM({self._sexpr(t2, m2, qualified_table=t2)}) AS v2
                    FROM {_safe_name(t1)}
                    {join_clause}
                    WHERE {_safe_name(t1)}.{_safe_name(dt1)} IS NOT NULL
                    GROUP BY period
                    ORDER BY period
                    LIMIT 24
                """
                _, rows = _run_sql(self.conn, sql, limit=24)
                if len(rows) < 6:
                    continue

                v1 = [_safe_float(r["v1"]) for r in rows if _safe_float(r["v1"]) is not None and _safe_float(r["v2"]) is not None]
                v2 = [_safe_float(r["v2"]) for r in rows if _safe_float(r["v1"]) is not None and _safe_float(r["v2"]) is not None]
                periods = [str(r["period"])[:7] for r in rows if _safe_float(r["v1"]) is not None and _safe_float(r["v2"]) is not None]

                if len(v1) < 6:
                    continue

                n = len(v1)
                mean1 = statistics.mean(v1)
                mean2 = statistics.mean(v2)
                std1 = statistics.stdev(v1)
                std2 = statistics.stdev(v2)
                if std1 == 0 or std2 == 0:
                    continue

                corr = sum((a - mean1) * (b - mean2) for a, b in zip(v1, v2)) / ((n - 1) * std1 * std2)
                if abs(corr) > 0.6:
                    confidence = min(90, max(35, int(abs(corr) * 100)))
                    direction = "positive" if corr > 0 else "negative"
                    strength = "strong" if abs(corr) > 0.8 else "moderate"
                    tdesc1 = self.intel.tables[t1].get("description", t1)
                    tdesc2 = self.intel.tables[t2].get("description", t2)
                    viz_data = [{"period": periods[i], "table1": v1[i], "table2": v2[i]} for i in range(len(v1))]

                    insights.append(self._build_insight(
                        insight_id=f"corr_{t1}_{t2}",
                        insight_type="Correlation",
                        severity="medium",
                        title=(f"{humanize(tdesc1)} and {humanize(tdesc2)} move together ({strength})"
                               if direction == "positive" else
                               f"{humanize(tdesc1)} and {humanize(tdesc2)} move in opposite directions ({strength})"),
                        description=f"There is a {strength} {direction} relationship between {humanize(tdesc1).lower()} and {humanize(tdesc2).lower()} over {n} shared periods: when one rises, the other tends to {'rise' if direction == 'positive' else 'fall'}.",
                        metric=f"{humanize(tdesc1)} vs {humanize(tdesc2)}",
                        magnitude=f"{strength} {direction} link",
                        confidence=confidence,
                        affected_period=f"{periods[0]} to {periods[-1]}",
                        tags=[t1, t2, "correlation", direction],
                        observations=[
                            f"{strength.title()} {direction} relationship across {n} shared periods.",
                            "Note: moving together does not prove one causes the other.",
                        ],
                        support_metrics=[f"{strength} {direction} link over {n} periods"],
                        sample_size=n,
                        impact_value=0,
                        impact_direction="neutral",
                        impact_explanation=f"The {strength} link may reflect shared underlying drivers. This does not prove causation.",
                        next_question=f"What shared factors could explain the link between {humanize(tdesc1).lower()} and {humanize(tdesc2).lower()}?",
                        next_dims=[t1, t2],
                        viz_type="scatter",
                        viz_data=viz_data,
                        source_tables=[t1, t2],
                        source_columns=[dt1, dt2],
                    ))

        return insights

    # ── Non-temporal detectors (work without date columns) ────────────────

    def _detect_value_outliers(self, table: str, measure: str | None = None):
        """Detect row-level outliers on numeric columns using IQR."""
        insights = []
        tdesc = self.intel.tables[table].get("description", table)
        num_cols = self.intel.get_numeric_cols(table)
        if not num_cols:
            return []

        for col in num_cols[:3]:
            sql = f"SELECT {self._sexpr(table, col)} AS val FROM {_safe_name(self._t(table))} WHERE {_safe_name(col)} IS NOT NULL"
            _, rows = _run_sql(self.conn, sql, limit=_MAX_RAW_VALUES)
            vals = [_safe_float(r["val"]) for r in rows if _safe_float(r.get("val")) is not None]
            if len(vals) < 10:
                continue
            _bands = _iqr_bands(vals)
            if _bands is None:
                continue
            q1, q3, iqr, lower, upper, _sev_lo, _sev_hi = _bands
            outliers = [v for v in vals if v < lower or v > upper]
            if not outliers:
                continue
            total = len(vals)
            pct = len(outliers) / total * 100
            if pct > 30 or pct < 1:
                continue
            direction = "above" if statistics.mean(outliers) > (q1 + q3) / 2 else "below"
            high_low = "high" if direction == "above" else "low"
            # Histogram spec (controlled vocab): box plots have no renderer,
            # so the distribution is preserved as value bins, with outlier
            # bins flagged so the UI can tell normal vs unusual apart.
            # Method detail (IQR) stays in evidence only; title/summary stay
            # plain-language (normal vs unusual, no jargon).
            _lo, _hi = min(vals), max(vals)
            _span = (_hi - _lo) or 1.0
            _nbins = 12
            _counts = [0] * _nbins
            for v in vals:
                _b = min(_nbins - 1, max(0, int((v - _lo) / _span * _nbins)))
                _counts[_b] += 1
            _viz = []
            for i in range(_nbins):
                if not _counts[i]:
                    continue
                _b_lo = _lo + _span * i / _nbins
                _b_hi = _lo + _span * (i + 1) / _nbins
                _is_out = _b_hi < lower or _b_lo > upper
                _viz.append({"name": f"{_fmt(_b_lo)}-{_fmt(_b_hi)}",
                             "value": _counts[i], "outlier": bool(_is_out)})
            _sorted_out = sorted(outliers)
            _examples = [round(float(_sorted_out[0]), 2),
                         round(float(_sorted_out[len(_sorted_out) // 2]), 2),
                         round(float(_sorted_out[-1]), 2)] if _sorted_out else []
            insights.append(dict(
                id=f"outlier_{table}_{col}",
                table=table,
                type="outlier",
                severity="medium" if pct < 10 else "low",
                title=f"{humanize(col)}: {len(outliers)} records unusually {high_low} vs the normal distribution ({pct:.0f}% of rows)",
                summary=f"{len(outliers)} records were unusually {high_low} vs the normal distribution for {humanize(col).lower()} ({pct:.1f}% of {total} rows); the rest sit in the normal range {_fmt(lower)} to {_fmt(upper)}.",
                observations=[
                    f"Normal range: {_fmt(lower)} to {_fmt(upper)} (most records).",
                    f"Unusual {high_low} values range from {_fmt(min(outliers))} to {_fmt(max(outliers))}.",
                ],
                support_metrics=[f"{len(outliers)} outliers ({pct:.1f}%)"],
                sample_size=total,
                impact_value=pct,
                impact_direction="negative" if direction == "above" else "neutral",
                impact_explanation=f"{pct:.1f}% of rows have unusual {humanize(col).lower()} values.",
                next_question=f"What drives the unusual {humanize(col).lower()} values near {_fmt(max(outliers))}?",
                next_dims=[col],
                viz_type="histogram",
                viz_data=_viz,
                source_tables=[table],
                source_columns=[col],
                evidence={
                    "observations": [
                        f"Normal range: {_fmt(lower)} to {_fmt(upper)} (most records sit inside it).",
                        f"Unusual {high_low} values range from {_fmt(min(outliers))} to {_fmt(max(outliers))}.",
                    ],
                    "supportingMetrics": [f"{len(outliers)} outliers ({pct:.1f}%)"],
                    "sampleSize": total,
                    "affectedValues": {
                        "count": len(outliers),
                        "direction": high_low,
                        "normalRange": [round(float(lower), 2), round(float(upper), 2)],
                        "examples": _examples,
                    },
                },
            ))
        return insights

    def _detect_concentration(self, table: str, measure: str | None = None):
        """Detect concentration risk: top N entities dominate total."""
        insights = []
        tdesc = self.intel.tables[table].get("description", table)
        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []
        dim = self._usable_dimension(table)
        if not dim:
            return []

        sql = f"""
            SELECT {_safe_name(dim)} AS dim_val, SUM({self._sexpr(table, measure)}) AS val
            FROM {_safe_name(self._t(table))}
            WHERE {_safe_name(dim)} IS NOT NULL
            GROUP BY {_safe_name(dim)}
            ORDER BY val DESC
            LIMIT 50
        """
        _, rows = _run_sql(self.conn, sql, limit=50)
        if len(rows) < 5:
            return []

        values = [(r["dim_val"], _safe_float(r["val"])) for r in rows if _safe_float(r.get("val")) is not None]
        total = sum(v[1] for v in values)
        if total <= 0:
            return []

        # Top 20% concentration
        top_n = max(1, len(values) // 5)
        top_sum = sum(v[1] for v in values[:top_n])
        top_pct = top_sum / total * 100

        if top_pct >= 60:
            top_entities = ", ".join(str(v[0])[:30] for v in values[:3])
            _conc_viz = [{"name": str(v[0])[:30], "value": v[1]} for v in values[:10]]
            insights.append(dict(
                id=f"concentration_{table}_{dim}",
                table=table,
                type="concentration",
                severity="high" if top_pct >= 80 else "medium",
                title=f"Top {top_n} {humanize(dim).lower()} account for {top_pct:.0f}% of total {humanize(measure).lower()}",
                summary=f"The top {top_n} {humanize(dim).lower()} ({top_entities}) dominate, holding {top_pct:.0f}% of total {humanize(measure).lower()}.",
                observations=[
                    f"Top {top_n} = {_fmt(top_sum)} out of {_fmt(total)} total.",
                    f"Remaining {len(values) - top_n} segments share only {100 - top_pct:.0f}%.",
                ],
                support_metrics=[f"{top_pct:.0f}% concentration in top {top_n}"],
                sample_size=len(values),
                impact_value=top_pct,
                impact_direction="negative",
                impact_explanation=f"High concentration means {humanize(dim).lower()} diversification is limited.",
                next_question=f"What makes {top_entities} so dominant?",
                next_dims=[dim, measure],
                viz_type="pie",
                viz_data=_conc_viz,
                source_tables=[table],
                source_columns=[dim, measure],
                evidence={
                    "observations": [
                        f"Top {top_n} = {_fmt(top_sum)} out of {_fmt(total)} total.",
                        f"Remaining {len(values) - top_n} segments share only {100 - top_pct:.0f}%.",
                    ],
                    "supportingMetrics": [f"{top_pct:.0f}% concentration in top {top_n}"],
                    "sampleSize": len(values),
                },
            ))

        # Pareto check: top 20% of entities = 80%+ of value
        cumulative = 0
        pareto_n = 0
        for v in values:
            cumulative += v[1]
            pareto_n += 1
            if cumulative / total >= 0.8:
                break
        pareto_pct = pareto_n / len(values) * 100
        if pareto_pct <= 30 and len(values) >= 5:
            _pareto_viz = [{"name": str(v[0])[:30], "value": v[1]} for v in values[:10]]
            insights.append(dict(
                id=f"pareto_{table}_{dim}",
                table=table,
                type="pareto",
                severity="low",
                title=f"Pareto pattern: top {pareto_n} {humanize(dim).lower()} = 80% of {humanize(measure).lower()}",
                summary=f"Just {pareto_n} out of {len(values)} {humanize(dim).lower()} ({pareto_pct:.0f}%) generate 80% of total {humanize(measure).lower()}.",
                observations=[
                    f"Top {pareto_n} entities contribute {_fmt(cumulative)} out of {_fmt(total)}.",
                    "This 80/20 pattern is typical but worth monitoring.",
                ],
                support_metrics=[f"{pareto_n}/{len(values)} entities = 80% of total"],
                sample_size=len(values),
                impact_value=pareto_pct,
                impact_direction="neutral",
                impact_explanation="Classic Pareto distribution detected.",
                next_question=f"Which {humanize(dim).lower()} are in the top {pareto_n}?",
                next_dims=[dim],
                viz_type="bar",
                viz_data=_pareto_viz,
                source_tables=[table],
                source_columns=[dim, measure],
                evidence={
                    "observations": [
                        f"Top {pareto_n} entities contribute {_fmt(cumulative)} out of {_fmt(total)}.",
                        "This 80/20 pattern is typical but worth monitoring.",
                    ],
                    "supportingMetrics": [f"{pareto_n}/{len(values)} entities = 80% of total"],
                    "sampleSize": len(values),
                },
            ))
        return insights

    def _detect_distributions(self, table: str, measure: str | None = None):
        """Detect interesting distribution patterns in numeric columns."""
        insights = []
        tdesc = self.intel.tables[table].get("description", table)
        num_cols = self.intel.get_numeric_cols(table)
        if not num_cols:
            return []

        for col in num_cols[:3]:
            sql = f"SELECT {self._sexpr(table, col)} AS val FROM {_safe_name(self._t(table))} WHERE {_safe_name(col)} IS NOT NULL"
            _, rows = _run_sql(self.conn, sql, limit=_MAX_RAW_VALUES)
            vals = [_safe_float(r["val"]) for r in rows if _safe_float(r.get("val")) is not None]
            if len(vals) < 20:
                continue
            mean_val = statistics.mean(vals)
            stdev_val = statistics.stdev(vals) if len(vals) > 1 else 0
            if stdev_val == 0 or mean_val == 0:
                continue
            cv = stdev_val / abs(mean_val)  # coefficient of variation

            # High variability
            if cv > 1.5:
                _dist_viz = _histogram_bins(vals)
                insights.append(dict(
                    id=f"high_var_{table}_{col}",
                    table=table,
                    type="distribution",
                    severity="medium",
                    title=f"{humanize(col)} varies widely across rows",
                    summary=f"{humanize(col)} varies widely, typically around {_fmt(mean_val)} with values from {_fmt(min(vals))} to {_fmt(max(vals))}.",
                    observations=[
                        f"Range: {_fmt(min(vals))} to {_fmt(max(vals))}.",
                        f"Typical level: {_fmt(mean_val)} (spread {_fmt(stdev_val)}).",
                    ],
                    support_metrics=["Wide spread across individual rows", f"Values run from {_fmt(min(vals))} to {_fmt(max(vals))}"],
                    sample_size=len(vals),
                    impact_value=cv,
                    impact_direction="negative",
                    impact_explanation="High variability may indicate inconsistent quality or pricing.",
                    next_question=f"What drives the wide variation in {humanize(col).lower()}?",
                    next_dims=[col],
                    viz_type="histogram",
                    viz_data=_dist_viz,
                    source_tables=[table],
                    source_columns=[col],
                    evidence={
                        "observations": [
                            f"Range: {_fmt(min(vals))} to {_fmt(max(vals))}.",
                            f"Values are spread widely around the typical level." ,
                        ],
                        "supportingMetrics": ["Wide spread across individual rows"],
                        "sampleSize": len(vals),
                    },
                ))

            # Skewed distribution
            if len(vals) > 20:
                median_val = statistics.median(vals)
                skew_ratio = (mean_val - median_val) / stdev_val if stdev_val > 0 else 0
                if abs(skew_ratio) > 0.8:
                    direction = "right" if skew_ratio > 0 else "left"
                    _skew_viz = _histogram_bins(vals)
                    insights.append(dict(
                        id=f"skew_{table}_{col}",
                        table=table,
                        type="distribution",
                        severity="low",
                        title=f"Most {humanize(col).lower()} values cluster at the {'lower' if direction == 'right' else 'higher'} end",
                        summary=f"Most {humanize(col).lower()} values sit near {_fmt(median_val)}, with a typical level around {_fmt(mean_val)}.",
                        observations=[
                            f"Typical value: {_fmt(median_val)}; overall level: {_fmt(mean_val)}.",
                            f"Most values cluster around the {'lower' if direction == 'right' else 'higher'} end.",
                        ],
                        support_metrics=["Most values sit at one end of the range"],
                        sample_size=len(vals),
                        impact_value=abs(skew_ratio),
                        impact_direction="neutral",
                        impact_explanation=f"Most {humanize(col).lower()} values are {'below' if direction == 'right' else 'above'} average, so the typical value differs from the overall level.",
                        next_question=f"What is the typical {humanize(col).lower()}?",
                        next_dims=[col],
                        viz_type="histogram",
                        viz_data=_skew_viz,
                        source_tables=[table],
                        source_columns=[col],
                        evidence={
                            "observations": [
                                f"Values pile up at one end of the range." ,
                                f"Most values cluster around the {'lower' if direction == 'right' else 'higher'} end.",
                            ],
                            "supportingMetrics": ["Most values sit at one end of the range"],
                            "sampleSize": len(vals),
                        },
                    ))
        return insights

    def _detect_segment_comparison(self, table: str, measure: str | None = None):
        """Compare average metrics across categorical segments."""
        insights = []
        tdesc = self.intel.tables[table].get("description", table)
        measure = measure or self.intel.find_best_measure(table)
        if not measure:
            return []
        num_cols = self.intel.get_numeric_cols(table)
        text_cols = [c for c in self.intel.get_text_cols(table) if c != measure]

        for dim in text_cols[:2]:
            # Check cardinality
            sql_count = f"SELECT COUNT(DISTINCT {_safe_name(dim)}) AS cnt FROM {_safe_name(self._t(table))} WHERE {_safe_name(dim)} IS NOT NULL"
            _, count_rows = _run_sql(self.conn, sql_count, limit=1)
            distinct_count = _safe_float(count_rows[0].get("cnt")) if count_rows else 0
            if not distinct_count or distinct_count < 3 or distinct_count > 30:
                continue

            sql = f"""
                SELECT {_safe_name(dim)} AS seg,
                       AVG({self._sexpr(table, measure)}) AS avg_val,
                       COUNT(*) AS cnt
                FROM {_safe_name(self._t(table))}
                WHERE {_safe_name(dim)} IS NOT NULL
                GROUP BY {_safe_name(dim)}
                ORDER BY avg_val DESC
                LIMIT 20
            """
            _, rows = _run_sql(self.conn, sql, limit=20)
            if len(rows) < 3:
                continue
            segs = [(r["seg"], _safe_float(r["avg_val"]), int(r.get("cnt") or 0)) for r in rows
                    if _safe_float(r.get("avg_val")) is not None]
            if len(segs) < 3:
                continue

            overall_avg = sum(s[1] * s[2] for s in segs) / sum(s[2] for s in segs) if sum(s[2] for s in segs) > 0 else 0
            if overall_avg == 0:
                continue

            top_seg = segs[0]
            bottom_seg = segs[-1]
            ratio = top_seg[1] / bottom_seg[1] if bottom_seg[1] > 0 else 0

            if ratio >= 2.0:
                _seg_viz = [{"name": str(s[0])[:30], "value": s[1]} for s in segs[:10]]
                insights.append(dict(
                    id=f"segcomp_{table}_{dim}_{measure}",
                    table=table,
                    type="segment_comparison",
                    severity="medium",
                    title=f"{humanize(top_seg[0])} has {ratio:.1f}x the average {humanize(measure).lower()} vs {humanize(bottom_seg[0])}",
                    summary=f"{humanize(dim)} segments differ significantly in average {humanize(measure).lower()}: top is {top_seg[1]:.1f}, bottom is {bottom_seg[1]:.1f}.",
                    observations=[
                        f"Overall average: {_fmt(overall_avg)}.",
                        f"Top segment: {humanize(top_seg[0])} ({_fmt(top_seg[1])}, {top_seg[2]} rows).",
                        f"Bottom segment: {humanize(bottom_seg[0])} ({_fmt(bottom_seg[1])}, {bottom_seg[2]} rows).",
                    ],
                    support_metrics=[f"Ratio: {ratio:.1f}x", f"Segments: {len(segs)}"],
                    sample_size=sum(s[2] for s in segs),
                    impact_value=ratio,
                    impact_direction="positive",
                    impact_explanation=f"The gap between segments suggests potential for targeted optimization.",
                    next_question=f"What drives the high {humanize(measure).lower()} in {humanize(top_seg[0]).lower()}?",
                    next_dims=[dim, measure],
                    viz_type="bar",
                    viz_data=_seg_viz,
                    source_tables=[table],
                    source_columns=[dim, measure],
                    evidence={
                        "observations": [
                            f"Overall average: {_fmt(overall_avg)}.",
                            f"Top segment: {humanize(top_seg[0])} ({_fmt(top_seg[1])}, {top_seg[2]} rows).",
                            f"Bottom segment: {humanize(bottom_seg[0])} ({_fmt(bottom_seg[1])}, {bottom_seg[2]} rows).",
                        ],
                        "supportingMetrics": [f"Ratio: {ratio:.1f}x", f"Segments: {len(segs)}"],
                        "sampleSize": sum(s[2] for s in segs),
                    },
                ))
        return insights

    # ── Helpers ───────────────────────────────────────────────────────────

    def _period_label(self, later, earlier):
        l = later.strftime('%b %Y') if hasattr(later, 'strftime') else str(later)[:7]
        e = earlier.strftime('%b %Y') if hasattr(earlier, 'strftime') else str(earlier)[:7]
        return f"{e} to {l}"

    def _compute_confidence(self, change_pct, sample_size, current, previous):
        mag_score = min(40, abs(change_pct) * 2)
        sample_score = min(30, sample_size * 3)
        consistency_score = 20 if abs(change_pct) > 10 else 10
        baseline_score = 10 if previous != 0 else 0
        return min(95, max(25, int(mag_score + sample_score + consistency_score + baseline_score)))

    def _compute_severity(self, magnitude, confidence, value):
        score = (magnitude / 10) * 0.4 + (confidence / 100) * 0.6
        if score > 6:
            return "high"
        if score > 3:
            return "medium"
        return "low"

    def _build_insight(self, **kw):
        """Build a standardized insight dict.

        `visualization` is a CONTROLLED chart spec only:
        {type,x,y,series,title,interactive,data}. `type` is drawn from
        CHART_SPEC_VOCAB; no LLM-supplied frontend code is ever included.
        `source` (provenance) stays in this backend payload for admin/debug
        use — user-facing pages choose whether to display it.
        """
        now = datetime.now().isoformat()
        _viz_type = kw.get("viz_type", "bar")
        _viz_data = kw.get("viz_data", [])
        _title = kw.get("title", "")
        spec = _normalize_chart_spec(_viz_type, _viz_data, _title)
        _desc = kw.get("description", "")
        _conf = kw.get("confidence", 50)
        try:
            _conf = int(_conf)
        except (TypeError, ValueError):
            _conf = 50
        _ev_sample = kw.get("sample_size")
        _out = {
            "id": kw.get("insight_id", f"ins_{now}"),
            "type": kw.get("insight_type", "Trend"),
            "severity": kw.get("severity", "low"),
            "title": kw.get("title", ""),
            "description": _desc,
            "summary": _desc,
            "metric": kw.get("metric", ""),
            "metricValue": kw.get("metric_value"),
            "previousValue": kw.get("previous_value"),
            "magnitude": kw.get("magnitude", ""),
            "confidence": _conf,
            "detectedAt": now,
            "affectedPeriod": kw.get("affected_period", ""),
            "tags": kw.get("tags", []),
            "evidence": {
                "observations": kw.get("observations", []),
                "supportingMetrics": kw.get("support_metrics", []),
                "sampleSize": _ev_sample,
                "baseline": kw.get("baseline"),
                "currentValue": kw.get("current_value"),
                "changePercent": round(kw.get("change_pct", 0), 1) if kw.get("change_pct") else None,
            },
            "businessImpact": {
                "value": kw.get("impact_value"),
                "direction": kw.get("impact_direction", "neutral"),
                "explanation": kw.get("impact_explanation", ""),
            },
            "recommendedNextAnalysis": {
                "question": kw.get("next_question", ""),
                "metric": kw.get("metric", ""),
                "dimensions": kw.get("next_dims", []),
            },
            "visualization": {
                "type": spec["type"],
                "x": spec["x"],
                "y": spec["y"],
                "series": spec["series"],
                "title": spec["title"],
                "interactive": spec["interactive"],
                "data": spec["data"],
                "insufficient": spec.get("insufficient", False),
                "unsupported": spec.get("unsupported", False),
            },
            "source": {
                "tables": kw.get("source_tables", []),
                "columns": kw.get("source_columns", []),
            },
        }
        return _enrich_insight(_out)

    def _deduplicate(self, insights):
        seen = set()
        unique = []
        for ins in insights:
            if not isinstance(ins, dict):
                continue
            _t = ins.get("type", "")
            _m = ins.get("metric", "")
            _a = ins.get("affectedPeriod", ins.get("affected_period", ""))
            key = (_t, _m, _a)
            if key not in seen:
                seen.add(key)
                unique.append(ins)
        return unique

    def _rank(self, insights):
        """Multi-signal ranking: severity + confidence + magnitude + impact +
        persistence + quality. Every signal is a measured field on the insight;
        missing signals contribute 0 rather than blocking the insight."""
        def score(ins):
            sev_map = {"high": 3, "medium": 2, "low": 1}
            sev = sev_map.get(str(ins.get("severity", "low")).lower(), 1)
            conf = (ins.get("confidence") or 0) / 100
            ev = ins.get("evidence") or {}
            bi = ins.get("businessImpact") or {}
            try:
                change = abs(float(ev.get("changePercent") or 0))
            except (TypeError, ValueError):
                change = 0
            try:
                impact = abs(float(bi.get("value") or 0))
            except (TypeError, ValueError):
                impact = 0
            try:
                n = int(ev.get("sampleSize") or 0)
            except (TypeError, ValueError):
                n = 0
            persistence = min(1.0, n / 12)
            quality = min(1.0, (len(ev.get("observations") or [])
                                + len(ev.get("supportingMetrics") or [])) / 6)
            magnitude = min(1.0, change / 50) + min(1.0, math.log1p(impact) / 10)
            return sev * (0.5 + conf) * (1 + magnitude) * (1 + 0.5 * persistence + 0.5 * quality)
        return sorted(insights, key=score, reverse=True)


# ── API Endpoints ─────────────────────────────────────────────────────────

class SaveInsightRequest(BaseModel):
    insight_id: str
    title: str
    insight_type: str
    severity: str
    metric: str = ""
    insight_payload: dict = {}


class AskInsightRequest(BaseModel):
    question: str = ""
    insight_id: str = ""
    # Cheap path: frontend already holds the full insight card, so it may
    # POST it here to skip regenerating all insights server-side. Optional
    # for backward compatibility (id-only calls still regenerate + match).
    insight: Optional[dict] = None


@router.get("/api/insights")
def get_insights(request: Request, user: dict = Depends(require_writable)):
    """Generate real insights from the connected database."""
    import json as _json
    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path or not schema_path.exists():
        raise HTTPException(status_code=400, detail="No database connected. Please connect first.")
    mapping = _json.loads(schema_path.read_text(encoding="utf-8"))
    tables = mapping.get("tables", {})
    if not tables:
        raise HTTPException(status_code=400, detail="No tables found. Please extract schema first.")

    from core.secret_box import decrypt as _decrypt_secret
    db_config = dict(session.get("db") or {})
    db_type = session.get("db_type", "postgresql")
    if not db_config:
        raise HTTPException(status_code=400, detail="No database configuration found.")

    conn = None
    sqlite_compat = False
    if session.get("source_type") != "relational" or db_type in ("sqlite", "document"):
        # CSV-backed SQLite sources: run the same detectors against the .db file.
        db_uri = session.get("db_uri") or ""
        if not db_uri.startswith("sqlite:///"):
            raise HTTPException(status_code=400, detail="Insights require a connected database.")
        import sqlite3 as _sqlite3

        try:
            conn = _SQLiteCompatConnection(_sqlite3.connect(db_uri[len("sqlite:///"):]))
            sqlite_compat = True
        except (sqlite3.Error, OSError, ValueError) as exc:
            log.exception("insights sqlite open failed request_id=%s: %s",
                          get_request_id(request), exc)
            raise HTTPException(status_code=500, detail="Could not open the connected database file.")
    if conn is None and not db_config:
        raise HTTPException(status_code=400, detail="Insights require a connected relational database.")
    if conn is None:
        if db_config.get("password"):
            try:
                db_config["password"] = _decrypt_secret(db_config["password"])
            except (ValueError, KeyError, TypeError) as exc:
                log.debug("insights secret decrypt skipped: %s", exc)

        if db_type == "mysql":
            try:
                import pymysql
                conn = pymysql.connect(
                    host=db_config.get("host", "localhost"),
                    port=int(db_config.get("port", 3306)),
                    user=db_config.get("user"),
                    password=db_config.get("password"),
                    database=db_config.get("db"),
                    charset="utf8mb4",
                    connect_timeout=10,
                )
            except (pymysql.MySQLError, OSError, ValueError, TypeError) as exc:
                log.exception("insights mysql connect failed request_id=%s: %s",
                              get_request_id(request), exc)
                raise HTTPException(status_code=500, detail="Could not connect to database.")
        else:
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=db_config.get("host", "localhost"),
                    port=int(db_config.get("port", 5432)),
                    dbname=db_config.get("db"),
                    user=db_config.get("user"),
                    password=db_config.get("password"),
                    options=f"-c search_path={db_config.get('db_schema', 'public')}",
                    connect_timeout=10,
                )
            except (psycopg2.Error, OSError, ValueError, TypeError) as exc:
                log.exception("insights postgres connect failed request_id=%s: %s",
                              get_request_id(request), exc)
                raise HTTPException(status_code=500, detail="Could not connect to database.")

    try:
        _t0 = time.time()
        log.info("insights start user=%s", user.get("user_id"))
        # Centralized evidence gate BEFORE any detector/LLM work: consume the
        # row counts the mapping (csv_handler/csv_orchestrator contract)
        # exposes; fall back to a live COUNT(*) only when the mapping has no
        # counts. rows < 10 -> honest insufficient state, never LLM content.
        _row_total, _row_known = 0, False
        for _t, _ti in (mapping.get("tables") or {}).items():
            _rc = (_ti or {}).get("row_count")
            if isinstance(_rc, int):
                _row_known = True
                _row_total += max(0, _rc)
        try:
            from core.evidence import assess_evidence as _assess2
            from core.evidence import dataset_facts as _facts2
            if _row_known:
                _ev = _assess2(_row_total, "insights",
                               dataset_name=session.get("db_name"))
            else:
                _ev = None
        except (ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
            log.debug("insights evidence pre-gate skipped: %s", exc)
            _ev = None
        if _ev is not None and _ev.get("status") != "supported":
            try:
                _cols = []
                for _ti in (mapping.get("tables") or {}).values():
                    _cc = (_ti or {}).get("columns") or []
                    if isinstance(_cc, dict):
                        _cols.extend(list(_cc.keys())[:20])
                    else:
                        for _c in _cc[:20]:
                            _cols.append((_c.get("column") if isinstance(_c, dict) else str(_c)) or "")
                    if len(_cols) >= 12:
                        break
                _facts = _facts2(session.get("db_name"), _row_total, _cols, [], _ev.get("needed"))
            except Exception:
                _facts = {"dataset": session.get("db_name"),
                          "usable_records": _row_total}
            log.info("insights insufficient user=%s rows=%s", user.get("user_id"), _row_total)
            return {"ok": True, "insights": [], "count": 0,
                    "evidence_state": _ev,
                    "message": _ev.get("reason"),
                    "dataset_facts": _facts}
        engine = InsightsEngine(conn, mapping)
        insights = engine.generate_all_insights(max_insights=20)
        # Legitimate empty case: no patterns detected -> honest ok:true with
        # empty list (frontend shows the no-patterns state, not the error one).
        log.info("insights done user=%s count=%d elapsed=%.2fs",
                 user.get("user_id"), len(insights), time.time() - _t0)
        _payload = {"ok": True, "insights": insights, "count": len(insights)}
        if getattr(engine, "sample_info", {}).get("sampled"):
            _payload["sampling"] = engine.sample_info
        # Additive supported state so the frontend can render evidence
        # uniformly (Insights.tsx empty branch already handles []).
        if _ev is not None:
            _payload["evidence_state"] = _ev
        else:
            try:
                from core.evidence import assess_evidence as _assess3
                _payload["evidence_state"] = _assess3(
                    _row_total if _row_known else max(0, len(insights)),
                    "insights", dataset_name=session.get("db_name"))
            except Exception:
                pass
        return _payload
    except Exception as e:
        # Honest failure shape: ok:false + sanitized detail (no secrets/paths).
        # Full traceback stays server-side. Frontend shows its error branch
        # (Insights.tsx error state reads `detail`) instead of an empty page.
        log.exception("insights failed user=%s", user.get("user_id"))
        return JSONResponse(status_code=500, content={
            "ok": False, "detail": sanitize_error(e),
            "insights": [], "count": 0,
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/api/insights/saved")
def get_saved_insights(request: Request, user: dict = Depends(require_writable)):
    """Return saved/bookmarked insights for this user."""
    session = get_session(user["user_id"])
    saved = session.get("saved_insights", [])
    return {"ok": True, "saved": saved}


@router.post("/api/insights/save")
def save_insight(req: SaveInsightRequest, request: Request, user: dict = Depends(require_writable)):
    """Bookmark/save an insight."""
    session = get_session(user["user_id"])
    saved = session.get("saved_insights", [])
    if any(s.get("insight_id") == req.insight_id for s in saved):
        return {"ok": True, "saved": saved}
    saved.append({
        "insight_id": req.insight_id,
        "title": req.title,
        "type": req.insight_type,
        "severity": req.severity,
        "metric": req.metric,
        "created_at": datetime.now().isoformat(),
        "insight_payload": req.insight_payload,
    })
    session["saved_insights"] = saved
    save_session(user["user_id"], session)
    return {"ok": True, "saved": saved}


@router.delete("/api/insights/save/{insight_id}")
def unsave_insight(insight_id: str, request: Request, user: dict = Depends(require_writable)):
    """Remove bookmark from an insight."""
    session = get_session(user["user_id"])
    saved = session.get("saved_insights", [])
    saved = [s for s in saved if s.get("insight_id") != insight_id]
    session["saved_insights"] = saved
    save_session(user["user_id"], session)
    return {"ok": True, "saved": saved}


@router.post("/api/insights/{insight_id}/ask")
def ask_about_insight(insight_id: str, req: AskInsightRequest, request: Request, user: dict = Depends(require_writable)):
    """Analyze a specific insight further using the GoalAgent."""
    import json as _json
    from pathlib import Path
    from core.config import get_session
    from core.secret_box import decrypt as _decrypt_secret

    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path or not Path(str(schema_path)).exists():
        raise HTTPException(status_code=400, detail="No database connected.")

    mapping = _json.loads(Path(str(schema_path)).read_text(encoding="utf-8"))
    db_config = dict(session.get("db") or {})
    if not db_config:
        raise HTTPException(status_code=400, detail="No database configuration found.")
    if db_config.get("password"):
        try:
            db_config["password"] = _decrypt_secret(db_config["password"])
        except Exception:
            pass

    db_type = session.get("db_type", "postgresql")
    dialect = "postgresql" if "postgres" in db_type else "mysql" if "mysql" in db_type else "sqlite"
    db_uri = session.get("db_uri", "")
    provider = session.get("provider")

    # Cheap path: use the caller-supplied insight payload when present so we
    # avoid regenerating all insights. Falls back to id-only lookup below.
    target = None
    if isinstance(getattr(req, "insight", None), dict) and req.insight.get("id", insight_id) == insight_id:
        target = req.insight
    elif isinstance(getattr(req, "insight", None), dict) and not insight_id:
        target = req.insight

    if target is None:
        # Generate insights to find the specific one (legacy id-only path)
        from sqlalchemy import create_engine
        _t0 = time.time()
        log.info("insights ask start user=%s id=%s", user.get("user_id"), insight_id)
        try:
            try:
                from core.db import search_path_connect_args as _spa_ins
                _sch_ins = [str(s).strip() for s in
                            (session.get("schemas") or []) if str(s or "").strip()]
                _ca_ins = _spa_ins(session.get("db_type", "postgresql"), _sch_ins)
            except Exception:
                _ca_ins = {}
            engine = create_engine(db_uri, pool_pre_ping=True,
                                   **({"connect_args": _ca_ins} if _ca_ins else {}))
            with engine.connect() as conn:
                ins_engine = InsightsEngine(conn, mapping, llm=None)
                all_insights = ins_engine.generate_all_insights(max_insights=20)
        except Exception as e:
            log.exception("insights ask regenerate failed user=%s", user.get("user_id"))
            raise HTTPException(status_code=500, detail=f"Failed to generate insights: {sanitize_error(e)}")

        # Find the target insight
        for ins in all_insights:
            if ins.get("id") == insight_id:
                target = ins
                break
        log.info("insights ask lookup done user=%s elapsed=%.2fs", user.get("user_id"), time.time() - _t0)

    if not target:
        raise HTTPException(status_code=404, detail="Insight not found.")

    # Build a contextual question from the insight
    question = req.question or ""
    if not question:
        title = target.get("title", "")
        metric = target.get("metric", "")
        insight_type = target.get("type", "")
        description = target.get("description", "")
        question = f"Provide a detailed analysis of: {title}. {description}"

    # Run through GoalAgent
    try:
        from goal_agent.agent import GoalAgent
        from goal_agent.schema import load_user_overrides
        import time

        output_dir = Path("artifacts/processed_data") / f"user_{user['user_id']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"insight_analysis_{int(time.time())}.json"

        from core.config import resolve_provider_key
        api_key = resolve_provider_key(provider, session.get("api_keys", {}))
        agent = GoalAgent(
            schema_json_path=str(schema_path),
            db_uri=db_uri,
            provider=provider,
            api_key=api_key,
            dialect=dialect,
            overrides=load_user_overrides(user["user_id"], session.get("db_name", "")),
            schemas=session.get("schemas"),
        )
        result = agent.process_goal(question, output_path=str(output_path))
    except Exception as e:
        log.exception("insights ask analysis failed user=%s", user.get("user_id"))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {sanitize_error(e)}")

    return {
        "ok": True,
        "insight_id": insight_id,
        "question": question,
        "result": result,
    }


@router.get("/api/insights/{insight_id}")
def get_insight_detail(insight_id: str, request: Request, user: dict = Depends(require_writable)):
    """Return one normalized insight + related ids (reuse generation, no duplicate).

    Regenerates the list via the existing InsightsEngine path, id-matches,
    and returns {ok, insight (full normalized + takeaways + flags),
    related_ids}. Honest 404-style {ok:False, code NOT_FOUND} when missing.
    """
    import json as _json
    session = get_session(user["user_id"])
    reject_document_sql(session)
    schema_path = session.get("schema_path")
    if not schema_path:
        raise HTTPException(status_code=400, detail="No database connected. Please connect first.")
    from pathlib import Path as _Path
    if not _Path(str(schema_path)).exists():
        raise HTTPException(status_code=400, detail="No database connected. Please connect first.")
    mapping = _json.loads(_Path(str(schema_path)).read_text(encoding="utf-8"))
    if not (mapping.get("tables") or {}):
        raise HTTPException(status_code=400, detail="No tables found. Please extract schema first.")

    from core.secret_box import decrypt as _decrypt_secret
    db_config = dict(session.get("db") or {})
    db_type = session.get("db_type", "postgresql")
    conn = None
    if session.get("source_type") != "relational" or db_type in ("sqlite", "document"):
        db_uri = session.get("db_uri") or ""
        if not db_uri.startswith("sqlite:///"):
            raise HTTPException(status_code=400, detail="Insights require a connected database.")
        import sqlite3 as _sqlite3
        try:
            conn = _SQLiteCompatConnection(_sqlite3.connect(db_uri[len("sqlite:///"):]))
        except Exception:
            raise HTTPException(status_code=500, detail="Could not open the connected database file.")
    if conn is None:
        if not db_config:
            raise HTTPException(status_code=400, detail="No database configuration found.")
        if db_config.get("password"):
            try:
                db_config["password"] = _decrypt_secret(db_config["password"])
            except Exception:
                pass
        if db_type == "mysql":
            try:
                import pymysql
                conn = pymysql.connect(
                    host=db_config.get("host", "localhost"),
                    port=int(db_config.get("port", 3306)),
                    user=db_config.get("user"),
                    password=db_config.get("password"),
                    database=db_config.get("db"),
                    charset="utf8mb4",
                    connect_timeout=10,
                )
            except (pymysql.MySQLError, OSError, ValueError, TypeError) as exc:
                log.exception("insights mysql connect failed request_id=%s: %s",
                              get_request_id(request), exc)
                raise HTTPException(status_code=500, detail="Could not connect to database.")
        else:
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=db_config.get("host", "localhost"),
                    port=int(db_config.get("port", 5432)),
                    dbname=db_config.get("db"),
                    user=db_config.get("user"),
                    password=db_config.get("password"),
                    options=f"-c search_path={db_config.get('db_schema', 'public')}",
                    connect_timeout=10,
                )
            except (psycopg2.Error, OSError, ValueError, TypeError) as exc:
                log.exception("insights postgres connect failed request_id=%s: %s",
                              get_request_id(request), exc)
                raise HTTPException(status_code=500, detail="Could not connect to database.")

    try:
        engine = InsightsEngine(conn, mapping)
        # Same path as the list endpoint — no duplicated detector logic.
        insights = engine.generate_all_insights(max_insights=20)
    except Exception as e:
        log.exception("insights detail failed user=%s", user.get("user_id"))
        return JSONResponse(status_code=500, content={
            "ok": False, "detail": sanitize_error(e),
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass

    target = None
    for _ins in insights:
        if isinstance(_ins, dict) and _ins.get("id") == insight_id:
            target = _ins
            break
    if target is None:
        return JSONResponse(status_code=404, content={
            "ok": False, "code": "NOT_FOUND",
            "detail": f"Insight '{insight_id}' not found.",
        })
    # related_ids: generic — same type or overlapping source tables, max 5.
    try:
        _t_type = str(target.get("type", "")).lower()
        _t_tables = set(((target.get("source") or {}) if isinstance(target.get("source"), dict) else {}).get("tables", []) or [])
        _rel = []
        for _ins in insights:
            if not isinstance(_ins, dict) or _ins.get("id") == insight_id:
                continue
            _o_tables = set(((_ins.get("source") or {}) if isinstance(_ins.get("source"), dict) else {}).get("tables", []) or [])
            if (str(_ins.get("type", "")).lower() == _t_type) or (_t_tables & _o_tables):
                _rel.append(_ins.get("id"))
            if len(_rel) >= 5:
                break
    except Exception:
        _rel = []
    return {"ok": True, "insight": target, "related_ids": _rel}
