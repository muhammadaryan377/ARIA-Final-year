"""Component: data-layer execution with retries and null-preserving data cleaning."""

import difflib
import json
import os
import re
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from llm_provider import LLMProvider, create_provider
from core.validation import unknown_sql_tables, empty_sql_tables, unknown_sql_columns
from core.config import SCHEMA_DIR, BASE_DIR
from schema_engine.lexical import singularize

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None

logger = logging.getLogger(__name__)


def _plain_words(identifier):
    """Identifier -> plain words for user-facing text ('order_date' -> 'order date')."""
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(identifier or "")) if p]
    return " ".join(w.lower() for w in parts) or str(identifier or "")


def _natural_unmatched(kind, names, available):
    """Natural-English failure text naming what was asked for in plain words
    plus the available datasets/fields — never raw SQL or identifiers."""
    named = ", ".join(f"'{_plain_words(n)}'" for n in list(names)[:3]) or "that"
    avail = ", ".join(f"'{_plain_words(a)}'" for a in list(available)[:10]) or "your connected data"
    if kind == "table":
        return (f"I couldn't match {named} to your data. "
                f"Available datasets: {avail}. Try asking with one of these.")
    return (f"I couldn't match {named} to your data. "
            f"Available fields: {avail}. Try rephrasing with one of these fields.")


_UNDEFINED_COLUMN_RES = (
    re.compile(r'no such column:?\s*["`\'\[]?([\w.]+)', re.IGNORECASE),
    re.compile(r'column\s+["`\'\[]?([\w.]+)["`\'\]]?\s+does not exist', re.IGNORECASE),
    re.compile(r"unknown column\s+['\"`]?([\w.]+)", re.IGNORECASE),
    re.compile(r"invalid column name\s+['\"`]?([\w.]+)", re.IGNORECASE),
)


def _extract_undefined_column(exc):
    """Pull the offending column name out of a DB undefined-column error."""
    try:
        text = str(exc or "")
    except Exception:
        return None
    for rx in _UNDEFINED_COLUMN_RES:
        m = rx.search(text)
        if m:
            return m.group(1).strip().strip("\"'`[]")
    if re.search(r"undefinedcolumn|undefined_column", text, re.IGNORECASE):
        m = re.search(r'["`\'\[]?([\w.]+)["`\'\]]?', text)
        if m:
            return m.group(1)
    return None


class DataMixin:
    # Bounded budget for each LLM repair call. Without this a slow/unavailable
    # local model (Ollama) call (capped at 120s in llm_provider) inside a
    # repair loop can block a request for many minutes. Repairs that exceed
    # this are abandoned and the caller degrades gracefully instead of hanging.
    _REPAIR_TIMEOUT = int(os.getenv("ARIA_REPAIR_TIMEOUT", "60"))

    # ---- _no_llm_rebuild (_no_llm_rebuild) ----

    def _no_llm_rebuild(self, user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql):
        """Deterministic rebuild when an LLM repair call fails (LLM disabled).

        Returns cleaned IR-driven SQL to continue the loop with, or None when
        no distinct rebuild exists (caller raises the original error).
        """
        try:
            det = self._deterministic_sql(  # type: ignore[attr-defined]
                user_goal, join_path, kpi_map, ir, join_plan)
        except Exception:
            return None
        if det and det.strip() != (current_sql or "").strip():
            try:
                return self._clean_sql(det)  # type: ignore[attr-defined]
            except Exception:
                return None
        return None

    # ---- _execute_with_retry (_execute_with_retry) ----

    def _execute_with_retry(self, user_goal, join_path, kpi_map, raw_sql, retries=6,
                            ir=None, join_plan=None):
        """Execute the SQL with a validation layer + LLM repair.

        Bounded retry policy: at most `retries` repair attempts (never raised
        to force queries to pass). Tracks generation time, validation
        failures, repair count, and outcome in logs. Final failure returns a
        structured "ARIA could not safely construct a query" note.

        Returns (records, final_sql, note). note is None on success, or a dict
        {"message": str} explaining a graceful empty outcome so the UI never
        shows a bare "no data" for a valid-but-empty query.
        """
        # Hard cap: callers may pass a lower bound, never a higher one.
        retries = min(int(retries if retries is not None else 6), 6)
        _t_start = time.monotonic()
        _repair_count = 0
        _validation_failures: list = []
        _SAFE_FAIL = "ARIA could not safely construct a query for this request."
        if raw_sql is None:
            # raw_sql was None -> the model could not produce a query (provider
            # outage, rate limit, or a controlled non-SELECT* failure for an
            # aggregating goal); degrade gracefully instead of crash-looping.
            logger.warning(
                "SQL retry: generation_time_ms=%d validation_failures=[] "
                "repair_count=0 outcome=generation_failed goal=%.120s",
                int((time.monotonic() - _t_start) * 1000), (user_goal or "")[:120])
            return [], None, {"failed": True, "message": _SAFE_FAIL}
        current_sql = self._clean_sql(raw_sql)
        # Fail-closed allowlist gate: read-only + unknown tables/columns must
        # pass before EVERY execution (enforced at the execution point below,
        # after repairs). Resolves via SqlMixin when composed in GoalAgent,
        # else falls back to the shared core.validation guard directly.
        try:
            _enforce = self._enforce_execution_guards  # type: ignore[attr-defined]
        except AttributeError:
            from core.validation import enforce_readonly_guards as _core_enforce
            _tables = getattr(self, "tables", {}) or {}
            _enforce = lambda s: _core_enforce(  # noqa: E731
                s, {n: list((i.get("columns", {}) or {})) for n, i in _tables.items()})
        # Entry check is read-only ONLY: unknown tables/columns are repaired
        # by the loop below, so the full allowlist gate runs at execution.
        if not self._is_read_only_sql(current_sql):
            raise RuntimeError(
                "The generated SQL is not read-only. Only read-only SELECT "
                "queries are allowed (no INSERT/UPDATE/DELETE or DDL)."
            )
        columns_by_table = {
            name: list(info["columns"])
            for name, info in self.tables.items()
        }
        for attempt in range(retries + 1):
            # Validation layer: reject hallucinated table names before hitting
            # the database (e.g. `order_items` when the real table is
            # `order_details`). Provider-agnostic.
            bad_tables = unknown_sql_tables(current_sql, self.tables)
            if bad_tables:
                if attempt >= retries:
                    raise RuntimeError(_natural_unmatched(
                        "table", bad_tables, list(self.tables.keys())))
                logger.warning(
                    f"SQL references unknown table(s) {bad_tables} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below references table(s) that DO NOT EXIST in the schema: {', '.join(bad_tables)}.
You MUST only use the exact table names listed in the schema below.

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query using real tables from the schema.
No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1,
                                            num_predict=500, timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM table repair unavailable; continuing with "
                            "deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(_natural_unmatched(
                        "table", bad_tables, list(self.tables.keys())))

            # Validation layer: reject hallucinated COLUMN references before a
            # database round-trip (e.g. `orders.quantity` when quantity lives
            # on `order_details`). Deterministic + provider-agnostic.
            bad_columns = unknown_sql_columns(current_sql, columns_by_table)
            if bad_columns:
                repaired, changed = self._repair_unknown_columns(
                    current_sql, columns_by_table, bad_columns
                )
                if changed:
                    logger.warning(
                        f"SQL referenced unknown column(s) {bad_columns}; "
                        f"repaired deterministically (no LLM)."
                    )
                    current_sql = repaired
                    continue
                # Near-miss names (typos/plurals/case) fixed without an LLM.
                try:
                    repaired, changed = self._repair_fuzzy_columns(
                        current_sql, columns_by_table, bad_columns
                    )
                except Exception:
                    repaired, changed = current_sql, False
                if changed:
                    logger.warning(
                        f"SQL referenced unknown column(s) {bad_columns}; "
                        f"repaired by fuzzy match (no LLM)."
                    )
                    current_sql = repaired
                    continue
                if attempt >= retries:
                    raise RuntimeError(_natural_unmatched(
                        "column", bad_columns,
                        sorted({c for cols in columns_by_table.values() for c in cols})))
                logger.warning(
                    f"SQL references unknown column(s) {bad_columns} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below references column(s) that DO NOT EXIST in the schema: {', '.join(bad_columns)}.
Use only the real columns listed under each table in the schema below.

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query using real columns from the schema.
No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1,
                                            num_predict=500, timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM column repair unavailable; continuing with "
                            "deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(_natural_unmatched(
                        "column", bad_columns,
                        sorted({c for cols in columns_by_table.values() for c in cols})))

            # Validation layer: reject SQL that references a table (as `tbl.col`
            # or in a JOIN ON) without that table in FROM/JOIN. This is a plain,
            # database-independent grammar bug ("missing FROM-clause entry"). When
            # the schema has a clear FK edge we inject the correct JOIN ourselves
            # (deterministic, no LLM); otherwise ask the LLM to repair.
            missing = self._find_missing_from_tables(current_sql)
            if missing:
                auto_sql, injected = self._inject_missing_joins(current_sql, missing)
                if injected:
                    logger.warning(
                        f"SQL referenced {missing} without a FROM clause; injected "
                        f"{injected} deterministic FK JOIN(s)."
                    )
                    current_sql = auto_sql
                    continue
                if attempt >= retries:
                    raise RuntimeError(
                        "I couldn't build that breakdown with the current data. "
                        "Try a simpler question about totals, averages or top items."
                    )
                logger.warning(
                    f"SQL references {missing} without a FROM clause on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below references the table(s) {missing} (e.g. `{missing[0]}.column`) but
never includes them in the FROM or JOIN clauses, which crashes with
"missing FROM-clause entry". Add each missing table to the FROM/JOIN using the
correct FOREIGN KEY join from the schema (child.fk_id = parent.id).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1,
                                            num_predict=500, timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM missing-table repair unavailable; continuing "
                            "with deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(
                        "I couldn't build that breakdown with the current data. "
                        "Try a simpler question about totals, averages or top items."
                    )

            # Validation layer: reject nested aggregate functions before a
            # database round-trip. AVG(SUM(x)) etc. are invalid in PostgreSQL
            # and this is a frequent LLM mistake that would otherwise waste a
            # full execution + repair cycle.
            nested = self._find_nested_aggregates(current_sql)
            if nested:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL nested aggregate function(s) "
                        f"({', '.join(nested)}) and could not be repaired."
                    )
                logger.warning(
                    f"SQL nests aggregate function(s) {nested} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below nests aggregate functions like {', '.join(nested)} (e.g. AVG(SUM(x))),
which is invalid SQL in PostgreSQL.
To compute an average per group use SUM(x) / COUNT(DISTINCT key), or move the outer
aggregate into a subquery (SELECT AVG(v) FROM (SELECT SUM(x) AS v FROM ... GROUP BY ...) t).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1,
                                            num_predict=500, timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM nested-aggregate repair unavailable; continuing "
                            "with deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(
                        f"Generated SQL nested aggregate function(s) "
                        f"({', '.join(nested)}) and could not be repaired."
                    )

            # Validation layer: reject SUM/AVG over incompatible datatypes
            # (e.g. SUM(name) on a TEXT column). Deterministic; repair via LLM.
            bad_types = self._find_incompatible_aggregations(current_sql)
            if bad_types:
                # Deterministic (no-LLM) repair first: swap the offending
                # column for the table's verified-safe numeric measure.
                try:
                    _fixed, _ok = self._repair_incompatible_aggregation(
                        current_sql)
                except Exception:
                    _fixed, _ok = current_sql, False
                if _ok:
                    logger.warning(
                        "SQL aggregates incompatible datatype(s) %s; repaired "
                        "deterministically (no LLM).", bad_types)
                    current_sql = _fixed
                    continue
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL aggregates incompatible datatype(s) "
                        f"({', '.join(bad_types)}) and could not be repaired."
                    )
                logger.warning(
                    f"SQL aggregates incompatible datatype(s) {bad_types} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below applies an aggregate to an incompatible datatype: {', '.join(bad_types)}.
SUM/AVG require a numeric column; use COUNT(*) for row counts or pick the real
numeric measure from the schema below.

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1,
                                            num_predict=500, timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM datatype repair unavailable; continuing with "
                            "deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(
                        f"Generated SQL aggregates incompatible datatype(s) "
                        f"({', '.join(bad_types)}) and could not be repaired."
                    )

            # Validation layer: reject joins that equate the PRIMARY KEYS of two
            # different tables, or that join an FK column to a table it does not
            # actually reference. In a normalized schema both are wrong
            # correlations (e.g. tracks.id = artists.id, tracks.album_id = artists.id);
            # real joins go PK -> FK through the declared relationships.
            bad_pk_joins = self._find_suspicious_pk_joins(current_sql)
            bad_fk_joins = self._find_fk_mismatch_joins(current_sql)
            bad_joins = bad_pk_joins + bad_fk_joins
            if bad_joins:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL joins unrelated tables "
                        f"({', '.join(bad_joins)}) and could not be repaired."
                    )
                logger.warning(
                    f"SQL joins unrelated tables {bad_joins} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below contains broken JOIN conditions: {', '.join(bad_joins)}.
Two different tables having an equal column name does NOT make them related,
and an FK column must be joined to the exact table it references.
Join ONLY through the actual FOREIGN KEY relationships in the schema (e.g. child.fk_id = parent.id).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query using real FK relationships. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500,
                                            timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM join repair unavailable; continuing with "
                            "deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(
                        f"Generated SQL joins unrelated tables "
                        f"({', '.join(bad_joins)}) and could not be repaired."
                    )

            # Validation layer: reject a bare `SELECT *` when the goal asks for
            # an aggregation. `SELECT * FROM plays LIMIT 50` is valid SQL but does
            # not answer "total play count per genre" â€” better to repair than to
            # chart the raw ID columns. Deterministic + provider-agnostic.
            if self._goal_asks_for_aggregation(user_goal) and self._find_bare_select_star(current_sql):
                if attempt >= retries:
                    raise RuntimeError(
                        "Generated SQL is a plain `SELECT *` that does not aggregate the "
                        "data the goal asks for, and could not be repaired."
                    )
                logger.warning(
                    f"SQL is a bare SELECT * for an aggregating goal on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below is a plain `SELECT *` (no aggregate functions, no GROUP BY), but the
user's goal clearly asks for an aggregation or breakdown (total, count, average,
per group, etc.). Rewrite it as an aggregated query that answers the goal using the
real columns in the schema (e.g. SELECT t.col, COUNT(...) ... GROUP BY t.col).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500,
                                            timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM aggregation repair unavailable; continuing with "
                            "deterministic rebuild (no LLM).")
                        current_sql = _rb
                        continue
                    raise RuntimeError(
                        "Generated SQL is a plain `SELECT *` that does not aggregate the "
                        "data the goal asks for, and could not be repaired."
                    )

            # Semantic guard: a compound-extremes goal ("highest and lowest",
            # "most and least", ...) must NOT be answered with a single LIMIT 1
            # row. Strip a single-sided trailing LIMIT so the full ranked set is
            # returned (both extremes present). Deterministic + provider-agnostic.
            current_sql, _changed = self._fix_extremes_sql(user_goal, current_sql)

            # Semantic validation (spec §9/§10): the SQL must implement the
            # resolved Goal IR and the explicit join plan. Runs before every
            # execution, so a repaired query is always re-validated first.
            self._last_semantic_warnings = []
            if ir:
                sem_issues, sem_warnings = self._semantic_validate_sql(
                    ir, current_sql, join_path, join_plan
                )
                self._last_semantic_warnings = list(sem_warnings)
                if sem_issues:
                    if attempt >= retries:
                        raise RuntimeError(
                            "Generated SQL violates the semantic goal plan: "
                            + "; ".join(sem_issues)
                            + " and could not be repaired."
                        )
                    logger.warning(
                        f"SQL violates the semantic plan on attempt {attempt + 1}: "
                        f"{sem_issues}. Asking LLM to repair."
                    )
                    plan_json = json.dumps({
                        k: ir.get(k)
                        for k in ("aggregation", "metrics", "dimensions",
                                  "filters", "time", "ranking")
                    }, default=str)
                    fix_prompt = f"""
The SQL below does not implement the validated goal plan:

{chr(10).join('- ' + i for i in sem_issues)}

User goal:
{user_goal}

Resolved goal plan (treat as ground truth):
{plan_json}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query that satisfies the goal plan.
No explanations, no markdown.
"""
                    try:
                        raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500,
                                                timeout=self._REPAIR_TIMEOUT)
                        current_sql = self._clean_sql(raw)
                        continue
                    except Exception:
                        _rb = self._no_llm_rebuild(
                            user_goal, join_path, kpi_map, ir, join_plan,
                            current_sql)
                        if _rb:
                            logger.warning(
                                "LLM semantic-plan repair unavailable; continuing "
                                "with deterministic rebuild (no LLM).")
                            current_sql = _rb
                            continue
                        raise RuntimeError(
                            "Generated SQL violates the semantic goal plan: "
                            + "; ".join(sem_issues)
                            + " and could not be repaired."
                        )

            # Fail-closed gate before EVERY execution: repaired SQL is
            # re-validated here so no path reaches the DB unchecked.
            current_sql = _enforce(current_sql)
            try:
                _t0 = time.monotonic()
                _MAX_ROWS = 10_000
                _STMT_TIMEOUT_MS = 30_000
                with self.engine.connect() as conn:
                    if self.dialect == "postgresql":
                        conn.execute(text(f"SET statement_timeout = '{_STMT_TIMEOUT_MS}'"))
                    elif self.dialect == "mysql":
                        conn.execute(text(f"SET SESSION max_execution_time = {_STMT_TIMEOUT_MS}"))
                    result = conn.execute(text(current_sql))
                    rows = result.fetchmany(_MAX_ROWS + 1)
                    truncated = len(rows) > _MAX_ROWS
                    if truncated:
                        rows = rows[:_MAX_ROWS]
                    cols = list(result.keys())
                self._last_execution_ms = int((time.monotonic() - _t0) * 1000)
                df = pd.DataFrame(rows, columns=cols)
                if df.empty:
                    empty_tabs = empty_sql_tables(current_sql, self._row_counts())
                    if empty_tabs:
                        # The data genuinely is not there; repairing is pointless.
                        names = ", ".join(sorted(empty_tabs))
                        return [], current_sql, {
                            "message": (
                                f"The query is valid but returned no rows because "
                                f"the table(s) {names} are empty in this database. "
                                f"Try a goal based on the tables that contain data."
                            )
                        }
                    # LLM produced 0-row SQL: try deterministic template as
                    # fallback (handles TEXT-date columns the LLM misses).
                    _det = None
                    try:
                        _det = self._deterministic_sql(
                            user_goal, join_path, kpi_map, ir, join_plan)
                    except Exception:
                        pass
                    if _det and _det.strip() != current_sql.strip():
                        try:
                            _det = self._clean_sql(_det)
                            with self.engine.connect() as _c:
                                _r = self._execute_guarded(_c, _det)
                                _rows = _r.fetchmany(10_001)
                                _cols = list(_r.keys())
                            if _rows:
                                _df = pd.DataFrame(_rows, columns=_cols)
                                _df = _df.where(pd.notnull(_df), None)
                                for _c in _df.select_dtypes(include=['float64', 'int64']).columns:
                                    _df[_c] = _df[_c].astype(float)
                                records = _df.to_dict(orient="records")
                                logger.info(
                                    "Deterministic fallback recovered %d rows "
                                    "for trend goal",
                                    len(records))
                                return records, _det, None
                        except Exception:
                            pass
                    # Spec §14: zero rows are NOT an error. Do NOT auto-broaden
                    # joins / loosen filters just because the result is empty.
                    # Return success + row_count 0 with an honest message.
                    return [], current_sql, {
                        "message": (
                            "The query is valid and ran successfully, but returned "
                            "no rows for the current data."
                        )
                    }

                df = df.where(pd.notnull(df), None)
                for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                for col in df.select_dtypes(include=['float64', 'int64']).columns:
                    df[col] = df[col].astype(float)
                records = df.to_dict(orient="records")
                # Result validation (n.txt item 2): never hand a broken frame
                # to the UI — confirm shape/emptiness is genuine first.
                self._validate_result_shape(user_goal, current_sql, records, cols)
                note = None
                if truncated:
                    note = {
                        "message": (
                            f"Results were truncated to {_MAX_ROWS:,} rows "
                            f"(the query returned more). Add filters or a LIMIT to "
                            f"narrow the result set."
                        )
                    }
                logger.info(
                    "SQL retry: generation_time_ms=%d repair_count=%d outcome=success "
                    "rows=%d goal=%.120s",
                    int((time.monotonic() - _t_start) * 1000), attempt,
                    len(records), (user_goal or "")[:120])
                return records, current_sql, note
            except Exception as exc:
                _validation_failures.append(f"exec:{str(exc)[:120]}")
                # The database itself rejects an unknown column (bare reference
                # the pre-checks skip): fuzzy-repair against the real schema
                # and retry within budget instead of failing the question.
                _bad_name = _extract_undefined_column(exc)
                if _bad_name and attempt < retries:
                    try:
                        _fixed, _ok = self._repair_fuzzy_columns(
                            current_sql, columns_by_table, [_bad_name])
                    except Exception:
                        _fixed, _ok = current_sql, False
                    if _ok:
                        logger.warning(
                            "Execution hit undefined column %s; repaired by "
                            "fuzzy match (no LLM).", _bad_name)
                        current_sql = _fixed
                        continue
                if attempt >= retries:
                    # Give up on the LLM SQL, but never fail the request: try
                    # the IR-driven deterministic fallback (verified measure,
                    # real FK joins, preserved aggregation) first.
                    # Bounded like the main path: never fetchall() an
                    # unbounded result into the JSON payload.
                    try:
                        safe_sql = self._fallback_sql(
                            join_path, ir=ir, join_plan=join_plan,
                            user_goal=user_goal)
                        safe_sql = _enforce(safe_sql)
                        with self.engine.connect() as conn:
                            result = self._execute_guarded(conn, safe_sql)
                            rows = result.fetchmany(_MAX_ROWS + 1)
                            truncated = len(rows) > _MAX_ROWS
                            if truncated:
                                rows = rows[:_MAX_ROWS]
                            cols = list(result.keys())
                        df = pd.DataFrame(rows, columns=cols)
                        df = df.where(pd.notnull(df), None)
                        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                        for col in df.select_dtypes(include=['float64', 'int64']).columns:
                            df[col] = df[col].astype(float)
                        if self._goal_asks_for_aggregation(user_goal) \
                                and self._find_bare_select_star(safe_sql):
                            # A raw `SELECT *` fallback would be misleading for an
                            # aggregating goal (it charts IDs, not the breakdown the
                            # user asked for). Explain instead of guessing. An
                            # IR-driven fallback that aggregates answers the goal
                            # and is returned as success below.
                            logger.warning(
                                "SQL retry: generation_time_ms=%d repair_count=%d "
                                "outcome=safe_fail detail=aggregate_fallback_refused goal=%.120s",
                                int((time.monotonic() - _t_start) * 1000), attempt,
                                (user_goal or "")[:120])
                            return [], safe_sql, {"failed": True, "message": _SAFE_FAIL}
                        logger.warning("LLM SQL exhausted retries (%s); returning safe fallback.", exc)
                        _fb_note = None
                        if truncated:
                            _fb_note = {
                                "message": (
                                    f"Results were truncated to {_MAX_ROWS:,} rows "
                                    f"(the query returned more). Add filters or a LIMIT to "
                                    f"narrow the result set."
                                )
                            }
                        return df.to_dict(orient="records"), safe_sql, _fb_note
                    except Exception as safe_exc:
                        raise RuntimeError(
                            f"Could not build a working query for this goal after retrying "
                            f"({exc}). Last fallback also failed: {safe_exc}"
                        ) from exc
                logger.warning(f"SQL failed on attempt {attempt + 1}: {exc}. Asking LLM to repair.")
                fix_prompt = f"""
The SQL below failed with error:
{exc}

User goal:
{user_goal}

Relevant schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Fix it and return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500,
                                            timeout=self._REPAIR_TIMEOUT)
                    current_sql = self._clean_sql(raw)
                except Exception:
                    _rb = self._no_llm_rebuild(
                        user_goal, join_path, kpi_map, ir, join_plan,
                        current_sql)
                    if _rb:
                        logger.warning(
                            "LLM execution-error repair unavailable; continuing "
                            "with deterministic rebuild (no LLM).")
                        current_sql = _rb
                    continue

        logger.warning(
            "SQL retry: generation_time_ms=%d repair_count=%d outcome=safe_fail "
            "validation_failures=%s goal=%.120s",
            int((time.monotonic() - _t_start) * 1000), retries,
            _validation_failures[-3:] if _validation_failures else [],
            (user_goal or "")[:120])
        return [], current_sql, {"failed": True, "message": _SAFE_FAIL}

    # ---- _clean_data (_clean_data) ----

    def _clean_data(self, records):
        """Preprocess query results before writing processed_data.json.

        Handles gracefully:
          - fully-empty columns  -> dropped
          - fully-empty rows     -> dropped
          - NULL values          -> PRESERVED as null (spec §13: the raw
                                   analytical result must never fabricate
                                   0 / '' for missing data)
          - types               -> coerced (numeric/text/datetime)

        Stores a human-readable report on self.preprocessing_report so the
        payload and UI can explain exactly what was cleaned.
        """
        report = {
            "dropped_empty_columns": [],
            "dropped_empty_rows": 0,
            "nulls_preserved": 0,
            "columns_kept": [],
            "notes": [],
        }
        if not records:
            self.preprocessing_report = report
            return []

        df = pd.DataFrame(records)
        df = df.replace({pd.NA: None})

        def _is_empty(series):
            return series.isna() | (series.astype(str).str.strip().isin(["", "nan", "None", "null"]))

        # 1) Drop columns that are entirely empty.
        empty_cols = [c for c in df.columns if len(df[c]) and _is_empty(df[c]).all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            report["dropped_empty_columns"] = empty_cols
            report["notes"].append(
                f"Dropped {len(empty_cols)} fully-empty column(s): {', '.join(empty_cols)}."
            )

        if df.empty or len(df.columns) == 0:
            report["notes"].append("No usable columns remain after cleaning.")
            self.preprocessing_report = report
            return []

        # 2) Drop rows that are entirely empty across every remaining column.
        all_empty = df.apply(lambda r: _is_empty(r).all(), axis=1)
        dropped_rows = int(all_empty.sum())
        if dropped_rows:
            df = df.loc[~all_empty].reset_index(drop=True)
            report["dropped_empty_rows"] = dropped_rows
            report["notes"].append(f"Dropped {dropped_rows} entirely-empty row(s).")

        # 3) Coerce types but PRESERVE NULL values (spec §13): no 0 / '' fill.
        for col in df.columns:
            col_null = int(df[col].isna().sum())
            kind = df[col].dtype.kind
            if kind in "fc":  # float / complex -> numeric (NaN kept as null)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif kind in "iub":  # int / bool -> numeric as float (NaN kept as null)
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif kind == "M":  # datetime
                df[col] = df[col].astype(object).where(df[col].notna(), None)
            elif kind == "O":  # object / string (includes Decimal, datetime objects)
                # SQLAlchemy returns Postgres NUMERIC as Decimal objects; coerce
                # numeric-looking object columns to float so they are treated as
                # measures downstream (KPIs, charts, predictions), for any database.
                if len(df[col]) and df[col].dropna().map(
                    lambda v: isinstance(v, (datetime, pd.Timestamp))
                ).all():
                    df[col] = df[col].astype(object).where(df[col].notna(), None)
                    continue
                sample = df[col].dropna()
                if len(sample):
                    coerced = pd.to_numeric(sample, errors="coerce")
                    num_ok = int(coerced.notna().sum())
                    if num_ok / len(sample) >= 0.8:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                        report["nulls_preserved"] += col_null
                        continue
                df[col] = df[col].astype(object).where(df[col].notna(), None)
            report["nulls_preserved"] += col_null

        report["columns_kept"] = list(df.columns)
        if report["nulls_preserved"]:
            report["notes"].append(
                f"Preserved {report['nulls_preserved']} NULL value(s) as null "
                "(no 0/'' fabrication)."
            )
        self.preprocessing_report = report

        # Explicitly materialize NaN cells as None (float/datetime columns hold
        # NaN, not None) so the JSON payload preserves NULL (spec §13).
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].astype(object).where(df[col].notna(), None)

        records = df.to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        return records

    # ---- _validate_result_shape (_validate_result_shape) ----

    def _validate_result_shape(self, user_goal, sql, records, cols):
        """Post-execution result validation (n.txt item 2).

        Confirms the frame is safe for chart/KPI/NL stages: non-empty columns,
        no all-NULL frame, aggregation goals return grouped (not raw *-dump)
        rows. Zero rows are NOT auto-repaired (spec §14) — they are logged and
        returned honestly. Raises only on structurally broken frames.
        """
        cols = list(cols or [])
        shape = {"rows": len(records or []), "columns": cols}
        if not cols:
            raise RuntimeError("Execution returned rows with no columns; refusing to visualize.")
        if records and all(all(v is None for v in r.values()) for r in records):
            raise RuntimeError("Execution returned only NULL rows; refusing to visualize.")
        if self._goal_asks_for_aggregation(user_goal) and records:
            bare = self._find_bare_select_star(sql or "")
            if bare:
                logger.warning("Result validation: aggregating goal returned bare SELECT * shape.")
        logger.info(
            "Result validation: question=%r rows=%d cols=%s sql=%.120s",
            (user_goal or "")[:160], shape["rows"], cols, (sql or "")[:120],
        )
        return shape

    # ---- _log_pipeline_diagnostic (_log_pipeline_diagnostic) ----

    def _log_pipeline_diagnostic(self, trace: dict):
        """Internal diagnostic log for the query-generation pipeline.

        P3 §31 contract (internal only, never returned to the UI):
        request_id, session_id, dataset_id, question, intent, tables,
        columns, sql, validation, exec_status, result_shape, context,
        model, latency, final_status. One structured INFO line; values are
        truncated so logs stay bounded.
        """
        def _t(v, n=300):
            s = str(v if v is not None else "")
            return s[:n]
        logger.info(
            "pipeline trace: request_id=%s session_id=%s dataset_id=%s "
            "question=%r intent=%s tables=%s columns=%s sql=%r "
            "validation=%s exec_status=%s result_shape=%s context=%s "
            "model=%s latency_ms=%s final_status=%s answer=%r",
            _t(trace.get("request_id"), 64),
            _t(trace.get("session_id"), 64),
            _t(trace.get("dataset_id"), 128),
            _t(trace.get("question"), 160),
            _t(trace.get("intent"), 64),
            trace.get("tables"),
            trace.get("columns"),
            _t(trace.get("sql"), 300),
            trace.get("validation"),
            trace.get("exec_status", trace.get("exec_ms")),
            trace.get("result_shape", trace.get("shape")),
            _t(trace.get("context"), 200),
            _t(trace.get("model"), 64),
            trace.get("latency_ms", trace.get("exec_ms")),
            trace.get("final_status"),
            _t(trace.get("answer"), 200),
        )
        self._last_pipeline_trace = dict(trace)

