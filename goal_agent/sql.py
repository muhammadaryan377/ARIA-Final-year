"""Component: SQL generation, repair and semantic validation (read-only, extremes, missing joins, unknown columns, fan-out)."""

import difflib
import json
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

# ---- Deterministic fallback vocabulary (schema-agnostic) -------------------
# Verified-numeric type fragments: a SUM/AVG measure must carry one of these
# in its declared data_type. Never blob/binary/bytes/photo/picture/counter.
_SAFE_NUMERIC_TYPE_TOKENS = (
    "smallint", "integer", "bigint", "decimal", "numeric",
    "real", "double precision", "float", "money", "serial",
    "tinyint", "int",
)
_UNSAFE_MEASURE_TYPE_TOKENS = (
    "bytea", "blob", "binary", "varbinary", "bytes", "image",
)
# Name tokens that disqualify a column as a measure (token equality, never
# bare substrings, so "encounter" does not trip on "counter").
_UNSAFE_MEASURE_NAME_TOKENS = frozenset({
    "binary", "blob", "bytea", "bytes", "image", "photo", "picture",
    "thumb", "thumbnail", "thumbs", "icon", "counter", "rowversion",
})
_UNVERIFIED_DT = frozenset({"", "unknown", "inferred", "none", "null"})


def _fallback_name_tokens(name):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name or ""))
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


class SqlMixin:
    # ---- _is_read_only_sql (_is_read_only_sql) ----

    def _is_read_only_sql(self, sql):
        """True only for read-only SQL (SELECT / CTE-WITH). Deterministically
        rejects INSERT/UPDATE/DELETE/DDL/DCL/DML before the DB round-trip.
        Returns False for empty/None SQL (not a valid statement).
        Delegates to the single shared guard in core.validation."""
        from core.validation import is_read_only_sql as _shared_is_read_only
        return _shared_is_read_only(sql)

    # ---- _warnings_for_sql (_warnings_for_sql) ----

    def _warnings_for_sql(self, sql):
        """Warn when the final query joins through an UNCERTAIN/REVIEW
        relationship (spec §5: do not silently use uncertain relationships) or
        through a low-confidence inferred relationship (spec §6)."""
        warnings = []
        if not sql:
            return warnings
        ref_tables = {t.lower() for t in self._sql_structure(sql)["in_from"]}
        if not ref_tables:
            return warnings
        for key in sorted(self.uncertain_edge_keys):
            src_table = key.split("->")[0].split(".")[0].lower()
            tgt_table = key.split("->")[1].split(".")[0].lower()
            if src_table in ref_tables and tgt_table in ref_tables:
                warnings.append(
                    f"Relationship {key} is UNCERTAIN/REVIEW in the Schema Agent "
                    "output. It was used in this query - validate the join manually."
                )
        low_trust = {"MEDIUM_CONFIDENCE_INFERRED", "LOW_CONFIDENCE_INFERRED"}
        for edge in getattr(self, "relationship_edges", []):
            if edge.get("trust") not in low_trust:
                continue
            if (edge["left_table"].lower() in ref_tables
                    and edge["right_table"].lower() in ref_tables):
                warnings.append(
                    f"Relationship {edge['left_table']}.{edge['left_column']} -> "
                    f"{edge['right_table']}.{edge['right_column']} is "
                    f"{edge['trust']} in the Schema Agent output. It was used in "
                    "this query - validate the join manually."
                )
        return warnings

    # ---- _clean_sql (_clean_sql) ----

    def _clean_sql(self, raw_sql):
        if not raw_sql or not str(raw_sql).strip():
            return "SELECT 1;"
        cleaned = str(raw_sql).strip()
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        # Only strip double-quotes that wrap the ENTIRE statement (the LLM
        # sometimes wraps its answer in quotes).  Never strip quotes around
        # individual identifiers — they protect reserved words and
        # case-sensitive names (e.g. SELECT "order" FROM "order").
        if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
            cleaned = cleaned[1:-1]
        # Small local models often ECHO the prompt's instruction sentences into
        # the answer ("with null values last. The final SQL statement is
        # provided without explanations or markdown; ..."). A real statement is
        # SELECT, or a genuine CTE (WITH <name> AS ( ... )). Plain English
        # "with ..." must NOT be mistaken for a WITH clause (it silently
        # corrupts the SQL), and the statement start must be the FIRST
        # top-level keyword: taking the last SELECT would capture inner
        # SELECTs from subqueries/CTEs and corrupt the query.
        stmt_pattern = re.compile(
            r"\bWITH\s+[A-Za-z_]\w*\s+AS\s*\(|\bSELECT\b|"
            r"\bINSERT\s+INTO\b|\bUPDATE\b|\bDELETE\s+FROM\b",
            re.IGNORECASE,
        )
        tail_pattern = re.compile(
            r"(?i)\b(the final sql statement|without explanations|no explanations|"
            r"here(?:'s| is) (?:the|your) final|output only[^\n]*|markdown|"
            r"with null values (?:last|first)[^\n]*|"
            r"this (?:query|sql) (?:is|uses|returns)[^\n]*)"
        )
        statements = []
        for fragment in cleaned.split(";"):
            match = stmt_pattern.search(fragment)
            if not match:
                continue
            frag = fragment[match.start():].strip()
            cut = tail_pattern.search(frag)
            if cut:
                frag = frag[:cut.start()].strip()
            if frag:
                statements.append(frag)
        if statements:
            return statements[-1].rstrip(";") + ";"
        # No real statement found: return the raw text so the validation layer /
        # repair loop sees it as broken and asks the model to fix it.
        return str(raw_sql).strip() + ";"

    # ---- _find_nested_aggregates (_find_nested_aggregates) ----

    def _find_nested_aggregates(self, sql):
        """Return the nested-aggregate snippets found in sql, e.g. AVG(SUM(...)).

        Matches an aggregate keyword directly wrapping another aggregate keyword.
        """
        matches = re.findall(
            r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(AVG|SUM|COUNT|MIN|MAX)\s*\(",
            sql,
            re.IGNORECASE,
        )
        return [f"{a}({b}" for a, b in matches]

    # ---- _quote (_quote) ----

    @staticmethod
    def _quote(name, dialect=None):
        """Quote an identifier (same pattern as core/schema_intel._safe_name).

        Escapes embedded double quotes; backticks for MySQL. Used for
        identifiers only — never for values (values always go via text()
        bound parameters).
        """
        safe = str(name or "")
        if (dialect or "") == "mysql":
            return f"`{safe.replace('`', '``')}`"
        return f'"{safe.replace(chr(34), chr(34) + chr(34))}"'

    def _quoted(self, name):
        return self._quote(name, getattr(self, "dialect", "postgresql"))

    # ---- _columns_by_type (_columns_by_type) ----

    def _columns_by_type(self):
        """Return {table: {column: data_type}} for datatype validation."""
        out = {}
        for t, info in (self.tables or {}).items():
            cols = info.get("columns", {}) or {}
            if isinstance(cols, dict):
                out[t] = {c: (v.get("data_type", "") if isinstance(v, dict) else "")
                          for c, v in cols.items()}
            else:
                out[t] = {str(c): "" for c in cols}
        return out

    # ---- _find_incompatible_aggregations (_find_incompatible_aggregations) ----

    def _find_incompatible_aggregations(self, sql):
        """SUM/AVG over text/datetime columns. Delegates to core.validation."""
        from core.validation import find_incompatible_aggregations as _inc
        return _inc(sql, self._columns_by_type())

    # ---- _validate_generated_sql (_validate_generated_sql) ----

    def _validate_generated_sql(self, sql):
        """Post-generation gate (n.txt item 2): one call covering nonexistent
        table/column, nested aggregates and incompatible datatypes. Returns a
        list of issue strings ([] = PASS). JOIN-shape checks
        (_find_missing_from_tables / pk-fk) run alongside in the retry loop."""
        from core.validation import validate_generated_sql as _validate
        columns_by_table = {n: list(i.get("columns", {})) for n, i in self.tables.items()}
        return _validate(
            sql,
            schema_tables=list(self.tables),
            columns_by_table=columns_by_table,
            columns_by_type=self._columns_by_type(),
        )

    def _enforce_execution_guards(self, sql):
        """Fail-closed allowlist gate before EVERY execution.

        Asserts ``is_read_only_sql()`` AND ``validate_generated_sql()``
        unknown tables/columns == []. Raises RuntimeError on failure; never
        coerces to a forgiving default.
        """
        from core.validation import enforce_readonly_guards as _enforce
        mapping = {
            name: list(info.get("columns", {}) or {})
            for name, info in (getattr(self, "tables", {}) or {}).items()
        }
        return _enforce(sql, mapping)

    def _execute_guarded(self, conn, sql, timeout_ms=30_000):
        """Guard + timeout + execute: single choke point for every execution.

        Applies ``_enforce_execution_guards`` (fail-closed read-only +
        unknown table/column allowlist — the same call the main execution
        path makes), then sets the same dialect-conditional statement
        timeout the main execution path sets (PostgreSQL
        ``statement_timeout`` / MySQL ``max_execution_time``; SQLite and
        other dialects get no SET), then executes ``sql`` and returns the
        SQLAlchemy Result. Raises from the guard before touching the DB.
        """
        guarded = self._enforce_execution_guards(sql)
        dialect = getattr(self, "dialect", "")
        if dialect == "postgresql":
            conn.execute(text(f"SET statement_timeout = '{int(timeout_ms)}'"))
        elif dialect == "mysql":
            conn.execute(text(f"SET SESSION max_execution_time = {int(timeout_ms)}"))
        return conn.execute(text(guarded))

    # ---- _goal_asks_for_aggregation (_goal_asks_for_aggregation) ----

    def _goal_asks_for_aggregation(self, user_goal):
        """True if the goal text clearly wants an aggregation / breakdown, in
        which case a bare `SELECT *` answer is almost certainly wrong."""
        goal = (user_goal or "").lower()
        markers = (
            "total", "count", "how many", "number of", "sum of", "average", "avg",
            "mean", "per ", "by ", "maximum", "minimum", "highest", "lowest",
            "top ", "share", "percentage", "breakdown", "distribution",
        )
        return any(m in goal for m in markers)

    # ---- _is_extremes_goal (_is_extremes_goal) ----

    def _is_extremes_goal(self, user_goal):
        """True if the goal explicitly asks for BOTH the top and bottom of a
        ranking (e.g. 'highest and lowest', 'most and least', 'max and min',
        'best and worst', 'top and bottom')."""
        goal = re.sub(r"[^a-z\s]", " ", (user_goal or "").lower())
        words = set(goal.split())
        has_max = bool(words & self._EXTREMES_MAX)
        has_min = bool(words & self._EXTREMES_MIN)
        return has_max and has_min

    # ---- _fix_extremes_sql (_fix_extremes_sql) ----

    def _fix_extremes_sql(self, user_goal, sql):
        """If the goal asks for both extremes and `sql` is a single-sided
        `ORDER BY ... (ASC|DESC) ... LIMIT 1`, strip the trailing LIMIT so the
        full ranking comes back. Returns (sql, changed)."""
        if not sql or not self._is_extremes_goal(user_goal):
            return sql, False
        if not re.search(r"\bORDER\s+BY\b[\s\S]*\b(ASC|DESC)\b", sql, re.IGNORECASE):
            return sql, False
        new_sql, n = re.subn(
            r"\s+LIMIT\s+\d+(?:\s+OFFSET\s+\d+)?(\s*;?\s*)$",
            r"\1",
            sql.strip(),
            count=1,
            flags=re.IGNORECASE,
        )
        if n:
            logger.warning(
                "Extremes goal detected; stripped trailing LIMIT so both the "
                "highest and lowest appear in the result set."
            )
            return new_sql.strip(), True
        return sql, False

    # ---- _sql_structure (_sql_structure) ----

    def _sql_structure(self, sql):
        """Return a dict describing the FROM/JOIN structure of `sql`:

          - in_from:    set of table names present in FROM/JOIN clauses
          - alias_to_name, name_to_alias: alias mappings
          - first_ref:  reference token (alias or bare name) of the FIRST
                        FROM table — the only table guaranteed visible at the
                        point right after the FROM clause.
          - intro:      table name -> byte position just after the clause that
                        introduces it (end of the FROM/JOIN block).

        Implemented as a positional scan that skips parenthesized subqueries
        so a JOIN/ON keyword inside a subquery is never mistaken for a
        top-level clause.
        """
        out = {
            "in_from": set(),
            "alias_to_name": {},
            "name_to_alias": {},
            "first_ref": None,
            "intro": {},
        }
        pos = 0
        first = True
        depth = 0
        while pos < len(sql):
            ch = sql[pos]
            if ch == "(":
                depth += 1
                pos += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                pos += 1
                continue
            if depth > 0:
                pos += 1
                continue
            # Quoted identifiers (e.g. FROM "walmart") must resolve: the old
            # trailing \b never matches after a closing quote (quote and space
            # are both non-word chars), so quoted tables were invisible here.
            # The lookahead accepts a closing quote or any non-identifier char.
            m = re.search(
                r"\b(?:FROM|JOIN)\s+[\"'`\[]?([A-Za-z_][A-Za-z0-9_]*)"
                r"(?:[\"'`\]])?(?:\s*\.\s*[\"'`\[]?([A-Za-z_][A-Za-z0-9]*)"
                r"(?:[\"'`\]])?)?(?![A-Za-z0-9_])",
                sql[pos:], re.IGNORECASE,
            )
            if not m:
                break
            table = m.group(1)
            if m.group(2):
                table = f"{table}.{m.group(2)}"
            out["in_from"].add(table)
            after = pos + m.end()
            alias = None
            tail = sql[after:]
            am = re.match(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
            if am:
                alias = am.group(1)
            else:
                am = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
                if am and am.group(1).lower() not in self._SQL_KEYWORDS:
                    alias = am.group(1)
            if alias and alias.lower() not in self._SQL_KEYWORDS:
                out["alias_to_name"][alias] = table
                out["name_to_alias"][table] = alias
                after += am.end()
            if first:
                out["first_ref"] = out["name_to_alias"].get(table, table)
                first = False
            out["intro"][table] = self._clause_end(sql, after)
            pos = after
        return out

    # ---- _clause_end (_clause_end) ----

    def _clause_end(self, sql, start):
        """Byte position where the clause starting at `start` ends: the next
        top-level JOIN/WHERE/GROUP/ORDER/HAVING/LIMIT/OFFSET keyword or `;`."""
        depth = 0
        i = start
        n = len(sql)
        while i < n:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ";":
                return i
            elif depth == 0:
                m = re.match(
                    r"\b(?:JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|OFFSET)\b",
                    sql[i:], re.IGNORECASE,
                )
                if m:
                    return i
            i += 1
        return n

    # ---- _missing_refs (_missing_refs) ----

    def _missing_refs(self, sql):
        """Return a list of {'table', 'ref'} for tables referenced via a
        qualified column but missing from FROM/JOIN. `ref` is the token the
        query actually uses for that table (an alias like `b`, or the bare
        table name). Unknown prefixes are resolved to the single schema table
        whose name starts with that prefix (the model invented an alias without
        declaring `FROM <table> <alias>`)."""
        if not sql:
            return []
        struct = self._sql_structure(sql)
        in_from_lc = {t.lower() for t in struct["in_from"]}
        if not in_from_lc:
            return []
        table_names_lc = {t.lower() for t in self.tables}
        found = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|\*)\b", sql):
            prefix = m.group(1).lower()
            if prefix in struct["alias_to_name"]:
                table = struct["alias_to_name"][prefix].lower()
            elif prefix in table_names_lc:
                table = prefix
            else:
                candidates = [t for t in self.tables if t.lower().startswith(prefix)]
                if len(candidates) != 1:
                    # Schema-qualified keys: match the bare segment too
                    # (salesorderheader -> sales.salesorderheader).
                    candidates = [t for t in self.tables
                                  if t.lower().split(".")[-1] == prefix]
                    if len(candidates) != 1:
                        continue
                table = candidates[0].lower()
            if table not in in_from_lc and table in table_names_lc:
                found[table] = prefix
        return [{"table": t, "ref": r} for t, r in sorted(found.items())]

    # ---- _find_missing_from_tables (_find_missing_from_tables) ----

    def _find_missing_from_tables(self, sql):
        """Return table names that are referenced (via `tbl.col` or a JOIN ON
        condition) but never appear in a FROM/JOIN clause."""
        return [m["table"] for m in self._missing_refs(sql)]

    # ---- _inject_missing_joins (_inject_missing_joins) ----

    def _inject_missing_joins(self, sql, missing):
        """Deterministically inject JOINs for tables that are referenced but
        missing from FROM/JOIN. Only inject when there is a clear FK edge to the
        FIRST table in FROM â€” the one guaranteed to be visible at the insertion
        point (a table referenced in an earlier ON clause must be introduced
        before that clause). Returns (new_sql, injected_count) or (sql, 0)."""
        if not missing:
            return sql, 0
        struct = self._sql_structure(sql)
        first_ref = struct["first_ref"]
        intro_pos = struct["intro"]
        if first_ref is None or not intro_pos:
            return sql, 0
        first_table = struct["alias_to_name"].get(first_ref, first_ref)
        if first_table not in intro_pos:
            return sql, 0
        edges = self._fk_edges()
        joins = []
        for item in self._missing_refs(sql):
            table = item["table"]
            ref = item["ref"]
            real = next((t for t in self.tables if t.lower() == table), table)
            alias_txt = "" if ref == table else f" {ref}"
            candidate = None
            for (src, col, ref_t, ref_col) in edges:
                if src.lower() == table and ref_t.lower() == first_table.lower():
                    candidate = f"JOIN {real}{alias_txt} ON {ref}.{col} = {first_ref}.{ref_col}"
                    break
            if candidate is None:
                for (src, col, ref_t, ref_col) in edges:
                    if ref_t.lower() == table and src.lower() == first_table.lower():
                        candidate = f"JOIN {real}{alias_txt} ON {first_ref}.{col} = {ref}.{ref_col}"
                        break
            if candidate:
                joins.append(candidate)
        if not joins:
            return sql, 0
        inject_at = intro_pos[first_table]
        injection = " " + " ".join(joins)
        new_sql = sql[:inject_at] + injection + " " + sql[inject_at:]
        return new_sql, len(joins)

    # ---- _join_clauses (_join_clauses) ----

    def _join_clauses(self, sql):
        """Return (start, end, table, on_start, on_end) for every JOIN clause,
        where on_start/on_end span the ON condition (None if no ON clause)."""
        clauses = []
        pos = 0
        while True:
            # Quote-tolerant trailing lookahead (same pattern as
            # _sql_structure): a closing quote is a non-word char, so the old
            # trailing \b never matched after it and quoted JOINs
            # (JOIN "products" ...) were invisible here.
            m = re.search(
                r"\bJOIN\s+([\"'`\[]?)([A-Za-z_][A-Za-z0-9_]*)(?:[\"'`\]])?(?![A-Za-z0-9_])",
                sql[pos:], re.IGNORECASE,
            )
            if not m:
                break
            start = pos + m.start()
            table = m.group(2)
            after = pos + m.end()
            tail = sql[after:]
            am = re.match(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
            if not am:
                am = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
                if am and am.group(1).lower() in self._SQL_KEYWORDS:
                    am = None
            if am:
                after += am.end()
            om = re.search(r"\bON\b", sql[after:], re.IGNORECASE)
            if not om:
                end = self._clause_end(sql, after)
                clauses.append((start, end, table, None, None))
            else:
                cond_start = after + om.end()
                cond_end = self._clause_end(sql, cond_start)
                clauses.append((start, cond_end, table, cond_start, cond_end))
            pos = cond_end if om else end
        return clauses

    # ---- _repair_unknown_columns (_repair_unknown_columns) ----

    def _repair_unknown_columns(self, sql, columns_by_table, bad_columns):
        """Deterministically fix hallucinated column references WITHOUT an LLM.

        Handles two failure modes the LLM repair loop repeatedly fails on:
          1. `tbl.col` where `col` actually lives on another table in the query's
             schema -> rewrite the qualifier to the owning table (the missing-FROM
             guard then injects the required FK join).
          2. a JOIN whose ON condition uses an unresolvable column -> drop that
             hallucinated JOIN entirely (only when the joined table is not used
             anywhere else in the query).

        JOIN ON predicates whose rewrite would self-compare the same table
        (``rooms.room_id = rooms.room_id``) are refused and instead dropped or
        left for the LLM. Returns (repaired_sql, changed).
        """
        if not bad_columns or not sql:
            return sql, False
        bad_set = {b.lower() for b in bad_columns}
        table_cols_lc = {t: {c.lower() for c in cols} for t, cols in columns_by_table.items()}
        col_owners = {}
        for t, cols in table_cols_lc.items():
            for c in cols:
                col_owners.setdefault(c, []).append(t)
        struct = self._sql_structure(sql)
        in_from_lc = {t.lower() for t in struct["in_from"]}

        def resolve_prefix(prefix):
            p = prefix.lower()
            if p in struct["alias_to_name"]:
                return struct["alias_to_name"][p].lower()
            if p in table_cols_lc:
                return p
            cands = [t for t in table_cols_lc if t.startswith(p)]
            return cands[0].lower() if len(cands) == 1 else None

        def near_match(table, col):
            norm = re.sub(r"[^a-z0-9]", "", col.lower())
            for c in table_cols_lc.get(table, ()):
                if re.sub(r"[^a-z0-9]", "", c) == norm:
                    return c
            for c in table_cols_lc.get(table, ()):
                if len(c) >= 4 and (norm.rstrip("s") == c.rstrip("s") or norm == c[:-1]):
                    return c
            return None

        on_regions = [(s, e) for (_, _, _, s, e) in self._join_clauses(sql) if s is not None]

        def other_side_table(pos, on_start, on_end):
            """Resolve the table of the operand on the other side of the `=`
            nearest to `pos` within an ON condition."""
            seg = sql[on_start:on_end]
            local = pos - on_start
            eq = seg.rfind("=", 0, local)
            eq2 = seg.find("=", local)
            bounds = []
            if eq != -1:
                bounds.append((seg.rfind("=", 0, eq - 1), eq))
            if eq2 != -1:
                bounds.append((eq, seg.find("=", eq2 + 1)))
            region = None
            for lo, hi in bounds:
                if lo <= local <= hi:
                    region = seg[lo + 1:hi if hi != -1 else len(seg)]
                    break
            if not region:
                return None
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b", region)
            return resolve_prefix(m.group(1)) if m else None

        def in_on(pos):
            return next(((s, e) for s, e in on_regions if s <= pos < e), None)

        # --- pass 1: rewrite qualifiers to the column's real owner -------------
        replacements = []
        for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|\*)\b", sql
        ):
            prefix, col = m.group(1), m.group(2)
            if col == "*":
                continue
            tbl = resolve_prefix(prefix)
            if tbl is None or f"{tbl}.{col}".lower() not in bad_set:
                continue
            real = near_match(tbl, col)
            if real:
                replacements.append((m.start(), m.end(), f"{prefix}.{real}"))
                continue
            owners = [t for t in col_owners.get(col.lower(), []) if t != tbl and t not in in_from_lc]
            if len(owners) != 1:
                continue
            region = in_on(m.start())
            if region:
                other = other_side_table(m.start(), region[0], region[1])
                if other == owners[0]:
                    continue  # would become X.x = X.x (self-comparison); drop/LLM instead
            replacements.append((m.start(), m.end(), f"{owners[0]}.{col}"))
        if replacements:
            out = sql
            for start, end, repl in sorted(replacements, reverse=True):
                out = out[:start] + repl + out[end:]
            if not unknown_sql_columns(out, columns_by_table):
                return out, True

        # --- pass 2: drop hallucinated JOINs (unresolvable ON column) ----------
        for (start, end, table, on_start, on_end) in self._join_clauses(sql):
            if on_start is None:
                continue
            cond = sql[on_start:on_end]
            used_elsewhere = False
            alias = struct["alias_to_name"].get(table)
            for token in {table, alias} - {None}:
                remaining = sql[:start] + sql[end:]
                if re.search(rf"\b{re.escape(token)}\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b",
                             remaining, re.IGNORECASE):
                    used_elsewhere = True
                    break
            if used_elsewhere:
                continue
            cond_bad = [b for b in bad_set if re.search(
                rf"\b{re.escape(b.split('.')[0])}\s*\.\s*{re.escape(b.split('.')[1])}\b",
                cond, re.IGNORECASE)]
            if not cond_bad:
                continue
            out = sql[:start] + sql[end:]
            if not unknown_sql_columns(out, columns_by_table):
                return out.strip(), True
        return sql, False

    # ---- _repair_fuzzy_columns (_repair_fuzzy_columns) ----

    def _repair_fuzzy_columns(self, sql, columns_by_table, bad_columns):
        """Fix near-miss column names WITHOUT an LLM (typos, pluralization,
        case and separator variants: `revnue` -> `revenue`, `amounts` ->
        `amount`, `orderDate` -> `order_date`).

        Same-table close matches win; otherwise a unique schema-wide owner
        wins with its qualifier rewritten. Returns (repaired_sql, changed);
        only returns changed=True when the result validates clean.
        """
        if not bad_columns or not sql:
            return sql, False
        table_cols = {t: list(cols) for t, cols in (columns_by_table or {}).items()}
        try:
            _aliases = {a.lower(): t.lower()
                        for a, t in (self._sql_structure(sql).get("alias_to_name") or {}).items()}
        except Exception:
            _aliases = {}

        def _norm(s):
            return re.sub(r"[^a-z0-9]", "", str(s or "").lower())

        def _best(cands, target):
            if not cands:
                return None
            tnorm = _norm(target)
            for c in cands:
                if _norm(c) == tnorm:
                    return c
            for c in cands:
                cn = _norm(c)
                if len(cn) >= 4 and (tnorm.rstrip("s") == cn.rstrip("s")):
                    return c
            scored = difflib.get_close_matches(target, list(cands), n=1, cutoff=0.78)
            return scored[0] if scored else None

        def _score(a, b):
            try:
                return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            except Exception:
                return 0.0

        out = sql
        for bad in bad_columns:
            if "." in bad:
                prefix, _, col = bad.partition(".")
                tbl = next((t for t in table_cols if t.lower() == prefix.lower()), None)
                if tbl is None and prefix.lower() in _aliases:
                    # Alias (e.g. `s.amont` for `sales AS s`): repair against
                    # the aliased table but keep the alias in the rewrite.
                    _at = next((t for t in table_cols
                                if t.lower() == _aliases[prefix.lower()]), None)
                    if _at is not None:
                        real = _best(table_cols[_at], col)
                        if real and real.lower() != col.lower():
                            out = re.sub(
                                rf"\b{re.escape(prefix)}\s*\.\s*{re.escape(col)}\b",
                                f"{prefix}.{real}", out, flags=re.IGNORECASE)
                    continue
                if tbl is None:
                    continue
                real = _best(table_cols[tbl], col)
                if real and real.lower() != col.lower():
                    out = re.sub(
                        rf"\b{re.escape(prefix)}\s*\.\s*{re.escape(col)}\b",
                        f"{prefix}.{real}", out, flags=re.IGNORECASE)
                    continue
                owners = [t for t, cols in table_cols.items()
                          if any(c.lower() == col.lower() for c in cols)
                          and t.lower() != tbl.lower()]
                if len(owners) == 1:
                    out = re.sub(
                        rf"\b{re.escape(prefix)}\s*\.\s*{re.escape(col)}\b",
                        f"{owners[0]}.{col}", out, flags=re.IGNORECASE)
                    continue
                best, best_tbl, best_score = None, None, 0.0
                for t, cols in table_cols.items():
                    m = _best(cols, col)
                    if m:
                        s = _score(col, m)
                        if s > best_score:
                            best, best_tbl, best_score = m, t, s
                if best and best_tbl:
                    out = re.sub(
                        rf"\b{re.escape(prefix)}\s*\.\s*{re.escape(col)}\b",
                        f"{best_tbl}.{best}", out, flags=re.IGNORECASE)
            else:
                # Bare identifier (usually from a DB "no such column" error):
                # unique schema-wide match wins, qualified to its owner.
                hits = [(t, _best(cols, bad)) for t, cols in table_cols.items()]
                hits = [(t, m) for t, m in hits if m]
                if len(hits) == 1:
                    t, m = hits[0]
                    out = re.sub(rf"\b{re.escape(bad)}\b", f"{t}.{m}",
                                 out, flags=re.IGNORECASE)
        if out != sql and not unknown_sql_columns(out, columns_by_table):
            return out, True
        return sql, False

    # ---- _find_bare_select_star (_find_bare_select_star) ----

    def _find_bare_select_star(self, sql):
        """Return a short description if `sql` is a bare `SELECT *` (no aggregate
        function, no GROUP BY). None otherwise. Never flags `SELECT t.*` that is
        accompanied by an aggregate."""
        m = re.search(r"\bselect\b([\s\S]*?)\bfrom\b", sql, re.IGNORECASE)
        if not m:
            return None
        select_part = m.group(1)
        if "*" not in select_part:
            return None
        has_agg = re.search(
            r"\b(?:count|sum|avg|average|min|max|stddev|std|variance|var_pop|var_samp|median|percentile_cont|percentile_disc)\s*\(",
            sql, re.IGNORECASE,
        )
        if has_agg:
            return None
        return "SELECT * with no aggregation"

    # ---- _find_suspicious_pk_joins (_find_suspicious_pk_joins) ----

    def _find_suspicious_pk_joins(self, sql):
        """Return equality join pairs that equate the PRIMARY KEYS of two
        DIFFERENT tables. In a normalized schema that is almost always a bug
        (e.g. `tracks.id = artists.id`); real joins go PK -> FK.

        Exception: a column that is both a member of a COMPOSITE primary key and
        a foreign key to the joined parent (e.g. order_details.product_id when
        order_details PK is (order_id, product_id)) is a legitimate FK join and
        is never flagged."""
        pk_of = {}
        for name, info in self.tables.items():
            _pk = set(info.get("primary_key", []) or info.get("inferred_primary_key", []) or [])
            # Generic composite members also count as key columns for the
            # PK=PK suspicious-join check (a composite member joined as an
            # FK to its parent is still exempted via the FK check below).
            try:
                _pk |= set(info.get("composite_primary_key", []) or [])
            except Exception:
                pass
            pk_of[name] = _pk
        # {table: {column: (referenced_table, referenced_column)}} from the FK edges.
        fk_of = {}
        for name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                fk_of.setdefault(name, {})[fk["column"]] = (
                    fk["referenced_table"], fk["referenced_column"],
                )

        # Resolve aliases to table names: FROM x [AS] a, JOIN y [AS] b.
        alias_to_table = {}
        for m in re.finditer(
            r"\b(FROM|JOIN)\s+([\"'`]?)(\w+)\2\s+(?:AS\s+)?([a-zA-Z_]\w*)",
            sql,
            re.IGNORECASE,
        ):
            table, alias = m.group(3), m.group(4)
            alias_to_table[alias.lower()] = table
            if not m.group(4):
                alias_to_table[table.lower()] = table

        suspects = []
        # Capture each ON clause and the equality conditions inside it.
        for on_m in re.finditer(
            r"\bJOIN\s+([\"'`]?)(\w+)\1\s+(?:AS\s+)?([a-zA-Z_]\w*)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|;)|$)",
            sql,
            re.IGNORECASE,
        ):
            joined_table, joined_alias, on_clause = on_m.group(2), on_m.group(3).lower(), on_m.group(4)
            for eq in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause):
                a, ca, b, cb = eq.groups()
                ta = alias_to_table.get(a.lower()) or a
                tb = alias_to_table.get(b.lower()) or b
                if ta == tb:
                    continue
                if ca in pk_of.get(ta, set()) and cb in pk_of.get(tb, set()):
                    if (ca in fk_of.get(ta, {}) and fk_of[ta][ca] == (tb, cb)) or (
                        cb in fk_of.get(tb, {}) and fk_of[tb][cb] == (ta, ca)
                    ):
                        continue
                    suspects.append(f"{ta}.{ca} = {tb}.{cb} (joins primary keys of unrelated tables)")
        return suspects

    # ---- _find_fk_mismatch_joins (_find_fk_mismatch_joins) ----

    def _find_fk_mismatch_joins(self, sql):
        """Return join equalities where an FK column is joined to a table it does
        NOT reference. E.g. tracks.album_id (FK -> albums.id) joined to artists.id
        is wrong: the FK definition says it must join albums, not artists."""
        # {table: {column: (referenced_table, referenced_column)}}
        fk_of = {}
        for name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                fk_of.setdefault(name, {})[fk["column"]] = (fk["referenced_table"], fk["referenced_column"])

        alias_to_table = {}
        for m in re.finditer(
            r"\b(FROM|JOIN)\s+([\"'`]?)(\w+)\2\s+(?:AS\s+)?([a-zA-Z_]\w*)",
            sql,
            re.IGNORECASE,
        ):
            table, alias = m.group(3), m.group(4)
            alias_to_table[alias.lower()] = table
            if not m.group(4):
                alias_to_table[table.lower()] = table

        suspects = []
        for on_m in re.finditer(
            r"\bJOIN\s+([\"'`]?)(\w+)\1\s+(?:AS\s+)?([a-zA-Z_]\w*)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|;)|$)",
            sql,
            re.IGNORECASE,
        ):
            on_clause = on_m.group(4)
            for eq in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause):
                a, ca, b, cb = eq.groups()
                ta = alias_to_table.get(a.lower()) or a
                tb = alias_to_table.get(b.lower()) or b
                if ta == tb:
                    continue
                if ca in fk_of.get(ta, {}):
                    ref_table, ref_col = fk_of[ta][ca]
                    if ref_table != tb or ref_col != cb:
                        suspects.append(
                            f"{ta}.{ca} references {ref_table}.{ref_col} but is joined to {tb}.{cb}"
                        )
                elif cb in fk_of.get(tb, {}):
                    ref_table, ref_col = fk_of[tb][cb]
                    if ref_table != ta or ref_col != ca:
                        suspects.append(
                            f"{tb}.{cb} references {ref_table}.{ref_col} but is joined to {ta}.{ca}"
                        )
        return suspects

    # ---- _llm_unavailable (_llm_unavailable) ----

    def _llm_unavailable(self, exc):
        """True when the language-model provider is down or rate-limited, as
        opposed to a genuine SQL-generation problem we could repair."""
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
            return True
        if any(k in msg for k in ("connection", "timed out", "timeout", "connect", "refused")):
            return True
        if "ollama" in msg and ("down" in msg or "refused" in msg or "connect" in msg):
            return True
        return False

    # ---- _generate_sql (_generate_sql) ----

    def _generate_sql(self, user_goal, join_path, kpi_map, ir=None, join_plan=None):
        schema_ddl = self._build_schema_ddl(join_path, full=False)
        if not schema_ddl:
            return "SELECT 1;"

        kpi_desc = ", ".join(k["description"] for k in kpi_map["kpis"]) or "none"

        ir_summary = ""
        if ir:
            bits = []
            if ir.get("aggregation"):
                bits.append(f"aggregation: {ir['aggregation']}")
            resolved_metrics = [m for m in ir.get("metrics", []) if m.get("resolved_expression")]
            if resolved_metrics:
                def _render_measure(m):
                    agg = m.get("aggregation")
                    expr = (m.get("resolved_expression") or "").strip()
                    if not expr:
                        return None
                    if expr.upper() == "COUNT(*)" or agg == "COUNT":
                        return "COUNT(*) (MANDATORY — count records, do not substitute SUM/AVG)"
                    if agg:
                        return f"{agg}({expr})"
                    return expr
                bits.append("resolved measures: " + ", ".join(
                    r for r in (_render_measure(m) for m in resolved_metrics) if r))
            if ir.get("dimensions"):
                bits.append("dimensions: " + ", ".join(
                    d["concept"] for d in ir["dimensions"]))
            if ir.get("time"):
                t = ir["time"]
                bits.append(
                    f"time: {t.get('constraint')} {t.get('value', '')} "
                    f"on {t.get('column') or 'the relevant date column'}")
            if ir.get("ranking"):
                r = ir["ranking"]
                bits.append(
                    f"ranking: top {r.get('limit', 'N')} by the measure "
                    f"({r['direction']})")
            if ir.get("filters"):
                bits.append("filters: " + ", ".join(
                    f"{f.get('column') or 'date'} {f['operator']} {f['value']}"
                    for f in ir["filters"]))
            if bits:
                ir_summary = (
                    "### Resolved goal plan (deterministic, treat as ground truth)\n"
                    + "\n".join(f"- {b}" for b in bits) + "\n"
                )

        join_plan_text = ""
        if join_plan and join_plan.get("edges"):
            lines = [
                f"  {e['left_table']}.{e['left_column']} = "
                f"{e['right_table']}.{e['right_column']} [{e.get('trust', 'declared')}]"
                for e in join_plan["edges"]
            ]
            join_plan_text = (
                "### Deterministic join plan (MUST be followed exactly)\n"
                "Join ONLY through these validated relationships:\n"
                + "\n".join(lines) + "\n"
            )

        if self.dialect == "sqlite":
            dialect_note = (
                "### SQLite notes\n"
                "Use standard SQLite syntax: strftime('%Y', col) / strftime('%Y-%m', col) "
                "for dates instead of EXTRACT, and || for concatenation.\n"
            )
        elif self.dialect == "mysql":
            dialect_note = (
                "### MySQL notes\n"
                "Quote identifiers with backticks when needed. Use DATE_FORMAT(col, '%Y') / "
                "DATE_FORMAT(col, '%Y-%m') for date grouping instead of EXTRACT/strftime. "
                "Use NOW()/CURDATE() for current time, LIKE (there is no ILIKE), LIMIT for "
                "paging, and CONCAT() instead of ||. Backslash is the string escape character.\n"
            )
        else:
            dialect_note = ""

        prompt = f"""
### Task
Generate a single {self.dialect} SQL query that answers the user's business goal.
User goal: {user_goal}
Aligned KPIs: {kpi_desc}
Dimensions: {', '.join(kpi_map['dimensions']) if kpi_map['dimensions'] else 'auto-detect'}

{ir_summary}
### Database Schema
The query will run on a database with the following schema:
{schema_ddl}
Preferred starting tables (inspect these first, but you may use any table
needed to answer the goal, joining via the FOREIGN KEY relationships above):
{join_path}
{join_plan_text}{dialect_note}
### SQL
Output only the final SQL statement. No explanations, no markdown, nothing after it.

### CRITICAL RULES
1. Aggregate functions must NEVER be nested (AVG(SUM(x)), SUM(COUNT(x)) are INVALID).
2. An average per group = SUM(x) / COUNT(DISTINCT key) or a subquery.
3. For "highest AND lowest" / "most and least" / "top and bottom" / "best and worst":
   do NOT use LIMIT 1. Return the full ranked list so both ends are present.
4. For "the highest" / "the lowest" / "top N": use ORDER BY + LIMIT.
5. Never add a trailing semicolon-only statement; a single trailing semicolon is fine.
6. NEVER assume a JOIN exists. If the question can be answered from ONE table,
   query ONLY that table with NO JOIN. JOINs are allowed ONLY through the
   validated relationships in the deterministic join plan above (or the
   FOREIGN KEY lines in the schema); with zero relationships the answer MUST
   be a single-table query.
7. For simple questions ("average X", "total X", "how many"), return exactly ONE
   aggregate column. Do NOT add extra aggregates (e.g. do not do SUM+AVG).
8. For "compare X and Y", return a simple GROUP BY or two columns, NOT a complex
   multi-join analysis. Keep queries under 30 words.

### QUERY PATTERNS — match the goal to the correct pattern:

#### COUNT pattern ("how many", "count of", "number of")
SELECT COUNT(*) FROM table;

#### TOTAL/SUM pattern ("total <measure>", "sum of <measure>")
SELECT SUM(measure_column) FROM table;
Only synthesize quantity * unit_price expressions when the schema shows
line-item grain AND the resolved measure is absent; never invent columns.

#### AVERAGE pattern ("average <measure>")
SELECT AVG(measure_column) FROM table;

#### GROUP BY pattern ("<measure> by <dimension>")
SELECT dimension, SUM(measure) FROM fact_table JOIN dimension_table GROUP BY dimension ORDER BY SUM(measure) DESC;

#### RANKING pattern ("top N <entities> by <measure>")
SELECT dimension, SUM(measure) FROM table GROUP BY dimension ORDER BY SUM(measure) DESC LIMIT N;

#### TREND/TIME SERIES pattern ("<measure> over time", "monthly <measure>")
SELECT date_trunc('month', date_col) AS month, SUM(measure) FROM table GROUP BY month ORDER BY month;

#### COMPARISON pattern ("compare <measure> across <dimension>")
SELECT dimension, SUM(measure) FROM table GROUP BY dimension ORDER BY SUM(measure) DESC;

#### ANTI-JOIN pattern ("<entities> never <associated>", "items with no <association>")
SELECT t1.* FROM table1 t1 LEFT JOIN table2 t2 ON t1.key = t2.key WHERE t2.key IS NULL;

#### PERCENTAGE/SHARE pattern ("share of <measure> by <dimension>")
SELECT dimension, SUM(measure) * 100.0 / SUM(SUM(measure)) OVER() AS pct FROM table GROUP BY dimension;

#### WINDOW FUNCTION pattern ("rank by <measure>", "running total", "percentile")
Use ROW_NUMBER(), RANK(), SUM() OVER(), etc.

#### SUBQUERY pattern ("<entities> above average <measure>")
SELECT * FROM table WHERE column > (SELECT AVG(column) FROM table);

#### DISTINCT pattern ("how many unique <entities>", "list all <dimension values>")
SELECT COUNT(DISTINCT column) FROM table;
SELECT DISTINCT column FROM table;

#### NULL HANDLING ("<entities> never <associated>" = LEFT JOIN + IS NULL)
Always use LEFT JOIN ... WHERE t2.key IS NULL for anti-joins.

RULES: use ONLY tables/columns from the schema mapping provided. Never
copy example nouns (products, revenue, regions) into the query — resolve
measure/dimension/date from the mapping. Output SQL only, no prose.

#### DATE FILTERING ("last year", "Q3", "this month", "2024")
Use WHERE date_col >= '2024-01-01' AND date_col < '2025-01-01' for year 2024.
Use date_trunc('quarter', date_col) for quarter filtering.
"""
        # ---- Deterministic-first path for simple patterns ----
        # Small LLMs (1-3B) frequently get simple GROUP BY / trend / ranking
        # queries wrong despite explicit instructions.  Deterministic templates
        # are faster, provider-agnostic, and always correct for these patterns.
        _det = self._deterministic_sql(user_goal, join_path, kpi_map, ir,
                                           join_plan)
        if _det:
            return _det

        try:
            # Fast, bounded LLM call: if it exceeds a short cap we fall back
            # to the fully dynamic deterministic SQL (no hardcoded table/column
            # names) so any database query is handled rather than hanging.
            _timeout = 60 if getattr(self.llm, "provider", "local") == "local" else 20
            raw = self.llm.complete("sql", prompt, temperature=0.1, num_predict=400, timeout=_timeout)
            sql = self._clean_sql(raw)
            try:
                sql = self._validate_sql_columns(user_goal, sql, ir)
            except Exception as e:
                logger.warning(f"SQL validation error: {e}")
            return sql
        except Exception as exc:
            logger.warning(f"LLM SQL generation failed: {exc}.")
            # Always try the fallback SQL rather than giving up on timeout
            return self._fallback_sql(join_path, ir=ir, join_plan=join_plan,
                                      user_goal=user_goal)

    # ---- _validate_sql_columns (_validate_sql_columns) ----

    def _validate_sql_columns(self, goal, sql, ir):
        """Post-validate: fix wrong columns AND simplify multi-aggregate queries."""
        if not sql:
            return sql
        import re
        goal_l = goal.lower()
        # Detect intent
        agg_match = None
        for kw, agg in [("average", "AVG"), ("avg", "AVG"), ("mean", "AVG"),
                         ("total", "SUM"), ("sum", "SUM")]:
            if kw in goal_l:
                agg_match = agg
                break
        if not agg_match:
            return sql
        # Build column lookup from schema
        all_cols = {}
        tables_list = []
        if ir:
            tables_list = ir.get("required_tables", []) or []
        if not tables_list:
            tables_list = list(self.tables.keys())[:1]
        for table in tables_list:
            try:
                for c, dt in self._table_columns_norm(table):
                    all_cols[c.lower()] = (table, c)
            except Exception:
                pass
        if not all_cols:
            return sql
        # Extract candidate column from the goal
        stripped = re.sub(r'\b(average|avg|mean|total|sum|what|is|the|of|are|there|how|many|price|value)\b', ' ', goal_l)
        words = [w.strip("?.!,;:'\"") for w in stripped.split() if len(w.strip("?.!,;:'\"")) >= 3]
        target = None
        for w in words:
            if w in all_cols:
                target = all_cols[w]
                break
        if not target:
            # Try multi-word: "discounted price" -> "discounted_price"
            for i in range(len(words)):
                for j in range(i + 2, min(i + 4, len(words) + 1)):
                    candidate = "_".join(words[i:j])
                    if candidate in all_cols:
                        target = all_cols[candidate]
                        break
                if target:
                    break
        if not target:
            # Fallback: find any numeric column whose name partially matches goal words
            for w in words:
                for col_lower, val in all_cols.items():
                    parts = col_lower.split("_")
                    if w in parts:
                        target = val
                        break
                if target:
                    break
        if not target:
            return sql
        target_table, target_col = target
        # Coerced text-numerics aggregate the CAST expression, never bare.
        try:
            if self._is_coerced_numeric_measure(target_table, target_col):
                _tgt = self._coercion_expression(target_table, target_col)
            else:
                _tgt = f'"{target_col}"'
        except Exception:
            _tgt = f'"{target_col}"'
        # If multiple aggregates, simplify to just the requested one
        agg_count = len(re.findall(r'\b(AVG|SUM|COUNT)\s*\(', sql, re.IGNORECASE))
        if agg_count > 1:
            return f'SELECT {agg_match}({_tgt}) AS result FROM {self._qtab(target_table)};'
        # Fix wrong column in the aggregate
        agg_pattern = re.compile(rf'{agg_match}\s*\(\s*["\']?(\w+)["\']?\s*\)', re.IGNORECASE)
        m = agg_pattern.search(sql)
        if m:
            sql_col = m.group(1).lower()
            if sql_col != target_col.lower():
                sql = sql[:m.start(1)] + _tgt + sql[m.end(1):]
        return sql

    # ---- _table_columns_norm (_table_columns_norm) ----

    def _table_columns_norm(self, table):
        """[(name, data_type)] for a table in EITHER mapping shape.

        The schema mapping may carry columns as a dict {name: {data_type}} (the
        normalized GoalAgent shape) or as a list [{column, data_type}] (raw
        extraction shape). The old deterministic templates only handled the
        list shape, so single-table dict-shape datasets silently fell through
        to None. Shared path: every template below uses this.
        """
        info = self.tables.get(table, {}) or {}
        cols_raw = info.get("columns", [])
        out = []
        if isinstance(cols_raw, dict):
            for name, meta in cols_raw.items():
                dt = ""
                if isinstance(meta, dict):
                    dt = str(meta.get("data_type", ""))
                out.append((str(name), dt))
        elif isinstance(cols_raw, list):
            for c in cols_raw:
                if isinstance(c, dict):
                    out.append((str(c.get("column", "")), str(c.get("data_type", ""))))
                elif isinstance(c, str):
                    out.append((str(c), ""))
        return [(n, dt) for n, dt in out if n]

    def _qtab(self, table):
        """Dialect-aware table reference; schema-qualified keys
        (sales.salesorderheader) become "sales"."salesorderheader" so the
        dot is a schema separator, never part of the name."""
        q = lambda n: self._quote(n, getattr(self, "dialect", "postgresql"))
        try:
            segs = [s for s in str(table or "").split(".") if s]
            if len(segs) > 1:
                return ".".join(q(s) for s in segs)
        except Exception:
            pass
        return self._quote(table, getattr(self, "dialect", "postgresql"))

    def _qcol(self, table, col):
        """Dialect-aware qualified column reference (quotes identifiers)."""
        q = lambda n: self._quote(n, getattr(self, "dialect", "postgresql"))
        if table:
            try:
                segs = [s for s in str(table).split(".") if s]
                if len(segs) > 1:
                    return ".".join([q(s) for s in segs] + [q(col)])
            except Exception:
                pass
            return f"{q(table)}.{q(col)}"
        return q(col)

    def _best_measure_for_table(self, table, ir=None):
        """Best SUM-able measure for `table`: IR-resolved metric wins, else the
        schema's best-guess measure column (dimension-like numerics skipped).
        Generic, never hardcoded to a column name."""
        if ir:
            for m in ir.get("metrics", []) or []:
                if m.get("resolved_table") == table and m.get("resolved_column"):
                    agg = str(m.get("aggregation", "") or "SUM").upper()
                    if agg in ("SUM", "AVG", "MIN", "MAX"):
                        return m["resolved_column"]
                if m.get("resolved_table") == table and m.get("resolved_expression"):
                    expr = str(m["resolved_expression"] or "")
                    if expr and expr.upper() != "COUNT(*)" and "(" not in expr:
                        part = expr.split(".")[-1]
                        if part:
                            return part
        try:
            picked = self._pick_measure_column(table)
            if picked:
                return picked
        except Exception:
            pass
        for name, _dt in self._table_columns_norm(table):
            low = name.lower()
            if low == "id" or low.endswith("_id"):
                continue
            try:
                if self._classify_column(table, name) == "numeric" \
                        and not self._is_dimension_like(table, name):
                    return name
            except Exception:
                continue
        return None

    def _groupable_columns(self, table):
        """Dimension-like columns usable as GROUP BY keys (text, boolean-as-int
        flags, dimension-like numerics such as store/holiday_flag, dates).

        Shared path: TEXT-only detection missed numeric dimensions (Walmart
        Store/Holiday_Flag are bigint) and picked the date instead of the
        requested store. PK/FK/identifier-like columns (any separator:
        order_id, "Order ID") are never automatic GROUP BY keys — grouping
        by a unique ID is only allowed when the user explicitly names it
        (handled by the explicit-breakdown path, not here).
        """
        info = self.tables.get(table, {}) or {}
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", []) or []}
        try:
            from core.data_intelligence import is_id_like_name as _id_like
        except Exception:
            _id_like = lambda n: str(n or "").lower() == "id" or str(n or "").lower().endswith("_id")  # noqa: E731
        out = []
        for name, _dt in self._table_columns_norm(table):
            low = name.lower()
            if name in pk or name in fk_cols or low == "id" or low.endswith("_id"):
                continue
            try:
                if _id_like(name):
                    continue
            except Exception:
                pass
            try:
                if self._is_dimension_like(table, name):
                    out.append(name)
                    continue
            except Exception:
                pass
            try:
                if self._classify_column(table, name) in ("text", "date"):
                    out.append(name)
            except Exception:
                out.append(name)
        return out

    def _date_col_for_table(self, table):
        """(col, data_type) of the table's date column, TEXT-date aware."""
        for name, dt in self._table_columns_norm(table):
            try:
                if self._classify_column(table, name) == "date":
                    return name, dt
            except Exception:
                continue
        for name, dt in self._table_columns_norm(table):
            d = (dt or "").lower()
            if any(k in d for k in ("date", "time", "timestamp")) or \
               any(k in name.lower() for k in ("date", "time", "month", "year", "day", "week", "quarter", "period")):
                return name, dt
        return None, None

    def _month_expr(self, table, col, dtype):
        """Dialect-aware month truncation, TEXT-date safe (DD-MM-YYYY + ISO)."""
        try:
            from core.data_intelligence import month_trunc_expression as _mte
            return _mte(self._qcol(table, col), dtype,
                        getattr(self, "dialect", "postgresql"))
        except Exception:
            q = self._qcol(table, col)
            dialect = getattr(self, "dialect", "postgresql")
            if dialect == "mysql":
                return f"DATE_FORMAT({q}, '%Y-%m')"
            if dialect == "sqlite":
                return f"strftime('%Y-%m', {q})"
            return f"date_trunc('month', {q})"

    # ---- _is_safe_numeric_measure (_is_safe_numeric_measure) ----

    def _is_safe_numeric_measure(self, table, col):
        """True when `col` on `table` is a verified SUM/AVG-able measure.

        Schema-agnostic: the declared data_type must carry a numeric
        fragment (INTEGER/DECIMAL/FLOAT/...) and must never be blob/binary/
        bytes/image-like; the column NAME must never carry a binary/photo/
        picture/counter token. Untyped sources ("", "unknown", "inferred")
        fall back to the shared classifier but keep the name exclusion.
        """
        if not table or not col:
            return False
        base = str(col).split(".")[-1].strip().strip("\"'`[]")
        if not base:
            return False
        if any(t in _UNSAFE_MEASURE_NAME_TOKENS
               for t in _fallback_name_tokens(base)):
            return False
        dtype = ""
        try:
            for name, dt in self._table_columns_norm(table):
                if name.lower() == base.lower():
                    dtype = str(dt or "")
                    break
        except Exception:
            dtype = ""
        dt = dtype.lower().strip()
        if any(k in dt for k in _UNSAFE_MEASURE_TYPE_TOKENS):
            return False
        if not dt or dt in _UNVERIFIED_DT:
            try:
                if self._classify_column(table, base) == "numeric" \
                        and not self._is_dimension_like(table, base):
                    return True
            except Exception:
                pass
            try:
                from core.text_numeric import is_measure_like_name as _mln0
                if _mln0(base) and self._is_coerced_numeric_measure(
                        table, base):
                    return True
            except Exception:
                pass
            try:
                return self._pick_measure_column(table) == base
            except Exception:
                return False
        if any(k in dt for k in ("bool", "bit")) and "double" not in dt \
                and "bigint" not in dt and "smallint" not in dt \
                and "int" not in dt.replace("point", ""):
            return False
        if any(k in dt for k in _SAFE_NUMERIC_TYPE_TOKENS):
            return True
        # Text-stored numerics (e.g. 'Rs.1,299' / '15%' in a TEXT column):
        # a measure-like name plus a bounded DISTINCT probe showing >=80%
        # coercible values makes this a COERCED measure (aggregates wrap the
        # coercion expression; GROUP BY/WHERE stay on the raw column).
        try:
            from core.text_numeric import is_measure_like_name as _mln
            if _mln(base) and self._is_coerced_numeric_measure(table, base):
                return True
        except Exception:
            pass
        return False

    # ---- text-numeric coercion (schema-agnostic, all 3 dialects) ----

    def _db_cache_uri(self):
        """Cache identity for text-numeric probes: (db_uri, table, column)."""
        try:
            return str(self.engine.url)
        except Exception:
            return str(getattr(self, "dialect", "") or "")

    def _coercion_budget_ok(self, consume=True):
        """At most 3 probed candidate columns per question (cache hits free)."""
        try:
            remaining = getattr(self, "_coercion_budget", 3)
        except Exception:
            remaining = 3
        if remaining is None:
            remaining = 3
        if remaining <= 0:
            return False
        if consume:
            try:
                self._coercion_budget = int(remaining) - 1
            except Exception:
                pass
        return True

    def _probe_text_numeric_column(self, table, col, limit=100):
        """Bounded read-only probe: SELECT DISTINCT col LIMIT `limit`.

        Returns the diagnostics record (cached per (db_uri, table, column)
        in-memory for the process). Never raises; failure means not
        coercible. Timeouts are respected via a tight statement timeout on
        postgres/mysql; the LIMIT bound keeps every dialect cheap.
        """
        try:
            from core.text_numeric import (
                diagnose_text_numeric as _diag,
                get_coercion_cache as _get,
                set_coercion_cache as _set,
            )
        except Exception:
            return None
        try:
            db_key = self._db_cache_uri()
        except Exception:
            db_key = ""
        try:
            hit = _get(db_key, table, col)
        except Exception:
            hit = None
        if isinstance(hit, dict):
            return hit
        if not self._coercion_budget_ok(consume=True):
            return None
        record = {"coercible": False, "sample": 0, "parsed": 0, "ratio": 0.0}
        try:
            qcol = self._qcol(table, col)
            qtab = self._qtab(table)
            sql = f"SELECT DISTINCT {qcol} FROM {qtab} LIMIT {int(limit)}"
            with self.engine.connect() as conn:
                rows = self._execute_guarded(
                    conn, sql, timeout_ms=5000).fetchall()
            values = [r[0] for r in (rows or [])]
            record = dict(_diag(values) or {})
        except Exception:
            record = {"coercible": False, "sample": 0, "parsed": 0, "ratio": 0.0}
        try:
            _set(db_key, table, col, record)
        except Exception:
            pass
        return record

    def _coerced_status_cached(self, table, col):
        """Tri-state cache read: True/False when probed, None when unknown."""
        try:
            from core.text_numeric import get_coercion_cache as _get
            hit = _get(self._db_cache_uri(), table, col)
        except Exception:
            hit = None
        if isinstance(hit, dict):
            return bool(hit.get("coercible"))
        return None

    def _is_coerced_numeric_measure(self, table, col):
        """True when a TEXT column is a probed-coercible numeric measure.

        Requires a measure-like name (generic fragments only) plus >=80% of
        a bounded DISTINCT sample parsing as float after currency/comma/
        percent/whitespace stripping. Cached per (db_uri, table, column).
        """
        if not table or not col:
            return False
        try:
            from core.text_numeric import is_measure_like_name as _mln
            if not _mln(col):
                return False
        except Exception:
            return False
        try:
            dtype = ""
            for name, dt in self._table_columns_norm(table):
                if name.lower() == str(col).split(".")[-1].lower():
                    dtype = str(dt or "")
                    break
        except Exception:
            dtype = ""
        dt = dtype.lower().strip()
        # Already-verified numerics never need coercion; binary/blob never
        # qualifies even when the name looks measure-like.
        if any(k in dt for k in _SAFE_NUMERIC_TYPE_TOKENS):
            return False
        if any(k in dt for k in _UNSAFE_MEASURE_TYPE_TOKENS):
            return False
        if dt and not any(k in dt for k in (
                "char", "text", "varchar", "citext", "uuid", "clob",
                "string", "enum", "name")) and dt not in _UNVERIFIED_DT:
            return False
        try:
            rec = self._probe_text_numeric_column(table, col)
        except Exception:
            return False
        return bool(isinstance(rec, dict) and rec.get("coercible"))

    def _coercion_expression(self, table, col):
        """Dialect-aware coercion expression for a text-numeric column."""
        try:
            from core.text_numeric import (
                coercion_sql as _cs, ensure_dialect_support as _eds)
            try:
                _eds(self.engine, getattr(self, "dialect", "postgresql"))
            except Exception:
                pass
            return _cs(getattr(self, "dialect", "postgresql"),
                       self._qcol(table, col))
        except Exception:
            return self._qcol(table, col)

    def _measure_agg_expr(self, table, col, agg="SUM"):
        """`AGG(col)` wrapping the coercion expression for coerced measures."""
        fn = str(agg or "SUM").upper()
        if fn not in ("SUM", "AVG", "MIN", "MAX"):
            fn = "SUM"
        try:
            if self._is_coerced_numeric_measure(table, col):
                return f"{fn}({self._coercion_expression(table, col)})"
        except Exception:
            pass
        return f"{fn}({self._qcol(table, col) if table else self._quoted(col)})"

    def _resolve_coerced_measure(self, metric, tables, goal_l, concept_tokens):
        """Bind one metric to a text-stored numeric column (bounded probe).

        Scans `tables` for measure-like TEXT columns (skipping PK/FK/id),
        probes at most the per-question budget (cache hits are free), and
        binds the best concept/goal-name match as a COERCED measure carrying
        its coercion expression. Returns (table, column) or None. Never
        raises; never invents values (unparsable cells stay NULL in SQL).
        """
        try:
            agg = str(metric.get("aggregation") or "").upper()
        except Exception:
            agg = ""
        if agg not in ("SUM", "AVG", "MIN", "MAX"):
            return None
        if metric.get("resolved_table"):
            return None
        try:
            from core.text_numeric import is_measure_like_name as _mln
        except Exception:
            return None
        goal_words = set()
        try:
            goal_words = set(re.findall(r"[a-z0-9]+", str(goal_l or "").lower()))
        except Exception:
            pass
        # Two-phase so the per-question probe budget (max 3) is spent on the
        # goal-relevant columns first: rank every measure-like candidate by
        # name (no probing), then probe only the top-ranked ones.
        ranked = []
        for table in tables or []:
            if table not in self.tables:
                continue
            info = self.tables.get(table, {}) or {}
            try:
                pk = set(info.get("primary_key", []) or [])
                fk_cols = {fk.get("column")
                           for fk in info.get("foreign_keys", []) or []}
            except Exception:
                pk, fk_cols = set(), set()
            try:
                cols = [n for n, _dt in self._table_columns_norm(table)]
            except Exception:
                continue
            for col in sorted(cols):
                low = col.lower()
                if col in pk or col in fk_cols or low == "id" \
                        or low.endswith("_id"):
                    continue
                try:
                    if not _mln(col):
                        continue
                except Exception:
                    continue
                # Skip already-probed non-coercible columns without spending.
                try:
                    cached = self._coerced_status_cached(table, col)
                except Exception:
                    cached = None
                if cached is False:
                    continue
                s = 0
                try:
                    col_tokens = set(self._normalize_tokens(col))
                except Exception:
                    col_tokens = set(re.split(r"[^a-z0-9]+", low))
                for tok in (concept_tokens or set()):
                    if tok in col_tokens:
                        s += 2
                    elif tok.endswith("s") and tok[:-1] in col_tokens:
                        s += 1
                try:
                    ctoks = _fallback_name_tokens(col)
                    goal_hit = 0
                    for _ct in ctoks:
                        if _ct in goal_words or _ct + "s" in goal_words \
                                or (_ct.endswith("s")
                                    and _ct[:-1] in goal_words):
                            goal_hit = 2
                            break
                    # Exact goal naming beats a compound sharing the word
                    # ("average rating" -> rating, not rating_count).
                    exact = 1 if low in goal_words else 0
                except Exception:
                    goal_hit, exact = 0, 0
                # Goal-named columns outrank pure concept-token matches
                # ("average discount" -> discount_percentage, not a price
                # column merely matching the generic "value/price" tokens).
                ranked.append(((goal_hit, exact, s, col.lower()), table, col))
        ranked.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
        best, best_key = None, None
        for _key, table, col in ranked:
            try:
                coerced = self._is_coerced_numeric_measure(table, col)
            except Exception:
                coerced = False
            if not coerced:
                continue
            # Rank order is fixed, so the first coercible hit is the best
            # goal-relevant candidate.
            best, best_key = (table, col), _key
            break
        if not best:
            return None
        table, col = best
        try:
            expr = self._coercion_expression(table, col)
        except Exception:
            expr = f"{table}.{col}"
        try:
            rec = self._probe_text_numeric_column(table, col)
        except Exception:
            rec = {}
        unit = None
        try:
            if isinstance(rec, dict):
                if rec.get("percent_unit"):
                    unit = "percent (values stay as-is, e.g. 15% -> 15)"
                elif rec.get("currency_unit"):
                    unit = "currency-denominated"
        except Exception:
            unit = None
        metric["resolved_table"] = table
        metric["resolved_column"] = col
        metric["resolved_expression"] = expr
        metric["coerced"] = True
        if unit:
            metric["coercion_unit"] = unit
        metric["confidence"] = 0.85
        metric["reason"] = (f"text column holds numbers "
                            f"(coerced {table}.{col} for {agg})")
        return best

    # ---- _verified_measure_for_table (_verified_measure_for_table) ----

    def _verified_measure_for_table(self, table, ir=None):
        """Best verified-safe measure column for `table` (IR wins, then schema).

        Returns the bare column name, or None when the table has no SUM/AVG-able
        numeric (binary/blob/photo/picture/counter columns never qualify).
        """
        if ir:
            for m in ir.get("metrics", []) or []:
                if m.get("resolved_table") != table:
                    continue
                agg = str(m.get("aggregation", "") or "SUM").upper()
                if agg not in ("SUM", "AVG", "MIN", "MAX"):
                    continue
                rc = m.get("resolved_column")
                if rc:
                    base = str(rc).split(".")[-1]
                    if self._is_safe_numeric_measure(table, base):
                        return base
                    try:
                        if m.get("coerced") and self._is_coerced_numeric_measure(
                                table, base):
                            return base
                    except Exception:
                        pass
        try:
            picked = self._pick_measure_column(table)
        except Exception:
            picked = None
        if picked and self._is_safe_numeric_measure(table, picked):
            return picked
        try:
            info = self.tables.get(table, {}) or {}
            pk = set(info.get("primary_key", []) or [])
            fk_cols = {fk.get("column")
                       for fk in info.get("foreign_keys", []) or []}
        except Exception:
            pk, fk_cols = set(), set()
        for name, _dt in self._table_columns_norm(table):
            low = name.lower()
            if name in pk or name in fk_cols or low == "id" \
                    or low.endswith("_id"):
                continue
            if not self._is_safe_numeric_measure(table, name):
                continue
            try:
                if self._is_dimension_like(table, name):
                    continue
            except Exception:
                pass
            return name
        # Coerced fallback: text-stored numerics are dimension-like by dtype
        # (never _is_measure_column), so they need their own pass guarded by
        # the measure-like name + bounded probe instead of the dtype gate.
        try:
            from core.text_numeric import is_measure_like_name as _mln2
        except Exception:
            return None
        for name, _dt in self._table_columns_norm(table):
            low = name.lower()
            if name in pk or name in fk_cols or low == "id" \
                    or low.endswith("_id"):
                continue
            try:
                if not _mln2(name):
                    continue
                if self._is_coerced_numeric_measure(table, name):
                    return name
            except Exception:
                continue
        return None

    # ---- _parse_top_n (_parse_top_n) ----

    def _parse_top_n(self, goal_text, default=10):
        """Parse N from a ranking question ("top 5", "bottom 3", "first 20").

        Schema-agnostic; clamps to [1, 1000]; default 10 when no N is stated.
        """
        try:
            g = str(goal_text or "").lower()
            m = re.search(
                r"\b(?:top|bottom|first|last|next|highest|lowest)\s+(\d+)\b", g)
            if not m:
                m = re.search(r"\b(?:limit|show(?:ing)?|list(?:ing)?)\s+(\d+)\b", g)
            if not m:
                m = re.search(r"\b(\d+)\s+(?:top|best|rows|records|items|results)\b", g)
            if m:
                return max(1, min(1000, int(m.group(1))))
        except Exception:
            pass
        return default

    # ---- _quote_computed_expression (_quote_computed_expression) ----

    def _quote_computed_expression(self, expr):
        """Quote table.column tokens inside a computed measure expression."""
        def _rep(m):
            return f"{self._qtab(m.group(1))}.{self._quote(m.group(2), getattr(self, 'dialect', 'postgresql'))}"
        try:
            return re.sub(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)",
                _rep, str(expr or ""))
        except Exception:
            return expr

    # ---- _join_from_for_tables (_join_from_for_tables) ----

    def _join_from_for_tables(self, tables, base):
        """Build FROM + JOIN SQL covering `tables` through real FK edges only.

        Starts from `base` (the measure's home table) and attaches only
        tables reachable via the deterministic join plan; unreachable tables
        are dropped (single-home-table fallback) rather than cross-joined.
        Every emitted JOIN carries a validated ON condition whose columns
        exist in the mapping. Returns (from_join_sql, included_tables).

        Generic multi-schema binding: table names resolve case-insensitively
        + bare-segment aware so search_path-bare or differently-cased names
        still bind; ambiguous bare names are dropped.
        """
        def _res(name):
            try:
                if name in self.tables:
                    return name
                if hasattr(self, "_resolve_table_key"):
                    return self._resolve_table_key(name)
            except Exception:
                pass
            return None
        wanted = []
        for t in (tables or []):
            r = _res(t)
            if r is not None and r not in wanted:
                wanted.append(r)
        base = _res(base) or (wanted[0] if wanted else None)
        if base is None:
            return "", []
        if not wanted:
            wanted = [base]
        try:
            plan = self._plan_joins(wanted)
            edges = list(plan.get("edges", []) or [])
            plan_nodes = set(plan.get("nodes", []) or []) | set(wanted)
        except Exception:
            edges = []
            plan_nodes = set(wanted)

        def _col_exists(table, col):
            try:
                names = {n.lower()
                         for n, _dt in self._table_columns_norm(table)}
                return str(col or "").lower() in names
            except Exception:
                return False

        valid = [e for e in edges
                 if e.get("left_table") in self.tables
                 and e.get("right_table") in self.tables
                 and _col_exists(e.get("left_table"), e.get("left_column"))
                 and _col_exists(e.get("right_table"), e.get("right_column"))]
        # Composite FKs share one table pair across several edges
        # (specialofferid + productid): group by pair so the JOIN below can
        # AND every condition. Emitting only the first silently fans out and
        # inflates aggregates. Generic (edge-driven, no table names).
        pair_edges: dict = {}
        for e in valid:
            _key = (e["left_table"], e["right_table"])
            pair_edges.setdefault(_key, []).append(e)
        included = [base]
        used_pairs = set()
        progressed = True
        while progressed:
            progressed = False
            for (a, b) in pair_edges:
                if (a, b) in used_pairs:
                    continue
                if (a in included) == (b in included):
                    continue
                nxt = b if a in included else a
                # Intermediate plan nodes (e.g. products between order_details
                # and categories) are valid stepping stones even when not in
                # `wanted`; tables outside the plan are never attached.
                if nxt not in plan_nodes or nxt in included:
                    continue
                used_pairs.add((a, b))
                included.append(nxt)
                progressed = True
        parts = [f"FROM {self._qtab(base)}"]
        joined = {base}
        for (a, b) in pair_edges:
            if (a, b) not in used_pairs:
                continue
            # Attach the side not yet in the query (both orders validated).
            nxt = b if a in joined else a
            try:
                _conds = " AND ".join(
                    f"{self._qcol(e['left_table'], e['left_column'])} = "
                    f"{self._qcol(e['right_table'], e['right_column'])}"
                    for e in pair_edges[(a, b)]
                )
            except Exception:
                continue
            parts.append(f"JOIN {self._qtab(nxt)} ON {_conds}")
            joined.add(nxt)
        return " ".join(parts), included

    # ---- _render_fallback_where (_render_fallback_where) ----

    def _render_fallback_where(self, filters, included):
        """Render IR filters whose columns resolve to `included` tables."""
        if not filters:
            return ""
        inc_lc = {str(t).lower(): t for t in included}
        conds = []
        for f in filters or []:
            col = f.get("column")
            if not col:
                continue
            if "." in str(col):
                tname, _, cname = str(col).partition(".")
            else:
                tname, cname = (included[0] if included else None), str(col)
            real_t = inc_lc.get(str(tname or "").lower())
            if real_t is None:
                continue
            try:
                names = {n.lower(): n
                         for n, _dt in self._table_columns_norm(real_t)}
            except Exception:
                continue
            real_c = names.get(str(cname or "").lower())
            if not real_c:
                continue
            op = str(f.get("operator") or "=").upper().strip() or "="
            if op not in ("=", "!=", "<>", ">", "<", ">=", "<=", "LIKE",
                          "ILIKE", "IS", "IS NOT"):
                op = "="
            val = f.get("value")
            if val is None:
                conds.append(
                    f"{self._qcol(real_t, real_c)} IS NULL"
                    if op == "IS" else
                    f"{self._qcol(real_t, real_c)} IS NOT NULL")
                continue
            v = str(val).strip()
            if (len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"')) \
                    or re.fullmatch(r"-?\d+(\.\d+)?", v):
                lit = v
            else:
                lit = "'" + v.replace("'", "''") + "'"
            conds.append(f"{self._qcol(real_t, real_c)} {op} {lit}")
        return ("WHERE " + " AND ".join(conds)) if conds else ""

    # ---- _ir_fallback_sql (_ir_fallback_sql) ----

    def _ir_fallback_sql(self, user_goal, join_path, ir=None, join_plan=None):
        """One IR-driven deterministic builder for every no-LLM path.

        Schema-agnostic (no table/column literals). Guarantees validation-safe
        SQL when tables exist:
          1. measure is verified numeric (never blob/binary/photo/picture);
          2. every emitted table joins through a real FK edge (else the
             single home table of the measure is used alone);
          3. the planned aggregation is preserved (SUM stays SUM, AVG stays
             AVG), including on filtered queries;
          4. ranking always emits ORDER BY + LIMIT N (N parsed, default 10);
          5. trends GROUP BY the datetime month truncation with ORDER BY;
          6. counts emit COUNT(*) with no measure.
        Returns the SQL string, or None only when no tables exist.
        """
        goal_text = str((ir or {}).get("original_goal")
                        or user_goal
                        or getattr(self, "_active_original_goal", "") or "")
        goal_l = goal_text.lower()
        def _res2(name):
            try:
                if name in self.tables:
                    return name
                if hasattr(self, "_resolve_table_key"):
                    return self._resolve_table_key(name)
            except Exception:
                pass
            return None
        tables = []
        for t in (join_path or []):
            r = _res2(t)
            if r is not None and r not in tables:
                tables.append(r)
        try:
            required = list(self._required_tables_from_ir(ir, join_path or [])) \
                if ir else list(tables)
        except Exception:
            required = list(tables)
        _req = []
        for t in required:
            r = _res2(t)
            if r is not None and r not in _req:
                _req.append(r)
        required = _req
        if not required:
            required = list(tables) or list(self.tables.keys())[:1]
        if not required:
            return None

        metrics = (ir or {}).get("metrics", []) or []
        dims = (ir or {}).get("dimensions", []) or []
        filters = (ir or {}).get("filters", []) or []
        time = (ir or {}).get("time") if ir else None
        ranking = (ir or {}).get("ranking") if ir else None
        if ranking is None:
            try:
                ranking = self._detect_ranking(goal_text)
            except Exception:
                ranking = None
        agg = (ir or {}).get("aggregation")
        primary = next((m for m in metrics if m.get("aggregation")), None) \
            or (metrics[0] if metrics else None)
        if not agg and primary:
            agg = primary.get("aggregation")
        if not agg:
            try:
                agg = self._detect_aggregation(goal_text)
            except Exception:
                agg = None
        agg = str(agg or "").upper() or None

        wants_count = (
            agg == "COUNT"
            or (primary and str(primary.get("aggregation") or "").upper() == "COUNT")
            or (primary and str(primary.get("resolved_expression") or "").upper() == "COUNT(*)")
        )
        if not wants_count and agg is None and re.search(
                r"\b(how many|number of|count of|count\b)", goal_l):
            wants_count = True

        # --- aggregate slots: EVERY resolved metric is emitted ---------------
        # One question can resolve several metrics (e.g. "total" + "sales" ->
        # SUM(store) + SUM(weekly_sales)); the semantic plan requires each
        # resolved column to be referenced, so each gets its own aggregate
        # slot with its OWN planned aggregation preserved (SUM stays SUM,
        # AVG stays AVG). Slots are (agg, table, column|None, expr|None).
        slots = []
        seen_slots = set()

        def _add_slot(slot_agg, slot_table, slot_col=None, slot_expr=None):
            key = (slot_agg, slot_table,
                   (slot_col or "").lower(), (slot_expr or "").lower())
            if key in seen_slots:
                return
            seen_slots.add(key)
            slots.append((slot_agg, slot_table, slot_col, slot_expr))

        if not wants_count:
            for m in metrics:
                m_agg = str(m.get("aggregation") or "").upper()
                if m_agg == "COUNT":
                    continue
                if m_agg not in ("SUM", "AVG", "MIN", "MAX"):
                    continue
                rt = m.get("resolved_table")
                if rt not in self.tables:
                    continue
                rc, rx = m.get("resolved_column"), m.get("resolved_expression")
                if rc:
                    base = str(rc).split(".")[-1]
                    if self._is_safe_numeric_measure(rt, base):
                        _add_slot(m_agg, rt, slot_col=base)
                        continue
                # A computed measure (quantity * unit_price) carries
                # operators but no parens: it must stay an expression and
                # never be truncated to its last dotted token ("unit_price").
                if rx and str(rx).upper() != "COUNT(*)" \
                        and any(op in str(rx) for op in ("*", "/", "+", "-")):
                    _add_slot(m_agg, rt, slot_expr=str(rx))
                    continue
                if rx and str(rx).upper() != "COUNT(*)" \
                        and "(" not in str(rx):
                    base = str(rx).split(".")[-1]
                    if self._is_safe_numeric_measure(rt, base):
                        _add_slot(m_agg, rt, slot_col=base)
                        continue
            if not slots:
                ordered = [t for t in required]
                for m in metrics:
                    rt = m.get("resolved_table")
                    if rt in ordered:
                        ordered = [rt] + [t for t in ordered if t != rt]
                for t in ordered:
                    col = self._verified_measure_for_table(t, ir)
                    if col:
                        _add_slot(agg if agg in ("SUM", "AVG", "MIN", "MAX")
                                  else "SUM", t, slot_col=col)
                        break
            if agg is None:
                agg = slots[0][0] if slots else "SUM"
            if agg == "COUNT":
                wants_count = True
        if wants_count:
            agg = "COUNT"
            slots = []
            # COUNT home: resolved fact table wins, else first required table.
            count_home = None
            if primary and primary.get("resolved_table") in self.tables:
                count_home = primary["resolved_table"]
            base_table = count_home or required[0]
        else:
            if not slots:
                # No safe numeric anywhere: COUNT(*) is the only valid math.
                wants_count, agg = True, "COUNT"
                base_table = required[0]
            else:
                base_table = slots[0][1]
        measure_table = None if wants_count else slots[0][1]
        measure_col = None if wants_count else slots[0][2]
        computed_expr = None if wants_count else slots[0][3]

        # --- join scope: tables reachable from the base through real edges --
        try:
            scope_plan = join_plan if (join_plan and join_plan.get("edges")) \
                else self._plan_joins(required)
        except Exception:
            scope_plan = {"edges": []}
        _edges = list((scope_plan or {}).get("edges", []) or [])

        def _edge_ok(e):
            try:
                return e.get("left_table") in self.tables \
                    and e.get("right_table") in self.tables
            except Exception:
                return False
        _edges = [e for e in _edges if _edge_ok(e)]
        # Reachable set from the base through plan edges. The walk traverses
        # intermediate plan nodes (never fabricated tables); required tables
        # with no join path are dropped (single-home-table fallback).
        try:
            _plan_nodes = set((scope_plan or {}).get("nodes", []) or [])
        except Exception:
            _plan_nodes = set()
        _walkable = set(required) | _plan_nodes
        reachable = {base_table}
        changed = True
        while changed:
            changed = False
            for e in _edges:
                a, b = e["left_table"], e["right_table"]
                if a in reachable and b not in reachable and b in _walkable:
                    reachable.add(b)
                    changed = True
                elif b in reachable and a not in reachable and a in _walkable:
                    reachable.add(a)
                    changed = True
        use_tables = [t for t in required if t in reachable]
        for t in [base_table] + list(_plan_nodes):
            if t in reachable and t not in use_tables:
                use_tables.append(t)
        use_tables = use_tables or [base_table]
        from_join, included = self._join_from_for_tables(use_tables, base_table)
        if not from_join:
            from_join = f"FROM {self._qtab(base_table)}"
            included = [base_table]

        where = self._render_fallback_where(filters, included)

        # Single-table convention (matches the legacy templates and test
        # contract): bare quoted columns; multi-table stays qualified.
        _single_tbl = len(included) == 1
        _dialect = getattr(self, "dialect", "postgresql")

        def _colref(t, c):
            if _single_tbl:
                return self._quote(c, _dialect)
            return self._qcol(t, c)

        # --- ranking / trend / grouping detection ---------------------------
        try:
            extremes = bool(self._is_extremes_goal(goal_text))
        except Exception:
            extremes = False
        rank_words = bool(re.search(
            r"\b(top|bottom|best|worst|highest|lowest|most|least|lead(ing|ers?)|"
            r"largest|smallest|rank)", goal_l))
        wants_ranking = bool(ranking) or rank_words
        trend_words = bool(re.search(
            r"\b(trend|over time|monthly|daily|weekly|yearly|by month|by year|"
            r"by day|growth|per month|per year|months?|decline|declined|"
            r"declining|decrease|decreased|decreasing|drop|dropped|"
            r"dropping|fall|fell|falling|falls)\b", goal_l))
        try:
            decline_words = bool(re.search(
                r"\b(decline|declined|declining|declines|decrease|decreased|"
                r"decreasing|decreases|drop|dropped|dropping|drops|fall|"
                r"fell|falling|falls)\b", goal_l))
        except Exception:
            decline_words = False
        try:
            ir_intent = (ir or {}).get("intent")
        except Exception:
            ir_intent = None
        wants_trend = bool(ir_intent == "trend" or trend_words
                            or (isinstance(time, dict) and time.get("granularity")))
        # DEFECT 2: month-over-month decline ("in which month did revenue
        # decline/drop/fall"): trend intent + decline words forces a monthly
        # GROUP BY even when the phrasing uses bare "month" instead of
        # "monthly" / "over time". Generic, no schema names.
        try:
            if decline_words and (ir_intent == "trend" or "month" in goal_l):
                wants_trend = True
        except Exception:
            pass
        by_words = bool(re.search(
            r"\b(by|per|versus|vs\.?|breakdown|distribution|across|among|"
            r"compare|comparison)\b", goal_l))
        limit_n = None
        if wants_ranking and not extremes:
            try:
                limit_n = (ranking or {}).get("limit") \
                    if isinstance(ranking, dict) else None
            except Exception:
                limit_n = None
            if not limit_n:
                limit_n = self._parse_top_n(goal_text, default=10)
        direction = "DESC"
        try:
            if isinstance(ranking, dict) and str(
                    ranking.get("direction") or "").lower() == "asc":
                direction = "ASC"
            elif any(k in goal_l for k in
                     ("bottom", "lowest", "least", "worst", "smallest")):
                direction = "ASC"
        except Exception:
            pass

        def _dim_usable(d):
            try:
                rt, rc = d.get("resolved_table"), d.get("resolved_column")
                if not rt or not rc or rt not in included:
                    return None
                base_c = str(rc).split(".")[-1]
                names = {n.lower(): n
                         for n, _dt in self._table_columns_norm(rt)}
                real = names.get(base_c.lower())
                return (rt, real) if real else None
            except Exception:
                return None

        # Goal-named dimensions win over subject-table labels: "how many
        # customers by city" must group by city, not customer_name. A dim
        # whose column tokens appear in the goal outranks a bare subject
        # label (concept == table name); ties keep IR order.
        try:
            _goal_words = set(re.findall(r"[a-z0-9]+", goal_l))
        except Exception:
            _goal_words = set()

        def _dim_score(d, got):
            _rt, _rc = got
            _toks = _fallback_name_tokens(_rc)
            # Table-name parts count too (countryregion.name for "country"):
            # synonym/substring-aware so compounds match. Generic, no names.
            _table_hit = False
            try:
                _stm = getattr(self, "_schema_token_matches_goal", None)
                if callable(_stm):
                    for _seg in str(_rt or "").lower().split("."):
                        for _p in _seg.split("_"):
                            if not _p:
                                continue
                            for _gw in _goal_words:
                                if _stm(_p, _gw):
                                    _table_hit = True
                                    break
                            if _table_hit:
                                break
                        if _table_hit:
                            break
            except Exception:
                _table_hit = False
            _exact = (_table_hit or any(t in _goal_words for t in _toks))
            _plural = any(t + "s" in _goal_words
                          or (t.endswith("s") and t[:-1] in _goal_words)
                          for t in _toks)
            # Exact name hits ("city" in "by city") outrank plural/stem hits
            # ("customer" via "customers"), so an explicitly requested
            # breakdown beats the subject-table label.
            _name_score = 2 if _exact else (1 if _plural else 0)
            _subject_label = str(d.get("concept") or "") == _rt
            return (_name_score,
                    0 if (_name_score > 0 or not _subject_label) else -1)

        # Breakdown grouping only answers a breakdown question: an overall
        # scalar ("total sales", "count of orders") must stay a single row
        # even when the IR carries subject-label dimensions — grouping it
        # would both over-group (semantic violation) and mis-answer. Only an
        # explicit breakdown signal (ranking, by/per/breakdown words, or a
        # comparison block) unlocks GROUP BY; a bare intent never does.
        _breakdown_ok = bool(
            wants_ranking or by_words
            or (ir or {}).get("comparison") is not None)
        group_table, group_col = None, None
        _best_score = None
        if _breakdown_ok:
            for d in dims:
                got = _dim_usable(d)
                if not got:
                    continue
                _sc = _dim_score(d, got)
                if _best_score is None or _sc > _best_score:
                    _best_score = _sc
                    group_table, group_col = got
        if group_col is None and (wants_ranking or by_words
                                   or (ir or {}).get("comparison") is not None):
            # Goal-named groupable column wins; else prefer low-cardinality
            # geo/category-like text dims over person names/IDs (generic:
            # name signals + bounded DISTINCT cardinality probe). Explicit
            # 'by X' already resolved above — this is the no-explicit
            # fallback so "best territory" never groups by customer name.
            def _rank_groupables(t, cands):
                gl = " " + goal_l + " "
                # Table-name parts also count: "revenue by country" names the
                # countryregion table even though its label column is just
                # "name". Synonym/substring-aware (compounds match). Generic.
                _tnamed = False
                try:
                    _stm2 = getattr(self, "_schema_token_matches_goal", None)
                    if callable(_stm2):
                        _gwords = set(re.findall(r"[a-z0-9]+", gl))
                        for _seg in str(t or "").lower().split("."):
                            for _p in _seg.split("_"):
                                if not _p:
                                    continue
                                for _gw in _gwords:
                                    if _stm2(_p, _gw):
                                        _tnamed = True
                                        break
                                if _tnamed:
                                    break
                            if _tnamed:
                                break
                except Exception:
                    _tnamed = False
                rows = []
                for c in cands:
                    parts = [p for p in c.lower().split("_") if p]
                    named = _tnamed or any(p in gl or p.rstrip("s") in gl
                                or (p + "s") in gl for p in parts)
                    try:
                        geo = bool(self._is_geo_cat_like(c)) if hasattr(
                            self, "_is_geo_cat_like") else False
                    except Exception:
                        geo = False
                    try:
                        person = bool(self._is_person_name_like(t, c)) if hasattr(
                            self, "_is_person_name_like") else False
                    except Exception:
                        person = False
                    try:
                        card = self._estimate_column_cardinality(t, c) if hasattr(
                            self, "_estimate_column_cardinality") else None
                    except Exception:
                        card = None
                    rows.append((0 if named else 1, 0 if geo else 1,
                                 1 if person else 0,
                                 card if isinstance(card, int) else 10 ** 9, c))
                rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
                return rows[0][4] if rows else None
            for t in [base_table] + [x for x in included if x != base_table]:
                try:
                    cands = self._groupable_columns(t)
                except Exception:
                    cands = []
                if cands:
                    group_table, group_col = t, _rank_groupables(t, cands)
                    break

        # --- datetime for trends --------------------------------------------
        date_table, date_col, date_dtype = None, None, None
        if wants_trend:
            tc = (time or {}).get("column") if isinstance(time, dict) else None
            if tc and "." in str(tc):
                t0, _, c0 = str(tc).partition(".")
                if t0 in included:
                    try:
                        names = {n.lower(): (n, dt)
                                 for n, dt in self._table_columns_norm(t0)}
                        if c0.lower() in names:
                            date_table, (date_col, date_dtype) = t0, names[c0.lower()]
                    except Exception:
                        pass
            if date_col is None:
                _date_search = list(dict.fromkeys(
                    [base_table] + [x for x in included if x != base_table]
                    + [x for x in (required or []) if x not in included]
                    + [x for x in (use_tables or []) if x not in included]))
                for t in _date_search:
                    try:
                        c, dt = self._date_col_for_table(t)
                    except Exception:
                        c, dt = None, None
                    if c:
                        date_table, date_col, date_dtype = t, c, dt
                        break
                if date_col is not None and date_table not in included:
                    # Pull the date table into the join scope so the trend
                    # GROUP BY is valid (join through real FK edges only).
                    try:
                        _fj, _inc = self._join_from_for_tables(
                            list(dict.fromkeys(list(use_tables) + [date_table])),
                            base_table)
                        if _inc and date_table in _inc:
                            from_join, included = _fj, _inc
                            _single_tbl = len(included) == 1
                    except Exception:
                        pass

        # --- scalar scope: ungrouped non-trend aggregates ---------------
        # A scalar COUNT/AVG/SUM must not fan out across dimension tables:
        # scope to the measure home + filter tables only (filters stay via
        # real FK joins; subject-label dims never force extra JOINs here).
        if group_col is None and not (wants_trend and date_col is not None):
            try:
                _ftables = []
                for _f in filters or []:
                    _c = str((_f or {}).get("column") or "")
                    if "." in _c and _c.split(".")[0] in self.tables \
                            and _c.split(".")[0] not in _ftables:
                        _ftables.append(_c.split(".")[0])
                _narrow = [base_table] + [t for t in _ftables if t != base_table]
                if set(_narrow) != set(included):
                    _fj, _inc = self._join_from_for_tables(_narrow, base_table)
                    if _inc and base_table in _inc:
                        from_join, included = _fj, _inc
                        _single_tbl = len(included) == 1
                        where = self._render_fallback_where(filters, included)
            except Exception:
                pass
        # --- prune the semantic plan to the emitted FROM/JOIN ------------
        # The fallback deliberately drops dimension tables that contribute
        # no column to SELECT/GROUP/WHERE (scalar scope above) and drops
        # unreachable tables with no FK path. The IR/plan must reflect the
        # emitted query, otherwise semantic validation demands a dropped
        # table (or a dropped join edge) that the SQL correctly omits.
        # Schema-agnostic: only intersects with actually-emitted tables;
        # the money-measure slots above are untouched so AVG/SUM stay
        # correct. Mutates the passed IR/plan in place so the retry loop
        # re-validates against the pruned plan.
        try:
            if ir is not None and included:
                _inc_set = set(included)
                _dims = ir.get("dimensions") or []
                _kept = [d for d in _dims
                         if d.get("resolved_table") in _inc_set
                         and d.get("resolved_column")]
                if len(_kept) != len(_dims):
                    ir["dimensions"] = _kept
                try:
                    _req = set(ir.get("required_tables") or [])
                    if _req - _inc_set:
                        ir["required_tables"] = sorted(_req & _inc_set) or sorted(_inc_set)
                except Exception:
                    pass
                for _p in (scope_plan, join_plan):
                    try:
                        if isinstance(_p, dict) and _p.get("edges"):
                            _edges0 = list(_p.get("edges") or [])
                            _edges1 = [e for e in _edges0
                                       if e.get("left_table") in _inc_set
                                       and e.get("right_table") in _inc_set]
                            if len(_edges1) != len(_edges0):
                                _p["edges"] = _edges1
                    except Exception:
                        pass
        except Exception:
            pass

        # --- measure SELECT fragments (every aggregation preserved) --------
        # One fragment per resolved metric slot so each planned aggregation
        # and column is referenced (SUM stays SUM, AVG stays AVG). The first
        # slot drives ORDER BY / scalar aliases.
        def _slot_expr(slot):
            s_agg, s_tab, s_col, s_expr = slot
            if s_agg not in ("SUM", "AVG", "MIN", "MAX"):
                s_agg = "SUM"
            if s_expr:
                return f"{s_agg}({self._quote_computed_expression(s_expr)})"
            # Coerced text-numerics aggregate the CAST expression, never the
            # bare TEXT column (GROUP BY/WHERE elsewhere stay raw).
            try:
                if self._is_coerced_numeric_measure(s_tab, s_col):
                    return f"{s_agg}({self._coercion_expression(s_tab, s_col)})"
            except Exception:
                pass
            return f"{s_agg}({_colref(s_tab, s_col)})"

        if wants_count:
            agg_exprs = ["COUNT(*)"]
        else:
            if agg not in ("SUM", "AVG", "MIN", "MAX"):
                agg = "SUM"
            agg_exprs = [_slot_expr(s) for s in slots]
        agg_expr = agg_exprs[0]
        value_list = ", ".join(
            f"{e} AS value{i + 1}" if len(agg_exprs) > 1 else f"{e} AS value"
            for i, e in enumerate(agg_exprs))

        # --- assemble --------------------------------------------------------
        if wants_trend and date_col is not None:
            try:
                month_expr = self._month_expr(date_table, date_col, date_dtype)
            except Exception:
                month_expr = self._qcol(date_table, date_col)
            sql = (f"SELECT {month_expr} AS period, {value_list} "
                   f"{from_join} "
                   f"{where + ' ' if where else ''}"
                   f"GROUP BY period ORDER BY period;")
            return " ".join(sql.split())
        if group_col is not None:
            gq = _colref(group_table, group_col)
            order = f"ORDER BY value {direction}" \
                if len(agg_exprs) == 1 else f"ORDER BY value1 {direction}"
            lim = f" LIMIT {int(limit_n)}" if limit_n else ""
            sql = (f"SELECT {gq}, {value_list} {from_join} "
                   f"{where + ' ' if where else ''}"
                   f"GROUP BY {gq} {order}{lim};")
            return " ".join(sql.split())
        if wants_ranking and limit_n and not wants_count:
            # Ranking with no dimension resolved: rank base rows by measure.
            order = f"ORDER BY {agg_expr} {direction}" \
                if agg in ("SUM", "AVG", "MIN", "MAX") else ""
            lim = f" LIMIT {int(limit_n)}"
            if date_col is not None:
                try:
                    month_expr = self._month_expr(
                        date_table, date_col, date_dtype)
                except Exception:
                    month_expr = self._qcol(date_table, date_col)
                sql = (f"SELECT {month_expr} AS period, {value_list} "
                       f"{from_join} {where + ' ' if where else ''}"
                       f"GROUP BY period ORDER BY period;")
            else:
                sql = (f"SELECT {value_list} {from_join} "
                       f"{where + ' ' if where else ''}"
                       f"{order}{lim};")
            return " ".join(sql.split())
        if wants_count and limit_n:
            # "top N by count" with no dimension: count rows, cap output.
            sql = (f"SELECT COUNT(*) AS value {from_join} "
                   f"{where + ' ' if where else ''}LIMIT {int(limit_n)};")
            return " ".join(sql.split())
        alias = {"SUM": "total", "AVG": "average", "MIN": "minimum",
                 "MAX": "maximum", "COUNT": "count"}.get(agg, "value")
        if len(agg_exprs) > 1:
            sql = (f"SELECT {value_list} {from_join} "
                   f"{where + ' ' if where else ''};")
        else:
            sql = (f"SELECT {agg_expr} AS {alias} {from_join} "
                   f"{where + ' ' if where else ''};")
        return " ".join(sql.split())

    # ---- _repair_incompatible_aggregation (_repair_incompatible_aggregation) ----

    def _repair_incompatible_aggregation(self, sql):
        """Deterministically fix SUM/AVG over text/datetime/binary columns.

        Swaps the offending column for the table's verified-safe numeric
        measure; when the table has none, rewrites that aggregate to
        COUNT(*). Returns (sql, changed); changed=True only when the result
        has no incompatible aggregations left.
        """
        try:
            bad = self._find_incompatible_aggregations(sql)
        except Exception:
            return sql, False
        if not bad or not sql:
            return sql, False
        out = sql
        try:
            alias_map = {a.lower(): t for a, t in
                         self._sql_structure(sql).get("alias_to_name", {}).items()}
        except Exception:
            alias_map = {}
        try:
            tables_lc = {t.lower(): t for t in self.tables}
        except Exception:
            tables_lc = {}
        for issue in bad:
            m = re.search(r"(SUM|AVG)\s+over\s+incompatible\s+\w+\s+column\s+"
                          r"([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)",
                          issue, re.IGNORECASE)
            if not m:
                continue
            fn, qual, col = m.group(1).upper(), m.group(2), m.group(3)
            table = alias_map.get(qual.lower(), tables_lc.get(qual.lower(), qual))
            if table not in self.tables:
                continue
            safe = self._verified_measure_for_table(table)
            if safe:
                try:
                    repl = self._measure_agg_expr(table, safe, fn)
                except Exception:
                    repl = f"{fn}({self._qcol(table, safe)})"
            else:
                repl = "COUNT(*)"
            pat = re.compile(
                rf"{fn}\s*\(\s*(?:[A-Za-z_][\w$]*\s*\.\s*)?"
                rf"[\"'`\[]?{re.escape(col)}[\"'`\]]?\s*\)",
                re.IGNORECASE)
            out, n = pat.subn(repl, out, count=1)
            if not n:
                return sql, False
        try:
            if not self._find_incompatible_aggregations(out):
                return out, True
        except Exception:
            pass
        return sql, False

    # ---- _deterministic_sql (_deterministic_sql) ----

    def _deterministic_sql(self, user_goal, join_path, kpi_map, ir=None,
                           join_plan=None):
        """Generate SQL deterministically (no LLM) by pattern-matching the
        user goal against the schema. Used as a fallback when the LLM fails."""
        # IR-driven builder first: it preserves the planned aggregation,
        # joins every required table through real FK edges, parses top-N and
        # groups trends by month — so the result passes validation with no
        # LLM repair. Legacy keyword templates below are the last resort.
        try:
            planned = self._ir_fallback_sql(user_goal, join_path, ir,
                                            join_plan)
        except Exception:
            planned = None
        if planned:
            return planned
        # Use the clean original goal from the IR (no conversation history
        # pollution) so multi-turn context doesn't trigger wrong patterns.
        goal = str((ir or {}).get("original_goal", user_goal) or user_goal or "")
        goal = goal.lower().strip()
        relevant = [t for t in join_path if t in self.tables]
        if not relevant:
            return None

        def _find_table(patterns):
            for pat in patterns:
                for t in relevant:
                    if pat in t.lower():
                        return t
            for pat in patterns:
                for t in self.tables:
                    if pat in t.lower():
                        return t
            return None

        def _find_col(table, patterns):
            for c, _dt in self._table_columns_norm(table):
                for pat in patterns:
                    if pat in c.lower():
                        return c
            return None

        def _revenue_expr(table):
            """Build revenue expression for a table (quoted, dialect-aware)."""
            cols = [c for c, _dt in self._table_columns_norm(table)]
            price = next((c for c in cols if "price" in c.lower() and "unit" in c.lower()), None)
            qty = next((c for c in cols if "quantity" in c.lower() or "qty" in c.lower()), None)
            disc = next((c for c in cols if "discount" in c.lower()), None)
            if price and qty:
                expr = f'{self._qcol(None, qty)} * {self._qcol(None, price)}'
                if disc:
                    expr += f' * (1 - COALESCE({self._qcol(None, disc)}, 0))'
                return expr, table
            return None, None

        def _measure_sum(table):
            """SUM(<best measure>) for `table`, quoted; None when no measure."""
            best = self._best_measure_for_table(table, ir)
            if best:
                try:
                    if self._is_coerced_numeric_measure(table, best):
                        from core.text_numeric import coercion_sql as _cs
                        bare = self._quote(best, getattr(
                            self, "dialect", "postgresql"))
                        return f'SUM({_cs(getattr(self, "dialect", "postgresql"), bare)})', best
                except Exception:
                    pass
                return f'SUM({self._qcol(None, best)})', best
            return None, None

        # --- COUNT pattern ---
        if any(kw in goal for kw in ("how many", "count", "number of")):
            # Dynamic: extract entity tokens from goal, match against schema table names
            goal_words = set(goal.split())
            entity_tokens = [w for w in goal_words
                            if len(w) > 2
                            and w not in ("how", "many", "count", "number", "the", "are",
                                          "there", "in", "of", "for", "with", "total", "sum")]
            t = None
            for tok in entity_tokens:
                t = _find_table([tok])
                if t:
                    break
            if not t:
                t = relevant[0]
            return f'SELECT COUNT(*) AS count FROM {self._qtab(t)};'

        # --- TOTAL/SUM pattern ---
        if any(kw in goal for kw in ("total", "sum", "overall")):
            # Resolved measure first (semantic roles); the synthesized
            # price×qty expression is only a fallback when no measure
            # resolves — never preferred on name fragments alone.
            for t in relevant:
                sum_expr, _col = _measure_sum(t)
                if sum_expr:
                    return f'SELECT {sum_expr} AS total FROM {self._qtab(t)};'
            for t in relevant:
                expr, _ = _revenue_expr(t)
                if expr:
                    return f'SELECT SUM({expr}) AS total FROM {self._qtab(t)};'

        # --- AVERAGE pattern ---
        if any(kw in goal for kw in ("average", "avg", "mean")):
            fact_tables = [t for t in relevant if self._is_pure_fact_table(t)]
            if not fact_tables:
                fact_tables = [t for t in relevant
                              if len(self.tables.get(t, {}).get("columns", {})) > 3
                              and any(c.lower().endswith("_id") for c in self.tables.get(t, {}).get("columns", {}))]
            if fact_tables:
                ft = fact_tables[0]
                ft_info = self.tables.get(ft, {})
                ft_cols = list(ft_info.get("columns", {}))
                # Find the FK in the fact table that joins to another relevant table
                fk_col = next((c for c in ft_cols if c.lower().endswith("_id") and c.lower() != "id"), None)
                if fk_col:
                    # Find the dimension table this FK references
                    ref_table = None
                    for t2 in relevant:
                        if t2 == ft:
                            continue
                        t2_info = self.tables.get(t2, {})
                        t2_pk = next((c for c in t2_info.get("columns", {})
                                      if c.lower().endswith("_id") or c.lower() == "id"), None)
                        if t2_pk and t2_pk.lower() == fk_col.lower():
                            ref_table = t2
                            break
                    if not ref_table:
                        ref_table = next((t2 for t2 in relevant if t2 != ft), None)
                    if ref_table:
                        expr, _ = _revenue_expr(ft)
                        ref_pk = next((c for c in self.tables.get(ref_table, {}).get("columns", {})
                                       if c.lower().endswith("_id") or c.lower() == "id"), None)
                        if expr and ref_pk:
                            return (f'SELECT SUM({expr}) / COUNT(DISTINCT {self._qcol(ref_table, ref_pk)}) AS average '
                                    f'FROM {self._qtab(ft)} JOIN {self._qtab(ref_table)} ON {self._qcol(ref_table, ref_pk)} = {self._qcol(ft, fk_col)};')
            for t in relevant:
                # Goal-specific: try to find the column named in the user's question
                all_cols = {c.lower(): c for c, _dt in self._table_columns_norm(t)}
                goal_words = goal.replace("average", "").replace("avg", "").replace("mean", "").replace("of", "").replace("the", "").split()
                target_col = None
                for w in goal_words:
                    w = w.strip()
                    if len(w) >= 3 and w in all_cols:
                        target_col = all_cols[w]
                        break
                if target_col:
                    try:
                        _tgt_expr = self._measure_agg_expr(t, target_col, "AVG").replace(
                            f"{self._qtab(t)}.", "")
                    except Exception:
                        _tgt_expr = f'AVG({self._qcol(None, target_col)})'
                    return f'SELECT {_tgt_expr} AS average FROM {self._qtab(t)};'
                sum_expr, _col = _measure_sum(t)
                if sum_expr:
                    best = self._best_measure_for_table(t, ir)
                    try:
                        _best_expr = self._measure_agg_expr(t, best, "AVG").replace(
                            f"{self._qtab(t)}.", "")
                    except Exception:
                        _best_expr = f'AVG({self._qcol(None, best)})'
                    return f'SELECT {_best_expr} AS average FROM {self._qtab(t)};'

        # --- TREND/TIME SERIES pattern (TEXT-date safe) ---
        if any(kw in goal for kw in ("trend", "over time", "monthly", "daily", "weekly", "yearly", "by month", "by year", "by day")):
            for t in relevant:
                date_col, date_dtype = self._date_col_for_table(t)
                if not date_col:
                    continue
                month_expr = self._month_expr(t, date_col, date_dtype)
                sum_expr, _col = _measure_sum(t)
                if sum_expr:
                    return (f"SELECT {month_expr} AS period, "
                            f'{sum_expr} AS value '
                            f'FROM {self._qtab(t)} '
                            f'GROUP BY period ORDER BY period;')
                expr, _ = _revenue_expr(t)
                if expr:
                    return (f"SELECT {month_expr} AS period, "
                            f'SUM({expr}) AS value '
                            f'FROM {self._qtab(t)} '
                            f'GROUP BY period ORDER BY period;')
                # No measure: count rows over time (still TEXT-date safe).
                return (f"SELECT {month_expr} AS period, "
                        f'COUNT(*) AS value '
                        f'FROM {self._qtab(t)} '
                        f'GROUP BY period ORDER BY period;')

        def _resolve_group_col(t):
            """Goal-named groupable column wins; else first groupable.

            IR dimensions pointing at keys/identifiers (PK, order_id,
            "Order ID") are skipped: grouping by a unique ID answers no
            breakdown question. Explicit user-named ID columns still work
            through the dedicated explicit-breakdown templates below.
            """
            try:
                from core.data_intelligence import is_id_like_name as _id_like
            except Exception:
                _id_like = lambda n: False  # noqa: E731
            try:
                dims = (ir or {}).get("dimensions") or []
                pk = set((self.tables.get(t, {}) or {}).get("primary_key", []) or [])
                for d in dims:
                    col = d.get("resolved_column")
                    if d.get("resolved_table") == t and col:
                        if col in pk:
                            continue
                        try:
                            if _id_like(col):
                                continue
                        except Exception:
                            pass
                        return col
            except Exception:
                pass
            groupables = self._groupable_columns(t)
            if not groupables:
                return None
            gl = " " + goal + " "
            for c in groupables:
                parts = [p for p in c.lower().split("_") if p]
                if any(p in gl or p.rstrip("s") in gl or (p + "s") in gl for p in parts):
                    return c
            return groupables[0]

        # --- GROUP BY / BY pattern ---
        if any(kw in goal for kw in (" by ", " per ", "breakdown", "distribution")):
            for t in relevant:
                cat = _resolve_group_col(t)
                if not cat:
                    continue
                sum_expr, _col = _measure_sum(t)
                if sum_expr:
                    return (f'SELECT {self._qcol(t, cat)}, {sum_expr} AS value '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY value DESC;')
                expr, _ = _revenue_expr(t)
                if expr:
                    return (f'SELECT {self._qcol(t, cat)}, SUM({expr}) AS value '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY value DESC;')
                return (f'SELECT {self._qcol(t, cat)}, COUNT(*) AS value '
                        f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY value DESC;')

        # --- RANKING pattern ---
        if any(kw in goal for kw in ("top", "best", "highest", "most", "leading", "largest")):
            for t in relevant:
                cat = _resolve_group_col(t)
                if not cat:
                    continue
                sum_expr, _col = _measure_sum(t)
                if sum_expr:
                    return (f'SELECT {self._qcol(None, cat)}, {sum_expr} AS value '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(None, cat)} ORDER BY value DESC LIMIT 10;')
                expr, _ = _revenue_expr(t)
                if expr:
                    return (f'SELECT {self._qcol(None, cat)}, SUM({expr}) AS value '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(None, cat)} ORDER BY value DESC LIMIT 10;')
                return (f'SELECT {self._qcol(None, cat)}, COUNT(*) AS value '
                        f'FROM {self._qtab(t)} GROUP BY {self._qcol(None, cat)} ORDER BY value DESC LIMIT 10;')

        # --- COMPARISON pattern (incl. holiday vs non-holiday) ---
        if any(kw in goal for kw in ("compare", "vs", "versus", "between")):
            for t in relevant:
                cat = _resolve_group_col(t)
                if not cat:
                    continue
                sum_expr, _col = _measure_sum(t)
                if sum_expr:
                    return (f'SELECT {self._qcol(t, cat)}, {sum_expr} AS value '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY value DESC;')
                expr, _ = _revenue_expr(t)
                if expr:
                    return (f'SELECT {self._qcol(t, cat)}, SUM({expr}) AS value '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY value DESC;')
                return (f'SELECT {self._qcol(t, cat)}, COUNT(*) AS value '
                        f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY value DESC;')

        # --- ANTI-JOIN pattern ---
        if any(kw in goal for kw in ("never", "without", "not sold", "no orders", "no sales", "no records")):
            fact_tables = [t for t in relevant if self._is_pure_fact_table(t)]
            if fact_tables and len(relevant) >= 2:
                dim_table = [t for t in relevant if t not in fact_tables]
                if dim_table:
                    dt, ft = dim_table[0], fact_tables[0]
                    dt_info = self.tables.get(dt, {})
                    ft_info = self.tables.get(ft, {})
                    dt_pk = next((c for c in dt_info.get("columns", {})
                                   if c.lower().endswith("_id") or c.lower() == "id"), None)
                    ft_fk = next((c for c in ft_info.get("columns", {})
                                   if c.lower() == f"{dt.lower()}_id" or c.lower().endswith("_id")), None)
                    if dt_pk and ft_fk:
                        return (f'SELECT t1.* FROM {self._qtab(dt)} t1 '
                                f'LEFT JOIN {self._qtab(ft)} t2 ON t1.{self._qcol(None, dt_pk)} = t2.{self._qcol(None, ft_fk)} '
                                f'WHERE t2.{self._qcol(None, ft_fk)} IS NULL LIMIT 50;')

        # --- PERCENTAGE pattern ---
        if any(kw in goal for kw in ("percentage", "percent", "share", "proportion")):
            for t in relevant:
                cat = _resolve_group_col(t)
                if not cat:
                    continue
                expr, _ = _revenue_expr(t)
                sum_expr, _col = _measure_sum(t)
                agg = expr or (sum_expr.replace("SUM(", "", 1)[:-1] if sum_expr else None)
                if cat and agg:
                    return (f'SELECT {self._qcol(t, cat)}, '
                            f'SUM({agg}) * 100.0 / SUM(SUM({agg})) OVER() AS percentage '
                            f'FROM {self._qtab(t)} GROUP BY {self._qcol(t, cat)} ORDER BY percentage DESC;')

        # --- BARE MEASURE fallback ("What are my sales?") ---
        # A sales/revenue-flavoured question with no other pattern still needs
        # SUM(main measure), never None and never SELECT *. Shared path: the IR
        # resolved metric wins so Walmart binds weekly_sales on every dialect.
        try:
            dims = (ir or {}).get("dimensions") or []
        except Exception:
            dims = []
        if not dims:
            for t in relevant:
                sum_expr, _col = _measure_sum(t)
                if sum_expr and any(kw in goal for kw in (
                        "sale", "revenue", "profit", "amount", "total", "value",
                        "quantity", "qty", "price", "cost", "weekly")):
                    return f'SELECT {sum_expr} AS total FROM {self._qtab(t)};'

        return None

    # ---- _semantic_validate_sql (_semantic_validate_sql) ----

    def _semantic_validate_sql(self, ir, sql, join_path, join_plan=None):
        """Semantic SQL validation (spec §9/§10): the final SQL must actually
        implement the resolved Goal IR â€” required tables in FROM, measures
        aggregated, filters/time/ranking present â€” and must respect the
        explicit deterministic join plan.

        Returns (issues, warnings). Empty issues = PASS. Deterministic; the
        LLM is NOT the authority on the plan. Runs after every repair so a
        repaired query is re-validated before execution.
        """
        issues = []
        warnings = []
        if not ir or not sql:
            return issues, warnings
        # Defense in depth (multi-turn poisoning): a COUNT intent whose every
        # metric is unresolvable (e.g. an AVG leaked from history that binds
        # to no column) degrades to the COUNT(*) success path instead of
        # hard-failing. Other intents' validation is untouched.
        try:
            _is_count = (
                str(ir.get("aggregation") or "").upper() == "COUNT"
                or str(ir.get("operation") or "").lower() == "count"
                or any(str(m.get("aggregation") or "").upper() == "COUNT"
                       for m in (ir.get("metrics") or []))
            )
            _metrics = list(ir.get("metrics") or [])
            if _is_count and _metrics and all(
                    not m.get("resolved_table")
                    and not m.get("resolved_expression")
                    for m in _metrics):
                _home = next((t for t in (join_path or []) if t in self.tables),
                             None)
                if _home is None:
                    try:
                        _home = next(iter(self.tables))
                    except Exception:
                        _home = None
                ir["aggregation"] = "COUNT"
                ir["operation"] = "count"
                ir["metrics"] = [{
                    "concept": "row count",
                    "aggregation": "COUNT",
                    "expression": None,
                    "resolved_expression": "COUNT(*)",
                    "resolved_table": _home,
                    "resolved_column": None,
                    "inferred": True,
                    "confidence": 0.6,
                    "reason": ("COUNT intent with no resolvable measure; "
                               "degraded to COUNT(*) (defense in depth)"),
                }]
        except Exception:
            pass
        struct = self._sql_structure(sql)
        in_from = {t.lower() for t in struct["in_from"]}
        sql_lower = sql.lower()
        sql_upper = sql.upper()

        required = set()
        for m in ir.get("metrics", []):
            if m.get("resolved_table"):
                required.add(m["resolved_table"])
        for d in ir.get("dimensions", []):
            if d.get("resolved_table"):
                required.add(d["resolved_table"])
        if not required and join_path:
            required.add(join_path[0])
        # Schema-alias aware: required 'sales.order' is satisfied by bare
        # 'order' in SQL (search_path resolves it) and vice versa.
        _present = set(in_from)
        try:
            for _t in list(in_from):
                if "." in _t:
                    _present.add(_t.split(".")[-1])
        except Exception:
            pass
        missing = [t for t in sorted(required)
                   if t.lower() not in _present
                   and str(t).split(".")[-1].lower() not in _present]
        if missing:
            issues.append(
                f"required table(s) {', '.join(missing)} are missing from FROM/JOIN"
            )

        wants_count = ir.get("aggregation") == "COUNT" or any(
            str(m.get("aggregation", "")).upper() == "COUNT"
            for m in ir.get("metrics", [])
        )
        if wants_count and "COUNT(" not in sql_upper:
            issues.append("the goal asks for a count but the SQL has no COUNT(...)")
        elif ir.get("aggregation") and not re.search(
                r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(", sql_upper):
            issues.append(
                f"the goal requires {ir['aggregation']} aggregation but the SQL "
                "has no aggregate function"
            )

        for m in ir.get("metrics", []):
            agg = m.get("aggregation")
            col = m.get("resolved_column")
            if not agg:
                continue
            if agg.upper() == "COUNT":
                if "COUNT(" not in sql_upper:
                    issues.append(
                        "the resolved metric requires a COUNT of records "
                        "but the SQL has no COUNT(...)"
                    )
                continue
            if f"{agg.upper()}(" not in sql_upper:
                issues.append(
                    f"the resolved metric requires {agg.upper()} over "
                    f"{m.get('resolved_expression')} but the SQL has no {agg.upper()}"
                )
            elif col and col.lower() not in sql_lower:
                # Coerced text-numeric measures are referenced via their
                # coercion expression (SUM(CAST(...col...))), not the bare
                # column: match by expression-in-SQL instead.
                try:
                    from goal_agent.sql_contract import (
                        sql_references_coerced_measure as _coerced_ref)
                    _rx = str(m.get("resolved_expression") or "")
                    if (m.get("coerced") or "CAST(" in _rx.upper()) and _coerced_ref(
                            sql, m.get("resolved_table"), col, agg):
                        continue
                except Exception:
                    pass
                issues.append(
                    f"the resolved measure column {m.get('resolved_expression')} "
                    "is not referenced by the SQL"
                )

        where_l = self._where_clause(sql).lower()
        for f in ir.get("filters", []):
            col = f.get("column")
            if not col:
                continue
            token = col.split(".")[-1].lower()
            if token and token not in where_l:
                issues.append(
                    f"the requested filter on {col} is not applied in the SQL WHERE clause"
                )

        # A bare period granularity ("weekly" in "weekly sales", "monthly
        # trend" with no year/value) is a grouping hint, not a WHERE constraint:
        # it must not fail a scalar aggregate whose measure name merely
        # contains the period word. Enforce the date-column requirement only
        # when the goal carries an actual time constraint (a value/year or an
        # explicit date filter); otherwise record a warning.
        if ir.get("time") and ir["time"].get("column"):
            date_token = ir["time"]["column"].split(".")[-1].lower()
            has_time_constraint = (
                ir["time"].get("value") is not None
                or any(f.get("column") for f in ir.get("filters", []))
            )
            if date_token and date_token not in sql_lower:
                if has_time_constraint:
                    issues.append(
                        "the requested time constraint is not applied to any date column"
                    )
                else:
                    warnings.append(
                        "the goal hints at a time granularity but the SQL does "
                        "not reference the date column"
                    )

        if ir.get("ranking"):
            if ir["ranking"].get("limit") is not None:
                if not re.search(r"\bLIMIT\s+\d+", sql_upper):
                    issues.append(
                        "the goal asks for a top/bottom N but the SQL has no LIMIT"
                    )
            if not re.search(r"\bORDER\s+BY\b", sql_upper):
                issues.append(
                    "the goal asks for a ranked result but the SQL has no ORDER BY"
                )

        if join_plan and join_plan.get("edges"):
            def _bare(t):
                try:
                    return str(t or "").split(".")[-1].lower()
                except Exception:
                    return str(t or "").lower()
            joined_pairs = set()
            for j in self._extract_sql_joins(sql):
                joined_pairs.add((j["left_table"].lower(), j["right_table"].lower()))
                joined_pairs.add((j["right_table"].lower(), j["left_table"].lower()))
                joined_pairs.add((_bare(j["left_table"]), _bare(j["right_table"])))
                joined_pairs.add((_bare(j["right_table"]), _bare(j["left_table"])))
            plan_nodes = {t.lower() for t in join_plan.get("nodes", [])}
            for e in join_plan["edges"]:
                pair = (e["left_table"].lower(), e["right_table"].lower())
                bare_pair = (_bare(e["left_table"]), _bare(e["right_table"]))
                if pair not in joined_pairs and bare_pair not in joined_pairs:
                    issues.append(
                        f"the planned join {e['left_table']}.{e['left_column']} = "
                        f"{e['right_table']}.{e['right_column']} is missing from the SQL"
                    )
            try:
                _pn_bare = {str(_pn).split(".")[-1].lower() for _pn in plan_nodes}
            except Exception:
                _pn_bare = set()
            extra = [t for t in in_from
                     if t not in plan_nodes
                     and str(t).split(".")[-1].lower() not in _pn_bare]
            if extra:
                warnings.append(
                    f"SQL uses table(s) {', '.join(sorted(extra))} outside the "
                    "deterministic join plan"
                )

        # ---- GROUP-BY granularity (spec §12 / Fix 3: AOV over-grouping) ----
        # A scalar overall aggregate (e.g. "average order value", "total sales")
        # must yield a single row: the outer SELECT must NOT carry a GROUP BY.
        # An LLM frequently over-groups AOV (e.g. AVG(total) GROUP BY customer_id),
        # splitting the single scalar metric into per-group rows and returning the
        # wrong answer. Flag it so the query is repaired to a single-row aggregate.
        #
        # Per-entity / "top N" / "by X" / "per X" / comparison goals are NOT
        # flagged: they carry intent ranking/comparison/distribution/trend, a
        # ranking block, a comparison block, or explicit dimensions, so a GROUP BY
        # is correct for them.
        #
        # Only the OUTERMOST (parenthesis-depth-0) GROUP BY is considered, so a
        # correct per-order AOV subquery
        #   SELECT AVG(v) FROM (SELECT SUM(x) AS v FROM ... GROUP BY order_id) t
        # is not mis-flagged -- its GROUP BY lives inside a subquery.
        _gb_match = re.search(r"\bGROUP\s+BY\s+", sql, re.IGNORECASE)
        _outer_gb = []
        if _gb_match:
            _depth = sql[:_gb_match.start()].count("(") - sql[:_gb_match.start()].count(")")
            if _depth == 0:
                _outer_gb = self._extract_sql_group_by(sql)
        if (
            _outer_gb
            and ir.get("aggregation")
            and ir.get("intent") == "summary"
            and not ir.get("ranking")
            and not ir.get("comparison")
            and not ir.get("dimensions")
            and re.search(r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(", sql_upper)
        ):
            issues.append(
                "the goal asks for an overall aggregate (no breakdown) but the SQL "
                "groups by "
                + ", ".join(_outer_gb)
                + " (over-grouping: a single aggregate metric must not be split "
                "into per-group rows)"
            )

        # Trend goals REQUIRE a datetime GROUP BY: a scalar aggregate with
        # no GROUP BY can never answer "how has X changed over time".
        try:
            _is_trend = (str(ir.get("intent") or "").lower() == "trend"
                         or str(ir.get("operation") or "").lower() == "trend")
            if _is_trend and re.search(
                    r"\b(SUM|AVG|MIN|MAX|COUNT)\s*\(", sql_upper):
                _gb = re.search(r"\bGROUP\s+BY\b", sql_upper)
                _date_tok = ""
                try:
                    _date_tok = str((ir.get("time") or {}).get("column") or "").split(".")[-1].lower()
                except Exception:
                    _date_tok = ""
                _dim_date = False
                try:
                    for d in ir.get("dimensions", []) or []:
                        if self._classify_column(
                                d.get("resolved_table"), d.get("resolved_column")) == "date":
                            _dim_date = True
                            break
                except Exception:
                    pass
                if not _gb and not _dim_date and _date_tok not in sql_lower:
                    issues.append(
                        "the goal asks for a trend over time but the SQL has "
                        "no GROUP BY on a date column (scalar aggregates are "
                        "invalid for trends)"
                    )
        except Exception:
            pass

        for w in self._detect_fan_out(ir, join_plan, in_from):
            warnings.append(w)
        return issues, warnings

    # ---- _detect_fan_out (_detect_fan_out) ----

    def _detect_fan_out(self, ir, join_plan, in_from):
        """Spec §18: warn when a SUM/AVG metric is summed over a table that is
        an ANCESTOR of the row-grain table. In a many-to-one chain such as
        customers -> orders -> order_items, summing an order_items grain by a
        customers-level value would multiply it by the number of matching child
        rows (row fan-out / double counting). Deterministic; warning only.
        """
        if not ir or not join_plan or not join_plan.get("edges"):
            return []
        metrics = [m for m in ir.get("metrics", []) if m.get("resolved_table")]
        if not metrics:
            return []
        sums = [
            m for m in metrics
            if str(m.get("aggregation", "")).upper() in ("SUM", "AVG")
        ]
        if not sums:
            return []
        grain = None
        for d in ir.get("dimensions", []):
            if d.get("resolved_table"):
                grain = d["resolved_table"]
                break
        if not grain:
            grain = next(iter(in_from)) if in_from else None
        if not grain:
            return []
        child_to_parent = {}
        for e in join_plan["edges"]:
            if e.get("cardinality") == "many_to_one":
                child_to_parent.setdefault(
                    e["left_table"].lower(), set()
                ).add(e["right_table"].lower())
        if not child_to_parent:
            return []
        warnings = []
        for m in sums:
            mt = m["resolved_table"].lower()
            if mt == grain.lower():
                continue
            if mt not in in_from or grain.lower() not in in_from:
                continue
            # Walk up from the grain table along many-to-one (child -> parent)
            # edges. If we reach the metric's table, the metric table is an
            # ancestor of the grain -> row fan-out is possible.
            seen, stack = {grain.lower()}, [grain.lower()]
            reached = False
            while stack:
                cur = stack.pop()
                for parent in child_to_parent.get(cur, []):
                    if parent == mt:
                        reached = True
                        break
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
                if reached:
                    break
            if reached:
                warnings.append(
                    f"possible row fan-out: the {m.get('aggregation')} over "
                    f"{m.get('resolved_expression') or m.get('resolved_column')} is "
                    f"taken at table {m['resolved_table']}, an ancestor of the row "
                    f"grain table {grain}; the metric may be multiplied across "
                    "matching child rows"
                )
        return warnings

    # ---- _fallback_sql (_fallback_sql) ----

    def _fallback_sql(self, join_path, ir=None, join_plan=None,
                        user_goal=None):
        """Smart template-based fallback when the LLM fails.
        Tries to match the user goal to common SQL patterns."""
        # IR-driven builder first: always emits validation-safe SQL (verified
        # numeric measure, real FK joins, preserved aggregation, LIMIT for
        # top-N, month GROUP BY for trends, COUNT(*) for counts). The legacy
        # keyword templates below only run when no plan context exists.
        try:
            planned = self._ir_fallback_sql(
                user_goal if user_goal is not None
                else getattr(self, "_active_original_goal", ""),
                join_path, ir, join_plan)
        except Exception:
            planned = None
        if planned:
            return planned
        relevant = [t for t in join_path if t in self.tables]
        if not relevant:
            if self.dialect == "sqlite":
                return "SELECT name FROM sqlite_master WHERE type='table' LIMIT 10;"
            return "SELECT table_name FROM information_schema.tables LIMIT 10;"

        goal = (user_goal if user_goal is not None
                else getattr(self, "_active_original_goal", "") or "").lower()

        NUMERIC_TYPES = {"int", "integer", "bigint", "decimal", "numeric", "real",
                        "double precision", "float", "money", "smallint"}
        TEXT_TYPES = {"char", "character", "varchar", "character varying",
                      "text", "citext", "string", "uuid", "clob", "enum", "name"}

        def _safe_num(table, col):
            try:
                return self._is_safe_numeric_measure(table, col)
            except Exception:
                return False

        # Anti-join: "never ordered", "without orders", "not sold"
        if any(kw in goal for kw in ("never", "without", "not sold", "not ordered", "no orders", "no sales")):
            fact_tables = [t for t in relevant if self._is_pure_fact_table(t)]
            if fact_tables and len(relevant) >= 2:
                dim_table = [t for t in relevant if t not in fact_tables]
                if dim_table:
                    dt = dim_table[0]
                    ft = fact_tables[0]
                    dt_info = self.tables.get(dt, {})
                    ft_info = self.tables.get(ft, {})
                    dt_pk = next((c for c in dt_info.get("columns", {})
                                   if c.lower().endswith("_id") or c.lower() == "id"), None)
                    ft_fk = next((c for c in ft_info.get("columns", {})
                                   if c.lower() == f"{dt.lower()}_id" or c.lower().endswith("_id")), None)
                    if dt_pk and ft_fk:
                        return (f'SELECT t1.* FROM {self._qtab(dt)} t1 '
                                f'LEFT JOIN {self._qtab(ft)} t2 ON t1."{dt_pk}" = t2."{ft_fk}" '
                                f'WHERE t2."{ft_fk}" IS NULL LIMIT 50;')

        # COUNT pattern
        if any(kw in goal for kw in ("how many", "count", "number of")):
            table = relevant[0]
            return f'SELECT COUNT(*) AS count FROM {self._qtab(table)};'

        # TOTAL/SUM pattern
        if any(kw in goal for kw in ("total", "sum", "overall")):
            for t in relevant:
                num_cols = [c for c, dt in self._table_columns_norm(t)
                            if dt.upper() in NUMERIC_TYPES or any(k in dt.lower() for k in ("int", "decimal", "numeric", "float", "real", "double", "money"))
                            and not c.lower().endswith("_id") and c.lower() != "id"]
                num_cols = [c for c in num_cols if _safe_num(t, c)]
                if num_cols:
                    return (f'SELECT {self._measure_agg_expr(t, num_cols[0], "SUM")}'
                            f' AS total FROM {self._qtab(t)};')

        # AVERAGE pattern — try to match the specific column the user asked about
        if any(kw in goal for kw in ("average", "avg", "mean")):
            for t in relevant:
                all_cols = {c.lower(): c for c, _dt in self._table_columns_norm(t)}
                # Try to find the column named in the goal (e.g. "average rating" -> "rating")
                goal_words = goal.replace("average", "").replace("avg", "").replace("mean", "").replace("of", "").replace("the", "").split()
                target_col = None
                for w in goal_words:
                    w = w.strip()
                    if len(w) >= 3 and w in all_cols:
                        target_col = all_cols[w]
                        break
                if target_col and _safe_num(t, target_col):
                    return (f'SELECT {self._measure_agg_expr(t, target_col, "AVG")}'
                            f' AS average FROM {self._qtab(t)};')
                # Fallback: pick any verified-safe numeric column
                num_cols = [c for c, dt in self._table_columns_norm(t)
                            if dt.upper() in NUMERIC_TYPES or any(k in dt.lower() for k in ("int", "decimal", "numeric", "float", "real", "double", "money"))
                            and not c.lower().endswith("_id") and c.lower() != "id"]
                num_cols = [c for c in num_cols if _safe_num(t, c)]
                if num_cols:
                    return (f'SELECT {self._measure_agg_expr(t, num_cols[0], "AVG")}'
                            f' AS average FROM {self._qtab(t)};')

        # GROUP BY pattern ("by category", "by region", "by country")
        by_match = None
        for kw in (" by ", " per ", " breakdown", " distribution"):
            if kw in goal:
                by_match = kw.strip()
                break

        if by_match:
            for t in relevant:
                all_cols = self._table_columns_norm(t)
                cat_cols = [c for c, dt in all_cols
                            if dt.upper() in TEXT_TYPES or any(k in dt.lower() for k in ("char", "text", "varchar"))
                            and not c.lower().endswith("_id") and c.lower() != "id"]
                num_cols = [c for c, dt in all_cols
                            if dt.upper() in NUMERIC_TYPES or any(k in dt.lower() for k in ("int", "decimal", "numeric", "float", "real", "double", "money"))
                            and not c.lower().endswith("_id") and c.lower() != "id"]
                num_cols = [c for c in num_cols if _safe_num(t, c)]
                if cat_cols and num_cols:
                    cat = cat_cols[0]
                    num = num_cols[0]
                    return (f'SELECT t."{cat}", {self._measure_agg_expr(t, num, "SUM")} AS total '
                            f'FROM {self._qtab(t)} t GROUP BY t."{cat}" ORDER BY total DESC;')

        # RANKING pattern ("top", "best", "highest", "most")
        if any(kw in goal for kw in ("top", "best", "highest", "most", "leading", "largest")):
            for t in relevant:
                all_cols = self._table_columns_norm(t)
                name_cols = [c for c, dt in all_cols
                             if dt.upper() in TEXT_TYPES or any(k in dt.lower() for k in ("char", "text", "varchar"))
                             and not c.lower().endswith("_id") and c.lower() != "id"]
                num_cols = [c for c, dt in all_cols
                            if dt.upper() in NUMERIC_TYPES or any(k in dt.lower() for k in ("int", "decimal", "numeric", "float", "real", "double", "money"))
                            and not c.lower().endswith("_id") and c.lower() != "id"]
                num_cols = [c for c in num_cols if _safe_num(t, c)]
                if name_cols and num_cols:
                    name = name_cols[0]
                    num = num_cols[0]
                    return (f'SELECT "{name}", {self._measure_agg_expr(t, num, "SUM")} AS total '
                            f'FROM {self._qtab(t)} GROUP BY "{name}" ORDER BY total DESC LIMIT 10;')

        # Default: SELECT from the first table
        return f'SELECT * FROM {self._qtab(relevant[0])} LIMIT 50;'

    # ---- _row_counts (_row_counts) ----

    def _row_counts(self):
        """Count rows per table (cached per agent) so a genuinely empty table
        can be told apart from a faulty query."""
        if self._row_counts_cache is not None:
            return self._row_counts_cache
        counts = {}
        try:
            with self.engine.connect() as conn:
                for table_name in self.tables:
                    try:
                        from core.relational import quote_identifier
                        result = conn.execute(text(f'SELECT count(*) FROM {quote_identifier(table_name, "postgresql")}'))
                        counts[table_name] = int(result.scalar() or 0)
                    except Exception:
                        counts[table_name] = -1
        except Exception:
            pass
        self._row_counts_cache = counts
        return counts

