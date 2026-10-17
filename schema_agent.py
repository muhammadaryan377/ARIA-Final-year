"""
ARIA Schema Agent
-----------------
Role: Data Understanding

Runs once (or on schema change) to parse and map messy/undocumented relational
databases and semi-structured (JSON/CSV/PDF) sources. It identifies tables,
columns, data types, null patterns, primary keys, declared foreign keys, and
INFERRED relationships for columns that look like foreign keys but have no
formal constraint declared.

Semi-structured sources (JSON/CSV/PDF) are parsed with the Qwen3-4B model via
Ollama, which extracts entities/fields and maps them into the unified schema.
Qwen3-4B is used exclusively here — never for SQL generation or storytelling.

Output: schema_mapping.json  (single source of truth for the Goal Agent)

Usage:
    python schema_agent.py
    python schema_agent.py --json-source reports/export.json --csv-source exports/stock.csv
    python schema_agent.py --host localhost --port 5432 --db northwind --user postgres --password 12345
"""

import argparse
import json
import logging
import os
import re
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras
from psycopg2 import sql

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from llm_provider import LLMProvider, create_provider
from core.config import SCHEMA_DIR


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------

def get_connection(args):
    """Build a database connection (PostgreSQL or MySQL) from CLI args / env vars."""
    db_type = getattr(args, "db_type", None) or "postgresql"
    host = args.host or os.getenv("DB_HOST", "localhost")
    # The port must never leak across engine types: DB_PORT in .env targets
    # PostgreSQL, so MySQL must not inherit it. Only --port (or MYSQL_PORT) can
    # override MySQL's default.
    if db_type == "mysql":
        port = args.port or os.getenv("MYSQL_PORT", "3306")
    else:
        port = args.port or os.getenv("DB_PORT", "5432")
    dbname = args.db or os.getenv("DB_NAME")
    user = args.user or os.getenv("DB_USER")
    password = args.password or os.getenv("DB_PASSWORD")

    missing = [name for name, val in
               [("database", dbname), ("user", user), ("password", password)]
               if not val]
    if missing:
        sys.exit(
            f"ERROR: missing required connection info: {', '.join(missing)}.\n"
            f"Set them via .env or pass --db --user --password on the command line."
        )

    if db_type == "mysql":
        if not MYSQL_AVAILABLE:
            sys.exit("ERROR: pymysql is not installed. Run: pip install pymysql")
        try:
            conn = pymysql.connect(
                host=host, port=int(port), user=user, password=password,
                database=dbname, charset="utf8mb4", connect_timeout=10,
            )
        except pymysql.MySQLError as e:
            sys.exit(
                f"ERROR: could not connect to MySQL at {host}:{port}/{dbname} as {user}: {e}\n"
                "Check: is the MySQL server running on that host/port? Are the "
                "database name, user, and password correct, and does the server "
                "accept TCP connections?"
            )
        return conn

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
            connect_timeout=10,
        )
    except psycopg2.OperationalError as e:
        sys.exit(
            f"ERROR: could not connect to PostgreSQL at {host}:{port}/{dbname} as {user}: {e}\n"
            "Check: is the PostgreSQL service running on that host/port? Are the "
            "database name, user, and password correct? Does pg_hba.conf allow "
            "TCP connections from this host?"
        )

    return conn


def _dict_cursor(conn, db_type="postgresql"):
    """Return a dictionary-row cursor for the connected database type."""
    if db_type == "mysql":
        return conn.cursor(pymysql.cursors.DictCursor)
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def _lower_keys(row):
    """Normalize a dict row's keys to lowercase.

    PostgreSQL's information_schema returns lowercase column names, but MySQL
    returns UPPERCASE ones (TABLE_NAME, COLUMN_NAME, ...). All downstream code
    reads lowercase keys, so MySQL rows must be normalized or every
    information_schema query fails with KeyError.
    """
    if not row:
        return row
    return {str(k).lower(): v for k, v in row.items()}


def _quote_ident(db_type, name):
    """Return a SQL-safe quoted identifier (double quotes for PG, backticks for MySQL)."""
    if db_type == "mysql":
        return "`" + str(name).replace("`", "") + "`"
    return str(name)


# ---------------------------------------------------------------------------
# Metadata extraction (pure SQL, no LLM)
# ---------------------------------------------------------------------------

def get_tables_and_columns(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [{column, data_type, nullable}, ...]}"""
    query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        rows = cur.fetchall()

    tables = {}
    for row in rows:
        row = _lower_keys(row)
        tables.setdefault(row["table_name"], []).append({
            "column": row["column_name"],
            "data_type": row["data_type"],
            "nullable": row["is_nullable"] == "YES",
        })
    return tables


def get_primary_keys(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [pk_column, ...]}"""
    # NOTE: the join must include tc.table_name. In MySQL every primary key
    # constraint is literally named 'PRIMARY', so joining on constraint_name
    # alone cross-joins every table's PK columns to every other table.
    query = """
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
        ORDER BY tc.table_name, kcu.ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        rows = cur.fetchall()

    pks = {}
    for row in rows:
        row = _lower_keys(row)
        pks.setdefault(row["table_name"], []).append(row["column_name"])
    return pks


def get_declared_foreign_keys(conn, schema="public", db_type="postgresql"):
    """Return list of {table_name, column_name, references_table, references_column}"""
    if db_type == "mysql":
        # MySQL exposes the referenced table/column directly on key_column_usage.
        query = """
            SELECT
                kcu.table_name AS table_name,
                kcu.column_name AS column_name,
                kcu.referenced_table_name AS references_table,
                kcu.referenced_column_name AS references_column
            FROM information_schema.key_column_usage kcu
            WHERE kcu.referenced_table_name IS NOT NULL
              AND kcu.table_schema = %s
            ORDER BY kcu.table_name, kcu.ordinal_position;
        """
    else:
        # PostgreSQL: join pg_constraint directly and align conkey/confkey by
        # ordinal position. Joining information_schema.key_column_usage to
        # constraint_column_usage on the constraint name alone would cross-produce
        # composite foreign keys (a 2-column FK would yield 4 wrong pairs).
        query = """
            SELECT
                refcon.relname AS table_name,
                refatt.attname AS column_name,
                conrel.relname AS references_table,
                conatt.attname AS references_column
            FROM pg_constraint c
            JOIN pg_namespace n ON n.oid = c.connamespace
            JOIN pg_class refcon ON refcon.oid = c.conrelid
            JOIN pg_class conrel ON conrel.oid = c.confrelid
            JOIN unnest(c.conkey) WITH ORDINALITY AS srckeys(attnum, ord) ON TRUE
            JOIN unnest(c.confkey) WITH ORDINALITY AS tgtkeys(attnum, ord) ON srckeys.ord = tgtkeys.ord
            JOIN pg_attribute refatt ON refatt.attrelid = c.conrelid AND refatt.attnum = srckeys.attnum
            JOIN pg_attribute conatt ON conatt.attrelid = c.confrelid AND conatt.attnum = tgtkeys.attnum
            WHERE c.contype = 'f'
              AND n.nspname = %s
            ORDER BY refcon.relname, srckeys.ord;
        """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        return [dict(row) for row in cur.fetchall()]


def get_null_stats(conn, tables, schema="public", db_type="postgresql"):
    """Return {table_name: {column_name: {"total_rows": n, "nulls": n, "null_pct": p}}}"""
    stats = {}
    with conn.cursor() as cur:
        for table_name, columns in tables.items():
            table_stats = {}
            if db_type == "mysql":
                qualified = f"{_quote_ident(db_type, schema)}.{_quote_ident(db_type, table_name)}"
                cur.execute(f"SELECT COUNT(*) FROM {qualified}")
                total_rows = cur.fetchone()[0]

                for col in columns:
                    col_name = col["column"]
                    qcol = _quote_ident(db_type, col_name)
                    cur.execute(
                        f"SELECT COUNT(*) - COUNT({qcol}) FROM {qualified}"
                    )
                    nulls = cur.fetchone()[0]
                    null_pct = round((nulls / total_rows) * 100, 2) if total_rows else 0.0
                    table_stats[col_name] = {
                        "total_rows": total_rows,
                        "nulls": nulls,
                        "null_pct": null_pct,
                    }
            else:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}.{}")
                       .format(sql.Identifier(schema), sql.Identifier(table_name))
                )
                total_rows = cur.fetchone()[0]

                for col in columns:
                    col_name = col["column"]
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) - COUNT({}) FROM {}.{}")
                           .format(sql.Identifier(col_name), sql.Identifier(schema), sql.Identifier(table_name))
                    )
                    nulls = cur.fetchone()[0]
                    null_pct = round((nulls / total_rows) * 100, 2) if total_rows else 0.0
                    table_stats[col_name] = {
                        "total_rows": total_rows,
                        "nulls": nulls,
                        "null_pct": null_pct,
                    }
            stats[table_name] = table_stats
    return stats


# ---------------------------------------------------------------------------
# Heuristic relationship inference (fills the gap for "messy" databases)
# ---------------------------------------------------------------------------

def normalize_identifier(value):
    """Normalize names to compare database identifiers across styles."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _singularize(name):
    """Best-effort singular of an English identifier (for PK/name matching)."""
    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 2:
        return name[:-1]
    return name


# Tokens that signal "this word is about identifiers/keys", never content words.
_ID_TOKENS = {"id", "no", "num", "number", "code", "key", "fk", "ref", "uuid", "sk", "nk"}


def _tokenize_words(name):
    """Split an identifier into lowercase word tokens (snake_case, camelCase, acronyms).

    "MediaTypeId" -> ["media", "type", "id"]; "support_rep_id" -> ["support", "rep", "id"].
    This is what lets name matching work on ANY naming convention, not just X_id.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if w]


def _norm_value(v):
    """Normalize a sampled value for cross-column comparison (numeric-aware)."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    return ("str", str(v))


class _ValueProfiler:
    """Samples distinct column values so relationships can be confirmed from DATA.

    This is what makes inference work for ANY database: a column whose values
    overlap a primary key's values is a de-facto foreign key, regardless of how
    it is named (e.g. support_rep_id -> employee, reports_to -> employee).
    """

    MIN_OVERLAP = 0.85
    MAX_SAMPLE = 500
    MAX_PK_SAMPLE = 100000

    def __init__(self, conn, schema, db_type, tables, primary_keys):
        self.conn = conn
        self.schema = schema
        self.db_type = db_type
        self.tables = tables
        self._src_cache = {}
        self._pk_cache = {}
        self._rows_cache = {}
        self.pk_index = []
        for table, cols in primary_keys.items():
            for col in cols:
                self.pk_index.append(
                    (table, col, self._distinct(table, col, self._pk_cache, self.MAX_PK_SAMPLE)))

    def _distinct(self, table, col, cache, limit):
        key = (table, col)
        if key in cache:
            return cache[key]
        try:
            values = set()
            with _dict_cursor(self.conn, self.db_type) as cur:
                cur.execute(
                    f"SELECT DISTINCT {_quote_ident(self.db_type, col)} AS v "
                    f"FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)} "
                    f"WHERE {_quote_ident(self.db_type, col)} IS NOT NULL LIMIT {limit}"
                )
                for row in cur.fetchall():
                    values.add(_norm_value(row["v"]))
        except Exception:
            values = set()
        cache[key] = values
        return values

    def source_values(self, table, col):
        return self._distinct(table, col, self._src_cache, self.MAX_SAMPLE)

    def row_count(self, table):
        if table not in self._rows_cache:
            try:
                with _dict_cursor(self.conn, self.db_type) as cur:
                    cur.execute(
                        f"SELECT COUNT(*) AS n FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)}"
                    )
                    self._rows_cache[table] = cur.fetchone()["n"] or 0
            except Exception:
                self._rows_cache[table] = 0
        return self._rows_cache[table]

    @staticmethod
    def _overlap(src, target):
        if not src:
            return 0.0
        return len(src & target) / len(src)

    def strongest_overlap(self, source_set, only_tables=None, min_source=1, high_cardinality=False, row_count=None, exclude=None, require_unambiguous=False):
        """Return (table, pk_col, overlap_ratio, matched) for the best PK target.

        Ties (same ratio and matched count) are broken in favour of the target
        whose primary key set is the smallest that still contains the values —
        the tightest containment is the more likely real reference. When
        `require_unambiguous` is set and the top two targets are indistinguishable
        on (ratio, matched), no target is returned: with opaque names and no name
        evidence a wrong guess is worse than none.
        """
        cands = []
        for table, pk_col, target_set in self.pk_index:
            if only_tables and table not in only_tables:
                continue
            if exclude and (table, pk_col) == exclude:
                continue
            if len(source_set) < min_source:
                continue
            ratio = self._overlap(source_set, target_set)
            if ratio < self.MIN_OVERLAP:
                continue
            if high_cardinality:
                if len(source_set) < 10:
                    continue
                if row_count and len(source_set) < 0.05 * row_count:
                    continue
            cands.append((ratio, len(source_set & target_set), -len(target_set),
                          table, pk_col, len(source_set & target_set)))
        if not cands:
            return None
        cands.sort(reverse=True)
        if require_unambiguous and len(cands) >= 2 and cands[0][:2] == cands[1][:2]:
            return None
        ratio, _, _, table, pk_col, matched = cands[0]
        return table, pk_col, ratio, matched

    def _distinct_raw(self, table, col, limit):
        """Raw (un-normalized) distinct values, for exact SQL containment checks."""
        values = []
        try:
            with _dict_cursor(self.conn, self.db_type) as cur:
                cur.execute(
                    f"SELECT DISTINCT {_quote_ident(self.db_type, col)} AS v "
                    f"FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)} "
                    f"WHERE {_quote_ident(self.db_type, col)} IS NOT NULL LIMIT {limit}"
                )
                for row in cur.fetchall():
                    v = row["v"]
                    if v is not None:
                        values.append(v)
        except Exception:
            return []
        return values

    def _exact_matched(self, table, pk_col, values):
        """Count how many of `values` actually occur in the FULL target key column."""
        if not values:
            return None
        ident = _quote_ident(self.db_type, pk_col)
        try:
            with _dict_cursor(self.conn, self.db_type) as cur:
                if self.db_type == "mysql":
                    marks = ", ".join(["%s"] * len(values))
                    cur.execute(
                        f"SELECT count(DISTINCT {ident}) AS matched "
                        f"FROM {_quote_ident(self.db_type, table)} "
                        f"WHERE {ident} IN ({marks})",
                        tuple(values),
                    )
                else:
                    cur.execute(
                        f"SELECT count(DISTINCT {ident}) AS matched "
                        f"FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)} "
                        f"WHERE {ident} = ANY(%s)",
                        (list(values),),
                    )
                return cur.fetchone()["matched"] or 0
        except Exception:
            return None

    def strongest_exact_overlap(self, source_table, source_col, target_tables, exclude=None):
        """Like strongest_overlap but verifies against the FULL target key column.

        Sampling both sides dilutes containment when the target table is huge
        (e.g. a 1M-row sales key vs a 30k-row warranty that references it),
        hiding real relationships. When the column name already points at a
        concrete target, an exact SQL containment count is cheap and accurate.
        """
        src = self._distinct_raw(source_table, source_col, self.MAX_SAMPLE)
        if len(src) < 2:
            return None
        best = None
        for table, pk_col, _ in self.pk_index:
            if table not in target_tables:
                continue
            if exclude and (table, pk_col) == exclude:
                continue
            matched = self._exact_matched(table, pk_col, src)
            if matched is None:
                continue
            ratio = matched / len(src)
            if ratio < self.MIN_OVERLAP:
                continue
            cand = (ratio, matched, table, pk_col)
            if best is None or cand > best:
                best = cand
        if best is None:
            return None
        ratio, matched, table, pk_col = best
        return table, pk_col, ratio, matched

    def should_profile(self, col):
        """Only sample columns that could plausibly reference a key (bounds cost)."""
        name = str(col["column"]).lower()
        if name.endswith(("_id", "_code", "id", "_no", "_num", "_number", "_key", "_uuid")):
            return True
        dtype = str(col.get("data_type", "")).upper()
        return any(tok in dtype for tok in ("INT", "NUMERIC", "DECIMAL", "NUMBER", "REAL", "FLOAT"))


def infer_primary_keys(tables, declared_primary_keys):
    """Return likely PKs when the database does not declare them explicitly."""
    inferred = {}

    for table_name, columns in tables.items():
        if declared_primary_keys.get(table_name):
            inferred[table_name] = declared_primary_keys[table_name]
            continue

        table_norm = normalize_identifier(table_name)
        table_singular = _singularize(table_norm)
        candidates = []
        for col in columns:
            col_name = col["column"]
            normalized = normalize_identifier(col_name)
            if (normalized == "id"
                    or normalized == table_norm + "id"
                    or normalized == table_singular + "id"):
                candidates.append(col_name)

        if len(candidates) == 1:
            inferred[table_name] = candidates
            continue

        non_null_columns = [col["column"] for col in columns if not col["nullable"]]
        if len(non_null_columns) == 1:
            # Never trust a single non-null column as PK if it looks like a
            # foreign key (e.g. a table whose only NOT NULL column is
            # `customer_id` almost certainly points elsewhere). Only accept it
            # when it does not look like a reference to another table.
            candidate = non_null_columns[0]
            normalized = normalize_identifier(candidate)
            is_id_like = normalized.endswith("id") and normalized not in ("id", normalize_identifier(table_name) + "id")
            if not is_id_like:
                inferred[table_name] = non_null_columns
            continue

        id_like = [col["column"] for col in columns if col["column"].lower().endswith("_id") or col["column"].lower() == "id"]
        if len(id_like) == 1:
            inferred[table_name] = id_like

    return inferred


def _qname(db_type, name):
    """SQL-safe quoted identifier (double quotes for PG, backticks for MySQL)."""
    if db_type == "mysql":
        return "`" + str(name).replace("`", "") + "`"
    return '"' + str(name).replace('"', "") + '"'


def _table_rows(conn, table_name, schema, db_type):
    """Best-effort row-count estimate for the table (None when unknown)."""
    try:
        if db_type == "postgresql":
            cur = conn.cursor()
            cur.execute(
                "SELECT c.reltuples::bigint FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s",
                (schema, table_name),
            )
            row = cur.fetchone()
            return row[0] if row else None
        cur = conn.cursor()
        cur.execute(
            "SELECT TABLE_ROWS FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
            (table_name,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _column_is_unique_key(conn, table_name, column, schema, db_type):
    """Return True when the column is fully populated and every value is unique."""
    try:
        cur = conn.cursor()
        table_ref = (f"{_qname(db_type, schema)}.{_qname(db_type, table_name)}"
                     if db_type == "postgresql" else _qname(db_type, table_name))
        col_ref = _qname(db_type, column)
        cur.execute(
            f"SELECT count(*) AS total, count({col_ref}) AS non_null, "
            f"count(DISTINCT {col_ref}) AS distinct_values FROM {table_ref}"
        )
        row = cur.fetchone()
        if not row:
            return False
        total, non_null, distinct_values = row[0], row[1], row[2]
        return (total is not None and total > 0
                and non_null == total and distinct_values == total)
    except Exception:
        return False


def _column_is_empty(conn, table_name, column, schema, db_type):
    """Return True when a column holds no populated values (all NULL or blank).

    A primary key or foreign key built entirely on such a column carries no
    information, so it is dropped from the mapping instead of being reported as
    a real key.
    """
    try:
        cur = conn.cursor()
        table_ref = (f"{_qname(db_type, schema)}.{_qname(db_type, table_name)}"
                     if db_type == "postgresql" else _qname(db_type, table_name))
        col_ref = _qname(db_type, column)
        if db_type == "mysql":
            cur.execute(
                f"SELECT COUNT({col_ref}) AS non_null, "
                f"COUNT(NULLIF(TRIM(CAST({col_ref} AS CHAR)), '')) AS non_blank "
                f"FROM {table_ref}"
            )
        else:
            cur.execute(
                f"SELECT COUNT({col_ref}) AS non_null, "
                f"COUNT(NULLIF(TRIM(CAST({col_ref} AS TEXT)), '')) AS non_blank "
                f"FROM {table_ref}"
            )
        row = cur.fetchone()
        return row is not None and (row[0] or 0) == 0 and (row[1] or 0) == 0
    except Exception:
        return False


def infer_primary_keys_from_data(conn, tables, schema, db_type, existing_pks, max_rows=500000, exclude_columns=None):
    """Data-backed PK inference for tables the name heuristics could not resolve.

    When a table declares no PK and no column *name* points to one, look at the
    actual data: an id-like column that is fully populated and whose every value
    is unique (distinct count == row count) is the de-facto key. This handles
    flat/denormalized tables whose natural key is a prefixed text id
    (e.g. 'T0000001'), which naming conventions cannot detect, and never guesses
    for a column that is merely shared with other tables (customer_id, ...).

    `exclude_columns` ((table, column) pairs) lets confirmed foreign-key columns
    be ruled out as key candidates — a column that provably references another
    table is a reference, not the table's own identity.
    """
    exclude_columns = exclude_columns or set()
    inferred = {}
    for table_name, columns in tables.items():
        if existing_pks.get(table_name):
            continue
        id_like = [c["column"] for c in columns
                   if str(c["column"]).lower().endswith("_id") or str(c["column"]).lower() == "id"]
        id_like = [c for c in id_like if (table_name, c) not in exclude_columns]
        if not id_like:
            continue
        rows = _table_rows(conn, table_name, schema, db_type)
        if rows is not None and rows > max_rows:
            # Too large to verify with a scan; don't guess on partial evidence.
            continue
        unique = [col for col in id_like
                  if _column_is_unique_key(conn, table_name, col, schema, db_type)]
        if not unique:
            continue
        if len(unique) == 1:
            inferred[table_name] = unique
            continue
        # Several unique id columns: only commit when exactly one carries the
        # primary-looking name (id / <table>id / <table_singular>id).
        table_norm = normalize_identifier(table_name)
        primary_named = [c for c in unique if normalize_identifier(c) in
                         ("id", table_norm + "id", _singularize(table_norm) + "id")]
        if len(primary_named) == 1:
            inferred[table_name] = primary_named
    return inferred


def infer_relationships(tables, primary_keys, declared_fks, conn=None, schema="public", db_type="postgresql"):
    """
    Infer relationships for columns that look like foreign keys but have NO
    declared constraint. Two general signals, in order of strength:

      1. DATA: when `conn` is given, sample distinct column values and confirm a
         relationship by value overlap with another table's primary key. This
         works for ANY naming convention (support_rep_id -> employee, reports_to
         -> employee, ...) because it relies on the values, not the names.
      2. NAME: tokenized word matching (snake_case / camelCase / plurals), which
         is a generic convention rule, not a per-database patch.

    Confidence levels:
      - "data-confirmed"     value overlap PLUS a name hint (strongest)
      - "data-inferred"      value overlap only (e.g. opaque names)
      - "heuristic-name-match"  name match only (offline / no data)
    """
    declared_pairs = {(fk["table_name"], fk["column_name"]) for fk in declared_fks}

    pk_owner = {}
    table_pk = {}
    for table_name, pk_cols in primary_keys.items():
        table_pk[table_name] = pk_cols
        for pk_col in pk_cols:
            pk_owner.setdefault(pk_col, []).append(table_name)

    profiler = _ValueProfiler(conn, schema, db_type, tables, primary_keys) if conn is not None else None

    def table_name_candidates(col_name):
        """For 'X_id', return tables whose name (singular or plural) is X."""
        base = col_name[:-3].lower() if col_name.endswith("_id") else col_name.lower()
        forms = {base, base + "s", base + "es"}
        if base.endswith("y") and len(base) > 1:
            forms.add(base[:-1] + "ies")
        forms.add(_singularize(base))
        names = [t for t in tables if t.lower() in forms]
        if not names:
            # Generalized token match: any naming style, not just X_id.
            col_tokens = {_singularize(tok) for tok in _tokenize_words(col_name) if tok not in _ID_TOKENS}
            if col_tokens:
                for t in tables:
                    t_tokens = {_singularize(tok) for tok in _tokenize_words(t)}
                    if col_tokens <= t_tokens:
                        names.append(t)
        return sorted(set(names))

    def _emit(table_name, col_name, candidate_table, ref_col, confidence, note):
        inferred.append({
            "table": table_name,
            "column": col_name,
            "references_table": candidate_table,
            "references_column": ref_col,
            "confidence": confidence,
            "note": note,
        })

    inferred = []
    for table_name, columns in tables.items():
        for col in columns:
            col_name = col["column"]

            if (table_name, col_name) in declared_pairs:
                continue

            own_pk = primary_keys.get(table_name, [])
            if col_name in own_pk:
                # A sole primary key is the table's own identity, never a FK.
                # A column of a COMPOSITE key may also reference another table
                # (junction tables), so it stays eligible but only via DATA.
                if len(own_pk) == 1:
                    continue
                is_composite_key_col = True
            else:
                is_composite_key_col = False

            is_ref_flavored = (col_name.endswith("_id") or col_name.endswith("id")
                               or col_name.lower() == "id")
            if not is_ref_flavored:
                # Data-only candidates must still look plausibly reference-ish so
                # we do not scan every text column of every table.
                if not (profiler and profiler.should_profile(col)):
                    continue

            candidate_tables = set(pk_owner.get(col_name, []))
            table_cands = set(table_name_candidates(col_name))
            if not table_cands:
                # Fallback: a column that shares its name with ANOTHER table's
                # primary key (only when that name is meaningful). A same-named
                # non-key column is a coincidental attribute, not a relationship.
                same_name = [
                    other_table
                    for other_table in tables
                    if other_table != table_name
                    and col_name in table_pk.get(other_table, [])
                    and len(normalize_identifier(col_name)) >= 3
                ]
                if len(same_name) == 1:
                    candidate_tables.update(same_name)
            candidate_tables.update(table_cands)
            candidate_tables.discard(table_name)

            if profiler is not None:
                src_set = profiler.source_values(table_name, col_name)
                rows = profiler.row_count(table_name)
                hit = None
                if is_ref_flavored or is_composite_key_col:
                    # id-suffixed columns: the name is already a strong signal,
                    # so data only needs a modest overlap to confirm. When the
                    # name gives NO target at all (opaque support_rep_id), let
                    # the data speak against every primary key — but only pick a
                    # target it can identify unambiguously.
                    hit = profiler.strongest_overlap(
                        src_set,
                        only_tables=candidate_tables or None,
                        min_source=2,
                        exclude=(table_name, col_name),
                        require_unambiguous=not candidate_tables,
                    )
                elif not candidate_tables:
                    # Non-id columns (reports_to): only data can find them, and
                    # only when high-cardinality enough to not be a code/status.
                    hit = profiler.strongest_overlap(
                        src_set,
                        high_cardinality=True,
                        row_count=rows,
                        exclude=(table_name, col_name),
                    )
                if hit is None and candidate_tables and (is_ref_flavored or is_composite_key_col):
                    # Name already narrows the target; verify against the FULL
                    # target key (exact containment) so huge key tables are not
                    # diluted by sampling (e.g. warranty -> 1M-row sales).
                    hit = profiler.strongest_exact_overlap(
                        table_name, col_name, candidate_tables,
                        exclude=(table_name, col_name),
                    )
                if hit is not None:
                    target_table, target_col, ratio, matched = hit
                    if target_table == table_name:
                        hit = None
                    elif target_table in candidate_tables:
                        confidence, note = "data-confirmed", (
                            f"No FK constraint declared; confirmed by value overlap "
                            f"({matched}/{len(src_set)} sample values) with {target_table}.{target_col}.")
                    else:
                        confidence, note = "data-inferred", (
                            f"No FK constraint declared and no name match; inferred from "
                            f"value overlap ({matched}/{len(src_set)} sample values) with {target_table}.{target_col}.")
                    if hit is not None:
                        ref_cols = table_pk.get(target_table) or [target_col]
                        for ref_col in ref_cols:
                            _emit(table_name, col_name, target_table, ref_col, confidence, note)
                        continue

            if is_composite_key_col:
                # Composite-key columns only produce relationships when the data
                # confirms them; a name match alone is not enough.
                continue

            for candidate_table in sorted(candidate_tables):
                ref_cols = table_pk.get(candidate_table) or [col_name]
                for ref_col in ref_cols:
                    _emit(table_name, col_name, candidate_table, ref_col,
                          "heuristic-name-match",
                          "No FK constraint declared in the database; relationship inferred from column naming convention.")
    return inferred


# ---------------------------------------------------------------------------
# LLM-assisted schema reasoning (optional enrichment over the heuristics)
#
# Runs a single structured pass with whichever LLM backend is active — local
# (Ollama via `aria-goal`) or cloud (Groq) — through the unified
# LLMProvider.chat() interface. The LLM:
#   * confirms or disproves the heuristic candidate relationships,
#   * may add genuine FK-like links the heuristics missed,
#   * suggests a primary key only when none was declared or inferred,
#   * attaches plain-English table descriptions and measure/dimension hints.
#
# Every suggestion is validated against the real schema before it is merged, so
# a hallucinated table/column is never accepted. Any timeout, failure or
# non-JSON response leaves the deterministic mapping untouched.
# ---------------------------------------------------------------------------

_SCHEMA_REASONING_ROLE = "schema"

_SCHEMA_REASONING_PROMPT = """\
You are a careful database schema analyst for a BI system. Below is a schema
summary (tables, columns, data types, nullability, declared/inferred primary
keys, and UNVERIFIED candidate relationships). Reason about it and return STRICT
JSON only, with exactly this shape:

{
  "tables": {
    "<table_name>": {
      "description": "one-line plain-English purpose of this table",
      "pk_candidates": ["column names likely to be a primary key, ONLY if the table has none declared or inferred; else []"],
      "measures": ["numeric columns that are business measurements to aggregate (never ids, codes, flags or timestamps)"],
      "dimensions": ["categorical/text columns useful for grouping and filtering"]
    }
  },
  "relationships": [
    {"table": "...", "column": "...", "references_table": "...", "references_column": "...",
     "kind": "confirm", "reason": "short justification"}
  ]
}

Rules:
- Refer ONLY to tables and columns that exist in the schema above. Never invent names.
- "kind": "confirm" for a candidate relationship you believe is a real foreign key;
  "add" for a genuine FK-like link the candidates missed. Skip same-name columns
  that merely coincide (e.g. two unrelated "amount" columns).
- Use exact, case-sensitive table and column names.
- When unsure, omit rather than guess. No markdown, no prose, JSON only.
"""


_SCHEMA_REASONING_PROMPT_TAIL = "\nSchema:\n"


def enrich_with_llm(mapping, llm):
    """Optional LLM-assisted reasoning pass over a built schema mapping.

    Works identically for every provider (Ollama 'local' or Groq 'cloud').
    Returns the mapping unchanged when no LLM is given, the call fails/times
    out, or the response is unusable.
    """
    if llm is None:
        return mapping

    summary = _condense_schema(mapping)
    if not summary.strip():
        return mapping

    try:
        timeout = 300 if getattr(llm, "provider", None) == "local" else 60
        content = llm.chat(
            _SCHEMA_REASONING_ROLE,
            messages=[{"role": "user", "content": _SCHEMA_REASONING_PROMPT + _SCHEMA_REASONING_PROMPT_TAIL + summary}],
            temperature=0.0,
            num_predict=2500,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"WARNING: schema LLM reasoning skipped ({exc}); keeping heuristic results.")
        return mapping

    parsed = _parse_schema_reasoning(content)
    if not parsed or not (isinstance(parsed.get("tables"), dict)
                          or isinstance(parsed.get("relationships"), list)):
        print("WARNING: schema LLM reasoning returned no usable JSON; keeping heuristic results.")
        return mapping

    try:
        model = llm.model_for(_SCHEMA_REASONING_ROLE)
    except Exception:
        model = None
    mapping["llm_reasoning"] = {
        "provider": getattr(llm, "provider", None),
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _apply_schema_reasoning(mapping, parsed)
    return mapping


def _condense_schema(mapping, max_tables=60, max_chars=12000):
    """Compact text summary of the mapping for the reasoning LLM."""
    tables = mapping.get("tables", {})
    if not tables:
        return ""
    lines = [f"Database: {mapping.get('database', 'unknown')} "
             f"(schema: {mapping.get('schema', '?')})"]

    for idx, (table_name, info) in enumerate(tables.items()):
        if idx >= max_tables:
            lines.append("... (additional tables omitted)")
            break
        pk = info.get("primary_key") or info.get("inferred_primary_key") or []
        pk_str = ", ".join(pk) if isinstance(pk, (list, tuple)) and pk else "none declared/inferred"
        lines.append(f"TABLE {table_name} (PK: {pk_str})")
        for col in info.get("columns", []):
            npct = info.get("null_stats", {}).get(col["column"], {}).get("null_pct")
            null_info = "NULL" if col.get("nullable") else "NOT NULL"
            if npct is not None:
                null_info += f" ({npct}% null)"
            lines.append(f"  {col['column']} {col.get('data_type', '?')} {null_info}")
        lines.append("")

    declared = mapping.get("declared_relationships", [])
    if declared:
        lines.append("DECLARED FOREIGN KEYS:")
        for d in declared:
            lines.append(f"  {d['table_name']}.{d['column_name']} -> "
                         f"{d['references_table']}.{d['references_column']}")
        lines.append("")

    inferred = mapping.get("inferred_relationships", [])
    if inferred:
        lines.append("CANDIDATE RELATIONSHIPS (unverified, from naming heuristics):")
        for r in inferred:
            lines.append(f"  {r['table']}.{r['column']} -> "
                         f"{r['references_table']}.{r['references_column']}")

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n... (truncated)"
    return summary


def _parse_schema_reasoning(content):
    """Robustly parse the reasoning response into a dict ({} on any failure).

    Falls back to the largest object-bounded prefix when the model's reply is
    truncated mid-JSON (the token cap can be hit before the closing brace), so
    a partial but otherwise-correct answer is not discarded wholesale.
    """
    if not content or not isinstance(content, str):
        return {}
    text = content.strip()
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Truncation salvage: try each `{` boundary before the final `}` and keep
    # the largest prefix that still parses as a dict.
    for open_idx in range(start, end):
        if text[open_idx] != "{":
            continue
        try:
            data = json.loads(text[open_idx:end + 1])
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


_NUMERIC_TYPE_HINTS = (
    "int", "numeric", "decimal", "float", "real", "double", "serial", "money",
)


def _is_numeric_type(data_type):
    t = (data_type or "").lower()
    return any(h in t for h in _NUMERIC_TYPE_HINTS)


def _column_data_type(table_info, col_name):
    for c in table_info.get("columns", []):
        if c["column"] == col_name:
            return c.get("data_type", "")
    return ""


def _apply_schema_reasoning(mapping, parsed):
    """Merge validated LLM suggestions into the mapping (never invents names)."""
    tables = mapping.setdefault("tables", {})
    declared = mapping.get("declared_relationships", [])
    declared_pairs = {(d["table_name"], d["column_name"]) for d in declared}
    inferred = mapping.setdefault("inferred_relationships", [])
    edges = mapping.setdefault("relationship_edges", [])

    def pair(r):
        return (r.get("table"), r.get("column"), r.get("references_table"), r.get("references_column"))

    # -- table-level annotations ------------------------------------------
    llm_tables = parsed.get("tables") or {}
    if isinstance(llm_tables, dict):
        for table_name, info in llm_tables.items():
            if table_name not in tables or not isinstance(info, dict):
                continue
            table_info = tables[table_name]
            real_columns = {c["column"] for c in table_info.get("columns", [])}

            desc = info.get("description")
            if isinstance(desc, str) and desc.strip():
                table_info["description"] = desc.strip()

            measures = [c for c in (info.get("measures") or []) if isinstance(c, str)]
            dims = [c for c in (info.get("dimensions") or []) if isinstance(c, str)]
            measures = [c for c in measures if c in real_columns and _is_numeric_type(_column_data_type(table_info, c))]
            dims = [c for c in dims if c in real_columns and c not in measures]
            if measures or dims:
                table_info.setdefault("semantic_tags", {}).update(
                    {"measures": measures, "dimensions": dims}
                )

            existing_pk = table_info.get("primary_key") or table_info.get("inferred_primary_key") or []
            pk_cands = [c for c in (info.get("pk_candidates") or []) if isinstance(c, str) and c in real_columns]
            if not existing_pk and len(pk_cands) == 1:
                cand = pk_cands[0]
                nullable = next((c.get("nullable") for c in table_info.get("columns", []) if c["column"] == cand), True)
                null_pct = table_info.get("null_stats", {}).get(cand, {}).get("null_pct")
                if nullable is False or null_pct in (0, 0.0):
                    table_info["inferred_primary_key"] = [cand]
                    if not table_info.get("primary_key"):
                        table_info["primary_key"] = [cand]

    # -- relationship confirm / add ----------------------------------------
    llm_rels = parsed.get("relationships") or []
    if isinstance(llm_rels, list):
        for rel in llm_rels:
            if not isinstance(rel, dict):
                continue
            table = rel.get("table")
            column = rel.get("column")
            ref_table = rel.get("references_table")
            ref_col = rel.get("references_column")
            kind = rel.get("kind")
            if not all(isinstance(x, str) and x for x in (table, column, ref_table, ref_col)):
                continue
            if table not in tables or ref_table not in tables:
                continue  # hallucinated table
            if column not in {c["column"] for c in tables[table].get("columns", [])}:
                continue
            if ref_col not in {c["column"] for c in tables[ref_table].get("columns", [])}:
                continue
            # The source column must look like a reference (id-like name or an
            # exact match to the target column): this stops the LLM from
            # "confirming" a numeric measure (e.g. `total`) as a foreign key.
            if not (normalize_identifier(column).endswith("id")
                    or normalize_identifier(column) == normalize_identifier(ref_col)):
                continue
            if (table, column) in declared_pairs:
                continue  # already constrained; don't second-guess the DB

            new_pair = (table, column, ref_table, ref_col)
            if kind == "confirm":
                # Replace any heuristic target for this column with the
                # LLM-confirmed target (upgrades confidence, removes ambiguity).
                inferred[:] = [r for r in inferred
                               if not (r.get("table") == table and r.get("column") == column)]
                edges[:] = [e for e in edges
                            if not (e.get("source_table") == table and e.get("source_column") == column)]
                inferred.append({
                    "table": table, "column": column,
                    "references_table": ref_table, "references_column": ref_col,
                    "confidence": "llm-confirmed",
                    "note": "Confirmed by LLM reasoning (no FK constraint declared in the database).",
                })
                edges.append({
                    "source_table": table, "source_column": column,
                    "target_table": ref_table, "target_column": ref_col,
                    "type": "inferred", "confidence": "llm-confirmed",
                })
            elif kind == "add" and new_pair not in {pair(r) for r in inferred}:
                # Only add when this column has no inferred FK yet (no ambiguity).
                if not any(r.get("table") == table and r.get("column") == column for r in inferred):
                    inferred.append({
                        "table": table, "column": column,
                        "references_table": ref_table, "references_column": ref_col,
                        "confidence": "llm-reasoned",
                        "note": "Proposed by LLM reasoning (no FK constraint declared in the database).",
                    })
                    edges.append({
                        "source_table": table, "source_column": column,
                        "target_table": ref_table, "target_column": ref_col,
                        "type": "inferred", "confidence": "llm-reasoned",
                    })


# ---------------------------------------------------------------------------
# Semi-structured source extraction (JSON / CSV / PDF) via Qwen3-4B
# ---------------------------------------------------------------------------

def _read_sample_rows(path, max_rows=20):
    """Return a small text sample of a JSON or CSV file for LLM extraction."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = [row for _, row in zip(range(max_rows + 1), reader)]
        if not rows:
            return ""
        return "\n".join("|".join(r) for r in rows)

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:4000]

    # PDF or anything else: fall back to the first chunk of raw text.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()[:4000]


def extract_semi_structured_schema(source_path, model=None, llm=None):
    """
    Ask the configured LLM (Qwen3 locally, Llama via Groq) to extract
    entities/fields from a semi-structured source and map them to the unified
    schema. Returns a dict describing the discovered table(s) plus declared
    relationships from the source.
    """
    llm = llm or create_provider()
    sample = _read_sample_rows(source_path)
    if not sample:
        print(f"WARNING: could not read sample from {source_path}")
        return {}

    prompt = f"""
You are the Schema Agent of a BI system. Analyse the following semi-structured
data sample and extract its schema. Return JSON only (no markdown) with this shape:
{{
  "tables": {{
    "table_name": {{
      "columns": [{{"column": "...", "data_type": "...", "nullable": true}}],
      "primary_key": ["..."],
      "source": "filename"
    }}
  }},
  "relationships": [{{"table": "...", "column": "...", "references_table": "...", "references_column": "..."}}]
}}

Data sample:
{sample}
"""
    try:
        content = llm.chat("semi", messages=[{"role": "user", "content": prompt}],
                           temperature=0.0, num_predict=800)
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        return json.loads(content)
    except Exception as exc:
        print(f"WARNING: semi-structured extraction failed for {source_path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Assemble & write schema_mapping.json
# ---------------------------------------------------------------------------

def find_schema_with_tables(conn):
    """Return the first non-system Postgres schema that contains tables."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_schema NOT LIKE 'pg_%'
              AND table_schema <> 'public'
            GROUP BY table_schema
            ORDER BY count(*) DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else "public"


def _drop_empty_key_columns(mapping, conn, schema, db_type):
    """Remove any primary/foreign key whose source column is fully empty.

    A key built on a column with no populated values (all NULL or blank) cannot
    identify or reference anything, so it is dropped from the mapping. The
    removals are recorded under mapping["empty_key_columns_removed"] so the
    decision is transparent. On any query failure the column is kept (safe).
    """
    tables = mapping.setdefault("tables", {})
    removed_pks, removed_fks = [], []

    for table_name, info in tables.items():
        col_names = {c["column"] for c in info.get("columns", [])}
        for key_field in ("primary_key", "inferred_primary_key"):
            kept = []
            for pk in info.get(key_field, []):
                if pk in col_names and _column_is_empty(conn, table_name, pk, schema, db_type):
                    removed_pks.append((table_name, pk))
                else:
                    kept.append(pk)
            info[key_field] = kept
        if not info.get("primary_key"):
            info["primary_key"] = []

    def source_pair(rel, declared):
        if declared:
            return rel.get("table_name"), rel.get("column_name")
        return rel.get("table"), rel.get("column")

    removed_fk_pairs = set()
    for rel_list, declared in ((mapping.get("declared_relationships", []), True),
                               (mapping.get("inferred_relationships", []), False)):
        kept = []
        for rel in rel_list:
            table_name, col_name = source_pair(rel, declared)
            col_names = {c["column"] for c in tables.get(table_name, {}).get("columns", [])}
            if (table_name in tables and col_name in col_names
                    and _column_is_empty(conn, table_name, col_name, schema, db_type)):
                removed_fk_pairs.add((table_name, col_name))
                removed_fks.append((table_name, col_name, declared))
            else:
                kept.append(rel)
        rel_list[:] = kept

    if removed_fk_pairs:
        mapping["relationship_edges"] = [
            e for e in mapping.get("relationship_edges", [])
            if (e.get("source_table"), e.get("source_column")) not in removed_fk_pairs
        ]

    mapping["empty_key_columns_removed"] = {
        "primary_keys": sorted(set(removed_pks)),
        "foreign_keys": sorted(set(removed_fks)),
    }
    return mapping


def build_schema_mapping(conn, schema="public", semi_sources=None, db_type="postgresql", llm=None, database_name=None):
    tables = get_tables_and_columns(conn, schema, db_type)
    primary_keys = get_primary_keys(conn, schema, db_type)
    inferred_primary_keys = infer_primary_keys(tables, primary_keys)
    inferred_primary_keys.update(infer_primary_keys_from_data(
        conn, tables, schema, db_type,
        {**primary_keys, **inferred_primary_keys},
    ))
    declared_fks = get_declared_foreign_keys(conn, schema, db_type)
    null_stats = get_null_stats(conn, tables, schema, db_type)

    merged_primary_keys = {**primary_keys, **inferred_primary_keys}
    inferred_fks = infer_relationships(tables, merged_primary_keys, declared_fks,
                                       conn=conn, schema=schema, db_type=db_type)

    # Refinement: a column confirmed (by data) to reference another table is a
    # foreign key, not the table's own identity. Ruling those out can resolve
    # ambiguous key candidates (e.g. a warranty table where both claim_id and
    # sale_id are unique — sale_id provably points at sales, so claim_id wins).
    fk_sources = {(r["table"], r["column"]) for r in inferred_fks
                  if r.get("confidence") in ("data-confirmed", "data-inferred")}
    if fk_sources:
        refined = infer_primary_keys_from_data(
            conn, tables, schema, db_type,
            {**primary_keys, **inferred_primary_keys},
            exclude_columns=fk_sources,
        )
        if refined:
            inferred_primary_keys.update(refined)
            merged_primary_keys = {**primary_keys, **inferred_primary_keys}
            inferred_fks = infer_relationships(tables, merged_primary_keys, declared_fks,
                                               conn=conn, schema=schema, db_type=db_type)

    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "database": database_name or os.getenv("DB_NAME") or "unknown_database",
        "schema": schema,
        "tables": {},
        "declared_relationships": declared_fks,
        "inferred_relationships": inferred_fks,
        "relationship_edges": [],
        "semi_structured_sources": {},
    }

    for rel in declared_fks:
        mapping["relationship_edges"].append({
            "source_table": rel["table_name"],
            "source_column": rel["column_name"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "declared",
        })

    for rel in inferred_fks:
        mapping["relationship_edges"].append({
            "source_table": rel["table"],
            "source_column": rel["column"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "inferred",
            "confidence": rel.get("confidence", "heuristic"),
        })

    for table_name, columns in tables.items():
        table_primary_key = primary_keys.get(table_name, inferred_primary_keys.get(table_name, []))
        mapping["tables"][table_name] = {
            "columns": columns,
            "primary_key": table_primary_key,
            "inferred_primary_key": inferred_primary_keys.get(table_name, []),
            "null_stats": null_stats.get(table_name, {}),
        }

    # Semi-structured sources: merge extracted tables/relationships into the map.
    for source_path in (semi_sources or []):
        extracted = extract_semi_structured_schema(source_path, llm=llm)
        if not extracted:
            continue
        for table_name, table_info in extracted.get("tables", {}).items():
            if table_name in mapping["tables"]:
                # Merge new columns that don't already exist.
                existing_cols = {c["column"] for c in mapping["tables"][table_name]["columns"]}
                for col in table_info.get("columns", []):
                    if col["column"] not in existing_cols:
                        mapping["tables"][table_name]["columns"].append(col)
            else:
                mapping["tables"][table_name] = {
                    "columns": table_info.get("columns", []),
                    "primary_key": table_info.get("primary_key", []),
                    "inferred_primary_key": table_info.get("primary_key", []),
                    "null_stats": {},
                    "source": table_info.get("source", source_path),
                }

        for rel in extracted.get("relationships", []):
            mapping["inferred_relationships"].append(rel)
            mapping["relationship_edges"].append({
                "source_table": rel["table"],
                "source_column": rel["column"],
                "target_table": rel["references_table"],
                "target_column": rel["references_column"],
                "type": "inferred",
                "confidence": "semi-structured-llm",
            })

        mapping["semi_structured_sources"][str(source_path)] = {
            "tables": list(extracted.get("tables", {}).keys()),
        }

    if llm is not None:
        mapping = enrich_with_llm(mapping, llm)

    if mapping.get("tables"):
        _drop_empty_key_columns(mapping, conn, schema, db_type)

    return mapping


SNAPSHOT_KEEP = 20  # max snapshot files retained per (user, database) file set


def _prune_snapshots(directory, stem, suffix, keep=SNAPSHOT_KEEP):
    """Delete the oldest snapshot files for a file set, keeping the newest `keep`.

    Never touches the base file ({stem}{suffix}) or the {stem}_latest alias.
    """
    try:
        snapshots = [
            p for p in Path(directory).glob(f"{stem}_*.json")
            if p.name != f"{stem}_latest{suffix}"
        ]
        if len(snapshots) <= keep:
            return
        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in snapshots[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def get_output_paths(output_path, db_name=None, snapshot_id=None):
    """Return the base output, a unique snapshot, the latest alias, and the snapshot id.

    Every database (when `db_name` is given) gets its own file set:

        schema_mapping_<db>.json          - canonical base (overwritten each run)
        schema_mapping_<db>_latest.json   - stable pointer for the active session
        schema_mapping_<db>_<id>.json     - unique snapshot, never overwritten

    The snapshot file name embeds the same `snapshot_id` that the mapping JSON
    carries, so the JSON and its file are always correlated. Old snapshots beyond
    SNAPSHOT_KEEP are pruned. Returns (base, snapshot, latest, snapshot_id).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    stem, suffix = output.stem, output.suffix or ".json"

    if db_name:
        safe_db = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(db_name))[:64] or "db"
        stem = f"{stem}_{safe_db}"

    if not snapshot_id:
        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    # Guarantee a filename that does not already exist, so a snapshot is never
    # overwritten (unique within and across users, DBs, and processes).
    candidate = f"{stem}_{snapshot_id}{suffix}"
    counter = 1
    while (output.parent / candidate).exists():
        candidate = f"{stem}_{snapshot_id}_{counter}{suffix}"
        counter += 1

    base = output.with_name(f"{stem}{suffix}")
    snapshot = output.with_name(candidate)
    latest = output.with_name(f"{stem}_latest{suffix}")

    _prune_snapshots(output.parent, stem, suffix)
    used_id = candidate[len(stem) + 1:-len(suffix)]
    return base, snapshot, latest, used_id


# ---------------------------------------------------------------------------
# File source handling (semi-structured path: CSV / tabular PDF)
# ---------------------------------------------------------------------------

def _coerce_numeric(df):
    """Best-effort numeric coercion so SQLite SUM/AVG work on text-sourced columns."""
    out = df.copy()
    for col in out.columns:
        coerced = pd.to_numeric(out[col], errors="coerce")
        non_null = coerced.notna().sum()
        if non_null > 0 and non_null >= len(out) * 0.9:
            out[col] = coerced
    return out


def build_tables_schema(tables, source_path, database):
    """Build a schema mapping from a dict of {table_name: DataFrame} (deterministic, no LLM)."""
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to read file sources.")

    tables_mapping = {}
    for table_name, df in tables.items():
        columns = []
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                data_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                data_type = "REAL"
            else:
                data_type = "TEXT"
            columns.append({"column": str(col), "data_type": data_type, "nullable": True})
        tables_mapping[table_name] = {
            "columns": columns,
            "primary_key": [],
            "inferred_primary_key": [],
            "null_stats": {},
            "source": str(source_path),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "database": database,
        "schema": "file",
        "tables": tables_mapping,
        "declared_relationships": [],
        "inferred_relationships": [],
        "relationship_edges": [],
        "semi_structured_sources": {str(source_path): {"tables": sorted(tables_mapping)}},
    }


def load_tables_to_sqlite(tables, sqlite_path):
    """Load a dict of {table_name: DataFrame} into a file-based SQLite DB.

    Returns (connection_uri, [(table_name, row_count), ...]).
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to load file sources.")

    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{sqlite_path}")
    result = []
    for table_name, df in tables.items():
        df = _coerce_numeric(df)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        result.append((table_name, len(df)))
    engine.dispose()
    return f"sqlite:///{sqlite_path}", result


def build_csv_schema(csv_path, table_name=None):
    """Build a schema mapping from a CSV file (deterministic, no LLM)."""
    df = pd.read_csv(csv_path)
    df.columns = [_normalize_csv_column(c) for c in df.columns]
    if table_name is None:
        table_name = _safe_identifier(Path(csv_path).stem)
    return build_tables_schema({table_name: df}, csv_path, table_name)


def load_csv_to_sqlite(csv_path, sqlite_path, table_name=None):
    """Load a CSV into a file-based SQLite database and return its connection URI."""
    df = pd.read_csv(csv_path)
    df.columns = [_normalize_csv_column(c) for c in df.columns]
    if table_name is None:
        table_name = _safe_identifier(Path(csv_path).stem)
    uri, tables = load_tables_to_sqlite({table_name: df}, sqlite_path)
    return uri, table_name, len(df)


def extract_pdf_tables(pdf_path):
    """Extract tables from a tabular PDF into {table_name: DataFrame} (no LLM).

    Uses pdfplumber: every detected table on every page becomes its own
    SQLite table (table_1, table_2, ...). Returns {} if none found.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is required to read PDF sources. pip install pdfplumber")

    tables = {}
    table_index = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for raw in page.extract_tables():
                if not raw:
                    continue
                header = [str(c).strip() if c is not None else "" for c in raw[0]]
                rows = [[str(c).strip() if c is not None else "" for c in r] for r in raw[1:]]
                if len(rows) == 0:
                    continue
                ncols = max(len(header), max(len(r) for r in rows))
                header = (header + [""] * ncols)[:ncols]
                for r in rows:
                    r.extend([""] * (ncols - len(r)))
                header = [_normalize_csv_column(c) or f"col_{i + 1}" for i, c in enumerate(header)]
                table_index += 1
                table_name = _safe_identifier(f"{Path(pdf_path).stem}_table_{table_index}")
                df = pd.DataFrame(rows, columns=header)
                tables[table_name] = df
    return tables


def pdf_has_text(pdf_path):
    """True if the PDF has an extractable text layer (i.e. it is NOT scanned/handwritten)."""
    try:
        import pdfplumber
    except ImportError:
        return False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if (page.extract_text() or "").strip():
                    return True
    except Exception:
        return False
    return False


def extract_handwritten_pdf(pdf_path, llm, table_name=None):
    """Extract records from a scanned/handwritten PDF via a vision LLM.

    Renders every page to a PNG (pypdfium2 + Pillow), sends the images to a
    vision model asking for a structured JSON table, and returns
    {table_name: DataFrame}. Raises RuntimeError if no tabular records are found.
    """
    try:
        import base64
        import io

        import pypdfium2 as pdfium
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(f"Handwritten PDF extraction needs pypdfium2 + Pillow: {exc}")

    images = []
    doc = pdfium.PdfDocument(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=2.0)
            pil = bitmap.to_pil()
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    finally:
        doc.close()

    prompt = (
        "You are a careful data-entry clerk reading a scanned/handwritten document page.\n"
        "Extract ALL records from THIS page into tables and return ONLY valid JSON.\n"
        'The JSON must have a key "tables": a list of table objects, each with a short "name" '
        'and "rows": an array of objects, one per record. Every row object in a table must use '
        "the SAME keys (these become column names; infer them from the page headers/labels).\n"
        "Rules:\n"
        "- One JSON table object per distinct table/section on the page.\n"
        "- Transcribe the writing literally, preserving numbers and dates exactly.\n"
        "- Do not invent records that are not present. Empty cells become null.\n"
        '- If this page has NO recognizable records, return {"tables": []}.\n'
        'Example: {"tables": [{"name": "sales", "rows": [{"region": "North", "units": 120}]}]}\n'
        "Return nothing but the JSON:\n"
    )

    base = _safe_identifier(Path(pdf_path).stem)
    frames_by_schema: dict[tuple, list] = {}
    for idx, b64 in enumerate(images, start=1):
        page_prompt = f"This is page {idx} of the scanned document. {prompt}"
        try:
            text = llm.vision(page_prompt, [b64])
        except Exception as exc:
            raise RuntimeError(f"Vision model could not read the document: {exc}")
        for df in _parse_vision_frames(text):
            frames_by_schema.setdefault(tuple(df.columns), []).append(df)

    if not frames_by_schema:
        raise RuntimeError(
            "Could not recognize any tabular records in the handwritten PDF. "
            "The file was rejected — please upload a clearer scan or a text-based PDF."
        )

    tables = {}
    multi = len(frames_by_schema) > 1
    for i, (key, dfs) in enumerate(frames_by_schema.items(), start=1):
        merged = pd.concat(dfs, ignore_index=True)
        name = f"{base}_handwritten" if not multi else f"{base}_handwritten_{i}"
        tables[name] = merged
    return tables


def _parse_vision_frames(text):
    """Parse a vision response into one or more DataFrames (supports single 'rows'
    or multi-table 'tables' payloads)."""
    import re as _re

    if not text:
        return []
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except Exception:
        arr = _re.search(r"\[.*\]", candidate, _re.DOTALL)
        if not arr:
            return []
        try:
            data = json.loads(arr.group(0))
        except Exception:
            return []

    frames = []
    if isinstance(data, dict):
        tables = data.get("tables")
        if isinstance(tables, list) and tables:
            for tbl in tables:
                rows = tbl.get("rows", tbl) if isinstance(tbl, dict) else tbl
                if isinstance(rows, list) and rows:
                    frames.append(pd.DataFrame(rows))
        else:
            rows = data.get("rows", [])
            if isinstance(rows, dict):
                rows = rows.get("rows", [])
            if isinstance(rows, list) and rows:
                frames.append(pd.DataFrame(rows))
    elif isinstance(data, list) and data:
        frames.append(pd.DataFrame(data))

    cleaned = []
    for df in frames:
        if len(df.columns) == 0:
            continue
        df.columns = [_normalize_csv_column(c) for c in df.columns]
        cleaned.append(df)
    return cleaned


def _normalize_csv_column(name):
    return re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_") or "column"


def _safe_identifier(name):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip()).strip("_") or "csv_table"


def main():
    parser = argparse.ArgumentParser(description="ARIA Schema Agent - automated schema understanding")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--db-type", choices=["postgresql", "mysql"], default="postgresql")
    parser.add_argument("--output", default=str(SCHEMA_DIR / "schema_mapping.json"))
    parser.add_argument("--json-source", action="append", default=[], help="JSON file(s) to parse")
    parser.add_argument("--csv-source", action="append", default=[], help="CSV file(s) to parse")
    parser.add_argument("--pdf-source", action="append", default=[], help="PDF/report file(s) to parse")
    parser.add_argument(
        "--provider", choices=["local", "cloud"], default=None,
        help="Enable LLM-assisted reasoning over the extracted schema via this backend "
             "(local = Ollama, cloud = Groq). "
             "When omitted, only deterministic heuristics are used.",
    )
    args = parser.parse_args()

    semi_sources = args.json_source + args.csv_source + args.pdf_source

    # MySQL has no named schemas separate from the database: its
    # information_schema uses the database name as the "schema". Resolve the
    # filter the same way the API route does, so `--db-type mysql --db X`
    # works out of the box instead of silently returning zero tables.
    schema = args.schema or (args.db if args.db_type == "mysql" else "public")

    llm = None
    if args.provider:
        llm = create_provider(provider=args.provider)

    conn = get_connection(args)
    try:
        print(f"Connected ({args.db_type}). Extracting schema for '{schema}' schema...")
        mapping = build_schema_mapping(conn, schema, semi_sources, args.db_type, llm=llm,
                                       database_name=args.db)
        if not semi_sources and not mapping.get("tables"):
            sys.exit(
                f"ERROR: no tables found in schema '{schema}' ({args.db_type}). "
                "The schema is empty or does not exist; nothing was mapped."
            )
        db_name = mapping.get("database", args.db or "unknown")
    finally:
        conn.close()

    output_path, timestamped_path, latest_path, snapshot_id = get_output_paths(
        args.output, db_name or None, snapshot_id=mapping.get("snapshot_id")
    )
    mapping["snapshot_id"] = snapshot_id

    for path in (output_path, timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    n_tables = len(mapping["tables"])
    n_declared = len(mapping["declared_relationships"])
    n_inferred = len(mapping["inferred_relationships"])
    print(f"Done. {n_tables} tables, {n_declared} declared relationships, "
          f"{n_inferred} inferred relationships.")
    if mapping.get("llm_reasoning"):
        print(f"LLM-assisted reasoning applied via {mapping['llm_reasoning'].get('provider')} "
              f"(model {mapping['llm_reasoning'].get('model')}).")
    print(f"Written to {output_path}")
    print(f"Saved fresh snapshot to {timestamped_path}")
    print(f"Updated latest copy at {latest_path}")


if __name__ == "__main__":
    main()
