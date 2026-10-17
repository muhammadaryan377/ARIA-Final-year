"""Shared schema intelligence — dynamic column classification using data_type.

Used by both KPI routes and Insights engine. Zero hardcoded table/column names.
Classifies columns by their actual database data_type from the schema metadata.
"""

import re


def _name_tokens(name: str) -> list:
    """Lowercase tokens, separator-agnostic (snake_case, spaces, camelCase)."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name or ""))
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _is_id_like(name: str) -> bool:
    """Identifier-like columns ("order_id", "Order ID", "postal_code").

    Token equality — never substrings — so "country" is not an ID.
    """
    toks = _name_tokens(name)
    if not toks:
        return False
    if toks == ["id"]:
        return True
    return toks[-1] in ("id", "uuid", "guid", "key", "code")


def _hint_hit(name: str, hints) -> bool:
    """Any name token equals a hint (singular/plural tolerant)."""
    for t in _name_tokens(name):
        for h in (hints or ()):
            hl = str(h or "").lower()
            if not hl:
                continue
            if t == hl or t == hl + "s" or (t.endswith("s") and len(t) > 3 and t[:-1] == hl):
                return True
    return False

# ── Type classification (fully dynamic from data_type) ────────────────────

NUMERIC_TYPES = {
    "smallint", "integer", "bigint", "decimal", "numeric",
    "real", "double precision", "float", "money",
    "tinyint", "int",
}
DATETIME_TYPES = {
    "date", "time", "timestamp", "timestamp without time zone",
    "timestamp with time zone", "datetime", "smalldatetime",
}
TEXT_TYPES = {
    "character varying", "varchar", "character", "char",
    "text", "citext", "ltree", "uuid",
}
BINARY_TYPES = {"bytea", "blob", "binary", "varbinary", "image"}

# Binary/blob-like column-name tokens. A column whose name carries one of
# these tokens (e.g. product_photo, picture, thumbnail, counter internals)
# is never an analytical measure, even when its declared type looks
# numeric. Token equality (never bare substrings) so "encounter" does not
# trip on "counter".
_BINARY_NAME_TOKENS = frozenset({
    "binary", "blob", "bytea", "bytes", "image", "photo", "picture",
    "thumb", "thumbnail", "thumbs", "icon", "counter", "rowversion",
})

# Binary/blob-like type fragments (substring match on the data_type).
_BINARY_TYPE_TOKENS = (
    "bytea", "blob", "binary", "varbinary", "bytes", "image",
)


def _is_binary_like_name(name: str) -> bool:
    """True when a column name carries a binary/blob/image token."""
    return any(t in _BINARY_NAME_TOKENS for t in _name_tokens(name))


def _is_binary_like_type(data_type: str) -> bool:
    """True when a declared data_type is binary/blob-like."""
    dt = str(data_type or "").lower().strip()
    return any(k in dt for k in _BINARY_TYPE_TOKENS)

# Amount-like column name hints (used to prefer revenue columns over IDs)
AMOUNT_HINTS = (
    "price", "cost", "amount", "total", "sum", "revenue",
    "sales", "income", "expense", "profit", "value",
    "quantity", "qty", "discount", "freight", "tax",
    "payment", "balance", "weight", "unit_price", "margin",
    "turnover", "billing", "charge", "fee", "rate",
)

# Store-like entity column hints (retail: Store/branch/outlet counts).
# Used to surface entity KPIs on single-table sources where the store column
# is numeric (INTEGER) rather than TEXT.
STORE_HINTS = (
    "store", "shop", "branch", "outlet", "location", "site",
    "warehouse", "depot",
)

# Date-like name hints: a date column drives trends, it is not the headline
# breakdown for "which X has the highest ...". Token-based (never bare
# substrings) so "holiday" does not trip on "day".
_DATE_NAME_HINTS = (
    "date", "time", "day", "month", "year", "created",
    "updated", "timestamp", "period", "quarter", "week",
    "shipped", "ordered",
)

# Holiday/flag column hints (retail: Holiday_Flag lift analysis).
HOLIDAY_HINTS = (
    "holiday", "is_holiday", "holiday_flag", "festival",
    "event_flag", "promo_flag", "is_promo",
)


def classify_type(data_type: str) -> str:
    """Classify a database column type into a semantic category."""
    dt = data_type.lower().strip()
    if _is_binary_like_type(dt) or dt in BINARY_TYPES:
        return "binary"
    if dt in NUMERIC_TYPES or any(k in dt for k in (
        "int", "decimal", "numeric", "float", "real", "double", "money"
    )):
        return "numeric"
    if dt in DATETIME_TYPES or any(k in dt for k in ("date", "time", "timestamp")):
        return "datetime"
    if dt in TEXT_TYPES or any(k in dt for k in ("char", "text", "varchar", "uuid")):
        return "text"
    if dt in BINARY_TYPES:
        return "binary"
    return "other"


def _safe_name(name: str, dialect: str = "postgresql") -> str:
    """Quote an identifier for the given dialect (escapes embedded quotes).

    Schema-qualified keys (sales.employee) quote per segment
    ("sales"."employee"); bare names are unchanged.
    """
    try:
        segs = [s for s in str(name or "").split(".") if s]
        if len(segs) > 1:
            if dialect == "mysql":
                return ".".join(f"`{s.replace('`', '``')}`" for s in segs)
            return ".".join(
                f'"{s.replace(chr(34), chr(34) + chr(34))}"' for s in segs)
    except Exception:
        pass
    safe = str(name or "")
    if dialect == "mysql":
        return f"`{safe.replace('`', '``')}`"
    return f'"{safe.replace(chr(34), chr(34) + chr(34))}"'


def _run_sql(conn, sql: str, limit: int = 500):
    """Execute read-only SQL and return (columns, rows_as_dicts).

    Distinguishes empty from broken: DB errors are logged with the
    statement prefix and rolled back, then ([], []) is returned ONLY for
    missing-relation/undefined cases. Programming/connection errors still
    return ([], []) but are logged at warning level with the SQL so silent
    detector blindness is diagnosable server-side.
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit)
        cur.close()
        return cols, [dict(zip(cols, row)) for row in rows]
    except Exception as exc:
        import logging as _logging
        _logging.getLogger("aria.schema_intel").warning(
            "sql failed (returning empty): sql=%.200s error=%s", sql, exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return [], []


class SchemaIntelligence:
    """Analyze any database schema using data_type, PKs, and FKs.

    Works with any PostgreSQL/MySQL/SQLite database. Zero hardcoded assumptions
    about table names, column names, or schema structure.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.tables = mapping.get("tables", {})
        self.fks = mapping.get("declared_relationships", [])
        self.edges = mapping.get("relationship_edges", [])

        self._pk_map = {}
        self._fk_map = {}
        self._col_types = {}
        self._col_category = {}
        self._table_categories = {}

        self._build_lookups()

    def _build_lookups(self):
        for tname, tinfo in self.tables.items():
            pks = tinfo.get("primary_key", []) or tinfo.get("inferred_primary_key", [])
            self._pk_map[tname] = pks

            cols_raw = tinfo.get("columns", [])
            col_names = []
            if isinstance(cols_raw, list):
                if len(cols_raw) > 0 and isinstance(cols_raw[0], str):
                    col_names = [str(c).strip() for c in cols_raw if c]
                    synthetic_cols = [{"column": name, "data_type": "unknown"} for name in col_names]
                    cols_raw = synthetic_cols
                else:
                    col_names = [str(c.get("column", "")).strip() for c in cols_raw if isinstance(c, dict) and c.get("column")]
            elif isinstance(cols_raw, dict):
                col_names = [str(k).strip() for k in cols_raw.keys() if k]
                synthetic_cols = [{"column": str(k).strip(), "data_type": "unknown"} for k in cols_raw.keys()]
                cols_raw = synthetic_cols

            cat_map = {"numeric": [], "text": [], "datetime": [], "binary": [], "other": []}
            for c in cols_raw:
                if isinstance(c, dict):
                    cname = str(c.get("column", "")).strip()
                    dtype = str(c.get("data_type", "")).lower().strip()
                elif isinstance(c, str):
                    cname = str(c).strip()
                    dtype = "unknown"
                else:
                    continue
                if not cname:
                    continue
                cat = classify_type(dtype)
                self._col_types.setdefault(tname, {})[cname] = dtype
                self._col_category.setdefault(tname, {})[cname] = cat
                cat_map[cat].append(cname)

            if not cat_map.get("numeric") and col_names:
                cat_map = {"numeric": [], "text": [], "datetime": [], "binary": [], "other": []}
                for cname in col_names:
                    if _hint_hit(cname, ("amount", "price", "cost", "total", "value", "revenue", "sales", "qty", "quantity", "count")):
                        cat_map["numeric"].append(cname)
                    elif _hint_hit(cname, ("date", "time", "created", "updated", "timestamp")):
                        cat_map["datetime"].append(cname)
                    elif _is_id_like(cname) or cname.lower() in ("key", "code"):
                        cat_map["other"].append(cname)
                    else:
                        cat_map["text"].append(cname)
                # Preserve the mapping's real dtypes (e.g. "text" for a
                # relational TEXT column) instead of blanking them: dtype
                # drives TEXT-numeric coercion downstream, and "inferred"
                # would hide a genuine text type. Only truly unknown dtypes
                # keep the "inferred" marker.
                _orig_dtype: dict[str, str] = {}
                try:
                    for _c in (cols_raw if isinstance(cols_raw, list) else []):
                        if isinstance(_c, dict) and str(_c.get("column", "")).strip():
                            _orig_dtype[str(_c.get("column")).strip()] = str(
                                _c.get("data_type", "") or "")
                except Exception:
                    _orig_dtype = {}
                for cname in col_names:
                    low = cname.lower()
                    if cname in cat_map.get("numeric", []):
                        cat = "numeric"
                    elif cname in cat_map.get("datetime", []):
                        cat = "datetime"
                    elif cname in cat_map.get("binary", []):
                        cat = "binary"
                    elif cname in cat_map.get("other", []):
                        cat = "other"
                    else:
                        cat = "text"
                    self._col_types.setdefault(tname, {})[cname] = (
                        _orig_dtype.get(cname) or "inferred")
                    self._col_category.setdefault(tname, {})[cname] = cat
            self._table_categories[tname] = cat_map

        for fk in self.fks:
            t = fk.get("table_name", "")
            col = fk.get("column_name", "")
            ref_t = fk.get("references_table", "")
            ref_c = fk.get("references_column", "")
            if t and col and ref_t and ref_c:
                self._fk_map.setdefault(t, []).append((col, ref_t, ref_c))

    def dtype_of(self, table: str, col: str) -> str:
        """Declared data_type for one column ("" when unknown).

        Schema-agnostic accessor for TEXT-numeric coercion decisions:
        callers coerce SUM/AVG over text-dtype measures instead of failing
        with ``function sum(text) does not exist``. Never raises.
        """
        try:
            return str((self._col_types.get(table, {}) or {}).get(col, "") or "")
        except Exception:
            return ""

    def get_numeric_cols(self, table: str) -> list:
        return self._table_categories.get(table, {}).get("numeric", [])

    def get_text_cols(self, table: str) -> list:
        return self._table_categories.get(table, {}).get("text", [])

    def get_datetime_cols(self, table: str) -> list:
        return self._table_categories.get(table, {}).get("datetime", [])

    def get_pk(self, table: str) -> list:
        return self._pk_map.get(table, [])

    def get_fks(self, table: str) -> list:
        return self._fk_map.get(table, [])

    def get_col_names(self, table: str) -> list:
        cols_raw = self.tables.get(table, {}).get("columns", [])
        if isinstance(cols_raw, list):
            return [c.get("column", "") for c in cols_raw if isinstance(c, dict)]
        elif isinstance(cols_raw, dict):
            return list(cols_raw.keys())
        return []

    def is_fact_table(self, table: str) -> bool:
        numerics = self.get_numeric_cols(table)
        fks = self.get_fks(table)
        return len(numerics) >= 1 and len(fks) >= 1

    def is_dimension_table(self, table: str) -> bool:
        texts = self.get_text_cols(table)
        pks = self.get_pk(table)
        outgoing_fks = self.get_fks(table)
        return len(texts) >= 1 and len(pks) >= 1 and len(outgoing_fks) <= 1

    def _usable_measure_cols(self, table: str) -> list:
        """Numeric columns verified safe for SUM/AVG.

        Excludes binary/blob/bytea/image/photo/picture columns by type AND
        by name (a photo/thumbnail/counter column is never a measure, even
        when its declared type looks numeric), plus PK/FK/identifier keys,
        plus calendar-year columns (release_year: SUM of years is nonsense;
        years stay usable as dimensions and trend axes).
        """
        from core.semantic import is_year_like as _is_year_like
        col_types = self._col_types.get(table, {})
        return [c for c in self.get_numeric_cols(table)
                if c not in set(self.get_pk(table))
                and c not in set(fk[0] for fk in self.get_fks(table))
                and not _is_id_like(c)
                and not _is_binary_like_name(c)
                and not _is_binary_like_type(col_types.get(c, ""))
                and not _is_year_like(c)]

    def find_best_measure(self, table: str):
        """Pick the most analytically useful numeric column (prefer amounts over IDs).

        Binary/blob/bytea/image/photo/picture columns are excluded by type
        AND by name — SUM over a photo column is always wrong.
        """
        numerics = self._usable_measure_cols(table)
        if not numerics:
            return None

        # Tier 1: amount-like names, not PK/FK
        tier1 = [c for c in numerics
                 if any(h in c.lower() for h in AMOUNT_HINTS)]
        if tier1:
            return tier1[0]

        # Tier 2: non-PK/FK, non-ID columns
        if numerics:
            return numerics[0]

        return None

    def find_all_measures(self, table: str):
        """Return all usable numeric measure columns for a table."""
        return list(self._usable_measure_cols(table))

    def find_best_dimension(self, table: str):
        """Pick the most descriptive text column for grouping (skip IDs).

        Date-named text columns are skipped while a non-date alternative
        exists — a date drives trends, it is not the headline answer to
        "which X has the highest ...". Falls back to the date column when
        it is the only text available.
        """
        texts = self.get_text_cols(table)
        if not texts:
            # Single-table retail sources (e.g. Walmart CSV): the entity
            # column (Store) is numeric, not text. Fall back to a store-like
            # numeric column so entity counts and breakdowns still work.
            for c in self.get_numeric_cols(table):
                low = c.lower()
                if any(h in low for h in STORE_HINTS):
                    return c
            return None
        pks = set(self.get_pk(table))
        fk_cols = set(fk[0] for fk in self.get_fks(table))
        usable = [c for c in texts
                  if c not in pks and c not in fk_cols
                  and not _is_id_like(c)]
        if usable:
            non_date = [c for c in usable if not _hint_hit(c, _DATE_NAME_HINTS)]
            return (non_date or usable)[0]
        non_pk = [c for c in texts if c not in pks]
        if non_pk:
            non_date = [c for c in non_pk if not _hint_hit(c, _DATE_NAME_HINTS)]
            return (non_date or non_pk)[0]
        return texts[0] if texts else None

    def find_store_column(self, table: str):
        """Find a low-cardinality store-like entity column (any type).

        Checks text columns first, then numeric columns with store-like
        names (retail CSVs store Store as INTEGER). Returns None when no
        store-like column exists — never guesses an unrelated column.
        """
        texts = self.get_text_cols(table)
        pks = set(self.get_pk(table))
        for c in texts:
            if any(h in c.lower() for h in STORE_HINTS) and c not in pks:
                return c
        for c in self.get_numeric_cols(table):
            low = c.lower()
            if low in pks or low == "id" or low.endswith("_id"):
                continue
            if any(h in low for h in STORE_HINTS):
                return c
        return None

    def find_holiday_column(self, table: str):
        """Find a holiday/flag column for lift analysis (any type).

        Returns None when no holiday-like column exists.
        """
        for t in (self.get_numeric_cols(table), self.get_text_cols(table)):
            for c in t:
                if any(h in c.lower() for h in HOLIDAY_HINTS):
                    return c
        # Generic boolean-ish flag fallback: 0/1 integer columns whose name
        # suggests a flag but which were not claimed as measures.
        for c in self.get_numeric_cols(table):
            low = c.lower()
            if low in ("id",) or low.endswith("_id"):
                continue
            if "flag" in low or low.startswith("is_") or low.startswith("has_"):
                return c
        return None

    def find_all_dimensions(self, table: str):
        """Return all usable text dimension columns.

        Measure-like names (amount/price/total/...) are never segments —
        even when the source types them as TEXT (CSV money arrives as TEXT
        in SQLite). See hyp.txt §4: a numeric measure is not a segment.
        """
        texts = self.get_text_cols(table)
        pks = set(self.get_pk(table))
        fk_cols = set(fk[0] for fk in self.get_fks(table))
        return [c for c in texts
                if c not in pks and c not in fk_cols
                and not _is_id_like(c)
                and not _hint_hit(c, AMOUNT_HINTS)]

    def find_best_datetime(self, table: str):
        """Pick the most analytically useful datetime column."""
        dts = self.get_datetime_cols(table)
        if dts:
            # No hardcoded name preferences — just return the first datetime column
            return dts[0]
        # Fallback for sources with weak typing (CSV dates arrive as TEXT):
        # a text column whose NAME is date-like is usable with strftime/date_trunc.
        date_hints = ("date", "time", "day", "month", "year", "created",
                      "updated", "timestamp", "period")
        for col in self.get_text_cols(table):
            low = col.lower()
            if any(h in low for h in date_hints):
                return col
            # Bare abbreviations (ts/dt/...) only on word boundaries —
            # avoids matching "hits", "parts", "assets".
            import re as _re
            if _re.search(r"(^|_)(ts|dt|dttm|datetime|ymd)(_|$)", low):
                return col
        return None

    def find_join_path(self, from_table: str, to_table: str) -> list:
        """Find a join path using FK relationships.
        Returns list of (from_table, from_col, to_table, to_col)."""
        if from_table == to_table:
            return []

        for fk_col, ref_table, ref_col in self.get_fks(from_table):
            if ref_table == to_table:
                return [(from_table, fk_col, to_table, ref_col)]

        for fk_col, ref_table, ref_col in self.get_fks(to_table):
            if ref_table == from_table:
                return [(to_table, fk_col, from_table, ref_col)]

        for fk1_col, ref1_table, ref1_col in self.get_fks(from_table):
            for fk2_col, ref2_table, ref2_col in self.get_fks(ref1_table):
                if ref2_table == to_table:
                    return [
                        (from_table, fk1_col, ref1_table, ref1_col),
                        (ref1_table, fk2_col, to_table, ref2_col),
                    ]

        return []

    def build_join_clause(self, joins: list, dialect: str = "postgresql") -> str:
        """Build SQL JOIN clause from a list of join steps."""
        clauses = []
        for from_t, from_c, to_t, to_c in joins:
            clauses.append(
                f'JOIN {_safe_name(to_t, dialect)} ON '
                f'{_safe_name(from_t, dialect)}.{_safe_name(from_c, dialect)} = '
                f'{_safe_name(to_t, dialect)}.{_safe_name(to_c, dialect)}'
            )
        return " ".join(clauses)

    def get_table_row_count(self, table: str) -> bool:
        """Check if table has a row_count in metadata."""
        info = self.tables.get(table, {})
        return info.get("row_count", 0) if "row_count" in info else None
