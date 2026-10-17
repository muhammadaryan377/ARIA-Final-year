"""ARIA universal Data Intelligence Layer (n.txt items 1, 11, 12 — P0 foundation).

Single shared contract consumed by Insights, Forecast, Prediction,
Prescription, RAG and Copilot so no agent re-discovers the dataset shape.

Pure and deterministic: no LLM calls, no I/O. Works for every source ARIA
supports (PostgreSQL, MySQL, CSV/SQLite, AdventureWorks-style multi-schema)
because it consumes the normalized schema-mapping dict, not engine internals:

    mapping["tables"][name] = {
        "columns": [{"column": ..., "data_type": ..., "nullable": ...}, ...]
                   | {name: {"data_type": ..., "nullable": ...}},
        "primary_key": [...], "inferred_primary_key": [...],
        "null_stats": {col: {"total_rows": n, "nulls": n, "null_pct": p}},
        "row_count": n, "empty": bool,
    }
    mapping["declared_relationships"] / ["inferred_relationships"] /
            ["relationship_edges"]  (any may be empty; empty == valid)

Key guarantees:
- 1 table, 2-10 tables and many tables all flow through this SAME path.
  There is deliberately no single-table special case anywhere in this module.
- ``relationships == []`` / ``relational_analysis == False`` is a VALID
  outcome, never an error. Downstream agents must read the capability flags
  instead of assuming a JOIN exists.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Shared type vocabularies (PostgreSQL + MySQL + SQLite + CSV logical types).
# ---------------------------------------------------------------------------

_NUMERIC_TOKENS = (
    "int", "bigint", "smallint", "serial", "bigserial",
    "numeric", "decimal", "number", "float", "double", "real",
    "money", "single", "byte",
)
# NOTE: "single"/"byte" are MS-Access-heritage tokens seen in some SQLite
# dumps; harmless as substring hints.
_DATETIME_TOKENS = ("date", "time", "timestamp", "datetime", "interval", "year")
_BOOLEAN_TOKENS = ("bool", "boolean", "bit")
_TEXT_TOKENS = ("char", "varchar", "text", "citext", "string", "uuid",
                "clob", "enum", "name")

_ID_NAME_RE = re.compile(r"(?:^|_)(?:id|uuid|guid|p?key|code)$", re.IGNORECASE)
_DATE_NAME_HINTS = ("date", "time", "day", "month", "year", "created",
                    "updated", "timestamp", "period", "shipped", "ordered")

# Numeric columns whose NAME signals a category/flag rather than a math
# measure (store id, holiday flag, ...). Token-based (never bare
# substrings): a column is dimension-like when its LAST token is
# dimension-ish ("holiday_flag", "store"), while a trailing measure token
# ("store_sales", "brand_revenue") keeps measure status. Generic name
# fragments only — never a table/column literal.
_DIMENSION_LIKE_NUMERIC_TOKENS = (
    "year", "month", "day", "status", "state", "code", "rank", "rating",
    "level", "version", "quarter", "age",
    "store", "flag", "holiday", "brand", "category", "segment", "region",
    "channel", "department", "division", "group", "type", "class", "gender",
    "outlet", "branch", "location", "site", "shop",
)
_MEASURE_TAIL_TOKENS = (
    "sales", "revenue", "profit", "amount", "total", "price", "cost",
    "margin", "income", "expense", "balance", "quantity", "qty", "count",
    "value", "rate", "score", "hours", "salary", "wage",
)


def name_tokens(column_name: str) -> list:
    """Split a column/table name into lowercase tokens, separator-agnostic.

    Handles snake_case, spaces, dots AND camelCase ("OrderDate" ->
    ["order", "date"]) so name-hint matching works on every naming
    convention instead of only underscore-joined names.
    """
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(column_name or ""))
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def is_id_like_name(column_name: str) -> bool:
    """True when a column name is identifier-like (id/key/code/uuid/guid).

    Separator-agnostic: "order_id", "Order ID" and "postal_code" all match,
    while "country", "monkey" or "barcode" do not (token equality, never
    bare substrings).
    """
    toks = name_tokens(column_name)
    if not toks:
        return False
    if toks == ["id"]:
        return True
    return toks[-1] in ("id", "uuid", "guid", "key", "code")


def hint_hit(column_name: str, hints) -> bool:
    """True when any token of the name equals a hint (singular/plural).

    Token equality — never substring — so "count" does not fire on
    "country", "order" does not fire on "order_status", and "class"
    does not fire on "classification".
    """
    for t in name_tokens(column_name):
        for h in (hints or ()):
            hl = str(h or "").lower()
            if not hl:
                continue
            if t == hl or t == hl + "s" or (t.endswith("s") and len(t) > 3 and t[:-1] == hl):
                return True
    return False


def is_dimension_like_numeric(column_name: str) -> bool:
    """True when a NUMERIC column's name signals a dimension, not a measure."""
    toks = [t for t in re.split(r"[^a-z0-9]+", str(column_name or "").lower()) if t]
    if not toks:
        return False
    if toks[-1] in _MEASURE_TAIL_TOKENS:
        return False
    return any(t in _DIMENSION_LIKE_NUMERIC_TOKENS for t in toks)


def quote_identifier(name: Any, dialect: str = "postgresql") -> str:
    """Dialect-aware identifier quoting (shared path, no hardcoding)."""
    safe = str(name or "")
    if (dialect or "") == "mysql":
        return f"`{safe.replace('`', '``')}`"
    return f'"{safe.replace(chr(34), chr(34) + chr(34))}"'


def is_text_date_column(column_name: str, data_type: Any) -> bool:
    """True when a column is a datetime concept stored as TEXT.

    Covers CSV/SQLite ISO TEXT dates AND DD-MM-YYYY TEXT dates (Walmart):
    the shared classifier already maps date-named TEXT to datetime, so this
    is just the explicit predicate SQL builders need.
    """
    try:
        if classify_column(column_name, data_type) != "datetime":
            return False
        dtype = _norm_dtype(data_type)
        return any(tok in dtype for tok in _TEXT_TOKENS) or not dtype
    except Exception:
        low = str(column_name or "").lower()
        return any(h in low for h in _DATE_NAME_HINTS)


def date_parse_expression(quoted_col: str, dialect: str = "postgresql") -> str:
    """Parse a TEXT date mixing DD-MM-YYYY and ISO (Walmart) into a real date.

    `quoted_col` must already be quoted for `dialect`. Each branch first
    tries the DD-MM-YYYY shape, then falls back to the native ISO parse, so
    05-02-2010 and 2010-02-05 both work and bad values become NULL (never an
    error). Shared by Goal Agent + Copilot so trends work on every dialect.
    """
    col = quoted_col
    if (dialect or "") == "mysql":
        return (
            f"CASE WHEN {col} REGEXP '^[0-9]{{2}}-[0-9]{{2}}-[0-9]{{4}}$' "
            f"THEN STR_TO_DATE({col}, '%d-%m-%Y') "
            f"WHEN {col} REGEXP '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' "
            f"THEN CAST({col} AS DATE) "
            f"WHEN {col} REGEXP '^[0-9]{{4}}-[0-9]{{2}}$' "
            f"THEN CAST(CONCAT({col}, '-01') AS DATE) ELSE NULL END"
        )
    if (dialect or "") == "sqlite":
        iso = (
            f"substr({col}, 7, 4) || '-' || substr({col}, 4, 2) "
            f"|| '-' || substr({col}, 1, 2)"
        )
        return (
            f"CASE WHEN substr({col}, 3, 1) = '-' AND length({col}) >= 10 "
            f"THEN date({iso}) "
            f"WHEN length({col}) = 7 AND {col} GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]' "
            f"THEN date({col} || '-01') ELSE date({col}) END"
        )
    return (
        f"CASE WHEN {col} ~ '^[0-9]{{2}}-[0-9]{{2}}-[0-9]{{4}}$' "
        f"THEN TO_DATE({col}, 'DD-MM-YYYY') "
        f"WHEN {col} ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' "
        f"THEN {col}::date "
        f"WHEN {col} ~ '^[0-9]{{4}}-[0-9]{{2}}$' "
        f"THEN ({col} || '-01')::date ELSE NULL END"
    )


def month_trunc_expression(quoted_col: str, data_type: Any,
                           dialect: str = "postgresql") -> str:
    """Month-grain expression for a date column, TEXT-date safe.

    Native date/datetime columns use the fast native truncation; TEXT dates
    go through date_parse_expression() first (DD-MM-YYYY + ISO fallback).
    """
    dtype = _norm_dtype(data_type)
    is_text = any(tok in dtype for tok in _TEXT_TOKENS) or not dtype
    expr = date_parse_expression(quoted_col, dialect) if is_text else quoted_col
    if (dialect or "") == "mysql":
        return f"DATE_FORMAT({expr}, '%Y-%m')"
    if (dialect or "") == "sqlite":
        return f"strftime('%Y-%m', {expr})"
    return f"TO_CHAR({expr}, 'YYYY-MM')"


def _norm_dtype(dtype: Any) -> str:
    return str(dtype or "").strip().lower()


def classify_column(column_name: str, data_type: Any = "") -> str:
    """Classify one column into a coarse role.

    Returns one of: ``identifier`` | ``datetime`` | ``boolean`` |
    ``numerical`` | ``categorical``. ``categorical`` covers low-cardinality
    text-like dimensions; free-form text also lands here (downstream agents
    refine with cardinality when available) so callers never have to handle
    an ``UNKNOWN`` role.
    """
    name = str(column_name or "").strip().lower()
    dtype = _norm_dtype(data_type)
    if any(tok in dtype for tok in _BOOLEAN_TOKENS) or dtype in ("boolean", "bit"):
        return "boolean"
    if any(tok in dtype for tok in _DATETIME_TOKENS):
        return "datetime"
    if any(tok in dtype for tok in _NUMERIC_TOKENS):
        if is_id_like_name(name):
            return "identifier"
        return "numerical"
    if any(tok in dtype for tok in _TEXT_TOKENS) or not dtype:
        if is_id_like_name(name):
            return "identifier"
        if hint_hit(name, _DATE_NAME_HINTS):
            return "datetime"
        return "categorical"
    # Unknown vendor type: fall back to the name.
    if is_id_like_name(name):
        return "identifier"
    if hint_hit(name, _DATE_NAME_HINTS):
        return "datetime"
    return "categorical"


def _iter_columns(table_info: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    """Normalize both column shapes to [(name, data_type, nullable)]."""
    columns = table_info.get("columns", [])
    out: List[Tuple[str, str, bool]] = []
    if isinstance(columns, dict):
        for name, meta in columns.items():
            meta = meta or {}
            out.append((str(name), str(meta.get("data_type", "TEXT")),
                        bool(meta.get("nullable", True))))
    else:
        for col in columns or []:
            if isinstance(col, dict):
                out.append((str(col.get("column", "")),
                            str(col.get("data_type", "TEXT")),
                            bool(col.get("nullable", True))))
            else:
                out.append((str(col), "TEXT", True))
    return out


def _primary_key(table_info: Dict[str, Any]) -> List[str]:
    pk = list(table_info.get("primary_key", []) or [])
    if not pk:
        pk = list(table_info.get("inferred_primary_key", []) or [])
    if not pk:
        # Generic composite fallback: a fully-populated unique pair exposed
        # by profilers (CSV handler / schema agent) counts as the table's
        # key for intelligence (measures vs dimensions) when no single
        # column is unique. Never hardcoded names.
        pk = list(table_info.get("inferred_composite_primary_key", []) or [])
    if not pk:
        try:
            _cck = table_info.get("composite_candidate_keys", []) or []
            if _cck and isinstance(_cck[0], (list, tuple)):
                pk = [str(c) for c in _cck[0]]
        except Exception:
            pk = []
    return [str(c) for c in pk]


def _row_count(table_name: str, table_info: Dict[str, Any]) -> int | None:
    if isinstance(table_info.get("row_count"), int):
        return table_info["row_count"]
    for stats in (table_info.get("null_stats") or {}).values():
        if isinstance(stats, dict) and isinstance(stats.get("total_rows"), int):
            return stats["total_rows"]
    return None


# Profilers may attach extra per-column evidence under these generic keys
# (null_stats entries, column dicts or their ``statistics`` sub-dicts).
# Only these evidence keys are read — never table/column literals.
_DISTINCT_EVIDENCE_KEYS = (
    "distinct_count", "distinct", "distinct_values",
    "n_distinct", "ndistinct", "unique_count",
)
_VARIANCE_EVIDENCE_KEYS = ("variance", "var")
_STD_EVIDENCE_KEYS = ("std", "stddev", "stdev", "std_dev")
_MIN_EVIDENCE_KEYS = ("min", "minimum", "lo", "low")
_MAX_EVIDENCE_KEYS = ("max", "maximum", "hi", "high")


def _evidence_number(entry: Any, keys: Tuple[str, ...]) -> float | int | None:
    if not isinstance(entry, dict):
        return None
    for key in keys:
        try:
            value = entry.get(key)
        except AttributeError:
            value = None
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _column_entry(table_info: Dict[str, Any], col_name: str) -> Dict[str, Any]:
    cols = table_info.get("columns", [])
    if isinstance(cols, dict):
        meta = cols.get(col_name)
        return meta if isinstance(meta, dict) else {}
    for item in cols or []:
        if isinstance(item, dict) and str(
                item.get("column") or item.get("name") or "") == str(col_name):
            return item
    return {}


def column_target_evidence(table_info: Dict[str, Any],
                           col_name: str) -> Dict[str, Any]:
    """Per-column evidence for prediction validation (pure, no I/O).

    Reads only profiler-supplied evidence already present in the mapping
    (null counts, distinct counts, variance/std, min/max statistics).
    Missing evidence stays ``None`` (unknown, never negative evidence).
    """
    raw_ns = (table_info.get("null_stats") or {}).get(col_name)
    ns = raw_ns if isinstance(raw_ns, dict) else {}
    entry = _column_entry(table_info, col_name)
    stats = ns.get("statistics") if isinstance(ns.get("statistics"), dict) else {}
    if not stats and isinstance(entry.get("statistics"), dict):
        stats = entry["statistics"]

    non_null: int | None = None
    total, nulls = ns.get("total_rows"), ns.get("nulls")
    if (isinstance(total, (int, float)) and not isinstance(total, bool)
            and isinstance(nulls, (int, float)) and not isinstance(nulls, bool)):
        try:
            non_null = int(total) - int(nulls)
        except (TypeError, ValueError):
            non_null = None
    if non_null is None:
        for bucket in (ns, entry):
            found = _evidence_number(
                bucket, ("non_null", "non_null_count", "nonnull", "count"))
            if isinstance(found, (int, float)):
                non_null = int(found)
                break

    distinct = _evidence_number(ns, _DISTINCT_EVIDENCE_KEYS)
    if distinct is None:
        distinct = _evidence_number(entry, _DISTINCT_EVIDENCE_KEYS)
    if distinct is None:
        distinct = _evidence_number(stats, _DISTINCT_EVIDENCE_KEYS)
    if isinstance(distinct, float):
        distinct = int(distinct)

    variance: float | None = None
    found_var = _evidence_number(ns, _VARIANCE_EVIDENCE_KEYS)
    if found_var is None:
        found_var = _evidence_number(entry, _VARIANCE_EVIDENCE_KEYS)
    if found_var is None:
        found_var = _evidence_number(stats, _VARIANCE_EVIDENCE_KEYS)
    if isinstance(found_var, (int, float)):
        variance = float(found_var)
    if variance is None:
        for bucket in (ns, entry, stats):
            found_std = _evidence_number(bucket, _STD_EVIDENCE_KEYS)
            if isinstance(found_std, (int, float)):
                try:
                    variance = float(found_std) ** 2
                except (TypeError, ValueError):
                    variance = None
                break

    low = _evidence_number(ns, _MIN_EVIDENCE_KEYS)
    if low is None:
        low = _evidence_number(entry, _MIN_EVIDENCE_KEYS)
    if low is None:
        low = _evidence_number(stats, _MIN_EVIDENCE_KEYS)
    high = _evidence_number(ns, _MAX_EVIDENCE_KEYS)
    if high is None:
        high = _evidence_number(entry, _MAX_EVIDENCE_KEYS)
    if high is None:
        high = _evidence_number(stats, _MAX_EVIDENCE_KEYS)

    return {"non_null": non_null, "distinct": distinct,
            "variance": variance, "min": low, "max": high}


def profile_table(table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-table intelligence profile (n.txt item 1 checklist).

    Always returns: table_name, columns+types, primary_key, nullable_columns,
    categorical/numerical/datetime columns, measures, dimensions, date_columns,
    row_count. Never raises on odd shapes; unknown values stay ``None``/``[]``
    (missing evidence, not negative evidence).
    """
    cols = _iter_columns(table_info)
    pk = _primary_key(table_info)
    pk_set = set(pk)
    fk_cols = set()
    for fk in table_info.get("foreign_keys", []) or []:
        if isinstance(fk, dict) and fk.get("column"):
            fk_cols.add(str(fk["column"]))

    categorical, numerical, datetimes = [], [], []
    measures, dimensions = [], []
    nullable = [name for name, _dt, nl in cols if nl]
    for name, dtype, _nl in cols:
        role = classify_column(name, dtype)
        if role == "datetime":
            datetimes.append(name)
            dimensions.append(name)
        elif role == "numerical":
            numerical.append(name)
            # Numeric dimension-likes (year, status codes, store ids,
            # holiday flags, ...) stay dimensions. Shared predicate so
            # Walmart Store/Holiday_Flag (bigint) classify as dimensions.
            if is_dimension_like_numeric(name):
                dimensions.append(name)
            elif name not in pk_set and name not in fk_cols:
                measures.append(name)
            else:
                dimensions.append(name)
        elif role == "identifier":
            dimensions.append(name)
        elif role == "boolean":
            dimensions.append(name)
            categorical.append(name)
        else:  # categorical
            categorical.append(name)
            dimensions.append(name)

    null_stats = table_info.get("null_stats") or {}
    missing = {}
    for col, stats in null_stats.items():
        if isinstance(stats, dict) and stats.get("null_pct") is not None:
            missing[str(col)] = stats.get("null_pct")

    row_count = _row_count(table_name, table_info)
    evidence: Dict[str, Dict[str, Any]] = {}
    for name, _dt, _nl in cols:
        try:
            evidence[name] = column_target_evidence(table_info, name)
        except Exception:
            evidence[name] = {"non_null": None, "distinct": None,
                              "variance": None, "min": None, "max": None}
    return {
        "table_name": table_name,
        "columns": [{"column": n, "data_type": dt, "nullable": nl}
                    for n, dt, nl in cols],
        "column_count": len(cols),
        "primary_key": pk,
        "nullable_columns": nullable,
        "categorical_columns": categorical,
        "numerical_columns": numerical,
        "datetime_columns": datetimes,
        "date_columns": list(datetimes),
        "measures": measures,
        "dimensions": dimensions,
        "row_count": row_count,
        "empty": bool(table_info.get("empty")) or (row_count == 0),
        "missingness_pct": missing,
        "column_evidence": evidence,
    }


def _relationship_count(mapping: Dict[str, Any]) -> Tuple[int, int, int]:
    declared = mapping.get("declared_relationships") or []
    inferred = mapping.get("inferred_relationships") or []
    edges = mapping.get("relationship_edges") or []
    return len(declared), len(inferred), len(edges)


def build_data_intelligence(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Build the dataset-level intelligence record (n.txt item 11).

    Works for 1 table, 2-10 tables and many tables through this single path.
    ``relationships == 0`` is valid (single-table datasets) and yields
    ``relational_analysis == False`` downstream instead of an error.
    """
    tables = mapping.get("tables") or {}
    profiles = {name: profile_table(name, info or {})
                for name, info in tables.items()}

    usable = [t for t, p in profiles.items() if not p["empty"] and p["column_count"]]
    total_rows = sum(p["row_count"] for p in profiles.values()
                     if isinstance(p["row_count"], int))
    has_numeric = any(p["numerical_columns"] or p["measures"] for p in profiles.values())
    has_categorical = any(p["categorical_columns"] for p in profiles.values())
    has_date = any(p["datetime_columns"] for p in profiles.values())

    missing_all: Dict[str, float] = {}
    for t, p in profiles.items():
        for col, pct in (p.get("missingness_pct") or {}).items():
            if isinstance(pct, (int, float)):
                missing_all[f"{t}.{col}"] = float(pct)
    avg_missing = (round(sum(missing_all.values()) / len(missing_all), 2)
                   if missing_all else 0.0)

    duplicates = None
    quality = mapping.get("quality") or {}
    for t in tables:
        info = tables[t] or {}
        csv_q = ((info.get("csv_profile") or {}).get("quality")
                 or (mapping.get("semi_structured_sources") or {}))
        _ = csv_q  # reserved: CSV dup counts surface via csv_profile when present
    if isinstance(quality.get("duplicate_rows"), int):
        duplicates = quality["duplicate_rows"]

    cardinality: Dict[str, Any] = {}
    for t, p in profiles.items():
        info = tables[t] or {}
        for col in p["dimensions"]:
            stats = (info.get("null_stats") or {}).get(col) or {}
            _ = stats  # distinct counts surface here when profilers provide them

    table_count = len(tables)
    if total_rows == 0 and not usable:
        volume = "empty"
    elif total_rows < 100:
        volume = "small"
    elif total_rows < 100_000:
        volume = "medium"
    elif total_rows < 5_000_000:
        volume = "large"
    else:
        volume = "very_large"

    n_decl, n_inf, n_edges = _relationship_count(mapping)
    return {
        "dataset_connected": bool(tables),
        "schema_valid": bool(tables) and all(
            p["column_count"] > 0 for p in profiles.values()),
        "table_count": table_count,
        "tables": sorted(tables),
        "rows_available": sum(1 for p in profiles.values()
                              if isinstance(p["row_count"], int)
                              and (p["row_count"] or 0) > 0),
        "total_rows": total_rows,
        "usable_columns": sum(p["column_count"] for p in profiles.values()),
        "date_column_available": has_date,
        "numeric_measures_available": has_numeric,
        "categorical_dimensions_available": has_categorical,
        "missingness_avg_pct": avg_missing,
        "missingness_by_column": missing_all,
        "duplicate_rows": duplicates,
        "cardinality": cardinality,
        "data_volume": volume,
        "relationships": {
            "declared": n_decl,
            "inferred": n_inf,
            "edges": n_edges,
            "none": (n_decl + n_inf + n_edges) == 0,
        },
        "per_table": profiles,
    }


# ---------------------------------------------------------------------------
# Per-target prediction validation.
# ---------------------------------------------------------------------------

def _usable_features_for(profile: Dict[str, Any], target: str,
                         evidence_map: Dict[str, Dict[str, Any]]) -> int:
    """Usable feature columns alongside `target` (same table).

    Excludes the target itself, primary-key members, identifier-like
    columns, proven-constant columns and unique free-form (high-cardinality)
    text. Unknown evidence is never a reason to exclude.
    """
    pk_set = set(profile.get("primary_key") or [])
    cols = profile.get("columns") or []
    dtypes = {}
    for item in cols:
        if isinstance(item, dict):
            dtypes[str(item.get("column") or "")] = str(item.get("data_type") or "")
    usable = 0
    for item in cols:
        name = str(item.get("column") if isinstance(item, dict) else item or "")
        if not name or name == target or name in pk_set:
            continue
        if classify_column(name, dtypes.get(name, "")) == "identifier":
            continue
        ev = (evidence_map or {}).get(name) or {}
        distinct, non_null = ev.get("distinct"), ev.get("non_null")
        low, high = ev.get("min"), ev.get("max")
        if isinstance(distinct, int) and distinct < 2:
            continue
        if low is not None and high is not None and low == high:
            continue
        if (isinstance(distinct, int) and isinstance(non_null, int)
                and non_null > 0 and distinct >= non_null
                and classify_column(name, dtypes.get(name, "")) == "categorical"):
            continue
        usable += 1
    return usable


def _validate_prediction_target(profile: Dict[str, Any], column: str,
                                evidence_map: Dict[str, Dict[str, Any]],
                                has_rows: bool) -> Dict[str, Any]:
    ev = (evidence_map or {}).get(column) or {}
    non_null, distinct = ev.get("non_null"), ev.get("distinct")
    variance, low, high = ev.get("variance"), ev.get("min"), ev.get("max")
    usable = _usable_features_for(profile, column, evidence_map)
    base = {"table": profile.get("table_name"), "column": column,
            "non_null": non_null, "distinct": distinct,
            "variance": variance, "usable_features": usable}

    def _no(reason: str) -> Dict[str, Any]:
        return {**base, "eligible": False, "reason": reason}

    if profile.get("empty") or not has_rows:
        return _no(f"'{column}' has no usable rows — connect data with "
                   f"populated values, then try a different number.")
    if isinstance(non_null, int) and non_null <= 0:
        return _no(f"'{column}' has no non-null values — try a different "
                   f"number with populated values.")
    if isinstance(distinct, int) and distinct == 0:
        return _no(f"'{column}' has no distinct values — try a different "
                   f"number with changing values.")
    if isinstance(distinct, int) and distinct == 1:
        return _no(f"'{column}' looks constant (a single distinct value) — "
                   f"there is no variation to learn from. Try a different "
                   f"number with changing values.")
    if isinstance(variance, (int, float)) and variance <= 0:
        return _no(f"'{column}' shows no variance (constant history) — "
                   f"there is no trend to project. Try a different number "
                   f"with changing values.")
    if low is not None and high is not None and low == high:
        return _no(f"'{column}' never varies (same value throughout) — "
                   f"try a different number with changing values.")
    if isinstance(non_null, int) and non_null < 2:
        return _no(f"'{column}' has too few non-null values for prediction "
                   f"({non_null}) — connect more history, then try again.")
    if usable < 1:
        return _no(f"'{column}' has no usable feature column alongside it "
                   f"(ids and constant columns do not count) — try a "
                   f"dataset with more descriptive columns.")
    scope = f"{non_null} non-null values" if isinstance(non_null, int) \
        else "populated values"
    return {**base, "eligible": True,
            "reason": f"'{column}' has {scope} with variation and "
                      f"{usable} usable feature(s) — eligible for prediction."}


def build_prediction_targets(intel: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-target prediction validation across every profiled table.

    One entry per numeric candidate (measures + numerical columns,
    deduplicated) carrying ``{table, column, non_null, distinct, variance,
    usable_features, eligible, reason}``. Pure and generic: no table/column
    literals, no table-count special cases.
    """
    per_table = intel.get("per_table") or {}
    has_rows = (intel.get("total_rows") or 0) > 0
    targets: List[Dict[str, Any]] = []
    for tname in sorted(per_table):
        profile = per_table[tname] or {}
        evidence_map = profile.get("column_evidence") or {}
        ordered: List[str] = []
        for col in (profile.get("measures") or []) + \
                (profile.get("numerical_columns") or []):
            if col and col not in ordered:
                ordered.append(col)
        for col in ordered:
            try:
                targets.append(_validate_prediction_target(
                    profile, col, evidence_map, has_rows))
            except Exception:
                targets.append({"table": profile.get("table_name"),
                                "column": col, "non_null": None,
                                "distinct": None, "variance": None,
                                "usable_features": 0, "eligible": False,
                                "reason": f"'{col}' could not be validated."})
    targets.sort(key=lambda t: (not t.get("eligible"),
                                str(t.get("table") or ""),
                                str(t.get("column") or "")))
    return targets


# ---------------------------------------------------------------------------
# Capability detection (n.txt item 12).
# ---------------------------------------------------------------------------

def detect_capabilities(intel: Dict[str, Any]) -> Dict[str, Any]:
    """Map intelligence -> per-analysis capability flags with reasons.

    Rules (dynamic, data-driven):
    - KPIs / distributions / categorical / numerical / correlations: need rows
      + the relevant column kinds on at least one usable table.
    - relational / cross-table: need >= 2 usable tables + relationships.
    - segmentation / product analysis: need rows + categorical dimensions
      (+ a numeric measure for product analysis) on ANY table count — a
      single table with a categorical dimension validly supports GROUP BY
      segmentation; requiring >= 2 tables was the shared-layer multi-table
      assumption that broke single-table datasets.
    - prediction: per-target validation (see build_prediction_targets) —
      each numeric candidate is checked for populated non-null values,
      variation (distinct/variance evidence when profilers supply it) and
      at least one usable feature column (ids, constants and unique
      free-form text excluded). Enabled when >= 1 target is eligible.
    - forecasting / trends: need a date column AND a numeric measure.
    A single-table dataset therefore gets KPIs/distributions/forecasting when
    its columns allow it, and ``relational_analysis == False`` — which is the
    correct, valid answer, not a failure.
    """
    per_table = intel.get("per_table") or {}
    usable = [p for p in per_table.values()
              if not p.get("empty") and p.get("column_count")]
    n_usable = len(usable)
    has_rows = (intel.get("total_rows") or 0) > 0
    has_numeric = bool(intel.get("numeric_measures_available"))
    has_cat = bool(intel.get("categorical_dimensions_available"))
    has_date = bool(intel.get("date_column_available"))
    rel = intel.get("relationships") or {}
    has_rel = not rel.get("none", True)

    def _flag(ok: bool, why: str) -> Dict[str, Any]:
        return {"enabled": bool(ok), "reason": why}

    caps = {
        "kpis": _flag(has_rows and has_numeric,
                      "needs rows + numeric measure" if not (has_rows and has_numeric)
                      else "rows + numeric measure available"),
        "distributions": _flag(has_rows and (has_numeric or has_cat),
                               "needs rows + numeric/categorical column" if not (has_rows and (has_numeric or has_cat))
                               else "rows + distribution column available"),
        "categorical_analysis": _flag(has_rows and has_cat,
                                      "needs rows + categorical dimension" if not (has_rows and has_cat)
                                      else "categorical dimensions available"),
        "numerical_analysis": _flag(has_rows and has_numeric,
                                    "needs rows + numeric measure" if not (has_rows and has_numeric)
                                    else "numeric measures available"),
        "correlations": _flag(has_rows and sum(len(p.get("numerical_columns", [])) for p in usable) >= 2,
                              "needs >= 2 numeric columns with rows" if sum(len(p.get("numerical_columns", [])) for p in usable) < 2 or not has_rows
                              else "multiple numeric columns available"),
        "relational_analysis": _flag(n_usable >= 2 and has_rel,
                                     "needs >= 2 tables with relationships" if not (n_usable >= 2 and has_rel)
                                     else "multi-table relationships available"),
        "forecasting": _flag(has_rows and has_date and has_numeric,
                             "needs date column + numeric measure + rows" if not (has_rows and has_date and has_numeric)
                             else "date + measure available for forecasting"),
        "trends": _flag(has_rows and has_date,
                        "needs date column + rows" if not (has_rows and has_date)
                        else "date column available"),
        "segmentation": _flag(has_rows and has_cat,
                              "needs rows + categorical dimensions" if not (has_rows and has_cat)
                              else "categorical dimensions available for grouping"),
        "product_analysis": _flag(has_rows and has_cat and has_numeric,
                                  "needs categorical dimension + numeric measure" if not (has_rows and has_cat and has_numeric)
                                  else "dimensions + measures available"),
        "cross_table_insights": _flag(n_usable >= 2 and has_rel and has_rows,
                                      "needs >= 2 related tables with rows" if not (n_usable >= 2 and has_rel and has_rows)
                                      else "related tables available"),
    }
    # Per-target prediction validation replaces the coarse gate: one entry
    # per numeric candidate with its own eligibility + reason.
    prediction_targets = build_prediction_targets(intel)
    if any(t.get("eligible") for t in prediction_targets):
        first_ok = next(t for t in prediction_targets if t.get("eligible"))
        caps["prediction"] = _flag(True, first_ok.get("reason") or
                                   "eligible prediction target available")
    elif not has_rows:
        caps["prediction"] = _flag(False, "needs rows with populated values")
    elif not prediction_targets:
        caps["prediction"] = _flag(False, "needs a numeric measure to predict")
    else:
        caps["prediction"] = _flag(
            False, prediction_targets[0].get("reason") or
            "no eligible prediction target")
    suitable = sorted(k for k, v in caps.items() if v["enabled"])
    # Flat can_* aliases (P0 contract): same dynamic flags, no second source
    # of truth — every value is read straight from the derived caps above.
    can = {
        "can_kpi": caps["kpis"]["enabled"],
        "can_insights": caps["distributions"]["enabled"],
        "can_forecast": caps["forecasting"]["enabled"],
        "can_prediction": caps["prediction"]["enabled"],
        "can_relational_analysis": caps["relational_analysis"]["enabled"],
        "can_segmentation": caps["segmentation"]["enabled"],
    }
    out = {
        "capabilities": caps,
        "suitable_analyses": suitable,
        "table_count": intel.get("table_count", 0),
        "usable_tables": n_usable,
        "prediction_targets": prediction_targets,
    }
    out.update(can)
    return out
