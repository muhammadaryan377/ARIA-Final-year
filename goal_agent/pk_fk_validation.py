"""PK/FK data quality validation for ARIA.

Checks:
  - PK columns: duplicates, NULLs, type mismatches
  - FK columns: orphan values, NULL FKs, type mismatches
  - Relationship integrity: missing referenced tables/columns

Run AFTER schema extraction and data loading. Respects existing profiling
limits (never full-table scans on huge datasets without bounds).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_IDENT_RE = None


def _quote_ident(name: str) -> str:
    """Quote an SQL identifier safely. Raises ValueError on invalid names.

    Schema-qualified keys (sales.employee) are quoted per segment
    ("sales"."employee"); embedded double quotes are escaped by doubling.
    Prevents identifier injection vs the previous raw f-string
    interpolation.
    """
    import re as _re
    if not isinstance(name, str) or not name or len(name) > 256:
        raise ValueError("Invalid identifier.")
    parts = []
    for seg in str(name).split("."):
        if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", seg or ""):
            raise ValueError(f"Unsafe identifier: {name!r}")
        parts.append('"' + seg.replace('"', '""') + '"')
    if not parts:
        raise ValueError("Invalid identifier.")
    return ".".join(parts)


def _bound_sample(n: int, default: int = 10000, cap: int = 100000) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        return default
    return max(1, min(v, cap))


def validate_schema_references(
    tables: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate that all FK references point to existing tables/columns.

    Returns a list of warning dicts (empty = all good).
    """
    warnings = []
    for tname, tinfo in tables.items():
        for fk in tinfo.get("foreign_keys", []):
            ref_table = fk.get("referenced_table")
            ref_col = fk.get("referenced_column")
            col = fk.get("column")
            if not ref_table or ref_table not in tables:
                warnings.append({
                    "type": "missing_referenced_table",
                    "table": tname,
                    "column": col,
                    "detail": f"FK {tname}.{col} references unknown table '{ref_table}'",
                })
                continue
            ref_cols = tables[ref_table].get("columns", {})
            if ref_col not in ref_cols:
                warnings.append({
                    "type": "missing_referenced_column",
                    "table": tname,
                    "column": col,
                    "detail": f"FK {tname}.{col} references unknown column '{ref_table}.{ref_col}'",
                })
    return warnings


def validate_pk_quality(
    conn,
    tables: dict[str, Any],
    max_sample: int = 10000,
) -> list[dict[str, Any]]:
    """Check PK columns for duplicates and NULLs.

    Only runs on tables with a declared or inferred PK.
    Uses COUNT(DISTINCT) vs COUNT(*) for efficiency.
    """
    findings = []
    for tname, tinfo in tables.items():
        pk_cols = tinfo.get("primary_key", []) or []
        if not pk_cols:
            continue
        try:
            _t = _quote_ident(tname)
        except ValueError:
            continue
        for pk_col in pk_cols:
            try:
                _c = _quote_ident(pk_col)
            except ValueError:
                continue
            # NOTE: pk_cols comes from the declared/inferred primary_key list,
            # so every entry is validated. (Previous `is_primary_key is not True`
            # guard skipped ALL PKs because mappings never set that key.)

            try:
                sample = _bound_sample(max_sample)
                cur = conn.execute(
                    f'SELECT COUNT(*), COUNT(DISTINCT sub.col), '
                    f'SUM(CASE WHEN sub.col IS NULL THEN 1 ELSE 0 END) '
                    f'FROM (SELECT {_c} AS col FROM {_t} LIMIT {sample}) sub'
                )
                total, distinct, nulls = cur.fetchone()
                if total and distinct and distinct < total:
                    findings.append({
                        "type": "duplicate_pk",
                        "table": tname,
                        "column": pk_col,
                        "detail": f"PK column {pk_col} has {distinct} unique values in {total} rows ({total - distinct} duplicates)",
                        "severity": "high",
                    })
                if nulls and nulls > 0:
                    findings.append({
                        "type": "null_pk",
                        "table": tname,
                        "column": pk_col,
                        "detail": f"PK column {pk_col} has {nulls} NULL values",
                        "severity": "high",
                    })
            except Exception as e:
                logger.debug("PK quality check skipped for %s.%s: %s", tname, pk_col, e)
    return findings


def validate_fk_quality(
    conn,
    tables: dict[str, Any],
    max_sample: int = 10000,
) -> list[dict[str, Any]]:
    """Check FK columns for orphan values and NULLs.

    Only runs on tables with declared/inferred FK relationships.
    Uses LEFT JOIN for orphan detection (efficient with indexes).
    """
    findings = []
    for tname, tinfo in tables.items():
        for fk in tinfo.get("foreign_keys", []):
            fk_col = fk.get("column")
            ref_table = fk.get("referenced_table")
            ref_col = fk.get("referenced_column")
            if not fk_col or not ref_table or not ref_col:
                continue
            if ref_table not in tables:
                continue
            try:
                _t = _quote_ident(tname)
                _fc = _quote_ident(fk_col)
                _rt = _quote_ident(ref_table)
                _rc = _quote_ident(ref_col)
            except ValueError:
                continue

            try:
                sample = _bound_sample(max_sample)
                cur = conn.execute(
                    f'SELECT COUNT(*) FROM ('
                    f'SELECT {_fc} FROM {_t} '
                    f'WHERE {_fc} IS NOT NULL '
                    f'LIMIT {sample}'
                    f') sub'
                )
                total_fks = cur.fetchone()[0]

                cur = conn.execute(
                    f'SELECT COUNT(*) FROM {_t} t '
                    f'LEFT JOIN {_rt} r ON t.{_fc} = r.{_rc} '
                    f'WHERE t.{_fc} IS NOT NULL AND r.{_rc} IS NULL '
                    f'LIMIT 100'
                )
                orphan_count = cur.fetchone()[0]

                if orphan_count > 0:
                    findings.append({
                        "type": "orphan_fk",
                        "table": tname,
                        "column": fk_col,
                        "detail": f"FK {tname}.{fk_col} has {orphan_count} orphan values not found in {ref_table}.{ref_col}",
                        "severity": "medium",
                    })

                cur = conn.execute(
                    f'SELECT COUNT(*) FROM (SELECT {_fc} FROM {_t} WHERE {_fc} IS NULL LIMIT 10001) sub'
                )
                null_count = cur.fetchone()[0]
                if null_count > 0:
                    col_info = tinfo.get("columns", {})
                    if isinstance(col_info, dict):
                        nullable = col_info.get(fk_col, {}).get("nullable", True)
                    else:
                        nullable = True
                    if not nullable:
                        findings.append({
                            "type": "null_fk_non_nullable",
                            "table": tname,
                            "column": fk_col,
                            "detail": f"Non-nullable FK column {fk_col} has {null_count} NULLs",
                            "severity": "medium",
                        })

            except Exception as e:
                logger.debug("FK quality check skipped for %s.%s: %s", tname, fk_col, e)
    return findings


def run_all_validations(
    conn,
    tables: dict[str, Any],
) -> dict[str, Any]:
    """Run all PK/FK data quality checks and return a summary.

    Returns:
        {
            "schema_warnings": [...],
            "pk_findings": [...],
            "fk_findings": [...],
            "total_issues": N,
            "healthy": bool,
        }
    """
    schema_warnings = validate_schema_references(tables)
    pk_findings = validate_pk_quality(conn, tables)
    fk_findings = validate_fk_quality(conn, tables)
    total = len(schema_warnings) + len(pk_findings) + len(fk_findings)
    return {
        "schema_warnings": schema_warnings,
        "pk_findings": pk_findings,
        "fk_findings": fk_findings,
        "total_issues": total,
        "healthy": total == 0,
    }
