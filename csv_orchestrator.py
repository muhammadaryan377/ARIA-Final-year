"""CSV orchestration for ARIA: validated CSV -> SQLite + Goal-Agent schema mapping.

Composes Aryan's deterministic CSV handler (``csv_handler``) with the rest of
ARIA without modifying the Schema Agent or Goal Agent:

- ``build_csv_schema_mapping`` converts the CSV handler's profile into a schema
  mapping JSON in the ``tables`` format the Goal Agent consumes (columns /
  data_type / nullable / inferred_primary_key / row_count / empty).
- ``load_csv_to_sqlite`` loads the raw rows (typed, null-preserving) into a
  file-based SQLite database so the Goal Agent can execute SQL against it with
  ``dialect="sqlite"``, and the Insight Agent can analyse query results through
  the normal processed-data path.
- ``build_multi_csv_schema_mapping`` / ``load_multiple_csvs_to_sqlite`` /
  ``build_sqlite_schema_mapping`` provide the GENERIC multi-table path: table
  discovery, per-table PK inference (single + composite), cross-table FK
  inference (name/type overlap + value containment), relationship graph, join
  planning metadata and duplicate-column disambiguation. No dataset, table or
  column names are hardcoded anywhere in this module.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from csv_handler import CsvSchemaExtractor

_log = logging.getLogger("aria.csv")

# Inferred logical types from the CSV handler -> SQLite storage types.
_SQLITE_TYPE = {
    "INTEGER": "INTEGER",
    "DECIMAL": "REAL",
    "BOOLEAN": "INTEGER",
    "DATE": "TEXT",
    "DATETIME": "TEXT",
    "STRING": "TEXT",
    "UNKNOWN": "TEXT",
}

_TRUE_TOKENS = {"true", "yes", "y", "1"}
_FALSE_TOKENS = {"false", "no", "n", "0"}

_INSERT_BATCH = 20_000
# Bounded streaming for SQLite loads (spec §1): chunks of this size keep the
# insert working set flat regardless of size (50k rows x ~10 cols of short
# strings ≈ single-digit MB). No analytical row cap. Larger chunks cut
# pandas per-chunk parse overhead, which dominates load time on big files.
_LOAD_CHUNK_ROWS = 50_000
# Bounded sample used to resolve ambiguous date formats before streaming.
_DATE_RESOLVE_SAMPLE_ROWS = 5_000
# Value-containment sets for cross-table FK inference are capped per column
# (matches the SQL path's LIMIT 2000 in build_sqlite_schema_mapping).
_MAX_VALUE_SET = 2_000

_DATE_NAME_HINTS = ("date", "time", "day", "month", "year", "created",
                    "updated", "timestamp", "period", "shipped", "ordered")

_SLASH_DATE_RE = re.compile(
    r"\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})"
    r"(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*"
)


def _resolve_slash_date_format(samples: list[str]) -> str | None:
    """Resolve ambiguous slash dates via the >12 rule.

    Returns "MDY"/"DMY", or None when inconsistent, contradictory, or fully
    ambiguous (all components ≤ 12 — guessing would fabricate meaning).
    """
    mdy_evidence = dmy_evidence = False
    seen = False
    for raw in samples:
        s = str(raw).strip()
        if not s:
            continue
        m = _SLASH_DATE_RE.fullmatch(s)
        if not m:
            return None
        seen = True
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 and b > 12:
            return None
        if a > 12:
            dmy_evidence = True
        elif b > 12:
            mdy_evidence = True
    if not seen or (mdy_evidence and dmy_evidence):
        return None
    if mdy_evidence:
        return "MDY"
    if dmy_evidence:
        return "DMY"
    return None


def _slash_to_iso(value: str, fmt: str) -> str | None:
    """Convert a resolved slash date to ISO (YYYY-MM-DD[ HH:MM:SS])."""
    m = _SLASH_DATE_RE.fullmatch(str(value).strip())
    if not m:
        return None
    a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    month, day = (a, b) if fmt == "MDY" else (b, a)
    if year < 100:
        year += 2000 if year < 70 else 1900
    hh, mm, ss = m.group(4), m.group(5), m.group(6)
    try:
        if hh is None:
            datetime(year, month, day)
            return f"{year:04d}-{month:02d}-{day:02d}"
        datetime(year, month, day, int(hh), int(mm), int(ss or 0))
        return f"{year:04d}-{month:02d}-{day:02d} {int(hh):02d}:{int(mm):02d}:{int(ss or 0):02d}"
    except ValueError:
        return None


def _normalize_date_columns(df, column_logical_types: dict) -> dict[str, str]:
    """Normalize resolvable non-ISO date columns to ISO in place.

    Only touches STRING columns with date-like names whose slash format is
    provably resolved. Ambiguous columns are left untouched (values preserved).
    Returns {column: note} for observability.
    Kept for small-frame callers; the streaming loader resolves formats once
    from a bounded sample (_resolve_date_formats) and applies them per chunk
    (_apply_date_formats) so no full column is ever held for this.
    """
    formats = _resolve_date_formats(df, column_logical_types)
    return _apply_date_formats(df, formats)


def _resolve_date_formats(df_sample, column_logical_types: dict) -> dict[str, str]:
    """Decide {column: MDY/DMY} once from a bounded sample (best-effort).

    Sampling is head+tail bounded (up to 1000 evenly spaced non-empty values)
    so resolution never holds the full column and late-file format switches
    are still observed. Mixed-format files beyond the sample may resolve
    differently than a full scan would; ambiguous columns are always left
    untouched, so the worst case is un-normalized (raw, preserved) values.
    """
    formats: dict[str, str] = {}
    for name, logical in column_logical_types.items():
        if logical != "STRING" or name not in df_sample.columns:
            continue
        low = str(name).lower()
        if not any(h in low for h in _DATE_NAME_HINTS):
            continue
        try:
            series = df_sample[name].dropna().astype(str)
            non_empty = [v for v in series.tolist() if str(v).strip()]
            # Evenly spaced head+tail sample (not head-only) so a format
            # switch late in the file is still observed. Bounded at 1000.
            if len(non_empty) > 1000:
                try:
                    import numpy as _np
                    _idx = _np.linspace(0, len(non_empty) - 1, 1000).astype(int)
                    samples = [non_empty[int(i)] for i in _idx]
                except Exception:
                    samples = non_empty[:500] + non_empty[-500:]
            else:
                samples = non_empty
        except Exception as exc:
            _log.debug("sample extraction fallback for column %s: %s", name, exc)
            continue
        if not samples:
            continue
        # Skip only values that are already ISO (YYYY-MM-DD...). A dash alone
        # does not mean ISO: DD-MM-YYYY (e.g. Walmart 05-02-2010) also uses
        # dashes and must still be resolved via the >12 rule below.
        if all(re.match(r"^\d{4}-\d{2}-\d{2}", str(s).strip()) for s in samples[:5]):
            continue  # already ISO-like
        fmt = _resolve_slash_date_format(samples)
        if fmt:
            formats[name] = fmt
    return formats


def _apply_date_formats(df, formats: dict[str, str]) -> dict[str, str]:
    """Convert format-resolved columns to ISO in place. Never raises."""
    notes: dict[str, str] = {}
    if not formats:
        return notes
    for name, fmt in formats.items():
        if name not in df.columns:
            continue
        try:
            converted = 0
            iso_vals = []
            for v in df[name].tolist():
                if v is None or (isinstance(v, str) and not v.strip()):
                    iso_vals.append(v)
                    continue
                iso = _slash_to_iso(v, fmt)
                if iso is None:
                    iso_vals.append(v)
                else:
                    iso_vals.append(iso)
                    converted += 1
            if converted:
                df[name] = iso_vals
                notes[name] = f"normalized {converted} values from {fmt} to ISO"
            else:
                notes[name] = "non-ISO date format left as-is (ambiguous)"
        except Exception as exc:
            _log.debug("date conversion skipped for column %s: %s", name, exc)
    return notes


def _safe_db_name(name: str) -> str:
    normalised = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return normalised or "file_upload"


def safe_table_name(name: str) -> str:
    normalised = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return normalised or "csv_table"


def _csv_column_null_stats(column: dict, row_count: int) -> dict[str, Any]:
    """null_stats entry enriched with profiler evidence (generic).

    Carries the handler's distinct_count + numeric min/max into the shared
    mapping so per-target validation can spot constant columns without
    re-reading the file. No names are referenced; unknown stays absent.
    """
    stats: dict[str, Any] = {
        "total_rows": row_count,
        "nulls": column.get("null_count", 0),
        "null_pct": column.get("null_percentage", 0.0),
    }
    try:
        distinct = column.get("distinct_count")
        if isinstance(distinct, int) and not isinstance(distinct, bool):
            stats["distinct_count"] = distinct
        col_stats = column.get("statistics")
        if isinstance(col_stats, dict):
            for key in ("min", "max"):
                value = col_stats.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    stats[key] = value
    except Exception as exc:
        _log.debug("column stats extraction failed: %s", exc)
        pass
    return stats


def _column_classes(profile: dict[str, Any]) -> dict[str, str]:
    """Per-column analytical class for the sufficiency/profile contract.

    Deterministic rule over the handler's inferred types (no name matching):
    INTEGER/DECIMAL -> "numeric", DATE/DATETIME -> "date", BOOLEAN ->
    "boolean", UNKNOWN -> "unknown", low-cardinality STRING -> "categorical"
    (distinct <= 100 and <= 50% of rows), other STRING -> "text". Additive:
    existing data_type/nullable keys are untouched.
    """
    row_count = int(profile.get("row_count") or 0)
    classes: dict[str, str] = {}
    for column in profile.get("columns", []) or []:
        name = column.get("name")
        logical = column.get("inferred_type", "UNKNOWN")
        if logical in ("INTEGER", "DECIMAL"):
            classes[name] = "numeric"
        elif logical in ("DATE", "DATETIME"):
            classes[name] = "date"
        elif logical == "BOOLEAN":
            classes[name] = "boolean"
        elif logical == "UNKNOWN":
            classes[name] = "unknown"
        else:
            try:
                distinct = int(column.get("distinct_count") or 0)
            except (TypeError, ValueError):
                distinct = 0
            if distinct <= 100 and (row_count == 0 or distinct / max(1, row_count) <= 0.5):
                classes[name] = "categorical"
            else:
                classes[name] = "text"
    return classes


def _time_coverage(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Time span of the first DATE/DATETIME column with a recorded range.

    Carries the handler's exact earliest/latest into the mapping so planners
    see time coverage without re-reading the file. None when no temporal
    column has a usable range (valid: not an error).
    """
    for column in profile.get("columns", []) or []:
        if column.get("inferred_type") not in ("DATE", "DATETIME"):
            continue
        stats = column.get("statistics") or {}
        if stats.get("earliest") and stats.get("latest"):
            return {
                "column": column.get("name"),
                "earliest": stats.get("earliest"),
                "latest": stats.get("latest"),
                "row_count": int(profile.get("row_count") or 0),
            }
    return None


def _sampling_block(profile: dict[str, Any]) -> dict[str, Any]:
    """Normalise the handler's sampling disclosure (defaults: exact)."""
    block = profile.get("sampling")
    if isinstance(block, dict):
        return block
    row_count = int(profile.get("row_count") or 0)
    return {
        "sampled": False,
        "sample_rows": row_count,
        "total_rows": row_count,
        "exact_fields": ["row_count"],
        "sample_based_fields": [],
    }


def _table_entry_from_profile(
    profile: dict[str, Any], table: str, source: str,
    *, with_pk_flags: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build one ``tables`` entry from a handler profile (generic).

    Shared by the single- and multi-CSV paths so row_count/empty/null_stats
    plus the profile contract (column_classes, time_coverage, sampling) flow
    identically everywhere. Returns (entry, logical_types).
    """
    columns: list[dict[str, Any]] = []
    null_stats: dict[str, dict[str, Any]] = {}
    inferred_pk: list[str] = []
    logical: dict[str, str] = {}
    for column in profile["columns"]:
        name = column["name"]
        logical[name] = column["inferred_type"]
        entry_col: dict[str, Any] = {
            "column": name,
            "data_type": _SQLITE_TYPE.get(column["inferred_type"], "TEXT"),
            "nullable": column["null_count"] > 0,
        }
        if with_pk_flags:
            entry_col["is_primary_key"] = False
            entry_col["is_foreign_key"] = False
        columns.append(entry_col)
        null_stats[name] = _csv_column_null_stats(column, profile["row_count"])
        if column["candidate_key"]:
            inferred_pk.append(name)
    if with_pk_flags:
        for col in columns:
            if col["column"] in inferred_pk:
                col["is_primary_key"] = True
    # Composite detection is generic (profiling/uniqueness/nullability/type
    # ranking inside the handler, never hardcoded names). A composite pair
    # is exposed SEPARATELY from `inferred_primary_key` so the single-column
    # contract stays honest ([] when no single column is unique) while the
    # 2-column uniqueness is still available to planners via
    # `inferred_composite_primary_key` / `composite_candidate_keys`.
    composite_pk: list[str] = []
    if not inferred_pk:
        for pair in (profile.get("composite_candidate_keys") or []):
            if (isinstance(pair, (list, tuple)) and len(pair) == 2
                    and all(isinstance(c, str) and c in null_stats for c in pair)
                    and all(null_stats[c]["nulls"] == 0 for c in pair)):
                composite_pk = list(pair)
                break
    entry = {
        "columns": columns,
        "primary_key": [],
        "inferred_primary_key": inferred_pk,
        "inferred_composite_primary_key": composite_pk,
        "composite_candidate_keys": [composite_pk] if composite_pk else [],
        "null_stats": null_stats,
        "row_count": int(profile["row_count"]),
        "empty": int(profile["row_count"]) == 0,
        "source": source,
        "column_classes": _column_classes(profile),
        "time_coverage": _time_coverage(profile),
        "sampling": _sampling_block(profile),
    }
    return entry, logical


def build_csv_schema_mapping(csv_path: str | Path, table_name: str | None = None,
                             profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a Goal-Agent-compatible schema mapping from a validated CSV file.

    Candidate keys detected by the CSV handler become ``inferred_primary_key``
    (they are suggested, never forced, matching the CSV handler's contract);
    ``row_count``/``empty`` let the Goal Agent distinguish truly empty tables.
    ``column_classes``/``time_coverage``/``sampling`` carry the usable row
    count, analytical column roles and any sampling disclosure through the
    existing contract (additive keys; the sufficiency gate is owned elsewhere
    and is not duplicated here).

    ``profile`` accepts a precomputed CsvSchemaExtractor profile so upload
    paths parse the file once and share it with the SQLite loader.
    """
    if profile is None:
        extractor = CsvSchemaExtractor(csv_path)
        profile = extractor.extract()
    table = safe_table_name(table_name or profile["table_name"])

    entry, _logical = _table_entry_from_profile(
        profile, table, str(csv_path), with_pk_flags=True)

    tables_mapping = {table: entry}
    # File-level provenance stays under csv_profile (existing contract).
    entry["csv_profile"] = {
        "size_bytes": profile["file"]["size_bytes"],
        "sha256": profile["file"]["sha256"],
        "encoding": profile["file"]["encoding"],
        "delimiter": profile["file"]["delimiter"],
        "quality": profile["quality"],
    }

    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + _uuid.uuid4().hex[:8],
        "database": _safe_db_name(Path(csv_path).stem),
        "schema": "file",
        "tables": tables_mapping,
        "declared_relationships": [],
        "inferred_relationships": [],
        "relationship_edges": [],
        "semi_structured_sources": {
            str(csv_path): {"tables": sorted(tables_mapping)},
        },
    }
    # Universal Data Intelligence Layer (same contract as the DB path; a lone
    # CSV table validly reports relationships.none == True).
    try:
        from core.data_intelligence import build_data_intelligence, detect_capabilities
        _intel = build_data_intelligence(mapping)
        mapping["data_intelligence"] = _intel
        mapping["capabilities"] = detect_capabilities(_intel)
    except Exception:
        logging.getLogger("aria.csv").debug("data intelligence attach failed", exc_info=True)
    return mapping


def _coerce_value(raw: Any, logical_type: str) -> Any:
    """Cast one raw string cell to its SQLite value (NULL for missing values).

    ``logical_type`` is the CSV handler's inferred type (INTEGER, DECIMAL,
    BOOLEAN, DATE, DATETIME, STRING, UNKNOWN) so bools are stored as 0/1 and
    identifiers/ISO dates stay as text.
    """
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if logical_type == "BOOLEAN":
        lowered = value.lower()
        if lowered in _TRUE_TOKENS:
            return 1
        if lowered in _FALSE_TOKENS:
            return 0
        return None
    if logical_type == "INTEGER":
        try:
            return int(value)
        except ValueError:
            # Retry after stripping currency symbols and thousand-separator commas
            from csv_handler import _clean_currency
            cleaned = _clean_currency(value)
            try:
                return int(float(cleaned))
            except (ValueError, TypeError):
                return None
    if logical_type == "DECIMAL":
        try:
            return float(value)
        except ValueError:
            from csv_handler import _clean_currency
            cleaned = _clean_currency(value)
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return None
    return value


def _quote_ident(name: str) -> str:
    """Quote a SQLite identifier safely (validate + escape embedded quotes).

    Table/column names in :func:`build_sqlite_schema_mapping` come from
    ``sqlite_master`` / ``PRAGMA table_info`` (i.e. file content, not code),
    so they must never be interpolated raw. Validation rejects empty names
    and NUL bytes (SQLite rejects NUL in identifiers); everything else —
    spaces, quotes, semicolons, unicode — is preserved by escaping ``"``
    as ``""`` inside a double-quoted identifier. ``PRAGMA table_info(?)``
    cannot bind identifiers, so quoting is the correct parameterisation.
    """
    if not isinstance(name, str):
        raise TypeError(f"SQLite identifier must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("SQLite identifier must be non-empty.")
    if "\x00" in name:
        raise ValueError("SQLite identifier must not contain NUL bytes.")
    return '"' + name.replace('"', '""') + '"'


def load_csv_to_sqlite(csv_path: str | Path, sqlite_path: str | Path,
                       table_name: str | None = None,
                       profile: dict[str, Any] | None = None) -> tuple[str, str, int]:
    """Load a validated CSV into a file-based SQLite database.

    Uses the CSV handler's detected encoding/dialect and inferred column
    types. Numeric and boolean columns are stored as INTEGER/REAL values so
    aggregate SQL works; date/string columns stay TEXT (ISO strings and
    identifiers with leading zeros are preserved); missing tokens are NULL.

    Returns (db_uri, table_name, row_count).

    The file is streamed in bounded chunks (never a full-frame to_dict), so
    memory stays flat regardless of size. ``profile`` accepts a precomputed
    CsvSchemaExtractor profile so callers that already extracted (upload
    path) never parse the file twice.
    """
    extractor = CsvSchemaExtractor(csv_path)
    if profile is None:
        profile = extractor.extract()
    table = safe_table_name(table_name or profile["table_name"])

    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sqlite_path.unlink(missing_ok=True)
    except PermissionError:
        pass

    column_logical_types = {
        column["name"]: column["inferred_type"]
        for column in profile["columns"]
    }
    # Resolve provably-resolvable non-ISO dates once from a bounded sample,
    # then apply per chunk. Never breaks ingestion: wrapped defensively,
    # ambiguous formats untouched (raw values preserved).
    try:
        if int(profile.get("row_count") or 0) > 0:
            _sample = extractor.read_sample(_DATE_RESOLVE_SAMPLE_ROWS)
            date_formats = _resolve_date_formats(_sample, column_logical_types)
            del _sample
        else:
            date_formats = {}
    except Exception:
        logging.getLogger("aria.csv").exception(
            "Date format resolution failed; continuing with raw values")
        date_formats = {}
    column_types = {
        name: _SQLITE_TYPE.get(logical, "TEXT")
        for name, logical in column_logical_types.items()
    }
    col_names = list(column_types)

    if not col_names:
        raise ValueError("CSV has no columns; refusing to create an empty SQLite table.")
    ddl_cols = ", ".join(f'{_quote_ident(name)} {column_types[name]}' for name in col_names)
    col_sql = ", ".join(_quote_ident(name) for name in col_names)
    placeholders = ", ".join("?" for _ in col_names)

    streamed = 0
    conn = sqlite3.connect(str(sqlite_path))
    try:
        # Bulk-load tuning (ingest-only): skip journaling/fsync while the
        # table is being built. The file is disposable until commit — on any
        # failure both artifacts are deleted by the caller below.
        try:
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-64000")
        except Exception:
            pass
        conn.execute(f'CREATE TABLE {_quote_ident(table)} ({ddl_cols})')
        insert_sql = f'INSERT INTO {_quote_ident(table)} ({col_sql}) VALUES ({placeholders})'
        try:
            for chunk in extractor.iter_chunks(_LOAD_CHUNK_ROWS):
                if date_formats:
                    try:
                        _apply_date_formats(chunk, date_formats)
                    except Exception:
                        logging.getLogger("aria.csv").exception(
                            "Date normalization failed for a chunk; continuing with raw values")
                # itertuples fast path (measured faster than column-wise
                # vector ops on clean data: plain int()/float() parse at
                # C speed per cell). Column order verified per chunk.
                try:
                    if list(chunk.columns) != col_names:
                        raise ValueError("column order drift")
                    _rows = chunk.itertuples(index=False, name=None)
                    batch = [tuple(_coerce_value(rec[i], column_logical_types[col_names[i]])
                                   for i in range(len(col_names)))
                             for rec in _rows]
                except Exception:
                    batch = [
                        tuple(_coerce_value(rec.get(name), column_logical_types[name]) for name in col_names)
                        for rec in chunk.to_dict(orient="records")
                    ]
                del chunk
                # Sub-batch so a chunk larger than _INSERT_BATCH still commits bounded.
                for start in range(0, len(batch), _INSERT_BATCH):
                    part = batch[start:start + _INSERT_BATCH]
                    if part:
                        conn.executemany(insert_sql, part)
                streamed += len(batch)
                del batch
        except Exception:
            logging.getLogger("aria.csv").exception("CSV SQLite ingestion failed")
            raise
        conn.commit()
        # Query-speed indexes (generic): DATE/DATETIME columns back every
        # time filter/grouping the app runs; indexing them at load keeps
        # explore/forecast fast on million-row files. Bounded: dates only,
        # never every column (write cost + size).
        try:
            for _dc, _lt in column_logical_types.items():
                if str(_lt).upper() in ("DATE", "DATETIME"):
                    _in = f"idx_{table}_{_dc}"[:60].replace('"', '""')
                    conn.execute(
                        f'CREATE INDEX IF NOT EXISTS "{_in}" '
                        f'ON {_quote_ident(table)} ({_quote_ident(_dc)})')
            conn.commit()
        except Exception:
            logging.getLogger("aria.csv").debug("CSV index build skipped", exc_info=True)
    finally:
        conn.close()

    return f"sqlite:///{sqlite_path}", table, streamed


# ---------------------------------------------------------------------------
# Generic multi-table handling (any dataset: Walmart single-table, CSV
# folders, SQLite DBs). No hardcoded table/column names.
# ---------------------------------------------------------------------------

def discover_csv_tables(folder: str | Path) -> list[Path]:
    """Table discovery for a folder: every ``*.csv`` (case-insensitive)."""
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"CSV folder not found: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )
    if not files:
        raise ValueError(f"No .csv files found in: {folder}")
    return files


def _per_table_entry(csv_path: str | Path, table: str) -> tuple[dict, dict, dict]:
    """Build one ``tables`` entry + helpers from a single CSV (generic)."""
    extractor = CsvSchemaExtractor(csv_path)
    profile = extractor.extract()
    table = safe_table_name(table or profile["table_name"])
    entry, logical = _table_entry_from_profile(profile, table, str(csv_path))
    return entry, logical, profile


def _attach_generic_relationships(tables_mapping: dict,
                                  value_sets: dict | None = None) -> dict:
    """Cross-table FK inference + graph + disambiguation (generic).

    ``value_sets`` maps (table, column) -> iterable of raw values (or None
    for name-only inference). Single-table inputs validly yield zero edges.
    """
    from core.relational import (
        build_relationship_graph,
        duplicate_columns,
        infer_foreign_keys,
    )
    tables_columns = {
        t: {c["column"]: c.get("data_type", "TEXT")
            for c in (info.get("columns") or [])}
        for t, info in tables_mapping.items()
    }
    pk_map: dict[str, list] = {}
    for t, info in tables_mapping.items():
        pk = list(info.get("primary_key") or []) or \
            list(info.get("inferred_primary_key") or [])
        if not pk:
            pk = list(info.get("inferred_composite_primary_key") or [])
        pk_map[t] = [str(c) for c in pk]
    inferred, ambiguous, edges = infer_foreign_keys(
        tables_columns, pk_map, value_sets)
    # mapping contract uses source_/target_ keys; keep left_/right_ too.
    mapping_edges = []
    for e in edges:
        mapping_edges.append({
            "source_table": e["left_table"], "source_column": e["left_column"],
            "target_table": e["right_table"], "target_column": e["right_column"],
            "left_table": e["left_table"], "left_column": e["left_column"],
            "right_table": e["right_table"], "right_column": e["right_column"],
            "type": "inferred", "confidence": e.get("confidence"),
            "confidence_score": e.get("confidence_score"),
            "trust": e.get("trust"), "cardinality": e.get("cardinality"),
        })
    col_lists = {t: list(cols) for t, cols in
                 ({t: list(d) for t, d in
                   ({t: {c['column'] for c in (info.get('columns') or [])}
                     for t, info in tables_mapping.items()}).items()}).items()}
    dupes = duplicate_columns(col_lists)
    return {
        "declared_relationships": [],
        "inferred_relationships": inferred,
        "ambiguous_relationships": ambiguous,
        "relationship_edges": mapping_edges,
        "relationship_graph": build_relationship_graph(
            sorted(tables_mapping), edges),
        "duplicate_columns": dupes,
    }


def build_multi_csv_schema_mapping(csv_paths: list[str | Path] | str | Path,
                                   table_names: list[str] | None = None,
                                   database: str | None = None) -> dict[str, Any]:
    """Generic multi-table mapping for any set/folder of CSVs.

    Accepts a folder path or an explicit file list. Each CSV becomes one
    table (per-table PK inference is the single-table logic); cross-table
    relationships come from generic name/type overlap + pandas value
    containment. A single CSV degrades to the single-table contract
    (zero edges, ``relationships.none`` downstream).
    """
    if isinstance(csv_paths, (str, Path)) and Path(csv_paths).is_dir():
        files = discover_csv_tables(csv_paths)
    elif isinstance(csv_paths, (str, Path)):
        files = [Path(csv_paths)]
    else:
        files = [Path(p) for p in csv_paths]
    if table_names and len(table_names) != len(files):
        raise ValueError("table_names length must match csv file count")
    tables_mapping: dict[str, Any] = {}
    value_sets: dict[tuple, set] = {}
    for i, fp in enumerate(files):
        # Single extract per file: the profile feeds the entry AND the
        # (capped) value sets, so the file is never parsed twice here.
        extractor = CsvSchemaExtractor(fp)
        profile = extractor.extract()
        table = safe_table_name(table_names[i] if table_names else profile["table_name"])
        entry, _logical = _table_entry_from_profile(profile, table, str(fp))
        tables_mapping[table] = entry
        try:
            for chunk in extractor.iter_chunks(_LOAD_CHUNK_ROWS):
                for col in chunk.columns:
                    key = (table, str(col))
                    tracked = value_sets.setdefault(key, set())
                    if len(tracked) >= _MAX_VALUE_SET:
                        continue
                    for v in chunk[str(col)].tolist():
                        if v is not None and str(v).strip() != "":
                            tracked.add(v)
                            if len(tracked) >= _MAX_VALUE_SET:
                                break
                del chunk
        except Exception:
            logging.getLogger("aria.csv").debug(
                "value sampling failed for %s", fp, exc_info=True)
    rel = _attach_generic_relationships(tables_mapping, value_sets)
    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + _uuid.uuid4().hex[:8],
        "database": _safe_db_name(database or (Path(files[0]).stem if files else "csv_collection")),
        "schema": "file",
        "tables": tables_mapping,
        "declared_relationships": rel["declared_relationships"],
        "inferred_relationships": rel["inferred_relationships"],
        "ambiguous_relationships": rel["ambiguous_relationships"],
        "relationship_edges": rel["relationship_edges"],
        "relationship_graph": rel["relationship_graph"],
        "duplicate_columns": rel["duplicate_columns"],
        "semi_structured_sources": {
            str(p): {"tables": sorted(tables_mapping)} for p in files
        },
    }
    try:
        from core.data_intelligence import build_data_intelligence, detect_capabilities
        _intel = build_data_intelligence(mapping)
        mapping["data_intelligence"] = _intel
        mapping["capabilities"] = detect_capabilities(_intel)
    except Exception:
        logging.getLogger("aria.csv").debug("data intelligence attach failed", exc_info=True)
    return mapping


def load_multiple_csvs_to_sqlite(csv_paths: list[str | Path] | str | Path,
                                 sqlite_path: str | Path,
                                 table_names: list[str] | None = None,
                                 ) -> tuple[str, list[str], dict[str, int]]:
    """Load any set/folder of CSVs into ONE shared SQLite DB (generic)."""
    if isinstance(csv_paths, (str, Path)) and Path(csv_paths).is_dir():
        files = discover_csv_tables(csv_paths)
    elif isinstance(csv_paths, (str, Path)):
        files = [Path(csv_paths)]
    else:
        files = [Path(p) for p in csv_paths]
    if table_names and len(table_names) != len(files):
        raise ValueError("table_names length must match csv file count")
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sqlite_path.unlink(missing_ok=True)
    except PermissionError:
        pass
    tables: list[str] = []
    counts: dict[str, int] = {}
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for i, fp in enumerate(files):
            extractor = CsvSchemaExtractor(fp)
            profile = extractor.extract()
            table = safe_table_name(
                table_names[i] if table_names else profile["table_name"])
            column_logical_types = {c["name"]: c["inferred_type"]
                                    for c in profile["columns"]}
            try:
                if int(profile.get("row_count") or 0) > 0:
                    _sample = extractor.read_sample(_DATE_RESOLVE_SAMPLE_ROWS)
                    date_formats = _resolve_date_formats(_sample, column_logical_types)
                    del _sample
                else:
                    date_formats = {}
            except Exception as exc:
                _log.debug("date format resolution failed for table %s: %s", table, exc)
                date_formats = {}
            column_types = {n: _SQLITE_TYPE.get(l, "TEXT")
                            for n, l in column_logical_types.items()}
            col_names = list(column_types)
            if not col_names:
                continue
            ddl = ", ".join(f"{_quote_ident(n)} {column_types[n]}" for n in col_names)
            conn.execute(f"CREATE TABLE {_quote_ident(table)} ({ddl})")
            ins = (f"INSERT INTO {_quote_ident(table)} "
                   f"({', '.join(_quote_ident(n) for n in col_names)}) "
                   f"VALUES ({', '.join('?' for _ in col_names)})")
            streamed = 0
            for chunk in extractor.iter_chunks(_LOAD_CHUNK_ROWS):
                if date_formats:
                    try:
                        _apply_date_formats(chunk, date_formats)
                    except Exception as exc:
                        _log.debug("date normalization failed for table %s: %s", table, exc)
                batch = [[_coerce_value(r.get(n), column_logical_types[n])
                          for n in col_names]
                         for r in chunk.to_dict(orient="records")]
                del chunk
                for start in range(0, len(batch), _INSERT_BATCH):
                    part = batch[start:start + _INSERT_BATCH]
                    if part:
                        conn.executemany(ins, part)
                streamed += len(batch)
                del batch
            tables.append(table)
            counts[table] = streamed
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{sqlite_path}", tables, counts


def build_sqlite_schema_mapping(sqlite_path: str | Path,
                                database: str | None = None) -> dict[str, Any]:
    """Generic mapping for ANY SQLite DB file (table discovery via catalog).

    Reads ``sqlite_master`` + ``PRAGMA table_info`` + row counts, infers
    single-column PKs from pragma ``pk`` flags / value uniqueness and
    composite pairs generically, then runs the same generic FK inference
    (name/type + value containment) as the CSV path. No hardcoded names.
    """
    sqlite_path = Path(sqlite_path)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
        tables_mapping: dict[str, Any] = {}
        value_sets: dict[tuple, set] = {}
        for t in tables:
            # Table/column identifiers come from sqlite_master (file content):
            # always quote via _quote_ident (validated, double-quote-escaped).
            # PRAGMA / COUNT / DISTINCT cannot use "?" placeholders for
            # identifiers, so quoting is the parameterisation. CSV write path
            # below is unchanged (identifiers there are created by ARIA).
            pragma = conn.execute(f'PRAGMA table_info({_quote_ident(t)})').fetchall()
            # (cid, name, type, notnull, dflt, pk)
            cols = [{"column": r[1], "data_type": (r[2] or "TEXT"),
                     "nullable": not bool(r[3])} for r in pragma]
            pragma_pk = [r[1] for r in sorted(
                [r for r in pragma if r[5]], key=lambda r: r[5])]
            n = conn.execute(f'SELECT COUNT(*) FROM {_quote_ident(t)}').fetchone()[0]
            null_stats = {}
            for c in cols:
                name = c["column"]
                nulls = conn.execute(
                    f'SELECT COUNT(*) FROM {_quote_ident(t)} WHERE {_quote_ident(name)} IS NULL'
                ).fetchone()[0]
                null_stats[name] = {"total_rows": n, "nulls": nulls,
                                    "null_pct": round(nulls / n * 100, 2) if n else 0.0}
            inferred_pk: list[str] = []
            if len(pragma_pk) == 1:
                inferred_pk = pragma_pk
            elif not pragma_pk:
                # generic single-column uniqueness probe
                for c in cols:
                    name = c["column"]
                    if null_stats[name]["nulls"]:
                        continue
                    distinct = conn.execute(
                        f'SELECT COUNT(DISTINCT {_quote_ident(name)}) FROM {_quote_ident(t)}').fetchone()[0]
                    if n and distinct == n:
                        from csv_handler import CsvSchemaExtractor as _E
                        if _E._is_identifier_like(name):
                            inferred_pk = [name]
                            break
            composite_pk: list[str] = []
            if not inferred_pk and n and n > 1:
                import itertools as _it
                # Reuse the CSV handler's generic composite ranking
                # (identifier-like first, continuous measures last) while
                # preserving the DB native column types/indexes untouched.
                try:
                    from csv_handler import CsvSchemaExtractor as _E
                    _order = {"INTEGER": 1, "DATE": 2, "DATETIME": 2,
                              "STRING": 3, "BOOLEAN": 4, "DECIMAL": 5,
                              "UNKNOWN": 6}

                    def _rank(col_name: str) -> tuple:
                        if _E._is_identifier_like(col_name):
                            return (0, col_name)
                        # Map the native SQLite type to the handler's
                        # logical vocabulary for ranking purposes only.
                        _native = next(
                            (c for c in cols if c["column"] == col_name), {})
                        _dt = str(_native.get("data_type", "")).upper()
                        _logical = "STRING"
                        if "INT" in _dt:
                            _logical = "INTEGER"
                        elif any(k in _dt for k in ("REAL", "FLOA", "DOUB",
                                                   "DECIMAL", "NUMERIC")):
                            _logical = "DECIMAL"
                        elif "BOOL" in _dt:
                            _logical = "BOOLEAN"
                        elif "DATE" in _dt and "TIME" in _dt:
                            _logical = "DATETIME"
                        elif "DATE" in _dt or "TIME" in _dt:
                            _logical = "DATE"
                        return (_order.get(_logical, 3), col_name)
                except Exception as exc:
                    _log.debug("column rank inference failed: %s", exc)
                    def _rank(col_name: str) -> tuple:  # type: ignore[misc]
                        return (3, col_name)
                elig = sorted(
                    (c["column"] for c in cols
                     if not null_stats[c["column"]]["nulls"]),
                    key=_rank,
                )
                # Skip trailing continuous-measure columns when
                # dimension-like columns exist (mirrors handler policy).
                try:
                    _non_measure = [c for c in elig if _rank(c)[0] < 5]
                    _pool = (_non_measure or elig)[:6]
                except Exception as exc:
                    _log.debug("pool selection fallback: %s", exc)
                    _pool = elig[:6]
                _pairs = list(_it.combinations(_pool, 2))
                if _pool:
                    _top = _pool[0]
                    _pairs.sort(key=lambda p: (0 if _top in p else 1, p))
                for a, b in _pairs[:12]:
                    dup = conn.execute(
                        f'SELECT 1 FROM {_quote_ident(t)} GROUP BY {_quote_ident(a)}, {_quote_ident(b)} '
                        f'HAVING COUNT(*) > 1 LIMIT 1').fetchone()
                    if dup is None:
                        composite_pk = [a, b]
                        break
            tables_mapping[t] = {
                "columns": cols,
                "primary_key": pragma_pk if len(pragma_pk) > 1 else [],
                "inferred_primary_key": inferred_pk,
                "inferred_composite_primary_key": composite_pk,
                "composite_candidate_keys": [composite_pk] if composite_pk else [],
                "null_stats": null_stats,
                "row_count": n,
                "empty": n == 0,
                "source": str(sqlite_path),
            }
            for c in cols:
                try:
                    vals = conn.execute(
                        f'SELECT DISTINCT {_quote_ident(c["column"])} FROM {_quote_ident(t)} '
                        f'WHERE {_quote_ident(c["column"])} IS NOT NULL LIMIT 2000').fetchall()
                    value_sets[(t, c["column"])] = {r[0] for r in vals}
                except Exception as exc:
                    _log.debug("value set query failed for %s.%s: %s", t, c["column"], exc)
                    value_sets[(t, c["column"])] = set()
    finally:
        conn.close()
    rel = _attach_generic_relationships(tables_mapping, value_sets)
    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + _uuid.uuid4().hex[:8],
        "database": _safe_db_name(database or Path(sqlite_path).stem),
        "schema": "main",
        "tables": tables_mapping,
        "declared_relationships": rel["declared_relationships"],
        "inferred_relationships": rel["inferred_relationships"],
        "ambiguous_relationships": rel["ambiguous_relationships"],
        "relationship_edges": rel["relationship_edges"],
        "relationship_graph": rel["relationship_graph"],
        "duplicate_columns": rel["duplicate_columns"],
    }
    # Single-table rule: a 1-table SQLite DB is profiled exactly like a
    # CSV dataset (PK-composite / FK-none / relationship-empty / no-join)
    # while native types/indexes stay preserved on the table entry.
    try:
        from core.relational import normalize_single_table_mapping
        mapping = normalize_single_table_mapping(mapping)
    except Exception:
        logging.getLogger("aria.csv").debug(
            "single-table normalization failed", exc_info=True)
    try:
        from core.data_intelligence import build_data_intelligence, detect_capabilities
        _intel = build_data_intelligence(mapping)
        mapping["data_intelligence"] = _intel
        mapping["capabilities"] = detect_capabilities(_intel)
    except Exception:
        logging.getLogger("aria.csv").debug("data intelligence attach failed", exc_info=True)
    return mapping
