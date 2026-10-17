"""Generic tabular ingestion for Insight RAG.

PDF-only indexing is why analytics questions over row-oriented sources (any
CSV or database table) fail in RAG: there is no tabular evidence to retrieve,
so numeric answers are guessed or refused. This module converts an arbitrary
row set -- a DataFrame, a CSV file, a SQLite table, or a list of records --
into the same Markdown-table ``Document`` contract that PDF tables use:

- one small **summary chunk** with full-coverage deterministic aggregates
  (count/sum/mean/min/max over *all* rows) plus schema/column metadata, and
- a bounded stratified **sample of row chunks** with the header repeated in
  every chunk so lexical retrieval matches column names.

No dataset, table, or column names are referenced here. All labels, numeric
detection, and aggregates are derived generically from the frame itself.
Large tables never flood context: row chunks are capped while the summary
still covers every row, and sibling expansion elsewhere stays bounded.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from .config import (
    TABLE_ROWS_PER_CHUNK,
    TABULAR_MAX_ROW_CHUNKS,
    TABULAR_MAX_ROWS,
)
from .pdf_ingest import table_to_markdown

_CURRENCY_SYMBOLS = "$£€₹"
_TOTAL_LIKE = {"total", "subtotal", "grand total", "sub-total"}


def _clean_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).replace("\n", " ").strip()
    return re.sub(r"[ \t]+", " ", text)


def _parse_number(value):
    """Generic numeric parse mirroring the table reasoner's contract."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = re.sub(r"[$£€₹%]", "", text).strip()
    if "," in text and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        return None
    text = text.replace(",", "").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return -number if negative else number


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    names = [str(column).strip() or f"column_{i + 1}" for i, column in enumerate(frame.columns)]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    frame.columns = unique
    return frame


def _column_kind(values: list[str]) -> str:
    non_empty = [value for value in values if value.strip()]
    if not non_empty:
        return "UNKNOWN"
    parsed = [_parse_number(value) for value in non_empty]
    valid = [number for number in parsed if number is not None]
    if len(valid) < max(1, int(len(non_empty) * 0.5)):
        return "TEXT"
    if all(float(number).is_integer() for number in valid) and all(
        re.fullmatch(r"[+-]?[\d,()$£€₹%\s.]+", value) and "." not in value.replace(",", "")
        for value in non_empty
        if _parse_number(value) is not None
    ):
        return "INTEGER"
    return "REAL"


def _summarise_frame(frame: pd.DataFrame) -> tuple[dict, dict, bool]:
    """Return (aggregates, column_kinds, has_total_like_row) generically."""
    aggregates: dict[str, dict] = {}
    kinds: dict[str, str] = {}
    for column in frame.columns:
        raw = [_clean_cell(value) for value in frame[column].tolist()]
        kinds[str(column)] = _column_kind(raw)
        parsed = [_parse_number(value) for value in raw]
        valid = [number for number in parsed if number is not None]
        non_empty = [value for value in raw if value.strip()]
        missing = len(raw) - len(non_empty) + (len(non_empty) - len(valid))
        stripped = [value.strip() for value in raw if value.strip()]
        percent_any = any(value.endswith("%") for value in stripped)
        percent_all = bool(stripped) and all(value.endswith("%") for value in stripped)
        currencies = sorted({symbol for value in stripped for symbol in _CURRENCY_SYMBOLS if symbol in value})
        entry: dict = {
            "count": len(valid),
            "missing": int(missing),
            "percent_any": bool(percent_any),
            "percent_all": bool(percent_all),
            "currencies": currencies,
        }
        if valid:
            total = float(sum(valid))
            entry.update({
                "sum": total,
                "mean": float(total / len(valid)),
                "min": float(min(valid)),
                "max": float(max(valid)),
            })
        aggregates[str(column)] = entry
    has_total_like = False
    for column in frame.columns:
        if kinds[str(column)] != "TEXT":
            continue
        lowered = frame[column].astype(str).str.strip().str.casefold()
        if lowered.isin(_TOTAL_LIKE).any():
            has_total_like = True
            break
    return aggregates, kinds, has_total_like


def _stable_chunk_id(document_id: str, table_index: int, ordinal: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aria-tabular:{document_id}:{table_index}:{ordinal}"))


def _stratified_indices(total: int, limit: int) -> list[int]:
    if total <= limit or limit <= 0:
        return list(range(total))
    # Evenly spaced head/middle/tail sample; deterministic, no datasegmentation.
    step = total / limit
    picked = sorted({min(total - 1, int(round(i * step))) for i in range(limit)})
    return picked


def dataframe_to_table_documents(
    frame: pd.DataFrame,
    *,
    document_id: str,
    filename: str,
    table_name: str | None = None,
    table_index: int = 1,
    rows_per_chunk: int = TABLE_ROWS_PER_CHUNK,
    max_row_chunks: int = TABULAR_MAX_ROW_CHUNKS,
) -> tuple[list[Document], dict]:
    """Convert any DataFrame into summary + sampled Markdown-table Documents."""
    if frame is None or len(frame.columns) == 0:
        raise ValueError("Tabular source has no columns to index.")
    frame = _normalise_columns(frame)
    total_rows = int(len(frame))
    if total_rows > TABULAR_MAX_ROWS:
        keep = _stratified_indices(total_rows, TABULAR_MAX_ROWS)
        frame = frame.iloc[keep].reset_index(drop=True)
        total_rows = int(len(frame))
        sampled_for_summary = True
    else:
        sampled_for_summary = False

    aggregates, kinds, has_total_like = _summarise_frame(frame)
    columns = [str(column) for column in frame.columns]
    label = table_name or filename

    numeric_lines = []
    for column in columns:
        entry = aggregates[column]
        if not entry.get("count"):
            numeric_lines.append(f"- {column} ({kinds[column]}): no parseable numeric values")
            continue
        line = (
            f"- {column} ({kinds[column]}): count={entry['count']}, "
            f"sum={entry['sum']:g}, mean={entry['mean']:g}, "
            f"min={entry['min']:g}, max={entry['max']:g}"
        )
        if entry.get("missing"):
            line += f", {entry['missing']} missing/unparseable"
        numeric_lines.append(line)

    header_row = columns
    preview = frame.head(5)
    preview_rows = [[_clean_cell(value) for value in row] for row in preview.itertuples(index=False, name=None)]
    preview_markdown = table_to_markdown([header_row] + preview_rows) if total_rows else ""

    scope_note = (
        f"Row chunks are a stratified sample; aggregates below cover all {total_rows} rows."
        if (sampled_for_summary or total_rows > max_row_chunks * rows_per_chunk)
        else f"Aggregates below cover all {total_rows} rows."
    )
    summary_text = (
        f"Table summary for {label}: {total_rows} rows, {len(columns)} columns.\n"
        f"Columns: " + ", ".join(f"{name} ({kinds[name]})" for name in columns) + "\n"
        f"{scope_note}\n"
        "Numeric summaries:\n" + "\n".join(numeric_lines)
    )
    if preview_markdown:
        summary_text += "\nSample rows:\n" + preview_markdown

    documents: list[Document] = [
        Document(
            page_content=summary_text,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "page": 1,
                "content_type": "table",
                "source_type": "tabular",
                "table_name": label,
                "table_index": table_index,
                "table_summary": True,
                "table_chunk_index": -1,
                "table_chunk_count": 1,  # summary stands alone; row chunks are samples
                "table_row_count": total_rows,
                "table_sampled": bool(sampled_for_summary or total_rows > max_row_chunks * rows_per_chunk),
                "column_names": columns,
                "column_kinds": kinds,
                "table_summary_aggregates": aggregates,
                "table_has_total_like_row": bool(has_total_like),
                "chunk_id": _stable_chunk_id(document_id, table_index, "summary"),
            },
        )
    ]

    body_capacity = max(1, max_row_chunks * rows_per_chunk)
    if total_rows:
        wanted = _stratified_indices(total_rows, min(total_rows, body_capacity))
        sampled = len(wanted) < total_rows
        rows = [[_clean_cell(value) for value in frame.iloc[i].tolist()] for i in wanted]
        chunks = [
            table_to_markdown([header_row] + rows[start:start + rows_per_chunk])
            for start in range(0, len(rows), rows_per_chunk)
        ]
        for ordinal, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page": 1,
                        "content_type": "table",
                        "source_type": "tabular",
                        "table_name": label,
                        "table_index": table_index,
                        "table_summary": False,
                        "table_chunk_index": ordinal,
                        "table_chunk_count": len(chunks),
                        "table_row_count": total_rows,
                        "table_sampled": bool(sampled),
                        "column_names": columns,
                        "chunk_id": _stable_chunk_id(document_id, table_index, f"rows-{ordinal}"),
                    },
                )
            )
    else:
        sampled = False

    stats = {
        "rows": total_rows,
        "columns": len(columns),
        "column_names": columns,
        "table_chunks": len(documents) - 1,
        "total_chunks": len(documents),
        "sampled": bool(sampled_for_summary or (total_rows and len(wanted) < total_rows)) if total_rows else False,
    }
    return documents, stats


def records_to_table_documents(
    records: list[dict],
    *,
    document_id: str,
    filename: str,
    table_name: str | None = None,
    table_index: int = 1,
) -> tuple[list[Document], dict]:
    """Convert a list of row dicts (DB rows) into table Documents."""
    if not records:
        raise ValueError("Tabular source has no rows to index.")
    columns: list[str] = []
    for record in records:
        for key in record:
            if str(key) not in columns:
                columns.append(str(key))
    frame = pd.DataFrame([{column: record.get(column) for column in columns} for record in records])
    return dataframe_to_table_documents(
        frame, document_id=document_id, filename=filename,
        table_name=table_name, table_index=table_index,
    )


def csv_to_table_documents(
    csv_path: str | Path,
    *,
    document_id: str,
    filename: str | None = None,
    table_name: str | None = None,
    table_index: int = 1,
) -> tuple[list[Document], dict]:
    """Convert any CSV file into table Documents (generic dialect handling)."""
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Tabular CSV not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    name = filename or path.name
    return dataframe_to_table_documents(
        frame, document_id=document_id, filename=name,
        table_name=table_name or path.stem, table_index=table_index,
    )


def sqlite_table_to_table_documents(
    db_path: str | Path,
    table: str,
    *,
    document_id: str,
    filename: str | None = None,
    table_index: int = 1,
    row_limit: int = TABULAR_MAX_ROWS,
) -> tuple[list[Document], dict]:
    """Convert any SQLite table into table Documents without dumping the DB."""
    quoted = '"' + str(table).replace('"', '""') + '"'
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(f"SELECT * FROM {quoted} LIMIT {int(row_limit) + 1}")
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
    finally:
        conn.close()
    if not columns:
        raise ValueError(f"Table {table!r} has no columns to index.")
    truncated = len(rows) > row_limit
    frame = pd.DataFrame(rows[:row_limit], columns=columns)
    name = filename or f"{Path(str(db_path)).name}:{table}"
    documents, stats = dataframe_to_table_documents(
        frame, document_id=document_id, filename=name,
        table_name=str(table), table_index=table_index,
    )
    stats["truncated_at_limit"] = bool(truncated)
    return documents, stats
