"""Deterministic numeric reasoning over retrieved Markdown tables.

Natural-language intent is resolved by the semantic router. This module receives
explicit operations and performs only auditable pandas calculations.
"""

from __future__ import annotations

import re
import math
from collections import defaultdict

import pandas as pd
from langchain_core.documents import Document

from .retrieval import deduplicate_chunks


def _split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    placeholder = "__ARIA_PIPE__"
    line = line.replace("\\|", placeholder)
    return [cell.strip().replace(placeholder, "|") for cell in line.split("|")]


def markdown_to_frame(text: str) -> pd.DataFrame | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return None
    header = _split_markdown_row(lines[0])
    if not header or len(set(header)) != len(header):
        return None
    separator = _split_markdown_row(lines[1])
    if len(separator) != len(header) or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        return None
    rows = []
    for line in lines[2:]:
        row = _split_markdown_row(line)
        if len(row) != len(header):
            return None
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def _to_number(value):
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
        return (-number if negative else number) if math.isfinite(number) else None
    except ValueError:
        return None


def _normalise_column(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1
    }


def _numeric_series(frame: pd.DataFrame) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for column in frame.columns:
        numeric = frame[column].map(_to_number)
        valid = numeric.dropna()
        if valid.empty:
            continue
        non_empty = frame[column].astype(str).str.strip().ne("").sum()
        if len(valid) < max(1, int(non_empty * 0.5)):
            continue
        result[str(column)] = numeric
    return result


def _relevant_numeric_columns(question: str, numeric: dict[str, pd.Series]) -> list[str]:
    if not numeric:
        return []
    q_tokens = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    ranked = []
    for column in numeric:
        tokens = _normalise_column(column)
        overlap = len(tokens & q_tokens)
        ranked.append((overlap, column))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked and ranked[0][0] > 0:
        best = ranked[0][0]
        return [column for score, column in ranked if score == best]
    return list(numeric)


def _label_columns(frame: pd.DataFrame, numeric_columns: set[str]) -> list[str]:
    return [str(column) for column in frame.columns if str(column) not in numeric_columns][:3]


def _row_descriptor(frame: pd.DataFrame, index, label_columns: list[str]) -> str:
    parts = []
    for column in label_columns:
        value = str(frame.loc[index, column]).strip()
        if value and value.lower() != "nan":
            parts.append(f"{column}={value}")
    return ", ".join(parts) if parts else f"row {int(index) + 1}"


def _filter_matches(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    operator: str,
    threshold: float,
    label_columns: list[str],
    column: str,
) -> list[str]:
    operator = operator.upper()
    if operator == "GT":
        mask = series > threshold
    elif operator == "GTE":
        mask = series >= threshold
    elif operator == "LT":
        mask = series < threshold
    elif operator == "LTE":
        mask = series <= threshold
    else:
        mask = series == threshold

    matches = []
    for index in frame.index[mask.fillna(False)]:
        descriptor = _row_descriptor(frame, index, label_columns)
        value = series.loc[index]
        matches.append(f"{descriptor}; {column}={float(value):g}")
        if len(matches) >= 20:
            break
    return matches


def _rank_rows(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    label_columns: list[str],
    column: str,
    top_n: int,
) -> list[str]:
    valid = series.dropna().astype(float).sort_values(ascending=False).head(max(1, min(top_n, 20)))
    return [
        f"{_row_descriptor(frame, index, label_columns)}; {column}={float(value):g}"
        for index, value in valid.items()
    ]


# Generic aggregation-intent patterns. These describe *operations* (total,
# average, highest, ...) in plain language, never concrete dataset, table,
# or column names, so they apply to any tabular source.
_INFER_SUM = re.compile(r"\b(total|totals|sum|sums|combined|overall|grand\s+total)\b", re.IGNORECASE)
_INFER_MEAN = re.compile(r"\b(average|averages|avg|mean|typical)\b", re.IGNORECASE)
_INFER_MAX = re.compile(
    r"\b(max(?:imum)?|highest|higher|high|largest|greater|greatest|peak|best(?:-selling|-seller)?|top(?:-selling)?)\b",
    re.IGNORECASE,
)
_INFER_MIN = re.compile(
    r"\b(min(?:imum)?|lowest|lower|low|smallest|least|bottom|worst|poorest)\b",
    re.IGNORECASE,
)
_INFER_COUNT = re.compile(r"\b(count|counts|how\s+many|number\s+of|total\s+number)\b", re.IGNORECASE)
_INFER_COMPARE = re.compile(
    r"\b(compare|compares|comparison|vs\.?|versus|difference|breakdown|distribution|across|per\b.+|by\b.+)\b",
    re.IGNORECASE,
)
_INFER_RANK = re.compile(r"\btop\s+(\d{1,3})\b|\bbest\s+(\d{1,3})\b|\bhighest\s+(\d{1,3})\b", re.IGNORECASE)
_INFER_FILTER = re.compile(
    r"(over|above|greater\s+than|more\s+than|at\s+least|minimum\s+of|"
    r"under|below|less\s+than|fewer\s+than|at\s+most|maximum\s+of|"
    r"equal\s+to|exactly|>|<|=)\s*[$£€₹]?\s*(\d[\d,]*(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)

def infer_table_operations(question: str) -> tuple[tuple[str, ...], str, float | None, int | None]:
    """Infer deterministic table operations from generic question phrasing.

    Used only as a fallback when the semantic router supplied no explicit
    operations. Patterns are operation-level English (total, average,
    highest, top-N, over-X) and never name a dataset, table, or column.
    Returns ``(operations, filter_operator, filter_value, top_n)``.
    """
    text = (question or "").strip()
    if not text:
        return (), "NONE", None, None
    operations: list[str] = []

    def add(operation: str) -> None:
        if operation not in operations:
            operations.append(operation)

    if _INFER_SUM.search(text):
        add("SUM")
    if _INFER_MEAN.search(text):
        add("MEAN")
    if _INFER_COUNT.search(text):
        add("COUNT")
    has_rank_n: int | None = None
    rank_match = _INFER_RANK.search(text)
    if rank_match:
        digits = next((group for group in rank_match.groups() if group), None)
        try:
            has_rank_n = max(1, min(100, int(digits))) if digits else 5
        except (TypeError, ValueError):
            has_rank_n = 5
        add("RANK")
    if _INFER_MAX.search(text) and has_rank_n is None:
        add("MAX")
    if _INFER_MIN.search(text):
        add("MIN")
    if _INFER_COMPARE.search(text):
        add("COMPARE")

    filter_operator = "NONE"
    filter_value: float | None = None
    filter_match = _INFER_FILTER.search(text)
    if filter_match:
        word = (filter_match.group(1) or "").strip().casefold()
        raw_number = (filter_match.group(2) or "").replace(",", "")
        try:
            threshold = float(raw_number)
        except ValueError:
            threshold = None
        if threshold is not None and threshold == threshold:  # finite check
            if word in {">", "over", "above"} or "greater" in word or "more" in word:
                filter_operator = "GT"
            elif word in {"<", "under", "below"} or "less" in word or "fewer" in word:
                filter_operator = "LT"
            elif "at least" in word or "minimum" in word:
                filter_operator = "GTE"
            elif "at most" in word or "maximum" in word:
                filter_operator = "LTE"
            elif "equal" in word or "exactly" in word or word == "=":
                filter_operator = "EQ"
            else:
                filter_operator = "GT" if word in {"over", "above"} else "GTE"
            filter_value = threshold
            add("FILTER")

    # A bare "how many" with no other signal is a COUNT, already added.
    # Avoid emitting COMPARE alone for pure row listings ("top 5" is RANK).
    return tuple(operations), filter_operator, filter_value, has_rank_n


def _summary_facts_for_column(
    column: str,
    entry: dict,
    *,
    operations: set[str],
    label: str,
) -> list[str]:
    """Emit full-scope aggregate facts from an ingest-time summary entry."""
    facts: list[str] = []
    missing = int(entry.get("missing") or 0)
    if missing:
        facts.append(f"{column}: {missing} missing/unparseable cells; aggregates withheld")
        return facts
    if entry.get("percent_any") and not entry.get("percent_all"):
        facts.append(f"{column}: mixed percent and plain values; aggregates withheld")
        return facts
    currencies = list(entry.get("currencies") or [])
    if len(currencies) > 1:
        facts.append(f"{column}: mixed currencies; aggregates withheld")
        return facts
    if not entry.get("count"):
        return facts
    if "SUM" in operations or "COMPARE" in operations:
        if entry.get("percent_all"):
            facts.append(f"{column}: percentage sum withheld (denominator/weight required)")
        else:
            facts.append(f"{column} sum={float(entry['sum']):g}")
    if "MEAN" in operations or "COMPARE" in operations:
        facts.append(f"{column} mean={float(entry['mean']):g}")
    if "COUNT" in operations:
        facts.append(f"{column} count={int(entry['count'])}")
    if "MAX" in operations or "COMPARE" in operations:
        facts.append(f"{column} max={float(entry['max']):g} ({label} full-table scope)")
    if "MIN" in operations or "COMPARE" in operations:
        facts.append(f"{column} min={float(entry['min']):g} ({label} full-table scope)")
    return facts


def build_table_facts(
    question: str,
    docs: list[Document],
    *,
    operations: tuple[str, ...] | list[str] = (),
    filter_operator: str = "NONE",
    filter_value: float | None = None,
    top_n: int | None = None,
    sources: list[dict] | None = None,
    allow_inference: bool = True,
) -> str:
    """Create deterministic facts from retrieved table rows.

    ``operations`` comes from the validated semantic route when present. When
    the router supplied no explicit operations, a generic operation-level
    fallback (total, average, highest, top-N, over-X -- never dataset, table,
    or column names) is inferred from the question so tabular evidence still
    yields auditable math instead of guessed numbers.
    """
    op_set = {str(operation).upper() for operation in operations}
    if not op_set and filter_operator == "NONE" and allow_inference:
        inferred_ops, inferred_filter_op, inferred_filter_value, inferred_top_n = infer_table_operations(question)
        op_set = set(inferred_ops)
        if inferred_filter_op != "NONE" and inferred_filter_value is not None:
            filter_operator, filter_value = inferred_filter_op, inferred_filter_value
        if top_n is None:
            top_n = inferred_top_n
    if not op_set and filter_operator == "NONE":
        return ""

    grouped: dict[tuple, list[Document]] = defaultdict(list)
    for doc in deduplicate_chunks(docs):
        meta = doc.metadata
        if meta.get("content_type") != "table":
            continue
        key = (
            meta.get("document_id"),
            meta.get("filename"),
            meta.get("page"),
            meta.get("table_index"),
        )
        grouped[key].append(doc)

    sections = []
    for key, table_docs in grouped.items():
        row_docs = [doc for doc in table_docs if not doc.metadata.get("table_summary")]
        summary_docs = [doc for doc in table_docs if doc.metadata.get("table_summary")]
        summary_aggs: dict = {}
        summary_row_count: int | None = None
        summary_sampled = False
        summary_has_total_like = False
        for summary in summary_docs:
            candidate = summary.metadata.get("table_summary_aggregates")
            if isinstance(candidate, dict) and candidate:
                summary_aggs = candidate
                try:
                    summary_row_count = int(summary.metadata.get("table_row_count"))
                except (TypeError, ValueError):
                    summary_row_count = None
                summary_sampled = bool(summary.metadata.get("table_sampled"))
                summary_has_total_like = bool(summary.metadata.get("table_has_total_like_row"))
                break
        filename, page, table_index = key[1], key[2], key[3]
        labels = [str(source["source_id"]) for source in (sources or [])
                  if (source.get("document_id"), source.get("filename"), source.get("page"), source.get("table_index")) == key]
        provenance = " ".join(f"[{label}]" for label in labels)

        row_frame = None
        row_complete = False
        # Only chunks explicitly marked as a bounded ingest-time sample may
        # stand in for the full table. Ordinary incomplete chunk sets (e.g. a
        # partially retrieved PDF table) must still withhold aggregates.
        rows_are_sample = any(bool(doc.metadata.get("table_sampled")) for doc in row_docs)
        if row_docs:
            frames = []
            ordered = sorted(row_docs, key=lambda doc: doc.metadata.get("table_chunk_index", 0))
            expected_counts = {doc.metadata.get("table_chunk_count") for doc in ordered
                               if doc.metadata.get("table_chunk_count") is not None}
            row_complete = rows_are_sample or not expected_counts or (len(expected_counts) == 1 and
                        {doc.metadata.get("table_chunk_index") for doc in ordered} == set(range(next(iter(expected_counts)))))
            malformed = False
            for doc in ordered:
                frame = markdown_to_frame(doc.page_content)
                if frame is not None:
                    frames.append(frame)
                else:
                    malformed = True
            if not malformed and frames and all(
                list(item.columns) == list(frames[0].columns) for item in frames):
                try:
                    row_frame = pd.concat(frames, ignore_index=True).reset_index(drop=True)
                except (ValueError, TypeError):
                    row_frame = None
        elif summary_docs and not summary_aggs:
            # Summary chunk without machine-readable aggregates: parse its
            # embedded sample rows as a last resort.
            for doc in summary_docs:
                frame = markdown_to_frame(doc.page_content)
                if frame is not None:
                    row_frame = frame
                    row_complete = True
                    break

        if row_frame is None and not summary_aggs:
            if row_docs and not row_complete:
                sections.append(f"No calculation: incomplete table from {filename}, page {page}, table {table_index}.")
            continue
        if row_docs and not row_complete and not summary_aggs:
            sections.append(f"No calculation: incomplete table from {filename}, page {page}, table {table_index}.")
            continue

        numeric: dict[str, pd.Series] = {}
        if row_frame is not None:
            numeric = _numeric_series(row_frame)
        if not numeric and summary_aggs:
            numeric = {column: pd.Series([], dtype=float) for column in summary_aggs
                       if isinstance(summary_aggs[column], dict) and summary_aggs[column].get("count")}
        relevant_columns = _relevant_numeric_columns(question, numeric)
        if summary_aggs and not relevant_columns:
            relevant_columns = [column for column in summary_aggs
                                if isinstance(summary_aggs[column], dict) and summary_aggs[column].get("count")]
        if not relevant_columns:
            continue

        label_columns: list[str] = []
        if row_frame is not None:
            label_columns = _label_columns(row_frame, set(numeric if isinstance(numeric, dict) else {}))
            has_total_row = any(
                row_frame[column].astype(str).str.strip().str.casefold().isin(
                    {"total", "subtotal", "grand total", "sub-total"}
                ).any() for column in label_columns
            )
        else:
            has_total_row = summary_has_total_like
        if has_total_row and op_set & {"SUM", "MEAN", "COUNT", "COMPARE"}:
            sections.append(f"No aggregate calculation: {filename}, page {page}, table {table_index} contains total/subtotal rows; isolate detail rows first.")
            continue

        facts: list[str] = []
        scope_rows = summary_row_count if summary_aggs else (len(row_frame) if row_frame is not None else 0)
        scope_note = (
            f"Calculation scope: {scope_rows} rows (full-table summary). {provenance}"
            if summary_aggs
            else f"Calculation scope: {len(row_frame)} extracted rows from this table only. {provenance}"
        )
        if summary_sampled and summary_aggs:
            scope_note += " Row chunks are a bounded sample; aggregates cover all rows."
        elif rows_are_sample and row_frame is not None:
            scope_note += " Row chunks are a bounded sample; aggregates cover the sampled rows."
        for column in relevant_columns:
            summary_entry = summary_aggs.get(column) if summary_aggs else None
            if isinstance(summary_entry, dict) and summary_entry.get("count"):
                facts.extend(_summary_facts_for_column(
                    column, summary_entry, operations=op_set, label=filename or "table"))
                series = numeric.get(column) if isinstance(numeric, dict) else None
                try:
                    row_valid = (
                        row_frame is not None and series is not None
                        and not series.dropna().astype(float).empty
                    )
                except (ValueError, TypeError):
                    row_valid = False
            else:
                series = numeric.get(column) if isinstance(numeric, dict) else None
                if row_frame is None or series is None:
                    continue
                valid = series.dropna().astype(float)
                if valid.empty:
                    continue
                missing = int(series.isna().sum())
                if missing:
                    facts.append(f"{column}: {missing} missing/unparseable cells; aggregates withheld")
                    continue
                raw_values = row_frame[column].astype(str).str.strip()
                percent = raw_values.str.endswith("%")
                if percent.any() and not percent.all():
                    facts.append(f"{column}: mixed percent and plain values; aggregates withheld")
                    continue
                currencies = {symbol for value in raw_values for symbol in "$£€₹" if symbol in value}
                if len(currencies) > 1:
                    facts.append(f"{column}: mixed currencies; aggregates withheld")
                    continue
                if percent.all() and "SUM" in op_set:
                    facts.append(f"{column}: percentage sum withheld (denominator/weight required)")
                if ("SUM" in op_set or "COMPARE" in op_set) and not percent.all():
                    facts.append(f"{column} sum={float(valid.sum()):g}")
                if "MEAN" in op_set or "COMPARE" in op_set:
                    facts.append(f"{column} mean={float(valid.mean()):g}")
                if "COUNT" in op_set:
                    facts.append(f"{column} count={int(valid.count())}")
                if "MAX" in op_set or "COMPARE" in op_set:
                    max_index = valid.idxmax()
                    facts.append(
                        f"{column} max={float(valid.loc[max_index]):g} at "
                        f"{_row_descriptor(row_frame, max_index, label_columns)}"
                    )
                if "MIN" in op_set or "COMPARE" in op_set:
                    min_index = valid.idxmin()
                    facts.append(
                        f"{column} min={float(valid.loc[min_index]):g} at "
                        f"{_row_descriptor(row_frame, min_index, label_columns)}"
                    )
                row_valid = True

            if row_valid and row_frame is not None and isinstance(numeric, dict) and column in numeric:
                series = numeric[column]
                if "FILTER" in op_set and filter_operator != "NONE" and filter_value is not None:
                    try:
                        matches = _filter_matches(
                            row_frame,
                            series,
                            operator=filter_operator,
                            threshold=float(filter_value),
                            label_columns=label_columns,
                            column=column,
                        )
                    except (ValueError, TypeError, KeyError):
                        matches = []
                    facts.append(
                        f"{column} filter {filter_operator} {float(filter_value):g}: "
                        + (" | ".join(matches) if matches else "none")
                    )

                if "RANK" in op_set:
                    ranked = _rank_rows(
                        row_frame,
                        series,
                        label_columns=label_columns,
                        column=column,
                        top_n=int(top_n or 5),
                    )
                    if ranked:
                        facts.append(f"{column} ranked rows: " + " | ".join(ranked))

        if facts:
            facts.insert(0, scope_note)
            if row_frame is not None and any(
                    row_frame[column].astype(str).str.strip().str.endswith("%").all()
                    for column in relevant_columns if column in row_frame.columns):
                facts.insert(1, "Percentage values use percentage-point units; MEAN is an unweighted row mean, not an overall rate.")
            sections.append(
                f"Computed deterministically from {filename}, page {page}, table {table_index}:\n"
                + "\n".join(f"- {fact}" for fact in facts)
            )

    return "\n\n".join(sections)
