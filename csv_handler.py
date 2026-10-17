from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_CURRENCY_RE = re.compile(r"[\u20b9\u00a3\u00a5\u20ac$\u00a4\uffe0\uffe1\uffe5\uffe6\u20b1\u20a9]")
_THOUSAND_SEP_RE = re.compile(r"(?<=\d),(?=\d{3})")


def _clean_currency(value: str) -> str:
    """Strip currency symbols, thousand-separator commas, and percentage signs.

    Returns the cleaned numeric-looking string, or the original if no cleaning
    changed anything.  ``"₹1,099"`` -> ``"1099"``, ``"64%"`` -> ``"0.64"``,
    ``"$12,345.67"`` -> ``"12345.67"``.
    """
    v = value.strip()
    v = _CURRENCY_RE.sub("", v)
    v = v.replace(",", "")
    stripped = v.strip()
    if stripped.endswith("%"):
        try:
            num = float(stripped[:-1])
            return str(num / 100)
        except ValueError:
            pass
    return stripped


def _numeric_clean(series: "pd.Series") -> "pd.Series":
    """Vectorized _clean_currency: identical results, no per-element calls.

    Strips currency symbols/thousand commas via regex ops, folds trailing %
    to /100, coerces the rest (unparseable -> NaN, same as before).
    """
    s = series.astype("string").str.strip()
    s = s.str.replace(_CURRENCY_RE, "", regex=True).str.replace(",", "", regex=False)
    pct = s.str.endswith("%")
    core = s.where(~pct, s.str.slice(stop=-1))
    return pd.to_numeric(core, errors="coerce").mask(pct, pd.to_numeric(s.str.slice(stop=-1), errors="coerce") / 100)

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


class CsvValidationError(ValueError):
    """Raised when a CSV file cannot be safely interpreted as a table."""


# Scalability bounds (spec §1). Profiling a large CSV through a single
# full-frame pandas read holds several copies of the data at once (frame +
# string casts + boolean masks + to_dict records) and repeats per-column
# Python loops over every row. Files above FULL_PROFILE_MAX_ROWS use a
# bounded path instead: type inference from a head sample plus one chunked
# streaming pass with a bounded working set. There is NO analytical row cap —
# every data row still contributes to the exact aggregates (row_count,
# null counts, numeric min/max/mean, ISO date range); only the inference
# sample and the in-memory cardinality working sets are bounded. Whenever a
# field is sample-based or estimated, the profile's "sampling" block says so.
FULL_PROFILE_MAX_ROWS = 50_000
INFERENCE_SAMPLE_ROWS = 20_000
PROFILE_CHUNK_ROWS = 50_000
MAX_DISTINCT_TRACKED = 200_000
MAX_DUP_ROWS_TRACKED = 500_000


@dataclass(frozen=True)
class CsvDialectInfo:
    delimiter: str
    quotechar: str
    doublequote: bool
    escapechar: str | None
    skipinitialspace: bool


class CsvSchemaExtractor:
    """
    Deterministic CSV ingestion + schema profiling for ARIA.

    Responsibilities:
    - validate a CSV file before ingestion
    - detect encoding and CSV dialect
    - preserve raw values as strings during ingestion
    - profile row/column structure and data quality
    - conservatively infer logical column types
    - expose candidate keys without pretending they are declared primary keys
    - create a stable file fingerprint for change detection

    This class intentionally does NOT use an LLM for structural CSV parsing.
    An LLM can be added later for semantic enrichment of ambiguous column names.
    """

    DEFAULT_MISSING_TOKENS = frozenset(
        {"", "null", "none", "nan", "missing"}
    )
    CANDIDATE_DELIMITERS = ",;\t|"

    def __init__(
        self,
        file_path: str | Path,
        *,
        sample_values: int = 5,
        max_file_size_mb: int | None = None,
        missing_tokens: Iterable[str] | None = None,
    ) -> None:
        import os as _os
        self.path = Path(file_path)
        self.sample_values = max(1, int(sample_values))
        # Default follows ARIA_MAX_UPLOAD_CSV_MB so every construction site
        # (routes, orchestrator, tests) honors one tunable limit.
        self.max_file_size_mb = (
            max_file_size_mb
            if max_file_size_mb is not None
            else int(_os.getenv("ARIA_MAX_UPLOAD_CSV_MB", "512"))
        )
        self.missing_tokens = {
            token.strip().lower()
            for token in (missing_tokens or self.DEFAULT_MISSING_TOKENS)
        }

        self._encoding: str | None = None
        self._dialect: CsvDialectInfo | None = None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> dict[str, Any]:
        """Parse, validate and return a JSON-serialisable schema profile.

        Row counting rides inside the chunked aggregate pass (no separate
        full scan): small files take the exact full-frame path, large files
        the bounded sampled path. Results are identical to before.
        """
        self._validate_file()
        self._encoding = self._detect_encoding()
        self._dialect = self._detect_dialect(self._encoding)
        self._validate_header(self._encoding, self._dialect)
        row_count = self.count_rows()
        if row_count <= FULL_PROFILE_MAX_ROWS:
            return self._extract_full(row_count)
        return self._extract_sampled(row_count)

    def _extract_full(self, row_count: int) -> dict[str, Any]:
        """Exact single-frame profile for small files (previous behaviour)."""
        assert self._encoding is not None and self._dialect is not None
        self._df = self._read_dataframe(self._encoding, self._dialect)
        # count_rows() skips blank lines exactly like pandas (skip_blank_lines),
        # so the streaming count must agree with the parsed frame here.
        row_count = len(self._df)

        table_name = self._safe_table_name(self.path.stem)

        columns = [
            self._profile_column(self._df[column_name], column_name, row_count)
            for column_name in self._df.columns
        ]

        duplicate_rows = int(self._df.duplicated().sum()) if row_count else 0
        duplicate_pct = round((duplicate_rows / row_count) * 100, 2) if row_count else 0.0

        composite_keys = self._detect_composite_keys(self._df, columns, row_count)

        warnings = self._build_warnings(columns, row_count, duplicate_rows)

        return {
            "source_type": "csv",
            "source_name": self.path.name,
            "table_name": table_name,
            "file": self._file_block(),
            "row_count": row_count,
            "column_count": len(self._df.columns),
            "columns": columns,
            "composite_candidate_keys": composite_keys,
            "quality": {
                "duplicate_rows": duplicate_rows,
                "duplicate_row_percentage": duplicate_pct,
                "warnings": warnings,
            },
            "sampling": {
                "sampled": False,
                "sample_rows": row_count,
                "total_rows": row_count,
                "exact_fields": [
                    "row_count", "null_count", "distinct_count",
                    "duplicate_rows", "statistics", "composite_candidate_keys",
                ],
                "sample_based_fields": [],
            },
        }

    def _extract_sampled(self, row_count: int) -> dict[str, Any]:
        """Bounded-memory profile for large files.

        Inference (logical types, sample values, medians, composite ranking)
        comes from a bounded head sample; row_count, null counts, numeric
        min/max/mean and ISO date ranges are exact over ALL rows via one
        chunked streaming pass. Cardinality fields degrade gracefully with
        explicit flags instead of unbounded memory.
        """
        assert self._encoding is not None and self._dialect is not None
        sample_n = min(INFERENCE_SAMPLE_ROWS, row_count)
        sample_df = self.read_sample(sample_n)
        table_name = self._safe_table_name(self.path.stem)

        # --- inference from the bounded sample (same helpers as full path) ---
        inferred: dict[str, tuple[str, str | None]] = {}
        sample_values: dict[str, list[Any]] = {}
        sample_medians: dict[str, int | float | None] = {}
        for column_name in sample_df.columns:
            string_series = sample_df[column_name].astype("string")
            trimmed = string_series.str.strip()
            missing_mask = trimmed.str.lower().isin(self.missing_tokens) | trimmed.isna()
            non_missing = trimmed[~missing_mask]
            original_non_missing = string_series[~missing_mask]
            inferred[column_name] = self._infer_logical_type(non_missing.tolist())
            sample_values[column_name] = [
                self._json_scalar(value)
                for value in non_missing.drop_duplicates().head(self.sample_values).tolist()
            ]
            logical, _ = inferred[column_name]
            if logical in {"INTEGER", "DECIMAL"}:
                numeric = _numeric_clean(non_missing)
                sample_medians[column_name] = self._json_number(
                    numeric.median(), rounded=True)
            _ = original_non_missing  # whitespace counts come from the full pass below

        # --- one chunked pass: exact counts + aggregates, capped sets ---
        null_counts: dict[str, int] = {c: 0 for c in sample_df.columns}
        ws_counts: dict[str, int] = {c: 0 for c in sample_df.columns}
        num_acc: dict[str, list[float]] = {}  # col -> [n, sum, min, max]
        date_cands: dict[str, list[str]] = {}  # col -> chunk min/max candidates
        distinct_sets: dict[str, set] = {c: set() for c in sample_df.columns}
        distinct_capped: list[str] = []
        seen_rows: set[int] = set()
        duplicate_rows = 0
        duplicates_estimated = False

        for chunk in self.iter_chunks(PROFILE_CHUNK_ROWS):
            # counters must stay aligned even if a chunk is empty
            for column_name in sample_df.columns:
                if column_name not in chunk.columns:
                    continue
                string_series = chunk[column_name].astype("string")
                trimmed = string_series.str.strip()
                missing_mask = trimmed.str.lower().isin(self.missing_tokens) | trimmed.isna()
                null_counts[column_name] += int(missing_mask.sum())
                non_missing = trimmed[~missing_mask]
                original_non_missing = string_series[~missing_mask]
                ws_counts[column_name] += int((original_non_missing != non_missing).sum())

                logical = inferred[column_name][0]
                if logical in {"INTEGER", "DECIMAL"}:
                    numeric = _numeric_clean(non_missing).dropna()
                    if len(numeric):
                        acc = num_acc.setdefault(column_name, [0.0, 0.0, 0.0, 0.0])
                        acc[0] += len(numeric)
                        acc[1] += float(numeric.sum())
                        acc[2] = float(numeric.min()) if acc[0] == len(numeric) else min(acc[2], float(numeric.min()))
                        acc[3] = float(numeric.max()) if acc[0] == len(numeric) else max(acc[3], float(numeric.max()))
                elif logical in {"DATE", "DATETIME"}:
                    if len(non_missing):
                        date_cands.setdefault(column_name, []).extend(
                            [str(non_missing.min()), str(non_missing.max())])
                # capped distinct tracking (exact until the cap, flagged after)
                if column_name not in distinct_capped:
                    tracker = distinct_sets[column_name]
                    for value in non_missing.tolist():
                        tracker.add(value)
                        if len(tracker) > MAX_DISTINCT_TRACKED:
                            distinct_capped.append(column_name)
                            tracker.clear()
                            break
            # duplicate detection over raw (unstripped) values, like
            # DataFrame.duplicated() on the full frame. Stable blake2b-64
            # hashes (not builtin hash(), which is salted per-process) keep
            # the working set bounded; the cap flips to a sampled estimate.
            if not duplicates_estimated:
                try:
                    frozen = chunk.fillna("")
                    for row in frozen.itertuples(index=False, name=None):
                        # blake2b-64: same exactness as sha256 for dedup
                        # flagging within one run, ~2x faster per row.
                        h = hashlib.blake2b(
                            "|".join(map(str, row)).encode("utf-8"),
                            digest_size=8,
                        ).hexdigest()
                        if h in seen_rows:
                            duplicate_rows += 1
                        else:
                            seen_rows.add(h)
                            if len(seen_rows) > MAX_DUP_ROWS_TRACKED:
                                duplicates_estimated = True
                                seen_rows.clear()
                                break
                except Exception:
                    duplicates_estimated = True
                    seen_rows.clear()
        # free the dedup working set before building the payload
        seen_rows.clear()
        del seen_rows

        if duplicates_estimated:
            # Bounded fallback: estimate from the inference sample and flag it.
            sample_dupes = int(sample_df.duplicated().sum()) if len(sample_df) else 0
            duplicate_rows = round(sample_dupes / max(1, len(sample_df)) * row_count)
        duplicate_pct = round((duplicate_rows / row_count) * 100, 2) if row_count else 0.0

        columns = []
        for column_name in sample_df.columns:
            logical, note = inferred[column_name]
            null_count = null_counts[column_name]
            non_null_count = row_count - null_count
            if column_name in distinct_capped:
                distinct_count = MAX_DISTINCT_TRACKED + 1  # lower bound, flagged below
                distinct_estimated = True
            else:
                distinct_count = len(distinct_sets[column_name])
                distinct_estimated = False
            null_percentage = round((null_count / row_count) * 100, 2) if row_count else 0.0
            distinct_percentage = (
                round((distinct_count / row_count) * 100, 2) if row_count else 0.0
            )
            is_unique = non_null_count > 0 and distinct_count == non_null_count
            identifier_name = self._is_identifier_like(column_name)
            candidate_key = (
                is_unique and null_count == 0 and identifier_name
                and not distinct_estimated
            )
            statistics = self._sampled_statistics(
                column_name, logical, num_acc, date_cands,
                sample_df[column_name], sample_medians.get(column_name),
            )
            profile = {
                "name": column_name,
                "inferred_type": logical,
                "inference_note": note,
                "nullable": null_count > 0,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "null_percentage": null_percentage,
                "distinct_count": distinct_count,
                "distinct_percentage": distinct_percentage,
                "is_unique": is_unique,
                "candidate_key": candidate_key,
                "whitespace_value_count": ws_counts[column_name],
                "sample_values": sample_values[column_name],
            }
            if distinct_estimated:
                profile["distinct_estimated"] = True
            if statistics is not None:
                profile["statistics"] = statistics
            columns.append(profile)
        distinct_sets.clear()

        composite_keys = self._verify_composite_sampled(columns, row_count)

        warnings = self._build_warnings(columns, row_count, duplicate_rows)

        exact_fields = [
            "row_count", "null_count", "duplicate_rows",
            "numeric_min_max_mean", "date_range",
        ]
        sample_based_fields = ["inferred_type", "sample_values", "median"]
        if distinct_capped:
            sample_based_fields.append("distinct_count(capped columns)")
        if duplicates_estimated:
            sample_based_fields.append("duplicate_rows")
            exact_fields.remove("duplicate_rows")

        return {
            "source_type": "csv",
            "source_name": self.path.name,
            "table_name": table_name,
            "file": self._file_block(),
            "row_count": row_count,
            "column_count": len(sample_df.columns),
            "columns": columns,
            "composite_candidate_keys": composite_keys,
            "quality": {
                "duplicate_rows": duplicate_rows,
                "duplicate_row_percentage": duplicate_pct,
                "warnings": warnings,
            },
            "sampling": {
                "sampled": True,
                "sample_rows": int(len(sample_df)),
                "total_rows": row_count,
                "exact_fields": exact_fields,
                "sample_based_fields": sample_based_fields,
                "distinct_capped_columns": sorted(distinct_capped),
                "duplicates_estimated": duplicates_estimated,
            },
        }

    def _sampled_statistics(
        self,
        column_name: str,
        logical: str,
        num_acc: dict[str, list[float]],
        date_cands: dict[str, list[str]],
        sample_series: pd.Series,
        sample_median: int | float | None,
    ) -> dict[str, Any] | None:
        """Statistics for the sampled path: exact min/max/mean, sample median."""
        if logical in {"INTEGER", "DECIMAL"}:
            acc = num_acc.get(column_name)
            if not acc or not acc[0]:
                return None
            count, total, minimum, maximum = acc
            return {
                "min": self._json_number(minimum),
                "max": self._json_number(maximum),
                "mean": self._json_number(total / count, rounded=True),
                "median": sample_median,
            }
        if logical in {"DATE", "DATETIME"}:
            valid = sorted({
                c for c in date_cands.get(column_name, [])
                if self._is_iso_datetime(c)
            })
            if valid:
                # Render through pandas like the full path so both paths emit
                # identical ISO shapes ('YYYY-MM-DDT00:00:00' for dates).
                try:
                    parsed = pd.to_datetime(valid, errors="coerce",
                                            utc=(logical == "DATETIME"))
                    if len(parsed) and not parsed.isna().all():
                        return {"earliest": parsed.min().isoformat(),
                                "latest": parsed.max().isoformat()}
                except Exception:
                    pass
                return {"earliest": valid[0], "latest": valid[-1]}
            # No verifiable ISO extreme: fall back to the sample aggregate.
            string_series = sample_series.astype("string")
            trimmed = string_series.str.strip()
            missing_mask = trimmed.str.lower().isin(self.missing_tokens) | trimmed.isna()
            return self._column_statistics(trimmed[~missing_mask], logical)
        return None

    def _verify_composite_sampled(
        self, col_profiles: list[dict], row_count: int
    ) -> list:
        """Composite-key check for large files without a full-frame pass.

        Pairs are ranked from the sample profiles with the same generic
        policy as the full path; only the single best-ranked pair is then
        verified exactly with one chunked pass over just those two columns.
        Returns [] when nothing verifies (valid: no key, not an error).
        """
        import itertools as _it

        if row_count is None or row_count <= 1:
            return []
        try:
            eligible = [c for c in (col_profiles or [])
                        if c.get("null_count", 1) == 0
                        and not c.get("distinct_estimated", False)
                        and c.get("distinct_count", 0) > 1
                        and c.get("distinct_count", 0) < row_count]
            if len(eligible) < 2:
                return []
            ranked = sorted(eligible, key=self._composite_rank)
            non_measure = [c for c in ranked
                           if c.get("inferred_type") not in ("DECIMAL", "UNKNOWN")]
            pool = (non_measure or ranked)[:6]
            if len(pool) < 2:
                return []
            names = [c["name"] for c in pool]
            pairs = list(_it.combinations(names, 2))
            top = names[0]
            pairs.sort(key=lambda p: (0 if top in p else 1, p))
            best = list(pairs[0])
            seen: set[int] = set()
            for chunk in self.iter_chunks(PROFILE_CHUNK_ROWS, usecols=best):
                frozen = chunk.fillna("")
                for row in frozen.itertuples(index=False, name=None):
                    h = hash(row)
                    if h in seen:
                        return []
                    seen.add(h)
                    if len(seen) > MAX_DUP_ROWS_TRACKED:
                        return []
            return [best]
        except Exception:
            return []

    def _build_warnings(
        self, columns: list[dict], row_count: int, duplicate_rows: int
    ) -> list[str]:
        warnings: list[str] = []
        if row_count == 0:
            warnings.append("CSV contains headers but no data rows.")
        if duplicate_rows:
            warnings.append(f"{duplicate_rows} duplicate row(s) detected.")
        for column in columns:
            name = column["name"]
            if column["null_percentage"] > 50:
                warnings.append(f"Column '{name}' contains more than 50% missing values.")
            if column["inferred_type"] == "UNKNOWN":
                warnings.append(f"Column '{name}' has no usable values.")
            if column["distinct_count"] == 1 and column["non_null_count"] > 1:
                warnings.append(f"Column '{name}' is constant.")
            if column["whitespace_value_count"]:
                warnings.append(
                    f"Column '{name}' has {column['whitespace_value_count']} value(s) "
                    "with surrounding whitespace."
                )
        return warnings

    def _file_block(self) -> dict[str, Any]:
        assert self._encoding is not None and self._dialect is not None
        return {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size,
            "sha256": self._sha256(),
            "modified_at_utc": datetime.fromtimestamp(
                self.path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "encoding": self._encoding,
            "delimiter": self._dialect.delimiter,
            "quotechar": self._dialect.quotechar,
        }

    def load_dataframe(self) -> pd.DataFrame:
        """
        Return a defensive copy of the parsed DataFrame.

        The file is parsed if extract() has not been called yet. Values are kept
        as strings so identifiers such as 00123 are not silently converted to 123.
        """
        if self._df is None:
            self.extract()
        if self._df is None:
            raise RuntimeError("DataFrame not loaded. Call extract() first.")
        return self._df.copy()

    def save_schema_json(self, output_path: str | Path) -> Path:
        """Write the extracted profile to JSON and return the output path."""
        payload = self.extract()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output

    # ------------------------------------------------------------------
    # Bounded streaming access (spec §1: no full-frame copies for big files)
    # ------------------------------------------------------------------

    def detect(self) -> tuple[str, CsvDialectInfo]:
        """Validate + detect encoding/dialect/header without reading rows."""
        self._validate_file()
        if self._encoding is None:
            self._encoding = self._detect_encoding()
        if self._dialect is None:
            self._dialect = self._detect_dialect(self._encoding)
            self._validate_header(self._encoding, self._dialect)
        assert self._encoding is not None and self._dialect is not None
        return self._encoding, self._dialect

    def count_rows(self) -> int:
        """Exact data-row count via a streaming csv pass (bounded memory).

        Skips blank lines exactly like pandas ``skip_blank_lines`` so the
        count agrees with the parsed frame. Ragged rows are counted here
        but still rejected later by the strict pandas parse (on_bad_lines).
        """
        encoding, dialect = self.detect()
        try:
            with self.path.open("r", encoding=encoding, errors="strict", newline="") as fh:
                reader = csv.reader(
                    fh,
                    delimiter=dialect.delimiter,
                    quotechar=dialect.quotechar,
                    doublequote=dialect.doublequote,
                    escapechar=dialect.escapechar,
                    skipinitialspace=dialect.skipinitialspace,
                )
                next(reader, None)  # header (already validated)
                total = 0
                for row in reader:
                    if not row or all(not str(cell).strip() for cell in row):
                        continue
                    total += 1
        except (StopIteration, csv.Error, UnicodeDecodeError, ValueError) as exc:
            raise CsvValidationError(f"CSV parsing failed: {exc}") from exc
        return total

    def _read_options(self, encoding: str, dialect: CsvDialectInfo) -> dict[str, Any]:
        """Shared pandas read options: raw strings, strict errors."""
        return {
            "sep": dialect.delimiter,
            "quotechar": dialect.quotechar,
            "doublequote": dialect.doublequote,
            "escapechar": dialect.escapechar,
            "skipinitialspace": dialect.skipinitialspace,
            "encoding": encoding,
            "encoding_errors": "strict",
            "dtype": "string",
            "keep_default_na": False,
            "on_bad_lines": "error",
            "low_memory": False,
        }

    def read_sample(self, nrows: int = INFERENCE_SAMPLE_ROWS) -> pd.DataFrame:
        """Read at most ``nrows`` head rows (bounded inference sample)."""
        encoding, dialect = self.detect()
        try:
            df = pd.read_csv(
                self.path,
                nrows=max(1, int(nrows)),
                **self._read_options(encoding, dialect),
            )
        except EmptyDataError as exc:
            raise CsvValidationError("CSV contains no tabular data.") from exc
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            raise CsvValidationError(f"CSV parsing failed: {exc}") from exc
        df.columns = [str(column).strip() for column in df.columns]
        lowered = [column.casefold() for column in df.columns]
        if len(lowered) != len(set(lowered)):
            raise CsvValidationError("CSV column names are not unique after parsing.")
        return df

    def iter_chunks(self, chunksize: int = PROFILE_CHUNK_ROWS,
                    usecols: list[str] | None = None):
        """Yield bounded string-dtype chunks over ALL rows (one streaming pass).

        Parsing stays strict (``on_bad_lines="error"``) so malformed rows
        anywhere in the file still raise CsvValidationError.
        """
        encoding, dialect = self.detect()
        options = self._read_options(encoding, dialect)
        if usecols:
            options["usecols"] = list(usecols)
        try:
            reader = pd.read_csv(
                self.path, chunksize=max(1, int(chunksize)), **options)
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            raise CsvValidationError(f"CSV parsing failed: {exc}") from exc
        try:
            for chunk in reader:
                chunk.columns = [str(column).strip() for column in chunk.columns]
                yield chunk
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            raise CsvValidationError(f"CSV parsing failed: {exc}") from exc

    # ------------------------------------------------------------------
    # File and dialect validation
    # ------------------------------------------------------------------

    def _validate_file(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")
        if not self.path.is_file():
            raise CsvValidationError(f"Expected a file, got: {self.path}")
        if self.path.suffix.lower() != ".csv":
            raise CsvValidationError("Only .csv files are accepted by CsvSchemaExtractor.")

        size = self.path.stat().st_size
        if size == 0:
            raise CsvValidationError("CSV file is empty.")

        if self.max_file_size_mb is not None:
            max_bytes = self.max_file_size_mb * 1024 * 1024
            if size > max_bytes:
                raise CsvValidationError(
                    f"CSV is {size / (1024 * 1024):.1f} MB; "
                    f"configured limit is {self.max_file_size_mb} MB. "
                    "Use chunked ingestion for larger files."
                )

        # CSV is text. A NUL byte is a strong signal that the upload is binary
        # or otherwise not a normal text CSV.
        with self.path.open("rb") as fh:
            if b"\x00" in fh.read(8192):
                # UTF-16 legitimately contains NUL bytes, so defer if BOM exists.
                with self.path.open("rb") as prefix_fh:
                    raw_prefix = prefix_fh.read(4)
                utf16_boms = (b"\xff\xfe", b"\xfe\xff")
                if not raw_prefix.startswith(utf16_boms):
                    raise CsvValidationError("File appears to be binary, not a normal CSV text file.")

    def _detect_encoding(self) -> str:
        # Read only the sample needed for detection. read_bytes() would load the
        # complete file before slicing it, which is wasteful for large uploads.
        with self.path.open("rb") as fh:
            raw = fh.read(65536)

        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return "utf-16"

        for encoding in ("utf-8", "cp1252"):
            try:
                raw.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue

        # latin-1 maps every byte. It is a safe final decoding fallback, not a
        # claim that the source was semantically authored in Latin-1.
        return "latin-1"

    def _sample_text(self, encoding: str, char_limit: int = 65536) -> str:
        with self.path.open("r", encoding=encoding, errors="strict", newline="") as fh:
            return fh.read(char_limit)

    def _detect_dialect(self, encoding: str) -> CsvDialectInfo:
        sample = self._sample_text(encoding)
        if not sample.strip():
            raise CsvValidationError("CSV contains no readable text.")

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=self.CANDIDATE_DELIMITERS)
            delimiter = dialect.delimiter
            quotechar = dialect.quotechar or '"'
            doublequote = bool(dialect.doublequote)
            escapechar = dialect.escapechar
            skipinitialspace = bool(dialect.skipinitialspace)
        except csv.Error:
            # A valid one-column CSV has no delimiter to sniff.
            delimiter = ","
            quotechar = '"'
            doublequote = True
            escapechar = None
            skipinitialspace = False

        # Verify doublequote by checking if lines have inconsistent field counts
        # when parsed with the detected dialect.  The Sniffer often mis-detects
        # doublequote as False for CSVs that contain quoted fields with embedded
        # delimiters.
        if not doublequote:
            try:
                with self.path.open("r", encoding=encoding, errors="strict", newline="") as fh:
                    reader = csv.reader(
                        fh,
                        delimiter=delimiter,
                        quotechar=quotechar,
                        doublequote=True,
                        escapechar=escapechar,
                        skipinitialspace=skipinitialspace,
                    )
                    field_counts = [len(row) for _, row in zip(range(20), reader)]
                if field_counts and len(set(field_counts)) == 1:
                    doublequote = True
            except Exception:
                pass

        return CsvDialectInfo(
            delimiter=delimiter,
            quotechar=quotechar,
            doublequote=doublequote,
            escapechar=escapechar,
            skipinitialspace=skipinitialspace,
        )

    def _validate_header(self, encoding: str, dialect: CsvDialectInfo) -> None:
        try:
            with self.path.open("r", encoding=encoding, errors="strict", newline="") as fh:
                reader = csv.reader(
                    fh,
                    delimiter=dialect.delimiter,
                    quotechar=dialect.quotechar,
                    doublequote=dialect.doublequote,
                    escapechar=dialect.escapechar,
                    skipinitialspace=dialect.skipinitialspace,
                )
                header = next(reader)
        except (StopIteration, csv.Error) as exc:
            raise CsvValidationError("CSV header could not be read.") from exc

        if not header:
            raise CsvValidationError("CSV header is empty.")

        cleaned = [name.strip() for name in header]
        if any(not name for name in cleaned):
            raise CsvValidationError("CSV contains one or more blank column names.")

        from collections import Counter
        lowered = [name.casefold().strip() for name in cleaned]
        duplicates = sorted({name for name, count in Counter(lowered).items() if count > 1})
        if duplicates:
            raise CsvValidationError(
                "Duplicate CSV column names are not allowed: " + ", ".join(duplicates)
            )

    def _read_dataframe(self, encoding: str, dialect: CsvDialectInfo) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path, **self._read_options(encoding, dialect))
        except EmptyDataError as exc:
            raise CsvValidationError("CSV contains no tabular data.") from exc
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            raise CsvValidationError(f"CSV parsing failed: {exc}") from exc

        # Header whitespace is formatting noise; data values are left untouched.
        df.columns = [str(column).strip() for column in df.columns]

        # Re-check because pandas can alter duplicate names during parsing.
        lowered = [column.casefold() for column in df.columns]
        if len(lowered) != len(set(lowered)):
            raise CsvValidationError("CSV column names are not unique after parsing.")

        return df

    # ------------------------------------------------------------------
    # Profiling and type inference
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        series: pd.Series,
        column_name: str,
        row_count: int,
    ) -> dict[str, Any]:
        string_series = series.astype("string")
        trimmed = string_series.str.strip()
        missing_mask = trimmed.str.lower().isin(self.missing_tokens) | trimmed.isna()
        non_missing = trimmed[~missing_mask]
        original_non_missing = string_series[~missing_mask]

        null_count = int(missing_mask.sum())
        distinct_count = int(non_missing.nunique(dropna=True))
        non_null_count = int(len(non_missing))
        whitespace_value_count = int((original_non_missing != non_missing).sum())

        null_percentage = round((null_count / row_count) * 100, 2) if row_count else 0.0
        distinct_percentage = (
            round((distinct_count / row_count) * 100, 2) if row_count else 0.0
        )
        is_unique = non_null_count > 0 and distinct_count == non_null_count
        # Uniqueness alone is not enough: prices and names may be accidentally
        # unique in a small sample. Only identifier-like columns are suggested.
        identifier_name = self._is_identifier_like(column_name)
        candidate_key = is_unique and null_count == 0 and identifier_name

        sample_values = [
            self._json_scalar(value)
            for value in non_missing.drop_duplicates().head(self.sample_values).tolist()
        ]

        inferred_type, inference_note = self._infer_logical_type(non_missing.tolist())
        statistics = self._column_statistics(non_missing, inferred_type)

        profile = {
            "name": column_name,
            "inferred_type": inferred_type,
            "inference_note": inference_note,
            "nullable": null_count > 0,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "distinct_count": distinct_count,
            "distinct_percentage": distinct_percentage,
            "is_unique": is_unique,
            "candidate_key": candidate_key,
            "whitespace_value_count": whitespace_value_count,
            "sample_values": sample_values,
        }
        if statistics is not None:
            profile["statistics"] = statistics
        return profile

    def _column_statistics(
        self,
        values: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        """Return a small, useful summary appropriate for the logical type."""
        if values.empty:
            return None

        if inferred_type in {"INTEGER", "DECIMAL"}:
            numeric = _numeric_clean(values)
            return {
                "min": self._json_number(numeric.min()),
                "max": self._json_number(numeric.max()),
                "mean": self._json_number(numeric.mean(), rounded=True),
                "median": self._json_number(numeric.median(), rounded=True),
            }

        if inferred_type in {"DATE", "DATETIME"}:
            parsed = pd.to_datetime(values, errors="coerce", utc=inferred_type == "DATETIME")
            return {
                "earliest": parsed.min().isoformat(),
                "latest": parsed.max().isoformat(),
            }

        # sample_values already explains string and boolean columns. Repeating
        # those values with counts makes the schema noisy and redundant.
        return None

    @staticmethod
    def _json_number(value: Any, *, rounded: bool = False) -> int | float | None:
        if value is pd.NA or value is None:
            return None
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        if number != number:  # NaN check
            return None
        if number.is_integer():
            return int(number)
        return round(number, 4) if rounded else number

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if value is pd.NA:
            return None
        # Values are strings in this extractor; str() keeps this robust if the
        # DataFrame backend changes later.
        return str(value)

    @staticmethod
    def _is_identifier_like(column_name: str) -> bool:
        """Recognise common identifier suffixes across typical naming styles."""
        separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name.strip())
        normalised = re.sub(r"[^0-9a-zA-Z]+", "_", separated).strip("_").lower()
        return bool(re.search(r"(?:^|_)(?:id|key|code)$", normalised))

    @staticmethod
    def _composite_rank(col_profile: dict) -> tuple:
        """Rank composite-PK candidates generically (lower is better).

        No hardcoded names: identifier-like, then INTEGER, then DATE/DATETIME,
        then STRING, then BOOLEAN, then DECIMAL/UNKNOWN last (continuous
        measures almost never form a key; excluding them avoids accidental
        unique pairs like store+weekly_sales).
        """
        name = str(col_profile.get("name", ""))
        typ = str(col_profile.get("inferred_type", ""))
        if CsvSchemaExtractor._is_identifier_like(name):
            return (0, name)
        order = {"INTEGER": 1, "DATE": 2, "DATETIME": 2, "STRING": 3,
                 "BOOLEAN": 4, "DECIMAL": 5, "UNKNOWN": 6}
        return (order.get(typ, 3), name)

    def _detect_composite_keys(self, df, col_profiles, row_count,
                               max_cols: int = 6, max_pairs: int = 12) -> list:
        """Find minimal 2-column unique combos (generic composite PKs).

        Only fully-populated columns participate (PKs are NOT NULL); DECIMAL
        measures are skipped when dimension-like columns exist. Bounded
        (<= max_pairs pandas dup checks) so wide tables stay cheap. Returns
        a list of [col_a, col_b] pairs; [] is valid (no key, not an error).
        """
        import itertools as _it

        if df is None or row_count is None or row_count <= 1:
            return []
        try:
            eligible = [c for c in (col_profiles or [])
                        if c.get("null_count", 1) == 0
                        and c.get("distinct_count", 0) > 1
                        and c.get("distinct_count", 0) < row_count]
            if len(eligible) < 2:
                return []
            ranked = sorted(eligible, key=self._composite_rank)
            non_measure = [c for c in ranked
                           if c.get("inferred_type") not in ("DECIMAL", "UNKNOWN")]
            pool = (non_measure or ranked)[:max_cols]
            if len(pool) < 2:
                return []
            names = [c["name"] for c in pool]
            pairs = list(_it.combinations(names, 2))
            top = names[0]
            pairs.sort(key=lambda p: (0 if top in p else 1, p))
            found = []
            for pair in pairs[:max_pairs]:
                try:
                    if int(df.duplicated(subset=list(pair)).sum()) == 0:
                        found.append(list(pair))
                        break  # first (best-ranked) minimal key wins
                except Exception:
                    continue
            return found
        except Exception:
            return []

    def _infer_logical_type(self, values: list[Any]) -> tuple[str, str | None]:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            return "UNKNOWN", "No non-missing values are available for inference."

        lowered = {value.casefold() for value in cleaned}
        # Canonical boolean policy (shared with csv_orchestrator._TRUE/_FALSE_TOKENS):
        # true/yes/y/1 and false/no/n/0. Requires BOTH sides present so a
        # single-token column ({"true"} / {"1"}) does not misfire: {"1"} alone
        # stays INTEGER, {"true"} alone stays STRING. Keeps handler/orchestrator consistent.
        bool_tokens = {"true", "false", "yes", "no", "y", "n", "1", "0"}
        _true_side = {"true", "yes", "y", "1"}
        _false_side = {"false", "no", "n", "0"}
        if lowered.issubset(bool_tokens) and (lowered & _true_side) and (lowered & _false_side):
            return "BOOLEAN", None

        # Try numeric inference on currency-cleaned values.
        numeric_re = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
        cleaned_numeric = [_clean_currency(v) for v in cleaned]
        cleaned_for_test = [v for v in cleaned_numeric if v and numeric_re.fullmatch(v)]
        if len(cleaned_for_test) >= len(cleaned) * 0.8 and cleaned_for_test:
            integer_re = re.compile(r"^[+-]?\d+$")
            if all(integer_re.fullmatch(v) for v in cleaned_for_test):
                unsigned = [v.lstrip("+-") for v in cleaned_for_test]
                if any(len(v) > 1 and v.startswith("0") for v in unsigned):
                    return (
                        "STRING",
                        "Numeric-looking values contain leading zeros; kept as STRING to preserve identifiers.",
                    )
                return "INTEGER", None
            decimal_re = re.compile(r"^[+-]?(?:\d+\.\d+|\d+\.|\.\d+|\d+[eE][+-]?\d+|\d+\.\d+[eE][+-]?\d+)$")
            if any(decimal_re.fullmatch(v) for v in cleaned_for_test):
                return "DECIMAL", None
            # If most cleaned values parse as numeric, treat as DECIMAL
            if len(cleaned_for_test) >= len(cleaned) * 0.9:
                return "DECIMAL", None

        # Fall back to pure-numeric check (no currency cleaning)
        integer_re = re.compile(r"^[+-]?\d+$")
        if all(integer_re.fullmatch(value) for value in cleaned):
            unsigned = [value.lstrip("+-") for value in cleaned]
            if any(len(value) > 1 and value.startswith("0") for value in unsigned):
                return (
                    "STRING",
                    "Numeric-looking values contain leading zeros; kept as STRING to preserve identifiers.",
                )
            return "INTEGER", None

        decimal_re = re.compile(r"^[+-]?(?:\d+\.\d+|\d+\.|\.\d+|\d+[eE][+-]?\d+|\d+\.\d+[eE][+-]?\d+)$")
        numeric_re2 = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
        if all(numeric_re2.fullmatch(value) for value in cleaned) and any(
            decimal_re.fullmatch(value) for value in cleaned
        ):
            return "DECIMAL", None

        # Date inference is intentionally conservative. Only ISO-like values
        # are auto-classified so 03/04/2026 is not guessed as DD/MM vs MM/DD.
        if all(self._is_iso_datetime(value) for value in cleaned):
            if any("T" in value or " " in value for value in cleaned):
                return "DATETIME", None
            return "DATE", None

        return "STRING", None

    @staticmethod
    def _is_iso_datetime(value: str) -> bool:
        candidate = value.strip()
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?",
            candidate,
        ):
            return False

        try:
            datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_table_name(name: str) -> str:
        normalised = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
        if not normalised:
            return "csv_table"
        if normalised[0].isdigit():
            normalised = f"t_{normalised}"
        return normalised

    def _sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class CsvFolderSchemaExtractor:
    """Extract all CSV files in one folder as independent tabular sources."""

    def __init__(self, folder_path: str | Path, **csv_options: Any) -> None:
        self.folder = Path(folder_path)
        self.csv_options = csv_options

    def extract(self) -> dict[str, Any]:
        if not self.folder.exists() or not self.folder.is_dir():
            raise CsvValidationError(f"CSV folder not found: {self.folder}")

        files = sorted(
            path for path in self.folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        )
        if not files:
            raise CsvValidationError(f"No .csv files found in: {self.folder}")

        tables = [CsvSchemaExtractor(path, **self.csv_options).extract() for path in files]
        return {
            "source_type": "csv_collection",
            "folder": str(self.folder),
            "table_count": len(tables),
            "tables": tables,
        }
