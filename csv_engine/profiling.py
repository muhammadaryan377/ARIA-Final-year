import copy
import math
import sqlite3
import time
from pathlib import Path
from .storage import _connection, _quote, _text, CsvAnalysisError
from .constants import VERSION

def _fingerprint(session):
    path = Path(session["db_uri"][len("sqlite:///"):]).resolve()
    stat = path.stat()
    return [VERSION, str(path), stat.st_size, stat.st_mtime_ns, session.get("db_name")]


def _context(session):
    """Bounded aggregate profiling; no raw row samples are sent to a provider."""
    fingerprint = _fingerprint(session)
    cached = session.get("csv_dataset_profile")
    if cached and cached.get("fingerprint") == fingerprint:
        return copy.deepcopy(cached["context"])
    session.pop("csv_dataset_understanding", None)
    table = _text(session.get("db_name"), 200)
    issues, profiled = [], []
    with _connection(session) as conn:
        deadline = time.monotonic() + 5
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        columns = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        if not columns or len(columns) > 100:
            raise CsvAnalysisError("CSV analysis requires between one and one hundred columns.")
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
        except sqlite3.Error:
            raise CsvAnalysisError("CSV profiling exceeded its time budget. Try a smaller dataset.") from None
        # Profile at most 10k rows, with explicit scope; all analysis queries still use the full CSV.
        scope = f"(SELECT * FROM {_quote(table)} LIMIT 10000)"
        sample_count = min(count, 10000)
        for c in columns:
            item = {"name": c["name"], "type": c["type"]}
            col = _quote(c["name"])
            try:
                row = conn.execute(f"SELECT COUNT(*)-COUNT({col}) AS null_count, "
                    f"COUNT(DISTINCT {col}) AS distinct_count, "
                    f"SUM(CASE WHEN typeof({col}) IN ('integer','real') THEN 1 ELSE 0 END) AS numeric_count, "
                    f"SUM(CASE WHEN typeof({col})='text' AND trim({col})='' THEN 1 ELSE 0 END) AS blank_count "
                    f"FROM {scope}").fetchone()
                item.update({k: row[k] or 0 for k in row.keys()})
                if item["numeric_count"]:
                    vals = conn.execute(f"SELECT MIN({col}),MAX({col}),AVG({col}) FROM {scope} "
                        f"WHERE typeof({col}) IN ('integer','real')").fetchone()
                    item["numeric_range"] = {k: v if v is None or math.isfinite(v) else None
                                             for k, v in zip(("min", "max", "mean"), vals)}
                if item["null_count"] or item["blank_count"]:
                    issues.append(f"{c['name']}: missing or blank values in profiling scope.")
            except sqlite3.Error:
                issues.append("Profiling time limit reached; some column statistics are unavailable.")
                profiled.append(item)
                profiled.extend({"name": x["name"], "type": x["type"]} for x in columns[len(profiled):])
                break
            profiled.append(item)
        duplicates = None
        try:
            unique = conn.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {scope})").fetchone()[0]
            duplicates = sample_count - unique
            if duplicates:
                issues.append("Duplicate full rows exist in profiling scope; do not remove without checking row meaning.")
        except sqlite3.Error:
            issues.append("Duplicate-row check unavailable within profiling budget.")
    context = {"dialect": "sqlite", "table": table, "row_count": count, "columns": profiled,
        "profile": {"scope": "full dataset" if count <= 10000 else "first 10000 rows (not random)",
                    "profiled_rows": sample_count, "duplicate_rows": duplicates,
                    "issues": list(dict.fromkeys(issues)),
                    "note": "Profile statistics describe their stated scope; execute SQL for answer values. Text/date meaning must be checked with targeted SQL. No raw samples included."}}
    session["csv_dataset_profile"] = {"fingerprint": fingerprint, "context": copy.deepcopy(context)}
    return context


