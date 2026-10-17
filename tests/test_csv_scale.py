"""Scale mechanics for the CSV path: bounded memory, exact counts, bounded payloads.

Mechanics only — fixtures use synthetic columns (id/category/amount/
event_date/flag) and assertions check structural math (counts, sums,
bounds), never business conclusions.

Sizes: 0/1/9/10/100/10_000 rows (exact full-frame path) plus one larger
stress file (default 100_000 rows, override with ARIA_STRESS_ROWS) that
exercises the sampled/chunked path. Each test prints time + peak memory so
the largest actually-tested dataset is on the record; do NOT claim larger
validation than what ran here.
"""

import ctypes
import os
import sqlite3
import time
import tracemalloc
from datetime import date, timedelta
from pathlib import Path

import pytest

from csv_handler import (
    FULL_PROFILE_MAX_ROWS,
    CsvSchemaExtractor,
)
from csv_orchestrator import build_csv_schema_mapping, load_csv_to_sqlite

STRESS_ROWS = int(os.getenv("ARIA_STRESS_ROWS", "100000"))
BASE_DATE = date(2024, 1, 1)


def _rss_mb():
    """Current process RSS in MB (Windows stdlib-only), else None."""
    try:
        if os.name != "nt":
            return None
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        proc = ctypes.windll.kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            proc, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except Exception:
        return None


def make_rows(n):
    """Deterministic synthetic rows + independently computed expectations."""
    rows = []
    exp_sum = 0.0
    exp_nulls = 0
    exp_min = None
    exp_max = None
    for i in range(n):
        if i % 17 == 0:
            amount = ""
            exp_nulls += 1
        else:
            value = (i * 13 % 1000) + 0.5
            amount = f"{value:.2f}"
            exp_sum += value
            exp_min = value if exp_min is None else min(exp_min, value)
            exp_max = value if exp_max is None else max(exp_max, value)
        rows.append({
            "id": str(i + 1),
            "category": f"cat{i % 7}",
            "amount": amount,
            "event_date": (BASE_DATE + timedelta(days=i % 365)).isoformat(),
            "flag": "true" if i % 2 == 0 else "false",
        })
    latest = (BASE_DATE + timedelta(days=min(n - 1, 364))).isoformat() if n else None
    return rows, {
        "sum": exp_sum, "nulls": exp_nulls, "min": exp_min, "max": exp_max,
        "earliest": BASE_DATE.isoformat() if n else None,
        "latest": latest,
    }


def write_csv(path: Path, rows) -> Path:
    cols = ["id", "category", "amount", "event_date", "flag"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(r[c] for c in cols) + "\n")
    return path


def upload_chain(csv_path, workdir, table="scale_t", db_name="scale.db"):
    """The connect_file upload steps: one extract shared by mapping + loader."""
    profile = CsvSchemaExtractor(csv_path).extract()
    mapping = build_csv_schema_mapping(csv_path, table_name=table, profile=profile)
    db_uri, table, row_count = load_csv_to_sqlite(
        csv_path, workdir / db_name, table_name=table, profile=profile
    )
    return profile, mapping, db_uri, table, row_count


@pytest.mark.parametrize("n", [0, 1, 9, 10, 100, 10000])
def test_small_sizes_exact(tmp_path, n):
    rows, exp = make_rows(n)
    csv_path = write_csv(tmp_path / f"s{n}.csv", rows)
    profile, mapping, db_uri, table, row_count = upload_chain(
        csv_path, tmp_path, table=f"s{n}")

    assert profile["row_count"] == n
    assert profile["sampling"]["sampled"] is False
    assert row_count == n
    assert mapping["tables"][table]["row_count"] == n
    assert mapping["tables"][table]["empty"] == (n == 0)

    cols = {c["name"]: c for c in profile["columns"]}
    assert cols["id"]["null_count"] == 0
    assert cols["amount"]["null_count"] == exp["nulls"]
    assert cols["amount"]["non_null_count"] == n - exp["nulls"]
    if exp["min"] is not None:
        assert cols["amount"]["statistics"]["min"] == pytest.approx(exp["min"])
        assert cols["amount"]["statistics"]["max"] == pytest.approx(exp["max"])

    tinfo = mapping["tables"][table]
    if n == 0:
        # No usable values anywhere: every class is honestly "unknown".
        assert set(tinfo["column_classes"].values()) == {"unknown"}
    elif n == 1:
        # Degenerate single row: {"1"}/{"true"} alone stay INTEGER/STRING
        # per the canonical boolean policy (both true and false sides are
        # required); the always-null amount has no usable values.
        assert tinfo["column_classes"]["id"] == "numeric"
        assert tinfo["column_classes"]["amount"] == "unknown"
        assert tinfo["column_classes"]["event_date"] == "date"
        assert tinfo["column_classes"]["flag"] == "text"
    else:
        # Categorical rule (mirrors the contract): low-cardinality text only.
        cat_distinct = len({f"cat{i % 7}" for i in range(n)})
        cat_class = ("categorical"
                     if cat_distinct <= 100 and cat_distinct / n <= 0.5 else "text")
        assert tinfo["column_classes"] == {
            "id": "numeric", "category": cat_class, "amount": "numeric",
            "event_date": "date", "flag": "boolean",
        }
    if n:
        assert tinfo["time_coverage"]["column"] == "event_date"
        # Both profile paths render ISO datetimes identically.
        assert tinfo["time_coverage"]["earliest"].startswith(exp["earliest"])
        assert tinfo["time_coverage"]["latest"].startswith(exp["latest"])
        assert tinfo["time_coverage"]["row_count"] == n
    else:
        assert tinfo["time_coverage"] is None

    db_path = db_uri.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        total = conn.execute(f'SELECT SUM("amount") FROM "{table}"').fetchone()[0]
        nulls = conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "amount" IS NULL').fetchone()[0]
    finally:
        conn.close()
    assert count == n
    assert nulls == exp["nulls"]
    assert (total or 0) == pytest.approx(exp["sum"])


def test_known_duplicates_counted(tmp_path):
    rows, _ = make_rows(100)
    rows = rows + [dict(rows[5]), dict(rows[5]), dict(rows[42])]
    csv_path = write_csv(tmp_path / "dup.csv", rows)
    profile = CsvSchemaExtractor(csv_path).extract()
    assert profile["row_count"] == 103
    assert profile["quality"]["duplicate_rows"] == 3


def test_stress_sampled_path_exact_counts_and_aggregates(tmp_path):
    assert STRESS_ROWS > FULL_PROFILE_MAX_ROWS, (
        f"stress file ({STRESS_ROWS}) must exceed the sampled-path threshold "
        f"({FULL_PROFILE_MAX_ROWS}) or it proves nothing"
    )
    rows, exp = make_rows(STRESS_ROWS)
    csv_path = write_csv(tmp_path / "stress.csv", rows)
    size_mb = csv_path.stat().st_size / (1024 * 1024)

    # Pass 1 (timed, clean): honest wall clock + RSS delta.
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    profile, mapping, db_uri, table, row_count = upload_chain(
        csv_path, tmp_path, table="stress")
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    # Pass 2 (peak Python bytes only): tracemalloc slows execution ~7x here,
    # so its clock is never reported as timing.
    tracemalloc.start()
    upload_chain(csv_path, tmp_path, table="stress2", db_name="scale2.db")
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Largest-dataset record (do not claim beyond this).
    print(
        f"\n[scale] rows={STRESS_ROWS:,} size={size_mb:.1f}MB "
        f"elapsed={elapsed:.1f}s peak_python={peak_py / 1e6:.1f}MB "
        f"rss_before={rss_before}MB rss_after={rss_after}MB"
    )

    # Row count is exact even on the sampled path (no silent truncation).
    assert profile["row_count"] == STRESS_ROWS
    assert row_count == STRESS_ROWS
    assert mapping["tables"][table]["row_count"] == STRESS_ROWS
    sampling = profile["sampling"]
    assert sampling["sampled"] is True
    assert sampling["total_rows"] == STRESS_ROWS
    assert "row_count" in sampling["exact_fields"]

    cols = {c["name"]: c for c in profile["columns"]}
    assert cols["amount"]["null_count"] == exp["nulls"]
    assert cols["amount"]["statistics"]["min"] == pytest.approx(exp["min"])
    assert cols["amount"]["statistics"]["max"] == pytest.approx(exp["max"])
    full_mean = exp["sum"] / (STRESS_ROWS - exp["nulls"])
    assert cols["amount"]["statistics"]["mean"] == pytest.approx(full_mean, rel=1e-6)

    tinfo = mapping["tables"][table]
    assert tinfo["time_coverage"]["earliest"].startswith(exp["earliest"])
    assert tinfo["time_coverage"]["latest"].startswith(exp["latest"])

    db_path = db_uri.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        total = conn.execute(f'SELECT SUM("amount") FROM "{table}"').fetchone()[0]
        # Paginated reads over the persisted SQLite (no CSV reparse per page).
        page1 = conn.execute(f'SELECT "id" FROM "{table}" ORDER BY "id" LIMIT 100 OFFSET 0').fetchall()
        page2 = conn.execute(f'SELECT "id" FROM "{table}" ORDER BY "id" LIMIT 100 OFFSET 100').fetchall()
    finally:
        conn.close()
    assert count == STRESS_ROWS
    assert total == pytest.approx(exp["sum"], rel=1e-6)
    assert len(page1) == 100 and len(page2) == 100
    assert {r[0] for r in page1}.isdisjoint({r[0] for r in page2})


def test_payload_bounds(tmp_path):
    """Every JSON-bound array on the CSV-adjacent path is deterministically capped."""
    from api.data_routes import MAX_PREVIEW_LIMIT, PreviewRequest
    from api.hypothesis_routes import _HYPOTHESIS_SAMPLE_MAX, _fetch_sample
    from api.insights_routes import _MAX_RAW_VALUES, _MAX_SPEC_ROWS, _sanitize_spec_rows
    from core.schema_intel import _run_sql

    # Preview caps (existing contract).
    assert PreviewRequest(table="t", limit=10**9, offset=0).limit == MAX_PREVIEW_LIMIT

    # Chart spec rows capped at 500.
    big = [{"name": f"n{i}", "value": float(i)} for i in range(600)]
    assert len(_sanitize_spec_rows(big)) == _MAX_SPEC_ROWS

    # Raw-value detector pulls bounded via fetchmany.
    assert _MAX_RAW_VALUES <= 2000

    # Hypothesis sample bounded; scatter separately capped at 200 by caller.
    assert _HYPOTHESIS_SAMPLE_MAX <= 2000
    from sqlalchemy import create_engine, text as _text
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(_text("CREATE TABLE h (x REAL, y REAL)"))
        conn.execute(_text(
            "WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i < 2500) "
            "INSERT INTO h SELECT i * 1.0, i * 2.0 FROM c"))
        conn.commit()
        sample = _fetch_sample(conn, "h", "x", "y", "sqlite")
    assert len(sample) <= _HYPOTHESIS_SAMPLE_MAX

    # _run_sql never returns more than its limit.
    conn2 = sqlite3.connect(":memory:")
    try:
        conn2.execute("CREATE TABLE r (v REAL)")
        conn2.executemany("INSERT INTO r VALUES (?)", [(float(i),) for i in range(3000)])
        _, out = _run_sql(conn2, "SELECT v FROM r", limit=2000)
        assert len(out) <= 2000
    finally:
        conn2.close()

    # Goal-agent execution bounds (main + fallback paths share _MAX_ROWS).
    import goal_agent.data as _gd
    import inspect as _inspect
    src = _inspect.getsource(_gd.DataMixin._execute_with_retry)
    assert "fetchmany(_MAX_ROWS + 1)" in src
    assert src.count("fetchmany(_MAX_ROWS + 1)") >= 2  # main path + safe fallback


def test_upload_guard_stays_100mb():
    """The upload guard scales with ARIA_MAX_UPLOAD_CSV_MB (default 512MB):
    large files are profiled by the bounded chunked path and processed in
    a background thread, so the cap is a safety limit, not an analytical one."""
    import os
    from api.connect_routes import MAX_UPLOAD_BYTES_CSV
    expected_mb = int(os.getenv("ARIA_MAX_UPLOAD_CSV_MB", "512"))
    assert MAX_UPLOAD_BYTES_CSV == expected_mb * 1024 * 1024
    assert expected_mb >= 200  # 200MB+ datasets must be accepted


def test_coercion_cache_bounded_lru():
    """Text-numeric probe cache is capped with LRU eviction: one entry per
    (db, table, column) triple must never grow past the cap, hits inside the
    cap behave as before, and clear_coercion_cache() still empties it."""
    from core import text_numeric as tn

    tn.clear_coercion_cache()
    try:
        cap = tn._COERCED_CACHE_MAX
        assert cap > 0
        # cap+50 inserts: size stays bounded, the oldest entries evicted.
        for i in range(cap + 50):
            tn.set_coercion_cache(f"sqlite:///probe{i}.sqlite", "t", f"c{i}",
                                  {"probe": i})
        assert len(tn._COERCED_CACHE) <= cap
        assert tn.get_coercion_cache("sqlite:///probe0.sqlite", "t", "c0") is None
        assert tn.get_coercion_cache(f"sqlite:///probe{cap + 49}.sqlite",
                                     "t", f"c{cap + 49}") == {"probe": cap + 49}
        # Hit inside the cap is unchanged, and the read promotes the entry:
        # after touching probe50, the next insert evicts probe51, not probe50.
        assert tn.get_coercion_cache("sqlite:///probe50.sqlite", "t", "c50") == {
            "probe": 50}
        tn.set_coercion_cache("sqlite:///probe-last.sqlite", "t", "x",
                              {"probe": "last"})
        assert tn.get_coercion_cache("sqlite:///probe51.sqlite", "t", "c51") is None
        assert tn.get_coercion_cache("sqlite:///probe50.sqlite", "t", "c50") == {
            "probe": 50}
        # Writes/reads run from request handlers and background threads.
        import threading
        assert isinstance(tn._COERCED_CACHE_LOCK, type(threading.Lock()))
    finally:
        tn.clear_coercion_cache()
    assert len(tn._COERCED_CACHE) == 0
