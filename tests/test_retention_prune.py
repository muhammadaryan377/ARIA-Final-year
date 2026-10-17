"""Retention-prune verification (see RETENTION.md).

`core.config.prune_regenerable_outputs` bounds each per-user
`artifacts/processed_data/user_{id}/` cache to the newest 20 regenerable
outputs (mtime-based, oldest-first). These tests prove:

  - overflow (25 files) keeps exactly the newest 20,
  - active files (processed_data*.json, insights*.json, uploads, docs)
    are never touched,
  - per-user dirs are independent quotas (pruning user A never evicts B),
  - missing dir returns 0 and never raises.
"""

import os
import time
from pathlib import Path

from core.config import (
    REGENERABLE_KEEP_PER_USER,
    prune_regenerable_outputs,
)


def _seed(directory: Path, n: int, prefix: str = "forecast_input_") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(n):
        p = directory / f"{prefix}{i:03d}.json"
        p.write_text('{"ok": true}', encoding="utf-8")
        # Distinct mtimes, oldest first (index 0 = oldest).
        stamp = time.time() - (n - i) * 10
        os.utime(p, (stamp, stamp))
        made.append(p)
    return made


def test_keep_default_is_20():
    assert REGENERABLE_KEEP_PER_USER == 20


def test_overflow_keeps_newest_20(tmp_path):
    d = tmp_path / "user_7"
    made = _seed(d, 25)
    removed = prune_regenerable_outputs(d)
    assert removed == 5
    remaining = sorted(p.name for p in d.iterdir())
    assert len(remaining) == 20
    # Oldest five evicted, newest twenty kept.
    assert {p.name for p in made[:5]}.isdisjoint(remaining)
    assert {p.name for p in made[5:]} == set(remaining)


def test_under_limit_removes_nothing(tmp_path):
    d = tmp_path / "user_7"
    _seed(d, 10)
    assert prune_regenerable_outputs(d) == 0
    assert len(list(d.iterdir())) == 10


def test_active_files_never_touched(tmp_path):
    d = tmp_path / "user_7"
    _seed(d, 25)
    active = [
        "processed_data.json",
        "processed_data_latest.json",
        "insights.json",
        "insights_final.json",
        "upload.csv",
        "report.pdf",
    ]
    for name in active:
        (d / name).write_text("user data", encoding="utf-8")
    prune_regenerable_outputs(d)
    for name in active:
        assert (d / name).exists(), f"active file evicted: {name}"


def test_per_user_quotas_independent(tmp_path):
    # Heavy user A overflows; quiet user B must be unaffected.
    a = tmp_path / "user_1"
    b = tmp_path / "user_2"
    _seed(a, 30)
    b_files = _seed(b, 5)
    assert prune_regenerable_outputs(a) == 10
    assert prune_regenerable_outputs(b) == 0
    assert len(list(a.iterdir())) == 20
    assert {p.name for p in b.iterdir()} == {p.name for p in b_files}


def test_missing_dir_returns_zero(tmp_path):
    assert prune_regenerable_outputs(tmp_path / "no_such_user") == 0
