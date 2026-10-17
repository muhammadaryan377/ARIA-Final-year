"""Dev-workspace cleanup for ARIA (safe, report-by-default).

Removes regenerable dev clutter at the repo root:
  - ``*.log`` files (server restart/debug logs),
  - ``__pycache__/`` directories,
  - ``.pytest_cache/`` directories,
  - orphaned ``*.db`` / ``*.sqlite`` files at the repo root (ONLY with
    ``--include-db``; these are stray local artifacts, never user data).

NEVER touches:
  - ``data/`` (real uploads, per-user SQLite DBs, JWT secret),
  - ``artifacts/`` user outputs,
  - ``.env`` / secrets,
  - source packages (``core/``, ``api/``, ``insight_rag/``, ``goal_agent/``...).

Usage (PowerShell):
    python tools/cleanup_dev.py                 # report only, deletes nothing
    python tools/cleanup_dev.py --apply         # delete safe targets
    python tools/cleanup_dev.py --apply --include-db   # also root *.db/*.sqlite

See RETENTION.md ("Dev-workspace cleanup").
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Repo root = parent of this script's directory (tools/cleanup_dev.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are ALWAYS off-limits, no matter what flags are passed.
PROTECTED_DIRS = {
    "data",
    "artifacts",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "aria-frontend",
}

# Exact filenames that are never deleted.
PROTECTED_FILES = {".env", ".env.example"}


def _is_protected(path: Path) -> bool:
    """True if `path` lives under a protected directory or is protected."""
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return True  # outside the repo: never touch
    if rel.parts and rel.parts[0] in PROTECTED_DIRS:
        return True
    if path.name in PROTECTED_FILES:
        return True
    return False


def collect(include_db: bool = False) -> dict[str, list[Path]]:
    """Scan the repo root for safe-to-remove dev clutter (no deletion)."""
    targets: dict[str, list[Path]] = {"logs": [], "pycache": [], "pytest_cache": [], "db": []}

    for log in REPO_ROOT.glob("*.log"):
        if log.is_file() and not _is_protected(log):
            targets["logs"].append(log)

    for cache in REPO_ROOT.rglob("__pycache__"):
        if cache.is_dir() and not _is_protected(cache):
            targets["pycache"].append(cache)

    for cache in (REPO_ROOT / ".pytest_cache",):
        if cache.is_dir() and not _is_protected(cache):
            targets["pytest_cache"].append(cache)

    if include_db:
        for pat in ("*.db", "*.sqlite"):
            for db in REPO_ROOT.glob(pat):  # root only, never data//
                if db.is_file() and not _is_protected(db):
                    targets["db"].append(db)

    for key in targets:
        targets[key].sort()
    return targets


def _remove(path: Path) -> bool:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        print(f"  SKIP {path}: {exc}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe ARIA dev-clutter cleanup (report by default).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete. Without it, only report.")
    parser.add_argument("--include-db", action="store_true",
                        help="Also remove orphaned root *.db/*.sqlite files.")
    args = parser.parse_args(argv)

    # Refuse to run from outside the repo (paranoia: resolve + containment).
    if not (REPO_ROOT / "app.py").exists() or not (REPO_ROOT / "tools").exists():
        print(f"Refusing to run: {REPO_ROOT} does not look like the ARIA repo root.",
              file=sys.stderr)
        return 2

    targets = collect(include_db=args.include_db)
    total = sum(len(v) for v in targets.values())
    if total == 0:
        print("Nothing to clean: no *.log, __pycache__, .pytest_cache"
              + ("" if args.include_db else " (root *.db not scanned without --include-db)") + ".")
        return 0

    print(f"Dev clutter under {REPO_ROOT} (data/ and artifacts/ untouched):")
    for key, paths in targets.items():
        for p in paths:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            print(f"  [{key}] {rel}")

    if not args.apply:
        print(f"\nReport only: {total} target(s) listed, nothing deleted. Re-run with --apply.")
        return 0

    removed = 0
    for paths in targets.values():
        for p in paths:
            if _is_protected(p):  # re-check at delete time
                print(f"  SKIP (protected) {p}", file=sys.stderr)
                continue
            removed += _remove(p)
    print(f"\nRemoved {removed}/{total} target(s).")
    return 0 if removed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
