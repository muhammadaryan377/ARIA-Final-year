# ARIA retention / cleanup policy

Active user data is **never** deleted by background behavior: uploads
(`data/uploads/user_{id}/`), per-user SQLite DBs (`.db`), schemas,
`processed_data*.json`, `insights*.json`, documents, session store, and logs
are kept until the user deletes them or disconnects the source.

Bounded behavior only:

- **Temp/partial uploads on failure/overflow**: `_spool_limited`
  (`api/connect_routes.py`) streams uploads with a size cap and deletes the
  partial file on any error/overflow. CSV SQLite build failures also delete
  the partial `.db`. PDF table-load failures delete the partial
  `*_tables.db` (`_pdf_tables_to_sqlite`). RAG PDF spool temp files are
  deleted in a `finally` block (`api/insight_rag_routes.py`).
- **Regenerable run outputs (bounded cache)**: `forecast_input_*.json` files
  under `artifacts/processed_data/user_{id}/` are regenerable on every
  forecast run. `core.config.prune_regenerable_outputs()` keeps the newest
  20 per user (mtime-based eviction, oldest first). It never touches active
  files (`processed_data*.json`, `insights*.json`, uploads, documents).
- **Session memory**: in-memory session store caps at 100 users with 4h TTL
  (`core/config.py`); persisted sessions survive restarts.
- **Quarantined RAG chunks (injection containment)**: chunks flagged by
  `insight_rag/enterprise_quality.quarantine_score()` are excluded from every
  LLM context string (retrieval, sibling expansion, and context construction
  all filter them; counts are recorded as `quarantined_count` diagnostics and
  surfaced as `quarantined_excluded` on the chat result). Quarantined chunk
  files are retained alongside their document until the user deletes the
  document — exclusion is from context, not silent deletion — so the decision
  stays auditable.

## Prune policy: regenerable run outputs (bounded per-user cache)

`forecast_input_*.json` files under `artifacts/processed_data/user_{id}/` are
regenerated on every forecast run, so each per-user directory is a **bounded
cache capped at the newest 20 files**:

- `core.config.prune_regenerable_outputs(processed_dir, patterns=None, keep=None)`
  evicts oldest-first by **mtime**, keeping `keep` (default
  `REGENERABLE_KEEP_PER_USER = 20`).
- It only matches regenerable patterns (default `forecast_input_*.json`).
  It **never** touches active files (`processed_data*.json`,
  `insights*.json`, uploads, documents, schemas, session store, logs).
- It never raises: missing dir / unlink errors return `0` instead of failing
  the request. Called from `api/predictive_routes.py` before each forecast
  run (best-effort hygiene inside the request path).
- **Per-user quota**: pruning runs inside each `user_{id}/` directory, so
  every user independently keeps at most 20 regenerable outputs — one heavy
  user can never evict another user's files, and total disk use scales as
  `20 files x users`, never unbounded.
- Verified by `tests/test_retention_prune.py`: overflow to 25 files keeps the
  newest 20, active files are untouched, per-user dirs are independent, and a
  missing dir returns `0`.

No quota system beyond this bounded cache, no arbitrary destructive cleanup.

## Dev-workspace cleanup (safe, never touches user data)

Stray dev clutter accumulates at the repo root during local runs: `*.log`
(server restarts), `__pycache__/`, `.pytest_cache/`, and orphaned `*.db` /
`*.sqlite` files outside `data/` (which is gitignored and holds real user
state). Clean it with:

```powershell
python tools/cleanup_dev.py          # report only (default, deletes nothing)
python tools/cleanup_dev.py --apply  # delete the safe targets
python tools/cleanup_dev.py --apply --include-db  # also remove orphaned root *.db/*.sqlite
```

Safety rules (enforced by the script, see `tools/cleanup_dev.py`):

- Removes: `*.log` at repo root, `__pycache__/` dirs, `.pytest_cache/`,
  and — only with `--include-db` — orphaned `*.db`/`*.sqlite` files at the
  repo root.
- **Never** touches `data/` (real uploads, per-user DBs, JWT secret),
  `artifacts/` user outputs, `.env`, or anything under `core/`, `api/`,
  `insight_rag/`, `goal_agent/`.
- Default mode is report-only; deletions require `--apply`. Every candidate
  is printed before removal.
