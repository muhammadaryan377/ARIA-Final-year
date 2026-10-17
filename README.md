# ARIA — AI Business Analytics & Intelligence Platform

ARIA is a unified AI analytics system: connect **PostgreSQL / MySQL / CSV / PDF**,
ask natural-language business questions, and get validated answers with evidence,
provenance, visualizations, insights, forecasts, prescriptions and statistical
hypothesis tests. One semantic layer, many product views
(Dashboard, AI Analytics, Insights, Predictive, Prescription, Hypothesis, Data,
Copilot, History, Settings). RL-ready: deterministic truth layer with a
separate ranking/personalization layer (`core/rl.py`).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in DB_* + API keys (see Secrets below)
python app.py                 # serves API + built frontend on 127.0.0.1:8000
cd aria-frontend && npm install && npm run dev   # optional: frontend dev server
```

## Release / dev setup (one reproducible command)

```bash
pip install -r requirements.txt -r requirements-dev.txt
cd aria-frontend && npm ci && npm run build && cd ..
python app.py                 # always serves the freshly built frontend
python -m pytest tests/ -q    # backend suite must stay green
```

## Architecture

| Layer | Location | Role |
|---|---|---|
| API | `app.py`, `api/` (19 routers) | FastAPI routes, auth, CORS, static serving |
| Auth | `auth.py`, `api/auth_routes.py` | PBKDF2 passwords, HS256 JWT, per-user history |
| Shared | `core/` | config+DPAPI secrets, session store, guards, semantic helpers (`semantic.py`), RL (`rl.py`) |
| NL→SQL | `goal_agent/` | intent → semantic grounding → SQL contract → execution + validation |
| Schema intel | `schema_agent/` + `schema_engine/` | introspection, PK/FK inference, drift, snapshots |
| Insights | `insight_agent/` (+ root legacy) | trends/anomalies/correlations + chart catalog |
| Prediction | `prediction/` | walk-forward validated forecasting (gated: refuses weak data) |
| Prescription | `prescription_agent.py` | evidence-based actions (never invents financial impact) |
| Hypothesis | `api/hypothesis_routes.py` | real Pearson tests, PG/MySQL/SQLite |
| Documents | `document_processor/` | PDF text/table/OCR → chunks → embeddings (faiss) → RAG |
| CSV | `csv_handler.py`, `csv_orchestrator.py` | validated ingestion → per-upload SQLite |
| Frontend | `aria-frontend/src/figma/` | React 19 + Vite; hash-persisted pages (`#/<page>`); no mock data path (offline shows empty state) |
| Tests | `tests/` (pytest) | unit + contract tests; run `python -m pytest tests/ -q` |

## Secrets

API keys in `.env` are stored **DPAPI-encrypted** (`enc1:<base64>`, Windows-only) and
decrypted transparently at startup (`core/config.py`). To encrypt a new key:

```bash
python -c "import sys; sys.path.insert(0,'.'); from core.secret_box import encrypt; print(encrypt('<KEY>'))"
```

Never commit `.env`, `data/.jwt_secret`, `*.db`, or token files (all gitignored).
`TOKEN_TTL_DAYS` (default 1, clamped to 1-7) and login rate limits live in `auth.py` /
`api/auth_routes.py`. Relational `db_uri` is never written to disk (rebuilt from
encrypted `db` config on restart). Rotate provider keys in their dashboards if ever exposed.

## Key API contracts

- `POST /api/auth/register|login` → `{ok, token, username}` (30 req / 5 min / IP)
- `POST /api/connect_file` (multipart CSV) → `{ok, table, rows, columns, quality}`
- `GET /api/schema`, `POST /api/data/preview` (`{table, limit, offset, search, sort_*}`)
- `POST /api/data/schema/override` — user PK/FK corrections outrank inference
- `POST /api/ask` → answer + `provenance {sql, tables, methodology, confidence, timestamp}`
- `POST /api/predictive/forecast`, `POST /api/prescription/generate`, `POST /api/hypothesis`
- `POST /api/rl/event`, `GET /api/rl/events` — RL telemetry (summaries only, never raw datasets)

## Honesty rules (enforced)

No mock data in production responses; insufficient data → clear `detail`
(`needs_clarification` contract, `valid: false` forecasts, `ok: false` hypothesis);
every prescription carries "Impact could not be reliably estimated…";
document content is data, never instructions.

## Storage model / scaling (single-worker vs Redis)

Single-worker (default) uses local stores and needs no extra config:
file `data/session_store.json` for sessions, in-memory sliding windows for
rate limits (`core/ratelimit.py`) and the login limiter
(`api/auth_routes.py`, 30 req / 5 min / IP, pruned + capped at 10k IPs).

Multi-worker production requires a shared store:

```bash
ARIA_REDIS_URL=redis://localhost:6379/0
ARIA_REQUIRE_SHARED_STORE=1
UVICORN_WORKERS=4   # or WEB_CONCURRENCY / ARIA_WORKERS
```

Without `ARIA_REDIS_URL`, >1 workers diverge per-process. Startup then
fails fast (`RuntimeError`) and requests fail closed (`503`) when
`ARIA_REQUIRE_SHARED_STORE=1`, or when `ARIA_ENV=production` without an
explicit `ARIA_REQUIRE_SHARED_STORE=0` opt-out. Single-worker runs never
fail this check.
