# ARIA — Session Handoff & Progress

Location: `C:\Users\Go\Desktop\ARIA`
Purpose: This file lets a fresh AI coding session pick up exactly where the last one left off.

---

## Project overview

ARIA = **A**gentic **R**easoning for **I**ntelligent **A**nalytics. A BI testing app with 3 agents:

1. **Schema Agent** (`schema_agent.py`) — extracts table/column/relationship mapping from PostgreSQL via `information_schema`; also infers relationships heuristically. Output: `schema_mapping_<db>_latest.json`.
2. **Goal Agent** (`goal_agent.py`) — natural-language goal → KPI alignment → join path → text-to-SQL → executes → cleans → writes `processed_data_<db>.json`.
3. **Insight Agent** (`insight_agent.py`) — reads processed data → KPIs, trends, anomalies, hypotheses, chart specs, business story → writes `insights_<db>.json`.

Backend: FastAPI entry point `app.py` (port 8000, serves `static/index.html`) wired to a package structure — routes in `api/`, shared config/deps/validation in `core/` (see "Project structure" below).
LLM backends: **local** = Ollama (`sqlcoder:7b` for SQL, `aria-goal` for story/suggest, `qwen3:4b` for semi), **cloud** = Groq (`llama-3.3-70b-versatile`).
DBs available on local Postgres (`postgres`/`12345`): `northwind`, `postgres`, plus `aria_test` (test library schema created for generic testing). Relational sources support **PostgreSQL and MySQL** (MySQL needs a reachable server; none installed locally — tested only up to the connection error path).

Run: `.venv\Scripts\python app.py` → open http://127.0.0.1:8000

---

## Project structure (refactored this session)

`app.py` is now a thin entry point (app factory, CORS, router includes, `/` + `/api/providers`, static mount, port resolution). Business logic lives in packages:

```
app.py                      # entry point: routers + public endpoints + uvicorn main
api/                        # FastAPI routers
  auth_routes.py            # /api/auth/*, /api/history, /api/history/{id}
  connect_routes.py         # /api/connect, /api/connect_file
  schema_routes.py          # /api/extract_schema, /api/schema
  query_routes.py           # /api/suggestions, /api/ask, /api/insight
core/                       # shared helpers (no routers)
  config.py                 # BASE_DIR/paths, PROVIDER_NAMES, SOURCE_TYPES, SESSION state
  deps.py                   # require_provider/require_data/get_current_user/require_writable/is_mobile
  validation.py             # validate_goal_text + _GOAL_STOPWORDS/_GIBSMASH
  db.py                     # build_db_uri, safe_db_name
auth.py                     # user/history store (PBKDF2 + JWT), kept as-is
schema_agent.py, goal_agent.py, insight_agent.py, llm_provider.py   # agents, as-is
prediction/                 # PREDICTIVE ANALYSIS (math models, no LLM for numbers)
  models.py                 # naive/mean/moving-avg/linear-trend/ES/Holt-damped (numpy only)
  evaluate.py               # walk-forward MAPE, model selection, overfit guard, accuracy
  agent.py                  # PredictionAgent.forecast(processed_data, ...) -> structured forecast
prescription_agent.py       # PrescriptionAgent.prescribe(...) -> LLM numbered actions + template fallback
```

### Prediction + Prescription (added 2026-08-15)

- **Prediction** (`prediction/`): `InsightAgent.analyze()` now calls `PredictionAgent.forecast()` — aggregates the primary numeric by the date column (daily/weekly/monthly via median-gap inference, `sequence` fallback), fits math models (naive, mean, moving average, linear trend, exponential smoothing, damped Holt), picks the best with **expanding-window walk-forward validation** and an **overfit guard** (a model is dropped if out-of-sample MAPE is >1.5x its in-sample MAPE). Reports honest `accuracy_pct = 100 - MAPE` (≥90% on clean trends, labelled `excellent/good/fair/poor/insufficient_data`), 80/95% confidence bands, `projected_pct_change`, and `reliability`.
- **Prescription** (`prescription_agent.py`): `PrescriptionAgent.prescribe()` sends KPIs/trends/anomalies/hypotheses/**prediction** to the LLM (local `story` role or Groq) asking for a numbered list of 3-5 actions; falls back to template actions from the numbers if the LLM fails. Output: `{"actions": [...], "source": "llm"|"template"}`.
- Both are added to `insights.json` as `predictions` + `prescriptions`; the business-story prompt now includes the forecast so the story adds a one-sentence outlook.
- Frontend: `renderPredictionsInto` (forecast table + accuracy + 95% band) and `renderPrescriptionsInto` (numbered actions) render in both live insights and history detail. No new deps (numpy/pandas already present).
- Verified end-to-end via `/api/insight` on northwind (connect→schema→ask→insight): predictions sequence/moving_average 82%, LLM prescriptions (5 actions), story includes outlook. Tested model maths directly: clean monthly trend → Holt-damped 99.5%, noisy random walk → naive 88.6% (honest), 4-point series → `insufficient_data`.


Notes: `SESSION` lives in `core/config.py` and is imported by the route modules (shared mutable state). No behavior changed — endpoints, payloads and auth are identical. Verified after the split (see below).

---

## What was broken and fixed (this session)

### 1. Local SQL generation completely broken (root cause)
- `sqlcoder:7b` is a **completion-only** model (no chat template). Old code called it with `chat()`, so it echoed the prompt and silently fell back to `SELECT * FROM <table> LIMIT 50` — never answering real questions.
- **Fix**: added `LLMProvider.complete()` + `_complete_local()` in `llm_provider.py` that use Ollama's `generate()` API for completion-style models (with stop tokens). Goal Agent SQL now uses SQLCoder's `### Task / ### Database Schema / ### SQL` prompt format.
- Verified: sqlcoder produces real, correct SQL via `generate()` (takes ~1–2 min on this CPU — hardware limitation, not a code bug).

### 2. Cloud SQL occasionally got "This query:" explanation appended
- Fixed by hardening `_clean_sql()` in `goal_agent.py` (keep only first statement up to first `;`) + added "Output only the final SQL statement" instruction.

### 3. Copilot-added goal validation was DB-specific
- `_validate_goal_text` in `app.py` had a hardcoded `business_words` list that **rejected valid queries** on any DB that didn't use sales/product/customer words.
- **Fix**: rewrote validation to be DB-agnostic — rejects only empty/no-letters/<2 words/gibberish. Then further hardened to "reject every gibberish": repeated-char, low-unique-char, no-vowel (len≥4), keyboard-smash blacklist `_GIBSMASH`, stopword-only input.
- Removed dead code (`VALID_GOAL_SUGGESTIONS`, `INVALID_GOAL_STOPWORDS`, `_goal_error_response`, `_is_likely_valid_goal`) and unused `JSONResponse` import.

### 4. Interactive charts + chart templates in JSON (user request)
- `static/chart_templates.json` holds chart "code" templates (bar, line, scatter, pie, doughnut) with `shapes` (data-shape rules + priority) and Chart.js `config`.
- `insight_agent.build_dashboard_spec()` loads templates, **picks the best chart type per data shape** (category+numeric→bar, datetime+numeric→line, 2 numerics→scatter, share→pie), embeds config + labels/datasets into each spec.
- Frontend renders interactive charts with Chart.js 4.4.1 (CDN), no LLM at render time → fast.

### 5. CSV + PDF file upload support (done, verified)- `/api/connect_file` (renamed from `/api/connect_csv`) accepts `.csv` OR `.pdf` uploads via multipart (`provider` + `file`).
  - **CSV** → `build_csv_schema()` + `load_csv_to_sqlite()` → file-based SQLite, single table.
  - **PDF** → `schema_agent.extract_pdf_tables()` (pdfplumber) — every detected table on every page becomes its own SQLite table (`<stem>_table_N`); no tables found → 400. Numeric coercion (`_coerce_numeric`) so SUM/AVG work on text-sourced columns.
  - Schema mapping via new generic `build_tables_schema()` / `load_tables_to_sqlite()` (dict of {table: DataFrame}).
- Session switches to `dialect="sqlite"`; Goal Agent prompt then emits SQLite SQL (verified `strftime('%Y-%m', ...)` for month grouping).
- Frontend: one file input, `accept=".csv,text/csv,.pdf,application/pdf"`, button "CSV / PDF". `connectDb` posts FormData to `/api/connect_file`.
- Gibberish validation final: added "you"/"we"/etc. to stopwords — full battery now rejects `asdf asdf`, `qwerty`, `fjdk alksd jf`, `zxcvbnm`, `12345`, `!!!!`, `hi how are you`; still accepts real questions.

### 6. Handwritten / scanned PDF support (done, verified)
- New `auth.py` module + JWT auth (see below) — needed because handwritten extraction only works on the **cloud** provider (Groq vision).
- `schema_agent.pdf_has_text()` detects whether a PDF has a text layer. `/api/connect_file` routing for PDFs:
  - has text + tables  → pdfplumber path (printed/tabular)
  - has text, no tables → reject 400
  - **no text layer** (scanned/handwritten) → `schema_agent.extract_handwritten_pdf()`:
    1. renders every page to PNG via `pypdfium2` + Pillow (already installed),
    2. sends each page image to the Groq **vision** model `qwen/qwen3.6-27b` (JSON mode) asking for `{"tables":[{name, rows}]}`,
    3. parses into DataFrames, groups by identical column sets → one SQLite table per distinct schema,
    4. if no records found → reject 400 (per user choice).
  - Cloud-only: local provider raises "requires the Cloud provider" for scanned PDFs.
- `llm_provider.vision()` added (image parts + `response_format=json_object` for qwen).

### 7. Login + per-user history + mobile read-only (done, verified)
- **Auth**: `auth.py` — users stored in `data/aria.db` (SQLite), passwords PBKDF2-HMAC-SHA256 (stdlib, no bcrypt), HS256 JWTs via PyJWT (7-day expiry, secret in `data/.jwt_secret`).
  - Endpoints: `POST /api/auth/register`, `POST /api/auth/login` (return JWT), `GET /api/auth/me`, `POST /api/auth/logout`.
- **Protection**: all data endpoints require `Authorization: Bearer <token>` (`_get_current_user`, 401 otherwise). `/api/providers` and `/` stay public.
- **History**: every successful `/api/ask` saves a row (goal, sql_used, row_count, columns, source, db, dialect, processed JSON). `/api/insight` attaches the insights JSON to that row via `SESSION["pending_history_id"]`.
  - `GET /api/history` (metadata list, newest first, `mode: read-only|full`), `GET /api/history/{id}` (full record incl. processed data + insights).
- **Mobile read-only**: `_is_mobile()` checks `X-Client-Type` header then User-Agent. Write endpoints (connect, connect_file, extract_schema, suggestions, ask, insight) use `_require_writable` → 403 on mobile. History endpoints stay available (view history only). The static UI also disables the write buttons and shows a banner on mobile.
- **Frontend (`static/index.html`)**: login/register overlay, token in `localStorage`, `Authorization` header on all fetches (incl. FormData upload), user bar + logout, "My History" panel (list + detail with SQL, data table, insights, interactive charts). 401 auto-logs-out.
- **Known limitation**: the in-memory `SESSION` (active source/schema) is still a single global state shared across users — acceptable for now; the friend's pro frontend may want per-user sessions later.

---

## Files changed (all under C:\Users\Go\Desktop\ARIA)

| File | What changed |
|---|---|
| `auth.py` | **NEW** — users + PBKDF2 hashing + JWT issue/verify + per-user history CRUD (`data/aria.db`) |
| `app.py` | **RESTRUCTURED** — now a thin entry point; auth deps, all routes and helpers moved into `api/` + `core/` (see "Project structure" above). Behavior unchanged |
| `core/config.py`, `core/deps.py`, `core/validation.py`, `core/db.py` | **NEW** — shared session state, FastAPI dependencies, goal validation, DB/URI helpers |
| `api/auth_routes.py`, `api/connect_routes.py`, `api/schema_routes.py`, `api/query_routes.py` | **NEW** — routers split by domain, imported by `app.py` |
| `llm_provider.py` | Added `vision()` (Groq image parts, JSON mode for qwen); cloud model `vision: qwen/qwen3.6-27b` |
| `schema_agent.py` | **NEW** `pdf_has_text()`, `extract_handwritten_pdf()` (pypdfium2 render → vision → DataFrames), `_parse_vision_frames()` |
| `app.py` | `Depends/Header/Request`; auth deps `_get_current_user` / `_require_writable` + `_is_mobile`; `/api/auth/*`, `/api/history`, `/api/history/{id}`; all data endpoints protected; scanned-PDF routing in `/api/connect_file`; history recorded in ask + insight |
| `static/index.html` | Login/register overlay + user bar + logout; token in localStorage + auth headers everywhere; "My History" panel (list/detail/charts); mobile read-only detection + banner + disabled write buttons |
| `requirements.txt` | Added `python-multipart`, `pdfplumber`, `PyJWT` (all installed in `.venv`; `fpdf2` installed only for generating test PDFs) |

---

## Verified / tested so far

- Full cloud pipeline on **northwind**: 8 diverse queries (totals, counts, averages, joins, top-N, time-series) → correct SQL + data; Insight → KPIs, bar/pie charts, real story. ✅
- **Generic/DB-agnostic**: created custom library DB `aria_test` (authors/books/loans) → full pipeline worked (4 queries + insight + charts). ✅
- Gibberish validation battery: 7 garbage inputs rejected, real questions accepted. ✅
- Chart selection: bar + pie correctly picked from `chart_templates.json`. ✅
- **CSV end-to-end** (cloud): uploaded 15-row `retail_sales.csv` → schema inferred (5 cols) → 3 queries OK (total revenue by region, top product, revenue by month via `strftime`) → insight with bar+pie charts + story. ✅
- **PDF end-to-end** (cloud): generated 2-table `test_report.pdf` (sales + payroll) → both tables extracted to SQLite (8 + 4 rows) → "total revenue by region" and "average salary per dept" (LLM auto-routed to correct table, handled CAST for salary) → insight with charts. ✅
- **Handwritten/scanned PDF** (cloud): created image-only `scanned_report.pdf` (no text layer) → `pdf_has_text()` correctly False → vision extraction split both tables (8-row sales `..._handwritten_1`, 4-row payroll `..._handwritten_2`) → both queries + insight worked. ✅
- **Auth + history + mobile read-only** (API): register/login/`me` OK; wrong password → 401; no token → 401; wrong password rejected; connect+ask with token OK (history_id recorded); `GET /api/history` returns the goal; mobile UA header → `/api/ask` and `/api/connect` both 403, `GET /api/history` allowed with `mode=read-only`; desktop insight attached insights to the history row; `GET /api/history/1` returned processed data + insights. ✅
- **Post-refactor smoke test** (restructured `api/`+`core/`, port 8001): `py_compile` all modules OK; `/api/providers` 200; login/`me` OK; cloud connect northwind → extract_schema (14 tables) → ask (8 rows, category sales) → insight (KPIs+story) → history listed + detail returned with `insights` attached; CSV upload via `curl.exe -F` → 15 rows, columns correct → monthly revenue via `strftime` (6 rows) → insight OK; mobile UA → ask 403 + history `mode=read-only`; `/` serves index (200). ✅
- Test files: sample data at `C:\Users\Go\AppData\Local\Temp\opencode\` (`retail_sales.csv`, `test_report.pdf`, `scanned_report.pdf`, `make_test_pdf.py`, `make_scanned_pdf.py`, `test_handwritten.py`).

### 8. MySQL support (done — code verified, needs a live server for E2E)
- Relational sources now accept `db_type: "postgresql" | "mysql"` in `/api/connect` (validated against `DB_TYPES` in `core/config.py`).
- `schema_agent.py` is db-type aware: `get_connection` branches to `pymysql` (installed; added to requirements), `_dict_cursor` (RealDictCursor for PG / DictCursor for MySQL), and the metadata queries handle MySQL differences:
  - `get_declared_foreign_keys` uses `key_column_usage.referenced_table_name` for MySQL (the PG `constraint_column_usage` join doesn't work there).
  - `get_null_stats` backtick-quotes identifiers for MySQL (`\`db\`.\`table\``) vs psycopg2 `sql.Identifier` for PG.
  - MySQL has no separate schema: `api/schema_routes.py` filters by the database name (`schema_filter = config["db"]`).
- `/api/connect` probe branches: MySQL `SELECT DATABASE();`, PG `SELECT current_database(), current_schema();`. `SESSION["dialect"]` and `SESSION["db_type"]` = the chosen type.
- `core/db.py build_db_uri(config, db_type)` emits `mysql+pymysql://` (Goal Agent executes via SQLAlchemy engine).
- `goal_agent.py` adds a **MySQL notes** block to the SQL prompt (backtick quoting, `DATE_FORMAT` for date grouping, `NOW()`, no ILIKE, `CONCAT()`, backslash escapes) — verified the note renders in the prompt.
- Frontend: DB type selector (PostgreSQL/MySQL) in the relational form; switches default port 5432↔3306, hides the Schema field for MySQL, and sends `db_type` in the connect body.
- **Tested**: bad `db_type` → 400; MySQL connect to a dead server → clean 400 via pymysql; Postgres northwind regression (connect→schema→ask→insight) still passes; MySQL note present in the Goal Agent prompt.
- **NOT yet E2E-tested**: no MySQL server available locally — needs one to confirm schema extraction + SQL execution against real MySQL.

---

## Planned (user decision — fine-tuning)

- **Goal Agent keeps `sqlcoder:7b`** for SQL query generation (unchanged).
- **Mistral 7B fine-tuned for the Insight Agent** (explanations + anomaly detection + prescriptions), to replace the current local `aria-goal` story model (cloud stays the fast path).

### DONE — adapter trained (2026-08-15)

- **Kaggle notebook** `mariazaidi77/aria-fine-tune-mistral-story` (v11) + **dataset** `mariazaidi77/aria-kaggle-finetune` (v5, 227 rows).
- Data = 122 story + 60 anomaly-detection + 45 prescription examples (the extra two task types were generated by `prepare_aria_story_data.py` and appended).
- **P100 blocker solved**: free Kaggle ignores the T4 accelerator setting (UI/API) and always assigns a P100 (sm_60), which current torch/bnb can't use. Solution: notebook installs `torch==2.3.1+cu118` (supports sm_60) and trains **fp16 LoRA, no bitsandbytes**. Requirements pinned: `transformers==4.44.2, peft==0.12.0, trl==0.10.1, accelerate==0.34.2, datasets==2.21.0`.
- Result: **loss 1.64 → 0.23** over 168 steps (~16s/it, ~47 min). Adapter = `r=16, alpha=32`, 7 target modules, 41.9M trainable params, 449 tensors, base `mistralai/Mistral-7B-Instruct-v0.2`.
- Adapter stored locally at `C:\Users\Go\Desktop\ARIA\models\aria-story-mistral-7b-lora\` (adapter_config.json + adapter_model.safetensors 167.8MB + tokenizer files). Download via signed `kaggleusercontent` URLs (raw API `list_kernel_session_output` → `f.url`); use resumable Range downloads — plain `requests.get` breaks mid-transfer on this network.
- Integration still TODO: merge LoRA into base → GGUF → Ollama (or Ollama `ADAPTER` in Modelfile), then wire into `llm_provider.py` as a local `insight` model. See `USE_TRAINED_ADAPTER.md`.

## Deployment (target: ~2–3 months out — no code changes wanted at deploy time)

- App is already config-driven: `GROQ_API_KEY`, `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD`, `ARIA_PORT`; JWT secret auto-persists to `data/.jwt_secret`.
- Small prep items to do near deploy time: make Ollama URL configurable (`llm_provider.py` hardcodes `localhost:11434` — add `OLLAMA_HOST` env), then ship a `Dockerfile` + `docker-compose.yml` (app + Postgres + Ollama) + `.env.example`.

---

## In-progress / NOT yet done

1. **Browser UI check** — the new login/history/mobile UI was only syntax-checked + served; exercise it in a real browser: register, connect northwind, ask, generate insight, open **My History** and click a row (charts render). Also test the CSV/PDF picker and a mobile viewport (read-only banner + disabled buttons + history-only).
2. **Local path final check** — pipeline now works on local, but sqlcoder is CPU-slow (~1–2 min/query). If too slow for comfort, consider pointing local SQL at a faster model (e.g. `aria-goal`) or pre-warming the model.
3. **Cleanup** — delete `aria_test` database if the user no longer wants it (created for generic testing). Offer first.
4. **Backup** — re-create `C:\Users\Go\Desktop\ARIA_copy.zip` if the user wants the updated code snapshotted.
5. **User's server on port 8000** runs old code — restart it to pick up auth + all fixes.

---

## LLM validation layer + graceful data preprocessing (added 2026-08-15)

### Shared guardrail layer — `core/validation.py` (provider-agnostic)
- Every agent routes raw model output (local Ollama **and** cloud Groq) through the same checks before trusting it: `strip_code_fences` (SQL/JSON fences), `clean_llm_text` (rejects empty/refusal/gibberish), `parse_llm_json`, `sql_table_names`/`unknown_sql_tables` (catches hallucinated tables like `order_items` vs the real `order_details` — deterministically, before a DB round-trip), `empty_sql_tables`.

### Goal Agent hardening
- **SQL validation + repair**: `_execute_with_retry` now checks referenced tables against the schema each attempt; unknown tables trigger a targeted repair prompt (full schema is included in generation + all repair prompts, so the LLM can discover `orders`/`order_details`/etc. instead of inventing names). Never 500s: falls back to a safe generic query, and the route surfaces a clean message.
- **Full-schema DDL**: `_build_schema_ddl(join_path, full=True)` adds every table (FK-connected first) with join-path tables marked "preferred", fixing e.g. `sales by customer demographics` which previously hallucinated `order_items` or returned 0 rows.
- **Empty-table detection**: `_row_counts()` (cached) + `empty_sql_tables()` — a valid query on genuinely empty tables (northwind's `customer_demographics`/`customer_customer_demo` have **0 rows**) now returns `message: "The query is valid but returned no rows because the table(s) ... are empty..."` instead of a bare 0 rows / "no data".
- **Preprocessing**: `_clean_data()` drops fully-empty columns, drops fully-empty rows, fills remaining nulls (0 for numeric, '' for text), coerces types, and stores a `preprocessing` report (`dropped_empty_columns`, `dropped_empty_rows`, `nulls_filled`, `columns_kept`, `notes`). The report + graceful `message` are returned by `/api/ask` and rendered in the UI (`ask_meta`).
- Verified: `sales by customer demographics` → 200 + clear empty-table message; `unknown_sql_tables` catches `order_items`; `_clean_data` on synthetic nulls drops 2 empty cols + 1 empty row and fills 1 null.

### Prediction validation (earlier)
- Every forecast re-runs expanding-window walk-forward backtesting; forecasts below the 70% accuracy threshold (or unmeasurable) are flagged `valid:false` and excluded from story/prescriptions/UI (red "Not shown" notice). Confidence bands use out-of-sample walk-forward error (not optimistic in-sample std).

---

## Notes / gotchas for the next session

- `.env` holds `DB_*` (northwind, postgres/12345) and a `GROQ_API_KEY` — don't print the key.
- No git repo in the folder; there is a zip backup at `C:\Users\Go\Desktop\ARIA_copy.zip` (excludes `.venv`, `__pycache__`, `data/`).
- The user's own server on **port 8000 may still run OLD code** — it must be restarted to pick up all fixes.
- Python <3.12: f-string expressions cannot contain backslashes (avoid `\n` inside `{...}` in f-strings).
- After the refactor, `app.py` inserts its own dir into `sys.path` so `uvicorn app:app` works from any CWD; running `python app.py` from the ARIA dir still works as before.
- Minor fix during refactor: `except HTTPException: raise` added in `/api/connect_file` so a rejected handwritten PDF returns **400** (was being swallowed into a 500).
- `sqlcoder:7b` and `aria-goal` are the Ollama models; sqlcoder has NO chat template → must use `complete()`/`generate()`, never `chat()`.
- Uploaded CSVs/sqlite dbs land in `data/uploads/`; per-CSV schema files are `schema_mapping_<name>_latest.json`.