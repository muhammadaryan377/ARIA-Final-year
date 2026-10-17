# ARIA — Setup & Run Guide (read this first)

ARIA is a BI analytics app with a pipeline of agents:

```
connect DB/CSV/PDF -> schema agent (extract mapping) -> goal agent (NL -> SQL -> data)
-> insight agent (KPIs, trends, anomalies, charts, story, predictions, prescriptions)
```

Every agent can run on one of two LLM backends (chosen per session in the UI or
via the `provider` field in API calls):

| provider | backend              | models                                                             |
|----------|----------------------|--------------------------------------------------------------------|
| `local`  | Ollama (offline)     | `aria-sql-mistral` (sql), `aria-story-mistral` (story), `aria-goal` (suggest/semi/schema) |
| `cloud`  | Groq API (fast)      | `llama-3.3-70b-versatile` (all roles), `qwen/qwen3.6-27b` (vision, scanned PDFs) |

Local = your data never leaves the machine but is slow on CPU. Cloud = fast but
sends schema/queries/data samples to Groq. Switching provider mid-session is
explicit and never automatic (`/api/provider/switch`).

---

## 1. Prerequisites

- Python 3.11+
- PostgreSQL server (relational source; MySQL is also supported)
- Ollama — only for the `local` provider: https://ollama.com
- Groq API key — only for the `cloud` provider: https://console.groq.com

## 2. Install the app

```bash
git clone <this-repo> ARIA
cd ARIA
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure `.env`

```bash
# Windows:   copy .env.example .env
# macOS/Linux: cp .env.example .env
```

Set in `.env`:

- `DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD / DB_SCHEMA` — a reachable
  PostgreSQL. `DB_NAME` is only needed for the standalone schema-agent script; the
  web UI connects per-session.
- `GROQ_API_KEY` — your key (required only for the `cloud` provider).
- Optional: `OLLAMA_HOST` (default `http://127.0.0.1:11434`), `ARIA_PORT` (default `8000`).

The JWT secret auto-persists to `data/.jwt_secret` on first start.

## 4. Local LLM models (only for the `local` provider)

The `aria-*` models are **fine-tuned privately** and are NOT on any registry, so
`ollama pull aria-goal` will fail. They are also NOT stored in this repo (the
weights live in the model owner's Ollama store).

1. Read `model_share/README.md` — it documents the packaging/transfer steps.
2. Obtain `model_share.zip` from the owner (they run `model_share/copy_models.ps1`
   to export `aria-goal.gguf`, `base-4b.gguf` + two LoRA adapters, then zip the
   folder). The zip is the ONLY file you need besides this repo.
3. Unzip it, and from inside the unzipped folder, build the models:

   ```bash
   ollama create aria-goal -f Modelfile-goal
   ollama create aria-sql-mistral -f Modelfile-sql
   ollama create aria-story-mistral -f Modelfile-story
   ```

4. Verify: `ollama list` shows all three `aria-*` models. Ensure Ollama is running
   (`ollama serve`, or keep the tray app open).

`aria-goal` alone is enough to test the schema agent; the other two are used by the
goal/insight agents. Role→model mapping lives in `llm_provider.py`.

## 5. Health check

```bash
python core/checks.py        # package / env / Ollama / DB status (exit 1 only if a required package is missing)
python core/checks.py --json # machine-readable
```

`GET /api/health` exposes the same report.

## 6. Run the web app

```bash
python app.py                # or: uvicorn app:app
```

Open http://127.0.0.1:8000 (respects `ARIA_PORT`; auto-picks a free port if busy).

UI flow: register → login → pick a provider card → connect a database
(PostgreSQL/MySQL) or upload a CSV/PDF → **Extract Schema** → ask questions →
generate insights.

## 7. API sequence (for scripting / AI agents)

All data endpoints require `Authorization: Bearer <token>`. Example flow:

```bash
curl -s -X POST http://127.0.0.1:8000/api/auth/register -H "Content-Type: application/json" \
  -d '{"username":"mate","password":"pass123"}'
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login -H "Content-Type: application/json" \
  -d '{"username":"mate","password":"pass123"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
AUTH="Authorization: Bearer $TOKEN"

curl -s -X POST http://127.0.0.1:8000/api/connect -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"provider":"local","db_type":"postgresql","host":"localhost","port":5432,"db":"northwind","user":"postgres","password":"secret","schema":"public"}'
# or upload: POST /api/connect_file (multipart: provider + file; .csv or .pdf)

curl -s -X POST http://127.0.0.1:8000/api/extract_schema -H "$AUTH"   # runs the schema agent
curl -s -X POST http://127.0.0.1:8000/api/suggestions  -H "$AUTH" -H "Content-Type: application/json" -d '{"goal":"total sales by region"}'
curl -s -X POST http://127.0.0.1:8000/api/ask          -H "$AUTH" -H "Content-Type: application/json" -d '{"goal":"total sales by region"}'
curl -s -X POST http://127.0.0.1:8000/api/insight       -H "$AUTH"
```

- `POST /api/provider/switch {"provider":"cloud"}` switches the backend for the session.
- Schema output is written per user to `data/schemas/user_<id>/schema_mapping_<db>_latest.json`,
  plus timestamped snapshots (20 kept). Everything under `data/` is gitignored and regenerable.
- History: `GET /api/history`, `GET /api/history/{id}`.

## 8. Standalone schema agent (no web app)

```bash
python schema_agent.py --host localhost --port 5432 --db mydb --user postgres --password x [--schema public]
```

Add `--llm-provider local|cloud` (and `--api-key <key>` for cloud) to enable LLM
schema reasoning (table descriptions, semantic measures/dimensions, relationship
confirmation). All LLM output is validated before it is merged into the mapping
(`_apply_schema_reasoning` in `schema_agent.py`) — the mapping never contains names
the database does not actually have.

## 9. Gotchas / troubleshooting

- **Local is slow by design**: on CPU, a schema-reasoning call or a SQL query can
  take ~1–3 minutes. Timeouts are already generous (schema 300s, story 900s).
- `"model 'aria-goal' not found"` → the model is not created in this machine's
  Ollama; see §4.
- Cloud 401/403 → `GROQ_API_KEY` missing/wrong.
- Scanned/handwritten PDFs require the `cloud` provider (vision model); `local`
  rejects them with a clear 400.
- Local context-size errors → `llm_provider.py` already sets `num_ctx=8192`; if you
  still hit "exceeds the available context size", the model was created with a
  smaller context and must be re-created from the Modelfile.
- Python <3.12: f-strings cannot contain backslash escapes inside `{...}`.
- Never commit `.env`, `data/`, or `model_share/*.gguf` (all gitignored).
