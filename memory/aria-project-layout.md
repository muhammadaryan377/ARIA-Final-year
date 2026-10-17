---
name: aria-project-layout
description: ARIA is a FastAPI+React analytics app with two layers: old (src/pages/) and active figma (aria-frontend/src/figma/)
metadata:
  type: reference
---

# ARIA Project Layout

## Active Frontend: `aria-frontend/src/figma/`
- `App.tsx` — Figma router (switch on `page` state), passes all connection/question/goal props down to pages
- `pages/` — Dashboard, AIAnalytics, Insights, Predictive, Prescription, Hypothesis, Data, Copilot, Documents, History, Settings
- `components/AppShell.tsx` — shell layout (sidebar nav, header with dbName)
- `index.css` — Tailwind CSS with dark mode, custom properties

## Old/Legacy Frontend: `aria-frontend/src/pages/` and `aria-frontend/src/components/`
- These still exist and are imported by `App.jsx` (root `src/`) — the root App.jsx is the actual mounted entry
- The figma layer is rendered inside a `<FigmaApp>` component embedded in the old App — **not** a separate SPA
- `App.jsx` (root) handles auth flow, LLM mode, DB connection, goal/insight agent orchestration; passes state to FigmaApp

## Backend: `/`
- `app.py` — FastAPI main; mounts routers
- `api/` — auth_routes, connect_routes, query_routes, document_routes, copilot_routes, hypothesis_routes, kpi_routes, settings_routes, feedback_routes
- `goal_agent/`, `insight_agent/`, `prediction/`, `prescription_agent/`, `schema_agent/`, `schema_engine/`
- `llm_provider.py` — accepts `api_key` param; falls back to `os.getenv("GROQ_API_KEY")`
- `document_processor/` — PDF intelligence (extraction, chunking, embeddings, vector store)
- `data/` — SQLite `aria.db`, `schemas/`, `uploads/`, `kaggle_retail_sales/`

## Security
- `GROQ_API_KEY` stored DPAPI-encrypted in `.env` (507B)
- User-applied Groq key: POST `/api/settings/providers` → stored in session dict `session["api_keys"][provider]` — **never written to disk**
- `connect_routes.py` reads user key: `session.get("api_keys", {}).get(request.provider)`

## Data Source Routes
- `POST /api/connect` → general (line 165)
- `POST /api/connect/postgresql` → PostgreSQL (line 171)
- `POST /api/connect/mysql` → MySQL (line 177)
- `POST /api/connect_file` → CSV upload (line 183)
- `POST /api/connect_pdf` → PDF upload (line 278)
- `GET /api/schema` → fetch schema
- `GET /api/settings/providers` → list configured providers
- `POST /api/settings/providers` → save user API key

## Playwright E2E
- `aria-frontend/tests/e2e/` — 5 `.cjs` specs (01-auth, 02-figma-pages, 03-main-pages, 04-data-connect, full-journey)
- `aria-frontend/playwright.config.cjs` — CommonJS, baseURL http://localhost:8000 (backend-served dist/ build; vite :5173 is dev-only)
- Browser: chromium-1243 installed

## Dev Ports
- Frontend dev: `http://localhost:5173` (Vite)
- Backend: `http://localhost:8000` (FastAPI) with `/docs`

## DO NOT DELETE
- `aria-frontend/src/figma/` — active UI
- `.env` — encrypted GROQ key
- `data/aria.db` — SQLite data
- `document_processor/` — PDF intelligence
- `api/connect_routes.py` — data source routes

## Previously Deleted (commit 2a8df7c)
- `aria-frontend/aria-frontend/` (nested empty)
- `aria-frontend/.backup-20260905-031204/`
- `recycled-restore/`
- `artifacts/`
- 13 `*.log` files
- `.pytest_cache/`
