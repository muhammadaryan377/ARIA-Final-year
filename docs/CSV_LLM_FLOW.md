# CSV LLM analysis

CSV Dashboard and POST `/api/ask` now use `services/csv_llm_service.py`.
The existing relational, PDF, prediction, copilot, autocomplete and dedicated
Insights-page pipelines are not migrated by this change. No frontend files are
changed. The existing UI still has its own chart formatting and annotations.

## Flow

1. Upload validates/parses CSV and imports SQLite deterministically. Upload
   progress describes importing; it does not claim an LLM analyzed the file.
2. Opening the dashboard sends schema/types/row count to the selected provider.
   The LLM proposes KPI and chart SQL. Raw source-row samples are not sent.
3. SQLite executes validated queries in a read-only connection with an
   authorizer restricted to the connected table and approved SQL functions.
   Queries have a three-second execution budget and a 200-result-row bound.
4. A second LLM invocation explains executed results. Numeric statements use
   result-cell references replaced by actual values. No automatic prior-month
   comparison, business-cause template, or independent-verification claim is added.
5. Questions use the same plan/execute/explain stages. The model declares
   output measures/dimensions; existing keyword grounding and SQL rewrites are
   bypassed. Multiple ungrouped totals stay separate metrics, not categories.

Planning uses the `semi` model role (a chat-capable schema/planning model,
including for local Ollama); explanation uses `story`. The provider continues
to enforce cloud consent, redaction, configured models and rate limits.
Provider errors surface as errors. Invalid JSON, unsafe SQL or invalid evidence
references allow one repair per stage, then fail closed without template results.
Ambiguous schema/question meaning returns a clarification response.

The planner chooses SQL, aliases, measure roles and chart type. Execution
constraints are safety/shape validation, not rules selecting business metrics.
Currency is not inferred from column names: this CSV path returns number format.

## Traces and cache

`llm_trace` appears in `/api/ask` responses and dashboard `config`. It records
request ID, provider, per-invocation model, stage, status and elapsed milliseconds.
`provider_invocation_count` counts calls to the adapter, not internal SDK retries.
Token usage and exact network request counts are `null`: the current adapter
does not expose them. Logs contain timing/model/status, not prompts or keys.

Dashboard results are cached in the user's session with dataset path, mtime,
schema, provider name and service version in the key. A cached response marks
`source=cache`, `cached=true`, `calls_this_request=0`; its retained invocation
trace describes the original analysis. New uploads clear the cache and prior
conversation context. Ask responses are not cached by this service.

Provider invocation timeout is at most 25 seconds, with a 100-second service
budget checked before each invocation. SDK retry/backoff can add overhead;
this is not a hard wall-clock cancellation guarantee for an in-flight call.
Large schemas (>100 columns), oversized context and results exceeding bounds
fail explicitly rather than silently dropping columns/results.

## Verification and limits

Run without API credentials or optional ML packages:

```powershell
python -m unittest discover -s tests -p test_csv_llm_service.py -v
```

Tests use real temporary SQLite data and scripted provider responses. They
exercise separate totals, grouping, clarification, bounded repair, provider
failure, SQL authorization, output limits, NULLs, grounded numeric narration,
cache invalidation, route persistence and failure/history isolation.

These tests do not demonstrate live model quality. LLM semantics and qualitative
claims can still be wrong; `verified=false` states this explicitly. Numeric cell
references prevent invented literal numeric values, but do not prove that the
model chose the right metric or described a referenced value correctly.

After pulling, restart the backend with cloud consent enabled when using cloud.
No frontend rebuild is needed. Open the existing CSV dashboard to trigger the
new analysis; start a new question/conversation to test it. Old saved history
does not get rewritten. For ARIA_Test_Sales.csv, verify totals 285000 revenue,
90000 profit, 90 quantity and 30 orders, then a city grouping and an unknown
metric clarification. Inspect `llm_trace` in Network responses and backend logs.
