# CSV analyst flow (version 3)

Scope: CSV Dashboard, AI Analytics Ask, suggested questions, autocomplete and
conversation persistence. Relational/PDF analytics and dedicated Copilot,
Insights, forecasting and report generation remain separate pipelines. Existing
CSV parsing/import validation is retained; no model rewrites uploaded data.

## Data and planning

- Profile metadata, exact total record count, missing/blank counts, distinct
  counts, numeric ranges/means and full-row duplicates. The aggregate profile
  covers at most the first 10,000 rows (explicitly marked, not a random sample),
  with a five-second profiling budget. A timed-out check is unavailable rather
  than guessed. Actual analysis SQL reads the full connected CSV.
- No raw source-row samples are sent automatically. The model can plan targeted
  read-only queries to inspect date formats, categories or other required values;
  executed results are sent for explanation under the existing cloud consent.
- LLM interprets dataset meaning and uncertain assumptions. Broad exploratory
  questions explicitly request an overview, not automatic clarification. Specific
  missing fields or genuinely conflicting meanings still require clarification.
- Plan one to six complementary queries. Simple questions normally need one.
  The model may request one investigation round after seeing initial results,
  adding up to two queries. Maximum eight successful result sets and sixteen
  SQL attempts including repairs. This is a bounded workflow, not an unlimited
  autonomous analyst. Statistical work uses supported SQLite aggregates,
  expressions and windows; arbitrary Python execution is not exposed.
- Date interpretation and zero-safe ratios are instructed in the planner.
  These instructions do not mathematically prove correct metric semantics.
- Dashboard and Ask receive the same profile and saved dataset interpretation.
  A new dataset fingerprint clears the old interpretation. Explicit questions
  take priority over history; history includes all prior query definitions and
  clarification exchanges, scoped to the conversation.

## Execution and answers

Every query runs on a read-only SQLite connection restricted by an authorizer
to the connected table and allowlisted functions. Per-query budget: three
seconds and 200 result rows. SQL errors/invalid JSON receive one bounded repair
per stage. Provider failures surface without a template fallback.

The explanation sees the original question and executed evidence from all
queries. Numeric literals must reference result cells and are substituted by
backend values. NULL and empty results remain distinct from zero. This prevents
invented literal numbers but cannot prove that a referenced value is described
correctly. `verified=false` remains deliberate.

`analysis_results` and `evidence` contain every query. Legacy top-level `data`,
`sql_used`, `columns`, and chart fields retain the primary query for compatibility.
AI Analytics renders each result separately, with charts, all returned rows in a
scrollable table, SQL evidence, assumptions, quality limitations and model trace.
Charts omit NULL measures with an explicit note; charts exceeding 60 points are
labeled as truncated. Tables retain all returned rows. Dashboard exposes profile
scope, assumptions and quality issues. Old saved responses are not rewritten.

## Suggestions, caching and limits

CSV suggestions come from the selected LLM or its cached dashboard follow-ups.
Autocomplete only filters those existing questions; typing never triggers a
model call and does not invent template metrics. Profile and suggestion caches
include dataset identity; suggestion/dashboard caches include provider/model.
Ask analysis itself is not cached. Disconnect clears CSV caches.

Traces expose provider, model, stage, status, duration, invocation count and SQL
attempt count. Cached dashboard traces describe original generation and declare
zero calls for the current request. Token usage and internal SDK network counts
remain unknown. No success is fabricated after a model failure.

The service checks a 100-second budget before model calls, each capped at forty
seconds or remaining time. Client Ask allows 120 seconds. SDK retry overhead is
not a hard wall-clock guarantee. Profiles over 100 columns, excessive context,
invalid output shapes and oversized results fail explicitly.

## Validation and running

```powershell
python -m unittest discover -s tests -p test_csv_llm_service.py -v
python -m unittest discover -s tests -p test_provider_selection.py -v
```

Tests use real temporary SQLite databases and scripted model responses. They
cover multi-query evidence, investigation/stop, SQL isolation, repaired output,
profile quality, empty/dimension-only data, NULL/zero division, date coverage,
cache invalidation, conversation clarification/history and provider failures.
They do not measure live DeepSeek/OpenAI/Claude accuracy.

Pull, build `aria-frontend` using `npm run build`, and restart the backend.
Use your selected provider and model with backend cloud consent enabled.
Try a broad overview, separate totals, city ranking, ratio, follow-up and absent
field. For ARIA_Test_Sales.csv independently compare revenue 285000, profit
90000, quantity 90 and 30 records. Inspect actual model traces and all result
sets. Stronger models and these checks do not guarantee universally correct
business conclusions.

## Validation repair diagnostics

Rejected candidates are sent back to the same provider for a bounded repair,
with a validation reason. The reason appears in the trace and backend log; raw
model output, result rows and API keys are not logged. Explanations receive an
explicit evidence-reference catalogue (first twelve rows per query), plus the
full executed results, to reduce index/alias formatting errors. Invalid numeric
claims and invalid references still fail closed. Presentation-only mismatches
can render as tables; a multi-measure scalar dashboard query yields separate
cards without changing SQL or calculated values. Missing format defaults to
number, without inferring currency. Live failure causes from earlier logs cannot
be determined retroactively because those logs did not include validation reasons.

## Current-chat context

AI Analytics sends an explicit conversation ID on each turn and disables the
shared cross-page metric/filter fallback. New analysis aborts the active UI
request and allocates a fresh ID; a dataset change also resets the chat.
The session stores up to eight CSV conversations, each bound to its dataset
fingerprint, with ten turns each. The planner receives up to four recent turns
within a bounded context budget, including prior answers and small result
excerpts; omitted results require fresh SQL. Clarifications are included.
Previous turns can be inspected in the current-chat panel. A page reload or
remount starts a new UI chat; restoring full conversations from Recent is not
part of this change. Successful analyses still persist in the history database.
