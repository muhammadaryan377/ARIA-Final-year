# CSV analyst flow (version 5)

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
  The model may request up to three investigation rounds after seeing results,
  adding up to two steps per round within the overall eight-result budget. Maximum eight successful result sets and sixteen
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
Previous turns can be inspected in the current-chat panel. A page reload restores the latest durable CSV chat for the same user and dataset. A new empty chat is also persisted. The separate Intelligence Recent view is not the CSV conversation browser. Successful analyses still persist in the history database.

Follow-up resolution now runs as a separate traced LLM stage before SQL planning
when chat history exists. It produces a self-contained question (or asks for
clarification), preserving referenced entities while allowing metric changes.
The resolved question is used by planning and explanation and displayed in the
UI as Interpreted as. This adds a provider invocation per follow-up; scripted
regression tests verify orchestration and filtered SQL results, not live model
semantic accuracy.

## Statistical discovery (version 5)

Implementation now lives in `csv_engine/`; former entrypoints are compatibility
aliases. See its README for the module map. Existing data/history is unchanged.
The model chooses distribution, robust outliers, Spearman relationships,
group-versus-rest comparisons and daily change tools. Six combined SQL/tool
steps (at most three tools) are allowed initially, plus up to two steps per investigation round within eight results overall.
Statistics use up to 10,000 rows selected by uniform reservoir sampling (seed 42) after scanning all matching rows within a three-second sampling budget;
SQL aggregates use the full connected dataset. Tool filters use bound parameters.

Methods, scope and limitations accompany each result in Ask and Dashboard.
Benjamini-Hochberg adjustment covers comparisons within each invocation, not an
entire adaptive investigation. Small groups/constant columns can be insufficient.
Daily changes require ISO dates and sufficient adjacent dates; seasonality and
incomplete periods are not modeled. Outliers are candidates, not confirmed errors
or fraud. No candidates does not establish that no anomalies exist. Semantic
interpretations remain unverified. Actual computation regression tests:

```powershell
python -m unittest discover -s tests -p test_csv_statistics.py -v
```

## Review, reproducibility, and durable chats (version 5)

Each successful Ask/Dashboard plan receives a separately traced semantic_review
model pass. It checks source expressions, entities, dates, grain and scope. A
rejected plan is replanned once with correction feedback; a second rejection
fails explicitly. This uses the selected model, so it is not independent proof.
Review sees bounded row excerpts, along with SQL/tool definitions and schema.
SQL results are replayed on read-only connections and compared before narration;
this confirms reproducibility, not that the chosen SQL answers the right question.
Evidence shape and finite values are checked. verified remains false.

Investigation runs at most three follow-up rounds, two steps per round, eight
result sets total, sixteen planning SQL attempts and eight statistical attempts.
Time is reserved before starting another investigation; budget stops are labeled.
Replayed SQL is counted separately. Every provider call consumes the shared
100-second soft budget; provider SDK behavior can extend wall-clock duration.

The aggregate data-quality profile still describes its bounded prefix sample.
Statistical tools now scan all matching rows to build a bounded uniform reservoir;
if that scan times out they fail rather than present a prefix as representative.
Rare segments can be missed. Daily-change analysis rejects sampled records;
use narrowed filters or full-data SQL daily aggregates. Discovery rankings sort
within each method by absolute observed effect. They are not calibrated scores
of importance, confidence, or causality.

Chats persist in data/csv_chats.sqlite3, scoped by authenticated user and dataset
fingerprint. Latest twenty conversations per user, ten turns each; size limited
to four MB per conversation. New analysis saves an empty new conversation.
GET /api/csv/conversation restores the latest or a specified conversation;
POST /api/csv/conversation/new starts a new one; DELETE with its ID removes one.
Data/model credentials are never copied into this store. Result values are stored
locally. Reuploading or changing the file identity intentionally starts fresh
context. After session expiry, reconnect the same retained source before restore.

See CSV_MORNING_TEST.md for exact validation commands and expected fixture results.


### Narration failure isolation

Narration accepts typed text/cell segments as well as legacy evidence-reference strings.
Numbers and percentage formatting are rendered from executed cells, never invented by
the UI. Planning, read-only execution, integrity, model review and reproducibility
still must pass before results are exposed. If only narration fails after bounded
repair (or its provider call fails), the response has status `partial` and
`narrative_status=unavailable`: reviewed tables are retained, rejected prose and
follow-ups are omitted, and the UI shows an explicit explanation-unavailable notice.
This is not a fallback analytical answer or independent verification. Partial
dashboards are not cached as completed dashboards. Partial chats retain their
executed evidence for follow-ups and restore. Live evaluation does not count a
partial answer as a fully passed case.
