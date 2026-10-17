# ARIA: Aryan backend audit and implementation plan

Source: the supplied `aria audit plan.docx`.
Repository: `muhammadaryan377/ARIA-Final-year`, branch `main`.
Reviewed baseline: `60430d184d2ca54929a977dc1ce1932ebbda2543` (17 October 2026).

## Scope and completion status

This change set implements confirmed backend defects found during the review. It is **not a declaration that every acceptance criterion in the original audit has passed**. Remaining integration work and live verification are listed explicitly below. No frontend, Semal-owned route, label helper, or query suggestions block was edited. The new regression test file belongs to the audit's shared "tests in owned areas" scope. This document is the requested plan.

The coverage table mentions `schema_engine/lexical.py` under the mixed Ask area, but the detailed ownership section assigns that file to Semal. The more specific assignment was followed: it was not edited. `core/ratelimit.py` is explicitly assigned to Aryan in section 3.1.

## Implemented fixes and why they matter

| Priority | File / function | Confirmed problem | Implemented behavior |
|---|---|---|---|
| P0 | `api/auth_routes.py::_check_login_rate_limit` | An outer broad exception handler swallowed Redis's 429 rejection. Different workers could each allow another login budget. | Calls the shared limiter directly; 429 and 503 propagate. Existing local guard remains. |
| P0 | `core/ratelimit.py` | Merely configuring Redis bypassed the shared-store configuration check, even when actual operations failed. Memory updates also lacked a thread lock. | Configured/required shared-store failures return 503. Production multi-worker defaults and explicit opt-in use one policy. Unknown backend names fail. Local memory updates are serialized. |
| P0 | `core/config.py::_redis_load_one`, `_redis_save_one`, `get_session`, `save_session` | Redis read/write failures silently used potentially stale worker/file data. Save could acknowledge state never written to Redis. | Shared-store operations fail with 503. Missing/expired Redis sessions cannot resurrect a stale worker snapshot. Shared persistence happens before committing local state. |
| P0 | `core/config.py::_ensure_provider` | A selected hosted provider could silently become local Ollama after reconstruction failure. | Preserves selected provider name, clears unusable provider instance, records a safe actionable error. New sessions do not implicitly initialize an unselected provider. |
| P0 | `auth.py` and `app.py::lifespan` | JWT secret loaded during import; JWT validation did not require expiry; same-user logins within one second could produce identical tokens. | Lazy signing-secret resolution, startup validation, required `exp`/`iat`/`sub`, random `jti` per token. Existing `auth.SECRET` access/override and rotation remain compatible. |
| P0 | `core/config.py::_ensure_user_paths`, `_coerce_user_id` | Matching only a parent directory's name did not prove containment; bool/float IDs could coerce to another integer. | Checks resolved parent against the exact owning directory; rejects bool/float IDs. |
| P0 | `api/prescription_routes.py::prescription_save` | Arbitrary client payload overwrote server ID, creation date and question; nested SQL/debug/credentials and markup could be persisted. | Top-level field allowlist; bounded recursive cleanup; 64 KB payload limit; nonfinite values become null; random server IDs and UTC creation time always win. |
| P1 | `core/config.py::_atomic_write_text` | Concurrent writers shared one `.tmp` filename, so one could truncate/replace another's temporary file. | Unique private temporary files in the target directory. This does not claim to solve every whole-session lost-update race. |
| P1 | `core/secret_box.py::decrypt` | On non-Windows systems an `enc1:` Windows ciphertext was returned as if it were a plaintext password. | Explicit error for unavailable Windows decryption; legacy plaintext passthrough retained. |
| P1 | `api/connect_routes.py` progress and session status | First polling/SSE reader destroyed terminal progress; interrupted delivery could lose the result. Provider errors were not surfaced in session status. | Retains terminal progress for 10 minutes, expires it on read/write, exposes selected provider and safe provider error. |
| P1 | `api/schema_routes.py::_extract_schema_impl` | File-source schema path could be a string; refreshed schema did not update readiness status. | Normalizes with `Path`; records `ok` or `empty` after extraction. |
| P1 | `goal_agent/grounding.py::_pick_measure_column` | After excluding dimensions, fallback selected numeric years/flags again. | Returns no measure if only dimension-like numeric columns exist. It never invents an aggregate. |
| P1 | `goal_agent/grounding.py::_estimate_column_cardinality` | Stripping the schema could query a different same-named table through search path. | Quotes each segment and retains schema qualification. |
| P1 | `goal_agent/suggestions.py` | Multiword generic measures, qualified FK references and short template lists were handled incompletely. | Generic-token matching without swallowing specific metrics; exact or unambiguous FK matching; additional real-measure questions and bounded cardinality-aware backfill (at most eight extra probes). No fabricated padding on dimension-only schemas. |
| P1 | `api/query_routes.py` engine failure/envelope paths | Failure text claimed the connection/schema worked without verifying that; missing metadata fallback claimed partial findings even without data. | Honest failure text, complete metric field, backend-failure when no result exists. Suggestions block unchanged. |
| P1 | `prediction/evaluate.py` | Unequal arrays could break MAPE; NaN/negative error could produce a misleading accuracy. | Aligns available pairs, rejects nonfinite MAPE inputs and invalid accuracy inputs; zero-only actuals remain unscoreable. |
| P1 | `services/kpi_service.py::_coerce_num` | Numeric strings could return NaN/Infinity despite the numeric input guard. | Rejects nonfinite values in both parser and fallback paths. |
| P1 | `api/insight_rag_routes.py::_capability`, health, chat | DeepSeek ignored the user's saved key; session failures/unknown providers could silently route elsewhere; health was hardwired to the DeepSeek environment key. | Uses selected user's key, maps `rag_scope`, propagates session failures, rejects unknown providers, reports selected-provider configuration and safe errors. Configuration readiness is not proof of network reachability. |
| P1 | `copilot/pipeline.py` forecast error paths | Raw exception text was returned in user-facing forecast explanations. | Safe actionable messages; detailed exception remains server-side. |
| P1 | `hypothesis_engine/evidence_ladder.py::normalize` | Strong contradictory evidence failed to downgrade when supporting evidence count exceeded one. | Contradictory grade when contradiction strictly outweighs support, matching the documented rule. |
| P1 | `reporting/digests.py::generate_digest_now` | Session errors became empty sessions; an unsuccessful report response could be recorded as successful. | Propagates into the failed-run path; requires a saved report ID before recording success. |
| P1 | `api/rl_routes.py` | Oversized events were silently truncated by the downstream recorder, while API always reported success with no durability qualification. | Bounds event name and finite JSON payload, rejects oversized input, labels persistence as best-effort and reward as client feedback. |

## Existing behavior checked; no rewrite justified yet

- Password decryption cache already has a lock and a 256-entry bound. Avoided adding a second cache.
- Prediction already contains sparse-demand disclosures, timezone normalization metadata, exact measure/time resolution, structured filters and model-substitution notes. These require scenario validation, not indiscriminate replacement.
- Copilot already has schema retrieval, conversation fingerprinting, grounding checks and a guided failure explanation. The mocked retrieval/intent suites were used to guard compatibility.
- Hypothesis investigation already adds evidence split, contribution, ranking, confirm/disprove and follow-ups. These are not evidence that every rival explanation or unsettled-period scenario has been validated.
- Verification exposes authenticated dashboard/Ask/forecast/prescription routes and numerical/evidence/completeness checks. It does not replace live source verification.
- Autocomplete already has a bounded index and distinct-value cap. Broad schema and stale-cache evaluation remains below.

## Remaining plan: do not mark these acceptance criteria complete

### 1. Platform integration gate

**Files:** `app.py`, `core/config.py`, `core/ratelimit.py`, `auth.py`, `api/auth_routes.py`, `core/secret_box.py`.

1. Run two real application workers with one Redis instance and the same injected JWT secret; set `ARIA_WORKERS=2` and `ARIA_REQUIRE_SHARED_STORE=1`.
2. Authenticate through worker A, read source/provider through worker B, exhaust one shared login budget across both.
3. Stop Redis after initialization: rate-limited requests and session read/write must return 503. Restore Redis and confirm recovery without stale session resurrection.
4. Test concurrent read-modify-write for the **same user's** saved collections. Redis shares state but whole-dictionary saves remain last-writer-wins. Introduce revision/CAS or a scoped mutation API only after reviewing all callers and preserving shared helper signatures.
5. Test Windows DPAPI and file ACL behavior using Aryan's Windows account. Linux tests cannot certify DPAPI.
6. Session-provider error is available to the UI; Semal must wire the visible banner. Do not silently change her files.

**Important deployment behavior:** enabling Redis makes it authoritative. Existing file-only sessions are not silently migrated on a Redis miss; users may need to reconnect/reselect the provider. Local default remains file/memory when Redis is neither configured nor required.

### 2. Upload job lifecycle (backend work still required)

**Files:** `api/connect_routes.py`, `api/schema_routes.py`, `api/data_routes.py`; Data page is Semal-owned.

Current progress state is keyed by user, not job. Retaining terminal results fixes lost delivery, but does **not** isolate simultaneous uploads or implement cancellation.

Implementation sequence:

1. Generate a server job ID per ingestion; store `(user_id, job_id)` state in the shared backend with TTL. Keep the old latest-job progress endpoint compatible.
2. Add job-scoped status and cancellation endpoints with ownership checks.
3. Pass job identity into background workers. Check cancellation between spool/profile/load/schema/session-commit steps. Do not claim synchronous CPU/database work can be instantly interrupted.
4. Write schema/SQLite outputs to job-specific paths; only atomically activate a completed, non-cancelled job. Prevent an older job replacing a newer active connection.
5. Keep progress polling usable beyond the 120-second SSE window and across reconnects/workers; preserve terminal state for an explicit TTL.
6. Ask Semal to wire AbortController + cancel-on-unmount to the endpoint. Frontend abort alone does not cancel backend processing.

Acceptance: same-user same-filename parallel uploads; different users; cancel before/after load; >2 minute job; SSE disconnect/reconnect; worker restart; no cross-job results or orphaned active session.

### 3. Analytics and retrieval matrix

**Files:** `api/kpi_routes.py`, `services/kpi_service.py`, `api/query_routes.py` engine only, `goal_agent/grounding.py`, `goal_agent/suggestions.py`, `api/predictive_routes.py`, `prediction/*`, `api/copilot_routes.py`, `copilot/*`, `api/insights_routes.py`, `api/insight_rag_routes.py`, `insight_agent/*`.

Use at least three synthetic schema shapes: one flat file with IDs/years/locale numbers; a declared-FK relational star; a multi-schema database with duplicate bare table names. Use arbitrary generated names rather than production-specific names.

- Compare totals/counts/grouping against direct reference SQL. Include no-measure and high-cardinality-only data. Check useful suggestion completeness without invented questions or year sums.
- Forecast: sparse/zero demand, short history, locale ambiguity, timezone boundaries, requested measure/time mismatch, multiple split values, filters and substitutions. Report MAE/WAPE/coverage alongside MAPE; do not certify model quality from unit tests.
- Copilot: current question must override irrelevant history; dataset switching must invalidate context; unsupported SQL must fail with useful table hints.
- Retrieval: create held-out question-to-evidence labels on unseen schemas/documents; measure recall@k and precision@k, and test cross-user/document scope. Unit routing tests are not a precision/recall benchmark.
- Verify the query response envelope on success, clarification, zero rows, substitution, provider timeout and storage failure. Leave Semal's suggestions block alone.

### 4. Hypotheses, prescriptions and verification

**Files:** `api/hypothesis_routes.py`, `api/hypothesis_investigation_routes.py`, `hypothesis_engine/*`, `api/prescription_routes.py`, `prescription_engine/*`, `verification/*`, `api/verification_routes.py`.

- Rival explanations: distinguish measured alternatives from untested possibilities; do not upgrade evidence from wording or an LLM vote.
- Settled-data tests: incomplete latest periods, delayed ingestion and revised records; disclose exclusions before period comparisons.
- Exercise the prescription save/list/edit round trip with real engine cards. Extend nested-field schemas only with evidence of fields required by the existing UI; never allow client ownership metadata to win.
- Add evidence provenance checks so client-submitted cards are not mistaken for independently verified engine output. Text cleanup is not a general prompt-injection defense or proof of truth.
- Verify NaN/Infinity, empty evidence, substitutions and stale snapshots are not awarded a clean verification badge.

### 5. Reports, RL and autocomplete

**Files:** `reporting/*`, reporting parts of `api/intelligence_routes.py`, `api/rl_routes.py`, `api/autocomplete_routes.py`.

- Scheduler is per-process: add a renewable shared lease/idempotency key before certifying multi-worker scheduled digests. Test duplicate ticks, long generation, restart and lease loss; never mark a failed report run successful.
- `core/rl.py` swallows recorder failures and is not explicitly assigned in the audit. Do not edit it under the current strict ownership constraint. A durable acknowledgement contract requires ownership clarification. The route now explicitly reports best-effort persistence.
- Client-provided reward remains untrusted feedback; do not use it as verified business impact or let it bypass analytical validation.
- Autocomplete: test schema change invalidation, high-cardinality limits, cross-user isolation and hostile identifiers; modify only if a reproducible defect appears.

## Verification record

Final regression run: **315 passed, 101 subtests passed, 196 warnings** in 18.72 seconds. `compileall` and `git diff --check` passed. Warnings were not treated as proof of failure or suppressed in the implementation; the pytest command only omitted their expanded display.

Suites: `test_aryan_audit_regressions`, `test_auth_hardening`, `test_goal_agent`, `test_prediction_forecast`, `test_prescription_topnotch`, `test_session_isolation`, `test_security_guards`, `test_remediation`, `test_copilot_e2e_mocked`, `test_copilot_rag`, `test_hypothesis_investigate`, `test_query_pipeline`, `test_prediction_forecast_prelim`, `test_prescription_display`, `test_intelligence`, `test_analytical_result`, `test_rag_cloud_provider_stability` (all under `tests/`, `.py`). This is the relevant regression selection, not the entire repository's test suite.

The scratch runtime initially lacked test/application dependencies; those were installed locally for verification. SQLAlchemy was aligned with the repository's `2.0.51` pin after the runtime's newer default expected a different PostgreSQL driver. The full pinned production environment was not rebuilt. Tests use fake providers, temporary data and local fixtures. They do not validate hosted credentials, real Redis workers, Windows DPAPI, live PostgreSQL/MySQL or frontend behavior. No dependency lockfile or CI workflow was changed.

## Pull safely in PowerShell

In the local checkout of the **personal** repository:

```powershell
Set-Location 'C:\Users\malik\Downloads\ARIA-fyp'
git status --short
git branch --show-current
git remote -v
```

Confirm the checkout/remote is `muhammadaryan377/ARIA-Final-year` and the branch is `main`. If it is the team repository or branch `aryan`, do not run the pull below there. If there are local edits, commit or preserve them first; do not reset them.

```powershell
git pull --ff-only origin main
python -m pytest tests/test_aryan_audit_regressions.py -q
```

No team repository or team branch is updated by this personal-repository push.
