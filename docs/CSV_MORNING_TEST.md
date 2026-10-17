# CSV acceptance checks

## Pull and run (PowerShell)

```powershell
cd C:\Users\malik\Downloads\ARIA-fyp\ARIA-main
git pull origin main
npm run build --prefix aria-frontend
python -m pytest tests/test_csv_handler.py tests/test_csv_scale.py tests/test_csv_llm_service.py tests/test_csv_statistics.py tests/test_csv_completion.py -q
python -m csv_engine.evaluation
```

Restart your existing backend on port 8001, then refresh the browser. Keep your
chosen provider/key and existing cloud consent configuration. The production
requirements already include scipy; no new production dependency is added.
Do not paste PowerShell continuation prompts (`>>`) into commands.

Offline evaluation writes data/csv_evaluation_report.json. Eight SQL oracle cases
should pass; missing-age clarification is deliberately not_run offline. These
checks test ingestion/calculation, not model reasoning. The datasets in
tests/fixtures/csv_eval are synthetic fixtures, not real customer data.

## Manual app checks

Upload tests/fixtures/csv_eval/sales.csv (three records).

| Question/check | Expected |
| --- | --- |
| Total revenue and profit? | 285000 revenue; 90000 profit |
| Which city has highest revenue? | Islamabad; 150000 |
| What is its profit? | Islamabad only; 45000 |
| Overall profit margin? | 90000/285000 × 100 ≈ 31.578947% |
| Average customer age? | Clarification: source has no age column |
| Refresh the page | Same latest chat and answers restored |
| New analysis, then refresh | Empty new chat; old filters not inherited |
| Switch to another CSV | No old dataset conversation mixed in |
| Explain hidden patterns/anomalies | Evidence with method/sample limits; small data may be insufficient |

Ask and Dashboard should show a semantic_review trace and result checks.
A successful review still says interpretation is not independently verified.
Invalid model output gets a bounded repair; failure must show an error, never a
fabricated success. A fresh overview costs additional model calls versus simple
totals, and follow-up resolution adds a call. Dashboard cache reuse is labeled.

Upload sensors.csv: average reading is 20 (NULL omitted), capacity total is zero,
and reading/capacity ratio must be unavailable, never infinity or a made-up zero.
Upload categories.csv: blue has one record, red two; no invented financial metric.

## Opt-in live benchmark

This makes provider calls and may cost money. Set your provider's key in your
local environment or existing .env; never commit it. Backend consent must already
be enabled. Use the model ID you selected in the app:

```powershell
python -m csv_engine.evaluation --live --provider deepseek --model YOUR_SELECTED_MODEL_ID --output data/csv_live_evaluation.json
```

The CLI uses environment keys, not browser-saved encrypted keys. The nine cases
include a real two-turn winner/profit follow-up. Output aliases are specified in
these benchmark questions so expected answers can be checked automatically.
A blocked follow-up means its prerequisite case failed. Read failures and the
model traces; do not treat these nine synthetic cases as general model accuracy.

## Scope still requiring live acceptance

Automated unit tests use scripted provider responses. They do not establish live
DeepSeek accuracy, production authentication behavior, or browser reload behavior.
Statistical sampling covers at most 10000 matching records; the profile can still
use the first 10000. Daily checks do not model seasonality. A second LLM review
can share the planner's mistakes. The separate Intelligence Recent UI is not a
full CSV conversation archive. Arbitrary generated Python, forecasting and causal
inference are not part of this CSV engine.

## Build-time validation record

101 distinct targeted tests passed across the parsing, scale, LLM contract,
statistics, review and conversation suites (including six scale subtests).
The frontend TypeScript/production build passed. Eight offline oracle cases
passed; absent-field clarification was not run offline. Two legacy route-import
checks could not run in the build environment because FastAPI was unavailable:
`test_payload_bounds` and `test_upload_guard_stays_100mb`. No live provider or
browser end-to-end accuracy claim is made. The production app environment should
run the complete command above, including those two checks.
