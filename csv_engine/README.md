# CSV engine

CSV backend implementation is kept together here. Existing imports remain valid
through compatibility aliases at the former three entrypoints.

| File | Responsibility |
| --- | --- |
| handler.py | Parsing, type/schema extraction and validation |
| importer.py | Import orchestration and SQLite materialization |
| storage.py | Read-only connections and shared validation helpers |
| profiling.py | Bounded quality profile and dataset fingerprint |
| conversation.py | Current-chat context and follow-up resolution |
| analyst.py | LLM planning, safe SQL, investigation and narration |
| prompts.py | Planning and evidence-grounded explanation instructions |
| statistics.py | Distribution, outlier, relationship, group and time tools |
| discovery.py | Adapt statistical evidence to the result contract |
| verification.py | Evidence integrity, SQL reproducibility and LLM semantic review |
| investigation.py | Bounded multi-round follow-up investigation |
| chat_store.py | Durable per-user, per-dataset conversation storage |
| ranking.py | Within-method effect ranking for investigation |
| evaluation.py | Offline oracle and opt-in live-model acceptance runner |
| constants.py | Engine version and result limits |

The LLM selects analyses; SQL and statistical tools compute values. Arbitrary
model-generated Python is not executed. Results expose method and scope;
semantic correctness and causal explanations are not certified.
Statistical tools inspect a reproducible uniform reservoir sample of at most 10,000 matching rows after scanning all matches within a time budget. Daily-change tools require complete data. SQL aggregates
can inspect the complete source. Analysis never modifies uploaded values.
Conversation context remains scoped to the current chat and dataset. Separate
relational/PDF and other independent app pipelines are outside this package.

Tests: `tests/test_csv_llm_service.py` and `tests/test_csv_statistics.py`.
See `docs/CSV_LLM_FLOW.md` for operating instructions and limits.
