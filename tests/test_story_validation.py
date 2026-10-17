"""Structured story validation (A-H): CODE CALCULATES / LLM EXPLAINS."""

from insight_agent.agent import InsightAgent


class _FakeProvider:
    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc

    def chat(self, *a, **k):
        if self._exc:
            raise self._exc
        return self._text


def _evidence():
    processed = {"user_goal": "Revenue trend", "row_count": 120}
    kpis = [{"column": "revenue", "sum": 1840000.0, "mean": 15333.33,
             "median": 15000.0, "min": 9000.0, "max": 25000.0,
             "std": 3000.0, "count": 120}]
    trends = [{"dimension": "month", "direction": "increasing",
               "pct_change": 18.4, "columns": ["revenue"]}]
    anomalies = [{"column": "revenue", "row": 5, "value": 25000.0,
                  "z_score": 3.1, "note": "beyond 2 standard deviations"}]
    hypotheses = [{"hypothesis": "'North' in region contributes about 40% of revenue, "
                                 "making it the dominant region segment.",
                   "basis": "Ranked every region group.", "confidence": "medium", "score": 8.0}]
    return processed, kpis, trends, anomalies, hypotheses


def _agent(text=None, exc=None):
    return InsightAgent(provider=_FakeProvider(text, exc))


def test_a_normal_success():
    processed, kpis, trends, anomalies, hyps = _evidence()
    polished = ("Revenue increased 18.4% compared with the previous period. "
                "On average revenue sits around 15333.33, ranging in the usual band. "
                "The headline finding is that 'North' in region contributes about "
                "40% of revenue, making it a notable segment to watch further.")
    agent = _agent()
    ok, reason = agent._validate_polished_story(
        polished, processed, kpis, trends, anomalies, hyps)
    assert ok is True, reason
    # Full path with cleaner bypassed (core cleaner's diversity check is
    # out of scope here): validator acceptance is the success criterion.
    import insight_agent.agent as _mod
    real_clean = _mod.clean_llm_text
    _mod.clean_llm_text = lambda raw: ((raw or "").strip(), None)
    try:
        agent2 = _agent("  " + polished + "  ")
        story = agent2.generate_business_story(processed, kpis, trends, anomalies, hyps)
        assert "18.4" in story
        assert agent2.last_story_status == "SUCCESS"
    finally:
        _mod.clean_llm_text = real_clean


def test_b_deterministic_fallback_on_llm_error():
    processed, kpis, trends, anomalies, hyps = _evidence()
    agent = _agent(exc=RuntimeError("down"))
    story = agent.generate_business_story(processed, kpis, trends, anomalies, hyps)
    assert "revenue" in story.lower()
    assert agent.last_story_status == "LLM_UNAVAILABLE"


def test_c_empty_llm_response():
    processed, kpis, trends, anomalies, hyps = _evidence()
    agent = _agent(text="   ")
    story = agent.generate_business_story(processed, kpis, trends, anomalies, hyps)
    assert story  # deterministic fallback, truthful
    assert agent.last_story_status == "LLM_EMPTY"


def test_d_malformed_response():
    processed, kpis, trends, anomalies, hyps = _evidence()
    agent = _agent(text="```json {not a story")
    story = agent.generate_business_story(processed, kpis, trends, anomalies, hyps)
    assert "revenue" in story.lower()
    assert agent.last_story_status in ("INVALID_LLM_RESPONSE", "LLM_EMPTY")


def test_e_incorrect_numeric_claim_rejected():
    processed, kpis, trends, anomalies, hyps = _evidence()
    agent = _agent()
    ok, reason = agent._validate_polished_story(
        "Revenue surged to 9999999.99 last quarter, a historic record high for the "
        "business across all segments and channels worldwide today.",
        processed, kpis, trends, anomalies, hyps)
    assert ok is False
    assert "numeric" in reason


def test_f_incorrect_segment_claim_rejected():
    processed, kpis, trends, anomalies, hyps = _evidence()
    agent = _agent()
    ok, reason = agent._validate_polished_story(
        "The headline finding is that 'Atlantis' in region drives revenue growth "
        "with strong momentum across the observed months in the data sample.",
        processed, kpis, trends, anomalies, hyps)
    assert ok is False
    assert "entity" in reason


def test_g_incorrect_period_claim_rejected():
    processed, kpis, trends, anomalies, hyps = _evidence()
    agent = _agent()
    ok, reason = agent._validate_polished_story(
        "Revenue grew strongly in January with broad gains across the observed "
        "months in the dataset sample period under review here today.",
        processed, kpis, trends, anomalies, hyps)
    assert ok is False
    assert "period" in reason


def test_h_insufficient_evidence_honest_state():
    agent = _agent(exc=RuntimeError("down"))
    story = agent._narrative_story({"user_goal": "Unknown goal", "row_count": 5},
                                   [], [], [], [])
    assert "no strong pattern" in story.lower()
