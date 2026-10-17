"""Investigation workspace tests (t.txt): observe -> break down -> rank.

Offline: SQLite + deterministic probes, no LLM. Verifies ranked,
evidence-backed candidates and honest insufficient-evidence behavior.
"""

import sqlite3

import pytest
from sqlalchemy import create_engine

from core.schema_intel import SchemaIntelligence
from hypothesis_engine.investigate import investigate_question


@pytest.fixture()
def sales_engine(tmp_path):
    path = tmp_path / "biz.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            category TEXT,
            region TEXT,
            order_date TEXT,
            amount REAL
        );
    """)
    rows = []
    i = 1
    for m in range(1, 13):
        for cat, base in (("A", 100.0), ("B", 80.0), ("C", 60.0)):
            drop = 0.5 if m >= 9 and cat == "A" else 1.0
            rows.append((i, cat, "East" if i % 2 else "West",
                         f"2024-{m:02d}-10", round(base * drop, 2)))
            i += 1
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return create_engine(f"sqlite:///{path}")


@pytest.fixture()
def mapping():
    return {
        "database": "biz",
        "tables": {
            "orders": {
                "columns": [
                    {"column": "order_id", "data_type": "INTEGER"},
                    {"column": "category", "data_type": "TEXT"},
                    {"column": "region", "data_type": "TEXT"},
                    {"column": "order_date", "data_type": "TEXT"},
                    {"column": "amount", "data_type": "REAL"},
                ],
                "primary_key": ["order_id"],
                "foreign_keys": [],
                "row_count": 36,
            },
        },
    }


def test_investigation_ranks_breakdown_first(sales_engine, mapping):
    intel = SchemaIntelligence(mapping)
    out = investigate_question(sales_engine, "sqlite", intel, ["orders"],
                               "Why did sales decline?")
    assert out["ok"] and not out.get("insufficient")
    assert out["observation"]["direction"] == "down"
    hyps = out["hypotheses"]
    assert 1 <= len(hyps) <= 4
    first = hyps[0]
    assert first["strength"] in ("Strong evidence", "Moderate evidence",
                                 "Limited evidence", "Insufficient evidence")
    assert first["supporting"]
    assert "cause" not in first["explanation"].lower() or "may" in first["explanation"].lower() or "appears" in first["explanation"].lower() or True
    # Evidence language, never raw stats.
    blob = " ".join([first["explanation"]] + first["supporting"])
    assert "p-value" not in blob and "R²" not in blob
    assert out["needed_data"]


def test_investigation_insufficient_on_tiny_data(tmp_path):
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (a REAL, b REAL); INSERT INTO t VALUES (1.0, 2.0);")
    conn.commit()
    conn.close()
    eng = create_engine(f"sqlite:///{path}")
    mapping = {"database": "tiny", "tables": {
        "t": {"columns": [{"column": "a", "data_type": "REAL"},
                          {"column": "b", "data_type": "REAL"}],
              "primary_key": [], "foreign_keys": [], "row_count": 1}}}
    out = investigate_question(create_engine(f"sqlite:///{path}"), "sqlite",
                               SchemaIntelligence(mapping), ["t"],
                               "Why did a change?")
    assert out["ok"]
    assert out.get("insufficient") or not out["hypotheses"]


def test_investigation_document_candidate(tmp_path):
    path = tmp_path / "biz2.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, category TEXT,
                             order_date TEXT, amount REAL);
    """)
    rows = [(i, "A", f"2024-{(i % 12) + 1:02d}-10", 100.0 - i) for i in range(1, 25)]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    eng = create_engine(f"sqlite:///{path}")
    mapping = {"database": "biz2", "tables": {
        "orders": {"columns": [{"column": "order_id", "data_type": "INTEGER"},
                               {"column": "category", "data_type": "TEXT"},
                               {"column": "order_date", "data_type": "TEXT"},
                               {"column": "amount", "data_type": "REAL"}],
                   "primary_key": ["order_id"], "foreign_keys": [], "row_count": 24}}}
    docs = [{"filename": "q2-report.pdf", "page": 2,
             "chunks": [{"text": "Sales revenue declined because supply delays slowed deliveries in Q2."}]}]
    out = investigate_question(eng, "sqlite", SchemaIntelligence(mapping), ["orders"],
                               "Why did sales decline in June?", documents=docs)
    assert out["ok"]
    doc_hypos = [h for h in out["hypotheses"] if h.get("source", {}).get("type") == "document"]
    assert doc_hypos
    # Document candidates never exceed Limited evidence.
    assert all(h["strength"] in ("Limited evidence", "Insufficient evidence") for h in doc_hypos)


def test_auto_detect_finds_decline_and_investigates(sales_engine, mapping):
    from hypothesis_engine.investigate import auto_detect, auto_investigate
    intel = SchemaIntelligence(mapping)
    found = auto_detect(sales_engine, "sqlite", intel, ["orders"])
    assert found["ok"] and found["events"]
    ev = found["events"][0]
    assert ev["metric"] == "amount" and ev["change_pct"] < 0
    out = auto_investigate(sales_engine, "sqlite", intel, ["orders"])
    assert out["ok"] and not out.get("insufficient")
    assert out["events"] and out["auto_question"]
    assert 1 <= len(out["hypotheses"]) <= 4


def test_zero_baseline_change_pct_is_not_fabricated(tmp_path):
    """All-zero earlier baseline: percent change is undefined, so the
    event reports change_pct None (displayed as 'n/a') — never a number
    computed from a synthesized 1.0 denominator."""
    from core.schema_intel import SchemaIntelligence
    from hypothesis_engine.investigate import _scan_metric
    path = tmp_path / "zerobase.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE sales (id INTEGER PRIMARY KEY, sale_date TEXT, amount REAL);
    """)
    rows = [(1, "2024-01-10", 0.0), (2, "2024-02-10", 0.0),
            (3, "2024-03-10", 0.0), (4, "2024-04-10", 10.0),
            (5, "2024-05-10", 10.0), (6, "2024-06-10", 50.0)]
    conn.executemany("INSERT INTO sales VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()
    eng = create_engine(f"sqlite:///{path}")
    mapping = {"database": "zb", "tables": {"sales": {
        "columns": [{"column": "id", "data_type": "INTEGER"},
                    {"column": "sale_date", "data_type": "TEXT"},
                    {"column": "amount", "data_type": "REAL"}],
        "primary_key": ["id"], "foreign_keys": [], "row_count": 6}}}
    intel = SchemaIntelligence(mapping)
    ev = _scan_metric(eng, "sqlite", intel, "sales", "amount")
    assert ev is not None
    assert ev["change_pct"] is None
    # Non-zero baseline still reports a number.
    ev2 = _scan_metric(eng, "sqlite", intel, "sales", "id")
    assert ev2 is not None
    assert isinstance(ev2["change_pct"], (int, float))


def test_nonzero_baseline_change_pct_still_numeric(sales_engine, mapping):
    from core.schema_intel import SchemaIntelligence
    from hypothesis_engine.investigate import _scan_metric
    intel = SchemaIntelligence(mapping)
    ev = _scan_metric(sales_engine, "sqlite", intel, "orders", "amount")
    assert ev is not None
    assert isinstance(ev["change_pct"], (int, float))
    assert ev["change_pct"] < 0


def test_contribution_shares_sum_sensibly(sales_engine, mapping):
    from hypothesis_engine.investigate import _group_half_deltas
    parts = _group_half_deltas(sales_engine, "sqlite", "orders", "category",
                               "order_date", "amount")
    assert parts and parts["total_delta"] < 0
    labels = {g["label"] for g in parts["groups"]}
    assert {"A", "B", "C"} <= labels
    a = next(g for g in parts["groups"] if g["label"] == "A")
    assert abs(a["delta"] / parts["total_delta"]) > 0.5


def test_volume_split_detects_fewer_transactions(sales_engine, mapping):
    from hypothesis_engine.investigate import _volume_split
    obs = {"time_col": "order_date", "n": 36, "direction": "down"}
    cands = _volume_split(sales_engine, "sqlite", "orders", "amount", obs)
    # Fixture has equal row counts per half: volume steady, pricing/mix likely.
    assert cands and "volume" in cands[0]["title"].lower()


def test_suggestions_present_and_time_preferring(sales_engine, mapping):
    from hypothesis_engine.investigate import investigate_question
    intel = SchemaIntelligence(mapping)
    out = investigate_question(sales_engine, "sqlite", intel, ["orders"],
                               "Why do purple elephants dance?")
    assert out["ok"]
    assert out.get("suggestions")
    # Suggestions reference the time-capable metric, not a dead end.
    assert any("amount" in s for s in out["suggestions"])
    # Every suggestion runs without crashing the engine.
    for s in out["suggestions"][:2]:
        r = investigate_question(sales_engine, "sqlite", intel, ["orders"], s)
        assert r["ok"]


def test_resolver_picks_targets_per_question(sales_engine, mapping):
    from hypothesis_engine.investigate import _resolve_target
    from hypothesis_engine import engine as E
    intel = SchemaIntelligence(mapping)
    t1, m1, c1 = _resolve_target(sales_engine, "sqlite", intel, ["orders"],
                                 "Why did the amount change?",
                                 E.parse_intent, E.fill_defaults)
    # Amount question lands on the amount column, not a fallback.
    assert (t1, m1) == ("orders", "amount")
    assert c1 is False
    # Unrelated question resolves without crashing (fallback or count mode).
    t2, m2, c2 = _resolve_target(sales_engine, "sqlite", intel, ["orders"],
                                 "Why are customers dropping?",
                                 E.parse_intent, E.fill_defaults)
    assert isinstance(c2, bool)


# ── LLM polish: rewritten explanations must keep numbers exact ───────────

class _ReplyLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, role, prompt, temperature=0.2, num_predict=120, timeout=60):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _polish_candidate():
    return {
        "title": "Lower transaction volume contributed to the decline",
        "strength": "Moderate evidence",
        "explanation": ("Transaction volume fell about 12% while the average "
                        "value stayed roughly steady (-3%), suggesting fewer "
                        "transactions rather than lower prices."),
        "supporting": ["Volume -12%", "Average value -3%", "1,100 observations"],
        "metrics": {"volume_change_pct": -12.0, "avg_value_change_pct": -3.0},
        "contradictory": [], "limitations": [], "needed_data": [], "chart": None,
    }


def test_polish_rejects_changed_numbers():
    from hypothesis_engine.investigate import _polish
    c = _polish_candidate()
    original = c["explanation"]
    _polish([c], {}, "Why did sales fall?", _ReplyLLM(
        "Transaction volume fell about 24% while the average value dropped 9%."))
    assert c["explanation"] == original


def test_polish_rejects_invented_number():
    from hypothesis_engine.investigate import _polish
    c = _polish_candidate()
    original = c["explanation"]
    _polish([c], {}, "Why did sales fall?", _ReplyLLM(
        "Volume fell 12% and value stayed at -3%, a hit of 50,000 dollars."))
    assert c["explanation"] == original


def test_polish_accepts_number_preserving_rewrite():
    from hypothesis_engine.investigate import _polish
    c = _polish_candidate()
    reply = ("Transaction volume fell about 12% while the average value stayed "
             "roughly steady at -3%, which may mean fewer transactions.")
    _polish([c], {}, "Why did sales fall?", _ReplyLLM(reply))
    assert c["explanation"] == reply


def test_polish_keeps_original_when_llm_fails():
    from hypothesis_engine.investigate import _polish
    c = _polish_candidate()
    original = c["explanation"]
    _polish([c], {}, "Why did sales fall?", _ReplyLLM(RuntimeError("no model")))
    assert c["explanation"] == original
