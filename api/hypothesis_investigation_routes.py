"""Enriched Hypothesis investigation API (hyp.txt).

One endpoint over the existing evidence engine — no duplicate pipeline:
POST /api/hypotheses/investigation {question, top_n}

Runs hypothesis_engine.investigate.investigate_question, then enriches
each candidate with the small reusable services:
evidence_ladder (5 levels incl. Contradictory) / validator
(support-contradict-unknown + temporal note) / contribution ($+share%)
/ ranker (deterministic top 3-5) / next_steps (dynamic follow-ups +
confirm/disprove) + decision_handoff payload for Decision Center.
"""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from core.config import get_session
from core.deps import get_current_user, require_writable, require_data
from core.errors import sanitize_error as _sanitize_error

router = APIRouter(tags=["hypothesis-investigation"])


class InvestigationRequest(BaseModel):
    question: str = Field(default="", max_length=500)
    top_n: int = Field(default=4, ge=1, le=5)


@router.post("/api/hypotheses/investigation")
def enriched_investigation(req: InvestigationRequest, request: Request,
                           user: dict = Depends(require_writable)):
    from api.hypothesis_routes import (_hypothesis_setup, _linked_hypothesis_setup,
                                       _session_documents)
    from hypothesis_engine.investigate import investigate_question
    from hypothesis_engine import evidence_ladder as _ladder
    from hypothesis_engine import validator as _val
    from hypothesis_engine import contribution as _contrib
    from hypothesis_engine import ranker as _ranker
    from hypothesis_engine import next_steps as _next
    from insight_agent import sanitize_json
    require_data(request)
    session = get_session(user["user_id"])
    try:
        setup, err = _hypothesis_setup(user)
    except Exception:
        setup, err = None, {"rejected": True}
    if err:
        setup, err = _linked_hypothesis_setup(session)
    if err:
        return {"ok": False, **err} if "rejected" not in err else {
            "ok": False, "code": "INVALID_SOURCE",
            "detail": "This source is a text document — use Document Chat instead."}
    mapping, tables, engine, dialect, intel = setup
    try:
        llm = session.get("provider")
        if not isinstance(llm, str) and llm is not None and not hasattr(llm, "complete"):
            llm = None
        out = investigate_question(engine, dialect, intel, tables, req.question,
                                   documents=_session_documents(user["user_id"]), llm=llm)
        if not out.get("ok", True):
            return sanitize_json({"ok": False, **out})
        dims = _dims_for(intel, tables)
        enriched, shares = [], {}
        for h in (out.get("hypotheses") or []):
            sp = _val.split_evidence(h)
            lad = _ladder.normalize(h.get("strength") or h.get("evidence_strength") or "",
                                    sp["n_support"], sp["n_against"])
            chart = h.get("chart") or {}
            contrib = _contrib.build_table(chart.get("groups") or [])
            key = str(h.get("title") or h.get("id") or "")
            shares[key] = contrib.get("top_share", 0.0)
            h2 = dict(h)
            h2["strength"] = lad
            h2["evidence_split"] = sp
            h2["contribution"] = contrib
            tn = _val.temporal_note(h)
            if tn:
                h2["temporal_note"] = tn
            cd = _next.confirm_disprove(h)
            h2["confirm"] = cd["strengthen"]
            h2["disprove"] = cd["weaken"]
            h2["follow_ups"] = _next.follow_ups(h, dims, out.get("observation", {}).get("metric"))
            h2["decision_handoff"] = {
                "origin": "hypothesis",
                "metric": out.get("observation", {}).get("metric") or h.get("metric"),
                "driver": h.get("driver") or (contrib.get("top_label")),
                "question": req.question,
                "hypothesis_title": h.get("title"),
                "evidence_strength": lad,
            }
            enriched.append(h2)
        ranked = _ranker.rank(enriched, shares, req.top_n)
        out["hypotheses"] = ranked
        out["ladder"] = ["Strong evidence", "Moderate evidence", "Limited evidence",
                         "Insufficient evidence", "Contradictory evidence"]
        return sanitize_json({"ok": True, **out})
    except Exception as exc:
        try:
            _d = str(_sanitize_error(exc))[:200]
        except Exception:
            _d = "The investigation could not run."
        return {"ok": False, "code": "ANALYSIS_FAILED",
                "detail": f"The investigation could not run: {_d}"}
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def _dims_for(intel, tables) -> list[str]:
    dims: list[str] = []
    try:
        for t in (tables or [])[:3]:
            for c in (intel.get_categorical_cols(t) if hasattr(intel, "get_categorical_cols") else []):
                if c not in dims:
                    dims.append(c)
            if len(dims) >= 6:
                break
    except Exception:
        pass
    return dims[:6]
