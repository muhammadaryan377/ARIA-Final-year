"""Intent routing for ARIA Copilot (t.txt §§21, 34).

Every user question is classified BEFORE any tool runs. SQL is one tool
among several — the router decides which tools each question needs so that
WHY / forecast / anomaly / recommendation / document / schema questions are
never forced through a single SQL-generation step.

Deterministic and provider-independent: keyword + pattern rules only, no LLM
round-trip, so routing works offline and identically on every provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Intent names (stable strings; stored in conversation memory + API payloads).
SCHEMA = "schema"
DOCUMENT = "document"
GENERAL = "general"
FORECAST = "forecast"
HYPOTHESIS = "hypothesis"
RECOMMENDATION = "recommendation"
ANOMALY = "anomaly"
HEALTH = "health"
MIXED = "mixed"
COMBINED = "combined"
DATA = "data"


@dataclass
class IntentResult:
    """Router output: what the question needs and which tools to run."""

    intent: str
    tools: list[str] = field(default_factory=list)
    confidence: str = "moderate"  # high | moderate | low
    reason: str = ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


_SCHEMA_RE = re.compile(
    r"\b(what (data|tables?|columns?) (do i have|are (available|connected))|"
    r"which tables? (contains?|have|hold)|what does (this|the) column mean|"
    r"describe (my|the|our) (data|schema|database|tables?)|"
    r"list (my|the) tables?|show (me )?(my|the) schema)\b"
)

_DOCUMENT_RE = re.compile(
    r"\b(invoice|receipt|contract|pdf|document|report (says|say|shows?)|"
    r"terms (in|of)|what does (this|the) (document|report|invoice|pdf) (say|show|contain)|"
    r"uploaded (report|document|file))\b"
)

_FORECAST_RE = re.compile(
    r"\b(forecast|predict|projection|projected|outlook|next (month|quarter|year|week|period)|"
    r"will .* (increase|decrease|recover|grow|improve|fall|rise)|"
    r"what (do you expect|will happen)|demand (forecast|outlook))\b"
)

_WHY_RE = re.compile(
    r"\b(why|what (caused|is causing|could be causing|drives?|explains?)|"
    r"reason (for|behind)|driver|what.*(decline|drop|decrease|fall|fell|dropped) (.*(due to|because|caused))?)\b"
)

_ANOMALY_RE = re.compile(
    r"\b(unusual|anomal|outlier|concerning|something wrong|anything wrong|"
    r"problems? (in|with)|issues? (in|with)|find .* (concerning|interesting|odd)|"
    r"is anything|should i be worried)\b"
)

_RECOMMEND_RE = re.compile(
    r"\b(what should|recommend|suggest|what to do|which .* should we focus|"
    r"should we (increase|decrease|focus|invest)|advice|action|improve|"
    r"what should i (do|focus|pay attention))\b"
)

_HEALTH_RE = re.compile(
    r"\b(health check|how are we doing|how (is|are) (sales|business|revenue|we)|"
    r"biggest .* problems?|most important .* (know|focus)|important stuff|"
    r"what should i (pay attention|focus on)|give me .* (overview|summary)|"
    r"business performance|what happened)\b"
)

_GENERAL_RE = re.compile(
    r"^(what (is|are|does|do)|define|explain) "
)

_DATA_SCOPE_RE = re.compile(
    r"\b(my|our|sales|revenue|orders?|customers?|products?|data|month|quarter|"
    r"trend|region|inventory|profit|costs?|last (month|year|week|quarter)|"
    r"this month|total|average|compare|top|best|worst)\b"
)

_MIXED_JOIN_RE = re.compile(
    r"\b(why.*what should|what should.*why|fallen.*why.*what|decline.*what.*do|"
    r"why.*and.*(should|recommend|do))\b"
)

# Document + database in one question (§7): "compare the report's explanation
# with the actual data", "does the report explain why sales fell".
_COMBINED_RE = re.compile(
    r"\b(compare .* (report|document|pdf)|report.*(explain|say|says|show)|"
    r"according to both|report .* actual|actual .* report|document .* data|"
    r"data .* document|management report)\b"
)


def classify(question: str) -> IntentResult:
    """Classify a (resolved) user question into an intent + tool plan.

    Rule order is deliberate: schema/document/general are checked before the
    analytical intents so meta-questions never reach SQL; MIXED is checked
    before its parts so multi-need questions get the full pipeline.
    """
    q = _norm(question)
    if not q:
        return IntentResult(DATA, ["sql"], "low", "empty question; default data path")

    if _SCHEMA_RE.search(q):
        return IntentResult(SCHEMA, ["schema"], "high", "schema/data-understanding phrasing")

    if _COMBINED_RE.search(q):
        return IntentResult(
            COMBINED,
            ["sql", "document", "analysis", "comparison"],
            "high",
            "database + document phrasing in one question",
        )

    if _DOCUMENT_RE.search(q):
        return IntentResult(DOCUMENT, ["document"], "high", "document/PDF phrasing")

    if _MIXED_JOIN_RE.search(q) or (_WHY_RE.search(q) and _RECOMMEND_RE.search(q)):
        return IntentResult(
            MIXED,
            ["sql", "analysis", "hypothesis", "recommendation"],
            "high",
            "why + what-to-do in one question",
        )

    if _FORECAST_RE.search(q):
        return IntentResult(
            FORECAST, ["sql", "analysis", "forecast"], "high", "forecast phrasing"
        )

    if _ANOMALY_RE.search(q):
        return IntentResult(
            ANOMALY, ["sql", "analysis", "anomaly"], "high", "anomaly/problem phrasing"
        )

    if _WHY_RE.search(q):
        return IntentResult(
            HYPOTHESIS, ["sql", "analysis", "hypothesis"], "high", "why/driver phrasing"
        )

    if _RECOMMEND_RE.search(q):
        return IntentResult(
            RECOMMENDATION,
            ["sql", "analysis", "recommendation"],
            "high",
            "recommendation phrasing",
        )

    if _HEALTH_RE.search(q):
        return IntentResult(
            HEALTH,
            ["sql", "analysis", "recommendation"],
            "moderate",
            "general business health phrasing",
        )

    # Definitional questions with no data scope -> general knowledge, never SQL.
    if _GENERAL_RE.search(q) and not _DATA_SCOPE_RE.search(q):
        return IntentResult(GENERAL, ["llm"], "moderate", "definitional, no data scope")

    return IntentResult(DATA, ["sql"], "moderate", "default data lookup path")
