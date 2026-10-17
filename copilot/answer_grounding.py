"""Answer-grounding check for ARIA Copilot database answers (n.txt §6).

After the LLM narrates SQL results, verify deterministically:

1. Did the answer use retrieved context? (mentions a retrieved table/column/
   measure/dimension, or the intent needs no DB context)
2. Does it contain unsupported claims? (numbers in the answer that appear
   nowhere in the executed rows → invented)
3. Does the numerical answer match executed data? (SQL is authoritative; the
   LLM explains, never invents — "Total sales $X" must come from the query)
4. Does the answer actually answer the question? (key question terms covered)

Fail-closed on numeric mismatch for data/analytical intents: the caller must
fall back to the deterministic row summary. Reuses the enterprise-quality
signal philosophy (``insight_rag.enterprise_quality.grounding_quality``) for
the quality index and mirrors ``verification_stability``'s binary-verdict
discipline (deterministic verdict authoritative, optional LLM check advisory).
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_NUMBER_RE = re.compile(
    # The thousands-group alternative must contain at least one ",ddd" group,
    # otherwise "1100" parses as 110 + 0 and never matches the row value.
    r"\$?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s*[KMBkmb](?![a-zA-Z])|\s*%)?"
)
_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how", "does", "did",
    "with", "from", "that", "this", "about", "have", "total", "many",
    "much", "most", "best", "show", "tell", "give", "list", "find",
    "were", "was", "are", "the", "and", "for", "our", "your", "data",
}


def _norm_numbers(values: list[object]) -> set[float]:
    out: set[float] = set()
    for v in values:
        try:
            out.add(round(float(str(v).replace(",", "").replace("$", "")
                                .replace("%", "").strip()), 2))
        except (TypeError, ValueError):
            continue
    return out


def _number_tokens(text: str) -> list[dict]:
    """Every number in ``text`` with the metadata grounding needs.

    Each token carries the parsed ``value`` (K/M/B suffixes scaled), the raw
    digit string, the suffix, and whether it was written as currency — so
    the check can tell a prose year ("in 2024") from a formatted measure
    ("$1,999", which must never hide behind the year exemption).
    """
    out: list[dict] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group("num")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        suffix = (match.group("suffix") or "").strip()
        scaled = suffix.upper()
        if scaled == "K":
            value *= 1_000
        elif scaled == "M":
            value *= 1_000_000
        elif scaled == "B":
            value *= 1_000_000_000
        prefix = match.group(0)[:match.start("num") - match.start()]
        out.append({"value": round(value, 2), "raw": raw, "suffix": suffix,
                    "currency": "$" in prefix})
    return out


def _numbers_in_text(text: str) -> list[float]:
    return [t["value"] for t in _number_tokens(text)]


def _is_year_token(tok: dict) -> bool:
    """Plain 4-digit year in prose ("in 2024") — never a formatted measure."""
    return (len(tok["raw"]) == 4 and tok["raw"].isdigit()
            and not tok["suffix"] and not tok["currency"]
            and 1900 <= tok["value"] <= 2100)


def _key_terms(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (question or "").lower())
            if w not in _STOPWORDS and len(w) > 2}


def _row_numbers(rows: list[dict]) -> set[float]:
    vals: list[object] = []
    for row in rows or []:
        vals.extend(row.values())
    return _norm_numbers(vals)


def check_answer_grounding(
    *,
    question: str,
    answer: str,
    retrieved_tables: list[str],
    retrieved_columns: list[str] | None = None,
    sql: str | None = None,
    rows: list[dict] | None = None,
    columns: list[str] | None = None,
    intent: str = "data",
) -> dict:
    """Run the four deterministic grounding checks. Never raises."""
    try:
        return _check(question=question, answer=answer,
                      retrieved_tables=retrieved_tables or [],
                      retrieved_columns=retrieved_columns or [],
                      sql=sql, rows=rows or [], columns=columns or [],
                      intent=intent)
    except Exception as exc:
        log.info("Grounding check failed open (%s).", exc)
        return {"verdict": "needs_review", "reason": "check_error",
                "checks": {}, "quality": {"score": 0.0, "level": "low"}}


def _check(*, question, answer, retrieved_tables, retrieved_columns,
           sql, rows, columns, intent) -> dict:
    ans_low = (answer or "").lower()
    q_terms = _key_terms(question)

    # 1. Used retrieved context?
    ctx_tokens = {t.lower() for t in retrieved_tables}
    ctx_tokens |= {c.lower() for c in (retrieved_columns or [])}
    ctx_tokens |= {c.lower() for c in (columns or [])}
    used_context = (
        intent in {"general", "schema"}  # these intents need no row context
        or not ctx_tokens
        or any(tok and tok in ans_low for tok in ctx_tokens if len(tok) > 2)
        or any(tok in ans_low for tok in q_terms & ctx_tokens)
        or "no rows" in ans_low or "no rows matched" in ans_low
    )

    # 2/3. Numeric consistency vs SQL (SQL authoritative).
    row_nums = _row_numbers(rows)
    # Result context: the values the query returned, the size of the result
    # set, and every number literal in the executed SQL (LIMIT 50, ids,
    # date literals).
    ctx_nums = set(row_nums) | set(_numbers_in_text(sql or ""))
    if rows:
        ctx_nums.add(float(len(rows)))
    ans_tokens = _number_tokens(answer)
    ans_nums = [t["value"] for t in ans_tokens]

    def _near(v: float, pool) -> bool:
        return any(abs(v - p) <= max(0.01, abs(p) * 0.005) + 1e-9 for p in pool)

    matched = [v for v in ans_nums if _near(v, row_nums)]
    # Fail-closed: any number the result context cannot support is invented,
    # at any magnitude — the old "<1000 is prose" exemption let "Total sales
    # is 999" pass against rows of 50,000. Exempt: numbers present in the
    # rows / row count / SQL text, and plain 4-digit prose years.
    invented: list[float] = []
    for tok in ans_tokens:
        v = tok["value"]
        if _near(v, ctx_nums) or _is_year_token(tok):
            continue
        if row_nums and any(abs(v - r) / max(1.0, abs(r)) < 0.05 for r in row_nums):
            continue  # presentation rounding of a row value
        if v not in invented:
            invented.append(v)
    has_measure_numbers = bool(row_nums)
    numeric_consistent = not invented
    if not ans_nums:
        numeric_consistent = True  # qualitative answer over evidence: nothing to falsify

    # A numeric answer whose every number comes from the executed rows uses
    # the retrieved context by construction (covers deterministic fallbacks
    # like "The result is 1,100." which restate SQL output without naming tables).
    if has_measure_numbers and not invented and matched:
        used_context = True

    # 4. Answers the question? (lexical coverage of question key terms,
    # relaxed: answer may paraphrase, so require only partial overlap.)
    if not q_terms:
        answers_question = bool((answer or "").strip())
    else:
        ans_terms = set(re.findall(r"[a-z0-9]+", ans_low))
        # Numeric answers ("The result is 600.") legitimately cover the
        # question via the number, not vocabulary — but only when every
        # number is backed by the result context: invented numbers must
        # never auto-pass this check.
        if ans_nums and row_nums and not invented:
            answers_question = True
        else:
            overlap = q_terms & ans_terms
            answers_question = len(overlap) >= min(2, len(q_terms)) or len(overlap) / max(1, len(q_terms)) >= 0.25

    unsupported = bool(invented)
    if intent in {"data", "forecast", "hypothesis", "anomaly",
                  "recommendation", "health", "mixed", "combined"}:
        if not rows:
            verdict = "grounded" if ("no rows" in ans_low or not ans_nums) else "ungrounded"
            reason = "empty_result" if verdict == "grounded" else "numbers_without_evidence"
        elif not numeric_consistent or unsupported:
            verdict, reason = "ungrounded", "numeric_mismatch_vs_sql"
        elif not used_context:
            verdict, reason = "needs_review", "context_not_referenced"
        elif not answers_question:
            verdict, reason = "needs_review", "question_not_addressed"
        else:
            verdict, reason = "grounded", "sql_authoritative_match"
    else:
        verdict = "grounded" if answers_question else "needs_review"
        reason = "non_data_intent"

    quality = _quality_index(verdict, used_context, numeric_consistent,
                             answers_question, len(rows))
    return {
        "verdict": verdict,
        "reason": reason,
        "checks": {
            "used_retrieved_context": bool(used_context),
            "unsupported_claims": sorted(invented)[:5],
            "numeric_match_vs_sql": bool(numeric_consistent),
            "answers_question": bool(answers_question),
            "sql_authoritative": True,
            "row_count": len(rows),
        },
        "quality": quality,
    }


def _quality_index(verdict: str, used_ctx: bool, numeric_ok: bool,
                   answers_q: bool, row_count: int) -> dict:
    """Deterministic quality index reusing the enterprise-quality shape.

    Delegates to ``insight_rag.enterprise_quality.grounding_quality`` when
    importable so DB and document answers share one auditable scale; falls
    back to the local heuristic otherwise.
    """
    score = 0.0
    if verdict == "grounded":
        score = 0.55 + (0.15 if used_ctx else 0.0) + (0.15 if numeric_ok else 0.0) \
            + (0.10 if answers_q else 0.0) + (0.05 if row_count else 0.0)
        score = min(1.0, round(score, 4))
    elif verdict == "needs_review":
        score = 0.4
    try:
        from insight_rag.enterprise_quality import grounding_quality

        enterprise = grounding_quality(
            evidence_status="supported" if verdict == "grounded" else "insufficient",
            citation_status="cited" if used_ctx and verdict == "grounded" else "missing_citations",
            sources=[{}] if row_count else [],
        )
        return {"score": score, "level": _level(score),
                "enterprise_score": enterprise.get("score"),
                "enterprise_level": enterprise.get("level"),
                "calibrated_probability": False}
    except Exception:
        return {"score": score, "level": _level(score),
                "calibrated_probability": False}


def _level(score: float) -> str:
    return "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"


def llm_verify_db_answer(llm, *, question: str, answer: str,
                         sql: str | None, rows_preview: str) -> str | None:
    """Optional advisory LLM verification (mirrors verification_stability).

    Returns "grounded" / "ungrounded" / None (unavailable). The deterministic
    check above is always authoritative; this never overrides it — it only
    adds a semantic signal for logging.
    """
    try:
        prompt = (
            "You audit a database answer. SQL results are authoritative. "
            "Reply with exactly GROUNDED or UNGROUNDED.\n\n"
            f"QUESTION: {question}\nSQL: {sql or '(none)'}\n"
            f"ROWS (preview): {rows_preview[:1500]}\nANSWER: {answer}"
        )
        raw = llm.complete("story", prompt, temperature=0.0,
                           num_predict=10, timeout=30)
        text = (raw or "").strip().upper()
        if "UNGROUNDED" in text:
            return "ungrounded"
        if "GROUNDED" in text:
            return "grounded"
    except Exception as exc:
        log.info("Advisory DB verification unavailable (%s).", exc)
    return None
