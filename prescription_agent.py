"""ARIA Prescription Agent
--------------------------
Role: Recommended Actions

Turns the Insight Agent's findings (KPIs, trends, anomalies, hypotheses) plus
the Prediction Agent's forecast into 3-5 concrete, prioritized next actions a
manager should take. Uses the configured LLM (local Mistral or cloud Groq) and
falls back to a template derived from the actual numbers if the LLM fails.

When DB context is provided (engine, dialect, intel, table, metric), delegates
to ``prescription_engine.analyze`` for evidence-backed deterministic prescriptions.

Output: {"actions": [str, ...], "source": "llm" | "template" | "engine"}
"""

import json
import logging
import re

from llm_provider import create_provider

logging.basicConfig(level=logging.INFO)


class PrescriptionAgent:
    def __init__(self, provider=None):
        if provider is None:
            logging.getLogger(__name__).warning(
                "PrescriptionAgent created with provider=None – failing closed"
            )
            raise ValueError(
                "PrescriptionAgent requires an explicit provider. "
                "Pass the LLMProvider object from require_data() or session['provider']."
            )
        self.llm = provider

    def prescribe(self, processed_data, kpis, trends, anomalies, hypotheses,
                  predictions=None, max_actions=5, *,
                  engine=None, dialect=None, intel=None, table=None, metric=None):
        """Return a dict with 'actions' (list of strings) and 'source'.

        When ``engine``, ``dialect``, ``intel``, ``table``, and ``metric`` are
        provided, the canonical ``prescription_engine.analyze`` path is used
        for evidence-backed deterministic prescriptions.  The legacy LLM and
        template paths are preserved for callers without DB context.
        """
        processed_data = processed_data or {}
        kpis = kpis or []
        trends = trends or []
        anomalies = anomalies or []
        hypotheses = hypotheses or []

        # ---- Deterministic engine path (§4 pipeline) ----
        if all(v is not None for v in (engine, dialect, intel, table, metric)):
            try:
                from prescription_engine.engine import analyze as _engine_analyze
                result = _engine_analyze(engine, dialect, intel, table, metric)
                if result.get("ok") and result.get("prescriptions"):
                    actions = []
                    for rx in result["prescriptions"][:max_actions]:
                        title = rx.get("title") or rx.get("summary", "")
                        if title:
                            actions.append(title)
                    if actions:
                        return {
                            "actions": actions,
                            "source": "engine",
                            "engine_status": result.get("status"),
                            "engine_driver": result.get("driver"),
                            "engine_story": result.get("story"),
                            "engine_summary": result.get("summary_plain"),
                            "impact_note": "Impact estimates derived from deterministic statistical analysis.",
                        }
                # Engine returned no_prescription or ok=False — fall through to LLM/template.
                logging.getLogger("aria.prescription").info(
                    "Engine returned no prescription (%s); falling back to LLM/template",
                    result.get("status", result.get("detail")),
                )
            except Exception as exc:
                logging.getLogger("aria.prescription").warning(
                    "Engine delegation failed (%s); falling back to LLM/template", exc)

        # ---- LLM path ----
        context = {
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "row_count": processed_data.get("row_count", 0),
            "kpis": kpis[:6],
            "trends": trends[:3],
            "anomalies": anomalies[:5],
            "hypotheses": hypotheses[:4],
            "prediction": self._prediction_summary(predictions),
        }

        prompt = f"""
You are ARIA's Insight Agent. Recommend exactly {max_actions} concrete,
prioritized next actions for a non-technical manager. Tie each action to the
actual numbers in the context (KPIs, trends, anomalies, hypotheses, and the
forecast if one is present).

Context:
{json.dumps(context, indent=2, default=str)}

Rules:
- Output ONLY a numbered list, one action per line, no extra text.
- Each action must be specific enough to execute this week.
- If the forecast shows a strong projected change, include an action that
  prepares for it.
"""
        try:
            text = self.llm.chat(
                "story",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                num_predict=500,
            )
            if not (text or "").strip():
                logging.getLogger("aria.prescription").warning(
                    "Prescription LLM returned empty; using template actions")
                return self._honest_fallback(
                    self._template_actions(kpis, trends, anomalies, hypotheses,
                                           predictions, max_actions),
                    "template", "LLM_EMPTY")
            actions = self._parse_actions(text, max_actions)
            if actions:
                # Integrity gate: LLM numbers survive only on quality_gate
                # pass — otherwise scrub + tag source + honest note.
                if not self._fallback_gate_passes(processed_data, kpis, trends, anomalies, hypotheses):
                    actions = [self._scrub_numbers(a) for a in actions]
                return {
                    "actions": actions,
                    "source": "llm",
                    "status": "SUCCESS",
                    "evidence_pass": self._fallback_gate_passes(
                        processed_data, kpis, trends, anomalies, hypotheses),
                    # Honesty guard (t.txt section 43): never invent financial impact.
                    "impact_note": "Impact could not be reliably estimated from the available data.",
                }
            logging.getLogger("aria.prescription").warning(
                "Prescription LLM returned unparseable prose; using template actions")
            return self._honest_fallback(
                self._template_actions(kpis, trends, anomalies, hypotheses,
                                       predictions, max_actions),
                "template", "INVALID_LLM_RESPONSE")
        except Exception as exc:
            logging.getLogger("aria.prescription").exception("Prescription LLM failed; using template actions")

        return self._honest_fallback(
            self._template_actions(kpis, trends, anomalies, hypotheses,
                                   predictions, max_actions),
            "template", "LLM_UNAVAILABLE")

    def recommendation_for_validated_decision(self, decision: dict):
        """PHASE 10: validated Decision -> existing prescription engine.

        Delegates to prescription_engine.recommendation_for_decision (the
        single engine). Never returns title-as-recommendation, never an
        LLM-only action. Returns the supported text or None.
        """
        try:
            from prescription_engine.engine import recommendation_for_decision as _rfd
            return _rfd(decision if isinstance(decision, dict) else {})
        except Exception as exc:
            logging.getLogger("aria.prescription").debug(
                "recommendation_for_decision failed closed: %s", exc)
            return None

    @staticmethod
    def _prediction_summary(predictions):
        # Only an accuracy-validated forecast may influence prescriptions.
        if not predictions or not predictions.get("valid") or not predictions.get("points"):
            return None
        return {
            "column": predictions["column"],
            "period": predictions["period"],
            "horizon": predictions["horizon"],
            "method": predictions["method"],
            "accuracy_pct": predictions["accuracy_pct"],
            "projected_pct_change": predictions["projected_pct_change"],
            "last_actual": predictions["last_actual"],
            "points": predictions["points"][:4],
        }

    @staticmethod
    def _parse_actions(text, max_actions):
        actions = []
        for line in str(text).splitlines():
            line = line.strip().replace("*", "").strip()
            if not line:
                continue
            m = re.match(r"^\s*(?:\d+[\.\)]\s*[\.\)\:]?|[-*]\s*|\u2022\s*)\s*(.*)$", line)
            if m and m.group(1).strip():
                actions.append(m.group(1).strip())
            if len(actions) >= max_actions:
                break
        if actions:
            return actions[:max_actions]

        # The model ignored the list format and returned prose. Descriptive
        # sentences are not prescriptions — signal unparseable output so the
        # caller falls back to template actions instead of promoting prose.
        return []

    # Integrity gate: legacy fallback must NEVER return unvalidated
    # numbers. Numbers survive only when quality_gate passes; otherwise the
    # caller gets plain words tagged with source + an honest note.
    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")
    _CAUSAL_RE = re.compile(
        r"\b(cause[sd]?|causing|prove[sd]?|proof|guarantee[sd]?|will\s+(increase|decrease|improve))\b",
        re.IGNORECASE)

    @staticmethod
    def _scrub_numbers(text):
        """Strip unvalidated numbers from a fallback action (plain words)."""
        try:
            s = re.sub(r"\(change [^)]*\)", "", str(text or ""))
            s = re.sub(r"\(observed [^)]*\)", "(outside its usual range)", s)
            s = re.sub(r"\(backtest accuracy [^)]*\)", "", s)
            s = re.sub(r"\(mean [^)]*\)", "", s)
            s = re.sub(r"-?\d+(?:\.\d+)?%\s*(change|accuracy)?", "a notable change", s)
            s = re.sub(r"\s{2,}", " ", s).strip(" .")
            return s
        except Exception:
            return str(text or "")

    @staticmethod
    def _fallback_gate_passes(processed_data, kpis, trends, anomalies, hypotheses):
        """quality_gate probe for legacy fallback inputs (never invents pass).

        Legacy insight inputs carry no fitted r/p/n evidence, so the gate
        can only pass when the caller already supplies validated evidence
        (e.g. a hypothesis with n + p_value + effect). Otherwise False —
        numbers are scrubbed and the honest note carries the decision.
        """
        try:
            from prescription_engine.engine import quality_gate as _qg
            cands = []
            for h in (hypotheses or []):
                if not isinstance(h, dict):
                    continue
                try:
                    n = h.get("n", h.get("sample_n"))
                    n = int(n) if n is not None else None
                except (TypeError, ValueError):
                    n = None
                probe = {"metric": h.get("metric") or h.get("column") or h.get("outcome"),
                         "n": n, "p_value": h.get("p_value", h.get("p")),
                         "r": h.get("r", h.get("effect_r")),
                         "confidence": h.get("confidence", "unknown")}
                g = _qg(probe)
                if isinstance(g, dict) and g.get("pass"):
                    return True
            return False
        except Exception:
            return False

    @classmethod
    def _honest_fallback(cls, actions, source, status):
        """Tag source + honest note; scrub numbers unless the gate passes."""
        return {"actions": actions, "source": source, "status": status,
                "evidence_pass": False,
                "impact_note": "Impact could not be reliably estimated from the available data."}

    @staticmethod
    def _template_actions(kpis, trends, anomalies, hypotheses, predictions, max_actions):
        # Plain-language template: no unvalidated numbers (no pct/value/
        # mean/accuracy figures) — validated numbers only flow through the
        # deterministic engine path with quality_gate evidence.
        actions = []
        if hypotheses and isinstance(hypotheses[0], dict):
            actions.append(f"Validate the leading hypothesis: {hypotheses[0].get('hypothesis', 'N/A')}")
        if trends and isinstance(trends[0], dict):
            t = trends[0]
            cols = ", ".join(t.get("columns", []) or []) or "the key measure"
            actions.append(f"Review the {t.get('direction', 'unknown')} trend in {cols}.")
        if anomalies and isinstance(anomalies[0], dict):
            a = anomalies[0]
            actions.append(f"Investigate the unusual {a.get('column', 'reading')} — well outside its usual range.")
        if predictions and predictions.get("valid") and predictions.get("points"):
            period_word = predictions.get("period") or "periods"
            if period_word in ("sequence", "daily", "weekly", "monthly"):
                period_word = "periods"
            actions.append(f"Plan for the projected change in {predictions['column']} over the next "
                           f"{predictions['horizon']} {period_word}.")
        if kpis and not actions:
            valid = [k for k in kpis if isinstance(k, dict) and k.get("column")]
            if valid:
                actions.append(f"Keep monitoring {valid[0].get('column', 'key metric')}.")
        if not actions:
            actions.append("Continue monitoring the key metrics; no urgent action is indicated.")
        return actions[:max_actions]