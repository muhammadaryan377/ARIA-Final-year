"""Component: intent parsing - goal IR, KPI mapping, clarification gate and metrics overview (spec sections 3, 7, 9)."""

import difflib
import json
import re
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from llm_provider import LLMProvider, create_provider
from core.validation import unknown_sql_tables, empty_sql_tables, unknown_sql_columns
from core.config import SCHEMA_DIR, BASE_DIR
from schema_engine.lexical import singularize

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None

logger = logging.getLogger(__name__)

class IntentMixin:
    # ---- conversational inputs (greetings/thanks/bye/help/who/ack/empty) ----

    def _extract_current_question(self, user_goal):
        """Current-question slice of a (possibly multi-turn) goal string.

        Multi-turn callers prepend history ("Previous question:…" /
        "Conversation context:…Q:…Current question:…"). Conversational
        detection must judge ONLY the current turn — otherwise a 'hi'
        following an analytical question inherits the previous tables and
        answers with data. Schema-agnostic, no table/column names.
        """
        try:
            _m = re.search(r"current question\s*:\s*(.+)$", str(user_goal or ""),
                           re.IGNORECASE | re.DOTALL)
            if _m and _m.group(1).strip():
                return _m.group(1).strip()
        except Exception:
            pass
        return str(user_goal or "")

    # ---- multi-turn current-question isolation (schema-agnostic) ----
    # History lines ("Q: …", "Previous question: …") must never leak
    # measure/aggregation words (e.g. a previous turn's 'average') into the
    # current turn's IR. Every measure/aggregation scan below runs on the
    # CURRENT question only; history is inherited solely for elliptical
    # follow-ups that name no measure of their own.

    @staticmethod
    def _strip_history_marker(text):
        """Text after the last 'Current question:' marker, else text as-is."""
        try:
            _m = re.search(r"current question\s*:\s*(.+)$", str(text or ""),
                           re.IGNORECASE | re.DOTALL)
            if _m and _m.group(1).strip():
                return _m.group(1).strip()
        except Exception:
            pass
        return str(text or "")

    def _current_question_text(self, user_goal):
        """Raw current question: explicit metadata wins, else the marker."""
        try:
            _active = getattr(self, "_active_current_question", None)
            if _active and str(_active).strip():
                return str(_active).strip()
        except Exception:
            pass
        return self._strip_history_marker(user_goal)

    def _history_text(self, user_goal):
        """Everything before the last 'Current question:' marker (may be '')."""
        try:
            _m = re.search(r"current question\s*:\s*(.+)$", str(user_goal or ""),
                           re.IGNORECASE | re.DOTALL)
            if _m:
                return str(user_goal or "")[:_m.start()]
        except Exception:
            pass
        return ""

    def _is_count_explicit_text(self, text):
        """True when the text itself asks for a count (schema-agnostic)."""
        try:
            return bool(re.search(
                r"\bhow many\b|\bnumber of\b|\bcount of\b|\bcounts\b|\bcount\b",
                str(text or "").lower()))
        except Exception:
            return False

    def _is_elliptical_followup_text(self, current_text):
        """True when the current question names NO measure/aggregation word
        but carries a pronoun/continuation cue ('that', 'those', 'it',
        'them', 'and by', 'for that', 'compare them', 'why did it', leading
        'and'/'then'/'also', 'what/how about'). Only then may history lend
        a measure/aggregation; a turn naming its own aggregation ('how many'
        = COUNT) never inherits."""
        try:
            cur = str(current_text or "")
            cur_l = cur.lower()
            if not cur_l.strip():
                return False
            for keyword in (getattr(self, "kpi_index", None) or {}):
                try:
                    if re.search(r"\b" + re.escape(keyword) + r"\b", cur_l):
                        return False
                except Exception:
                    continue
            if self._detect_aggregation(cur) is not None:
                return False
            return bool(re.search(
                r"\b(that|those|it|them|they|this|these|same|above|previous|"
                r"earlier|also)\b|^\s*and\b|\band by\b|\bfor that\b|"
                r"\bcompare them\b|\bwhy did it\b|^\s*(then|also|what about|"
                r"how about|which|who)\b",
                cur_l))
        except Exception:
            return False

    def _conversational_kind(self, user_goal):
        """Kind of pure-conversational input, or None when analytical.

        Matches ONLY pure conversational turns (greetings, thanks, bye,
        help, 'who are you', acknowledgements, empty). Any turn carrying
        analytical content (metric/operation wording beyond the greeting
        itself) returns None so 'hi, show total sales' stays analytical.
        Case-insensitive; handles all-caps and surrounding punctuation.
        """
        try:
            cur = self._extract_current_question(user_goal)
            low = str(cur or "").strip().lower()
            if not low or not re.search(r"[a-z]", low):
                # Empty / whitespace / punctuation-only: conversational.
                return "empty"
            # Normalise punctuation to spaces for matching.
            norm = re.sub(r"[^a-z ]+", " ", low)
            norm = re.sub(r"\s+", " ", norm).strip()
            if not norm:
                return "empty"
            words = norm.split()
            # A pure conversational turn is short. Longer turns naming data
            # concepts stay analytical (never swallow a real question).
            if len(words) > 8:
                return None
            # Greeting prefix with an analytical remainder stays analytical:
            # "hi, show total sales" must not become a greeting.
            _ANALYTICAL_HINTS = (
                "total", "sum", "count", "average", "avg", "mean", "max",
                "minimum", "maximum", "min", "revenue", "revenues", "sales",
                "profit", "margin", "income", "turnover", "quantity", "qty",
                "units", "amount", "value", "price", "cost", "trend",
                "monthly", "daily", "weekly", "yearly", "growth", "decline",
                "decrease", "increase", "change", "drop", "dropped", "fall",
                "fell", "top", "best", "worst", "highest", "lowest", "most",
                "least", "compare", "comparison", "versus", "distribution",
                "breakdown", "share", "month", "year", "over time", "by ",
                "per ", "show", "list", "how many", "number of",
            )
            def _has_analytical_remainder(text):
                t = f" {text} "
                for h in _ANALYTICAL_HINTS:
                    if h in t:
                        return True
                return False
            # Strip a leading greeting then check the remainder.
            _stripped = re.sub(
                r"^(hi+|hello|helo|heloo|hey|yo|howdy|greetings|good\s+"
                r"(morning|afternoon|evening))[\s,!\.]*", "", norm).strip()
            if _stripped and _stripped != norm and _has_analytical_remainder(
                    f" {_stripped} "):
                return None
            if _has_analytical_remainder(f" {norm} ") and not re.fullmatch(
                    r"(hi|hello|helo|hey|yo|howdy|thanks|thank you|thankyou|"
                    r"thx|ok|okay|bye|goodbye|help)", norm):
                # Longer analytical phrasing (e.g. "overall sales",
                # "thank you for the sales report, now show…") is not pure.
                # Short exact conversational forms are whitelisted below;
                # anything else carrying analytical wording stays analytical.
                _pure_ok = False
                if norm in ("thank you", "thanks", "thankyou", "thx",
                            "hi", "hello", "hey"):
                    _pure_ok = True
                if not _pure_ok:
                    # Allow "thanks + short tail" only when the tail has no
                    # analytical hint (checked above it does) → analytical.
                    return None
            # Exact short forms.
            if norm in ("hi", "hii", "hiii", "hello", "helo", "heloo",
                        "hey", "yo", "howdy", "greetings", "hi there",
                        "hello there", "hey there", "hi aria", "hello aria",
                        "hey aria"):
                return "greeting"
            if re.fullmatch(
                    r"(good\s+morning|good\s+afternoon|good\s+evening)"
                    r"(\s+aria)?", norm):
                return "greeting"
            if norm in ("thanks", "thank you", "thankyou", "thx", "ty",
                        "thanks a lot", "thank you so much",
                        "thank you very much", "many thanks",
                        "thanks so much", "appreciated", "thank u"):
                return "thanks"
            if re.fullmatch(r"thank(s| you).*", norm) and len(words) <= 4 \
                    and not _has_analytical_remainder(f" {norm} "):
                return "thanks"
            if norm in ("bye", "goodbye", "good bye", "see you",
                        "see you soon", "cya", "farewell", "good night"):
                return "bye"
            if norm in ("help", "help me", "help please", "i need help",
                        "what can you do", "what can u do",
                        "what do you do", "how does this work",
                        "how do i use this", "how do i use",
                        "how to use", "guide", "guide me", "show help"):
                return "help"
            if re.search(r"\b(who are you|what are you|your name|about you|"
                         r"who is aria|what is aria|introduce yourself)\b",
                         norm) and len(words) <= 8:
                return "who"
            if norm in ("ok", "okay", "k", "got it", "gotit", "cool",
                        "nice", "great", "perfect", "sure", "yes", "yeah",
                        "yep", "nope", "alright", "all right", "fine",
                        "sounds good", "thank ok"):
                return "ack"
            if norm in ("?", "hi?", "hello?", "hey?"):
                return "greeting"
            return None
        except Exception:
            return None

    def _is_conversational(self, user_goal):
        """True for pure conversational inputs (no SQL may execute)."""
        try:
            return self._conversational_kind(user_goal) is not None
        except Exception:
            return False

    def _conversational_message(self, kind, suggestions):
        """Warm natural reply naming real starter questions (schema-driven)."""
        try:
            _s = [str(s).strip() for s in (suggestions or []) if str(s).strip()]
            _s = _s[:3]
        except Exception:
            _s = []
        if _s:
            if len(_s) >= 2:
                _q = "; ".join(f"'{s}'" for s in _s)
            else:
                _q = f"'{_s[0]}'"
            _invite = f" Try asking something like {_q}."
        else:
            _invite = " Try asking about totals, top items, or trends over time."
        if kind == "thanks":
            return ("You're very welcome! Happy to help with your data."
                    + _invite)
        if kind == "bye":
            return ("Goodbye! I'll be here whenever you want to explore "
                    "your data." + _invite)
        if kind == "help":
            return ("I can answer questions about your connected data — "
                    "totals, rankings, breakdowns, and trends over time."
                    + _invite)
        if kind == "who":
            return ("I'm ARIA, your data analyst — I answer questions "
                    "directly from your connected data." + _invite)
        if kind == "ack":
            return ("Got it! Let me know what you'd like to explore in "
                    "your data." + _invite)
        if kind == "empty":
            return ("Tell me what you'd like to know about your data."
                    + _invite)
        return ("Hi there! I'm ARIA, your data analyst." + _invite)

    def _converse(self, user_goal, output_path="processed_data.json"):
        """Write the ``conversation`` contract without running any SQL.

        Status ``conversation`` is new and frontend-friendly: NOT
        ``success`` with data, NOT ``query_failed``, NOT clarification.
        Carries a warm natural reply plus 2-3 real schema-derived starter
        questions. No SQL executes on this path by construction.
        """
        try:
            kind = self._conversational_kind(user_goal) or "greeting"
        except Exception:
            kind = "greeting"
        try:
            suggestions = self._template_suggestions(3)
        except Exception:
            suggestions = []
        try:
            suggestions = [s for s in (suggestions or []) if s][:3]
        except Exception:
            suggestions = []
        if len(suggestions) < 2:
            # Schema yielded too few: pad with honest answerable generics
            # (never invented specifics, never hardcoded columns).
            for _pad in ("How many records are in the data?",
                         "What are the top items by value?",
                         "How has the main metric changed over time?"):
                if _pad not in suggestions:
                    suggestions.append(_pad)
                if len(suggestions) >= 3:
                    break
        message = self._conversational_message(kind, suggestions)
        original = user_goal
        try:
            if getattr(self, "_active_original_goal", None):
                original = self._active_original_goal
        except Exception:
            pass
        output = {
            "user_goal": original,
            "status": "conversation",
            "conversation_kind": kind,
            "message": message,
            "question": message,
            "goal": {
                "original_question": original,
                "intent": "conversation",
                "analysis_type": "conversational",
            },
            "kpi_alignment": {},
            "join_path": [],
            "goal_ir": {"intent": "conversation", "original_goal": original},
            "data_selection": {"tables": [], "columns": [],
                               "filters": [], "joins": []},
            "analysis_plan": {
                "measures": [], "dimensions": [], "aggregation": [],
                "group_by": [], "order_by": [], "limit": None,
            },
            "sql": None,
            "sql_used": None,
            "execution": {"success": False, "row_count": 0,
                          "execution_time_ms": 0},
            "row_count": 0,
            "data": [],
            "columns": [],
            "suggested_questions": suggestions,
            "warnings": [],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8")
        return output_path

    # ---- _needs_clarification (_needs_clarification) ----

    def _needs_clarification(self, user_goal):
        """Return (True, question) only when the goal cannot be grounded to ANY
        table or column in the connected schema, even after normalization,
        synonym matching, singular/plural handling and analytical-intent
        checks (spec §7).

        The old behavior failed whenever no literal table/column name appeared
        in the question. That was too primitive: a generic analytical question
        ("top brands by model year", "overall sales") must NOT be rejected just
        because the user phrased a concept rather than a physical column.
        """
        # Conversational inputs never clarify — process_goal answers them as
        # status=conversation before reaching here; this guard keeps direct
        # _needs_clarification callers honest too.
        try:
            if self._is_conversational(user_goal):
                return False, None
        except Exception:
            pass
        tokens = self._normalize_tokens(user_goal)
        if not tokens:
            return True, (
                "Please provide a business question, for example "
                "'total sales by month' or 'top customers'."
            )

        goal_l = str(user_goal).lower()

        # 1-0) Pure count questions NEVER clarify (spec §7): "how many
        #    records/orders/rows", "number of X", "count of X" always ground
        #    to COUNT(*) on the most relevant table. Generic, no schema names.
        try:
            if re.search(
                r"\bhow many\b|\bhow much\b|\bnumber of\b|\bcount of\b"
                r"|\bcounts\b|\bnumber of rows\b|\bnumber of records\b",
                goal_l,
            ) or re.search(r"\bcount\b", goal_l):
                return False, None
        except Exception:
            pass

        # 1-0b) Revenue/measure-like questions ground via the measure column
        #    (+ a sampled cell-value filter), never via literal table names:
        #    "revenue for electronics" must not clarify. Generic, no names.
        try:
            if re.search(
                r"\b(revenue|revenues|sales|turnover|income|gmv|aov)\b"
                r"|\b(amount|total|quantity|qty|units|value|price|cost)\b",
                goal_l,
            ):
                return False, None
        except Exception:
            pass

        # 1) Table-level + column-level semantic matching (synonym-aware).
        tables = self._match_tables(goal_l)
        columns = []
        for _score, t in tables[:6]:
            columns += self._match_columns(goal_l, t)
        for t in self.tables:
            columns += self._match_columns(goal_l, t)

        if tables or columns:
            return False, None

        # 2) A genuinely analytical goal (metric concept + operation wording)
        #    must not be pre-blocked just because it does not name a physical
        #    column ("overall sales", "top brands by model year"). Only a vague
        #    goal with neither a grounded subject nor a resolvable metric
        #    concept + operation is ambiguous.
        grounded_subject = bool(tables or columns)
        analytical = self._has_resolvable_analytical_structure(goal_l)
        if analytical:
            return False, None

        # 2b) Pure time/trend phrasing ("monthly trend", "sales over time")
        #    grounds via the schema's date column + measure, not via a literal
        #    column name. Single-table datasets (Walmart Date TEXT) must not be
        #    rejected here; the semantic layer binds the date + measure next.
        trend_words = ("trend", "monthly", "daily", "weekly", "yearly",
                       "over time", "growth", "time series")
        if any(w in goal_l for w in trend_words):
            try:
                has_date = self._first_date_column(list(self.tables.keys() or [])) is not None
            except Exception:
                has_date = False
            if has_date:
                return False, None

        # 3-5) Anti-join / comparison / why patterns only bypass clarification
        #    when SOMETHING is grounded. "why did sales drop" grounds via
        #    sales->orders; but "explain the weather forecast" grounds to
        #    nothing and must clarify instead of hallucinating SQL.
        if not grounded_subject:
            table_names = ", ".join(sorted(self.tables))[:300] if self.tables else "none"
            return True, (
                "I couldn't find a table or column in the connected database that "
                "clearly corresponds to your question. Available data covers: "
                f"{table_names}. Please rephrase using one of those areas (for "
                "example 'total sales by month' or 'top products')."
            )

        # 3) Anti-join / existence patterns: "never", "without", "not", "no"
        anti_patterns = ("never", "without", "not sold", "no orders",
                         "no sales", "no records", "missing", "absent",
                         "not present", "has not", "have not", "did not")
        if any(p in goal_l for p in anti_patterns):
            return False, None

        # 4) Comparison patterns: "compare", "vs", "versus", "difference"
        comparison_patterns = ("compare", " vs ", "versus", "difference between",
                               "how does", "how do", "which is more", "which is less")
        if any(p in goal_l for p in comparison_patterns):
            return False, None

        # 5) "Why" patterns: "why", "what caused", "reason for"
        why_patterns = ("why", "what caused", "reason for", "explain",
                        "what happened", "what made")
        if any(p in goal_l for p in why_patterns):
            return False, None

        # 6) Nothing at all matched: graceful clarification that names what was
        #    looked for, never a raw exception or matching failure.
        table_names = ", ".join(sorted(self.tables))[:300] if self.tables else "none"
        return True, (
            "I couldn't find a table or column in the connected database that "
            "clearly corresponds to your question. Available data covers: "
            f"{table_names}. Please rephrase using one of those areas (for "
            "example 'total sales by month' or 'top products')."
        )

    # ---- _clarify (_clarify) ----

    def _clarify(self, user_goal, question, output_path="processed_data.json"):
        """Write the spec §3 needs_clarification contract without running SQL."""
        output = {
            "status": "needs_clarification",
            "question": question,
            "goal": {
                "original_question": user_goal,
                "intent": self._detect_intent(user_goal),
                "analysis_type": self._detect_analysis_type(user_goal),
            },
            "data_selection": {"tables": [], "columns": [], "filters": [], "joins": []},
            "analysis_plan": {
                "measures": [], "dimensions": [], "aggregation": [],
                "group_by": [], "order_by": [], "limit": None,
            },
            "sql": None,
            "execution": {"success": False, "row_count": 0, "execution_time_ms": None},
            "data": [],
            "suggested_questions": [],
            "warnings": [],
            "message": question,
        }
        Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        return output_path

    # ---- _build_kpi_index (_build_kpi_index) ----

    def _build_kpi_index(self):
        """Keyword -> (aggregate, description) map used for KPI alignment.

        Spec §3: these are EXPLICIT wording hints only. A word is resolved only
        when the user actually says it ("top products by quantity"); it is never
        a silently applied default metric. Generic measure words (quantity,
        amount, value, price) are listed so explicit wording grounds even on
        schemas with no 'sales' column.
        """
        return {
            "total": ("SUM", "aggregate total"),
            "sum": ("SUM", "aggregate total"),
            "amount": ("SUM", "aggregate total"),
            "value": ("SUM", "aggregate total"),
            "quantity": ("SUM", "aggregate total"),
            "qty": ("SUM", "aggregate total"),
            "units": ("SUM", "aggregate total"),
            "price": ("SUM", "aggregate total"),
            "cost": ("SUM", "aggregate total"),
            "count": ("COUNT", "row count"),
            "how many": ("COUNT", "row count"),
            "average": ("AVG", "average value"),
            "avg": ("AVG", "average value"),
            "mean": ("AVG", "average value"),
            "max": ("MAX", "maximum value"),
            "maximum": ("MAX", "maximum value"),
            "min": ("MIN", "minimum value"),
            "minimum": ("MIN", "minimum value"),
            "profit": ("SUM", "profit KPI"),
            "margin": ("SUM", "profit KPI"),
            "revenue": ("SUM", "revenue KPI"),
            "sales": ("SUM", "sales KPI"),
            "growth": ("AVG", "growth KPI"),
        }

    # ---- _detect_metrics_overview (_detect_metrics_overview) ----

    def _detect_metrics_overview(self, user_goal):
        """Return the entity table to summarize when the goal is an open-ended
        'most important metrics for X' style question, else None.

        Deterministic: no LLM. Concrete goals ('total sales by customer') never
        match because they do not use metric/overview vocabulary.
        """
        try:
            if self._is_conversational(user_goal):
                return None
        except Exception:
            pass
        if not re.search(self._OVERVIEW_RE, user_goal):
            return None
        goal_tokens = self._normalize_tokens(user_goal)
        for table_name in self._get_relevant_tables(user_goal, max_tables=3):
            if self._natural_label(table_name).replace(" ", "") in goal_tokens:
                return table_name
            if table_name.lower() in goal_tokens:
                return table_name
        return None

    # ---- _build_metrics_overview (_build_metrics_overview) ----

    def _qt(self, table_name):
        """Schema-aware table reference (sales.employee -> "sales"."employee").
        Falls back to the bare quoted segment when the sql-mixin quoter is
        unavailable (partial compositions in tests)."""
        try:
            _qtab = getattr(self, "_qtab", None)
            if callable(_qtab):
                return _qtab(table_name)
        except Exception:
            pass
        try:
            from core.db import base_table as _bt
            return f'"{_bt(table_name)}"'
        except Exception:
            return f'"{table_name}"'

    def _build_metrics_overview(self, entity_table):
        """Deterministic metric set for an entity: entity count + one count and
        one sum per fact table that references it. Returns [(label, sql)]."""
        entity_label = self._natural_label(entity_table)
        metrics = [
            (f"Total {self._plural_label(entity_table)}",
             f'SELECT COUNT(*) FROM {self._qt(entity_table)}'),
        ]
        for table_name, info in self.tables.items():
            if table_name == entity_table:
                continue
            for fk in info.get("foreign_keys", []):
                if fk["referenced_table"] != entity_table:
                    continue
                t_label = self._natural_label(table_name)
                metrics.append(
                    (f"Total {self._plural_label(table_name)}",
                     f'SELECT COUNT(*) FROM {self._qt(table_name)}')
                )
                measure_col = self._pick_measure_column(table_name)
                if measure_col:
                    col_label = self._natural_label(measure_col)
                    if col_label in self._GENERIC_TOTAL_WORDS:
                        label = f"Total {t_label} value"
                    else:
                        label = f"Total {col_label}"
                    metrics.append(
                        (label, f'SELECT COALESCE(SUM("{measure_col}"), 0) '
                                f'FROM {self._qt(table_name)}')
                    )
                break
        return metrics

    # ---- _process_metrics_overview (_process_metrics_overview) ----

    def _process_metrics_overview(self, user_goal, entity_table, output_path):
        """Answer an open-ended 'most important metrics for X' goal with a
        deterministic, always-working overview instead of a clarification."""
        entity_label = self._natural_label(entity_table)
        metrics = self._build_metrics_overview(entity_table)
        rows, sql_parts = [], []
        t0 = time.time()
        try:
            with self.engine.connect() as conn:
                for label, sql in metrics:
                    try:
                        value = self._execute_guarded(conn, sql).scalar()
                        rows.append({"metric": label, "value": value})
                        sql_parts.append(sql)
                    except Exception as exc:
                        logging.warning(
                            "Overview metric failed (%s): %s", label, exc
                        )
        except Exception as exc:
            return self._graceful_failure(user_goal, output_path, exc)
        execution_ms = int((time.time() - t0) * 1000)
        if not rows:
            return self._graceful_failure(
                user_goal, output_path,
                RuntimeError("Could not compute an overview for this goal."),
            )

        output = {
            "user_goal": self._active_original_goal or user_goal,
            "kpi_alignment": {
                "kpis": [{"aggregate": None, "description": "metrics overview",
                          "match": None}],
                "dimensions": [entity_table],
            },
            "join_path": [entity_table],
            "sql_used": "; ".join(sql_parts) or None,
            "row_count": len(rows),
            "data": rows,
            "message": f"Here's an overview of the most important metrics "
                       f"for {entity_label}.",
            "preprocessing": {"notes": ["Deterministic overview (no LLM)."]},
            "missing_values_handled": "not applicable (counts/sums)",
            "timestamp": datetime.now().isoformat(),
        }
        output.update(self._build_output_contract(
            user_goal, output["kpi_alignment"], [entity_table],
            sql_parts[0] if sql_parts else None, rows,
            success=True, message=output["message"], status="success",
            warnings=self._typo_warnings, execution_time_ms=execution_ms,
        ))
        output["sql"] = sql_parts
        output["data_selection"]["columns"] = [r["metric"] for r in rows]
        output["analysis_plan"]["measures"] = [r["metric"] for r in rows]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Goal Agent overview done! Saved to %s", output_path)
        return output_path

    # ---- map_goal_to_kpi (map_goal_to_kpi) ----

    def map_goal_to_kpi(self, user_goal):
        """Return detected KPIs + dimensions for a user goal."""
        # Current question only: history keywords must not become KPIs.
        try:
            _cur = self._current_question_text(user_goal)
        except Exception:
            _cur = str(user_goal or "")
        goal_lower = _cur.lower()
        kpis = []
        dimensions = set()
        seen = set()

        for keyword, (agg, label) in self.kpi_index.items():
            if keyword in goal_lower and (label, agg) not in seen:
                seen.add((label, agg))
                kpis.append({"aggregate": agg, "description": label, "match": keyword})
        if not kpis:
            # Elliptical follow-up ("and which category sells most?"): lend
            # the history's measure words instead of the generic fallback.
            try:
                _hist = self._history_text(user_goal)
                if _hist.strip() and self._is_elliptical_followup_text(_cur):
                    _hl = _hist.lower()
                    for keyword, (agg, label) in self.kpi_index.items():
                        if keyword in _hl and (label, agg) not in seen:
                            seen.add((label, agg))
                            kpis.append({"aggregate": agg, "description": label,
                                         "match": keyword, "inherited": True})
            except Exception:
                pass
        if not kpis:
            kpis = [{"aggregate": "SUM", "description": "general KPI", "match": None}]

        for table_name in self.tables:
            if table_name.lower() in goal_lower:
                dimensions.add(table_name)

        return {"kpis": kpis, "dimensions": sorted(dimensions)}

    # ---- _enrich_kpi_map (_enrich_kpi_map) ----

    def _enrich_kpi_map(self, kpi_map, ir):
        """Bind weak KPI hints to real schema columns (spec §17): the KPI map
        must never rely on a naive universal mapping — every hint that resolves
        to a measure column records the resolved expression/aggregation. The
        'general KPI' fallback binds to the first resolved IR metric so an
        inferred COUNT or a computed measure still reaches the SQL prompt.

        Also propagates resolved IR dimensions into kpi_map['dimensions'] so
        the LLM prompt always includes the actual breakdown columns (e.g.
        'region, product' for 'Compare segments breakdown')."""
        if not ir:
            return kpi_map
        resolved = []
        for m in ir.get("metrics", []):
            if m.get("resolved_expression") or m.get("resolved_column"):
                resolved.append(m)
        by_concept = {m["concept"]: m for m in resolved}
        for k in kpi_map.get("kpis", []):
            r = by_concept.get(k["description"])
            if not r and resolved and k.get("match") is None:
                r = resolved[0]
            if r:
                k["resolved_column"] = r.get("resolved_expression") or r.get("resolved_column")
                if r.get("aggregation"):
                    k["aggregate"] = r["aggregation"]
                if r.get("computed_measure"):
                    k["computed_measure"] = True
                if r.get("inferred"):
                    k["inferred"] = True
        # Propagate IR dimensions to kpi_map so the LLM prompt includes them.
        ir_dims = [d.get("resolved_column") for d in ir.get("dimensions", [])
                   if d.get("resolved_column")]
        if ir_dims:
            existing = set(kpi_map.get("dimensions", []))
            for d in ir_dims:
                if d not in existing:
                    kpi_map.setdefault("dimensions", []).append(d)
        return kpi_map

    # ---- _detect_aggregation (_detect_aggregation) ----

    def _detect_aggregation(self, goal):
        try:
            g = self._strip_history_marker(goal).lower()
        except Exception:
            g = str(goal).lower()
        if re.search(r"\b(total|sum|amount of|sum of)\b", g):
            return "SUM"
        if re.search(r"\b(count|how many|number of|total number)\b", g):
            return "COUNT"
        if re.search(r"\b(average|avg|mean)\b", g):
            return "AVG"
        if re.search(r"\b(max|maximum|largest|biggest)\b", g):
            return "MAX"
        if re.search(r"\b(min|minimum|smallest)\b", g):
            return "MIN"
        return None

    # ---- _detect_ranking (_detect_ranking) ----

    def _detect_ranking(self, goal):
        """Best-effort ranking extraction (top/bottom N, or a one-sided order)."""
        g = str(goal).lower()
        direction = "asc" if any(k in g for k in
                                 ("bottom", "lowest", "least", "worst")) else "desc"
        top_m = re.search(r"\b(top|bottom)\s+(\d+)\b", g)
        if top_m:
            return {"by": None, "direction": direction, "limit": int(top_m.group(2))}
        if any(k in g for k in ("top", "best", "highest", "most", "leader",
                                "worst", "lowest", "least", "bottom")):
            return {"by": None, "direction": direction, "limit": None}
        return None

    # ---- _extract_goal_time (_extract_goal_time) ----

    def _extract_goal_time(self, goal):
        """Time constraints mentioned by the user (year / period granularity)."""
        g = str(goal).lower()
        year = re.search(r"\b(19\d{2}|20\d{2})\b", g)
        if year:
            return {"constraint": "year", "value": int(year.group(1)),
                    "column": None, "granularity": None}
        for period in ("monthly", "quarterly", "yearly", "daily", "weekly"):
            if period in g:
                return {"constraint": "period", "value": None, "column": None,
                        "granularity": period[:-2]}
        return None

    # ---- _extract_goal_filters (_extract_goal_filters) ----

    def _extract_goal_filters(self, goal):
        """Conservative deterministic filter extraction (spec §11). Only
        constraints we can name without inventing data are captured; the rest
        is left to the SQL and mirrored back from it in the contract."""
        filters = []
        year = re.search(r"\b(19\d{2}|20\d{2})\b", goal)
        if year:
            y = int(year.group(1))
            filters.append({"column": None, "operator": ">=", "value": f"{y}-01-01"})
            filters.append({"column": None, "operator": "<=", "value": f"{y}-12-31"})
        return filters

    # ---- _goal_asks_for_comparison (_goal_asks_for_comparison) ----

    def _goal_asks_for_comparison(self, goal):
        g = str(goal).lower()
        return any(k in g for k in ("compare", "comparison", "versus",
                                    "difference between", "among"))

    # ---- _build_goal_ir (_build_goal_ir) ----

    def _build_goal_ir(self, goal):
        """Build a canonical, serializable Goal IR (spec §1/§2) BEFORE any SQL.

        The IR is intentionally deterministic (no LLM): intent, aggregation,
        metric concepts, dimensions (concepts only), filters, time, ranking and
        comparison. `_resolve_semantics` later binds the concepts to real schema
        columns/tables so the LLM is no longer the authority on the plan.
        """
        # Multi-turn calls prepend conversation history ("Previous question:…"
        # / "Conversation context:…Q:…Current question:…"). The IR must reflect
        # ONLY the current question — otherwise keywords from earlier turns
        # ("how many", "total", "average") poison this turn's intent and the
        # plan answers a previous question. History still reaches the LLM
        # prompt via the unmodified goal text upstream.
        _full_goal = str(goal or "")
        try:
            goal = self._current_question_text(_full_goal)
        except Exception:
            goal = self._strip_history_marker(_full_goal)
        try:
            _history = self._history_text(_full_goal)
        except Exception:
            _history = ""
        goal_l = str(goal).lower()
        ir = {
            "original_goal": goal,
            "intent": self._detect_intent(goal),
            "analysis_type": self._detect_analysis_type(goal),
            "operation": None,
            "aggregation": None,
            "metrics": [],
            "measures": [],
            "dimensions": [],
            "filters": [],
            "time": None,
            "ranking": None,
            "comparison": None,
            "ordering": [],
            "required_tables": [],
            "join_plan": [],
            "confidence": 0.5,
            "requires_sql": True,
        }
        agg = self._detect_aggregation(goal)
        if agg:
            ir["aggregation"] = agg
        seen_metrics = set()
        for keyword, (aggregate, description) in self.kpi_index.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", goal_l):
                metric_key = (description, aggregate)
                if metric_key not in seen_metrics:
                    seen_metrics.add(metric_key)
                    ir["metrics"].append({
                        "concept": description,
                        "aggregation": None if keyword == "growth" else aggregate,
                        "expression": None,
                        "resolved_expression": None,
                        "resolved_table": None,
                        "resolved_column": None,
                    })
        # One question, one primary aggregation: wording can trigger several
        # aggregates at once ("how many total records" hits COUNT + SUM,
        # "average amount" hits AVG + SUM) which makes the semantic plan
        # unsatisfiable and the answer degrades. Explicit function words
        # outrank generic measure words (COUNT > AVG > MAX/MIN > SUM), unless
        # the goal joins measures with "and" ("total and average amount"
        # genuinely wants both).
        try:
            _aggs = {m.get("aggregation") for m in ir["metrics"]}
            if len(_aggs) > 1 and " and " not in f" {goal_l} ":
                _order = ("COUNT", "AVG", "MAX", "MIN", "SUM")
                _triggers = {
                    "COUNT": ("how many", "number of", "count"),
                    "AVG": ("average", "avg", "mean"),
                    "MAX": ("max", "maximum", "highest", "largest"),
                    "MIN": ("min", "minimum", "lowest", "smallest"),
                    "SUM": ("total", "sum"),
                }
                _hit = {a for a, words in _triggers.items()
                        if any(w in goal_l for w in words)}
                for _winner in _order:
                    if _winner in _hit and _winner in _aggs:
                        ir["metrics"] = [m for m in ir["metrics"]
                                         if m.get("aggregation") == _winner]
                        break
        except Exception:
            pass
        # COUNT-explicit current questions ('how many records', 'number of
        # rows', 'count of') DROP every metric from history/current wording
        # (not just note it): semantic validation must never demand an
        # aggregation over an unresolvable measure for a pure count.
        # A turn naming its own aggregation never inherits history measures.
        try:
            _count_explicit = self._is_count_explicit_text(goal)
        except Exception:
            _count_explicit = False
        if _count_explicit:
            ir["aggregation"] = "COUNT"
            ir["metrics"] = [{
                "concept": "row count",
                "aggregation": "COUNT",
                "expression": None,
                "resolved_expression": None,
                "resolved_table": None,
                "resolved_column": None,
                "inferred": False,
                "confidence": 0.95,
                "reason": "current question explicitly asks for a count; "
                          "history measures dropped",
            }]
        elif _history.strip():
            # Elliptical follow-up ("and which category sells most?"): lend
            # the history's measure/aggregation words. Non-elliptical turns
            # keep their current-only IR.
            try:
                _elliptical = self._is_elliptical_followup_text(goal)
            except Exception:
                _elliptical = False
            if _elliptical:
                try:
                    _hist_l = _history.lower()
                    _have = {(m.get("concept"), m.get("aggregation"))
                             for m in ir["metrics"]}
                    for keyword, (aggregate, description) in (
                            self.kpi_index or {}).items():
                        try:
                            if re.search(r"\b" + re.escape(keyword) + r"\b",
                                         _hist_l):
                                _key = (description,
                                        None if keyword == "growth" else aggregate)
                                if _key not in _have:
                                    _have.add(_key)
                                    ir["metrics"].append({
                                        "concept": description,
                                        "aggregation": None if keyword == "growth"
                                        else aggregate,
                                        "expression": None,
                                        "resolved_expression": None,
                                        "resolved_table": None,
                                        "resolved_column": None,
                                        "inherited": True,
                                        "confidence": 0.55,
                                        "reason": "elliptical follow-up; measure "
                                                  "inherited from conversation history",
                                    })
                        except Exception:
                            continue
                    if ir.get("aggregation") is None:
                        try:
                            _h_agg = self._detect_aggregation(_history)
                            if _h_agg:
                                ir["aggregation"] = _h_agg
                        except Exception:
                            pass
                except Exception:
                    pass
        if any(k in goal_l for k in ("growth", "trend", "over time", "increase",
                                     "decrease", "decline", "declined",
                                     "declining", "drop", "dropped",
                                     "dropping", "fall", "fell", "falling",
                                     "falls", "change")) or re.search(
                r"\b(decline|declined|declining|drop|dropped|dropping|"
                r"fall|fell|falling|falls|decrease|decreased|decreasing)\b",
                goal_l):
            ir["intent"] = "trend"
            ir["analysis_type"] = "trend_analysis"
        ir["ranking"] = self._detect_ranking(goal)
        ir["time"] = self._extract_goal_time(goal)
        ir["filters"] = self._extract_goal_filters(goal)
        if self._goal_asks_for_comparison(goal):
            ir["comparison"] = {"dimension": None, "segments": []}

        # Spec §3: NEVER assume the metric, but when a ranking/aggregation goal
        # states NO measure at all ("top brands by model year"), a COUNT of the
        # related records is a safe generic default. It must be represented as
        # an INFERRED metric with low-medium confidence, never as certainty.
        # The schema-aware metric resolver may still upgrade it to a real
        # measure column when one clearly exists.
        if not ir["metrics"] and ir["intent"] in (
                "ranking", "aggregation", "summary", "distribution", "trend"):
            ir["metrics"].append({
                "concept": "row count",
                "aggregation": "COUNT",
                "expression": None,
                "resolved_expression": None,
                "resolved_table": None,
                "resolved_column": None,
                "inferred": True,
                "confidence": 0.6,
                "reason": "no explicit measure; ranked/aggregated by count of "
                          "related records",
            })

        # Spec §9: a database-independent operation label for the contract.
        operation = ir["intent"]
        if ir["ranking"]:
            operation = "ranking"
        elif ir["comparison"]:
            operation = "comparison"
        elif ir["aggregation"] == "COUNT":
            operation = "count"
        elif ir["aggregation"]:
            operation = ir["aggregation"].lower()
        elif ir["intent"] == "trend":
            operation = "trend"
        ir["operation"] = operation
        # Additive general-analytics capabilities (sections 1-8): one question
        # maps to multiple capabilities via classifier + schema grounding.
        # Never overrides the deterministic operation above (SQL planning is
        # unchanged); purely descriptive for the stable result contract.
        try:
            from goal_agent.analytics_intent import classify_analytics as _cls_a
            _c = _cls_a(goal)
            ir["capabilities"] = _c.get("capabilities", [ir["intent"]])
            ir["interpreted_intent"] = _c.get("primary", ir["intent"])
        except Exception:
            ir["capabilities"] = [ir["intent"]]
            ir["interpreted_intent"] = ir["intent"]
        return ir

    # ---- _detect_intent (_detect_intent) ----

    def _detect_intent(self, user_goal):
        """Heuristic intent classification (no LLM call) for the goal contract."""
        try:
            g = self._strip_history_marker(user_goal).lower()
        except Exception:
            g = str(user_goal).lower()
        if re.search(r"\b(top\b|best\b|worst\b|highest\b|lowest\b|most\b|least\b|rank\b|leader\b|bottom\b)", g):
            return "ranking"
        if re.search(r"\b(compare|comparison|versus|difference|among|between)\b", g):
            return "comparison"
        if re.search(r"\b(trend|over\s+time|monthly|daily|weekly|yearly|growth|decline|declined|declining|increase|decrease|decreased|decreasing|change|drop|dropped|dropping|fall|fell|falling|falls|dip|down)\b", g):
            return "trend"
        if re.search(r"\b(distribution|breakdown|per\b|by\b|count)\b", g):
            return "distribution"
        return "summary"

    # ---- _detect_analysis_type (_detect_analysis_type) ----

    def _detect_analysis_type(self, user_goal):
        """Heuristic analysis-type classification for the goal contract."""
        try:
            g = self._strip_history_marker(user_goal).lower()
        except Exception:
            g = str(user_goal).lower()
        if any(k in g for k in ("predict", "forecast", "future", "expected",
                                "will ", "what if", "what-if", "scenario",
                                "simulate", "projection")):
            return "predictive"
        if any(k in g for k in ("why", "cause", "reason", "driver", "because",
                                "anomaly", "outlier", "hypothes")):
            return "diagnostic"
        if any(k in g for k in ("recommend", "should", "action", "improve",
                                "optimize", "suggest", "what to do",
                                "next step")):
            return "prescriptive"
        return "descriptive"

