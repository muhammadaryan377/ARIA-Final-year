"""Goal Agent orchestration: builds the engine, runs the linear / LangGraph pipelines and composes all components."""

import difflib
import json
import os
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None


from .constants import ConstantsMixin
from .semantics import SemanticMixin
from .intent import IntentMixin
from .schema import SchemaMixin
from .grounding import GroundingMixin
from .sql import SqlMixin
from .sql_contract import SqlContractMixin
from .suggestions import SuggestionsMixin
from .data import DataMixin

DEFAULT_DB_URI = (
    "postgresql://{user}:{password}@{host}:{port}/{dbname}"
).format(
    user=os.getenv("DB_USER", ""),
    password=os.getenv("DB_PASSWORD", ""),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "postgres"),
)

class GoalAgent(ConstantsMixin, SemanticMixin, IntentMixin, SchemaMixin,
                GroundingMixin, SqlMixin, SqlContractMixin, SuggestionsMixin,
                DataMixin):
    # ---- __init__ (__init__) ----

    def __init__(self, schema_json_path=None, db_uri=None,
                  provider=None, api_key=None, dialect="postgresql", overrides=None,
                 schemas=None):
        if schema_json_path:
            schema_file = Path(schema_json_path)
        else:
            # Absolute project-relative default so the process CWD never matters.
            schema_file = SCHEMA_DIR / "schema_mapping_latest.json"
        if not schema_file.exists():
            # Absolute project-relative fallback so the process CWD never matters.
            fallback = BASE_DIR / "schema_mapping.json"
            if fallback.exists():
                schema_file = fallback
        if not schema_file.exists():
            available = sorted(p.name for p in SCHEMA_DIR.glob("schema_mapping_*.json"))
            raise FileNotFoundError(
                f"Schema mapping not found: {schema_file}. "
                + (f"Available mappings: {', '.join(available)}."
                   if available
                   else "Run the schema extraction first (python schema_agent.py <database>).")
            )
        with open(schema_file, "r", encoding="utf-8") as f:
            self.full_schema = json.load(f)

        self.tables = self._normalize_schema(self.full_schema)
        self.uncertain_edge_keys = self._collect_uncertain_edges(self.full_schema)
        self.relationship_graph = self._build_relationship_graph()
        self.relationship_edges = self._build_relationship_edges()
        # Journey E: user corrections outrank inference in all future planning.
        self.user_overrides = overrides or {}
        self._apply_user_overrides()
        self.dialect = dialect
        self.schema = (self.full_schema.get("schema") or "public").strip()
        _ms = schemas or (self.full_schema.get("schemas") if isinstance(
            self.full_schema, dict) else None) or []
        self.schemas = [str(s).strip() for s in _ms if str(s or "").strip()]
        self.engine = self._build_engine(db_uri)
        if isinstance(provider, str):
            self.llm = create_provider(provider=provider, api_key=api_key)
        elif provider is not None:
            self.llm = provider
        else:
            logger.warning("GoalAgent created with provider=None – failing closed to prevent silent local fallback")
            raise ValueError(
                "GoalAgent requires an explicit provider. Pass the LLMProvider "
                "object returned by require_data() or session['provider']."
            )
        self.kpi_index = self._build_kpi_index()
        # Universal Data Intelligence Layer (n.txt 11/12): reuse the mapping's
        # contract when this extraction already carries it; otherwise derive
        # the same contract from the normalized tables so every Goal Agent —
        # including ones loaded from older/stale mappings — exposes it.
        self.data_intelligence = (self.full_schema.get("data_intelligence")
                                  if isinstance(self.full_schema, dict) else None)
        self.capabilities = (self.full_schema.get("capabilities")
                             if isinstance(self.full_schema, dict) else None)
        if not isinstance(self.data_intelligence, dict):
            try:
                from core.data_intelligence import (
                    build_data_intelligence as _build_intel,
                    detect_capabilities as _detect_caps,
                )
                _rel_edges = [
                    {"source_table": e.get("left_table"),
                     "source_column": e.get("left_column"),
                     "target_table": e.get("right_table"),
                     "target_column": e.get("right_column"),
                     "type": e.get("relationship_type", "declared")}
                    for e in (self.relationship_edges or [])
                ]
                _adapter = {
                    "tables": {
                        t: {"columns": [{"column": c, "data_type": (mi or {}).get("data_type", "TEXT"),
                                        "nullable": (mi or {}).get("nullable", True)}
                                       for c, mi in (info.get("columns", {}) or {}).items()],
                            "primary_key": list(info.get("primary_key", []) or []),
                            "inferred_composite_primary_key": list(
                                info.get("composite_primary_key", []) or []),
                            "row_count": info.get("row_count"),
                            "empty": bool(info.get("empty")),
                            "foreign_keys": list(info.get("foreign_keys", []) or [])}
                        for t, info in (self.tables or {}).items()
                    },
                    "declared_relationships": [],
                    "inferred_relationships": [],
                    "relationship_edges": _rel_edges,
                }
                self.data_intelligence = _build_intel(_adapter)
                self.capabilities = _detect_caps(self.data_intelligence)
            except Exception:
                logger.debug("GoalAgent data intelligence derivation failed", exc_info=True)
                self.data_intelligence = {}
                self.capabilities = {"capabilities": {}, "suitable_analyses": []}
        self._row_counts_cache = None
        self.preprocessing_report = None
        self._last_execution_ms = None
        self._last_semantic_warnings = []
        self._schema_word_tokens = self._build_schema_word_tokens()
        self._active_original_goal = None
        self._active_current_question = None
        self._typo_warnings = []

    # ---- _build_engine (_build_engine) ----

    def _build_engine(self, db_uri):
        """Build the SQLAlchemy engine. For non-public PostgreSQL schemas the
        search_path is applied via connect_args (survives pool resets), so bare
        table names from the schema mapping resolve correctly.

        connect_timeout + pool_pre_ping bound the initial connect so a
        slow/unreachable database fails fast instead of hanging the request.

        Defense in depth: a missing/empty db_uri raises instead of silently
        falling back to DEFAULT_DB_URI (which would query the wrong
        database). Callers needing the local-dev default must pass
        DEFAULT_DB_URI explicitly.
        """
        if not db_uri or (isinstance(db_uri, str) and not db_uri.strip()):
            raise ValueError("No database connection in session — reconnect first")
        uri = db_uri
        if self.dialect == "postgresql":
            args: dict = {"connect_timeout": 10}
            try:
                from core.db import search_path_connect_args as _spa
                _schemas = list(getattr(self, "schemas", []) or [])
                if (not _schemas or _schemas == ["public"]) and getattr(
                        self, "schema", "") not in (None, "", "public"):
                    _schemas = [self.schema]
                args.update(_spa(self.dialect, _schemas))
            except Exception:
                if self.schema and self.schema != "public":
                    args["options"] = f"-csearch_path={self.schema}"
            return create_engine(uri, connect_args=args, pool_pre_ping=True)
        if self.dialect == "mysql":
            return create_engine(uri, connect_args={"connect_timeout": 10}, pool_pre_ping=True)
        return create_engine(uri, pool_pre_ping=True)

    # ---- process_goal (process_goal) ----

    def process_goal(self, user_goal, output_path="processed_data.json"):
        # The contract writers below target `output_path` (e.g. a per-user path
        # under artifacts/processed_data/). Create the parent directory so the
        # API route never crashes on a missing folder for a new user.
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            self._active_original_goal = user_goal
            # DEFECT 1: pure conversational inputs (greetings, thanks, bye,
            # help, 'who are you', empty) answer WITHOUT typo correction,
            # overview detection, clarification, or any SQL. Checked on the
            # CURRENT question only so multi-turn history never turns 'hi'
            # into a data answer. Schema-agnostic, no table/column names.
            try:
                if self._is_conversational(user_goal):
                    return self._converse(user_goal, output_path)
            except Exception:
                pass
            corrected_goal, corrections = self._correct_goal(user_goal)
            # A conversational turn that only became visible after typo
            # correction (e.g. 'helo' -> greeting) is still conversational:
            # never let it fall through to data.
            try:
                if self._is_conversational(corrected_goal):
                    return self._converse(user_goal, output_path)
            except Exception:
                pass
            self._typo_warnings = [
                f"Did you mean '{suggestion}' for '{typo}' in your question?"
                for typo, suggestion in corrections
            ]
            entity = self._detect_metrics_overview(corrected_goal)
            if entity is not None:
                logger.info(
                    "Open-ended metrics goal -> deterministic overview for '%s'.",
                    entity,
                )
                return self._process_metrics_overview(
                    corrected_goal, entity, output_path
                )
            needs, question = self._needs_clarification(corrected_goal)
            if needs:
                if corrections:
                    suggestions = ", ".join(f"'{suggestion}'" for _, suggestion in corrections)
                    question += f" Did you mean: {suggestions}?"
                logger.info("Goal is ambiguous; returning needs_clarification.")
                return self._clarify(user_goal, question, output_path)
            if LANGGRAPH_AVAILABLE and StateGraph is not None and END is not None:
                return self._process_goal_langgraph(corrected_goal, output_path)
            return self._process_goal_linear(corrected_goal, output_path)
        except Exception as exc:
            if isinstance(exc, (MemoryError, RecursionError, OverflowError)):
                logger.critical("Fatal exception in process_goal: %s", exc, exc_info=True)
                raise
            return self._graceful_failure(user_goal, output_path, exc)

    # ---- _process_goal_linear (_process_goal_linear) ----

    def _process_goal_linear(self, user_goal, output_path="processed_data.json"):
        logger.info("Goal Agent: '%s'", user_goal)

        kpi_map = self.map_goal_to_kpi(user_goal)
        logger.info("Aligned KPIs: %s | dimensions: %s", kpi_map['kpis'], kpi_map['dimensions'])

        ir = self._build_goal_ir(user_goal)
        relevant_tables = self._get_relevant_tables(user_goal)
        join_path = self._determine_join_path(relevant_tables)
        ir = self._resolve_semantics(user_goal, ir, join_path)
        kpi_map = self._enrich_kpi_map(kpi_map, ir)
        join_plan = self._plan_joins(self._required_tables_from_ir(ir, join_path))
        if join_plan["edges"]:
            join_path = join_plan["nodes"]
        ir["join_plan"] = join_plan["edges"]
        logger.info("Join path: %s", join_path)

        raw_sql = self._generate_sql(user_goal, join_path, kpi_map, ir=ir, join_plan=join_plan)
        logger.info("Generated SQL: %s", raw_sql)
        # Post-generation gate (n.txt item 2): log validation BEFORE execution.
        pre_issues = self._validate_generated_sql(raw_sql)
        if pre_issues:
            logger.warning("Post-generation validation issues: %s", pre_issues)

        data_rows, final_sql, note = self._execute_with_retry(
            user_goal, join_path, kpi_map, raw_sql, ir=ir, join_plan=join_plan
        )
        cleaned_data = self._clean_data(data_rows)
        # Internal diagnostic trace (P3 §31, internal only): full contract
        # fields; legacy keys (exec_ms/shape/viz) kept for compatibility.
        try:
            _cols = sorted({m.get("resolved_column") for m in (ir or {}).get("metrics", []) if m.get("resolved_column")} |
                           {d.get("resolved_column") for d in (ir or {}).get("dimensions", []) if d.get("resolved_column")})
            _shape = {"rows": len(cleaned_data),
                      "columns": list(cleaned_data[0].keys()) if cleaned_data else []}
            self._log_pipeline_diagnostic({
                "request_id": getattr(self, "_active_request_id", None),
                "session_id": getattr(self, "_active_session_id", None),
                "dataset_id": getattr(self, "_active_dataset_id", None),
                "question": self._active_original_goal or user_goal,
                "tables": list(join_path),
                "columns": _cols,
                "intent": (ir or {}).get("intent"),
                "sql": final_sql,
                "validation": pre_issues or [],
                "exec_status": "failed" if (note and note.get("failed")) else "ok",
                "exec_ms": self._last_execution_ms,
                "shape": _shape,
                "result_shape": _shape,
                "context": {"join_plan": (ir or {}).get("join_plan", [])},
                "model": getattr(getattr(self, "llm", None), "model", None) or getattr(self, "_provider_name", None),
                "latency_ms": self._last_execution_ms,
                "final_status": "query_failed" if (note and note.get("failed")) else "success",
                "viz": "deferred to /api/ask chart_spec",
                "answer": (note or {}).get("message"),
            })
        except Exception:
            pass

        failed = bool(note and note.get("failed"))
        output = {
            "user_goal": self._active_original_goal or user_goal,
            "kpi_alignment": kpi_map,
            "join_path": join_path,
            "goal_ir": ir,
            "sql_used": final_sql,
            "row_count": len(cleaned_data),
            "data": cleaned_data,
            "message": (note or {}).get("message") if note else None,
            "preprocessing": self.preprocessing_report,
            "missing_values_handled": "preserved (NULLs kept as-is)",
            "timestamp": datetime.now().isoformat(),
        }
        output.update(self._build_output_contract(
            user_goal, kpi_map, join_path, final_sql, cleaned_data,
            success=(not failed), message=output["message"],
            status="query_failed" if failed else "success",
            warnings=self._typo_warnings + self._warnings_for_sql(final_sql)
                    + self._last_semantic_warnings,
            execution_time_ms=self._last_execution_ms,
        ))
        # Additive diagnostics envelope (n.txt item 2): never breaks the
        # existing contract; the UI ignores unknown keys.
        output["diagnostics"] = getattr(self, "_last_pipeline_trace", None) or {}

        out_path = Path(output_path)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        logger.info("Goal Agent done! Saved to %s", output_path)
        return output_path

    # ---- _process_goal_langgraph (_process_goal_langgraph) ----

    def _process_goal_langgraph(self, user_goal, output_path):
        def load_schema(state):
            state["kpi_map"] = self.map_goal_to_kpi(state["user_goal"])
            state["relevant_tables"] = self._get_relevant_tables(state["user_goal"])
            state["join_path"] = self._determine_join_path(state["relevant_tables"])
            return state

        def understand_goal(state):
            state["goal_ir"] = self._build_goal_ir(state["user_goal"])
            return state

        def resolve_semantics(state):
            state["goal_ir"] = self._resolve_semantics(
                state["user_goal"], state["goal_ir"], state["join_path"]
            )
            state["kpi_map"] = self._enrich_kpi_map(state["kpi_map"], state["goal_ir"])
            return state

        def plan_joins(state):
            state["join_plan"] = self._plan_joins(
                self._required_tables_from_ir(state.get("goal_ir"), state["join_path"])
            )
            if state["join_plan"]["edges"]:
                state["join_path"] = state["join_plan"]["nodes"]
            if state.get("goal_ir"):
                state["goal_ir"]["join_plan"] = state["join_plan"]["edges"]
            return state

        def generate_sql(state):
            state["raw_sql"] = self._generate_sql(
                state["user_goal"], state["join_path"], state["kpi_map"],
                ir=state.get("goal_ir"), join_plan=state.get("join_plan"),
            )
            state["repair_attempts"] = int(state.get("repair_attempts") or 0)
            return state

        def validate_sql(state):
            """Validation gate 1: SQL must be read-only + schema-grounded.

            On failure (and attempts left) the error text is fed back into
            state so generate_sql repairs instead of executing a bad query.
            """
            import logging as _lg
            _log = _lg.getLogger(__name__)
            sql = state.get("raw_sql") or ""
            issues = []
            try:
                issues = self._validate_generated_sql(sql) or []
            except Exception as exc:
                issues = [str(exc)[:200]]
            # Semantic plan check (IR + join plan) — same gate as linear path.
            try:
                ir = state.get("goal_ir")
                if ir and not issues:
                    sem_issues, _w = self._semantic_validate_sql(
                        ir, sql, state.get("join_path") or [],
                        state.get("join_plan"))
                    issues = list(sem_issues or [])
            except Exception:
                pass
            state["sql_issues"] = issues
            if issues:
                _log.warning("LangGraph validate_sql failed (attempt %s): %s",
                             state.get("repair_attempts"), issues[:3])
            return state

        def repair_sql(state):
            """LLM repair pass: re-generate with the validation errors as context."""
            attempts = int(state.get("repair_attempts") or 0) + 1
            state["repair_attempts"] = attempts
            issues = state.get("sql_issues") or []
            try:
                fix_prompt = (
                    "The SQL below failed validation:\n"
                    + "\n".join("- " + str(i) for i in issues[:5])
                    + "\n\nUser goal:\n" + str(state.get("user_goal"))
                    + "\n\nBroken SQL:\n" + str(state.get("raw_sql"))
                    + "\n\nReturn only the corrected read-only SELECT query. "
                      "No explanations, no markdown."
                )
                raw = self.llm.complete("sql", fix_prompt, temperature=0.1,
                                        num_predict=500, timeout=self._REPAIR_TIMEOUT)
                state["raw_sql"] = self._clean_sql(raw)
            except Exception:
                pass
            return state

        def execute_sql(state):
            rows, final_sql, note = self._execute_with_retry(
                state["user_goal"], state["join_path"], state["kpi_map"],
                state["raw_sql"],
                ir=state.get("goal_ir"), join_plan=state.get("join_plan"),
            )
            state["rows"] = self._clean_data(rows)
            state["final_sql"] = final_sql
            state["_failed"] = bool(note and note.get("failed"))
            state["message"] = (note or {}).get("message") if note else None
            state["preprocessing"] = self.preprocessing_report
            return state

        def validate_result(state):
            """Validation gate 2: result frame must be chartable + intent-shaped.

            - non-empty columns, no all-NULL frame (raises in _validate_result_shape)
            - trend intent needs >=2 rows; ranking needs ordering; never empty+non-table
            On failure with attempts left, route back to repair_sql.
            """
            import logging as _lg
            _log = _lg.getLogger(__name__)
            try:
                rows = state.get("rows") or []
                cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
                self._validate_result_shape(state.get("user_goal"), state.get("final_sql"),
                                            rows, cols)
                intent = ((state.get("goal_ir") or {}).get("intent") or "").lower()
                if not rows:
                    state["result_issues"] = []
                elif "trend" in intent and len(rows) < 2:
                    state["result_issues"] = ["trend intent returned fewer than 2 points"]
                else:
                    state["result_issues"] = []
            except Exception as exc:
                state["result_issues"] = [str(exc)[:200]]
            if state.get("result_issues"):
                _log.warning("LangGraph validate_result failed: %s", state["result_issues"][:2])
            return state

        def finalize(state):
            failed = bool(state.get("message") and state.get("_failed"))
            output = {
                "user_goal": self._active_original_goal or state["user_goal"],
                "kpi_alignment": state["kpi_map"],
                "join_path": state["join_path"],
                "goal_ir": state.get("goal_ir"),
                "sql_used": state["final_sql"],
                "row_count": len(state["rows"]),
                "data": state["rows"],
                "message": state.get("message"),
                "preprocessing": state.get("preprocessing"),
                "missing_values_handled": "preserved (NULLs kept as-is)",
                "timestamp": datetime.now().isoformat(),
            }
            output.update(self._build_output_contract(
                state["user_goal"], state["kpi_map"], state["join_path"],
                state["final_sql"], state["rows"],
                success=(not failed),
                message=output["message"],
                status="query_failed" if failed else "success",
                warnings=self._typo_warnings + self._warnings_for_sql(state["final_sql"])
                + self._last_semantic_warnings,
                execution_time_ms=self._last_execution_ms,
            ))
            Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
            state["result_path"] = output_path
            return state

        def _sql_needs_repair(state):
            if state.get("sql_issues") and int(state.get("repair_attempts") or 0) < 3:
                return "repair"
            if state.get("sql_issues"):
                return "execute_anyway"
            return "execute"

        def _result_needs_repair(state):
            if state.get("result_issues") and int(state.get("repair_attempts") or 0) < 3:
                return "repair"
            return "finalize"

        workflow = StateGraph(dict)
        workflow.add_node("load_schema", load_schema)
        workflow.add_node("understand_goal", understand_goal)
        workflow.add_node("resolve_semantics", resolve_semantics)
        workflow.add_node("plan_joins", plan_joins)
        workflow.add_node("generate_sql", generate_sql)
        workflow.add_node("validate_sql", validate_sql)
        workflow.add_node("repair_sql", repair_sql)
        workflow.add_node("execute_sql", execute_sql)
        workflow.add_node("validate_result", validate_result)
        workflow.add_node("finalize", finalize)
        workflow.set_entry_point("load_schema")
        workflow.add_edge("load_schema", "understand_goal")
        workflow.add_edge("understand_goal", "resolve_semantics")
        workflow.add_edge("resolve_semantics", "plan_joins")
        workflow.add_edge("plan_joins", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_conditional_edges("validate_sql", _sql_needs_repair, {
            "repair": "repair_sql",
            "execute": "execute_sql",
            "execute_anyway": "execute_sql",
        })
        workflow.add_edge("repair_sql", "validate_sql")
        workflow.add_edge("execute_sql", "validate_result")
        workflow.add_conditional_edges("validate_result", _result_needs_repair, {
            "repair": "repair_sql",
            "finalize": "finalize",
        })
        workflow.add_edge("finalize", END)

        app = workflow.compile()
        result = app.invoke({
            "user_goal": user_goal, "kpi_map": {}, "relevant_tables": [],
            "join_path": [], "raw_sql": "", "rows": [], "final_sql": "",
            "repair_attempts": 0, "sql_issues": [], "result_issues": [],
        })
        logger.info("Goal Agent done! Saved to %s", result.get('result_path'))
        return result.get("result_path")

