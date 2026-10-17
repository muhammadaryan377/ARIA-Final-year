"""Component: schema normalization and relationship graph (confidence-annotated edges) plus join planning (spec sections 5, 10)."""

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

USER_TRUST_LEVELS = ("USER_CONFIRMED", "USER_CREATED")


def load_user_overrides(user_id, db_name) -> dict:
    """Load persisted user corrections for (user, db) — PK/FK/rejections.

    Returns {"pk": {}, "fk": {}, "rejected": []} when nothing stored.
    Same file data_routes writes, so corrections survive across sessions.
    """
    path = SCHEMA_DIR / f"user_{user_id}" / f"overrides_{db_name or ''}.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"pk": data.get("pk", {}), "fk": data.get("fk", {}),
                    "rejected": data.get("rejected", [])}
    except Exception:
        pass
    return {"pk": {}, "fk": {}, "rejected": []}


class SchemaMixin:
    # ---- user overrides (Journey E: corrections persist into planning) ----

    @staticmethod
    def _edge_key(left_table, left_column, right_table, right_column) -> str:
        return f"{left_table}.{left_column}->{right_table}.{right_column}"

    def _apply_user_overrides(self) -> None:
        """Apply user corrections with higher authority than inference.

        - rejected rel_ids: removed from edges + traversal graph.
        - fk entries: upserted as USER_CONFIRMED edges (top join priority).
        - confirmations of existing edges: upgraded to USER_CONFIRMED.
        """
        ov = getattr(self, "user_overrides", None) or {}
        rejected = set(ov.get("rejected", []) or [])
        fk_specs = ov.get("fk", {}) or {}
        if not rejected and not fk_specs:
            return

        def key_of(e) -> str:
            return self._edge_key(e.get("left_table"), e.get("left_column"),
                                  e.get("right_table"), e.get("right_column"))

        if rejected:
            self.relationship_edges = [e for e in self.relationship_edges
                                       if key_of(e) not in rejected]
            for rel_id in rejected:
                try:
                    left, _, right = rel_id.partition("->")
                    lt = left.rsplit(".", 1)[0]
                    rt = right.rsplit(".", 1)[0]
                    if lt in self.relationship_graph:
                        self.relationship_graph[lt] = [t for t in self.relationship_graph[lt] if t != rt]
                    if rt in self.relationship_graph:
                        self.relationship_graph[rt] = [t for t in self.relationship_graph[rt] if t != lt]
                except Exception:
                    continue

        for rel_id, spec in fk_specs.items():
            if rel_id in rejected or not isinstance(spec, dict):
                continue
            lt, lc = spec.get("table", ""), spec.get("column", "")
            rt, rc = spec.get("references_table", ""), spec.get("references_column", "")
            if not (lt and lc and rt and rc):
                # Bare confirmation of an inferred edge: upgrade in place.
                for e in self.relationship_edges:
                    if key_of(e) == rel_id:
                        e["relationship_type"] = "user_confirmed"
                        e["confidence"] = 1.0
                        e["trust"] = "USER_CONFIRMED"
                continue
            if lt not in self.tables or rt not in self.tables:
                continue
            # Remove any competing edge for the same source column first.
            self.relationship_edges = [
                e for e in self.relationship_edges
                if not (e.get("left_table") == lt and e.get("left_column") == lc)
            ]
            self.relationship_edges.append({
                "left_table": lt, "left_column": lc,
                "right_table": rt, "right_column": rc,
                "relationship_type": "user_confirmed",
                "confidence": 1.0,
                "trust": "USER_CONFIRMED",
                "cardinality": "many_to_one",
                "source": "user",
            })
            if rt not in self.relationship_graph.get(lt, []):
                self.relationship_graph[lt].append(rt)
            if lt not in self.relationship_graph.get(rt, []):
                self.relationship_graph[rt].append(lt)

    def _canon_ident(self, value):
        """Generic identifier match key: lower + strip quotes/brackets."""
        try:
            s = str(value or "").strip()
            while len(s) >= 2 and ((s[0] == '"' and s[-1] == '"')
                                   or (s[0] == "'" and s[-1] == "'")
                                   or (s[0] == "`" and s[-1] == "`")
                                   or (s[0] == "[" and s[-1] == "]")):
                s = s[1:-1].strip()
            return s.lower()
        except Exception:
            return str(value or "").lower()

    def _bare_ident(self, value):
        try:
            segs = [s for s in str(value or "").split(".") if s.strip()]
            last = segs[-1] if segs else str(value or "")
            return self._canon_ident(last)
        except Exception:
            return self._canon_ident(value)

    def _resolve_table_key(self, name):
        """Resolve `name` to a real table key (exact, case-insensitive, bare)."""
        if name in self.tables:
            return name
        low = self._canon_ident(name)
        for k in self.tables:
            if self._canon_ident(k) == low:
                return k
        bare = self._bare_ident(name)
        cands = [k for k in self.tables if self._bare_ident(k) == bare]
        return cands[0] if len(cands) == 1 else None

    def _normalize_schema(self, raw_schema):
        raw_tables = raw_schema.get("tables", {})
        if not isinstance(raw_tables, dict) or not raw_tables:
            available = sorted(p.name for p in SCHEMA_DIR.glob("schema_mapping_*.json"))
            raise ValueError(
                "Schema mapping has no tables (stale or hand-edited JSON?). "
                + (f"Available mappings: {', '.join(available)}."
                   if available
                   else "Re-run the schema extraction for this database.")
            )

        tables = {}
        all_edges = raw_schema.get("relationship_edges", [])
        declared = raw_schema.get("declared_relationships", [])
        inferred = raw_schema.get("inferred_relationships", [])
        known_tables = set(raw_tables)

        def add_fk(table_name, source_column, target_table, target_column, rel_type, confidence=None):
            if not table_name or not source_column or not target_table or not target_column:
                logger.warning(
                    "Skipping malformed relationship entry %r -> %r (missing endpoint names).",
                    table_name, target_table,
                )
                return
            tables.setdefault(table_name, {
                "columns": {}, "primary_key": [], "foreign_keys": [],
            })
            tables[table_name]["foreign_keys"].append({
                "column": source_column,
                "referenced_table": target_table,
                "referenced_column": target_column,
                "type": rel_type,
                "confidence": confidence,
            })

        for table_name, info in raw_schema.get("tables", {}).items():
            columns = info.get("columns", [])
            col_map = {}
            if isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict):
                        name = col.get("column")
                        col_map[name] = {
                            "data_type": col.get("data_type", "TEXT"),
                            "nullable": col.get("nullable", True),
                        }
                    else:
                        col_map[str(col)] = {"data_type": "TEXT", "nullable": True}
            else:
                col_map = columns

            pk = info.get("primary_key", []) or info.get("inferred_primary_key", []) or []
            for key in pk:
                if key in col_map:
                    col_map[key]["is_primary_key"] = True
            # Composite uniqueness (e.g. a fully-populated unique pair on a
            # 1-table dataset) is preserved separately and generically: it
            # informs relationship reasoning without removing its members
            # from GROUP-BY-able dimensions (composite members stay
            # dimensions, never ungroupable PKs).
            composite_pk = list(
                info.get("inferred_composite_primary_key", []) or []
            )
            if not composite_pk:
                try:
                    _cck = info.get("composite_candidate_keys", []) or []
                    if _cck and isinstance(_cck[0], (list, tuple)):
                        composite_pk = [str(c) for c in _cck[0]]
                except Exception:
                    composite_pk = []

            tables[table_name] = {
                "columns": col_map, "primary_key": pk, "foreign_keys": [],
                "empty": bool(info.get("empty")),
                "row_count": info.get("row_count"),
                "composite_primary_key": composite_pk,
            }
            for cname, cmeta in col_map.items():
                cmeta.setdefault("is_primary_key", cname in pk)
                cmeta.setdefault("is_foreign_key", False)
                cmeta.setdefault("references", None)

        for edge in all_edges:
            add_fk(
                edge.get("source_table"), edge.get("source_column"),
                edge.get("target_table"), edge.get("target_column"),
                edge.get("type", "declared"), edge.get("confidence"),
            )
        for rel in declared:
            add_fk(rel.get("table_name"), rel.get("column_name"),
                   rel.get("references_table"), rel.get("references_column"),
                   "declared", None)
        for rel in inferred:
            conf = rel.get("confidence")
            if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                # Schema Agent can emit non-numeric confidence labels such as
                # "llm-confirmed" / "llm-reasoned"; the numeric confidence_score
                # (0-100) is the real measure. Prefer it, never crash on labels.
                conf = rel.get("confidence_score")
                if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                    conf = None
            add_fk(rel.get("table"), rel.get("column"),
                   rel.get("references_table"), rel.get("references_column"),
                   "inferred", conf)

        for table_name, info in tables.items():
            for fk in info.get("foreign_keys", []):
                target = fk.get("referenced_table")
                real_target = None
                if target in tables:
                    real_target = target
                else:
                    # Case-insensitive / bare-segment resolution so quoted
                    # CamelCase or search_path-bare declared names still bind.
                    low = self._canon_ident(target)
                    for k in tables:
                        if self._canon_ident(k) == low:
                            real_target = k
                            break
                    if real_target is None:
                        bare = self._bare_ident(target)
                        cands = [k for k in tables
                                 if self._bare_ident(k) == bare]
                        real_target = cands[0] if len(cands) == 1 else None
                if real_target is not None and real_target != target:
                    fk["referenced_table"] = real_target
                    target = real_target
                if target and target not in tables:
                    logger.warning(
                        "Schema mapping references unknown table '%s' from %s.%s "
                        "(stale/manually-edited JSON?)",
                        target, table_name, fk.get("column"),
                    )
                elif target and self._canon_ident(fk.get("referenced_column") or "") not in {
                        self._canon_ident(c) for c in tables[target]["columns"]}:
                    logger.warning(
                        "Schema mapping references unknown column '%s.%s' from %s.%s",
                        target, fk.get("referenced_column"), table_name, fk.get("column"),
                    )
                else:
                    col_name = fk.get("column")
                    if col_name and col_name in tables[table_name]["columns"]:
                        tables[table_name]["columns"][col_name]["is_foreign_key"] = True
                        tables[table_name]["columns"][col_name]["references"] = {
                            "table": target,
                            "column": fk.get("referenced_column"),
                            "relationship_type": fk.get("type", "unknown"),
                            "confidence": fk.get("confidence"),
                        }

        return tables

    # ---- _build_relationship_graph (_build_relationship_graph) ----

    def _build_relationship_graph(self):
        """Bidirectional relationship graph (spec §5): for every FK edge
        A.column -> B.column the graph permits traversal A->B AND B->A. The
        authoritative FK direction is preserved in `relationship_edges`; this
        dict only supports traversal in both directions."""
        graph = defaultdict(list)
        for table_name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                target = self._resolve_table_key(fk.get("referenced_table")) \
                    or fk.get("referenced_table")
                if target in self.tables:
                    if target not in graph[table_name]:
                        graph[table_name].append(target)
                    if table_name not in graph[target]:
                        graph[target].append(table_name)
        return graph

    # ---- _relationship_trust (_relationship_trust) ----

    def _relationship_trust(self, rel_type, confidence):
        """Explicit relationship trust level (spec §6).

        USER_CONFIRMED / USER_CREATED -> outrank everything (Journey E).
        DECLARED                     -> safe for deterministic join planning
        HIGH_CONFIDENCE_INFERRED     -> may be used when required
        MEDIUM_CONFIDENCE_INFERRED   -> candidate only / needs stronger evidence
        LOW_CONFIDENCE_INFERRED      -> never joined automatically
        """
        if (rel_type or "") in ("user_confirmed", "user_created"):
            return "USER_CONFIRMED"
        if rel_type == "declared":
            return "DECLARED"
        try:
            conf = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            # Non-numeric confidence labels ("llm-confirmed", "llm-reasoned"):
            # never crash on schema output. A confirmation label is treated as
            # high confidence; everything else is treated conservatively.
            label = str(confidence).lower()
            if "confirm" in label or "high" in label:
                conf = 1.0
            elif "reason" in label or "medium" in label:
                conf = 0.7
            else:
                conf = 0.0
        if conf > 1.0:
            conf = conf / 100.0
        if conf >= 0.8:
            return "HIGH_CONFIDENCE_INFERRED"
        if conf >= 0.6:
            return "MEDIUM_CONFIDENCE_INFERRED"
        return "LOW_CONFIDENCE_INFERRED"

    # ---- _build_relationship_edges (_build_relationship_edges) ----

    def _build_relationship_edges(self):
        """Every FK edge with full metadata (spec §5): both endpoints, the
        original FK direction, relationship type, confidence and trust level.

        Generic multi-schema resolution: targets bind case-insensitively +
        bare-segment aware so quoted CamelCase or search_path-bare declared
        names still yield cross-schema edges; ambiguous bare names are
        dropped so the database stays the final arbiter.
        """
        edges = []
        for table_name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                target = fk.get("referenced_table")
                try:
                    real = self._resolve_table_key(target)
                except Exception:
                    real = None
                target = real or target
                if target not in self.tables:
                    continue
                rel_type = fk.get("type", "declared")
                conf = fk.get("confidence")
                cardinality = None
                right_info = self.tables.get(target)
                if right_info:
                    is_unique = (
                        fk["referenced_column"] in right_info.get("primary_key", [])
                    )
                    cardinality = "many_to_one" if is_unique else "many_to_many"
                edges.append({
                    "left_table": table_name,
                    "left_column": fk["column"],
                    "right_table": target,
                    "right_column": fk["referenced_column"],
                    "relationship_type": rel_type,
                    "confidence": conf,
                    "trust": self._relationship_trust(rel_type, conf),
                    "cardinality": cardinality,
                })
        return edges

    # ---- _collect_uncertain_edges (_collect_uncertain_edges) ----

    def _collect_uncertain_edges(self, raw_schema):
        """Collect edges whose Schema Agent state is UNCERTAIN/REVIEW. These are
        never silently used: when a query joins through one, a warning is added
        to the goal contract (spec §5). Ambiguous relationships are all
        uncertain by definition."""
        keys = set()
        for rel in raw_schema.get("ambiguous_relationships", []) or []:
            keys.add(
                f"{rel.get('table')}.{rel.get('column')}"
                f"->{rel.get('references_table')}.{rel.get('references_column')}"
            )
        for rel in raw_schema.get("inferred_relationships", []) or []:
            state = (rel.get("relationship_state") or "").upper()
            if state in self.UNCERTAIN_STATES:
                keys.add(
                    f"{rel.get('table')}.{rel.get('column')}"
                    f"->{rel.get('references_table')}.{rel.get('references_column')}"
                )
        for edge in raw_schema.get("relationship_edges", []) or []:
            if edge.get("type") != "inferred":
                continue
            state = (edge.get("relationship_state") or "").upper()
            if state in self.UNCERTAIN_STATES:
                keys.add(
                    f"{edge.get('source_table')}.{edge.get('source_column')}"
                    f"->{edge.get('target_table')}.{edge.get('target_column')}"
                )
        return keys

    # ---- _build_schema_ddl (_build_schema_ddl) ----

    def _qtn(self, table_name):
        """Table reference for DDL: schema-qualified keys become
        "sales"."salesorderheader" (schema separator, never a dotted name)."""
        def _strip(v):
            try:
                s = str(v or "").strip()
                while len(s) >= 2 and ((s[0] == '"' and s[-1] == '"')
                                       or (s[0] == "'" and s[-1] == "'")
                                       or (s[0] == "`" and s[-1] == "`")
                                       or (s[0] == "[" and s[-1] == "]")):
                    s = s[1:-1].strip()
                return s
            except Exception:
                return str(v or "")
        try:
            segs = [_strip(s) for s in str(table_name or "").split(".")]
            segs = [s for s in segs if s]
            if len(segs) > 1:
                return ".".join(f'"{s.replace(chr(34), chr(34) + chr(34))}"'
                                for s in segs)
        except Exception:
            pass
        return f'"{_strip(table_name).replace(chr(34), chr(34) + chr(34))}"'

    def _build_schema_ddl(self, join_path, full=False):
        """Build DDL for the tables in the join path.

        When full=True the DDL also includes every other table in the schema
        (FK-connected first), so the LLM can discover the metric tables it
        needs even when the keyword scoring missed them. The join-path tables
        are always listed first and marked as preferred.
        """
        relevant = [t for t in join_path if t in self.tables]
        logger.info("Relevant tables (join path): %s", relevant)

        tables = list(relevant)
        if full:
            remaining = [t for t in self.tables if t not in relevant]
            # Prefer tables that connect to the join path via foreign keys.
            connected = set()
            for t in relevant:
                for nb in self.relationship_graph.get(t, []):
                    if nb in remaining:
                        connected.add(nb)
            tables += [t for t in remaining if t in connected]
            tables += [t for t in remaining if t not in connected]

        ddl_parts = []
        for table_name in tables:
            info = self.tables[table_name]
            columns = info.get("columns", {})
            col_defs = []
            for column_name, col_info in columns.items():
                data_type = col_info.get("data_type", "TEXT")
                is_pk = column_name in info.get("primary_key", []) or col_info.get("is_primary_key")
                suffix = " PRIMARY KEY" if is_pk else ""
                col_defs.append(f'    "{column_name}" {data_type}{suffix}')
                # Text-stored numerics: annotate from the probe cache (no new
                # probes here -- the per-question budget was spent at
                # measure-resolution time) so LLM SQL also coerces.
                try:
                    if self._coerced_status_cached(table_name, column_name):
                        _cexpr = self._coercion_expression(
                            table_name, column_name)
                        col_defs.append(
                            f'    -- "{column_name}" is TEXT holding numbers: '
                            f'for SUM/AVG/MIN/MAX use {_cexpr} '
                            f'(unparsable -> NULL; percent stays as-is, e.g. '
                            f'15% -> 15); GROUP BY/WHERE use the raw column')
                except Exception:
                    pass
            fk_lines = []
            for fk in info.get("foreign_keys", []):
                fk_lines.append(
                    f'    FOREIGN KEY ("{fk["column"]}") REFERENCES {self._qtn(fk["referenced_table"])}("{fk["referenced_column"]}")'
                )
            all_defs = ",\n".join(col_defs + fk_lines)
            ddl_parts.append(f'CREATE TABLE {self._qtn(table_name)} (\n{all_defs}\n);')
        return "\n\n".join(ddl_parts)

    # ---- _fk_edges (_fk_edges) ----

    def _fk_edges(self):
        if getattr(self, "_fk_edges_cache", None) is None:
            edges = []
            for table_name, info in self.tables.items():
                for fk in info.get("foreign_keys", []):
                    edges.append((
                        table_name,
                        fk["column"],
                        fk["referenced_table"],
                        fk["referenced_column"],
                    ))
            self._fk_edges_cache = edges
        return self._fk_edges_cache

    # ---- _required_tables_from_ir (_required_tables_from_ir) ----

    def _required_tables_from_ir(self, ir, join_path):
        """The tables the goal actually needs (spec §8): resolved measure +
        dimension tables. Join planning and semantic validation run over this
        set only, so neighbor-expanded context tables never become mandatory
        joins."""
        if not ir:
            return list(join_path)
        required = set()
        for m in ir.get("metrics", []):
            if m.get("resolved_table"):
                real = self._resolve_table_key(m["resolved_table"]) \
                    or m["resolved_table"]
                if real in self.tables:
                    required.add(real)
        for d in ir.get("dimensions", []):
            if d.get("resolved_table"):
                real = self._resolve_table_key(d["resolved_table"]) \
                    or d["resolved_table"]
                if real in self.tables:
                    required.add(real)
        for f in ir.get("filters", []) or []:
            try:
                _col = str(f.get("column") or "")
                if "." in _col:
                    _t = _col.split(".")[0]
                    real = self._resolve_table_key(_t)
                    if real is not None:
                        required.add(real)
            except Exception:
                pass
        return sorted(required) if required else list(join_path)

    # ---- _plan_joins (_plan_joins) ----

    def _plan_joins(self, required_tables):
        """Explicit deterministic join plan (spec §7/§8): the minimum connected
        subgraph of the relationship graph that covers all required tables.

        Returns {"required_tables", "nodes", "edges"} where every edge carries
        table/column endpoints, relationship type and trust level. Declared
        relationships are preferred; LOW_CONFIDENCE_INFERRED edges are only
        used when there is no other way to connect a required table.
        """
        req = []
        for t in (required_tables or []):
            real = self._resolve_table_key(t)
            if real is not None and real not in req:
                req.append(real)
        if not req:
            return {"required_tables": [], "nodes": [], "edges": []}

        trust_rank = {"USER_CONFIRMED": -1, "DECLARED": 0, "HIGH_CONFIDENCE_INFERRED": 1,
                      "MEDIUM_CONFIDENCE_INFERRED": 2, "LOW_CONFIDENCE_INFERRED": 3}
        adj = defaultdict(list)
        for e in self.relationship_edges:
            rank = trust_rank.get(e.get("trust"), 4)
            left = self._resolve_table_key(e.get("left_table")) or e.get("left_table")
            right = self._resolve_table_key(e.get("right_table")) or e.get("right_table")
            if left in self.tables and right in self.tables:
                adj[left].append((rank, e))
                adj[right].append((rank, e))
        for table in adj:
            adj[table].sort(key=lambda pair: pair[0])

        connected = {req[0]}
        nodes = [req[0]]
        edges = []
        remaining = [t for t in req[1:] if t != req[0]]

        while remaining:
            target, path_tables, path_edges = None, None, None
            for t in remaining:
                found = self._bfs_join_path(adj, connected, t)
                if found is not None:
                    path_tables, path_edges = found
                    target = t
                    break
            if target is None:
                for t in remaining:
                    if t not in nodes:
                        nodes.append(t)
                break
            for e in path_edges:
                if e not in edges:
                    edges.append(e)
            for n in path_tables:
                if n not in connected:
                    connected.add(n)
                    nodes.append(n)
            remaining = [t for t in remaining if t != target]

        return {"required_tables": req, "nodes": nodes, "edges": edges}

    # ---- _bfs_join_path (_bfs_join_path) ----

    def _bfs_join_path(self, adj, connected, target):
        """Shortest BFS path (by hop count) from the connected component to
        `target`, returning (tables_on_path, edges_on_path) or None."""
        queue = [[n] for n in connected]
        seen = set(connected)
        while queue:
            path = queue.pop(0)
            tail = path[-1]
            for _rank, e in adj.get(tail, []):
                nb = e["right_table"] if e["left_table"] == tail else e["left_table"]
                if nb in seen:
                    continue
                new_path = path + [nb]
                if nb == target:
                    path_tables = new_path
                    path_edges = []
                    for a, b in zip(path_tables, path_tables[1:]):
                        for _r, edge in adj[a]:
                            if edge["left_table"] == b or edge["right_table"] == b:
                                path_edges.append(edge)
                                break
                    return path_tables, path_edges
                seen.add(nb)
                queue.append(new_path)
        return None

