"""Component: schema grounding - relevant tables, join path and semantic resolution of subjects, dimensions and metrics (spec section 6)."""

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


class GroundingMixin:
    # ---- _table_keyword_score (_table_keyword_score) ----

    def _table_keyword_score(self, table_name, info, user_goal_tokens):
        """Score how strongly a table matches the goal words.

        Name matches dominate (synonym/plural-aware). Column matches are only a
        small bonus ON TOP of a name hit; a bare column match (e.g. the generic
        word 'company' hitting `company_name` on every table) must never outrank
        a table whose name actually matches the subject.

        Generic multi-schema rules (no table names): identifiers fold
        CamelCase/quotes/separators, and the bare table segment outranks the
        schema prefix so a schema-synonym hit never ties a bare-entity hit.
        """
        try:
            _segs = [s for s in str(table_name or "").split(".") if s.strip()]
            _bare_seg = _segs[-1] if _segs else str(table_name or "")
            _schema_segs = _segs[:-1] if len(_segs) > 1 else []
            try:
                bare_parts = set(self._fold_ident_tokens(_bare_seg))
            except Exception:
                bare_parts = set(self._normalize_tokens(_bare_seg))
            schema_parts = set()
            for _s in _schema_segs:
                try:
                    schema_parts |= set(self._fold_ident_tokens(_s))
                except Exception:
                    schema_parts |= set(self._normalize_tokens(_s))
        except Exception:
            bare_parts = set(self._normalize_tokens(table_name))
            schema_parts = set()
        name_hit = False
        for tok in user_goal_tokens:
            for part in list(bare_parts) + list(schema_parts):
                try:
                    if self._schema_token_matches_goal(part, tok):
                        name_hit = True
                        break
                except Exception:
                    continue
            if name_hit:
                break
        score = 0
        if name_hit:
            # Strong: the table's name means something mentioned in the goal.
            # An EXACT name token outranks a partial/synonym match so "orders"
            # picks `orders`, not `order_details` (which only matches because
            # "order" is a prefix of "order_detail"). Bare-segment hits
            # outrank schema-prefix hits so multi-schema synonyms on the
            # schema never tie a real entity match. Generic, no names.
            for tok in user_goal_tokens:
                matched = False
                for part in bare_parts:
                    if tok == part or (tok.endswith("s") and tok[:-1] == part) \
                            or (part.endswith("s") and part[:-1] == tok):
                        score += 55
                        matched = True
                        break
                if matched:
                    continue
                for part in bare_parts:
                    try:
                        if self._schema_token_matches_goal(part, tok):
                            score += 40
                            matched = True
                            break
                    except Exception:
                        continue
                if matched:
                    continue
                for part in schema_parts:
                    if tok == part or (tok.endswith("s") and tok[:-1] == part) \
                            or (part.endswith("s") and part[:-1] == tok):
                        score += 10
                        matched = True
                        break
                if matched:
                    continue
                for part in schema_parts:
                    try:
                        if self._schema_token_matches_goal(part, tok):
                            score += 5
                            break
                    except Exception:
                        continue
        try:
            _col_fold = self._fold_ident_tokens
        except Exception:
            _col_fold = None
        for column_name in info.get("columns", {}).keys():
            try:
                col_parts = set(_col_fold(column_name)) if _col_fold else set(
                    self._normalize_tokens(column_name))
            except Exception:
                col_parts = set(self._normalize_tokens(column_name))
            for tok in user_goal_tokens:
                for part in col_parts:
                    try:
                        hit = self._schema_token_matches_goal(part, tok)
                    except Exception:
                        hit = False
                    if hit:
                        # Column matches are a bonus only on a name-matched
                        # table; otherwise they are a weak hint that must not
                        # outrank a subject-table name match.
                        score += 8 if name_hit else 3
                        break
        return score

    # ---- _expand_with_neighbors (_expand_with_neighbors) ----

    def _expand_with_neighbors(self, selected, limit=4):
        """Expand the selected tables through FK neighbors, depth-bounded so
        unrelated tables further out in the graph never get pulled in.
        Deterministic: neighbors are ordered by FK connectivity (more central
        tables first) then alphabetically, never by set/dict insertion order.
        Only fills up to `limit` tables so a small relevant set is not drowned
        by the whole graph."""
        expanded = set(selected)
        queue = [(t, 0) for t in selected]
        visited = set()
        while queue and len(expanded) < limit:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if depth >= 2:
                continue
            neighbors = self.relationship_graph.get(current, [])
            neighbors = sorted(
                set(neighbors),
                key=lambda n: (-len(self.tables[n].get("foreign_keys", [])) if n in self.tables else 0, n),
            )
            for neighbor in neighbors:
                if len(expanded) >= limit:
                    break
                if neighbor in self.tables and neighbor not in expanded:
                    expanded.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return sorted(
            list(expanded),
            key=lambda n: (-len(self.tables[n].get("foreign_keys", [])) if n in self.tables else 0, n),
        )

    # ---- _get_relevant_tables (_get_relevant_tables) ----

    def _get_relevant_tables(self, user_goal, max_tables=3):
        # Current question only: history table words must not outrank the
        # current subject. Elliptical follow-ups ("and which category sells
        # most?") additionally union history tokens so the prior context
        # still scopes the tables.
        try:
            _cur = self._current_question_text(user_goal)
        except Exception:
            _cur = str(user_goal or "")
        goal_tokens = self._normalize_tokens(_cur)
        try:
            _hist = self._history_text(user_goal)
            if _hist.strip() and self._is_elliptical_followup_text(_cur):
                goal_tokens = goal_tokens | self._normalize_tokens(_hist)
        except Exception:
            pass
        if not goal_tokens:
            goal_tokens = {"data", "summary", "report"}

        ranked = []
        for table_name, info in self.tables.items():
            if info.get("empty"):
                # An empty table (0 rows) can never contribute data to an
                # answer; selecting it as relevant forces an unsatisfiable
                # join into the semantic plan and degrades every goal that
                # mentions its name.
                continue
            score = self._table_keyword_score(table_name, info, goal_tokens)
            ranked.append((table_name, score))
        # Deterministic ordering: ties resolved by FK connectivity (more central
        # tables win) then alphabetically, never by dict insertion order.
        ranked.sort(key=lambda x: (-x[1], -len(self.tables[x[0]].get("foreign_keys", [])), x[0]))
        selected = [name for name, _ in ranked if _ > 0][:max_tables]

        if not selected:
            fallback = [name for name, _ in self.tables.items() if not self.tables[name].get("empty")]
            fallback.sort(key=lambda x: (-len(self.tables[x].get("foreign_keys", [])), x))
            selected = fallback[:max_tables] if fallback else []
            if len(selected) < max_tables:
                selected = self._expand_with_neighbors(selected, max_tables)
            return selected[:max_tables]

        # Positively-scored tables answer the goal on their own: never
        # auto-pull FK neighbors here. Neighbor expansion would force extra
        # tables (and JOINs) into every query — including single-table
        # questions such as "What are my sales?" which must run with NO JOIN.
        # Multi-table context is added later by deterministic join planning,
        # only for tables the resolved goal actually needs.
        return selected[:max_tables]

    # ---- _find_join_hops (_find_join_hops) ----

    def _find_join_hops(self, start, target):
        """BFS join path between two tables using the relationship graph.

        Generic multi-schema resolution (no table names): endpoints are
        resolved case-insensitively + bare-segment aware so `customer`,
        `SALES.CUSTOMER` and `"Customer"` all bind the same node; the
        database remains the final arbiter for ambiguous bare names.
        """
        try:
            _rs = self._resolve_table_key(start) if hasattr(
                self, "_resolve_table_key") else None
            _rt = self._resolve_table_key(target) if hasattr(
                self, "_resolve_table_key") else None
        except Exception:
            _rs, _rt = None, None
        start = _rs or start
        target = _rt or target
        if start == target:
            return [start]
        if start not in self.tables or target not in self.tables:
            return None
        queue = [[start]]
        visited = {start}
        while queue:
            path = queue.pop(0)
            try:
                _nbrs = self.relationship_graph.get(path[-1], [])
            except Exception:
                _nbrs = []
            for neighbor in _nbrs:
                if neighbor not in self.tables:
                    continue
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    # ---- _determine_join_path (_determine_join_path) ----

    def _determine_join_path(self, relevant_tables):
        """
        Return the ordered join path that connects all relevant tables, using
        the schema's relationship graph.

        Only FK-reachable tables enter the path. Tables with NO join path are
        never forced in: inventing a JOIN where no relationship exists produces
        cartesian products, and a single-table question must run with NO JOIN.

        Generic multi-schema resolution: inputs are bound case-insensitively
        + bare-segment aware (search_path semantics) so `customer` and
        `SALES.CUSTOMER` resolve to the same node; ambiguous bare names are
        dropped so the database stays the final arbiter.
        """
        if not relevant_tables:
            return []
        resolved = []
        try:
            _res = self._resolve_table_key if hasattr(
                self, "_resolve_table_key") else None
        except Exception:
            _res = None
        for t in (relevant_tables or []):
            try:
                r = _res(t) if callable(_res) else (t if t in self.tables else None)
            except Exception:
                r = t if t in self.tables else None
            if r is not None and r in self.tables and r not in resolved:
                resolved.append(r)
        if not resolved:
            return []
        path = [resolved[0]]
        remaining = resolved[1:]
        while remaining:
            connected = None
            best_local = None
            for table in remaining:
                hop = self._find_join_hops(path[-1], table)
                if hop:
                    best_local = hop
                    connected = table
                    break
            if best_local:
                for node in best_local[1:]:
                    if node not in path:
                        path.append(node)
                remaining = [t for t in remaining if t != connected]
            else:
                # No FK path to the remaining tables: leave them out rather
                # than assuming a JOIN exists. The resolved goal (IR) decides
                # the required tables later; unconnected tables would only
                # corrupt the query.
                logger.info(
                    "No FK join path to %s from %s; answering without them.",
                    remaining, path,
                )
                break
        return path

    # ---- _pick_measure_column (_pick_measure_column) ----

    def _pick_measure_column(self, table_name):
        """First numeric, non-key, non-id, non-FK, non-dimension column (best-guess measure).

        Dimension-like numerics (store ids, flags, years, ...) are skipped via
        the shared predicate so a single-table dataset resolves to its real
        measure (e.g. Walmart weekly_sales, not store).
        """
        info = self.tables[table_name]
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        for col in info["columns"]:
            low = col.lower()
            if col in pk or col in fk_cols or low == "id" or low.endswith("_id") or low.endswith("id"):
                continue
            if self._classify_column(table_name, col) == "numeric" and not self._is_dimension_like(table_name, col):
                return col
        return None

    # ---- _first_date_column (_first_date_column) ----

    def _first_date_column(self, tables):
        # Shared path: datetime role (handles TEXT dates like Walmart DD-MM-YYYY
        # via the name hint) wins; legacy dtype-substring check is the fallback.
        for table in tables:
            if table not in self.tables:
                continue
            for col_name in self.tables[table].get("columns", {}):
                try:
                    if self._classify_column(table, col_name) == "date":
                        return (table, col_name)
                except Exception:
                    pass
        for table in tables:
            if table not in self.tables:
                continue
            for col_name, col_info in self.tables[table].get("columns", {}).items():
                dt = (col_info.get("data_type") or "").lower()
                if any(k in dt for k in ("date", "time", "timestamp")) or "date" in col_name.lower():
                    return (table, col_name)
        return None

    # ---- _is_measure_column (_is_measure_column) ----

    def _is_measure_column(self, table, col):
        """True when a column is a numeric, non-key, non-FK measure candidate.

        Calendar-year columns (release_year, model_year) are never measures:
        years label a timeline (usable as dimensions/trends) but SUM/AVG
        over them is meaningless on any schema.
        """
        try:
            from core.semantic import is_year_like as _is_year_like
            if _is_year_like(col):
                return False
        except Exception:
            pass
        info = self.tables.get(table, {})
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        if col in pk or col in fk_cols:
            return False
        dt = str(info.get("columns", {}).get(col, {}).get("data_type", "")).lower()
        if any(k in dt for k in (
                "int", "decimal", "float", "numeric", "double",
                "money", "real", "bigint", "smallint")):
            return True
        return False

    # ---- _pick_label_column (_pick_label_column) ----

    def _pick_label_column(self, table):
        """Best human-readable grouping/display column of a table (the first
        text, non-key column; otherwise any non-key column)."""
        info = self.tables.get(table, {})
        pk = set(info.get("primary_key", []) or [])
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        text_col = None
        for col in info.get("columns", {}):
            low = col.lower()
            if col in pk or col in fk_cols or low.endswith("_id") or low == "id":
                continue
            if self._classify_column(table, col) == "text" and not text_col:
                text_col = col
            if "name" in low or "label" in low or "title" in low:
                return col
        return text_col

    # ---- _is_dimension_like (_is_dimension_like) ----

    def _is_dimension_like(self, table, col):
        """A column is a breakdown dimension (even when numeric) if its name
        suggests a category/time attribute such as model_year or order_status.
        This keeps quantity/price-style numeric columns as measures while still
        grouping by model_year on any schema."""
        if not self._is_measure_column(table, col):
            return True
        low = col.lower()
        return any(k in low for k in self._DIMENSION_LIKE_NUMERIC)

    # ---- _match_dimension_columns (_match_dimension_columns) ----

    def _match_dimension_columns(self, goal_l, table):
        """Columns of `table` that the goal means AND that are legitimate
        breakdown columns (excludes PK / FK / *_id and pure-measure numeric
        columns). Used for dimension grounding and for choosing the fact table
        of an inferred COUNT.

        Metric/analytical concept words ('sales', 'value', 'average', ...) are
        stripped BEFORE column matching: in 'average order value per customer'
        the word 'value' (or 'sales' -> synonym 'order') must not bind
        order_date / ship_name as a dimension and make the SQL over-group."""
        info = self.tables.get(table, {})
        fk_cols = {fk.get("column") for fk in info.get("foreign_keys", [])}
        pk = set(info.get("primary_key", []) or [])
        goal_tokens = self._normalize_tokens(goal_l)
        # Don't bind metric words, analytical words, or (when the goal carries
        # a metric) bare count/fact nouns like "order"/"sales" — "average ORDER
        # value per customer" must not group by order_date, and "by ORDER STATUS"
        # should bind the status column, not order_date.
        metric_present = bool(goal_tokens & self._METRIC_CONCEPT_WORDS)
        skip = set(self._ANALYTICAL_GOAL_WORDS) | set(self._METRIC_CONCEPT_WORDS)
        if metric_present:
            skip |= self._COUNT_OBJECT_TOKENS
        filtered = " ".join(t for t in goal_tokens if t not in skip)
        out = []
        for score, col in self._match_columns(filtered or goal_l, table):
            low = col.lower()
            if col in fk_cols or col in pk or low == "id" or low.endswith("_id"):
                continue
            if not self._is_dimension_like(table, col):
                continue
            out.append((score, col))
        return out

    # ---- _pick_fact_table (_pick_fact_table) ----

    def _pick_fact_table(self, goal_l, relevant, subject):
        """Best table to COUNT / aggregate when the metric is inferred
        (spec §3 / §10).

        Rules (generic, schema-driven):
          - if the SUBJECT itself carries the requested dimension columns,
            count the subject ("how many customers");
          - else if the goal EXPLICITLY names a child table (word in the goal
            matches the child's table name: "shipping company with most
            ORDERS" -> count orders, not order_details), prefer that child;
          - else if a CHILD table that references the subject carries the
            requested dimension columns, count there ("top brands by model
            year" -> count products, because model_year lives in products);
          - else fall back to the subject.
        """
        if not subject:
            return relevant[0] if relevant else None
        if self._match_dimension_columns(goal_l, subject):
            return subject
        goal_tokens = self._normalize_tokens(goal_l)
        best_child = None
        for t in relevant:
            if t == subject:
                continue
            info = self.tables.get(t, {})
            child_of_subject = any(
                fk.get("referenced_table") == subject
                for fk in info.get("foreign_keys", [])
            )
            if not child_of_subject:
                continue
            # The child must be named as a WHOLE table ("orders" names
            # `orders`, not `order_details` via a shared "order" prefix).
            name_parts = [p for p in str(t).lower().split("_") if p]
            if not name_parts:
                continue
            def _tok_matches(tok, part):
                return (tok == part or (tok.endswith("s") and tok[:-1] == part)
                        or (part.endswith("s") and part[:-1] == tok))
            named_in_goal = all(
                any(_tok_matches(tok, part) for tok in goal_tokens)
                for part in name_parts
            )
            if named_in_goal:
                return t
            if self._match_dimension_columns(goal_l, t):
                best_child = t
                break
        return best_child or subject

    # ---- _find_price_quantity_expression (_find_price_quantity_expression) ----

    def _find_price_quantity_expression(self, relevant, concept_tokens):
        """Spec §3: when no single measure column matches a sales-like concept,
        look for a fact table with BOTH a quantity and a unit-price column and
        synthesize quantity * unit_price.  Also includes discount when present.
        Generic, schema-driven, no DB names."""
        # Quantity times selling price estimates revenue, never profit/margin.
        # Do not replace a matched profit column with a revenue expression.
        if not set(concept_tokens or ()) & {"revenue", "sales"}:
            return None
        for table in relevant:
            info = self.tables.get(table, {})
            cols = list(info.get("columns", {}))

            def _fold(c):
                try:
                    return set(self._fold_ident_tokens(c))
                except Exception:
                    return {str(c or "").lower()}

            qty = next((c for c in cols if self._schema_token_matches_goal("quantity", c)
                        or "qty" in _fold(c) or "quantity" in _fold(c)), None)
            price = next((c for c in cols if self._schema_token_matches_goal("price", c)
                          or "price" in _fold(c) or "amount" in _fold(c)), None)
            if qty and price and qty != price:
                if self._is_measure_column(table, qty) and self._is_measure_column(table, price):
                    disc = next((c for c in cols if "discount" in c.lower()), None)
                    if disc:
                        expr = (f"{table}.{qty} * {table}.{price} "
                                f"* (1 - COALESCE({table}.{disc}, 0))")
                    else:
                        expr = f"{table}.{qty} * {table}.{price}"
                    return {
                        "table": table,
                        "expression": expr,
                        "quantity": qty,
                        "price": price,
                        "discount": disc,
                    }
        return None

    # ---- _is_pure_fact_table (_is_pure_fact_table) ----

    def _is_pure_fact_table(self, table):
        """True for a line-item/fact table: every non-key column is a measure
        (order_details, transaction_items, ...). Such tables are measures, never
        a breakdown subject, and must not contribute label dimensions."""
        info = self.tables.get(table, {})
        cols = [c for c in info.get("columns", {})
                if not (c.lower().endswith("_id") or c.lower() == "id")]
        if not cols:
            return True
        return all(self._is_measure_column(table, c) for c in cols)

    # ---- _pick_subject (_pick_subject) ----

    _COUNT_OBJECT_TOKENS = frozenset({
        "order", "orders", "sale", "sales", "transaction", "transactions",
        "quantity", "qty", "units", "unit", "amount", "price", "cost",
        "revenue", "income", "turnover", "profit", "margin",
    })

    _QUANTITY_COL_TOKENS = frozenset({
        "quantity", "qty", "units", "unit", "count", "num",
    })

    # Column-name preference tiers for measure resolution. When match scores
    # tie, higher-tier (KPI total) columns win so "overall sales" binds the
    # sales.amount total column rather than a per-line quantity/price column.
    _MONETARY_TOTAL_COLS = frozenset({
        "amount", "total", "revenue", "sales",
    })
    _MONETARY_LINE_COLS = frozenset({
        "price", "cost", "value", "margin", "income",
    })

    def _pick_subject(self, matched_tables, goal_tokens, entity_tokens):
        """Choose the subject (entity) table and the ordered list of entity
        (subject) tables from the matched tables.

        Entity tables are those whose NAME matches an entity token
        (non-metric, non-analytical; minus count/fact nouns when a metric is
        present). Pure-fact tables (order_details, ...) are never subjects.
        Falls back to non-fact matched tables, then the strongest name match.

        Generic multi-schema rules: CamelCase-folded bare-segment matching
        outranks schema-prefix synonyms, so a schema word never ties a real
        entity hit; ties break by bare exact/plural strength, then FK
        connectivity, then name. No table names.
        """
        def _bare_folded(table):
            try:
                _seg = str(table or "").split(".")[-1]
                return set(self._fold_ident_tokens(_seg))
            except Exception:
                try:
                    return set(self._normalize_tokens(table))
                except Exception:
                    return {str(table or "").lower()}

        def _bare_strength(table):
            try:
                parts = _bare_folded(table)
            except Exception:
                return 0
            best = 0
            for tok in (entity_tokens or set()):
                for part in parts:
                    if tok == part or (tok.endswith("s") and tok[:-1] == part) \
                            or (part.endswith("s") and part[:-1] == tok):
                        best = max(best, 2)
                    else:
                        try:
                            if self._schema_token_matches_goal(part, tok):
                                best = max(best, 1)
                        except Exception:
                            continue
            return best

        def name_matches_entity(table):
            return _bare_strength(table) > 0
        subject_tables = [
            mt for mt in matched_tables
            if not self._is_pure_fact_table(mt[1]) and name_matches_entity(mt[1])
        ]
        if not subject_tables:
            subject_tables = [
                mt for mt in matched_tables if not self._is_pure_fact_table(mt[1])
            ]
        if not subject_tables:
            subject_tables = matched_tables
        def key(mt):
            t = mt[1]
            return (
                self._table_keyword_score(t, self.tables[t], goal_tokens),
                _bare_strength(t),
                -len(self.tables[t].get("foreign_keys", [])),
                t,
            )
        best = max(subject_tables, key=key)
        return best[1], subject_tables

    def _explicit_breakdown_tokens(self, goal_l):
        """Goal tokens that name an explicit breakdown dimension via a
        'by X / per X / for each X' phrase — plus comparison phrasing
        ('vs X', 'versus X', 'compare ...', 'between ...').

        Empty when the goal has no explicit breakdown (so only the subject
        label becomes a dimension). Shared path: handles Walmart
        'holiday vs non-holiday' the same way as multi-table
        'compare sales between regions'.
        """
        goal_l = str(goal_l).lower()
        tokens = self._normalize_tokens(goal_l)
        breakdown = set()
        for m in re.finditer(r"\b(?:by|per)\s+([a-z][a-z0-9_\s]+)", goal_l):
            for w in m.group(1).split():
                breakdown.add(w.rstrip("s") if w.endswith("s") else w)
        if re.search(r"\bfor each\b|\bfor every\b", goal_l):
            # tokens after "for each" up to a punctuation/keyword
            for m in re.finditer(r"for each\s+([a-z][a-z0-9_\s]+)", goal_l):
                for w in m.group(1).split():
                    breakdown.add(w.rstrip("s") if w.endswith("s") else w)
        # Comparison phrasing names the breakdown dimension too:
        # "holiday vs non-holiday", "compare sales between regions".
        if re.search(r"\bvs\b|\bversus\b|\bcompare\b|\bbetween\b|\bamong\b", goal_l):
            cmp_tokens = set(tokens) - set(self._ANALYTICAL_GOAL_WORDS) - set(self._METRIC_CONCEPT_WORDS)
            # 'vs'/'non' style glue words carry no schema meaning on their own.
            cmp_tokens -= {"vs", "versus", "compare", "between", "among", "non", "not"}
            breakdown |= cmp_tokens
        # Intersect with what the goal actually contains so we never invent words.
        return breakdown & tokens

    # ---- generic helpers for unseen-schema grounding (no DB names) ----

    _REVENUE_WORD_RE = r"\b(revenue|revenues|sales|turnover|income|gmv)\b"
    _AOV_RE = r"\baov\b|average\s+order\s+value|average\s+revenue|average\s+sales"
    _TREND_RE = (r"\btrend\b|\bover time\b|\bmonthly\b|\bdaily\b|\bweekly\b"
                 r"|\byearly\b|\bhistory\b|\bhistorical\b|\bchanged?\b"
                 r"|\bgrowth\b|\btime series\b"
                 r"|\bdecline[sd]?\b|\bdeclining\b|\bdecrease[sd]?\b"
                 r"|\bdecreasing\b|\bdrops?\b|\bdropped\b|\bdropping\b"
                 r"|\bfalls?\b|\bfell\b|\bfalling\b")
    # Name signals for preferred breakdowns (generic business concepts only).
    _GEO_CAT_TOKENS = frozenset({
        "region", "territory", "area", "zone", "district", "country",
        "city", "state", "province", "location", "category", "department",
        "division", "segment", "group", "class", "type", "brand",
        "channel", "market",
    })
    _PERSON_NAME_TOKENS = frozenset({
        "first", "last", "middle", "full", "person", "contact",
        "customer", "client", "buyer", "employee", "staff", "member",
    })
    # Column-name tokens that must never answer a revenue-like question.
    _REVENUE_EXCLUDED_TOKENS = frozenset({"cost", "fuel", "expense", "tax"})

    def _is_revenue_like_goal(self, goal_l, ir=None):
        try:
            if re.search(self._REVENUE_WORD_RE, str(goal_l or "")):
                return True
        except Exception:
            pass
        try:
            for m in (ir or {}).get("metrics", []) or []:
                if str(m.get("concept") or "").lower() in (
                        "revenue kpi", "sales kpi"):
                    return True
        except Exception:
            pass
        return False

    def _is_aov_goal(self, goal):
        try:
            return bool(re.search(self._AOV_RE, str(goal or "").lower()))
        except Exception:
            return False

    def _is_trend_goal(self, goal_l, ir=None):
        try:
            if re.search(self._TREND_RE, str(goal_l or "")):
                return True
        except Exception:
            pass
        try:
            if (ir or {}).get("intent") == "trend":
                return True
        except Exception:
            pass
        return False

    def _safe_measure_candidates(self, measures):
        """Subset of `measures` passing the shared safe-numeric gate."""
        out = []
        for m in measures or []:
            try:
                if self._is_safe_numeric_measure(m.get("table"), m.get("column")):
                    out.append(m)
            except Exception:
                continue
        return out

    def _fold_ident_tokens(self, value):
        """Lowercase word tokens folding CamelCase + separators + quotes."""
        try:
            s = str(value or "").strip()
            while len(s) >= 2 and ((s[0] == '"' and s[-1] == '"')
                                   or (s[0] == "'" and s[-1] == "'")
                                   or (s[0] == "`" and s[-1] == "`")
                                   or (s[0] == "[" and s[-1] == "]")):
                s = s[1:-1].strip()
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
            s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
            return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w} | \
                {s.lower()} if s else set()
        except Exception:
            low = str(value or "").lower()
            return {low} if low else set()

    def _is_line_grain_table(self, table):
        """True for line-grain tables (generic grain words only, no names)."""
        try:
            toks = self._fold_ident_tokens(str(table or "").split(".")[-1])
            return bool(toks & {"line", "lines", "detail", "details",
                                "item", "items", "entry", "entries"})
        except Exception:
            return False

    def _is_parent_table(self, table, tables):
        """True when another table in `tables` holds an FK referencing `table`."""
        try:
            for other in (tables or []):
                if other == table:
                    continue
                info = self.tables.get(other, {}) or {}
                for fk in info.get("foreign_keys", []) or []:
                    if fk.get("referenced_table") == table:
                        return True
        except Exception:
            pass
        return False

    def _measure_name_tier(self, col_name, monetary_bias):
        """0 = money total, 1 = money line, 2 = neutral, 3 = quantity-like,
        10 = excluded for revenue-like (cost/fuel/id/count). Token-based so
        unit_price/fuel_price variants classify correctly."""
        try:
            toks = set(self._fold_ident_tokens(col_name))
        except Exception:
            toks = set()
        low = str(col_name or "").strip().strip("\"'`[]").lower()
        toks = set(toks) | {low}
        parts = set(low.split("_")) | set(toks)
        if monetary_bias and (parts & self._REVENUE_EXCLUDED_TOKENS):
            return 10
        if monetary_bias and ("id" in parts or parts & {"count", "cnt"}):
            return 10
        if not monetary_bias:
            return 2
        if parts & self._MONETARY_TOTAL_COLS or low in self._MONETARY_TOTAL_COLS:
            return 0
        if parts & self._MONETARY_LINE_COLS or low in self._MONETARY_LINE_COLS:
            return 1
        if parts & self._QUANTITY_COL_TOKENS:
            return 3
        return 2

    def _extract_value_candidates(self, goal):
        """Quoted phrases + Capitalized tokens that name no column.

        Generic: quoted/capitalized tokens matching a table/column name (or a
        protected/analytical/metric word) are NOT values; everything else is a
        candidate cell value to probe for in the data."""
        cands = []
        try:
            raw = str(goal or "")
            for m in re.finditer(r"'([^']{2,})'|\"([^\"]{2,})|`([^`]{2,})`", raw):
                v = next((g for g in m.groups() if g), None)
                if v and v.strip():
                    cands.append(v.strip())
            for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_\-]{2,}\b", raw):
                cands.append(m.group(0).strip())
            # De-dupe preserving order, drop junk.
            seen, out = set(), []
            schema_vocab = set()
            try:
                schema_vocab = {t.lower() for t in (self._schema_word_tokens or set())}
            except Exception:
                schema_vocab = set()
            stop = (set(self._ANALYTICAL_GOAL_WORDS) | set(self._METRIC_CONCEPT_WORDS)
                    | set(self._PROTECTED_GOAL_WORDS) | set(self._COUNT_OBJECT_TOKENS))
            for c in cands:
                low = c.lower()
                if low in seen or len(c.strip()) < 3:
                    continue
                if low in schema_vocab or low.rstrip("s") in schema_vocab:
                    continue
                if low in stop:
                    continue
                if re.fullmatch(r"\d+(\.\d+)?", low):
                    continue
                seen.add(low)
                out.append(c.strip())
            # Lowercase value words ("revenue for electronics"): goal tokens
            # naming no table/column/metric/analytical concept are cell-value
            # candidates too (bounded, generic — never invents values).
            try:
                _goal_toks = re.findall(r"[a-z][a-z0-9_\-]{2,}", str(goal or "").lower())
                for _tok in _goal_toks:
                    if len(out) >= 3 or _tok in seen:
                        break
                    if len(_tok) < 3 or _tok in stop or _tok in schema_vocab:
                        continue
                    if _tok.rstrip("s") in schema_vocab or (_tok + "s") in schema_vocab:
                        continue
                    if re.fullmatch(r"\d+(\.\d+)?", _tok):
                        continue
                    seen.add(_tok)
                    out.append(_tok)
            except Exception:
                pass
            return out[:3]
        except Exception:
            return []

    def _candidate_text_columns(self, tables, limit=3):
        """Up to `limit` text columns to probe for a cell value.

        Category/geo-like names first (generic signals only), then the rest;
        deterministic (alphabetical) and never ID/PK/FK columns."""
        try:
            scored = []
            for t in tables or []:
                info = self.tables.get(t, {}) or {}
                pk = set(info.get("primary_key", []) or [])
                fk_cols = {fk.get("column") for fk in info.get("foreign_keys", []) or []}
                for col in info.get("columns", {}) or {}:
                    low = str(col).lower()
                    if col in pk or col in fk_cols or low == "id" or low.endswith("_id"):
                        continue
                    try:
                        if self._classify_column(t, col) != "text":
                            continue
                    except Exception:
                        continue
                    toks = set(re.split(r"[^a-z0-9]+", low))
                    pref = 0 if (toks & self._GEO_CAT_TOKENS) else 1
                    scored.append((pref, t, col))
            scored.sort(key=lambda x: (x[0], x[1], x[2]))
            return [(t, c) for _, t, c in scored[:max(1, int(limit))]]
        except Exception:
            return []

    def _probe_value_filters(self, goal, tables):
        """Sample DISTINCT values (bounded: LIMIT 50 per column, max 3
        columns) to find which column holds a quoted/Capitalized token.

        Returns IR filter dicts with equality predicates. Never raises;
        returns [] when the engine is unavailable or nothing matches."""
        found = []
        try:
            cands = self._extract_value_candidates(goal)
            if not cands:
                return []
            cols = self._candidate_text_columns(tables, limit=3)
            if not cols:
                return []
            from sqlalchemy import text as _text
            try:
                quote = (lambda n: self._quote(n, getattr(self, "dialect", "postgresql")))
            except Exception:
                quote = lambda n: f'"{n}"'  # noqa: E731
            try:
                from core.db import base_table as _bt
            except Exception:
                _bt = lambda n: str(n or "").split(".")[-1]  # noqa: E731
            for cand in cands:
                if len(found) >= 2:
                    break
                for (t, c) in cols:
                    try:
                        # Bare segment: the engine search_path resolves
                        # schema-qualified keys (sales.employee -> employee).
                        sql = (f"SELECT DISTINCT {quote(c)} FROM {quote(_bt(t))} "
                               f"LIMIT 50")
                        with self.engine.connect() as conn:
                            rows = conn.execute(_text(sql)).fetchall()
                    except Exception:
                        continue
                    hit = False
                    hit_val = None
                    try:
                        for r in rows or []:
                            if str(r[0]).strip().lower() == str(cand).strip().lower():
                                hit = True
                                try:
                                    hit_val = str(r[0]).strip()
                                except Exception:
                                    hit_val = str(cand).strip()
                                break
                    except Exception:
                        continue
                    if hit:
                        # Preserve the database's own casing ('Electronics',
                        # not the goal's 'electronics'): equality is
                        # case-sensitive on SQLite/Postgres, so storing the
                        # goal casing would filter to zero rows.
                        _val = hit_val if hit_val else str(cand).strip()
                        found.append({
                            "column": f"{t}.{c}",
                            "operator": "=",
                            "value": _val,
                            "confidence": 0.85,
                            "reason": (f"value '{_val}' found in {t}.{c} "
                                       "(sampled distinct values)"),
                        })
                        break
        except Exception:
            return found
        return found

    def _estimate_column_cardinality(self, table, col, cap=101):
        """Bounded distinct-value estimate (DISTINCT LIMIT `cap`)."""
        try:
            from sqlalchemy import text as _text
            try:
                quote = (lambda n: self._quote(n, getattr(self, "dialect", "postgresql")))
            except Exception:
                quote = lambda n: f'"{n}"'  # noqa: E731
            try:
                from core.db import base_table as _bt2
            except Exception:
                _bt2 = lambda n: str(n or "").split(".")[-1]  # noqa: E731
            qualified = ".".join(quote(part) for part in str(table).split("."))
            sql = (f"SELECT DISTINCT {quote(col)} FROM {qualified} "
                   f"LIMIT {int(cap)}")
            with self.engine.connect() as conn:
                rows = conn.execute(_text(sql)).fetchall()
            return len(rows or [])
        except Exception:
            return None

    def _is_geo_cat_like(self, col):
        try:
            toks = set(self._fold_ident_tokens(col))
            return bool(toks & self._GEO_CAT_TOKENS)
        except Exception:
            return False

    def _is_person_name_like(self, table, col):
        try:
            toks = set(self._fold_ident_tokens(col))
            if not (toks & self._PERSON_NAME_TOKENS) and "name" not in toks:
                return False
            # A bare *product/region/category* name is a legit breakdown;
            # person names live on people tables or carry first/last tokens.
            if toks & {"product", "region", "territory", "category", "brand"}:
                return False
            if toks & {"first", "last", "middle", "full", "person", "contact"}:
                return True
            return True
        except Exception:
            return False

# ---- _resolve_semantics (_resolve_semantics) ----

    def _resolve_semantics(self, goal, ir, join_path):
        """Bind the Goal IR concepts to real schema columns/tables (spec §3).

        Two-layer and database-agnostic:
          1. Generic intent lives in the IR (no schema names).
          2. Schema grounding happens HERE: subjects, dimensions and metrics are
             resolved with synonym-aware matching (customers <-> clients,
             model year <-> model_year, ...) plus an explicit confidence and a
             human reason for every binding (spec §6). Never fabricates a
             mapping; unresolved concepts stay None and confidence drops.
        """
        relevant = [t for t in join_path if t in self.tables]
        if not relevant:
            relevant = [list(self.tables.keys())[0]] if self.tables else []
        # Current question only: a previous turn's measure words (e.g.
        # 'average') must not bias monetary/AOV/trend/value grounding. The
        # inherited IR metrics (elliptical follow-ups) already carry history
        # measures via their concepts, independent of this text.
        try:
            goal = self._current_question_text(goal)
        except Exception:
            goal = str(goal or "")
        goal_l = str(goal).lower()
        # Fresh per-question probe budget for text-numeric coercion (max 3
        # probed candidate columns; cache hits are free).
        try:
            self._coercion_budget = 3
        except Exception:
            pass

        measures = []
        for table in relevant:
            for col_name in self.tables[table].get("columns", {}):
                if self._is_measure_column(table, col_name):
                    measures.append({"table": table, "column": col_name})
        ir["measures"] = measures

        concept_tokens = {
            "aggregate total": {"total", "sum", "amount", "value", "revenue",
                                "sales", "price", "quantity", "qty", "profit",
                                "units", "cost", "volume"},
            "row count": set(),
            "average value": {"average", "avg", "mean", "value", "price", "amount", "total"},
            "maximum value": {"maximum", "max", "value", "price", "amount"},
            "minimum value": {"minimum", "min", "value", "price", "amount"},
            "profit kpi": {"profit", "margin", "net", "income"},
            "revenue kpi": {"revenue", "amount", "total", "sales", "price"},
            "sales kpi": {"sales", "revenue", "amount", "total", "quantity", "qty", "price"},
        }

        # -- 1) SUBJECT grounding (synonym-aware table matching) -------------
        matched_tables = self._match_tables(goal_l)
        goal_tokens = self._normalize_tokens(goal_l)
        metric_present = bool(goal_tokens & self._METRIC_CONCEPT_WORDS)
        entity_tokens = {
            t for t in goal_tokens
            if t not in self._ANALYTICAL_GOAL_WORDS and t not in self._METRIC_CONCEPT_WORDS
        }
        if metric_present:
            # When the goal states a metric ("average order VALUE per customer")
            # bare count/fact nouns ("order", "sales", "quantity") belong to the
            # measure, not the breakdown entity.
            entity_tokens -= self._COUNT_OBJECT_TOKENS
        if matched_tables:
            subject, subject_tables = self._pick_subject(
                matched_tables, goal_tokens, entity_tokens
            )
        else:
            subject, subject_tables = None, []

        # -- 2) DIMENSION grounding -----------------------------------------
        dims = []
        # Label dimensions are bound to the entity (subject) tables the goal
        # actually names — never to pure-fact tables that only appear because a
        # generic word matched their name. This keeps "average order value per
        # customer" from dragging orders.ship_name / order_date into GROUP BY.
        for _score, table in subject_tables:
            label_col = self._pick_label_column(table)
            if not label_col:
                # No human-readable label on this table (e.g. a pure link
                # table with only keys/dates): a dimension with a NULL
                # column would force its table into required_tables and
                # fail semantic validation even though the SQL can never
                # emit it. Skip it here.
                continue
            dims.append({
                "concept": table,
                "resolved_table": table,
                "resolved_column": label_col,
                "confidence": 0.9,
                "reason": "subject table named (or meant) in the question",
            })
        # Explicit breakdown columns ("by X", "per X", "for each X") are bound
        # across the join path from tables whose column names the breakdown
        # tokens resolve to ("by model year" -> products.model_year,
        # "by order status" -> transactions.status). Only from columns that are
        # legitimate dimensions (excludes PK/FK/*_id and pure measures).
        breakdown_tokens = self._explicit_breakdown_tokens(goal_l)
        _explicit_matched = set()
        if breakdown_tokens:
            breakdown_str = " ".join(sorted(breakdown_tokens))
            # Search every table (not just the join path) so "by region"
            # finds a territories/dim table the keyword scorer missed.
            _search_tables = list(dict.fromkeys(
                list(relevant) + [t for t in self.tables if t not in relevant]))
            for table in _search_tables:
                try:
                    cols = self._match_dimension_columns(breakdown_str, table)
                except Exception:
                    cols = []
                if cols:
                    label_col = self._pick_label_column(table)
                    for _score, col in cols:
                        if col != label_col:
                            dims.append({
                                "concept": col,
                                "resolved_table": table,
                                "resolved_column": col,
                                "confidence": 0.95,
                                "reason": "explicit breakdown column named in the question",
                            })
                            _explicit_matched.add((table, col))
        # Explicit 'by X' outranks in SQL ranking via higher confidence
        # (0.95) + name-score preference there; subject-label dims are kept
        # so "by category and model year" still grounds the category table
        # whose breakdown IS its label column.
        # Generic segment/breakdown/comparison intent with no specific column
        # matched: auto-detect all categorical columns as dimensions so queries
        # like "Compare segments breakdown" produce a real GROUP BY instead of
        # a single aggregate value.
        _GENERIC_SEGMENT_WORDS = {
            "segment", "segments", "breakdown", "category", "categories",
            "group", "groups", "split", "compare", "comparison",
        }
        if not dims and breakdown_tokens & _GENERIC_SEGMENT_WORDS:
            for table in relevant:
                for col_name in self.tables.get(table, {}).get("columns", {}):
                    try:
                        role = self._classify_column(table, col_name)
                    except Exception:
                        continue
                    if role == "text":
                        dims.append({
                            "concept": col_name,
                            "resolved_table": table,
                            "resolved_column": col_name,
                            "confidence": 0.7,
                            "reason": "generic segment/breakdown intent — auto-detected categorical column",
                        })
        if not dims and len(relevant) == 1:
            # Single-table fallback (Walmart 'holiday vs non-holiday', bare
            # 'monthly trend' with no by/per): the breakdown may live in the
            # goal tokens without an explicit marker, or the subject match may
            # be empty. Ground any dimension-like column the goal names rather
            # than returning no dimensions. Multi-table paths already carry
            # subject dims, so this only fires when nothing else grounded.
            table = relevant[0]
            cols = self._match_dimension_columns(goal_l, table)
            for _score, col in cols[:2]:
                if (table, col) not in {(d.get("resolved_table"), d.get("resolved_column")) for d in dims}:
                    dims.append({
                        "concept": col,
                        "resolved_table": table,
                        "resolved_column": col,
                        "confidence": 0.7,
                        "reason": "single-table dimension named in the question",
                    })
        # Trend/time intent REQUIRES a datetime GROUP BY: scalar aggregates
        # are invalid for trends. Detect robustly (regex, not token split,
        # so "changed over time?" and "history" match) and always ensure a
        # date dimension exists — even when subject dims already grounded.
        if self._is_trend_goal(goal_l, ir):
            _has_date_dim = False
            try:
                for d in dims:
                    if self._classify_column(
                            d.get("resolved_table"), d.get("resolved_column")) == "date":
                        _has_date_dim = True
                        break
            except Exception:
                pass
            if not _has_date_dim:
                _date_found = None
                for table in list(relevant) + [t for t in self.tables if t not in relevant]:
                    for col_name in self.tables.get(table, {}).get("columns", {}):
                        try:
                            role = self._classify_column(table, col_name)
                        except Exception:
                            continue
                        if role == "date":
                            _date_found = (table, col_name)
                            break
                    if _date_found:
                        break
                if _date_found:
                    dims.append({
                        "concept": _date_found[1],
                        "resolved_table": _date_found[0],
                        "resolved_column": _date_found[1],
                        "confidence": 0.85,
                        "reason": "trend/time intent — datetime grouping is required",
                    })
        # --- deterministic dimension pruning (schema-agnostic) ---------
        # 1) Drop NULL/unknown columns (can never be emitted in SQL).
        # 2) Scalar summaries (overall aggregate, no ranking/comparison/
        #    trend/breakdown) carry no GROUP BY: subject-label dims would
        #    force spurious tables into required_tables and fail semantic
        #    validation after the SQL correctly narrows to the measure
        #    home. Clear them.
        # 3) A single breakdown token matching dozens of columns across
        #    tables ("region" -> city/country/ship_region/... on every
        #    table) is capped to ONE best dimension: lowest-cardinality
        #    geo/category text column on a connected (non-isolated)
        #    table. Multi-token breakdowns ("by category and model
        #    year") keep up to one dim per token so genuine multi-dim
        #    goals still ground.
        try:
            dims = [d for d in dims
                    if d.get("resolved_table") in self.tables
                    and d.get("resolved_column")]
        except Exception:
            pass
        try:
            _is_trend_here = bool(self._is_trend_goal(goal_l, ir))
        except Exception:
            _is_trend_here = False
        try:
            _has_signal = bool(
                (ir or {}).get("ranking") is not None
                or (ir or {}).get("comparison") is not None
                or _explicit_matched
                or (breakdown_tokens & _GENERIC_SEGMENT_WORDS)
                or _is_trend_here
            )
        except Exception:
            _has_signal = True
        if not _has_signal and (ir or {}).get("intent") == "summary":
            dims = [d for d in dims
                    if d.get("reason") == "trend/time intent — datetime grouping is required"]
        elif breakdown_tokens and len(dims) > max(1, len(breakdown_tokens)):
            # Rank all candidate dims; keep the best N where N = number
            # of distinct breakdown tokens (single-token "by region" ->
            # exactly 1; multi-token keeps one per concept).
            try:
                _cap_n = max(1, len(set(breakdown_tokens)))
            except Exception:
                _cap_n = 1
            try:
                _graph = getattr(self, "relationship_graph", {}) or {}
            except Exception:
                _graph = {}
            try:
                _goal_toks = set(self._normalize_tokens(goal_l)) | set(breakdown_tokens or [])
            except Exception:
                _goal_toks = set()
            try:
                _bdn_toks = {str(_b).lower() for _b in (breakdown_tokens or set()) if str(_b or "")}
            except Exception:
                _bdn_toks = set()
            def _rank_key(d):
                _t = d.get("resolved_table")
                _c = d.get("resolved_column") or ""
                # Goal-named dims win the cap: "by country" must keep the
                # country table even when another table has fewer groups.
                # Synonym/substring-aware so compounds match. Generic.
                _named = 1
                _direct = 1
                try:
                    _stm = getattr(self, "_schema_token_matches_goal", None)
                    _parts = [p for seg in str(_t or "").split(".")
                              for p in seg.split("_") if p]
                    _parts += [p for p in str(_c or "").split("_") if p]
                    if callable(_stm) and _goal_toks:
                        for _p in _parts:
                            for _gw in _goal_toks:
                                if _stm(_p, _gw):
                                    _named = 0
                                    break
                            if _named == 0:
                                break
                    # Direct breakdown-token hit outranks synonym-only hits:
                    # 'country' is literally inside 'countryregion' but only
                    # a synonym of 'location'. Generic (leading-substring
                    # either way, min length 4).
                    if _bdn_toks:
                        for _p in _parts:
                            for _bw in _bdn_toks:
                                if len(_p) >= 4 and len(_bw) >= 4 and (
                                        _p.startswith(_bw) or _bw.startswith(_p)):
                                    _direct = 0
                                    break
                            if _direct == 0:
                                break
                except Exception:
                    pass
                try:
                    _geo = bool(self._is_geo_cat_like(_c))
                except Exception:
                    _geo = False
                try:
                    _deg = len(_graph.get(_t, []) or [])
                except Exception:
                    _deg = 0
                _isolated = 1 if _deg == 0 else 0
                try:
                    _card = self._estimate_column_cardinality(_t, _c)
                except Exception:
                    _card = None
                _card_v = _card if isinstance(_card, int) else 10 ** 9
                try:
                    _conf = float(d.get("confidence") or 0)
                except Exception:
                    _conf = 0
                return (_direct, _named, _isolated, 0 if _geo else 1, _card_v,
                        -_conf, str(_t or ""), str(_c or ""))
            try:
                # Keep date dims required by trends outside the cap;
                # cap only the categorical breakdown candidates.
                _date_ds = []
                _cat_ds = []
                for _d in dims:
                    try:
                        _role = self._classify_column(
                            _d.get("resolved_table"), _d.get("resolved_column"))
                    except Exception:
                        _role = None
                    if _role == "date":
                        _date_ds.append(_d)
                    else:
                        _cat_ds.append(_d)
                _cat_ds = sorted(_cat_ds, key=_rank_key)[:_cap_n]
                dims = _date_ds + _cat_ds
            except Exception:
                pass
        seen_dims = set()
        for d in dims:
            key = (d["resolved_table"], d["resolved_column"])
            if key in seen_dims:
                continue
            seen_dims.add(key)
            ir["dimensions"].append(d)

        # -- 3) METRIC resolution -------------------------------------------
        for metric in ir["metrics"]:
            concept = metric["concept"].lower()
            if metric.get("aggregation") == "COUNT" or concept == "row count":
                # A vague trend goal ("analyze trend over time") names no
                # measure: tracking the table's real measure over time is the
                # useful default, not a row count. Only inferred (guessed)
                # counts are upgraded — an explicit "how many"/"count" stays
                # COUNT(*). Fully schema-driven, no column names assumed.
                if metric.get("inferred") and ir.get("intent") == "trend":
                    upgraded = None
                    for table in relevant:
                        try:
                            col = self._pick_measure_column(table)
                        except Exception:
                            col = None
                        if col:
                            upgraded = (table, col)
                            break
                    if upgraded:
                        table, col = upgraded
                        metric["resolved_table"] = table
                        metric["resolved_column"] = col
                        metric["resolved_expression"] = f"{table}.{col}"
                        metric["aggregation"] = "SUM"
                        metric["confidence"] = 0.7
                        metric["reason"] = (
                            "no explicit measure for a trend goal; tracking "
                            f"the {table}.{col} measure over time"
                        )
                        continue
                fact = self._pick_fact_table(goal_l, relevant, subject)
                metric["resolved_table"] = fact
                metric["resolved_column"] = None
                metric["resolved_expression"] = "COUNT(*)"
                if metric.get("inferred"):
                    metric["confidence"] = 0.6
                    metric["reason"] = (
                        "no explicit measure; count of related records "
                        f"in {fact or 'the fact table'}"
                    )
                else:
                    metric["confidence"] = 0.9
                continue
            tokens = concept_tokens.get(concept, {concept})
            best, best_key = None, None
            # Aggregate-monetary concepts prefer a monetary total column
            # (amount/total/revenue/sales) over a per-line price or a
            # quantity/units fact column when match scores tie, so "overall
            # sales" binds sales.amount rather than sale_lines.quantity
            # regardless of the order tables appear in the join path.
            monetary_concepts = {
                "sales kpi", "revenue kpi", "profit kpi",
            }
            monetary_bias = concept in monetary_concepts or bool(
                self._is_revenue_like_goal(goal_l, {"metrics": [metric]}))
            try:
                if not monetary_bias and self._is_aov_goal(goal):
                    monetary_bias = True
            except Exception:
                pass
            # "amount/total/value" questions prefer money totals; an explicit
            # quantity question ("overall quantity") keeps quantity unbiased.
            # Generic: driven only by the goal wording, never by DB names.
            try:
                _qty_explicit = bool(re.search(
                    r"\bquantity\b|\bqty\b|\bunits?\b|\bvolume\b", goal_l))
                if concept == "aggregate total" and not _qty_explicit:
                    monetary_bias = True
            except Exception:
                pass
            # AOV intent = AVG of the money measure (never AVG(cost)).
            try:
                if self._is_aov_goal(goal) and not metric.get("inferred"):
                    metric["aggregation"] = "AVG"
            except Exception:
                pass
            # Candidate pool: verified safe-numeric measures first (shared
            # gate in goal_agent/sql.py); fall back to all measures only
            # when nothing safe exists.
            try:
                _safe_pool = self._safe_measure_candidates(measures)
            except Exception:
                _safe_pool = []
            _pool = _safe_pool or measures
            for m in _pool:
                s = 0
                try:
                    col_tokens = set(self._fold_ident_tokens(m["column"]))
                except Exception:
                    col_tokens = self._normalize_tokens(m["column"])
                col_name = str(m["column"] or "").lower()
                for tok in tokens:
                    if tok in col_tokens:
                        s += 2
                    elif tok.endswith("s") and tok[:-1] in col_tokens:
                        s += 1
                # A measure only competes if a concept token actually matched
                # its column; otherwise `best` stays None and the
                # computed-measure path (quantity * unit_price) runs below.
                if s == 0:
                    continue
                tier = self._measure_name_tier(m["column"], monetary_bias)
                if tier >= 10:
                    continue
                # Header-vs-line disambiguation (generic): when both a
                # header-level monetary total and a line-level total exist,
                # prefer the header total (non-line table, FK-referenced
                # parent). Line-grain tables and unreferenced children sort
                # last so qty*price synthesis only wins when no header
                # total exists. For monetary concepts the tier outranks the
                # raw token score so a per-line price with an extra plural
                # hit never beats a header total. No table/column literals.
                try:
                    _line_pen = 1 if self._is_line_grain_table(
                        m.get("table")) else 0
                except Exception:
                    _line_pen = 0
                try:
                    _parent_bonus = 0 if self._is_parent_table(
                        m.get("table"), relevant) else 1
                except Exception:
                    _parent_bonus = 1
                if monetary_bias:
                    key = (-tier, s, -_line_pen, -_parent_bonus, col_name)
                else:
                    key = (s, -tier, -_line_pen, -_parent_bonus, col_name)
                if best_key is None or key > best_key:
                    best_key = key
                    best = m
            # Goal-name override: if the user explicitly names a column
            # (e.g. "average rating" -> "rating"), prefer it over the
            # concept-token match which may pick the wrong column. Never
            # overrides a revenue-like goal with a cost/fuel/id/count column.
            if not monetary_bias:
                try:
                    goal_words = {w for w in self._normalize_tokens(goal_l) if len(w) >= 3}
                except Exception:
                    goal_words = {w for w in goal_l.split() if len(w) >= 3}
                for m in _pool:
                    col_lower = m["column"].lower()
                    col_base = col_lower.split("_")[0]
                    if col_lower in goal_words or col_base in goal_words:
                        best = m
                        break
            # If nothing matched (best is None) the computed-measure path
            # below synthesizes a quantity*unit_price expression.
            # Also prefer the computed expression when monetary_bias is on
            # and the best match is only a per-line price column (tier 1),
            # because revenue/sales should be qty * price * (1-discount).
            try:
                computed = self._find_price_quantity_expression(relevant, tokens)
            except Exception:
                computed = None
            if best:
                best_tier = self._measure_name_tier(best["column"], monetary_bias)
                # For monetary concepts, prefer computed expression over a bare
                # price/line column (tier >= 1), since revenue = qty*price*(1-disc).
                # A header-level monetary total (tier 0) always wins over the
                # line-grain synthesis: header totals are already the
                # qty*price rollup and re-multiplying lines double-counts.
                if computed and best_tier >= 1 and monetary_bias:
                    metric["resolved_table"] = computed["table"]
                    metric["resolved_column"] = None
                    metric["resolved_expression"] = computed["expression"]
                    metric["computed_measure"] = True
                    metric["confidence"] = 0.85
                    metric["reason"] = (
                        f"monetary concept; computed "
                        f"{computed['expression']} from the schema"
                    )
                else:
                    metric["resolved_table"] = best["table"]
                    metric["resolved_column"] = best["column"]
                    metric["resolved_expression"] = f"{best['table']}.{best['column']}"
                    metric["confidence"] = 0.9
                    metric["reason"] = "measure column matched the requested metric"
            else:
                # Spec §3: sales-like concept with no single column -> try a
                # computed quantity * unit_price expression from the schema.
                # Search the whole schema (not just the join path) so a
                # mis-scoped `relevant` set never forces AVG(cost).
                try:
                    computed = self._find_price_quantity_expression(relevant, tokens)
                except Exception:
                    computed = None
                if not computed:
                    try:
                        _all_tables = [t for t in self.tables if t not in relevant]
                        computed = self._find_price_quantity_expression(
                            list(relevant) + _all_tables, tokens)
                    except Exception:
                        computed = None
                if not computed and monetary_bias and concept != "profit kpi":
                    # Money-preferred fallback across the whole schema:
                    # first safe total/amount/sales-like column, never
                    # cost/fuel/id/counts. Generic, no DB names.
                    try:
                        for t in list(relevant) + [x for x in self.tables if x not in relevant]:
                            for col in self.tables.get(t, {}).get("columns", {}) or {}:
                                try:
                                    if not self._is_safe_numeric_measure(t, col):
                                        continue
                                except Exception:
                                    continue
                                if self._measure_name_tier(col, True) >= 10:
                                    continue
                                toks = self._normalize_tokens(col)
                                if toks & {"total", "amount", "sales", "revenue", "price", "value"}:
                                    best = {"table": t, "column": col}
                                    break
                            if best:
                                break
                    except Exception:
                        pass
                if computed:
                    metric["resolved_table"] = computed["table"]
                    metric["resolved_column"] = None
                    metric["resolved_expression"] = computed["expression"]
                    metric["computed_measure"] = True
                    metric["confidence"] = 0.8
                    metric["reason"] = (
                        f"no single measure column; computed "
                        f"{computed['expression']} from the schema"
                    )
                elif best:
                    # Money-preferred fallback hit (whole-schema scan).
                    metric["resolved_table"] = best["table"]
                    metric["resolved_column"] = best["column"]
                    metric["resolved_expression"] = f"{best['table']}.{best['column']}"
                    metric["confidence"] = 0.75
                    metric["reason"] = ("money-preferred safe measure "
                                        "for a revenue-like question")

        # Coerced text-numeric fallback (schema-agnostic): any SUM/AVG/MIN/
        # MAX metric still unresolved here (all-text schemas have no
        # dtype-numeric measures, so the pools above stay empty) is bound to
        # a measure-like TEXT column whose bounded DISTINCT probe shows
        # >=80% coercible values. Carries its coercion expression; GROUP
        # BY/WHERE on the column stay raw.
        try:
            _unresolved = [
                m for m in ir.get("metrics", []) or []
                if not m.get("resolved_table")
                and str(m.get("aggregation") or "").upper() in (
                    "SUM", "AVG", "MIN", "MAX")
                and str(m.get("concept") or "").lower() != "row count"
            ]
        except Exception:
            _unresolved = []
        if _unresolved:
            try:
                _scan_tables = list(dict.fromkeys(
                    list(relevant) + [t for t in self.tables
                                      if t not in relevant]))
            except Exception:
                _scan_tables = list(relevant)
            for _metric in _unresolved:
                try:
                    _concept = str(_metric.get("concept") or "").lower()
                    _tokens = concept_tokens.get(_concept, {_concept})
                    self._resolve_coerced_measure(
                        _metric, _scan_tables, goal_l, _tokens)
                except Exception:
                    continue

        # AOV with no explicit AVG metric (e.g. bare "AOV"): upgrade the
        # inferred COUNT to AVG over the money measure. Generic, no names.
        try:
            if self._is_aov_goal(goal):
                for metric in ir["metrics"]:
                    if metric.get("resolved_expression") == "COUNT(*)" and metric.get("inferred"):
                        _money = None
                        for t in list(relevant) + [x for x in self.tables if x not in relevant]:
                            for col in self.tables.get(t, {}).get("columns", {}) or {}:
                                try:
                                    if not self._is_safe_numeric_measure(t, col):
                                        continue
                                except Exception:
                                    continue
                                if self._measure_name_tier(col, True) >= 10:
                                    continue
                                if self._normalize_tokens(col) & {
                                        "total", "amount", "sales", "revenue", "price", "value"}:
                                    _money = (t, col)
                                    break
                            if _money:
                                break
                        if _money:
                            metric["resolved_table"] = _money[0]
                            metric["resolved_column"] = _money[1]
                            metric["resolved_expression"] = f"{_money[0]}.{_money[1]}"
                            metric["aggregation"] = "AVG"
                            metric["confidence"] = 0.75
                            metric["reason"] = ("AOV intent — AVG of the money measure")
        except Exception:
            pass

        # Value filters ("revenue for electronics"): quoted/Capitalized
        # tokens naming no column are probed as cell values (bounded
        # DISTINCT sampling, max 3 columns). Generic, works on any schema.
        try:
            _tables_for_values = list(dict.fromkeys(
                list(relevant) + [d.get("resolved_table") for d in ir.get("dimensions", []) or []
                                  if d.get("resolved_table")]
                + [m.get("resolved_table") for m in ir.get("metrics", []) or []
                   if m.get("resolved_table")]))
            _tables_for_values = [t for t in _tables_for_values if t in self.tables]
            if not _tables_for_values:
                _tables_for_values = list(relevant)
            _vf = self._probe_value_filters(goal, _tables_for_values)
            _have = {(str(f.get("column") or "").lower(), str(f.get("value") or "").lower())
                     for f in ir.get("filters", []) or []}
            for _f in _vf or []:
                if (_f.get("column", "").lower(), _f.get("value", "").lower()) not in _have:
                    ir.setdefault("filters", []).append(_f)
        except Exception:
            pass

        if ir.get("time"):
            date_col = self._first_date_column(relevant)
            if date_col is None:
                try:
                    date_col = self._first_date_column(
                        [t for t in self.tables if t not in relevant])
                except Exception:
                    date_col = None
            if date_col:
                ir["time"]["column"] = f"{date_col[0]}.{date_col[1]}"
                for f in ir["filters"]:
                    if f["column"] is None:
                        f["column"] = f"{date_col[0]}.{date_col[1]}"

        resolved_any = any(m.get("resolved_table") for m in ir["metrics"]) or bool(ir["dimensions"])
        ir["confidence"] = 0.85 if resolved_any else 0.5
        ir["reasoning"] = {
            "intent_confidence": 0.9 if ir.get("operation") else 0.5,
            "schema_confidence": ir["confidence"],
            "relationship_confidence": 0.7,
        }
        required = set()
        for m in ir["metrics"]:
            if m.get("resolved_table"):
                required.add(m["resolved_table"])
        for d in ir["dimensions"]:
            if d.get("resolved_table"):
                required.add(d["resolved_table"])
        for f in ir.get("filters", []) or []:
            try:
                _col = str(f.get("column") or "")
                if "." in _col and _col.split(".")[0] in self.tables:
                    required.add(_col.split(".")[0])
            except Exception:
                pass
        ir["required_tables"] = (sorted(required) if required else list(join_path))
        return ir

