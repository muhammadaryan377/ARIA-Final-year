"""Generic multi-table relational handling (no hardcoded dataset names).

Single shared layer for table discovery, column typing, PK inference
(single + composite), FK inference via name/type overlap + value validation,
relationship graph, join planning, and duplicate-column disambiguation.

Works for ANY dataset (Walmart single-table, stores/sales splits, SQLite DBs,
CSV folders) because it only uses structural signals:

- lexical: normalized names, token overlap, table-name hints
  (via ``schema_engine.lexical`` -- never literal table/column names)
- types: coarse families (numeric / text / datetime / boolean)
- values: containment of source distinct values in target distinct values
  (threshold-gated, NULL-ignorant)

A lone table validly yields zero relationships (``relationships.none``).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

try:  # reuse the generic lexical signals; fall back to local copies offline
    from schema_engine.lexical import (
        is_code_column,
        is_ref_flavored,
        normalize_identifier,
        singularize,
        table_name_candidates,
        tokenize_words,
    )
except Exception:  # pragma: no cover
    import re as _re

    def normalize_identifier(v: str) -> str:
        return _re.sub(r"[^a-z0-9]+", "", str(v).lower())

    def singularize(n: str) -> str:
        if n.endswith("ies") and len(n) > 4:
            return n[:-3] + "y"
        if n.endswith("s") and not n.endswith("ss") and len(n) > 2:
            return n[:-1]
        return n

    def tokenize_words(n: str):
        s = _re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(n))
        return [w for w in _re.split(r"[^a-zA-Z0-9]+", s.lower()) if w]

    def table_name_candidates(c, t):
        return []

    def is_code_column(c: str) -> bool:
        return False

    def is_ref_flavored(n) -> bool:
        low = str(n or "").lower()
        return low == "id" or low.endswith(("_id", "_key", "_ref", "_no", "_num"))

MIN_OVERLAP = 0.85
MIN_SAMPLES = 2

_NUMERIC_TOKENS = (
    "int", "bigint", "smallint", "serial", "bigserial", "numeric",
    "decimal", "number", "float", "double", "real", "money",
)
_TEXT_TOKENS = ("char", "varchar", "text", "citext", "string", "uuid",
                "clob", "enum", "name")
_DATETIME_TOKENS = ("date", "time", "timestamp", "datetime", "interval", "year")
_BOOLEAN_TOKENS = ("bool", "boolean", "bit")


def _canon_ident(value: Any) -> str:
    """Generic identifier match key: lower + strip quotes/brackets/whitespace."""
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


def _bare_key(value: Any) -> str:
    """Bare last dot-segment, canonicalized (multi-schema aware)."""
    try:
        segs = [s for s in str(value or "").split(".") if s.strip()]
        last = segs[-1] if segs else str(value or "")
        return _canon_ident(last)
    except Exception:
        return _canon_ident(value)


def _resolve_table_key(name: Any, tables) -> str | None:
    """Resolve `name` to a real table key (exact, case-insensitive, bare-segment).

    Bare-segment fallback applies only when exactly one table shares that
    bare name (search_path semantics); ambiguous bare names resolve to None
    so the database remains the final arbiter.
    """
    try:
        keys = list((tables or {}).keys() if isinstance(tables, dict) else (tables or []))
    except Exception:
        return None
    if name in keys:
        return name  # type: ignore[return-value]
    low = _canon_ident(name)
    for k in keys:
        if _canon_ident(k) == low:
            return k
    bare = _bare_key(name)
    cands = [k for k in keys if _bare_key(k) == bare]
    if len(cands) == 1:
        return cands[0]
    return None


def quote_identifier(name: Any, dialect: str = "sqlite") -> str:
    safe = str(name or "")
    if (dialect or "") == "mysql":
        return "`" + safe.replace("`", "``") + "`"
    return '"' + safe.replace('"', '""') + '"'


def qualify(table: str, column: str, dialect: str = "sqlite") -> str:
    return f"{quote_identifier(table, dialect)}.{quote_identifier(column, dialect)}"


def type_family(dtype: Any) -> str:
    dt = str(dtype or "").strip().lower()
    if any(t in dt for t in _BOOLEAN_TOKENS) or dt == "boolean":
        return "boolean"
    if any(t in dt for t in _DATETIME_TOKENS):
        return "datetime"
    if any(t in dt for t in _NUMERIC_TOKENS):
        return "numeric"
    if any(t in dt for t in _TEXT_TOKENS) or not dt:
        return "text"
    return "other"


def types_compatible(a: Any, b: Any) -> bool:
    fa, fb = type_family(a), type_family(b)
    if not a or not b:
        return True  # unknown -> never reject on types alone
    if fa == fb:
        return True
    if {fa, fb} <= {"numeric", "text", "other"}:
        # INTEGER pk <-> TEXT fk happens with zero-padded ids; allow when
        # the name corroborates, value overlap decides.
        return True
    return False


def duplicate_columns(tables_columns: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Map lowercase column name -> sorted tables sharing it (only dupes)."""
    owners: Dict[str, List[str]] = defaultdict(list)
    for table, cols in (tables_columns or {}).items():
        for col in cols or []:
            owners[_canon_ident(col)].append(table)
    return {c: sorted(set(ts)) for c, ts in owners.items() if len(set(ts)) > 1}


def resolve_column(column: str, tables_columns: Dict[str, List[str]],
                   preferred: str | None = None) -> List[str]:
    """All qualified ``table.column`` owners of ``column`` (generic disambiguation).

    ``preferred`` (a table name) is listed first when it owns the column.
    Returns [] when no table owns it.
    """
    low = _canon_ident(column)
    owners = [t for t, cols in (tables_columns or {}).items()
              if any(_canon_ident(c) == low for c in (cols or []))]
    owners = sorted(set(owners))
    real_preferred = _resolve_table_key(preferred, tables_columns)
    if real_preferred in owners:
        owners = [real_preferred] + [t for t in owners if t != real_preferred]
    real = {}
    for t in owners:
        for c in tables_columns[t]:
            if _canon_ident(c) == low:
                real[t] = c
                break
    return [f"{t}.{real[t]}" for t in owners]


def _norm_values(values) -> set:
    out = set()
    for v in values or []:
        if v is None:
            continue
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "null", "nan"):
            continue
        out.add(s)
    return out


def _name_hint(src_col: str, tgt_table: str, tgt_col: str,
               all_tables) -> bool:
    if normalize_identifier(src_col) == normalize_identifier(tgt_col):
        return True
    try:
        if tgt_table in (table_name_candidates(src_col, all_tables) or []):
            return True
    except Exception:
        pass
    # token overlap: e.g. src `store` <-> table `stores`
    try:
        src_toks = {singularize(t) for t in tokenize_words(src_col)} - {"id"}
        tgt_toks = {singularize(t) for t in tokenize_words(str(tgt_table).rsplit(".", 1)[-1])}
        if src_toks and (src_toks & tgt_toks):
            return True
    except Exception:
        pass
    return False


def infer_foreign_keys(tables_columns: Dict[str, Dict[str, str]],
                       pk_map: Dict[str, List[str]],
                       value_sets: Dict[Tuple[str, str], set] | None = None,
                       min_overlap: float = MIN_OVERLAP,
                       ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Generic FK inference.

    ``tables_columns``: {table: {column: data_type}}.
    ``pk_map``: {table: [pk cols]} (single or composite; [] valid).
    ``value_sets``: {(table, column): set(str values)} or None (name-only).

    Returns (inferred, ambiguous, edges). Every edge carries both
    ``source_/target_`` (schema-mapping contract) and ``left_/right_``
    (goal-agent contract) endpoint spellings plus trust/confidence.
    """
    tables = sorted((tables_columns or {}).keys())
    inferred: List[dict] = []
    ambiguous: List[dict] = []
    # unique cols: PK members are unique by definition; value_sets let us
    # additionally prove uniqueness for non-PK targets (distinct == count).
    unique: Dict[Tuple[str, str], bool] = {}
    for t in tables:
        for c in (tables_columns.get(t) or {}):
            unique[(t, c)] = c in (pk_map.get(t) or [])

    for src in tables:
        src_cols = tables_columns.get(src) or {}
        sole_pk = pk_map.get(src) or []
        for s_col, s_type in src_cols.items():
            if len(sole_pk) == 1 and s_col in sole_pk:
                continue  # own identity is never its own FK
            cands = []
            for tgt in tables:
                if tgt == src:
                    continue  # single-table datasets yield none (valid)
                for t_col, t_type in ((tables_columns.get(tgt)) or {}).items():
                    if not types_compatible(s_type, t_type):
                        continue
                    hint = _name_hint(s_col, tgt, t_col, tables)
                    ratio = None
                    matched = 0
                    n_src = 0
                    if value_sets is not None:
                        s_vals = _norm_values(value_sets.get((src, s_col), set()))
                        t_vals = _norm_values(value_sets.get((tgt, t_col), set()))
                        n_src = len(s_vals)
                        if n_src >= MIN_SAMPLES and t_vals:
                            matched = len(s_vals & t_vals)
                            ratio = matched / n_src
                    tgt_unique = unique.get((tgt, t_col), False)
                    if value_sets is not None and not tgt_unique and ratio is not None:
                        # prove target uniqueness from values when PK unknown
                        t_vals = _norm_values(value_sets.get((tgt, t_col), set()))
                        if t_vals and len(t_vals) >= MIN_SAMPLES:
                            tgt_unique = True  # distinct-sampled; containment decides
                    # ---- acceptance per candidate ----
                    if ratio is not None and ratio >= min_overlap and n_src >= MIN_SAMPLES:
                        if not tgt_unique and not hint:
                            continue  # uncorroborated overlap into non-key
                        code = False
                        try:
                            code = bool(is_code_column(s_col))
                        except Exception:
                            code = False
                        if code and not hint:
                            continue  # code-like veto (policy parity)
                        score = 85 if hint else 65
                        state = "CONFIRMED" if (hint and score >= 80) else (
                            "PROBABLE" if score >= 60 else "UNCERTAIN")
                        conf = "data-confirmed" if hint else "data-inferred"
                        cands.append((score, ratio, matched, tgt, t_col, hint,
                                      state, conf, tgt_unique))
                    elif value_sets is None and hint:
                        # name-only: visible for review, never auto-joinable
                        cands.append((45, 0.0, 0, tgt, t_col, True,
                                      "UNCERTAIN", "heuristic-name-match", tgt_unique))
            if not cands:
                continue
            cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
            best = cands[0]
            # ambiguity: runner-up within margin on a different table
            amb = any(b[3] != best[3] and (best[0] - b[0]) < 20
                      and abs(best[1] - b[1]) < 0.05 for b in cands[1:2])
            _score, _ratio, matched, tgt, t_col, hint, state, conf, tgt_unique = best
            card = "many_to_one" if tgt_unique else "many_to_many"
            record = {
                "table": src, "column": s_col,
                "references_table": tgt, "references_column": t_col,
                "confidence": conf, "confidence_score": _score,
                "relationship_state": state,
                "review_status": "auto-accept" if state == "CONFIRMED" else (
                    "flagged" if state == "PROBABLE" else "review"),
                "ambiguous": bool(amb),
                "cardinality": card,
                "note": (f"{'Name + value' if hint and matched else 'Value' if matched else 'Name'} "
                         f"evidence: {src}.{s_col} -> {tgt}.{t_col}"
                         + (f" ({matched} values, {best[1]:.0%})" if matched else "")),
            }
            (ambiguous if amb else inferred).append(record)

    edges = []
    for r in inferred:
        edges.append({
            "source_table": r["table"], "source_column": r["column"],
            "target_table": r["references_table"], "target_column": r["references_column"],
            "left_table": r["table"], "left_column": r["column"],
            "right_table": r["references_table"], "right_column": r["references_column"],
            "type": "inferred", "relationship_type": "inferred",
            "confidence": r["confidence"], "confidence_score": r.get("confidence_score"),
            "trust": ("HIGH_CONFIDENCE_INFERRED" if r["relationship_state"] == "CONFIRMED"
                      else "MEDIUM_CONFIDENCE_INFERRED"),
            "cardinality": r.get("cardinality", "many_to_one"),
            "source": "generic-relational",
        })
    return inferred, ambiguous, edges


def build_relationship_graph(tables: List[str], edges: List[dict]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {t: [] for t in tables}
    canon = {_canon_ident(t): t for t in tables}
    bare: Dict[str, List[str]] = {}
    for t in tables:
        bare.setdefault(_bare_key(t), []).append(t)
    for e in edges or []:
        a = e.get("left_table", e.get("source_table"))
        b = e.get("right_table", e.get("target_table"))
        ra = a if a in graph else canon.get(_canon_ident(a))
        rb = b if b in graph else canon.get(_canon_ident(b))
        if ra is None:
            cands = bare.get(_bare_key(a), [])
            ra = cands[0] if len(cands) == 1 else None
        if rb is None:
            cands = bare.get(_bare_key(b), [])
            rb = cands[0] if len(cands) == 1 else None
        if ra in graph and rb in graph:
            if rb not in graph[ra]:
                graph[ra].append(rb)
            if ra not in graph[rb]:
                graph[rb].append(ra)
    return graph


def is_single_table_mapping(mapping: Dict[str, Any]) -> bool:
    """True when the mapping holds exactly one user table (generic).

    No dataset/table/column names are inspected — only the table count.
    """
    try:
        tables = (mapping or {}).get("tables") or {}
        return len(tables) == 1
    except Exception:
        return False


def normalize_single_table_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Treat any 1-table DB (SQLite/Postgres/MySQL) like a CSV dataset.

    Generic only: no table/column literals. Preserves DB native column
    types and declared primary keys/indexes; only the *shape* is aligned
    with the CSV contract so GoalAgent, KPI, and RAG-tabular see an
    identical single-table dataset:

    - per-table profiling fields guaranteed: ``primary_key`` (kept),
      ``inferred_primary_key`` (single-col only), plus
      ``inferred_composite_primary_key`` / ``composite_candidate_keys``
      for a fully-populated unique pair (split out generically when an
      inferred key holds >1 column and no declared single PK exists);
    - ``row_count`` derived from ``null_stats`` when missing,
      ``empty`` set for every table;
    - FK-none / relationship-empty / no-join: declared/inferred/
      ambiguous/edges cleared (a lone table cannot join), duplicate
      columns emptied, relationship graph reduced to a single isolated
      node (both the ``{table: [neighbours]}`` and the detailed
      ``{table: {outgoing/incoming/...}}`` graph shapes are handled).
    """
    tables = (mapping or {}).get("tables") or {}
    if len(tables) != 1:
        return mapping
    sole = next(iter(tables))
    info = tables[sole] or {}
    if not isinstance(info, dict):
        return mapping

    info.setdefault("columns", [])
    if not isinstance(info.get("primary_key"), list):
        info["primary_key"] = list(info.get("primary_key") or [])
    if not isinstance(info.get("inferred_primary_key"), list):
        info["inferred_primary_key"] = list(info.get("inferred_primary_key") or [])

    # Generic composite split: an inferred key with >1 column is a composite
    # pair, not a single-column PK. The single-column contract stays honest
    # ([]) while the pair is exposed separately — exactly like the CSV path.
    declared = [str(c) for c in (info.get("primary_key") or [])]
    inferred = [str(c) for c in (info.get("inferred_primary_key") or [])]
    comp = list(info.get("inferred_composite_primary_key") or [])
    cck = list(info.get("composite_candidate_keys") or [])
    col_names = set()
    try:
        for c in (info.get("columns") or []):
            if isinstance(c, dict) and c.get("column"):
                col_names.add(str(c["column"]))
            elif isinstance(c, str):
                col_names.add(c)
    except Exception:
        col_names = set()
    if not comp and len(inferred) > 1 and len(declared) != 1:
        members = [c for c in inferred if not col_names or c in col_names]
        if len(members) > 1:
            comp = members
            info["inferred_primary_key"] = []
            inferred = []
    if comp and not cck:
        cck = [list(comp)]
    info["inferred_composite_primary_key"] = [str(c) for c in (comp or [])]
    info["composite_candidate_keys"] = [
        [str(c) for c in pair] for pair in (cck or []) if isinstance(pair, (list, tuple))
    ]

    # row_count / empty from null_stats when the extractor omitted them
    # (schema_agent only stamps empty tables; CSV stamps every table).
    row_count = info.get("row_count")
    if not isinstance(row_count, int):
        row_count = None
        try:
            for stats in (info.get("null_stats") or {}).values():
                if isinstance(stats, dict) and isinstance(stats.get("total_rows"), int):
                    row_count = stats["total_rows"]
                    break
        except Exception:
            row_count = None
    if isinstance(row_count, int):
        info["row_count"] = row_count
        info["empty"] = row_count == 0
    else:
        info.setdefault("empty", False)
    info.setdefault("null_stats", {})
    info.setdefault("source", info.get("source", ""))

    # FK-none / relationship-empty: a lone table has no cross-table edge.
    # Declared self-references (if any) are stashed transparently instead
    # of silently kept, so downstream sees the CSV-identical empty shape.
    try:
        _orig_declared = list(mapping.get("declared_relationships") or [])
        _orig_inferred = list(mapping.get("inferred_relationships") or [])
        _orig_edges = list(mapping.get("relationship_edges") or [])
        if any(((r.get("table_name") or r.get("table")) == sole) for r in (_orig_declared + _orig_inferred)) \
                or _orig_edges:
            mapping["single_table_original_relationships"] = {
                "declared": _orig_declared,
                "inferred": _orig_inferred,
                "edges": _orig_edges,
            }
    except Exception:
        pass
    mapping["declared_relationships"] = []
    mapping["inferred_relationships"] = []
    mapping["ambiguous_relationships"] = []
    mapping["relationship_edges"] = []
    mapping["duplicate_columns"] = {}

    # Single isolated graph node (handle both graph shapes).
    existing_graph = mapping.get("relationship_graph")
    if isinstance(existing_graph, dict) and sole in existing_graph \
            and isinstance(existing_graph[sole], dict):
        mapping["relationship_graph"] = {
            sole: {
                "primary_key": list(declared or inferred or comp),
                "columns": [str(c.get("column")) for c in (info.get("columns") or [])
                            if isinstance(c, dict) and c.get("column")],
                "outgoing_relationships": [],
                "incoming_relationships": [],
                "many_to_many": [],
            }
        }
    else:
        mapping["relationship_graph"] = build_relationship_graph([sole], [])

    mapping["single_table_as_csv"] = True
    mapping.setdefault(
        "single_table_note",
        "Single-table database treated like a CSV dataset: profiling + "
        "PK-composite + FK-none + relationship-empty + no-join.",
    )
    return mapping


_TRUST_RANK = {"USER_CONFIRMED": -1, "DECLARED": 0,
               "HIGH_CONFIDENCE_INFERRED": 1,
               "MEDIUM_CONFIDENCE_INFERRED": 2,
               "LOW_CONFIDENCE_INFERRED": 3}


def plan_joins(required_tables: List[str], graph: Dict[str, List[str]],
               edges: List[dict]) -> Dict[str, Any]:
    """Minimum connected subgraph covering ``required_tables`` (generic BFS)."""
    graph = graph or {}
    canon = {_canon_ident(t): t for t in graph}
    bare: Dict[str, List[str]] = {}
    for t in graph:
        bare.setdefault(_bare_key(t), []).append(t)

    def _resolve(name):
        if name in graph:
            return name
        hit = canon.get(_canon_ident(name))
        if hit is not None:
            return hit
        cands = bare.get(_bare_key(name), [])
        return cands[0] if len(cands) == 1 else None

    req = []
    for t in (required_tables or []):
        r = _resolve(t)
        if r is not None and r not in req:
            req.append(r)
    if not req:
        return {"required_tables": [], "nodes": [], "edges": []}
    if len(req) == 1:
        return {"required_tables": req, "nodes": [req[0]], "edges": []}
    adj: Dict[str, list] = defaultdict(list)
    for e in edges or []:
        a = _resolve(e.get("left_table", e.get("source_table")))
        b = _resolve(e.get("right_table", e.get("target_table")))
        rank = _TRUST_RANK.get(e.get("trust"), 4)
        if a is not None and b is not None and a in graph and b in graph:
            adj[a].append((rank, e))
            adj[b].append((rank, e))
    for t in adj:
        adj[t].sort(key=lambda p: p[0])

    def bfs(source_set: set, target: str):
        queue = deque([[n] for n in source_set])
        seen = set(source_set)
        while queue:
            path = queue.popleft()
            tail = path[-1]
            for _rank, e in adj.get(tail, []):
                a = e.get("left_table", e.get("source_table"))
                b = e.get("right_table", e.get("target_table"))
                nb = b if a == tail else a
                if nb in seen:
                    continue
                new_path = path + [nb]
                if nb == target:
                    path_edges = []
                    for x, y in zip(new_path, new_path[1:]):
                        for _r, edge in adj[x]:
                            ea = edge.get("left_table", edge.get("source_table"))
                            eb = edge.get("right_table", edge.get("target_table"))
                            if (ea == y or eb == y):
                                path_edges.append(edge)
                                break
                    return new_path, path_edges
                seen.add(nb)
                queue.append(new_path)
        return None

    connected = {req[0]}
    nodes = [req[0]]
    out_edges: List[dict] = []
    for target in [t for t in req[1:] if t != req[0]]:
        found = bfs(connected, target)
        if found is None:
            if target not in nodes:
                nodes.append(target)  # disconnected: report, don't fabricate
            continue
        path_tables, path_edges = found
        for e in path_edges:
            if e not in out_edges:
                out_edges.append(e)
        for n in path_tables:
            if n not in connected:
                connected.add(n)
                nodes.append(n)
    return {"required_tables": req, "nodes": nodes, "edges": out_edges}
