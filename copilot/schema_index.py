"""Dynamic schema index for Copilot RAG.

Turns a connected database's schema mapping (produced by schema_agent, any
PostgreSQL/MySQL/SQLite source) into a searchable set of text documents:
one per table, one per meaningful column, one per relationship.

Nothing here is schema-specific: table names, column names, types, keys and
relationships are all read from the mapping. The same index works on a schema
it has never seen before.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Imported lazily-safe: schema_intel only does pure type classification.
try:
    from core.schema_intel import classify_type
except Exception:  # pragma: no cover - import path fallback
    def classify_type(data_type: str) -> str:
        dt = str(data_type or "").lower()
        if any(k in dt for k in ("int", "decimal", "numeric", "float", "real", "double", "money")):
            return "numeric"
        if any(k in dt for k in ("date", "time", "timestamp")):
            return "datetime"
        if any(k in dt for k in ("char", "text", "varchar", "uuid")):
            return "text"
        if any(k in dt for k in ("bytea", "blob", "binary")):
            return "binary"
        return "other"


_ID_LIKE_RE = re.compile(r"(^|_)(id|key|code|uuid|guid)$", re.IGNORECASE)
_AMOUNT_HINTS = (
    "price", "cost", "amount", "total", "sum", "revenue", "sales", "income",
    "expense", "profit", "value", "quantity", "qty", "discount", "tax",
    "payment", "balance", "weight", "margin", "turnover", "fee", "rate",
)
_DATE_HINTS = ("date", "time", "created", "updated", "timestamp", "period", "year", "month", "day")


def _natural(name: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", str(name or "")) if w]
    return " ".join(w.lower() for w in words)


def _is_id_like(name: str) -> bool:
    low = str(name or "").lower()
    return low in ("id", "key", "code") or bool(_ID_LIKE_RE.search(low)) or low.endswith("_id")


def _column_names(info: dict) -> list[dict]:
    """Normalize a table's columns to [{column, data_type}] across mapping shapes."""
    cols_raw = info.get("columns", [])
    out: list[dict] = []
    if isinstance(cols_raw, dict):
        for name, meta in cols_raw.items():
            dtype = ""
            if isinstance(meta, dict):
                dtype = str(meta.get("data_type", ""))
            out.append({"column": str(name), "data_type": dtype})
    elif isinstance(cols_raw, list):
        for c in cols_raw:
            if isinstance(c, dict):
                name = c.get("column") or c.get("name")
                if name:
                    out.append({"column": str(name), "data_type": str(c.get("data_type", ""))})
            elif isinstance(c, str):
                out.append({"column": c, "data_type": ""})
    return out


def _pk_of(info: dict) -> list[str]:
    # Composite-aware: a composite-only table still reports its key members
    # instead of "inferred", so single-table datasets keep working.
    pk = list(info.get("primary_key") or info.get("inferred_primary_key") or [])
    if pk:
        return pk
    comp = info.get("composite_primary_key") or info.get("inferred_composite_primary_key")
    if comp and isinstance(comp, (list, tuple)):
        return [str(c) for c in comp]
    cck = info.get("composite_candidate_keys") or []
    if cck and isinstance(cck, (list, tuple)) and cck and isinstance(cck[0], (list, tuple)):
        return [str(c) for c in cck[0]]
    return []


def _fk_of(info: dict) -> list[dict]:
    return list(info.get("foreign_keys") or [])


@dataclass
class SchemaDocument:
    """One retrievable unit of schema knowledge."""

    doc_id: str
    kind: str            # "table" | "column" | "relationship"
    table: str
    text: str
    payload: dict = field(default_factory=dict)


class SchemaIndex:
    """Provider-independent, in-memory schema index with deterministic ranking.

    Semantic retrieval is layered on top by the retriever; this class owns the
    documents and a keyword/BM25-style lexical score that always works even
    when no embedding model is available (offline / cold start).
    """

    def __init__(self, mapping: dict, dialect: str = "postgresql"):
        self.mapping = mapping or {}
        self.dialect = dialect
        self.tables: dict[str, dict] = self.mapping.get("tables", {}) or {}
        self.documents: list[SchemaDocument] = []
        self._by_table: dict[str, list[SchemaDocument]] = {}
        self._edges: list[dict] = []
        self._build()

    # -- construction ---------------------------------------------------

    def _build(self) -> None:
        for table, info in self.tables.items():
            if not isinstance(info, dict):
                continue
            self._add_table_doc(table, info)
            for col in _column_names(info):
                self._add_column_doc(table, info, col)
        self._collect_edges()

    def _add_table_doc(self, table: str, info: dict) -> None:
        cols = _column_names(info)
        pk = _pk_of(info)
        fks = _fk_of(info)
        measures = [c["column"] for c in cols
                    if classify_type(c["data_type"]) == "numeric"
                    and not _is_id_like(c["column"])]
        dates = [c["column"] for c in cols
                 if classify_type(c["data_type"]) == "datetime"
                 or any(h in c["column"].lower() for h in _DATE_HINTS)]
        row_count = info.get("row_count")
        description = str(
            info.get("description") or info.get("business_meaning")
            or info.get("business_description") or ""
        )[:500]
        col_descriptions = {
            c["column"]: str(c.get("description", ""))[:200]
            for c in cols if isinstance(c, dict) and c.get("description")
        }
        rels = [f"{fk.get('column')} -> {fk.get('referenced_table')}.{fk.get('referenced_column')}"
                for fk in fks if fk.get("referenced_table")]
        text = (
            f"Table {table} ({_natural(table)}). "
            + (f"{description} " if description else "")
            + f"Columns: {', '.join(c['column'] for c in cols)}. "
            f"Measures: {', '.join(measures) or 'none'}. "
            f"Dates: {', '.join(dates) or 'none'}. "
            f"Primary key: {', '.join(pk) or 'inferred'}. "
            + (f"Rows: {row_count}. " if row_count is not None else "")
            + (f"Relationships: {'; '.join(rels)}." if rels else "")
        )
        self.documents.append(SchemaDocument(
            doc_id=f"table:{table}",
            kind="table",
            table=table,
            text=text,
            payload={
                "table": table,
                "columns": [c["column"] for c in cols],
                "column_types": {c["column"]: c["data_type"] for c in cols},
                "column_descriptions": col_descriptions,
                "description": description,
                "measures": measures,
                "dates": dates,
                "primary_key": pk,
                "foreign_keys": fks,
                "row_count": row_count,
            },
        ))

    def _add_column_doc(self, table: str, info: dict, col: dict) -> None:
        name = col["column"]
        dtype = col["data_type"]
        category = classify_type(dtype)
        description = str(col.get("description", "") or "")[:200]
        pk = set(_pk_of(info))
        fk_cols = {fk.get("column") for fk in _fk_of(info)}
        role = "key" if name in pk else ("foreign key" if name in fk_cols else category)
        if category == "numeric" and not _is_id_like(name) and any(h in name.lower() for h in _AMOUNT_HINTS):
            role = "measure"
        text = (
            f"Column {table}.{name} ({_natural(name)}) "
            f"of type {dtype or category}, role {role}."
            + (f" {description}" if description else "")
        )
        self.documents.append(SchemaDocument(
            doc_id=f"column:{table}.{name}",
            kind="column",
            table=table,
            text=text,
            payload={"table": table, "column": name, "data_type": dtype,
                     "category": category, "role": role,
                     "description": description},
        ))

    def _collect_edges(self) -> None:
        edges = list(self.mapping.get("declared_relationships") or [])
        edges += list(self.mapping.get("relationship_edges") or [])
        for table, info in self.tables.items():
            if not isinstance(info, dict):
                continue
            for fk in _fk_of(info):
                edges.append({
                    "left_table": table,
                    "left_column": fk.get("column"),
                    "right_table": fk.get("referenced_table"),
                    "right_column": fk.get("referenced_column"),
                    "trust": fk.get("trust", "declared"),
                })
        seen = set()
        for e in edges:
            lt, lc = e.get("left_table"), e.get("left_column")
            rt, rc = e.get("right_table"), e.get("right_column")
            if not (lt and lc and rt and rc):
                continue
            key = (lt, lc, rt, rc)
            if key in seen:
                continue
            seen.add(key)
            trust = e.get("trust", "declared")
            self._edges.append({"left_table": lt, "left_column": lc,
                                "right_table": rt, "right_column": rc, "trust": trust})
            text = (f"Relationship {lt}.{lc} relates to {rt}.{rc} ({trust}). "
                    f"Join {lt} with {rt} on these columns.")
            self.documents.append(SchemaDocument(
                doc_id=f"relationship:{lt}.{lc}->{rt}.{rc}",
                kind="relationship",
                table=lt,
                text=text,
                payload={"left_table": lt, "left_column": lc,
                         "right_table": rt, "right_column": rc, "trust": trust},
            ))

    # -- lexical scoring (always available) ------------------------------

    @staticmethod
    def _tokens(text: str) -> set[str]:
        out = set()
        for w in re.findall(r"[A-Za-z0-9_]+", str(text or "").lower()):
            out.add(w)
            out.update(p for p in w.split("_") if p)
        return {t for t in out if t}

    def lexical_search(self, query: str, limit: int = 20) -> list[tuple[float, SchemaDocument]]:
        q = self._tokens(query)
        if not q:
            return []
        scored: list[tuple[float, SchemaDocument]] = []
        for doc in self.documents:
            dt = self._tokens(doc.text)
            if not dt:
                continue
            overlap = q & dt
            if not overlap:
                continue
            score = float(len(overlap))
            # A table/column name match is a stronger signal than prose overlap.
            name_tokens = self._tokens(doc.payload.get("table", "")) | self._tokens(
                doc.payload.get("column", ""))
            score += 2.0 * len(q & name_tokens)
            if doc.kind == "table":
                score += 0.5
            scored.append((score, doc))
        scored.sort(key=lambda t: (-t[0], t[1].doc_id))
        return scored[:limit]

    # -- helpers ---------------------------------------------------------

    def all_tables(self) -> list[str]:
        return list(self.tables.keys())

    def edges(self) -> list[dict]:
        return list(self._edges)

    def table_doc(self, table: str) -> SchemaDocument | None:
        for d in self._by_table.get(table, []):
            if d.kind == "table":
                return d
        for d in self.documents:
            if d.kind == "table" and d.table == table:
                self._by_table.setdefault(table, []).append(d)
                return d
        return None

    def stats(self) -> dict:
        return {
            "tables": len(self.tables),
            "documents": len(self.documents),
            "relationships": len(self._edges),
        }


def build_schema_index(mapping_or_path, dialect: str = "postgresql") -> SchemaIndex:
    """Build a SchemaIndex from a mapping dict or a path to a mapping JSON."""
    if isinstance(mapping_or_path, (str, Path)):
        path = Path(mapping_or_path)
        mapping = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    else:
        mapping = mapping_or_path or {}
    return SchemaIndex(mapping, dialect=dialect)