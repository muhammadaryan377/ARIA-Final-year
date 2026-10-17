"""Generic multi-table handling: no hardcoded dataset/table/column names.

Covers the generic contract for ANY dataset:
- single-table (Walmart CSV) -> zero edges is valid, composite (Store, Date)
  uniqueness still detectable, no JOINs needed;
- synthetic stores/sales split -> PK inference, FK inference via name/type
  overlap + value containment, relationship graph, join planning, and
  duplicate-column disambiguation;
- SQLite multi-table mapping via catalog discovery;
- core.relational unit behavior (qualify/resolve/plan).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WALMART_CSV = (REPO / "data" / "kaggle_retail_sales"
               / "mikhail1681__walmart-sales" / "Walmart_Sales.csv")


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_walmart_single_table_needs_no_joins():
    import sqlite3

    from csv_orchestrator import build_csv_schema_mapping
    from core.relational import build_relationship_graph, plan_joins

    if not WALMART_CSV.exists():
        import pytest
        pytest.skip(f"Walmart CSV not found: {WALMART_CSV}")
    mapping = build_csv_schema_mapping(WALMART_CSV, table_name="walmart")
    info = mapping["tables"]["walmart"]
    assert info["row_count"] == 6435
    # no single-column PK forced; composite pair exposed generically
    assert info.get("inferred_primary_key", []) == []
    comp = info.get("inferred_composite_primary_key") or []
    assert len(comp) == 2
    assert mapping.get("inferred_relationships", []) == []
    assert mapping.get("relationship_edges", []) == []
    graph = build_relationship_graph(["walmart"], [])
    plan = plan_joins(["walmart"], graph, [])
    assert plan["nodes"] == ["walmart"] and plan["edges"] == []


def test_multi_csv_fk_graph_join_and_disambiguation(tmp_path):
    from csv_orchestrator import (
        build_multi_csv_schema_mapping,
        load_multiple_csvs_to_sqlite,
    )
    from core.relational import plan_joins, resolve_column

    _write(tmp_path / "stores.csv",
           "store_id,region\n1,North\n2,South\n3,East\n")
    _write(tmp_path / "sales.csv",
           "sale_id,store_id,region,amount\n"
           "10,1,North,100.0\n11,2,South,200.0\n"
           "12,1,North,150.0\n13,3,East,300.0\n")
    mapping = build_multi_csv_schema_mapping(tmp_path)
    assert set(mapping["tables"]) == {"sales", "stores"}
    assert mapping["tables"]["stores"]["inferred_primary_key"] == ["store_id"]
    assert mapping["tables"]["sales"]["inferred_primary_key"] == ["sale_id"]
    inferred = mapping["inferred_relationships"]
    assert any(r["table"] == "sales" and r["column"] == "store_id"
               and r["references_table"] == "stores"
               and r["references_column"] == "store_id" for r in inferred)
    edges = mapping["relationship_edges"]
    assert edges, "expected at least one cross-table edge"
    graph = mapping["relationship_graph"]
    assert "stores" in graph["sales"] and "sales" in graph["stores"]
    plan = plan_joins(["sales", "stores"], graph, edges)
    assert set(plan["nodes"]) == {"sales", "stores"}
    assert len(plan["edges"]) == 1
    # duplicate column `region` exists on both tables -> disambiguation
    dupes = mapping["duplicate_columns"]
    assert set(dupes.get("region", [])) == {"sales", "stores"}
    owners = resolve_column("region", {"sales": ["sale_id", "store_id", "region"],
                                       "stores": ["store_id", "region"]})
    assert owners == ["sales.region", "stores.region"]
    # shared SQLite load keeps both tables queryable
    db_uri, tables, counts = load_multiple_csvs_to_sqlite(
        tmp_path, tmp_path / "multi.db")
    assert set(tables) == {"sales", "stores"}
    assert counts == {"sales": 4, "stores": 3}
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "multi.db"))
    try:
        n = conn.execute(
            'SELECT COUNT(*) FROM "sales" JOIN "stores" '
            'ON "sales"."store_id" = "stores"."store_id"').fetchone()[0]
    finally:
        conn.close()
    assert n == 4


def test_sqlite_catalog_mapping_generic(tmp_path):
    from csv_orchestrator import (
        build_sqlite_schema_mapping,
        load_multiple_csvs_to_sqlite,
    )

    _write(tmp_path / "authors.csv", "author_id,name\n1,A\n2,B\n")
    _write(tmp_path / "books.csv",
           "book_id,author_id,title\n10,1,T1\n11,2,T2\n12,1,T3\n")
    load_multiple_csvs_to_sqlite(tmp_path, tmp_path / "lib.db")
    mapping = build_sqlite_schema_mapping(tmp_path / "lib.db")
    assert set(mapping["tables"]) == {"authors", "books"}
    assert mapping["tables"]["authors"]["row_count"] == 2
    assert mapping["tables"]["books"]["row_count"] == 3
    assert any(r["table"] == "books" and r["references_table"] == "authors"
               for r in mapping["inferred_relationships"])


def test_relational_helpers_generic():
    from core.relational import (
        duplicate_columns,
        plan_joins,
        qualify,
        resolve_column,
        type_family,
        types_compatible,
    )

    assert type_family("bigint") == "numeric"
    assert type_family("TEXT") == "text"
    assert types_compatible("INTEGER", "bigint")
    assert not types_compatible("INTEGER", "DATETIME")
    assert duplicate_columns({"a": ["x", "y"], "b": ["y", "z"]}) == {"y": ["a", "b"]}
    assert resolve_column("y", {"a": ["x", "y"], "b": ["y", "z"]}) == ["a.y", "b.y"]
    assert qualify("t", "c", "sqlite") == '"t"."c"'
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    edges = [
        {"left_table": "a", "left_column": "b_id", "right_table": "b",
         "right_column": "id", "trust": "DECLARED"},
        {"left_table": "b", "left_column": "c_id", "right_table": "c",
         "right_column": "id", "trust": "DECLARED"},
    ]
    plan = plan_joins(["a", "c"], graph, edges)
    assert set(plan["nodes"]) == {"a", "b", "c"}
    assert len(plan["edges"]) == 2
    solo = plan_joins(["a"], graph, edges)
    assert solo == {"required_tables": ["a"], "nodes": ["a"], "edges": []}
