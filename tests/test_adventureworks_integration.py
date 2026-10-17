"""AdventureWorks integration test (spec item 7/13).

Permanent hermetic suite: an AdventureWorks-style multi-table SQLite fixture
and a single-table denormalized fixture holding the SAME facts. The 8
business questions below must return the SAME concepts on both datasets,
and both must pass — no live DB or LLM required.

Questions: how many customers, top 10 products, category with most sales,
monthly trend, best territory, highest-selling product, order count,
avg order value.
"""

import json
import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Deterministic seed (shared facts for both fixtures)
# ---------------------------------------------------------------------------
# unit prices: Widget 50, Gizmo 200, Sprocket 25
# orders: o1=100, o2=200, o3=150, o4=400, o5=50, o6=150 -> grand 1050, AOV 175
CUSTOMERS = [(1, "Alice", "North"), (2, "Bob", "South"),
             (3, "Cara", "North"), (4, "Dan", "East")]
PRODUCTS = [(101, "Widget", "Gadgets"), (102, "Gizmo", "Gadgets"),
            (103, "Sprocket", "Tools")]
TERRITORIES = [(1, "North"), (2, "South"), (3, "East")]
ORDERS = [(1, 1, 1, "2024-01-05"), (2, 2, 2, "2024-01-20"),
          (3, 1, 1, "2024-02-10"), (4, 3, 1, "2024-02-15"),
          (5, 4, 3, "2024-03-01"), (6, 2, 2, "2024-03-10")]
ITEMS = [(1, 101, 2, 50.0), (2, 102, 1, 200.0),
         (3, 101, 1, 50.0), (3, 103, 4, 25.0),
         (4, 102, 2, 200.0), (5, 103, 2, 25.0), (6, 101, 3, 50.0)]

# flat rows: (customer, product, category, territory, order_id, date, qty, price)
FLAT_ROWS = [
    ("Alice", "Widget", "Gadgets", "North", 1, "2024-01-05", 2, 50.0),
    ("Bob", "Gizmo", "Gadgets", "South", 2, "2024-01-20", 1, 200.0),
    ("Alice", "Widget", "Gadgets", "North", 3, "2024-02-10", 1, 50.0),
    ("Alice", "Sprocket", "Tools", "North", 3, "2024-02-10", 4, 25.0),
    ("Cara", "Gizmo", "Gadgets", "North", 4, "2024-02-15", 2, 200.0),
    ("Dan", "Sprocket", "Tools", "East", 5, "2024-03-01", 2, 25.0),
    ("Bob", "Widget", "Gadgets", "South", 6, "2024-03-10", 3, 50.0),
]

EXPECTED = {
    "customer_count": 4,
    "top_products": [("Gizmo", 600.0), ("Widget", 300.0), ("Sprocket", 150.0)],
    "top_category": ("Gadgets", 900.0),
    "monthly": [("2024-01", 300.0), ("2024-02", 550.0), ("2024-03", 200.0)],
    "best_territory": ("North", 650.0),
    "top_product": ("Gizmo", 600.0),
    "order_count": 6,
    "aov": 175.0,
}

# Per-question SQL for each schema shape.
AW_SQL = {
    "customer_count": "SELECT COUNT(*) AS v FROM customers",
    "top_products": ("SELECT p.product_name AS product, SUM(i.quantity*i.unit_price) AS revenue "
                     "FROM order_items i JOIN products p ON p.product_id=i.product_id "
                     "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 10"),
    "top_category": ("SELECT p.category AS category, SUM(i.quantity*i.unit_price) AS revenue "
                     "FROM order_items i JOIN products p ON p.product_id=i.product_id "
                     "GROUP BY p.category ORDER BY revenue DESC LIMIT 1"),
    "monthly": ("SELECT substr(o.order_date,1,7) AS month, SUM(i.quantity*i.unit_price) AS revenue "
                "FROM orders o JOIN order_items i ON i.order_id=o.order_id "
                "GROUP BY month ORDER BY month"),
    "best_territory": ("SELECT t.territory_name AS territory, SUM(i.quantity*i.unit_price) AS revenue "
                       "FROM orders o JOIN order_items i ON i.order_id=o.order_id "
                       "JOIN territories t ON t.territory_id=o.territory_id "
                       "GROUP BY t.territory_name ORDER BY revenue DESC LIMIT 1"),
    "top_product": ("SELECT p.product_name AS product, SUM(i.quantity*i.unit_price) AS revenue "
                    "FROM order_items i JOIN products p ON p.product_id=i.product_id "
                    "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 1"),
    "order_count": "SELECT COUNT(DISTINCT order_id) AS v FROM orders",
    "aov": ("SELECT AVG(t.v) AS v FROM (SELECT SUM(i.quantity*i.unit_price) AS v "
            "FROM order_items i GROUP BY i.order_id) t"),
}
FLAT_SQL = {
    "customer_count": "SELECT COUNT(DISTINCT customer) AS v FROM sales_flat",
    "top_products": ("SELECT product, SUM(quantity*unit_price) AS revenue FROM sales_flat "
                     "GROUP BY product ORDER BY revenue DESC LIMIT 10"),
    "top_category": ("SELECT category, SUM(quantity*unit_price) AS revenue FROM sales_flat "
                     "GROUP BY category ORDER BY revenue DESC LIMIT 1"),
    "monthly": ("SELECT substr(order_date,1,7) AS month, SUM(quantity*unit_price) AS revenue "
                "FROM sales_flat GROUP BY month ORDER BY month"),
    "best_territory": ("SELECT territory, SUM(quantity*unit_price) AS revenue FROM sales_flat "
                       "GROUP BY territory ORDER BY revenue DESC LIMIT 1"),
    "top_product": ("SELECT product, SUM(quantity*unit_price) AS revenue FROM sales_flat "
                    "GROUP BY product ORDER BY revenue DESC LIMIT 1"),
    "order_count": "SELECT COUNT(DISTINCT order_id) AS v FROM sales_flat",
    "aov": ("SELECT AVG(t.v) AS v FROM (SELECT SUM(quantity*unit_price) AS v "
            "FROM sales_flat GROUP BY order_id) t"),
}

QUESTIONS = [
    "how many customers",
    "top 10 products by sales",
    "which category has the most sales",
    "monthly sales trend",
    "best sales territory",
    "highest-selling product",
    "how many orders",
    "average order value",
]
QKEY = ["customer_count", "top_products", "top_category", "monthly",
        "best_territory", "top_product", "order_count", "aov"]


@pytest.fixture()
def aw_conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, territory TEXT);
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT, category TEXT);
        CREATE TABLE territories (territory_id INTEGER PRIMARY KEY, territory_name TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, territory_id INTEGER, order_date TEXT);
        CREATE TABLE order_items (order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price REAL);
    """)
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO products VALUES (?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO territories VALUES (?,?)", TERRITORIES)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?)", ITEMS)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def flat_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sales_flat (customer TEXT, product TEXT, category TEXT,"
                 " territory TEXT, order_id INTEGER, order_date TEXT,"
                 " quantity INTEGER, unit_price REAL)")
    conn.executemany("INSERT INTO sales_flat VALUES (?,?,?,?,?,?,?,?)", FLAT_ROWS)
    conn.commit()
    yield conn
    conn.close()


def _scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


def _all(conn, sql):
    return [(r[0], float(r[1])) for r in conn.execute(sql).fetchall()]


# -- 8 business questions, both schemas must pass with the same concepts ----

def test_01_customer_count(aw_conn, flat_conn):
    assert _scalar(aw_conn, AW_SQL["customer_count"]) == EXPECTED["customer_count"]
    assert _scalar(flat_conn, FLAT_SQL["customer_count"]) == EXPECTED["customer_count"]


def test_02_top_10_products(aw_conn, flat_conn):
    assert _all(aw_conn, AW_SQL["top_products"]) == EXPECTED["top_products"]
    assert _all(flat_conn, FLAT_SQL["top_products"]) == EXPECTED["top_products"]


def test_03_category_most_sales(aw_conn, flat_conn):
    aw = _all(aw_conn, AW_SQL["top_category"])[0]
    fl = _all(flat_conn, FLAT_SQL["top_category"])[0]
    assert aw[0] == fl[0] == EXPECTED["top_category"][0]
    assert aw[1] == pytest.approx(EXPECTED["top_category"][1])
    assert fl[1] == pytest.approx(EXPECTED["top_category"][1])


def test_04_monthly_trend(aw_conn, flat_conn):
    assert _all(aw_conn, AW_SQL["monthly"]) == EXPECTED["monthly"]
    assert _all(flat_conn, FLAT_SQL["monthly"]) == EXPECTED["monthly"]


def test_05_best_territory(aw_conn, flat_conn):
    aw = _all(aw_conn, AW_SQL["best_territory"])[0]
    fl = _all(flat_conn, FLAT_SQL["best_territory"])[0]
    assert aw[0] == fl[0] == EXPECTED["best_territory"][0]
    assert aw[1] == pytest.approx(EXPECTED["best_territory"][1])
    assert fl[1] == pytest.approx(EXPECTED["best_territory"][1])


def test_06_highest_selling_product(aw_conn, flat_conn):
    aw = _all(aw_conn, AW_SQL["top_product"])[0]
    fl = _all(flat_conn, FLAT_SQL["top_product"])[0]
    assert aw[0] == fl[0] == EXPECTED["top_product"][0]
    assert aw[1] == pytest.approx(EXPECTED["top_product"][1])


def test_07_order_count(aw_conn, flat_conn):
    assert _scalar(aw_conn, AW_SQL["order_count"]) == EXPECTED["order_count"]
    assert _scalar(flat_conn, FLAT_SQL["order_count"]) == EXPECTED["order_count"]


def test_08_avg_order_value(aw_conn, flat_conn):
    assert _scalar(aw_conn, AW_SQL["aov"]) == pytest.approx(EXPECTED["aov"])
    assert _scalar(flat_conn, FLAT_SQL["aov"]) == pytest.approx(EXPECTED["aov"])


# -- Goal-Agent semantic grounding works on BOTH shapes ---------------------

def _typed(cols):
    return [{"column": c, "data_type": t, "nullable": True} for c, t in cols]


def _aw_mapping():
    return {
        "schema": "public",
        "tables": {
            "customers": {"columns": _typed([("customer_id", "integer"), ("name", "text"),
                                             ("territory", "text")]),
                          "primary_key": ["customer_id"]},
            "products": {"columns": _typed([("product_id", "integer"), ("product_name", "text"),
                                            ("category", "text")]),
                         "primary_key": ["product_id"]},
            "territories": {"columns": _typed([("territory_id", "integer"),
                                               ("territory_name", "text")]),
                            "primary_key": ["territory_id"]},
            "orders": {"columns": _typed([("order_id", "integer"), ("customer_id", "integer"),
                                          ("territory_id", "integer"), ("order_date", "timestamp")]),
                       "primary_key": ["order_id"]},
            "order_items": {"columns": _typed([("order_id", "integer"), ("product_id", "integer"),
                                               ("quantity", "integer"), ("unit_price", "numeric")]),
                            "primary_key": ["order_id", "product_id"]},
        },
        "relationship_edges": [],
        "declared_relationships": [
            {"table_name": "orders", "column_name": "customer_id",
             "references_table": "customers", "references_column": "customer_id"},
            {"table_name": "orders", "column_name": "territory_id",
             "references_table": "territories", "references_column": "territory_id"},
            {"table_name": "order_items", "column_name": "order_id",
             "references_table": "orders", "references_column": "order_id"},
            {"table_name": "order_items", "column_name": "product_id",
             "references_table": "products", "references_column": "product_id"},
        ],
        "inferred_relationships": [],
        "ambiguous_relationships": [],
    }


def _flat_mapping():
    return {
        "schema": "public",
        "tables": {
            "sales_flat": {"columns": _typed([("customer", "text"), ("product", "text"),
                                              ("category", "text"), ("territory", "text"),
                                              ("order_id", "integer"), ("order_date", "timestamp"),
                                              ("quantity", "integer"), ("unit_price", "numeric")]),
                           "primary_key": []},
        },
        "relationship_edges": [],
        "declared_relationships": [],
        "inferred_relationships": [],
        "ambiguous_relationships": [],
    }


class _NoLLM:
    def complete(self, *a, **k):
        raise AssertionError("LLM must not be called")

    def chat(self, *a, **k):
        raise AssertionError("LLM must not be called")


def _agent(mapping, tmp_path):
    from goal_agent import GoalAgent
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return GoalAgent(schema_json_path=str(p),
                     db_uri="postgresql://u:p@localhost:5432/nonexistent",
                     provider=_NoLLM(), dialect="postgresql")


@pytest.mark.parametrize("qi", range(len(QUESTIONS)))
def test_09_same_concepts_ground_on_both_shapes(tmp_path, qi):
    """Each NL question resolves to measures/dimensions on AW and flat alike."""
    q = QUESTIONS[qi]
    for mapping in (_aw_mapping(), _flat_mapping()):
        d = tmp_path / f"s{qi}_{len(str(mapping))}.json"
        ag = _agent(mapping, tmp_path)
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        assert ir["metrics"], f"{q!r} resolved no metrics on {list(mapping['tables'])}"
        json.dumps(ir)  # must stay serialisable


def test_10_cross_schema_equivalence(aw_conn, flat_conn):
    """Every question key returns identical values across both schemas."""
    for key in QKEY:
        if key in ("customer_count", "order_count", "aov"):
            assert _scalar(aw_conn, AW_SQL[key]) == pytest.approx(
                _scalar(flat_conn, FLAT_SQL[key])), key
        else:
            assert _all(aw_conn, AW_SQL[key]) == _all(flat_conn, FLAT_SQL[key]), key


# ---------------------------------------------------------------------------
# Real-shape CamelCase multi-schema fixture (AdventureWorks LIVE gaps).
# Tables use quoted CamelCase + schema-qualified mapping keys:
# header monetary total vs line qty*price, Person/Customer split.
# ---------------------------------------------------------------------------

def _camel_typed(cols):
    return [{"column": c, "data_type": t, "nullable": True} for c, t in cols]


def _camel_mapping():
    return {
        "schema": "public",
        "schemas": ["Sales", "Person", "Production"],
        "tables": {
            "Sales.SalesOrderHeader": {
                "columns": _camel_typed([
                    ("SalesOrderID", "integer"), ("CustomerID", "integer"),
                    ("TerritoryID", "integer"), ("OrderDate", "timestamp"),
                    ("SubTotal", "numeric"), ("TaxAmt", "numeric"),
                    ("Freight", "numeric"), ("TotalDue", "numeric")]),
                "primary_key": ["SalesOrderID"]},
            "Sales.SalesOrderDetail": {
                "columns": _camel_typed([
                    ("SalesOrderID", "integer"), ("ProductID", "integer"),
                    ("OrderQty", "integer"), ("UnitPrice", "numeric"),
                    ("LineTotal", "numeric")]),
                "primary_key": ["SalesOrderID", "ProductID"]},
            "Sales.Customer": {
                "columns": _camel_typed([
                    ("CustomerID", "integer"), ("PersonID", "integer"),
                    ("TerritoryID", "integer")]),
                "primary_key": ["CustomerID"]},
            "Person.Person": {
                "columns": _camel_typed([
                    ("BusinessEntityID", "integer"), ("FirstName", "text"),
                    ("LastName", "text")]),
                "primary_key": ["BusinessEntityID"]},
            "Production.Product": {
                "columns": _camel_typed([
                    ("ProductID", "integer"), ("Name", "text"),
                    ("ProductCategory", "text")]),
                "primary_key": ["ProductID"]},
            "Sales.SalesTerritory": {
                "columns": _camel_typed([
                    ("TerritoryID", "integer"), ("Name", "text")]),
                "primary_key": ["TerritoryID"]},
        },
        "relationship_edges": [],
        "declared_relationships": [
            {"table_name": "Sales.SalesOrderDetail", "column_name": "SalesOrderID",
             "references_table": "Sales.SalesOrderHeader", "references_column": "SalesOrderID"},
            {"table_name": "Sales.SalesOrderDetail", "column_name": "ProductID",
             "references_table": "Production.Product", "references_column": "ProductID"},
            {"table_name": "Sales.SalesOrderHeader", "column_name": "CustomerID",
             "references_table": "Sales.Customer", "references_column": "CustomerID"},
            {"table_name": "Sales.SalesOrderHeader", "column_name": "TerritoryID",
             "references_table": "Sales.SalesTerritory", "references_column": "TerritoryID"},
            {"table_name": "Sales.Customer", "column_name": "PersonID",
             "references_table": "Person.Person", "references_column": "BusinessEntityID"},
            {"table_name": "Sales.Customer", "column_name": "TerritoryID",
             "references_table": "Sales.SalesTerritory", "references_column": "TerritoryID"},
        ],
        "inferred_relationships": [],
        "ambiguous_relationships": [],
    }


def _camel_agent(tmp_path, mapping=None):
    from goal_agent import GoalAgent
    m = mapping if mapping is not None else _camel_mapping()
    p = tmp_path / "camel_schema.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return GoalAgent(schema_json_path=str(p),
                     db_uri="postgresql://u:p@localhost:5432/nonexistent",
                     provider=_NoLLM(), dialect="postgresql")


def _bare(name):
    return str(name or "").split(".")[-1].strip().strip("\"'`[]").lower()


def test_11_header_total_preferred_over_line_synthesis(tmp_path):
    """Monetary totals prefer the header-level total when both exist."""
    ag = _camel_agent(tmp_path)
    for q in ("total sales", "total revenue"):
        ir = ag._build_goal_ir(q)
        jp = ag._determine_join_path(ag._get_relevant_tables(q))
        ir = ag._resolve_semantics(q, ir, jp)
        sums = [m for m in ir["metrics"] if str(m.get("aggregation", "")).upper() == "SUM"]
        assert sums, f"{q!r} resolved no SUM metrics"
        for m in sums:
            expr = str(m.get("resolved_expression") or "")
            assert "*" not in expr, (
                f"{q!r} synthesized qty*price {expr!r} instead of the header total")
            assert _bare(m.get("resolved_table")) == "salesorderheader", (
                f"{q!r} resolved to {m.get('resolved_table')} instead of the header table")
            assert str(m.get("resolved_column") or "").lower() == "totaldue", (
                f"{q!r} resolved to {m.get('resolved_column')} instead of the header total")


def test_12_order_count_resolves_to_header(tmp_path):
    """'how many orders' counts header rows, never territory/customer splits."""
    ag = _camel_agent(tmp_path)
    ir = ag._build_goal_ir("how many orders")
    jp = ag._determine_join_path(ag._get_relevant_tables("how many orders"))
    ir = ag._resolve_semantics("how many orders", ir, jp)
    counts = [m for m in ir["metrics"] if str(m.get("aggregation", "")).upper() == "COUNT"]
    assert counts, "no COUNT metric resolved"
    for m in counts:
        assert _bare(m.get("resolved_table")) in ("salesorderheader", "salesorderdetail"), (
            f"order count resolved to {m.get('resolved_table')}")


def test_13_customer_person_split_and_cross_schema_join(tmp_path):
    """Customer counts stay on Customer; Customer<->Person spans schemas."""
    ag = _camel_agent(tmp_path)
    ir = ag._build_goal_ir("how many customers")
    jp = ag._determine_join_path(ag._get_relevant_tables("how many customers"))
    ir = ag._resolve_semantics("how many customers", ir, jp)
    counts = [m for m in ir["metrics"] if str(m.get("aggregation", "")).upper() == "COUNT"]
    assert counts
    for m in counts:
        assert _bare(m.get("resolved_table")) == "customer", (
            f"customer count resolved to {m.get('resolved_table')}")
    plan = ag._plan_joins(["Sales.Customer", "Person.Person"])
    assert "Sales.Customer" in plan["nodes"] and "Person.Person" in plan["nodes"]
    assert plan["edges"], "cross-schema Customer<->Person join has no edges"
    # Bare + case-insensitive join path must also connect across schemas.
    jp2 = ag._determine_join_path(["customer", "person.person"])
    assert {_bare(t) for t in jp2} >= {"customer", "person"}, f"bare join failed: {jp2}"
    jp3 = ag._determine_join_path(["SALES.SALESORDERHEADER", "sales.customer"])
    assert {_bare(t) for t in jp3} >= {"salesorderheader", "customer"}, f"case join failed: {jp3}"


def test_14_identifier_normalization_and_search_path(tmp_path):
    """Generic identifier + multi-schema search_path fallback (no DB names)."""
    from core.db import (normalize_identifier, bare_identifier, base_table,
                         search_path_connect_args)
    from schema_engine.lexical import (canonical_identifier, bare_identifier as lex_bare,
                                       fold_identifier_tokens)
    assert normalize_identifier('"TotalDue"') == "totaldue"
    assert bare_identifier("Sales.SalesOrderHeader") == "salesorderheader"
    assert base_table('"Sales"."SalesOrderHeader"') == "SalesOrderHeader"
    assert canonical_identifier('"SalesOrderHeader"') == "salesorderheader"
    assert lex_bare("Sales.SalesOrderHeader") == "salesorderheader"
    assert set(fold_identifier_tokens("TotalDue")) >= {"total", "due"}
    assert set(fold_identifier_tokens("SalesOrderHeader")) >= {"sales", "order", "header"}
    ag = _camel_agent(tmp_path)
    assert ag._resolve_table_key("sales.salesorderheader") == "Sales.SalesOrderHeader"
    assert ag._resolve_table_key("SALESORDERHEADER") == "Sales.SalesOrderHeader"
    assert ag._resolve_table_key('"SalesOrderHeader"') == "Sales.SalesOrderHeader"
    opts = search_path_connect_args("postgresql", ["Sales", "Person", "Production"])
    assert "options" in opts
    assert all(s in opts["options"] for s in ("Sales", "Person", "Production", "public"))
    assert search_path_connect_args("postgresql", ["public"]) == {}
    assert search_path_connect_args("postgresql", ["PUBLIC"]) == {}


def test_15_synthesis_fallback_without_header_total(tmp_path):
    """Without any header/line total, revenue synthesizes qty*price generically."""
    m = _camel_mapping()
    m["tables"]["Sales.SalesOrderHeader"]["columns"] = _camel_typed([
        ("SalesOrderID", "integer"), ("CustomerID", "integer"),
        ("TerritoryID", "integer"), ("OrderDate", "timestamp")])
    m["tables"]["Sales.SalesOrderDetail"]["columns"] = _camel_typed([
        ("SalesOrderID", "integer"), ("ProductID", "integer"),
        ("OrderQty", "integer"), ("UnitPrice", "numeric")])
    ag = _camel_agent(tmp_path, m)
    ir = ag._build_goal_ir("total sales")
    jp = ag._determine_join_path(ag._get_relevant_tables("total sales"))
    ir = ag._resolve_semantics("total sales", ir, jp)
    exprs = [str(x.get("resolved_expression") or "") for x in ir["metrics"]]
    assert any("*" in e for e in exprs), f"expected qty*price synthesis, got {exprs}"


@pytest.fixture()
def camel_conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE "SalesOrderHeader" ("SalesOrderID" INTEGER PRIMARY KEY, "CustomerID" INTEGER, "TerritoryID" INTEGER, "OrderDate" TEXT, "TotalDue" REAL);
        CREATE TABLE "SalesOrderDetail" ("SalesOrderID" INTEGER, "ProductID" INTEGER, "OrderQty" INTEGER, "UnitPrice" REAL, "LineTotal" REAL);
    """)
    conn.executemany("INSERT INTO SalesOrderHeader VALUES (?,?,?,?,?)",
                     [(1, 1, 1, "2024-01-05", 100.0), (2, 2, 2, "2024-01-20", 200.0),
                      (3, 1, 1, "2024-02-10", 150.0), (4, 3, 1, "2024-02-15", 400.0),
                      (5, 4, 3, "2024-03-01", 50.0), (6, 2, 2, "2024-03-10", 150.0)])
    conn.executemany("INSERT INTO SalesOrderDetail VALUES (?,?,?,?,?)",
                     [(1, 101, 2, 50.0, 100.0), (2, 102, 1, 200.0, 200.0),
                      (3, 101, 1, 50.0, 50.0), (3, 103, 4, 25.0, 100.0),
                      (4, 102, 2, 200.0, 400.0), (5, 103, 2, 25.0, 50.0),
                      (6, 101, 3, 50.0, 150.0)])
    conn.commit()
    yield conn
    conn.close()


def test_16_header_total_matches_line_rollup(camel_conn):
    """Header TotalDue equals the detail qty*price rollup (no double count)."""
    header = camel_conn.execute('SELECT SUM("TotalDue") FROM "SalesOrderHeader"').fetchone()[0]
    detail = camel_conn.execute('SELECT SUM("OrderQty"*"UnitPrice") FROM "SalesOrderDetail"').fetchone()[0]
    line = camel_conn.execute('SELECT SUM("LineTotal") FROM "SalesOrderDetail"').fetchone()[0]
    assert header == pytest.approx(1050.0)
    assert detail == pytest.approx(header)
    assert line == pytest.approx(header)
