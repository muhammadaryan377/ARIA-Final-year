"""Full audit: connect Amazon Sales, test every page endpoint."""
import requests, json, time, sys

BASE = "http://127.0.0.1:8000"
AMAZON = r"C:\Users\Go\Desktop\ARIA\data\kaggle_retail_sales\karkavelrajaj__amazon-sales-dataset\amazon.csv"


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # Register + connect (token is now an HttpOnly cookie, not in JSON body)
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/register", json={"username": f"audit_{int(time.time())}", "password": "Test1234!"})
    assert r.status_code == 200, f"Register failed: {r.status_code} {r.text[:200]}"
    TOKEN = s.cookies.get("aria_token") or s.cookies.get("__Host-aria_token") or ""
    assert TOKEN, f"Auth cookie missing: {r.json()} cookies={dict(s.cookies)}"
    H = {"Authorization": f"Bearer {TOKEN}"}

    with open(AMAZON, "rb") as fp:
        r = s.post(f"{BASE}/api/connect_file", files={"file": ("amazon.csv", fp, "text/csv")}, headers=H)
    conn = r.json()
    TABLE = conn.get("table", "amazon")
    print(f"CONNECTED: {TABLE} ({conn.get('rows')} rows)")

    # 1. DASHBOARD
    print("\n=== DASHBOARD ===")
    r = s.get(f"{BASE}/api/dashboard/config", headers=H).json()
    cfg = r.get("config", r)
    kpis = cfg.get("metrics", cfg.get("kpis", []))
    charts = cfg.get("visualizations", cfg.get("charts", []))
    print(f"KPIs: {len(kpis)}")
    for m in kpis[:4]:
        print(f"  {m.get('label','')}: {m.get('value','')}")
    print(f"Charts: {len(charts)}")
    for ch in charts:
        print(f"  {ch.get('title','')} type={ch.get('type','')} data={len(ch.get('data',[]))} pts")

    # 2. SCHEMA
    print("\n=== SCHEMA ===")
    r = s.get(f"{BASE}/api/schema", headers=H).json()
    sch = r.get("schema", r)
    tables = sch.get("tables", {})
    for tname, tdata in tables.items():
        cols = tdata.get("columns", [])
        types = {}
        for c in cols:
            dt = c.get("data_type", "")
            types[dt] = types.get(dt, 0) + 1
        print(f"  {tname}: {len(cols)} cols -> {types}")

    # 3. AI ANALYTICS (ask)
    print("\n=== AI ANALYTICS ===")
    questions = [
        "How many products are there?",
        "What is the average rating?",
        "Which category has the most products?",
    ]
    for q in questions:
        try:
            r = s.post(f"{BASE}/api/ask", json={"goal": q, "table": TABLE}, headers=H, timeout=90).json()
            ans = str(r.get("result", r.get("answer", "")))[:100]
            print(f"  Q: {q}")
            print(f"  A: {ans} (ok={r.get('ok')})")
        except Exception as e:
            print(f"  Q: {q} -> ERROR: {e}")

    # 4. INSIGHTS
    print("\n=== INSIGHTS ===")
    r = s.get(f"{BASE}/api/insights", headers=H)
    print(f"  Status: {r.status_code}")
    ins = r.json() if r.status_code == 200 else {}
    if isinstance(ins, dict):
        items = ins.get("insights", ins.get("items", []))
        print(f"  Insights: {len(items)}")
        for i in items[:3]:
            print(f"    {i.get('title', i.get('description', ''))[:80]}")
    elif isinstance(ins, list):
        print(f"  Insights: {len(ins)}")

    # 5. DECISION CENTER
    print("\n=== DECISION CENTER ===")
    ov = s.get(f"{BASE}/api/decision-center/overview", headers=H).json()
    print(f"Overview: ok={ov.get('ok')} priorities={len(ov.get('priorities',[]))} opps={len(ov.get('opportunities',[]))}")
    recs = s.get(f"{BASE}/api/decision-center/recommendations?top_n=3", headers=H).json()
    print(f"Recs: ok={recs.get('ok')} prescriptions={len(recs.get('prescriptions',[]))}")
    lev = s.get(f"{BASE}/api/decision-center/levers", headers=H).json()
    print(f"Levers: ok={lev.get('ok')} count={len(lev.get('levers',[]))}")

    # 6. FORECAST
    print("\n=== FORECAST ===")
    r = s.post(f"{BASE}/api/forecast/run", json={"table": TABLE, "metric": "discounted_price"}, headers=H, timeout=30)
    print(f"  Status: {r.status_code}")
    fc = r.json() if r.status_code in (200, 400, 422) else {}
    print(f"  Response: {json.dumps(fc, default=str)[:200]}")

    # 7. PREVIEW
    print("\n=== PREVIEW ===")
    r = s.post(f"{BASE}/api/data/preview", json={"table": TABLE, "limit": 3}, headers=H).json()
    rows = r.get("rows", r.get("data", []))
    print(f"  Rows: {len(rows)}")
    if rows:
        print(f"  Columns: {list(rows[0].keys())[:8]}")

    # 8. COPILOT
    print("\n=== COPILOT ===")
    try:
        r = s.post(f"{BASE}/api/copilot", json={"message": "What is the average discounted price?", "table": TABLE}, headers=H, timeout=60)
        cp = r.json() if r.status_code == 200 else {"status": r.status_code}
        print(f"  ok={cp.get('ok')} answer={str(cp.get('answer',''))[:150]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nDONE")


if __name__ == "__main__":
    main()
