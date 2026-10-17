"""Full end-to-end journey test through the real FastAPI app.

Uses TestClient (same routes, same session store, same CSV pipeline) so the
whole user flow is exercised without a live server or external database:

    register/login -> connect CSV -> /api/ask -> /api/copilot -> /api/insight
    -> /api/predictive/forecast -> /api/hypothesis -> history -> cleanup

The LLM provider is 'local' (no Ollama call is made at construction time).
Where an LLM call would be required, the flow is expected to degrade
gracefully rather than error — that graceful behaviour is what's asserted.
"""

import io
import os
import time
import unittest
from pathlib import Path

import auth

# The full-journey test drives a real LLM (SQL generation + narration). Local
# Ollama is intentionally slow on commodity hardware, so this test is opt-in:
# set E2E_PROVIDER=cloud (fast) or E2E_PROVIDER=local (slow) to run it. Without
# the variable only the fast, LLM-free endpoints run — the default suite never
# blocks on a model.
E2E_PROVIDER = os.getenv("E2E_PROVIDER", "").strip().lower()


CSV = (
    "region,product,order_date,amount,quantity\n"
    "North,Widget,2024-01-05,100.0,2\n"
    "North,Widget,2024-02-10,150.0,3\n"
    "South,Gadget,2024-01-20,300.0,1\n"
    "North,Gadget,2024-03-01,50.0,5\n"
    "South,Widget,2024-02-15,220.0,4\n"
    "East,Gadget,2024-03-10,410.0,2\n"
    "East,Widget,2024-01-30,90.0,1\n"
    "South,Gadget,2024-04-05,180.0,3\n"
)


class TestFullE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from fastapi.testclient import TestClient
        import app as app_module

        cls._tmp = tempfile.TemporaryDirectory()
        cls._orig_db = auth.DB_PATH
        auth.DB_PATH = Path(cls._tmp.name) / "e2e_auth.db"
        cls.client = TestClient(app_module.app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        auth.DB_PATH = cls._orig_db
        cls._tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def _auth(self):
        user = f"e2e_{int(time.time() * 1000) % 10000000}"
        r = self.client.post("/api/auth/register",
                             json={"username": user, "password": "Secret123!"})
        self.assertEqual(r.status_code, 200, r.text[:300])
        token = self.client.cookies.get("aria_token") or self.client.cookies.get("__Host-aria_token")
        self.assertTrue(token, f"Auth cookie missing: {r.json()} cookies={dict(self.client.cookies)}")
        return {"Authorization": f"Bearer {token}"}

    def _connect_csv(self, headers):
        files = {"file": ("sales.csv", io.BytesIO(CSV.encode()), "text/csv")}
        r = self.client.post("/api/connect_file",
                             data={"provider": E2E_PROVIDER or "local"},
                             files=files, headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:400])
        return r.json()

    # -- tests -----------------------------------------------------------

    def test_01_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("critical", r.json())

    def test_02_providers_endpoint(self):
        r = self.client.get("/api/providers")
        self.assertEqual(r.status_code, 200)
        self.assertIn("providers", r.json())

    def test_03_auth_guard(self):
        r = self.client.post("/api/ask", json={"goal": "top products"})
        self.assertEqual(r.status_code, 401)

    @unittest.skipUnless(E2E_PROVIDER in ("cloud", "local"),
                         "set E2E_PROVIDER=cloud|local to run the LLM journey")
    def test_04_full_journey(self):
        headers = self._auth()

        # Connect a CSV source (runs schema extraction + SQLite load).
        conn = self._connect_csv(headers)
        self.assertTrue(conn.get("ok"), conn)
        self.assertTrue(conn.get("table"))

        # Session status reflects the connection.
        r = self.client.get("/api/session/status", headers=headers)
        self.assertEqual(r.status_code, 200)

        # /api/ask must return the analytical envelope (LLM may fall back).
        r = self.client.post("/api/ask",
                             json={"goal": "total amount by region"}, headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:400])
        body = r.json()
        self.assertIn("ok", body)
        self.assertIn("message", body)
        if body.get("ok"):
            self.assertIn("provenance", body)
            self.assertIn("chart_spec", body)
            # New analytical layer keys must be present.
            self.assertIn("analysis", body)
            self.assertIn("result_valid", body)

        # /api/copilot must run the RAG pipeline (or degrade gracefully).
        r = self.client.post("/api/copilot",
                             json={"question": "total amount by region"}, headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:400])
        cop = r.json()
        self.assertIn("ok", cop)
        self.assertIn("answer", cop)
        self.assertTrue(cop["answer"])

        # /api/insight must return an insights envelope.
        r = self.client.post("/api/insight", headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:300])
        self.assertIn("insights", r.json())

        # Predictive forecast endpoint responds (may be invalid for tiny data).
        r = self.client.post("/api/predictive/forecast",
                             json={"horizon": 3}, headers=headers)
        self.assertIn(r.status_code, (200, 400, 422), r.text[:300])

        # Hypothesis endpoint responds with the honest contract.
        r = self.client.post("/api/hypothesis",
                             json={"hypothesis": "amount relates to quantity"},
                             headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:300])
        self.assertIn("ok", r.json())

        # History lists the run.
        r = self.client.get("/api/history", headers=headers)
        self.assertIn(r.status_code, (200, 404), r.text[:300])
        if r.status_code == 200:
            self.assertIn("items", r.json())

    def test_05_copilot_without_connection_is_graceful(self):
        headers = self._auth()
        r = self.client.post("/api/copilot", json={"question": "anything"}, headers=headers)
        self.assertEqual(r.status_code, 400, r.text[:300])

    @unittest.skipUnless(E2E_PROVIDER in ("cloud", "local"),
                         "set E2E_PROVIDER=cloud|local to run the LLM journey")
    def test_06_ask_rejects_gibberish(self):
        headers = self._auth()
        self._connect_csv(headers)
        r = self.client.post("/api/ask", json={"goal": "asdf asdf qwerty"}, headers=headers)
        self.assertEqual(r.status_code, 400, r.text[:300])


if __name__ == "__main__":
    unittest.main()