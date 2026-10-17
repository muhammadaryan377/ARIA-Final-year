"""Mocked copilot E2E (replaces legacy test_copilot_e2e.py).

The legacy script is gone on purpose: it required a live server on
127.0.0.1:8000, a hardcoded absolute CSV path, and a real GROQ_API_KEY.
This suite covers the same journey hermetically:

    register -> auth guard -> copilot without provider (honest 400, no LLM)
    -> suggestions fallback -> intent routing of the legacy questions

No absolute paths, no network, no live LLM, no server process. Auth uses an
isolated temp DB (same pattern as tests/test_aria_api.py).
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

import auth

PASSWORD = "Test1234!"

# The two questions the legacy live script asked (now routed locally).
LEGACY_QUESTIONS = [
    "How many products?",
    "What is the average discounted price?",
]


class TestCopilotE2EMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import app as app_module

        cls._tmp = tempfile.TemporaryDirectory()
        cls._orig_db = auth.DB_PATH
        auth.DB_PATH = Path(cls._tmp.name) / "copilot_mocked_auth.db"
        # Plain-HTTP TestClient needs the Secure-cookie opt-out (production
        # defaults remain Secure; see api/auth_routes._cookie_secure).
        cls._prev_cookie_secure = os.environ.get("ARIA_COOKIE_SECURE")
        os.environ["ARIA_COOKIE_SECURE"] = "0"
        cls.client = TestClient(app_module.app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        auth.DB_PATH = cls._orig_db
        cls._tmp.cleanup()
        if cls._prev_cookie_secure is None:
            os.environ.pop("ARIA_COOKIE_SECURE", None)
        else:
            os.environ["ARIA_COOKIE_SECURE"] = cls._prev_cookie_secure

    # -- helpers ---------------------------------------------------------

    def _register(self):
        user = f"copilotmock_{int(time.time() * 1000) % 10000000}"
        r = self.client.post("/api/auth/register",
                             json={"username": user, "password": PASSWORD})
        self.assertEqual(r.status_code, 200, r.text[:300])
        token = (self.client.cookies.get("aria_token")
                 or self.client.cookies.get("__Host-aria_token"))
        self.assertTrue(token, f"Auth cookie missing: {r.text[:200]}")
        return {"Authorization": f"Bearer {token}"}

    # -- tests -----------------------------------------------------------

    def test_01_register_with_test_password(self):
        headers = self._register()
        r = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:200])

    def test_02_copilot_requires_auth(self):
        # Fresh client with an empty cookie jar: the class client still
        # holds test_01's auth cookie, which would mask the guard.
        from fastapi.testclient import TestClient
        import app as app_module

        anon = TestClient(app_module.app, raise_server_exceptions=False)
        r = anon.post("/api/copilot", json={"question": LEGACY_QUESTIONS[0]})
        self.assertEqual(r.status_code, 401, r.text[:200])
        r = anon.get("/api/copilot/suggestions")
        self.assertEqual(r.status_code, 401, r.text[:200])

    def test_03_copilot_no_provider_honest_400(self):
        # Fresh user: no provider, no source. Must fail CLOSED with an
        # actionable 400 — never a 500, never a live-LLM attempt.
        # _ensure_provider falls back to local, so the 400 is about
        # missing source (not missing provider).
        headers = self._register()
        for q in LEGACY_QUESTIONS:
            r = self.client.post("/api/copilot", json={"question": q}, headers=headers)
            self.assertEqual(r.status_code, 400, r.text[:300])
            body = r.json().get("detail", "").lower()
            self.assertTrue(
                "provider" in body or "source" in body or "connect" in body,
                f"Expected provider/source/connect in error, got: {body[:200]}"
            )

    def test_04_suggestions_fallback_without_source(self):
        headers = self._register()
        r = self.client.get("/api/copilot/suggestions", headers=headers)
        self.assertEqual(r.status_code, 200, r.text[:300])
        body = r.json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(len(body.get("suggestions", [])), 4)

    def test_05_legacy_questions_route_locally(self):
        # Same questions the live script asked, classified with no LLM.
        from copilot.intent import classify

        for q in LEGACY_QUESTIONS:
            route = classify(q)
            self.assertEqual(route.intent, "data", f"{q!r} -> {route!r}")
            self.assertIn("sql", route.tools, f"{q!r} -> {route!r}")


if __name__ == "__main__":
    unittest.main()
