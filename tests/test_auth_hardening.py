"""Auth hardening + cloud-consent regression tests (security hardening pass).

Hermetic: no network, no LLM, no live database. Covers:
- valid / expired / malformed / invalid-signature tokens
- missing sub, unexpected sub type (bool/float/injection/negative/zero)
- secret rotation (SECRET_PREVIOUS verify-only; new issues use SECRET)
- TOKEN_TTL_DAYS clamp (1..7, non-integer raises)
- production fail-fast (ARIA_ENV=production requires strong JWT_SECRET)
- cloud consent at provider-selection layer (construction raises, no network)
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt

import auth


def _make_token(sub=None, secret=None, exp_days=1, include_sub=True, algo=None):
    now = datetime.now(timezone.utc)
    payload = {
        "username": "hardening_user",
        "iat": now,
        "exp": now + timedelta(days=exp_days),
    }
    if include_sub:
        payload["sub"] = sub
    return jwt.encode(
        payload, secret or auth.SECRET, algorithm=algo or auth.JWT_ALGO
    )


class _TempAuthDB(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_db = auth.DB_PATH
        auth.DB_PATH = Path(self._tmp.name) / "test_hardening.db"
        self.addCleanup(setattr, auth, "DB_PATH", self._orig_db)
        # Snapshot module secrets; tests that mutate them restore below.
        self._orig_secret = auth.SECRET
        self._orig_prev = auth.SECRET_PREVIOUS
        self.addCleanup(setattr, auth, "SECRET", self._orig_secret)
        self.addCleanup(setattr, auth, "SECRET_PREVIOUS", self._orig_prev)


class TestTokenVerification(_TempAuthDB):
    def test_valid_token_verifies(self):
        uid = auth.register("harden_valid", "Secret123!")
        token = auth.issue_token(uid, "harden_valid")
        result = auth.verify_token(token)
        self.assertIsNotNone(result)
        self.assertEqual(result["user_id"], uid)
        self.assertEqual(result["username"], "harden_valid")

    def test_expired_token_returns_none(self):
        token = _make_token(sub="123", exp_days=-1)
        self.assertIsNone(auth.verify_token(token))

    def test_malformed_tokens_return_none(self):
        for bad in ("not.a.jwt", "Bearer xyz", "eyJhbGciOi", "", None, 123, b"bytes"):
            with self.subTest(token=bad):
                try:
                    result = auth.verify_token(bad)
                except Exception as exc:
                    self.fail(f"verify_token raised {exc!r}")
                self.assertIsNone(result)

    def test_invalid_signature_returns_none(self):
        token = _make_token(sub="42", secret="wrong-secret-xyz-0000")
        self.assertIsNone(auth.verify_token(token))

    def test_missing_sub_returns_none(self):
        token = _make_token(sub=None, include_sub=False)
        self.assertIsNone(auth.verify_token(token))

    def test_unexpected_sub_types_return_none(self):
        for bad_sub in (True, False, 3.5, "abc", "", None, "1; DROP TABLE users",
                        "-5", "0", 0, -3, "1.0", "  ", "None"):
            with self.subTest(sub=bad_sub):
                token = _make_token(sub=bad_sub)
                try:
                    result = auth.verify_token(token)
                except Exception as exc:
                    self.fail(f"verify_token raised {exc!r} for sub={bad_sub!r}")
                self.assertIsNone(result, f"sub={bad_sub!r} must not verify")

    def test_positive_int_and_digit_string_subs_verify(self):
        uid = auth.register("harden_sub", "Secret123!")
        # issue_token emits str(user_id); a digit string must verify.
        token = _make_token(sub=str(uid))
        result = auth.verify_token(token)
        self.assertIsNotNone(result, f"sub={str(uid)!r} must verify")
        self.assertEqual(result["user_id"], uid)
        # Raw int sub: PyJWT strict subject validation rejects it at decode
        # (InvalidSubjectError -> None). Either way it must never verify as
        # another user and never raise.
        try:
            int_result = auth.verify_token(_make_token(sub=uid))
        except Exception as exc:
            self.fail(f"verify_token raised {exc!r} for int sub")
        self.assertTrue(int_result is None or int_result["user_id"] == uid)


class TestSecretRotation(_TempAuthDB):
    def test_previous_secret_verifies_only(self):
        import secrets as _secrets

        old_secret = _secrets.token_hex(32)
        new_secret = _secrets.token_hex(32)
        # Token minted under the OLD secret.
        now = datetime.now(timezone.utc)
        old_token = jwt.encode(
            {"sub": "77", "username": "rot",
             "iat": now, "exp": now + timedelta(days=1)},
            old_secret, algorithm=auth.JWT_ALGO,
        )
        # Without rotation configured the old token must fail.
        auth.SECRET = new_secret
        auth.SECRET_PREVIOUS = None
        self.assertIsNone(auth.verify_token(old_token))
        # With SECRET_PREVIOUS set, the old token verifies...
        auth.SECRET_PREVIOUS = old_secret
        result = auth.verify_token(old_token)
        self.assertIsNotNone(result)
        self.assertEqual(result["user_id"], 77)
        # ...but newly issued tokens still use SECRET (fail under old only).
        uid = auth.register("harden_rot", "Secret123!")
        fresh = auth.issue_token(uid, "harden_rot")
        fresh_payload = jwt.decode(
            fresh, new_secret, algorithms=[auth.JWT_ALGO], options={"verify_exp": False}
        )
        self.assertEqual(int(fresh_payload["sub"]), uid)
        with self.assertRaises(jwt.InvalidSignatureError):
            jwt.decode(fresh, old_secret, algorithms=[auth.JWT_ALGO])


class TestTTLClamp(_TempAuthDB):
    def test_ttl_clamped_1_to_7(self):
        old = os.environ.get("TOKEN_TTL_DAYS")
        try:
            os.environ.pop("TOKEN_TTL_DAYS", None)
            self.assertEqual(auth._token_ttl_days(), 1)
            os.environ["TOKEN_TTL_DAYS"] = "30"
            self.assertEqual(auth._token_ttl_days(), 7)
            os.environ["TOKEN_TTL_DAYS"] = "0"
            self.assertEqual(auth._token_ttl_days(), 1)
            os.environ["TOKEN_TTL_DAYS"] = "3"
            self.assertEqual(auth._token_ttl_days(), 3)
        finally:
            if old is None:
                os.environ.pop("TOKEN_TTL_DAYS", None)
            else:
                os.environ["TOKEN_TTL_DAYS"] = old

    def test_ttl_non_integer_raises(self):
        old = os.environ.get("TOKEN_TTL_DAYS")
        try:
            os.environ["TOKEN_TTL_DAYS"] = "forever"
            with self.assertRaises(ValueError):
                auth._token_ttl_days()
        finally:
            if old is None:
                os.environ.pop("TOKEN_TTL_DAYS", None)
            else:
                os.environ["TOKEN_TTL_DAYS"] = old


class TestProductionFailFast(unittest.TestCase):
    def test_production_requires_jwt_secret(self):
        old_env, old_secret = os.environ.get("ARIA_ENV"), os.environ.get("JWT_SECRET")
        try:
            os.environ["ARIA_ENV"] = "production"
            os.environ.pop("JWT_SECRET", None)
            with self.assertRaises(RuntimeError) as ctx:
                auth._load_secret()
            self.assertIn("JWT_SECRET", str(ctx.exception))
        finally:
            if old_env is None:
                os.environ.pop("ARIA_ENV", None)
            else:
                os.environ["ARIA_ENV"] = old_env
            if old_secret is None:
                os.environ.pop("JWT_SECRET", None)
            else:
                os.environ["JWT_SECRET"] = old_secret

    def test_production_rejects_short_secret(self):
        old_env, old_secret = os.environ.get("ARIA_ENV"), os.environ.get("JWT_SECRET")
        try:
            os.environ["ARIA_ENV"] = "production"
            os.environ["JWT_SECRET"] = "short"
            with self.assertRaises(RuntimeError):
                auth._load_secret()
            os.environ["JWT_SECRET"] = "a" * 64
            self.assertEqual(auth._load_secret(), "a" * 64)
        finally:
            if old_env is None:
                os.environ.pop("ARIA_ENV", None)
            else:
                os.environ["ARIA_ENV"] = old_env
            if old_secret is None:
                os.environ.pop("JWT_SECRET", None)
            else:
                os.environ["JWT_SECRET"] = old_secret

    def test_dev_keeps_autogenerated_fallback(self):
        # Dev (non-production) without JWT_SECRET must not raise: the
        # existing per-machine autogenerated secret mechanism is kept.
        old_env, old_secret = os.environ.get("ARIA_ENV"), os.environ.get("JWT_SECRET")
        try:
            os.environ.pop("ARIA_ENV", None)
            os.environ.pop("JWT_SECRET", None)
            self.assertFalse(auth._is_production())
            secret = auth._load_secret()  # must not raise
            self.assertTrue(isinstance(secret, str) and len(secret) > 0)
        finally:
            if old_env is None:
                os.environ.pop("ARIA_ENV", None)
            else:
                os.environ["ARIA_ENV"] = old_env
            if old_secret is None:
                os.environ.pop("JWT_SECRET", None)
            else:
                os.environ["JWT_SECRET"] = old_secret


class TestCloudConsentHermetic(unittest.TestCase):
    """CLOUD_CONSENT=0 blocks hosted providers at selection time (no network)."""

    def _set_cloud_env(self, allowed="1", consent="0"):
        old_allowed = os.environ.get("ARIA_CLOUD_ALLOWED")
        old_consent = os.environ.get("ARIA_CLOUD_CONSENT")
        os.environ["ARIA_CLOUD_ALLOWED"] = allowed
        os.environ["ARIA_CLOUD_CONSENT"] = consent
        self.addCleanup(
            lambda: (
                os.environ.pop("ARIA_CLOUD_ALLOWED", None),
                os.environ.pop("ARIA_CLOUD_CONSENT", None),
                (os.environ.__setitem__("ARIA_CLOUD_ALLOWED", old_allowed)
                 if old_allowed is not None else None),
                (os.environ.__setitem__("ARIA_CLOUD_CONSENT", old_consent)
                 if old_consent is not None else None),
            )
        )

    def test_construction_denied_without_consent(self):
        from llm_provider import create_provider

        self._set_cloud_env(allowed="1", consent="0")
        for name in ("cloud", "deepseek", "omniroute", "grok"):
            with self.subTest(provider=name):
                # Must raise BEFORE any network I/O (selection-layer guard).
                with self.assertRaises(RuntimeError) as ctx:
                    create_provider(provider=name, api_key="dummy-test-key")
                self.assertIn("consent", str(ctx.exception).lower())

    def test_kill_switch_denies_even_with_consent(self):
        from llm_provider import create_provider

        self._set_cloud_env(allowed="0", consent="1")
        with self.assertRaises(RuntimeError):
            create_provider(provider="cloud", api_key="dummy-test-key")

    def test_construction_permitted_with_consent(self):
        from llm_provider import create_provider

        self._set_cloud_env(allowed="1", consent="1")
        for name in ("cloud", "deepseek", "omniroute", "grok"):
            with self.subTest(provider=name):
                try:
                    llm = create_provider(provider=name, api_key="dummy-test-key")
                except RuntimeError as exc:
                    if "not installed" in str(exc):
                        self.skipTest(f"optional dependency for {name} missing: {exc}")
                    raise
                self.assertEqual(llm.provider, name)

    def test_call_denied_without_consent_no_network(self):
        from llm_provider import LLMProvider

        self._set_cloud_env(allowed="1", consent="1")
        try:
            llm = LLMProvider("deepseek", api_key="dummy-test-key")
        except RuntimeError as exc:
            if "not installed" in str(exc):
                self.skipTest(f"optional dependency missing: {exc}")
            raise
        # Revoke consent after construction: calls must still fail closed
        # before touching the network.
        os.environ["ARIA_CLOUD_CONSENT"] = "0"
        with self.assertRaises(RuntimeError):
            llm.chat("sql", [{"role": "user", "content": "hi"}])
        with self.assertRaises(RuntimeError):
            llm.ping(timeout=5)

    def test_local_always_allowed(self):
        from llm_provider import LLMProvider, create_provider

        self._set_cloud_env(allowed="0", consent="0")
        llm = create_provider(provider="local")
        self.assertEqual(llm.provider, "local")
        llm2 = LLMProvider("local")
        self.assertEqual(llm2.provider, "local")


if __name__ == "__main__":
    unittest.main()
