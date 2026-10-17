"""Session isolation regression tests (hyp.txt Phases 3,4,7-partial).

No mock data: uses real JWT verification, real session store, and real
path resolution. Auth DB is isolated via a temp file.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import jwt

import auth
from core.config import (
    DEFAULT_SESSION,
    INSIGHTS_PATH,
    PROCESSED_PATH,
    USERS_DIR,
    get_session,
    get_session_for_request,
    get_user_artifact_paths,
)


def _client():
    from fastapi.testclient import TestClient
    import app as app_module
    return TestClient(app_module.app, raise_server_exceptions=False)


def _fake_request(token=None, cookies=None):
    headers = {}
    if token is not None:
        headers["authorization"] = token
    return SimpleNamespace(headers=headers, cookies=cookies or {})


def test_missing_jwt_401():
    c = _client()
    r = c.get("/api/session/status")
    assert r.status_code == 401, r.text[:300]


def test_malformed_jwt_401():
    c = _client()
    for bad in [
        "Bearer not.a.jwt",
        "Bearer null",
        "Bearer undefined",
        "Bearer ",
        "Bearer xyz",
    ]:
        r = c.get("/api/session/status", headers={"Authorization": bad})
        assert r.status_code == 401, f"{bad!r} -> {r.status_code} {r.text[:200]}"


def test_expired_jwt_401():
    c = _client()
    payload = {
        "sub": "999999",
        "username": "expired_user",
        "iat": datetime.now(timezone.utc) - timedelta(days=2),
        "exp": datetime.now(timezone.utc) - timedelta(seconds=10),
    }
    expired = jwt.encode(payload, auth.SECRET, algorithm=auth.JWT_ALGO)
    assert auth.verify_token(expired) is None
    r = c.get("/api/session/status", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401, r.text[:300]


def test_no_default_fallback():
    from fastapi import HTTPException
    # Missing token must raise 401, never return DEFAULT_SESSION.
    for req in [
        _fake_request(token=None),
        _fake_request(token=""),
        _fake_request(token="Bearer garbage.token.here"),
        _fake_request(token="Bearer null"),
    ]:
        try:
            sess = get_session_for_request(req)
        except HTTPException as exc:
            assert exc.status_code == 401
        else:
            raise AssertionError(f"expected 401, got session keys {sorted(sess.keys())}")
    # Expired token must also raise 401.
    payload = {
        "sub": "999998",
        "username": "expired2",
        "iat": datetime.now(timezone.utc) - timedelta(days=2),
        "exp": datetime.now(timezone.utc) - timedelta(seconds=10),
    }
    expired = jwt.encode(payload, auth.SECRET, algorithm=auth.JWT_ALGO)
    try:
        get_session_for_request(_fake_request(token=f"Bearer {expired}"))
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("expired token must raise 401")


def test_cross_user_artifact_paths_differ():
    uid1, uid2 = 910101, 910102
    try:
        s1 = get_session(uid1)
        s2 = get_session(uid2)
        p1 = Path(str(s1["processed_path"]))
        p2 = Path(str(s2["processed_path"]))
        i1 = Path(str(s1["insights_path"]))
        i2 = Path(str(s2["insights_path"]))
        # Different users never share a file.
        assert p1 != p2
        assert i1 != i2
        # Paths embed the owning user id.
        assert f"user_{uid1}" in str(p1) and f"user_{uid1}" in str(i1)
        assert f"user_{uid2}" in str(p2) and f"user_{uid2}" in str(i2)
        # No user data on shared global paths.
        assert p1 != PROCESSED_PATH and p2 != PROCESSED_PATH
        assert i1 != INSIGHTS_PATH and i2 != INSIGHTS_PATH
        assert str(DEFAULT_SESSION["processed_path"]) == str(PROCESSED_PATH)
        # Dirs created deterministically.
        assert p1.parent.exists() and i1.parent.exists()
        assert p2.parent.exists() and i2.parent.exists()
        # Resolved paths stay within the project base.
        from core.config import BASE_DIR
        base = BASE_DIR.resolve()
        for p in (p1, p2, i1, i2):
            assert p.resolve().is_relative_to(base)
        # Helper roots are isolated per user too.
        r1 = get_user_artifact_paths(uid1)["root"]
        r2 = get_user_artifact_paths(uid2)["root"]
        assert r1 != r2
        assert r1.parent == USERS_DIR.resolve() or r1.parent == USERS_DIR
    finally:
        from core.config import SESSION_STORE, SESSION_LOCK
        with SESSION_LOCK:
            SESSION_STORE.pop(uid1, None)
            SESSION_STORE.pop(uid2, None)


def test_require_data_provider_enforce_auth():
    from fastapi import HTTPException
    from core.deps import require_data, require_provider
    req = _fake_request(token=None)
    for fn in (require_data, require_provider):
        try:
            fn(req)
        except HTTPException as exc:
            assert exc.status_code == 401, f"{fn.__name__} -> {exc.status_code}"
        else:
            raise AssertionError(f"{fn.__name__} must raise 401 without auth")
