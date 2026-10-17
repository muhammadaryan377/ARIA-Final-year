"""Gated live-relational regression tests (PostgreSQL / MySQL).

Consumes POSTGRES_TEST_URL / MYSQL_TEST_URL from the environment (see
.env.example). When absent, every test skips explicitly with
`NOT EXECUTED — TEST CREDENTIALS UNAVAILABLE`.

Never commit values, no fake creds, no weakened auth. The suite stays
available for environments where credentials exist (CI with secrets, local
dev with a throwaway database). Destructive operations are additionally
gated by ARIA_ALLOW_BENCHMARK=1 (benchmark_fk.py precedent).
"""

import os

import pytest

POSTGRES_TEST_URL = os.getenv("POSTGRES_TEST_URL", "").strip()
MYSQL_TEST_URL = os.getenv("MYSQL_TEST_URL", "").strip()

requires_postgres = pytest.mark.skipif(
    not POSTGRES_TEST_URL,
    reason="NOT EXECUTED — TEST CREDENTIALS UNAVAILABLE",
)
requires_mysql = pytest.mark.skipif(
    not MYSQL_TEST_URL,
    reason="NOT EXECUTED — TEST CREDENTIALS UNAVAILABLE",
)


@requires_postgres
def test_postgres_live_connection_select_only():
    """Read-only smoke: connect + SELECT 1 over the provided test URL."""
    import sqlalchemy as _sa

    assert POSTGRES_TEST_URL.startswith("postgresql"), "POSTGRES_TEST_URL must be a postgresql DSN"
    eng = _sa.create_engine(POSTGRES_TEST_URL, connect_args={"connect_timeout": 5})
    try:
        with eng.connect() as conn:
            assert conn.execute(_sa.text("SELECT 1")).scalar() == 1
    finally:
        eng.dispose()


@requires_mysql
def test_mysql_live_connection_select_only():
    """Read-only smoke: connect + SELECT 1 over the provided test URL."""
    import sqlalchemy as _sa

    assert MYSQL_TEST_URL.startswith("mysql"), "MYSQL_TEST_URL must be a mysql DSN"
    eng = _sa.create_engine(MYSQL_TEST_URL, connect_args={"connect_timeout": 5})
    try:
        with eng.connect() as conn:
            assert conn.execute(_sa.text("SELECT 1")).scalar() == 1
    finally:
        eng.dispose()
