"""Small database / identifier helpers."""

import re
from urllib.parse import quote_plus

_URI_SCHEMES = {
    "postgresql": "postgresql",
    "mysql": "mysql+pymysql",
}


def build_db_uri(config, db_type="postgresql"):
    """Build a SQLAlchemy connection URI for the given database type.

    config is a dict with user/password/host/port/db keys. For MySQL the
    scheme is ``mysql+pymysql`` (needs pymysql installed).
    """
    scheme = _URI_SCHEMES.get(db_type, "postgresql")
    return (
        f"{scheme}://{quote_plus(config['user'])}:{quote_plus(config['password'])}"
        f"@{config['host']}:{config['port']}/{quote_plus(config['db'])}"
    )


def search_path_connect_args(db_type, schemas):
    """libpq connect_args pinning PostgreSQL schema resolution.

    Multi-schema databases (AdventureWorks-style) keep tables outside
    ``public``; bare ``FROM "employee"`` then fails with UndefinedTable
    on every engine that does not set a search_path. Returns
    ``{"options": "-csearch_path=s1,s2,public"}`` for PostgreSQL with a
    non-trivial schema list, else ``{}`` (single-public, MySQL, SQLite
    and unknown inputs are untouched).
    """
    try:
        if str(db_type or "").lower() != "postgresql":
            return {}
        seen = set()
        clean = []
        for s in (schemas or []):
            norm = _strip_identifier_quotes(s)
            if not norm:
                continue
            key = norm.lower()
            if key not in seen:
                seen.add(key)
                clean.append(norm)
        if not clean:
            return {}
        # Single-public (any casing/quoting) stays untouched; anything else
        # is a multi-schema search_path fallback with public appended so
        # bare names resolve across ALL populated schemas.
        if len(clean) == 1 and clean[0].lower() == "public":
            return {}
        if "public" not in seen:
            clean = clean + ["public"]
        return {"options": "-csearch_path=" + ",".join(clean)}
    except Exception:
        return {}


def _strip_identifier_quotes(name):
    """Strip surrounding quotes/brackets/whitespace (generic, dialect-agnostic)."""
    try:
        s = str(name or "").strip()
        while len(s) >= 2 and ((s[0] == '"' and s[-1] == '"')
                               or (s[0] == "'" and s[-1] == "'")
                               or (s[0] == "`" and s[-1] == "`")
                               or (s[0] == "[" and s[-1] == "]")):
            s = s[1:-1].strip()
        return s
    except Exception:
        return str(name or "")


def normalize_identifier(name):
    """Lower + strip quotes/brackets/whitespace (generic identifier match key)."""
    try:
        return _strip_identifier_quotes(name).lower()
    except Exception:
        return str(name or "").lower()


def bare_identifier(name):
    """Bare last segment, lower + quote-stripped (schema-qualified aware)."""
    try:
        segs = [s for s in str(name or "").split(".") if s.strip()]
        last = segs[-1] if segs else str(name or "")
        return normalize_identifier(last)
    except Exception:
        return str(name or "").lower()


def base_table(name):
    """Bare table segment of a possibly schema-qualified key.

    ``sales.salesorderheader`` -> ``salesorderheader``; bare names pass
    through. Deterministic builders emit the bare segment and let the
    engine search_path resolve it; the database is the final arbiter.
    Quote/bracket-wrapped segments are stripped generically so quoted
    CamelCase keys resolve case-insensitively downstream.
    """
    try:
        segs = [_strip_identifier_quotes(s) for s in str(name or "").split(".")]
        segs = [s for s in segs if s]
        return segs[-1] if segs else _strip_identifier_quotes(name)
    except Exception:
        return str(name or "")


def safe_db_name(name):
    """Sanitize a filename into a valid SQLite table/database name."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip()).strip("_") or "file_upload"