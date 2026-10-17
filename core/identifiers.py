"""Canonical identifier helpers (single source of truth).

Consolidates the duplicated `_is_id_like` / `_quote_ident` / `_safe_name`
helpers previously copied across insight_agent, prediction, copilot, and
goal_agent. Existing call sites keep their local names as thin aliases so
imports never break; new code should import from here.
"""

from __future__ import annotations

import re


def is_id_like(name) -> bool:
    """True for surrogate-key-like columns (id/uuid/guid/key/code tokens)."""
    try:
        low = str(name or "").strip().lower()
    except Exception:
        return False
    if not low:
        return False
    try:
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "")).lower()
        toks = set(t for t in re.split(r"[^a-z0-9]+", split) if t)
    except Exception:
        toks = set()
    if toks & {"id", "uuid", "guid", "key", "code"}:
        return True
    return (
        low in ("id", "key", "code", "codeid")
        or low.endswith("_id")
        or low.endswith("_key")
        or low.endswith("_code")
    )


def quote_ident(name: str) -> str:
    """Quote an SQL identifier (double quotes, "" escape). Validates shape."""
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ValueError("Invalid identifier.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


# Back-compat alias (goal_agent/csv code uses _quote_ident name).
_quote_ident = quote_ident
_is_id_like = is_id_like
