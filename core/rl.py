"""ARIA RL-ready decision layer (t.txt RL addendum, Phase 9).

Design rules enforced here:
- Truth layer (SQL/stats/ML) is NEVER modified by this module.
- RL only ranks pre-validated candidates (charts, insights, suggestions).
- New users get deterministic baseline ranking (cold-start safe).
- Any failure falls back to the deterministic baseline.

Storage: lightweight JSON under data/preferences/ (no raw datasets copied).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

IMPACT_UNKNOWN_NOTE = "Impact could not be reliably estimated from the available data."

# Priority hierarchy (RL below explicit correction / validity).
PRIORITY_ORDER = [
    "security",
    "data_integrity",
    "user_correction",
    "analytical_validity",
    "semantic_correctness",
    "statistical_validity",
    "system_constraints",
    "rl_personalization",
    "presentation_preference",
]


def _events_path(user_id: str | int, base_dir: Path | None = None) -> Path:
    root = base_dir or (Path(__file__).resolve().parent.parent / "data" / "preferences")
    # Contain (mirrors core.config): only alnum/_/- tokens, so a crafted
    # user_id cannot escape preferences/ via ../ segments. Numeric IDs keep
    # the user_<id> convention; other safe tokens are used verbatim.
    import re as _re
    _tok = str(user_id).strip()
    if not _tok or not _re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", _tok) or ".." in _tok:
        raise ValueError("Invalid user id.")
    try:
        _dirname = f"user_{int(_tok)}"
    except (TypeError, ValueError):
        _dirname = f"user_{_tok}"
    candidate = (root / _dirname).resolve()
    if candidate.parent != root.resolve():
        raise ValueError("Invalid user id.")
    return candidate / "rl_events.jsonl"


def record_event(
    user_id: str | int,
    *,
    event: str,
    state: dict | None = None,
    action: dict | None = None,
    outcome: dict | None = None,
    reward: dict | None = None,
    base_dir: Path | None = None,
) -> None:
    """Append a structured RL telemetry event (context + action + outcome).

    Never stores raw datasets — callers must pass summaries/references only.
    Failures are swallowed (telemetry must not break analytics).
    """
    try:
        path = _events_path(user_id, base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "event": str(event)[:64],
            "state": state or {},
            "action": action or {},
            "outcome": outcome or {},
            "reward": reward or {},
        }
        # Bound telemetry size: references and summaries, never datasets.
        blob = json.dumps(record, default=str)
        if len(blob) > 8192:
            record = {"ts": record["ts"], "event": record["event"] + ":truncated",
                      "state": {}, "action": {}, "outcome": {},
                      "reward": {"note": "oversized event dropped"}}
            blob = json.dumps(record, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(blob + "\n")
        # Rotation: oldest-first trim so a long-lived server cannot fill
        # the disk. Atomic tmp+replace so a concurrent append cannot
        # interleave a half-written file. Cap 5k lines / ~5MB per file.
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                import os as _os
                _lines = path.read_text(encoding="utf-8").splitlines()
                _tmp = path.with_suffix(".jsonl.tmp")
                _tmp.write_text("\n".join(_lines[-5000:]) + "\n", encoding="utf-8")
                _os.replace(_tmp, path)
        except Exception:
            pass
    except Exception as exc:  # telemetry is best-effort
        log.debug("RL record_event failed: %s", exc)


def rank_candidates(
    candidates: list[dict],
    *,
    prefs: dict | None = None,
    key: str = "id",
    score_key: str = "score",
) -> list[dict]:
    """Deterministic baseline ranking with optional preference boost.

    - Only ranks the candidates given (caller validates them first).
    - Preference boost is small and can never promote an invalid candidate
      (caller must filter invalid ones before calling).
    - Stable: ties keep input order.
    """
    prefs = prefs or {}
    boosts: dict[str, float] = {}
    for section in ("chart_prefs", "insight_prefs", "suggestion_clicks"):
        items = prefs.get(section, {})
        if isinstance(items, dict):
            for name, count in items.items():
                try:
                    boosts[str(name).lower()] = boosts.get(str(name).lower(), 0.0) + float(count) * 0.01
                except (TypeError, ValueError):
                    continue
    scored: list[tuple[float, int, dict]] = []
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue
        base = cand.get(score_key, 0.0)
        try:
            base_f = float(base)
        except (TypeError, ValueError):
            base_f = 0.0
        boost = boosts.get(str(cand.get(key, "")).lower(), 0.0)
        scored.append((base_f + boost, idx, cand))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored]


class RLPolicy:
    """Replaceable policy abstraction (bandits/DQN/PPO later, baseline now)."""

    name = "baseline-v1"

    def rank(
        self,
        state: dict,
        candidates: list[dict],
        *,
        prefs: dict | None = None,
    ) -> dict:
        """Rank validated candidates; returns ordered list + metadata."""
        try:
            ordered = rank_candidates(candidates, prefs=prefs)
            return {
                "policy": self.name,
                "ordered": ordered,
                "rl_influence": bool(prefs),
                "fallback": False,
            }
        except Exception as exc:
            log.warning("RLPolicy.rank failed, falling back: %s", exc)
            return {
                "policy": self.name,
                "ordered": list(candidates),
                "rl_influence": False,
                "fallback": True,
            }


def load_recent_events(user_id: str | int, limit: int = 200, base_dir: Path | None = None) -> list[dict]:
    """Read recent events for debugging/observability (RL observability layer)."""
    path = _events_path(user_id, base_dir)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
    except Exception as exc:
        log.debug("RL load_recent_events failed: %s", exc)
        return []


def as_action(action_type: str, value: Any, **meta: Any) -> dict:
    """Build an extensible action dict for the RL action space."""
    return {"type": action_type, "value": value, **meta}


def log_decision_activity(user_id: str | int, decision_id: str, event: str,
                          detail: str = "", **extra: Any) -> None:
    """PHASE 14: persisted decision activity (status/scenario/monitor events).

    Thin wrapper over record_event with the decision action space. Never
    stores raw datasets — decision_id + event + short detail only.
    """
    record_event(
        user_id,
        event=f"decision_{str(event)[:48]}",
        state={"decision_id": str(decision_id)[:80]},
        action={"detail": str(detail)[:160], **{k: v for k, v in extra.items()}},
        outcome={},
    )
