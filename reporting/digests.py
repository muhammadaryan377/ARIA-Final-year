"""ARIA scheduled digests (Phase 4).

Per-user daily/weekly digest schedules stored as JSON per user, with an
in-process dependency-free background scheduler (threading + timer) that
reuses the existing engines:

- dashboard-style content via ``services.intelligence_service.generate_report``
  (latest analysis + hypotheses + goals + scenarios — all live values only),
- top insights = top live hypotheses embedded by ``generate_report``,
- forecast-vs-actual attached only where a saved forecast already exists
  (snapshot counts, never invented numbers).

Every successful run persists a normal report via ``generate_report`` so the
digest shows up in report history with its definition (re-run works through
the existing POST /reports/{id}/rerun path). Failures record a failed run
entry on the schedule and never crash the scheduler.

Threading model: one daemon thread, 60s poll, ``threading.Event`` for
graceful shutdown + ``atexit`` hook. Startup is lazy (``ensure_started`` is
called from the API routes) so no new infra or app wiring is needed.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("aria.digests")

POLL_SECONDS = 60
MAX_RUNS_KEPT = 20

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event = threading.Event()
_started = False


# ---------------------------------------------------------------------------
# Storage (per-user JSON, same layout as intelligence_service)
# ---------------------------------------------------------------------------

def _users_dir() -> Path:
    try:
        from core.config import USERS_DIR as _d
        _d.mkdir(parents=True, exist_ok=True)
        return _d
    except Exception:
        d = Path("data") / "users"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _user_intel_dir(user_id: int) -> Path:
    d = _users_dir() / f"user_{user_id}" / "intelligence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _schedules_path(user_id: int) -> Path:
    return _user_intel_dir(user_id) / "digest_schedules.json"


def _load_schedules(user_id: int) -> list:
    try:
        p = _schedules_path(user_id)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_schedules(user_id: int, items: list) -> None:
    try:
        _schedules_path(user_id).write_text(
            json.dumps(items or [], default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("digest: could not persist schedules for user=%s (%s)", user_id, exc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

VALID_FREQUENCIES = ("daily", "weekly")


def _clean_sections(sections: Any) -> list:
    if not isinstance(sections, list):
        return []
    out = []
    for s in sections:
        name = s.get("name") if isinstance(s, dict) else s
        name = str(name or "").strip()
        if name and name not in out:
            out.append(name)
    return out[:12]


def list_digests(user_id: int) -> list:
    return _load_schedules(user_id)


def get_digest(user_id: int, digest_id: str) -> dict | None:
    for d in _load_schedules(user_id):
        if str(d.get("id") or "") == str(digest_id):
            return d
    return None


def create_digest(user_id: int, data: dict) -> dict:
    data = data or {}
    freq = str(data.get("frequency") or "daily").strip().lower()
    if freq not in VALID_FREQUENCIES:
        freq = "daily"
    digest = {
        "id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "name": str(data.get("name") or "Daily Digest").strip()[:200] or "Daily Digest",
        "description": str(data.get("description") or "").strip()[:500],
        "frequency": freq,
        "enabled": bool(data.get("enabled", True)),
        "template": str(data.get("template") or "professional").strip()[:80] or "professional",
        "sections": _clean_sections(data.get("sections")) or [
            "Executive Summary", "Key Metrics", "Insights", "Goals",
        ],
        "dataset_id": data.get("dataset_id"),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "last_run_at": None,
        "last_status": "never",
        "last_report_id": None,
        "last_error": None,
        "run_count": 0,
        "runs": [],
    }
    items = _load_schedules(user_id)
    items.append(digest)
    _save_schedules(user_id, items)
    return digest


def update_digest(user_id: int, digest_id: str, patch: dict) -> dict | None:
    patch = patch or {}
    items = _load_schedules(user_id)
    for d in items:
        if str(d.get("id") or "") != str(digest_id):
            continue
        if "name" in patch and patch["name"] is not None:
            name = str(patch["name"]).strip()[:200]
            if name:
                d["name"] = name
        if "description" in patch and patch["description"] is not None:
            d["description"] = str(patch["description"]).strip()[:500]
        if "frequency" in patch and patch["frequency"] is not None:
            freq = str(patch["frequency"]).strip().lower()
            if freq in VALID_FREQUENCIES:
                d["frequency"] = freq
        if "enabled" in patch and patch["enabled"] is not None:
            d["enabled"] = bool(patch["enabled"])
        if "template" in patch and patch["template"] is not None:
            tpl = str(patch["template"]).strip()[:80]
            if tpl:
                d["template"] = tpl
        if "sections" in patch and patch["sections"] is not None:
            cleaned = _clean_sections(patch["sections"])
            if cleaned:
                d["sections"] = cleaned
        d["updated_at"] = _now_iso()
        _save_schedules(user_id, items)
        return d
    return None


def delete_digest(user_id: int, digest_id: str) -> bool:
    items = _load_schedules(user_id)
    kept = [d for d in items if str(d.get("id") or "") != str(digest_id)]
    if len(kept) < len(items):
        _save_schedules(user_id, kept)
        return True
    return False


# ---------------------------------------------------------------------------
# Run bookkeeping (failures record a failed run entry, never raise)
# ---------------------------------------------------------------------------

def _record_run(user_id: int, digest_id: str, status: str,
                report_id: str | None = None, error: str | None = None) -> dict | None:
    try:
        items = _load_schedules(user_id)
        for d in items:
            if str(d.get("id") or "") != str(digest_id):
                continue
            entry = {
                "at": _now_iso(),
                "status": status,
                "report_id": report_id,
                "error": (str(error)[:500] if error else None),
            }
            runs = list(d.get("runs") or [])
            runs.append(entry)
            d["runs"] = runs[-MAX_RUNS_KEPT:]
            d["last_run_at"] = entry["at"]
            d["last_status"] = status
            d["last_report_id"] = report_id
            d["last_error"] = entry["error"]
            d["run_count"] = int(d.get("run_count") or 0) + 1
            d["updated_at"] = _now_iso()
            _save_schedules(user_id, items)
            return d
    except Exception as exc:
        log.warning("digest: run bookkeeping failed (%s)", exc)
    return None


# ---------------------------------------------------------------------------
# Digest content: reuse existing engines, never invent numbers
# ---------------------------------------------------------------------------

def digest_to_definition(digest: dict) -> dict:
    """Report definition for a digest (persisted with the report for re-run)."""
    digest = digest or {}
    return {
        "name": digest.get("name") or "Digest",
        "description": digest.get("description") or (
            f"Scheduled {digest.get('frequency') or 'daily'} digest."
        ),
        "date_range": None,
        "sections": list(digest.get("sections") or []),
        "template": digest.get("template") or "professional",
        "digest_id": digest.get("id"),
        "digest_name": digest.get("name"),
        "digest_frequency": digest.get("frequency"),
        "scheduled": True,
    }


def _forecast_snapshot(user_id: int) -> dict | None:
    """Attach forecast-vs-actual context only where a saved forecast exists.

    Scans the user's known persistence spots for stored forecast payloads and
    returns counts + labels only. Returns None when nothing is stored (the
    digest then simply omits that section). Never invents numbers.
    """
    try:
        base = _users_dir() / f"user_{user_id}"
        candidates: list[dict] = []
        for pattern in ("forecasts.json", "predictions.json", "predictive.json"):
            for path in list(base.rglob(pattern))[:4]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                items = data if isinstance(data, list) else (
                    data.get("items") or data.get("forecasts") or []
                )
                if isinstance(items, list):
                    for it in items[:10]:
                        if isinstance(it, dict):
                            candidates.append({
                                "target": it.get("target") or it.get("metric") or "",
                                "points": len(it.get("points") or it.get("forecast") or []),
                                "created_at": it.get("created_at") or "",
                            })
        if candidates:
            return {"forecasts_available": len(candidates), "forecasts": candidates[:5]}
    except Exception as exc:
        log.debug("digest: forecast snapshot skipped (%s)", exc)
    return None


def generate_digest_now(user_id: int, digest_or_id: Any,
                        dataset_id: Optional[str] = None) -> dict:
    """Run one digest immediately and persist the result to report history.

    Returns the generated report dict, or {"ok": False, ...} after recording
    a failed run entry. Never raises.
    """
    try:
        digest = (digest_or_id if isinstance(digest_or_id, dict)
                  else get_digest(user_id, str(digest_or_id or "")))
        if not digest:
            return {"ok": False, "detail": "Digest not found.", "code": "NOT_FOUND"}

        from core.config import get_session
        from services import intelligence_service as intel

        session = get_session(user_id)
        did = dataset_id or digest.get("dataset_id") or intel._dataset_id(session or {})

        definition = digest_to_definition(digest)
        snapshot = _forecast_snapshot(user_id)
        if snapshot:
            # Informational only: generate_report ignores unknown keys, so
            # this never affects numbers — it documents availability.
            definition["forecast_snapshot"] = snapshot

        report = intel.generate_report(user_id, did, definition)
        if not isinstance(report, dict) or report.get("ok") is False or not report.get("id"):
            raise RuntimeError("Report generation did not produce a saved report.")
        # Tag the report back to its digest for the UI (best-effort; the
        # definition already carries digest_id for re-run).
        try:
            items = intel._load_user_items(user_id, "reports")
            for r in items:
                if str(r.get("id") or "") == str(report.get("id")):
                    r["digest_id"] = digest.get("id")
                    r["digest_name"] = digest.get("name")
                    break
            intel._save_user_items(user_id, "reports", items)
            report["digest_id"] = digest.get("id")
        except Exception:
            pass
        _record_run(user_id, digest.get("id"), "ok", report_id=report.get("id"))
        return report
    except Exception as exc:
        log.warning("digest: generation failed for user=%s (%s)", user_id, exc)
        try:
            did = digest_or_id.get("id") if isinstance(digest_or_id, dict) else digest_or_id
            _record_run(user_id, str(did or ""), "failed", error=str(exc))
        except Exception:
            pass
        return {"ok": False, "detail": "Digest run failed — recorded as a failed run.",
                "code": "DIGEST_FAILED"}


# ---------------------------------------------------------------------------
# Scheduler (dependency-free: threading + timer, graceful shutdown)
# ---------------------------------------------------------------------------

def _due(digest: dict, now_ts: float) -> bool:
    try:
        if not digest.get("enabled"):
            return False
        last = digest.get("last_run_at")
        if not last:
            return True  # newly created / never run -> run on next tick
        try:
            last_ts = datetime.fromisoformat(str(last)).timestamp()
        except Exception:
            return True
        freq = str(digest.get("frequency") or "daily").lower()
        interval = 7 * 24 * 3600 if freq == "weekly" else 24 * 3600
        return (now_ts - last_ts) >= interval
    except Exception:
        return False


def _iter_user_ids() -> list[int]:
    try:
        base = _users_dir()
        out = []
        for child in base.iterdir():
            if not child.is_dir() or not child.name.startswith("user_"):
                continue
            try:
                out.append(int(child.name.split("user_")[1]))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _tick() -> None:
    now_ts = time.time()
    for uid in _iter_user_ids():
        try:
            schedules = _load_schedules(uid)
        except Exception:
            continue
        for digest in schedules:
            try:
                if _due(digest, now_ts):
                    log.info("digest: scheduled run user=%s digest=%s freq=%s",
                             uid, digest.get("id"), digest.get("frequency"))
                    generate_digest_now(uid, digest)
            except Exception as exc:
                # Never crash the scheduler; the failure is already recorded
                # as a failed run entry inside generate_digest_now.
                log.warning("digest: tick error user=%s (%s)", uid, exc)


def _loop() -> None:
    log.info("digest: scheduler started (poll=%ss)", POLL_SECONDS)
    while not _stop_event.wait(POLL_SECONDS):
        try:
            _tick()
        except Exception as exc:
            log.warning("digest: scheduler tick failed (%s)", exc)
    log.info("digest: scheduler stopped")


def ensure_started() -> bool:
    """Start the background scheduler once (idempotent, lazy)."""
    global _thread, _started
    with _lock:
        if _started and _thread is not None and _thread.is_alive():
            return True
        _stop_event.clear()
        _thread = threading.Thread(target=_loop, name="aria-digests",
                                   daemon=True)
        try:
            _thread.start()
            _started = True
            log.info("digest: scheduler ensure_started ok")
            return True
        except Exception as exc:
            log.warning("digest: scheduler failed to start (%s)", exc)
            return False


def stop() -> None:
    """Graceful shutdown: signal the loop and join briefly."""
    global _started
    try:
        _stop_event.set()
        t = _thread
        if t is not None and t.is_alive():
            t.join(timeout=5)
    except Exception:
        pass
    finally:
        _started = False


atexit.register(stop)
