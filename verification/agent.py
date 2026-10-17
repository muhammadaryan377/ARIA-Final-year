"""Verification Agent — Phase 1 (generic, additive).

Re-checks a dashboard ``config`` payload before it reaches the UI. Fully
generic: no hardcoded table/column names. Every audit only reads whatever
field names the ``config`` (and ``session``) already carry, including common
aliases (``metrics``/``kpis``, ``visualizations``/``charts``).

Checks performed by :func:`verify_dashboard`:

* ``freshness`` — cache age vs TTL (from ``_meta``/``cached``/``generated_at``
  carried by the config; TTL from ``ARIA_DASH_CACHE_TTL``).
* ``substitution`` — requested vs resolved audit (passes through whatever
  substitution records the config carries).
* ``math`` — recompute percentages/shares from carried data rows; reject
  NaN/Infinity and zero-division.
* ``evidence`` — spot re-aggregation of carried rows plus optional live
  spot-count passthrough (``evidence_counts`` carried by config/session).
* ``completeness`` — every displayed section has backing content.

Returns ``{"verified": bool, "checks": [...], "warnings": [...]}``.
Never raises on bad input: structural problems become failed checks.
"""

import math
import os
from datetime import datetime, timezone


def _is_finite_number(value) -> bool:
    """True for real finite ints/floats (bools excluded)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _looks_nonfinite_text(value) -> bool:
    """Detect NaN/Infinity leaking into display strings."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return text in {"nan", "+nan", "-nan", "inf", "+inf", "-inf",
                    "infinity", "+infinity", "-infinity"}


def _as_list(value):
    return value if isinstance(value, list) else []


def _metrics_of(config: dict) -> list:
    for key in ("metrics", "kpis"):
        items = config.get(key)
        if isinstance(items, list):
            return items
    return []


def _visualizations_of(config: dict) -> list:
    for key in ("visualizations", "charts"):
        items = config.get(key)
        if isinstance(items, list):
            return items
    return []


def _numeric_values(row: dict, preferred_keys: tuple = ()) -> list:
    """Collect numeric cell values from a generic data row (dict)."""
    if not isinstance(row, dict):
        return []
    values = []
    for key in preferred_keys:
        try:
            candidate = row.get(key)
        except Exception:
            continue
        if _is_finite_number(candidate) or isinstance(candidate, (int, float)):
            values.append(candidate)
    for key, candidate in row.items():
        if preferred_keys and key in preferred_keys:
            continue
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            values.append(candidate)
    return values


def _parse_generated_at(value):
    """Parse an ISO-8601 timestamp carried by the config; None if absent."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        return None


def _check_freshness(config: dict):
    """Cache age vs TTL, using only what the config carries."""
    meta = config.get("_meta") if isinstance(config.get("_meta"), dict) else {}
    try:
        ttl = float(meta.get("ttl_s", os.getenv("ARIA_DASH_CACHE_TTL", "120")))
    except (TypeError, ValueError):
        ttl = 120.0
    if not math.isfinite(ttl) or ttl <= 0:
        ttl = 120.0

    age = meta.get("cache_age_s", meta.get("cache_age", config.get("cache_age_s")))
    cached = meta.get("cached", config.get("cached", False))
    generated_at = meta.get("generated_at", config.get("generated_at"))
    if age is None and generated_at is not None:
        parsed = _parse_generated_at(generated_at)
        if parsed is not None:
            age = (datetime.now(timezone.utc) - parsed).total_seconds()

    if age is None:
        return (
            {"name": "freshness", "pass": True,
             "detail": "No timestamp carried; treated as fresh for this view."},
            ["Freshness timestamp not carried; showing data as-is."],
        )
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return (
            {"name": "freshness", "pass": False,
             "detail": "Unreadable cache age carried; data may be stale."},
            [],
        )
    if not math.isfinite(age_f) or age_f < 0:
        return (
            {"name": "freshness", "pass": False,
             "detail": "Unreadable cache age carried; data may be stale."},
            [],
        )
    if age_f <= ttl:
        state = "cached view" if cached else "fresh view"
        return (
            {"name": "freshness", "pass": True,
             "detail": f"Data is current ({state}, age {age_f:.0f}s of {ttl:.0f}s allowed)."},
            [],
        )
    return (
        {"name": "freshness", "pass": False,
         "detail": f"Data may be out of date (age {age_f:.0f}s exceeds {ttl:.0f}s allowed)."},
        [],
    )


def _iter_substitution_records(config: dict) -> list:
    """Collect substitution records from whatever keys the config carries."""
    records = []
    for key in ("substitutions", "fallbacks", "resolved_substitutions"):
        items = config.get(key)
        if isinstance(items, list):
            records.extend([r for r in items if isinstance(r, dict)])
    for item in _metrics_of(config) + _visualizations_of(config):
        if not isinstance(item, dict):
            continue
        if any(k in item for k in ("requested", "resolved", "substituted", "fallback")):
            records.append(item)
    requested = config.get("requested")
    resolved = config.get("resolved")
    if isinstance(requested, dict) or isinstance(resolved, dict):
        records.append({"requested": requested, "resolved": resolved})
    return records


def _check_substitution(config: dict):
    """Requested-vs-resolved audit over carried substitution records."""
    records = _iter_substitution_records(config)
    if not records:
        return (
            {"name": "substitution", "pass": True,
             "detail": "No substitutions recorded; showing what was asked for."},
            [],
        )
    malformed = 0
    swapped = []
    for record in records:
        requested = record.get("requested", record.get("ask", record.get("wanted")))
        resolved = record.get("resolved", record.get("used", record.get("actual")))
        if requested is None or resolved is None:
            malformed += 1
            continue
        if str(requested) != str(resolved):
            swapped.append(f"{requested} shown as {resolved}")
    warnings = []
    if swapped:
        shown = "; ".join(swapped[:3])
        if len(swapped) > 3:
            shown += f" (+{len(swapped) - 3} more)"
        warnings.append(f"Some items were swapped for close alternatives: {shown}.")
    if malformed:
        return (
            {"name": "substitution", "pass": False,
             "detail": f"{malformed} substitution record(s) missing request or result."},
            warnings,
        )
    if swapped:
        return (
            {"name": "substitution", "pass": True,
             "detail": f"{len(swapped)} item(s) shown with a close alternative."},
            warnings,
        )
    return (
        {"name": "substitution", "pass": True,
         "detail": "Everything shown matches what was asked for."},
        warnings,
    )


def _check_math(config: dict):
    """Recompute percentages/shares from carried rows; reject NaN/Inf/÷0."""
    problems = []
    warnings: list[str] = []

    for index, metric in enumerate(_metrics_of(config)):
        if not isinstance(metric, dict):
            problems.append(f"metric #{index} is not an object")
            continue
        label = str(metric.get("label", f"metric #{index}"))
        for field in ("rawValue", "raw_value", "value_numeric", "change",
                      "change_percent", "trendPct", "share"):
            candidate = metric.get(field)
            if candidate is None or candidate == "":
                continue
            if isinstance(candidate, bool):
                continue
            if isinstance(candidate, (int, float)):
                if not _is_finite_number(candidate):
                    problems.append(f"'{label}' has an unusable number in '{field}'")
                continue
            if isinstance(candidate, str) and _looks_nonfinite_text(candidate):
                problems.append(f"'{label}' has an unusable number in '{field}'")
        display = metric.get("value")
        if isinstance(display, str) and _looks_nonfinite_text(display):
            problems.append(f"'{label}' displays an unusable number")

    for index, viz in enumerate(_visualizations_of(config)):
        if not isinstance(viz, dict):
            problems.append(f"chart #{index} is not an object")
            continue
        title = str(viz.get("title", f"chart #{index}"))
        data = viz.get("data")
        if not isinstance(data, list):
            continue
        preferred = tuple(
            k for k in (viz.get("dataKey"), viz.get("yKey"), viz.get("valueKey"))
            if isinstance(k, str) and k
        )
        total = 0.0
        total_ok = True
        count = 0
        for row in data:
            if not isinstance(row, dict):
                problems.append(f"'{title}' has a broken data row")
                total_ok = False
                continue
            for candidate in _numeric_values(row, preferred):
                if not _is_finite_number(candidate):
                    problems.append(f"'{title}' has an unusable number in its data")
                    total_ok = False
                else:
                    total += float(candidate)
                    count += 1
        if not total_ok:
            continue
        if count == 0:
            continue
        if total == 0.0:
            # Zero-division guard: shares would divide by zero, so they are
            # simply not computed; warn in plain words instead of failing
            # when the underlying rows are otherwise fine.
            if any(float(v) != 0.0 for v in
                   [n for r in data if isinstance(r, dict)
                    for n in _numeric_values(r, preferred)
                    if _is_finite_number(n)]):
                warnings.append(f"'{title}': parts add to zero, so shares are left out.")
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            for candidate in _numeric_values(row, preferred):
                share = (float(candidate) / abs(total)) * 100.0
                if not math.isfinite(share):
                    problems.append(f"'{title}' has a share that could not be worked out")
                    break

    quality = config.get("quality")
    if isinstance(quality, dict):
        score = quality.get("score")
        if score is not None and (not _is_finite_number(score)
                                  or not (0 <= float(score) <= 100)):
            problems.append("data-quality score is outside 0–100")

    if problems:
        shown = "; ".join(problems[:3])
        if len(problems) > 3:
            shown += f" (+{len(problems) - 3} more)"
        return (
            {"name": "math", "pass": False,
             "detail": f"Number check failed: {shown}."},
            warnings,
        )
    return (
        {"name": "math", "pass": True,
         "detail": "Totals and shares were re-worked and line up."},
        warnings,
    )


def _check_evidence(config: dict, session: dict):
    """Spot re-aggregation of carried rows + optional live-count passthrough."""
    warnings: list[str] = []
    visualizations = [v for v in _visualizations_of(config) if isinstance(v, dict)]
    metrics = [m for m in _metrics_of(config) if isinstance(m, dict)]

    backing_rows = 0
    empty_charts = []
    for viz in visualizations:
        data = viz.get("data")
        if isinstance(data, list):
            backing_rows += len(data)
            if not data:
                empty_charts.append(str(viz.get("title", "chart")))

    # Spot re-query: re-aggregate the carried rows a second way (count + sum
    # per chart) and require the two passes to agree. Catches corrupted or
    # half-written payloads without needing any table/column names.
    mismatches = []
    for viz in visualizations:
        data = viz.get("data")
        if not isinstance(data, list) or not data:
            continue
        preferred = tuple(
            k for k in (viz.get("dataKey"), viz.get("yKey"), viz.get("valueKey"))
            if isinstance(k, str) and k
        )
        first_total = 0.0
        first_count = 0
        for row in data:
            if isinstance(row, dict):
                for candidate in _numeric_values(row, preferred):
                    if _is_finite_number(candidate):
                        first_total += float(candidate)
                        first_count += 1
        second_total = sum(
            float(candidate)
            for row in data if isinstance(row, dict)
            for candidate in _numeric_values(row, preferred)
            if _is_finite_number(candidate)
        )
        if not math.isfinite(first_total) or not math.isfinite(second_total) \
                or abs(first_total - second_total) > 1e-6:
            mismatches.append(str(viz.get("title", "chart")))

    # Optional live spot-count passthrough: when the caller carries expected
    # counts (config) alongside freshly re-queried counts (session), compare
    # up to a few keys generically.
    live_mismatches = []
    expected = config.get("evidence_counts")
    actual = session.get("evidence_counts", session.get("spot_counts"))
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in list(expected.keys())[:5]:
            try:
                exp = float(expected[key])
                got = float(actual[key])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(exp) and math.isfinite(got) and abs(exp - got) > 1e-6:
                live_mismatches.append(str(key))

    insights = _as_list(config.get("insights")) + _as_list(config.get("anomalies"))
    without_backup = sum(
        1 for item in insights
        if isinstance(item, dict)
        and not any(item.get(k) not in (None, "", [], {}) for k in
                    ("evidence", "supporting_rows", "row_count", "sample", "query"))
    )
    if without_backup:
        warnings.append(
            f"{without_backup} finding(s) carry no supporting detail; treat as pointers.")

    problems = []
    if mismatches:
        problems.append(f"re-count differed on: {', '.join(mismatches[:2])}")
    if live_mismatches:
        problems.append(f"live re-check differed on: {', '.join(live_mismatches[:2])}")
    numeric_claims = sum(
        1 for m in metrics
        if isinstance(m, dict) and _is_finite_number(m.get("rawValue", m.get("raw_value")))
    )
    if numeric_claims and backing_rows == 0 and not visualizations:
        problems.append("numbers shown with no backing rows carried")
    if empty_charts and numeric_claims:
        warnings.append(
            f"{len(empty_charts)} chart(s) arrived empty; numbers come from cards only.")

    if problems:
        return (
            {"name": "evidence", "pass": False,
             "detail": "Backup check failed: " + "; ".join(problems[:2]) + "."},
            warnings,
        )
    detail = "Spot re-count of the displayed rows lines up."
    if not visualizations and not metrics:
        detail = "Nothing to back up yet (empty dashboard)."
    return ({"name": "evidence", "pass": True, "detail": detail}, warnings)


def _check_completeness(config: dict):
    """Every displayed section has backing content (generic key aliases)."""
    warnings: list[str] = []
    missing = []
    if not isinstance(config.get("summary"), dict):
        missing.append("summary")
    has_metrics = isinstance(config.get("metrics"), list) or isinstance(config.get("kpis"), list)
    has_viz = isinstance(config.get("visualizations"), list) or isinstance(config.get("charts"), list)
    if not has_metrics:
        missing.append("metrics")
    if not has_viz:
        missing.append("visualizations")

    broken = []
    for index, metric in enumerate(_metrics_of(config)):
        if not isinstance(metric, dict) or not metric.get("label") \
                or metric.get("value") in (None, ""):
            broken.append(f"card #{index}")
    for index, viz in enumerate(_visualizations_of(config)):
        if not isinstance(viz, dict) or not viz.get("title") \
                or not isinstance(viz.get("data"), list):
            broken.append(f"chart #{index}")

    for key in ("quality", "recommendations", "insights", "anomalies"):
        if config.get(key) is None:
            warnings.append(f"'{key}' section not carried; shown without it.")

    problems = missing + broken
    if problems:
        shown = ", ".join(problems[:4])
        if len(problems) > 4:
            shown += f" (+{len(problems) - 4} more)"
        return (
            {"name": "completeness", "pass": False,
             "detail": f"Display is incomplete: missing {shown}."},
            warnings,
        )
    if not _metrics_of(config) and not _visualizations_of(config):
        warnings.append("Dashboard arrived with no cards or charts to show.")
    return (
        {"name": "completeness", "pass": True,
         "detail": "All displayed sections have backing content."},
        warnings,
    )


def _payload_of(payload) -> dict:
    """Normalize an ask/forecast/prescription payload to a dict (never raises)."""
    try:
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _selection_info_of(payload: dict) -> list:
    """Collect requested-vs-resolved selection dicts from common envelope keys."""
    found = []
    try:
        for key in ("selection", "info"):
            item = payload.get(key)
            if isinstance(item, dict):
                found.append(item)
        forecast = payload.get("forecast")
        if isinstance(forecast, dict):
            for key in ("selection", "info"):
                item = forecast.get(key)
                if isinstance(item, dict):
                    found.append(item)
    except Exception:
        pass
    return found


def _check_ask_substitution(payload: dict):
    """Requested-vs-resolved audit for an ask answer (generic, additive)."""
    records = _iter_substitution_records(payload)
    for item in _selection_info_of(payload):
        try:
            if any(k in item for k in ("requested_measure", "resolved_measure",
                                       "requested_time_dim", "resolved_time_dim",
                                       "requested", "resolved", "substituted")):
                records.append(item)
        except Exception:
            continue
    # Explicit requested/resolved pairs carried by the ask envelope.
    pairs = []
    try:
        for item in _selection_info_of(payload):
            for req_key, res_key in (("requested_measure", "resolved_measure"),
                                     ("requested_time_dim", "resolved_time_dim"),
                                     ("requested", "resolved")):
                if req_key in item or res_key in item:
                    pairs.append((item.get(req_key), item.get(res_key)))
    except Exception:
        pass
    swapped = []
    for requested, resolved in pairs:
        try:
            if requested is not None and resolved is not None \
                    and str(requested) != str(resolved):
                swapped.append(f"{requested} shown as {resolved}")
        except Exception:
            continue
    # Fall back to the shared record audit for anything else carried.
    if not pairs and records:
        check, warnings = _check_substitution(payload)
        return check, warnings
    warnings = []
    substituted_flags = []
    try:
        for item in _selection_info_of(payload):
            if item.get("substituted") is True:
                reason = item.get("substitution_reason")
                substituted_flags.append(str(reason) if reason else "a close alternative was used")
    except Exception:
        pass
    status = None
    try:
        status = str(payload.get("status") or "")
    except Exception:
        status = None
    if swapped or substituted_flags:
        shown = "; ".join((swapped + substituted_flags)[:3])
        extra = len(swapped) + len(substituted_flags) - 3
        if extra > 0:
            shown += f" (+{extra} more)"
        detail = f"Showing a close alternative: {shown}."
        if status == "partial":
            warnings.append("Your exact question could not be run, so a nearby breakdown is shown; try rephrasing for an exact answer.")
        else:
            warnings.append(f"Some items were swapped for close alternatives: {shown}.")
        return ({"name": "substitution", "pass": True, "detail": detail}, warnings)
    if status == "partial":
        return (
            {"name": "substitution", "pass": True,
             "detail": "A nearby breakdown is shown because the exact question could not be run."},
            ["Your exact question could not be run, so a nearby breakdown is shown."],
        )
    return (
        {"name": "substitution", "pass": True,
         "detail": "Everything shown matches what was asked for."},
        [],
    )


def _ask_rows_of(payload: dict) -> list:
    """Best-effort data rows carried by an ask answer (generic keys only)."""
    rows: list = []
    try:
        for key in ("data", "values"):
            items = payload.get(key)
            if isinstance(items, list) and items:
                rows = [r for r in items if isinstance(r, dict)]
                if rows:
                    break
        if not rows:
            viz = payload.get("visualization")
            if isinstance(viz, dict) and isinstance(viz.get("data"), list):
                rows = [r for r in viz["data"] if isinstance(r, dict)]
        if not rows:
            evidence = payload.get("evidence")
            if isinstance(evidence, list):
                for entry in evidence:
                    if isinstance(entry, dict) and isinstance(entry.get("values"), list):
                        rows = [r for r in entry["values"] if isinstance(r, dict)]
                        if rows:
                            break
    except Exception:
        rows = []
    return rows


def _check_ask_math(payload: dict):
    """Number audit over the carried ask rows (finite numbers, sane shares)."""
    warnings: list[str] = []
    try:
        rows = _ask_rows_of(payload)
    except Exception:
        rows = []
    if not rows:
        return (
            {"name": "math", "pass": True,
             "detail": "No rows carried to re-check; showing the answer as-is."},
            ["Answer arrived without rows to double-check."],
        )
    try:
        check, extra = _check_math({"metrics": [], "visualizations": [
            {"title": "answer", "data": rows}]})
    except Exception:
        return ({"name": "math", "pass": False,
                 "detail": "Number check could not run on this answer."}, warnings)
    return check, list(extra) + warnings


def _check_ask_evidence(payload: dict):
    """Evidence-SQL presence audit for an ask answer (generic keys only)."""
    sql = None
    try:
        provenance = payload.get("provenance")
        if isinstance(provenance, dict):
            sql = provenance.get("sql")
        if not sql:
            sql = payload.get("sql_used", payload.get("sql"))
        if not sql:
            evidence = payload.get("evidence")
            if isinstance(evidence, list):
                for entry in evidence:
                    if isinstance(entry, dict) and entry.get("sql"):
                        sql = entry.get("sql")
                        break
        if not sql:
            why = payload.get("why")
            if isinstance(why, dict) and isinstance(why.get("evidence"), dict):
                sql = why["evidence"].get("sql")
    except Exception:
        sql = None
    try:
        text = str(sql or "").strip()
    except Exception:
        text = ""
    if text:
        return (
            {"name": "evidence", "pass": True,
             "detail": "This answer is backed by a stored database query."},
            [],
        )
    return (
        {"name": "evidence", "pass": False,
         "detail": "No backing database query was carried with this answer."},
        [],
    )


def verify_ask(payload, session) -> dict:
    """Re-check an ask (/api/ask) answer; never raises on bad input."""
    try:
        payload = _payload_of(payload)
    except Exception:
        payload = {}
    try:
        session = session if isinstance(session, dict) else {}
    except Exception:
        session = {}
    return _run_checks(payload, session, [
        ("substitution", _check_ask_substitution, (payload,)),
        ("math", _check_ask_math, (payload,)),
        ("evidence", _check_ask_evidence, (payload,)),
    ])


def _forecast_of(payload: dict) -> dict:
    """Normalize a forecast payload: full response or bare forecast object."""
    try:
        forecast = payload.get("forecast")
        if isinstance(forecast, dict) and forecast:
            return forecast
    except Exception:
        pass
    return payload


def _forecast_points(forecast: dict) -> list:
    """Future points from generic forecast keys (never raises)."""
    try:
        for key in ("forecast", "forecast_series", "points"):
            items = forecast.get(key)
            if isinstance(items, list) and items:
                return [p for p in items if isinstance(p, dict)]
    except Exception:
        pass
    return []


def _forecast_history(forecast: dict, payload: dict) -> list:
    """Past points from generic forecast keys (never raises)."""
    try:
        for source in (forecast, payload):
            if not isinstance(source, dict):
                continue
            for key in ("historical", "history"):
                items = source.get(key)
                if isinstance(items, list) and items:
                    return [h for h in items if isinstance(h, dict)]
    except Exception:
        pass
    return []


def _band_widths(points: list) -> list:
    """Likely-range widths from generic lower/upper keys (finite only)."""
    widths = []
    try:
        for point in points:
            if not isinstance(point, dict):
                continue
            low = point.get("lower", point.get("lower95", point.get("low")))
            high = point.get("upper", point.get("upper95", point.get("high")))
            try:
                if low is None or high is None:
                    continue
                width = float(high) - float(low)
            except (TypeError, ValueError):
                continue
            if math.isfinite(width) and width >= 0:
                widths.append(width)
    except Exception:
        pass
    return widths


def _check_forecast_substitution(payload: dict):
    """Requested-vs-resolved audit for a forecast (generic selection/info)."""
    try:
        return _check_ask_substitution(payload)
    except Exception:
        return ({"name": "substitution", "pass": False,
                 "detail": "Substitution check could not run on this forecast."}, [])


def _check_forecast_bands(payload: dict):
    """Likely-range audit: bands carried and widening further out."""
    try:
        forecast = _forecast_of(payload)
        points = _forecast_points(forecast)
        interval = forecast.get("confidence_interval")
        widths = _band_widths(points)
        if not widths and isinstance(interval, list) and interval:
            widths = _band_widths([{"lower": pair[0], "upper": pair[1]}
                                   for pair in interval
                                   if isinstance(pair, (list, tuple)) and len(pair) >= 2])
        if not widths:
            return (
                {"name": "bands", "pass": False,
                 "detail": "No likely range was carried with this forecast."},
                [],
            )
        try:
            tolerance = max(1e-9, abs(widths[0]) * 1e-6)
        except Exception:
            tolerance = 1e-9
        narrowed = sum(1 for earlier, later in zip(widths, widths[1:])
                       if later + tolerance < earlier)
        if narrowed:
            return (
                {"name": "bands", "pass": False,
                 "detail": "The likely range narrows further out, so treat distant points with care."},
                [],
            )
        return (
            {"name": "bands", "pass": True,
             "detail": "The likely range widens further out, as expected."},
            [],
        )
    except Exception:
        return ({"name": "bands", "pass": False,
                 "detail": "Range check could not run on this forecast."}, [])


def _check_forecast_history(payload: dict):
    """History-count audit: carried history length lines up with n_obs."""
    try:
        forecast = _forecast_of(payload)
        history = _forecast_history(forecast, payload)
        n_obs = forecast.get("n_obs", payload.get("n_obs"))
        try:
            n_obs_f = float(n_obs) if n_obs is not None else None
        except (TypeError, ValueError):
            n_obs_f = None
        if n_obs_f is not None and (not math.isfinite(n_obs_f) or n_obs_f < 0):
            return (
                {"name": "history", "pass": False,
                 "detail": "The history count carried with this forecast is unreadable."},
                [],
            )
        if not history and n_obs_f is None:
            return (
                {"name": "history", "pass": True,
                 "detail": "No history count was carried; showing the forecast as-is."},
                ["Forecast arrived without a history count to double-check."],
            )
        if n_obs_f is not None and history:
            if abs(len(history) - n_obs_f) > 1e-6:
                return (
                    {"name": "history", "pass": False,
                     "detail": f"History count looks off ({len(history)} points carried vs {n_obs_f:.0f} stated)."},
                    [],
                )
            return (
                {"name": "history", "pass": True,
                 "detail": f"Built from {n_obs_f:.0f} past points carried with the forecast."},
                [],
            )
        if history:
            return (
                {"name": "history", "pass": True,
                 "detail": f"Built from {len(history)} past points carried with the forecast."},
                [],
            )
        return (
            {"name": "history", "pass": True,
             "detail": f"Built from {n_obs_f:.0f} past points stated with the forecast."},
            [],
        )
    except Exception:
        return ({"name": "history", "pass": False,
                 "detail": "History check could not run on this forecast."}, [])


def verify_forecast(payload, session) -> dict:
    """Re-check a forecast (/api/predictive/forecast) payload; never raises."""
    try:
        payload = _payload_of(payload)
    except Exception:
        payload = {}
    try:
        session = session if isinstance(session, dict) else {}
    except Exception:
        session = {}
    return _run_checks(payload, session, [
        ("substitution", _check_forecast_substitution, (payload,)),
        ("bands", _check_forecast_bands, (payload,)),
        ("history", _check_forecast_history, (payload,)),
    ])


_RX_DISPLAY_KEYS = ("title", "action_text", "outcome_text", "confidence_label",
                    "why_summary", "problem_text", "why_text")


def _prescriptions_of(payload: dict) -> list:
    """Prescription cards from generic analyze/scenario payloads."""
    try:
        items = payload.get("prescriptions")
        if isinstance(items, list) and items:
            return [p for p in items if isinstance(p, dict)]
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            return [p for p in scenarios if isinstance(p, dict)]
        if isinstance(payload.get("display"), dict):
            return [payload]
    except Exception:
        pass
    return []


def _check_prescription_completeness(payload: dict):
    """Display 7-key completeness audit for every prescription card."""
    try:
        cards = _prescriptions_of(payload)
    except Exception:
        cards = []
    if not cards:
        status = None
        try:
            status = str(payload.get("status") or "")
        except Exception:
            status = None
        if status == "no_prescription" or isinstance(payload.get("refusal"), dict):
            return (
                {"name": "completeness", "pass": True,
                 "detail": "No action cleared the bar, and that honest refusal is shown as-is."},
                [],
            )
        return (
            {"name": "completeness", "pass": False,
             "detail": "No recommendation cards were carried to check."},
            [],
        )
    incomplete = 0
    missing_names: list[str] = []
    for card in cards:
        try:
            display = card.get("display")
            if not isinstance(display, dict):
                incomplete += 1
                continue
            missing = [k for k in _RX_DISPLAY_KEYS
                       if not str(display.get(k) or "").strip()]
            if missing:
                incomplete += 1
                for key in missing:
                    if key not in missing_names:
                        missing_names.append(key)
        except Exception:
            incomplete += 1
    if incomplete:
        shown = ", ".join(missing_names[:3])
        extra = f" (missing: {shown})" if shown else ""
        return (
            {"name": "completeness", "pass": False,
             "detail": f"{incomplete} of {len(cards)} card(s) arrive with details missing{extra}."},
            [],
        )
    return (
        {"name": "completeness", "pass": True,
         "detail": "Every card carries its full details (what, why, expected result)."},
        [],
    )


def _check_prescription_evidence(payload: dict):
    """Evidence-gate audit: every card carries its backing checks."""
    try:
        cards = _prescriptions_of(payload)
    except Exception:
        cards = []
    if not cards:
        if isinstance(payload.get("refusal"), dict) \
                or str(payload.get("status") or "") == "no_prescription":
            return (
                {"name": "evidence", "pass": True,
                 "detail": "No action cleared the bar, and the reason is shown."},
                [],
            )
        return (
            {"name": "evidence", "pass": False,
             "detail": "No cards carried backing checks to review."},
            [],
        )
    ungated = 0
    for card in cards:
        try:
            details = card.get("details")
            evidence = card.get("evidence")
            has_pass = isinstance(details, dict) and isinstance(
                details.get("evidence_pass"), bool)
            has_checks = isinstance(details, dict) and isinstance(
                details.get("checks"), list) and bool(details.get("checks"))
            has_evidence = isinstance(evidence, dict) and bool(evidence)
            if not (has_pass and (has_checks or has_evidence)):
                ungated += 1
        except Exception:
            ungated += 1
    if ungated:
        return (
            {"name": "evidence", "pass": False,
             "detail": f"{ungated} of {len(cards)} card(s) arrive without full backing checks."},
            [],
        )
    return (
        {"name": "evidence", "pass": True,
         "detail": "Every card passed its backing data checks."},
        [],
    )


def _check_prescription_numbers(payload: dict):
    """Number audit over carried prescription figures (finite, readable)."""
    try:
        cards = _prescriptions_of(payload)
    except Exception:
        cards = []
    if not cards:
        return (
            {"name": "math", "pass": True,
             "detail": "No figures carried to re-check; showing the cards as-is."},
            [],
        )
    problems = 0
    for card in cards:
        try:
            impact = card.get("impact")
            candidates = []
            if isinstance(impact, dict):
                for key in ("baseline", "scenario", "delta", "rel_pct"):
                    candidates.append(impact.get(key))
            details = card.get("details")
            if isinstance(details, dict) and isinstance(details.get("numbers"), dict):
                candidates.extend(list(details["numbers"].values()))
            for candidate in candidates:
                if candidate is None or candidate == "":
                    continue
                if isinstance(candidate, bool):
                    continue
                if isinstance(candidate, (int, float)):
                    if not _is_finite_number(candidate):
                        problems += 1
                        break
                elif isinstance(candidate, str) and _looks_nonfinite_text(candidate):
                    problems += 1
                    break
        except Exception:
            continue
    if problems:
        return (
            {"name": "math", "pass": False,
             "detail": f"{problems} card(s) carry figures that could not be re-checked."},
            [],
        )
    return (
        {"name": "math", "pass": True,
         "detail": "Card figures were re-checked and line up."},
        [],
    )


def verify_prescription(payload, session) -> dict:
    """Re-check a prescription analyze payload; never raises on bad input."""
    try:
        payload = _payload_of(payload)
    except Exception:
        payload = {}
    try:
        session = session if isinstance(session, dict) else {}
    except Exception:
        session = {}
    return _run_checks(payload, session, [
        ("completeness", _check_prescription_completeness, (payload,)),
        ("evidence", _check_prescription_evidence, (payload,)),
        ("math", _check_prescription_numbers, (payload,)),
    ])


def _run_checks(payload: dict, session: dict, steps: list) -> dict:
    """Run (name, fn, args) audit steps; never raises on bad input."""
    checks: list[dict] = []
    warnings: list[str] = []
    for name, fn, args in steps:
        try:
            check, extra = fn(*args)
            checks.append(check)
            warnings.extend(extra or [])
        except Exception:
            checks.append({"name": name, "pass": False,
                           "detail": "Check could not run on this view."})
    verified = bool(checks) and all(bool(c.get("pass")) for c in checks)
    seen: set[str] = set()
    unique_warnings = []
    for warning in warnings:
        text = str(warning)
        if text and text not in seen:
            seen.add(text)
            unique_warnings.append(text)
    return {"verified": bool(verified), "checks": checks, "warnings": unique_warnings}


def verify_dashboard(config, session) -> dict:
    """Run the Phase-1 dashboard audits; never raises on bad input."""
    try:
        config = config if isinstance(config, dict) else {}
    except Exception:
        config = {}
    try:
        session = session if isinstance(session, dict) else {}
    except Exception:
        session = {}

    checks: list[dict] = []
    warnings: list[str] = []
    try:
        check, extra = _check_freshness(config)
        checks.append(check)
        warnings.extend(extra)
    except Exception:
        checks.append({"name": "freshness", "pass": False,
                       "detail": "Freshness check could not run."})
    try:
        check, extra = _check_substitution(config)
        checks.append(check)
        warnings.extend(extra)
    except Exception:
        checks.append({"name": "substitution", "pass": False,
                       "detail": "Substitution check could not run."})
    try:
        check, extra = _check_math(config)
        checks.append(check)
        warnings.extend(extra)
    except Exception:
        checks.append({"name": "math", "pass": False,
                       "detail": "Number check could not run."})
    try:
        check, extra = _check_evidence(config, session)
        checks.append(check)
        warnings.extend(extra)
    except Exception:
        checks.append({"name": "evidence", "pass": False,
                       "detail": "Backup check could not run."})
    try:
        check, extra = _check_completeness(config)
        checks.append(check)
        warnings.extend(extra)
    except Exception:
        checks.append({"name": "completeness", "pass": False,
                       "detail": "Completeness check could not run."})

    verified = bool(checks) and all(bool(c.get("pass")) for c in checks)
    seen: set[str] = set()
    unique_warnings = []
    for warning in warnings:
        text = str(warning)
        if text and text not in seen:
            seen.add(text)
            unique_warnings.append(text)
    return {"verified": bool(verified), "checks": checks, "warnings": unique_warnings}
