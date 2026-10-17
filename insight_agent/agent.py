"""
ARIA Insight Agent
------------------
Role: Signal Detection

Analyses processed_data.json (produced by the Goal Agent) for KPIs, trends,
and anomalies. Generates chart/dashboard specs, produces 3-4 ranked hypotheses
plus a human-readable business story (Mistral 7B locally, or Llama via Groq).

LLM backend is selected by the user at startup:
    local - Ollama (private, offline, slower)
    cloud - Groq API (fast, but data leaves the machine)

Output: insights.json

Usage:
    python -m insight_agent processed_data.json
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llm_provider import LLMProvider, create_provider
from core.validation import clean_llm_text, validate_chart_choice
from .chart_catalog import compatible_chart_options
from prediction import PredictionAgent
from prescription_agent import PrescriptionAgent

logging.basicConfig(level=logging.INFO)


def sanitize_json(obj):
    """Recursively replace NaN/Inf floats with None.

    Statistic computations can produce NaN/Inf (e.g. 0/0 on a single-row
    dataset) which break strict JSON serializers like Starlette's
    JSONResponse (allow_nan=False). Applied to every agent payload before it
    is written or returned.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


CHART_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "static" / "chart_templates.json"

# ── n.txt items 9 + P2: controlled chart vocabulary ──────────────────────
# The backend may only emit these chart types. The frontend renders the spec
# object only — the LLM is never allowed to return arbitrary frontend code,
# templates, or render configs. Every proposal (including `chart_suggestions`)
# passes the deterministic guardrail in core.validation before use.
CHART_SPEC_VOCAB = ("line", "bar", "area", "scatter", "pie", "donut", "histogram", "heatmap")

# Internal template ids -> controlled vocabulary. Anything not listed falls
# back to "bar" (or "table-equivalent" kpi handling) so the frontend never
# receives an unsupported type.
_TEMPLATE_TO_VOCAB = {
    "line": "line", "multi_line": "line", "step_line": "line",
    "sparkline": "line", "slope": "line", "control_chart": "line",
    "area": "area", "stacked_area": "area", "density": "area", "ecdf": "area",
    "bar": "bar", "bar_horizontal": "bar", "bar_stacked": "bar",
    "bar_grouped": "bar", "bar_normalized": "bar", "bar_diverging": "bar",
    "lollipop": "bar", "pareto": "bar", "waterfall": "bar", "combo": "bar",
    "radar": "bar", "bullet": "bar", "funnel": "bar",
    "scatter": "scatter", "bubble": "scatter", "hexbin": "scatter", "contour": "scatter",
    "pie": "pie", "polar_area": "pie",
    "doughnut": "donut", "donut": "donut", "gauge": "donut",
    "histogram": "histogram", "box_plot": "histogram", "violin": "histogram",
    "heatmap": "heatmap", "calendar_heatmap": "heatmap",
}


def vocab_type_for_template(tid: str) -> str:
    """Map an internal template id to the controlled frontend vocabulary."""
    return _TEMPLATE_TO_VOCAB.get((tid or "").lower(), "bar")


def validate_spec_data(rows, limit: int = 500):
    """Validate + sanitize chart data before it leaves the backend.

    Drops non-dict rows, caps length, converts NaN/Inf to None. Returns a
    plain JSON-safe list. Never raises — returns [] when unusable so the
    frontend can show an insufficient-data state.
    """
    if not isinstance(rows, (list, tuple)):
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        clean = {}
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
        if len(out) >= limit:
            break
    return out


class InsightAgent:
    """Base Insight Agent (signal detection, z-score anomalies)."""

    VERSION = "2.0"
    ANOMALY_METHOD = "zscore"

    def __init__(self, provider=None):
        if provider is None:
            logging.getLogger(__name__).warning(
                "InsightAgent created with provider=None – failing closed"
            )
            raise ValueError(
                "InsightAgent requires an explicit provider. "
                "Pass the LLMProvider object from require_data() or session['provider']."
            )
        self.llm = provider
        self.prediction = PredictionAgent()
        self.prescription = PrescriptionAgent(provider=self.llm)
        self._templates = None
        self.last_story_status = "SUCCESS"

    # ------------------------------------------------------------------
    # Chart templates (code/config kept in a JSON file, not in code)
    # ------------------------------------------------------------------

    def _load_chart_templates(self):
        """Load the reusable chart config templates from chart_templates.json."""
        if self._templates is None:
            if CHART_TEMPLATES_PATH.exists():
                try:
                    self._templates = json.loads(CHART_TEMPLATES_PATH.read_text(encoding="utf-8")).get("templates", {})
                except Exception:
                    self._templates = {}
            else:
                self._templates = {}
        return self._templates

    def _pick_template(self, templates, shape):
        """Pick the chart template whose rules best fit the data shape."""
        best_id, best_priority = None, -1
        for tid, tpl in templates.items():
            for rule in tpl.get("shapes", []):
                if rule.get("shape") == shape and rule.get("priority", 0) > best_priority:
                    best_priority = rule.get("priority", 0)
                    best_id = tid
        return best_id

    def _labels_datasets(self, xcol, ycol, records):
        return {
            "labels": [r.get(xcol) for r in records],
            "datasets": [{"label": ycol, "data": [r.get(ycol) for r in records]}],
        }

    # ------------------------------------------------------------------
    # Chart guardrail (data-driven rules decide; LLM never overrides)
    # ------------------------------------------------------------------

    def _guard_chart_type(self, default_tid, shape, suggestions, templates):
        """Return the template id to use for one data shape.

        The data-driven `default_tid` is authoritative. `suggestions` is kept
        for signature compatibility but is ignored by design: chart selection
        is fully deterministic so every provider renders the same charts.
        """
        if default_tid in templates:
            return default_tid
        return default_tid

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_processed_data(self, processed_data_path="processed_data.json"):
        with open(processed_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Column classification
    # ------------------------------------------------------------------

    def _classify_columns(self, df):
        df = df.copy()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # Numeric-looking object columns (e.g. Decimal stringified by JSON, or
        # currencies with separators) must be treated as measures, not categories.
        for col in df.columns:
            if col in numeric:
                continue
            sample = df[col].dropna()
            if not len(sample):
                continue
            if sample.map(lambda v: isinstance(v, (datetime, pd.Timestamp))).all():
                continue
            coerced = pd.to_numeric(sample, errors="coerce")
            if int(coerced.notna().sum()) / len(sample) >= 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                numeric.append(col)

        category = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = []
        for col in df.columns:
            if col in numeric:
                continue
            try:
                pd.to_datetime(df[col], errors="raise")
                if col in category:
                    category.remove(col)
                datetime_cols.append(col)
            except Exception:
                pass
        return numeric, category, datetime_cols

    # ------------------------------------------------------------------
    # KPI analysis
    # ------------------------------------------------------------------

    def compute_kpis(self, df, numeric_cols):
        """Compute descriptive KPIs for every numeric column."""
        kpis = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.dropna().empty:
                continue
            count = int(series.count())
            kpis.append({
                "column": col,
                "sum": round(float(series.sum()), 2),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "std": (round(float(series.std()), 2) if count > 1 else None),
                "count": count,
                "null_count": int(series.isna().sum()),
            })
        return kpis

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trends(self, df, datetime_cols, numeric_cols):
        """Simple monotonic/up/down trend flag on time-series numeric columns."""
        trends = []
        for dt_col in datetime_cols[:1]:
            ts = pd.to_datetime(df[dt_col], errors="coerce")
            temp = df.copy()
            temp["_date"] = ts
            for num_col in numeric_cols:
                series = temp.groupby("_date")[num_col].sum().sort_index()
                if len(series) < 3:
                    continue
                values = series.astype(float).values
                diffs = np.diff(values)
                if np.all(diffs >= 0):
                    direction = "increasing"
                elif np.all(diffs <= 0):
                    direction = "decreasing"
                else:
                    direction = "mixed"
                pct_change = round(
                    float((values[-1] - values[0]) / values[0] * 100), 2
                ) if values[0] else 0.0
                trends.append({
                    "dimension": dt_col,
                    "direction": direction,
                    "pct_change": pct_change,
                    "columns": [num_col],
                })
        return trends

    # ------------------------------------------------------------------
    # Anomaly detection (z-score based)
    # ------------------------------------------------------------------

    def detect_anomalies(self, df, numeric_cols):
        """Flag numeric rows whose z-score exceeds the threshold as anomalies.
        Identifier-like columns (ids/keys/codes) are never flagged: they are
        synthetic keys, not measures. Output is capped at the strongest 20
        (by |z|) with a total count, so noisy columns cannot flood insights.
        """
        anomalies = []
        for col in numeric_cols:
            low = col.lower().strip()
            if low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid"):
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            if len(valid) < 4:
                continue
            mean = valid.mean()
            std = valid.std()
            if std == 0 or pd.isna(std):
                continue
            z = (series - mean) / std
            for idx, value in series.items():
                if abs(z[idx]) > 2.0:
                    anomalies.append({
                        "column": col,
                        "row": int(idx),
                        "value": round(float(value), 2),
                        "z_score": round(float(z[idx]), 2),
                        "note": "beyond 2 standard deviations",
                    })
        total = len(anomalies)
        if total > 20:
            anomalies = sorted(anomalies, key=lambda a: abs(a["z_score"]), reverse=True)[:20]
            for a in anomalies:
                a["total_anomaly_count"] = total
        return anomalies

    # ------------------------------------------------------------------
    # Chart / dashboard spec generation
    # ------------------------------------------------------------------

    def _histogram(self, series, bins=10):
        """Bin a numeric column into a histogram. Returns (labels, counts)."""
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return [], []
        lo, hi = float(s.min()), float(s.max())
        if not (hi > lo):
            return [], []
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(s.values, bins=edges)
        labels = [f"{edges[i]:,.1f}-{edges[i + 1]:,.1f}" for i in range(bins)]
        return labels, [int(c) for c in counts]

    @staticmethod
    def _deep_merge(base, extra):
        if not isinstance(base, dict) or not isinstance(extra, dict):
            return extra if extra is not None else base
        out = dict(base)
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = InsightAgent._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _spec_data_rows(spec_type, labels, datasets, x, y):
        """Flatten labels+datasets into frontend-ready row dicts for the spec.

        Single-series bar/line/area/histogram -> [{x, y}]. Multi-series ->
        [{x, s1, s2, ...}]. Scatter/bubble point lists pass through unchanged
        (validated later). Returns [] when the shape is unusable.
        """
        try:
            datasets = datasets or []
            if spec_type == "scatter":
                for d in datasets:
                    pts = (d or {}).get("data") if isinstance(d, dict) else None
                    if isinstance(pts, list) and pts and isinstance(pts[0], dict):
                        return pts[:500]
                return []
            if spec_type == "heatmap":
                # Heatmap rows: {x, y, value}; fall back to category matrix.
                rows = []
                for d in datasets:
                    if not isinstance(d, dict):
                        continue
                    for lbl, val in zip(labels or [], (d.get("data") or [])):
                        rows.append({"x": lbl, "y": d.get("label"), "value": val})
                return rows[:500]
            labels = labels or []
            if not labels or not datasets:
                return []
            if len(datasets) == 1:
                vals = (datasets[0] or {}).get("data") or []
                return [{str(x): lbl, str(y): val} for lbl, val in zip(labels, vals)][:500]
            rows = []
            for i, lbl in enumerate(labels):
                row = {str(x): lbl}
                for d in datasets:
                    if isinstance(d, dict) and d.get("label"):
                        vals = d.get("data") or []
                        row[str(d["label"])] = vals[i] if i < len(vals) else None
                rows.append(row)
            return rows[:500]
        except Exception:
            return []

    def build_dashboard_spec(self, df, numeric_cols, category_cols, datetime_cols,
                             chart_suggestions=None):
        """Build chart specs by picking the chart template best suited to the data.

        Selection is data-intelligent: it considers category cardinality, the
        number of numeric series, the presence of a time axis, label lengths and
        distribution shape, then picks the best template from
        static/chart_templates.json. An LLM may propose chart types via
        `chart_suggestions`, but every proposal passes the deterministic guardrail
        (core.validation.validate_chart_choice) so the final type always fits the
        data shape and is identical on both providers.

        Each spec embeds a ready-to-render Chart.js `config`, so the frontend can
        add new chart types without touching client code.
        """
        templates = self._load_chart_templates()
        specs = []
        palette = ["#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#f472b6",
                   "#2dd4bf", "#facc15", "#60a5fa", "#fb7185"]

        def pick(shape, default):
            tid = self._pick_template(templates, shape) or default
            return self._guard_chart_type(tid, shape, chart_suggestions, templates)

        def pick_type(shape, preferred, default):
            """Pick a specific preferred type if it fits the shape, else the
            best-fit template. Lets a context force e.g. radar or bubble while
            the deterministic guardrail still decides validity."""
            if preferred:
                final_type, _ = validate_chart_choice(preferred, shape)
                if final_type == preferred and preferred in templates:
                    return preferred
            return pick(shape, default)

        def make(tid, title, x, y, labels, datasets, extra_options=None):
            cfg = templates.get(tid, {}).get("config", {})
            options = self._deep_merge(dict(cfg.get("options", {})), extra_options or {})
            options.setdefault("responsive", True)
            options.setdefault("maintainAspectRatio", False)
            chart_type = cfg.get("type", "bar")
            # Controlled chart spec (n.txt-9): frontend renders ONLY this
            # {type,x,y,series,title,interactive,data} object. `type` is drawn
            # from CHART_SPEC_VOCAB; raw template/config internals stay as
            # legacy compat and must never carry LLM-supplied code.
            spec_type = vocab_type_for_template(tid)
            series = [d.get("label") for d in (datasets or []) if isinstance(d, dict) and d.get("label")]
            spec_data = validate_spec_data(
                self._spec_data_rows(spec_type, labels, datasets, x, y)
            )
            return {
                "chart_type": tid,
                "title": title,
                "x": x,
                "y": y,
                "labels": labels,
                "datasets": datasets,
                "template": cfg,
                "config": {
                    "type": chart_type,
                    "data": {"labels": labels, "datasets": datasets},
                    "options": options,
                },
                # ── controlled spec: the only contract the frontend renders ──
                "spec": {
                    "type": spec_type,
                    "x": x,
                    "y": y,
                    "series": series[:8],
                    "title": title,
                    "interactive": tid != "kpi_card",
                    "data": spec_data,
                },
                "type": spec_type,
                "series": series[:8],
                "interactive": tid != "kpi_card",
                "data": spec_data,
            }

        def ds_style(label, data, kind, color):
            if kind in ("line", "area"):
                return {
                    "label": label, "data": data, "fill": kind == "area", "tension": 0.3,
                    "borderColor": color, "backgroundColor": color + "33",
                    "borderWidth": 2, "pointRadius": 3,
                }
            return {"label": label, "data": data, "backgroundColor": color,
                    "borderColor": color, "borderWidth": 1}

        def cat_summary(dim, col, topn):
            grouped = df.groupby(dim)[col].sum().sort_values(ascending=False).head(topn)
            return grouped.reset_index().to_dict(orient="records")

        # Identifier-like columns are keys, not measures: charts must plot real
        # business measures, never row IDs or foreign keys, unless those are the
        # only numeric columns available.
        def _is_id_like(name):
            try:
                from core.identifiers import is_id_like as _canon
                return bool(_canon(name))
            except Exception:
                low = str(name or "").lower().strip()
                return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid")

        measures_all = [c for c in numeric_cols if not _is_id_like(c)] or numeric_cols

        # ---- 1) Category -> numeric ----
        if category_cols and measures_all:
            dim = category_cols[0]
            try:
                card = int(df[dim].nunique())
            except Exception:
                card = 0
            primary = measures_all[0]

            recs = cat_summary(dim, primary, 10)
            if recs:
                labels = [r[dim] for r in recs]
                data = [float(r[primary]) for r in recs]
                longest = max((len(str(l)) for l in labels), default=0)
                if longest > 14 or card > 12:
                    tid = pick_type("category_numeric", "bar_horizontal", "bar")
                else:
                    tid = pick_type("category_numeric", "bar", "bar")
                specs.append(make(tid, f"{primary} by {dim} (top {len(labels)})", dim, primary,
                                  labels, [ds_style(primary, data, "bar", palette[0])]))

            # Share chart is only meaningful for a handful of categories.
            if 1 < card <= 6:
                top6 = cat_summary(dim, primary, 6)
                if top6:
                    tid = pick("category_low_card", "pie")
                    labels = [r[dim] for r in top6]
                    data = [float(r[primary]) for r in top6]
                    specs.append(make(
                        tid, f"{primary} share by {dim}", dim, primary, labels,
                        [{"label": primary, "data": data, "backgroundColor": palette,
                          "borderColor": "#1e293b", "borderWidth": 1}],
                    ))

            # Multiple numeric series over the same category -> stacked bars.
            if len(measures_all) >= 2:
                top8 = cat_summary(dim, primary, 8)
                if top8:
                    labels = [r[dim] for r in top8]
                    cols = measures_all[:4]
                    datasets = []
                    grouped = {c: df.groupby(dim)[c].sum() for c in cols}
                    for i, c in enumerate(cols):
                        ordered = [float(grouped[c].get(l, 0)) if l in grouped[c].index else 0.0
                                   for l in labels]
                        datasets.append(ds_style(c, ordered, "bar", palette[i % len(palette)]))
                    tid = pick("category_numeric_multi", "bar_stacked")
                    specs.append(make(tid, f"{' + '.join(cols)} by {dim} (top {len(labels)})",
                                      dim, "+".join(cols), labels, datasets))

            # Radar compares several numeric dimensions across few categories.
            if 2 <= card <= 8 and len(measures_all) >= 2:
                cats = df[dim].value_counts().head(8).index.tolist()
                cols = measures_all[:4]
                grouped = {c: df.groupby(dim)[c].sum() for c in cols}
                datasets = [
                    ds_style(c, [float(grouped[c].get(ct, 0)) for ct in cats], "line",
                             palette[i % len(palette)])
                    for i, c in enumerate(cols)
                ]
                tid = pick_type("category_numeric_multi", "radar", "radar")
                specs.append(make(tid, f"radar comparison across {dim}", dim, "+".join(cols),
                                  cats, datasets))

        # ---- 2) Datetime -> numeric (time series) ----
        if datetime_cols and measures_all:
            ts = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            temp = df.copy()
            temp["_date"] = ts
            temp = temp.dropna(subset=["_date"]).sort_values("_date")
            if not temp.empty:
                if len(measures_all) == 1:
                    col = measures_all[0]
                    line = temp.groupby("_date")[col].sum().reset_index()
                    labels = [str(d) for d in line["_date"]]
                    data = [float(v) for v in line[col]]
                    tid = pick("datetime_numeric", "line")
                    specs.append(make(tid, f"{col} over time", datetime_cols[0], col, labels,
                                      [ds_style(col, data, "line", palette[0])]))
                else:
                    cols = measures_all[:3]
                    grouped = {c: temp.groupby("_date")[c].sum() for c in cols}
                    labels = [str(d) for d in grouped[cols[0]].index]
                    datasets = []
                    for i, c in enumerate(cols):
                        ordered = [float(grouped[c].get(d, 0)) for d in grouped[cols[0]].index]
                        color = palette[i % len(palette)]
                        if i == 0:
                            datasets.append({**ds_style(c, ordered, "line", color),
                                             "yAxisID": "y", "type": "line"})
                        else:
                            datasets.append({**ds_style(c, ordered, "bar", color),
                                             "yAxisID": "y1"})
                    tid = pick("datetime_numeric_multi", "combo")
                    specs.append(make(tid, f"{' + '.join(cols)} over time", datetime_cols[0],
                                      "+".join(cols), labels, datasets))

        # ---- 3) Numeric columns only (no category, no time) ----
        if not category_cols and not datetime_cols and measures_all:
            measures = measures_all
            if len(measures) == 1:
                col = measures[0]
                labels, counts = self._histogram(df[col], bins=10)
                if labels:
                    tid = pick("distribution", "histogram")
                    specs.append(make(tid, f"distribution of {col}", col, "count", labels,
                                      [ds_style("count", counts, "bar", palette[0])]))
            else:
                x, y = measures[0], measures[1]
                sample = df[[x, y]].dropna().head(80)
                if not sample.empty:
                    tid = pick("numeric_numeric", "scatter")
                    pts = [{"x": float(a), "y": float(b)} for a, b in zip(sample[x], sample[y])]
                    specs.append(make(tid, f"{y} vs {x}", x, y, [],
                                      [{"label": y, "data": pts, "backgroundColor": palette[0]}]))
                if len(measures) >= 3:
                    z = measures[2]
                    s3 = df[[x, y, z]].dropna().head(60)
                    if not s3.empty and s3[z].abs().max() > 0:
                        tid2 = pick("numeric_numeric_numeric", "bubble")
                        scale = float(s3[z].abs().max()) or 1.0
                        bpts = [
                            {"x": float(a), "y": float(b),
                             "r": max(2.0, min(25.0, abs(float(c)) / scale * 25))}
                            for a, b, c in zip(s3[x], s3[y], s3[z])
                        ]
                        specs.append(make(
                            tid2, f"{z} as bubble size ({y} vs {x})", x, y, [],
                            [{"label": y, "data": bpts, "backgroundColor": palette[0] + "66",
                              "borderColor": palette[0], "borderWidth": 1}],
                        ))

        # ---- 4) Category only -> counts ----
        if category_cols and not numeric_cols and not datetime_cols:
            dim = category_cols[0]
            counts = df[dim].value_counts().head(10)
            if not counts.empty:
                labels = [str(i) for i in counts.index]
                data = [int(c) for c in counts.values]
                longest = max((len(str(l)) for l in labels), default=0)
                if longest > 14 or len(counts) > 12:
                    tid = pick_type("category_counts", "bar_horizontal", "bar")
                else:
                    tid = pick_type("category_counts", "bar", "bar")
                specs.append(make(tid, f"records by {dim} (top {len(labels)})", dim, "count",
                                  labels, [ds_style("count", data, "bar", palette[0])]))

        # ---- 5) KPI cards ----
        # Defensive: when schema classification misses numeric columns
        # (e.g. new database format or Decimal objects that select_dtypes
        # skips), detect numeric values directly from the data so KPI cards
        # never vanish for any future database source.
        effective_numeric = list(numeric_cols)
        if not effective_numeric:
            for col in df.columns:
                if col in numeric_cols:
                    continue
                sample = df[col].dropna()
                if len(sample) >= 1:
                    try:
                        coerced = pd.to_numeric(sample, errors="coerce")
                        if int(coerced.notna().sum()) / len(sample) >= 0.7:
                            effective_numeric.append(col)
                    except Exception:
                        continue
                # Also treat pure float/int numpy dtypes that may have been
                # missed by the initial classification for any reason.
                if col not in effective_numeric and pd.api.types.is_numeric_dtype(df[col]):
                    effective_numeric.append(col)

        if effective_numeric:
            kpi_rows = validate_spec_data(self.compute_kpis(df, effective_numeric))
            specs.append({
                "chart_type": "kpi_card",
                "title": "Key performance indicators",
                "x": None, "y": None, "labels": [], "datasets": [],
                "template": templates.get("kpi_card", {}).get("config", {}),
                "config": {"type": "kpi_card", "data": {"labels": [], "datasets": []}, "options": {}},
                "data": kpi_rows,
                # Controlled spec for KPI cards (non-interactive by design).
                "spec": {
                    "type": "bar", "x": None, "y": None, "series": [],
                    "title": "Key performance indicators",
                    "interactive": False, "data": kpi_rows,
                },
                "type": "bar", "series": [], "interactive": False,
            })

        return specs

    # ------------------------------------------------------------------
    # Hypothesis generation + ranking (statistical)
    # ------------------------------------------------------------------

    def generate_hypotheses(self, df, numeric_cols, category_cols, datetime_cols, kpis, anomalies):
        """360-degree analysis -> hypotheses in plain natural English.

        Every hypothesis is a sentence a business person can read directly.
        No technical jargon (no z-scores, std dev, grouped-sum terminology) in
        the output text or its rationale. Ranking is by `score`; `confidence`
        reflects how strongly the numbers support the claim.
        """
        hyps = []
        score = 0.0

        def add(text, basis, confidence, pts):
            nonlocal score
            hyps.append({
                "hypothesis": text,
                "basis": basis,
                "confidence": confidence,
                "score": float(pts),
            })
            score += float(pts)

        # Identifier-like columns are keys, not measures: hypotheses must talk
        # about real business measures, never about row IDs or foreign keys.
        def _is_id_like(name):
            try:
                from core.identifiers import is_id_like as _canon
                return bool(_canon(name))
            except Exception:
                low = str(name or "").lower().strip()
                return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid")

        measures = [c for c in numeric_cols if not _is_id_like(c)] or numeric_cols

        def _human_name(name):
            try:
                from core.semantic import humanize as _hz
                return _hz(str(name or ""))
            except Exception:
                return str(name or "").replace("_", " ").strip() or str(name or "")

        def _human_scale(v):
            try:
                n = float(v)
            except (TypeError, ValueError):
                return str(v)
            a = abs(n)
            if a >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if a >= 1_000:
                return f"{n / 1_000:.1f}K"
            if a >= 100:
                return f"{round(n):,}"
            if a >= 1:
                return f"{round(n, 1)}"
            return f"{round(n, 2)}"

        # 1) Across every numeric measure x every category dimension: who
        #    dominates, and is the value concentrated in a few groups?
        for col in measures:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            total = float(series.sum())
            if total == 0:
                continue
            for dim in category_cols[:2]:
                try:
                    grp = df.groupby(dim)[col].sum().dropna().sort_values(ascending=False)
                except Exception:
                    continue
                if grp.empty:
                    continue
                top = grp.index[0]
                top_share = float(grp.iloc[0]) / total * 100.0
                if top_share >= 20:
                    confidence = "high" if top_share >= 60 else ("medium" if top_share >= 35 else "low")
                    add(
                        f"'{top}' leads {_human_name(dim).lower()} — about {top_share:.0f}% "
                        f"of all {_human_name(col).lower()}. It is the standout to watch.",
                        f"Compared every {_human_name(dim).lower()} group by its share of "
                        f"{_human_name(col).lower()}.",
                        confidence, 10 + top_share / 10,
                    )
                top3_share = float(grp.head(3).sum()) / total * 100.0
                if top3_share >= 60 and len(grp) > 3:
                    add(
                        f"The top three {_human_name(dim).lower()} groups together hold "
                        f"{top3_share:.0f}% of {_human_name(col).lower()} — so it is "
                        f"concentrated in a few names.",
                        f"Measured how much {_human_name(col).lower()} sits in the three "
                        f"largest {_human_name(dim).lower()} groups.",
                        "medium", 7,
                    )

        # 2) Compare the numeric measures with each other: which is the most
        #    volatile and which is the steadiest?
        if len(measures) >= 2:
            cv = []
            for col in measures:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) and s.mean() != 0:
                    cv.append((col, float(s.std() / abs(s.mean()))))
            if cv:
                vol_col, vol = max(cv, key=lambda x: x[1])
                if vol > 0.5:
                    add(
                        f"{_human_name(vol_col)} moves around the most — up and down "
                        f"against its own average. It deserves the closest watch.",
                        "Compared how much each measure moves against its own average.",
                        "medium", 6,
                    )
                stable_col, stab = min(cv, key=lambda x: x[1])
                if stab < 0.15 and len(cv) > 1:
                    add(
                        f"{_human_name(stable_col)} stays close to its average throughout — "
                        f"the steadiest measure here.",
                        "Compared how much each measure moves against its own average.",
                        "low", 4,
                    )

        # 3) Outliers / anomalies described in plain language.
        by_col = {}
        for a in anomalies:
            if _is_id_like(a.get("column", "")):
                continue
            by_col.setdefault(a["column"], []).append(a)
        for col, items in list(by_col.items())[:2]:
            n = len(items)
            add(
                f"{_human_name(col)} has {n} {'value' if n == 1 else 'values'} well "
                f"outside the usual range — worth a closer look.",
                "Flagged the rows sitting far outside the usual spread.",
                "medium", 6 + n,
            )

        # 4) Time dimension present.
        if datetime_cols:
            add(
                f"Things move over time here — recent shifts in the key numbers "
                f"may come from that.",
                "Found dated records and checked how the numbers move along them.",
                "low", 4,
            )

        # 5) Anchor on the descriptive statistics.
        if kpis:
            top = max(kpis, key=lambda k: k.get("mean") or 0)
            add(
                f"On average, {_human_name(top['column']).lower()} is about "
                f"{_human_scale(top['mean'])}, usually between "
                f"{_human_scale(top.get('min'))} and {_human_scale(top.get('max'))}.",
                "Summarised the typical level of each measure.",
                "high", 5,
            )

        hyps.sort(key=lambda h: h["score"], reverse=True)
        return hyps[:5], score

    # ------------------------------------------------------------------
    # Business story via Mistral 7B
    # ------------------------------------------------------------------

    def generate_business_story(self, processed_data, kpis, trends, anomalies, hypotheses,
                                predictions=None):
        """Return a short, plain-English business story for the findings.

        A deterministic narrative engine always produces the story from the
        actual numbers (identical on cloud and local, and never blocks). When
        the LLM is available it is asked to polish/expand that narrative; any
        failure, timeout or unusable output falls back to the deterministic
        version so the story is never missing or stuck.

        Evidence gate FIRST (core/evidence.py): rows < 10 -> honest
        deterministic insufficient state is returned WITHOUT calling the LLM.
        The LLM must never compensate for missing evidence.
        """
        try:
            from core.evidence import insufficient_narrative as _insuf
            _rows = processed_data.get("row_count", len(processed_data.get("data", []) or []))
            try:
                _rows = int(_rows)
            except (TypeError, ValueError):
                _rows = 0
            if _rows < 10:
                _cols = list(processed_data.get("columns", []) or [])
                if not _cols and isinstance(processed_data.get("data"), list) \
                        and processed_data["data"]:
                    _first = processed_data["data"][0]
                    if isinstance(_first, dict):
                        _cols = list(_first.keys())
                _fb = _insuf(_rows, "narrative",
                             dataset_name=processed_data.get("user_goal"),
                             columns=_cols)
                self.last_story_status = "INSUFFICIENT_EVIDENCE"
                return _fb["text"]
        except Exception:
            pass
        base = self._narrative_story(processed_data, kpis, trends, anomalies, hypotheses, predictions)
        self.last_story_status = "SUCCESS"

        context = {
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "row_count": processed_data.get("row_count", 0),
            "kpis": kpis[:6],
            "trends": trends[:3],
            "anomalies": anomalies[:5],
            "hypotheses": hypotheses,
        }
        if predictions and predictions.get("valid") and predictions.get("points"):
            context["prediction"] = {
                "column": predictions["column"],
                "horizon": predictions["horizon"],
                "method": predictions["method"],
                "accuracy_pct": predictions["accuracy_pct"],
                "projected_pct_change": predictions["projected_pct_change"],
                "points": predictions["points"][:4],
            }
        claims = self._build_story_claims(processed_data, kpis, trends, anomalies,
                                          hypotheses, predictions)

        prompt = f"""
You are the Insight Agent of a BI system. Below is a draft narrative written from
the data, followed by the raw findings. Rewrite it as a polished, human-readable
business story for a non-technical manager.

Draft narrative:
{base}

Raw findings:
{json.dumps(context, indent=2, default=str)}

STRICT EVIDENCE PAYLOAD — the story may ONLY use these computed facts.
Do not invent numbers, segments, entities, or periods outside this list:
{json.dumps(claims, indent=2, default=str)}

Rules:
- Keep every claim grounded in the numbers given; do not invent facts.
- Every numeric value must come from the evidence payload (formatting/rounding only).
- Every named segment/entity/period must appear in the evidence payload.
- Direction words (increased/decreased) must match the sign of the computed change.
- Start with the headline finding, then supporting points, then any outlook.
- Plain language, under 250 words, no markdown, no JSON, no lists of jargon.
"""
        try:
            polished = self.llm.chat("story", messages=[{"role": "user", "content": prompt}],
                                     temperature=0.4, num_predict=400, timeout=45)
            if not (polished or "").strip():
                self.last_story_status = "LLM_EMPTY"
                logging.warning("Story LLM returned empty; using deterministic narrative.")
                return base
            polished = (polished or "").strip()
            text, reason = clean_llm_text(polished)
            if not text:
                self.last_story_status = "INVALID_LLM_RESPONSE"
                logging.warning("Story LLM output rejected (%s); using deterministic narrative.",
                                reason or "empty after cleaning")
                return base
            if not (40 <= len(text) <= 1200):
                self.last_story_status = "INVALID_LLM_RESPONSE"
                logging.warning("Story LLM output rejected (bad length); using deterministic narrative.")
                return base
            ok, fail_reason = self._validate_polished_story(
                text, processed_data, kpis, trends, anomalies, hypotheses, predictions)
            if ok:
                return self._plain_english_story(text)
            self.last_story_status = "INVALID_LLM_RESPONSE"
            logging.warning("Story LLM output rejected (%s); using deterministic narrative.",
                            fail_reason or reason or "ungrounded numbers")
        except Exception as exc:
            self.last_story_status = "LLM_UNAVAILABLE"
            logging.warning(f"Storytelling LLM failed: {exc}. Using deterministic narrative.")
        return base

    def _build_story_claims(self, processed_data, kpis, trends, anomalies,
                              hypotheses=None, predictions=None):
        """Structured claims list from computed evidence (no LLM involved).

        Each claim carries metric, value, comparison_value, direction,
        pct_change, period/comparison_period, segment/entity so the polish
        prompt is constrained and the validator can check numbers (on
        underlying values, formatting only at presentation), entities,
        periods, and direction sign.
        """
        import re as _re
        claims = []
        for k in kpis or []:
            col = k.get("column")
            if not col:
                continue
            claims.append({
                "metric": col, "value": k.get("mean"),
                "comparison_value": None,
                "direction": None, "pct_change": None,
                "period": None, "comparison_period": None,
                "segment": None, "entity": None,
                "detail": {f: k.get(f) for f in
                           ("sum", "mean", "median", "min", "max", "std", "count")
                           if k.get(f) is not None},
            })
        for t in trends or []:
            cols = t.get("columns") or ([t["column"]] if t.get("column") else [])
            pct = t.get("pct_change")
            try:
                pct_f = float(str(pct).replace("%", "")) if pct is not None else None
            except (TypeError, ValueError):
                pct_f = None
            direction = t.get("direction")
            if not direction and pct_f is not None:
                direction = "increasing" if pct_f > 0 else ("decreasing" if pct_f < 0 else "mixed")
            claims.append({
                "metric": cols[0] if cols else t.get("dimension"),
                "value": None, "comparison_value": None,
                "direction": direction, "pct_change": pct_f,
                "period": t.get("dimension"), "comparison_period": "previous period",
                "segment": None, "entity": None,
            })
        for a in anomalies or []:
            claims.append({
                "metric": a.get("column"), "value": a.get("value"),
                "comparison_value": None,
                "direction": None, "pct_change": None,
                "period": None, "comparison_period": None,
                "segment": None, "entity": None,
            })
        for h in hypotheses or []:
            txt = (h or {}).get("hypothesis", "") if isinstance(h, dict) else str(h)
            ents = set(_re.findall(r"'([^']{1,60})'", txt))
            for e in ents:
                claims.append({
                    "metric": None, "value": None, "comparison_value": None,
                    "direction": None, "pct_change": None,
                    "period": None, "comparison_period": None,
                    "segment": e, "entity": e,
                })
        if predictions and predictions.get("valid"):
            claims.append({
                "metric": predictions.get("column"), "value": predictions.get("last_actual"),
                "comparison_value": None,
                "direction": None, "pct_change": predictions.get("projected_pct_change"),
                "period": f"next {predictions.get('horizon')} {predictions.get('period') or 'periods'}",
                "comparison_period": "recent period",
                "segment": None, "entity": None,
            })
        return claims

    def _story_numbers_grounded(self, text, processed_data, kpis, trends, anomalies,
                                hypotheses=None, predictions=None) -> bool:
        """Back-compat wrapper: numeric heuristic now delegates to structured validation."""
        ok, _ = self._validate_polished_story(
            text, processed_data, kpis, trends, anomalies, hypotheses, predictions)
        return ok

    def _validate_polished_story(self, text, processed_data, kpis, trends, anomalies,
                                 hypotheses=None, predictions=None):
        """Structured-claims validation: numbers, entities, periods, direction.

        Returns (ok, reason). Numbers are compared on underlying values with
        rounding-awareness (not string equality); entities/periods must exist
        in evidence; direction words must match the sign of computed change.
        """
        import re as _re
        import json as _json
        known: list = []
        try:
            known.append(float((processed_data or {}).get("row_count") or 0))
        except (TypeError, ValueError):
            pass
        for k in kpis or []:
            for f in ("sum", "mean", "median", "min", "max", "std", "count"):
                try:
                    if k.get(f) is not None:
                        known.append(float(k[f]))
                except (TypeError, ValueError):
                    continue
        for t in trends or []:
            try:
                if t.get("pct_change") is not None:
                    known.append(float(str(t["pct_change"]).replace("%", "")))
            except (TypeError, ValueError):
                continue
        for a in anomalies or []:
            try:
                if a.get("value") is not None:
                    known.append(float(a["value"]))
            except (TypeError, ValueError):
                continue
        if predictions and predictions.get("valid"):
            for f in ("projected_pct_change", "accuracy_pct", "last_actual"):
                try:
                    if predictions.get(f) is not None:
                        known.append(float(str(predictions[f]).replace("%", "")))
                except (TypeError, ValueError):
                    continue
            for p in predictions.get("points") or []:
                for f in ("forecast", "value", "actual", "fitted"):
                    try:
                        if isinstance(p, dict) and p.get(f) is not None:
                            known.append(float(p[f]))
                    except (TypeError, ValueError):
                        continue
        # Hypothesis prose is computed evidence: numbers inside it (e.g. "40%")
        # are grounded by construction.
        for h in hypotheses or []:
            txt = (h or {}).get("hypothesis", "") if isinstance(h, dict) else str(h)
            txt += " " + (((h or {}).get("basis", "")) if isinstance(h, dict) else "")
            for m in _re.finditer(r"-?\d[\d,]*\.?\d*", txt):
                try:
                    known.append(float(m.group(0).replace(",", "")))
                except ValueError:
                    continue
        known = [v for v in known if v == v]  # drop NaN
        cols = {str(k.get("column", "")).lower() for k in kpis or [] if k.get("column")}
        cols |= {str(c or "").lower() for t in trends or [] for c in (t.get("columns") or [])}
        cols |= {str(t.get("column", "")).lower() for t in trends or [] if t.get("column")}
        cols |= {str(a.get("column", "")).lower() for a in anomalies or [] if a.get("column")}
        if predictions and predictions.get("column"):
            cols.add(str(predictions["column"]).lower())
        evidence_text = _json.dumps({
            "kpis": kpis or [], "trends": trends or [], "anomalies": anomalies or [],
            "hypotheses": hypotheses or [], "predictions": predictions or {},
            "row_count": (processed_data or {}).get("row_count"),
        }, default=str).lower()

        def _close(a, b, suffix="") -> bool:
            if b == 0:
                return abs(a) < 1.0
            # Rounding-aware presentation: 1.84M for 1840000, 42.8K for 42800.
            if suffix in ("m", "k"):
                scale = 1_000_000 if suffix == "m" else 1_000
                try:
                    if abs(a - round(b / scale, 1)) <= max(0.15, abs(b / scale) * 0.05):
                        return True
                except Exception:
                    pass
                return False
            return abs(a - b) <= max(0.6, abs(b) * 0.05)

        for m in _re.finditer(r"-?\d[\d,]*\.?\d*\s*[kKmM%]?", text):
            token = m.group(0)
            suffix_m = _re.search(r"([kKmM%])\s*$", token)
            suffix = (suffix_m.group(1).lower() if suffix_m else "")
            raw = _re.sub(r"[,%\s kKmM]+$", "", token).replace(",", "")
            if not raw or raw in ("-", ".", "-."):
                continue
            try:
                num = float(raw)
            except ValueError:
                continue
            if suffix == "%":
                # Percent values must match a computed pct_change.
                pct_known = []
                for t in trends or []:
                    try:
                        if t.get("pct_change") is not None:
                            pct_known.append(float(str(t["pct_change"]).replace("%", "")))
                    except (TypeError, ValueError):
                        continue
                if predictions and predictions.get("projected_pct_change") is not None:
                    try:
                        pct_known.append(float(str(predictions["projected_pct_change"]).replace("%", "")))
                    except (TypeError, ValueError):
                        pass
                for h in hypotheses or []:
                    txt = (h or {}).get("hypothesis", "") if isinstance(h, dict) else str(h)
                    for m in _re.finditer(r"(-?\d[\d,]*\.?\d*)\s*%", txt):
                        try:
                            pct_known.append(float(m.group(1).replace(",", "")))
                        except ValueError:
                            continue
                if not pct_known:
                    continue  # no computed change to contradict; skip
                if any(abs(num - p) <= max(0.6, abs(p) * 0.05) for p in pct_known):
                    continue
                return False, f"unmapped percent {token.strip()}"
            if abs(num) < 10 and not suffix:
                continue  # counts / ranks / small prose numbers
            if 1900 <= abs(num) <= 2100 and num.is_integer() and not suffix:
                # Year-like periods must exist in evidence; otherwise invented.
                if str(int(num)) not in evidence_text:
                    return False, f"unmapped period {token.strip()}"
                continue
            if any(_close(num, k, suffix) for k in known):
                continue
            return False, f"unmapped numeric {token.strip()}"
        for tok in set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+", text)):
            if tok.lower() not in cols:
                return False, f"unknown segment {tok}"
        # Quoted segment/entity names must exist in evidence.
        for q in set(_re.findall(r"'([^']{1,60})'|\"([^\"]{1,60})\"", text)):
            name = (q[0] or q[1] or "").strip()
            if not name or len(name) < 2:
                continue
            if _re.fullmatch(r"-?\d[\d,.\s%kKmM]*", name):
                continue
            low = name.lower()
            if low in {"previous period", "recent period", "over time", "next period",
                       "next periods", "usual range", "typical range"}:
                continue
            if low not in evidence_text:
                return False, f"unknown entity {name[:40]}"
        # Month-name periods must exist in evidence.
        months = ("january", "february", "march", "april", "may", "june", "july",
                  "august", "september", "october", "november", "december",
                  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                  "q1", "q2", "q3", "q4", "quarter")
        for m in set(_re.findall(r"[A-Za-z]{3,9}", text.lower())):
            if m in months and m not in evidence_text:
                return False, f"unmapped period {m}"
        # Direction words must match the sign of computed change.
        pct_signs = []
        for t in trends or []:
            try:
                if t.get("pct_change") is not None:
                    pct_signs.append(float(str(t["pct_change"]).replace("%", "")))
            except (TypeError, ValueError):
                continue
        if predictions and predictions.get("projected_pct_change") is not None:
            try:
                pct_signs.append(float(str(predictions["projected_pct_change"]).replace("%", "")))
            except (TypeError, ValueError):
                pass
        if pct_signs:
            up_words = ("increas", "grew", "growth", "rose", "rising", " up ", "upward")
            down_words = ("decreas", "declin", "fell", "falling", "dropped", " down ", "downward")
            low_text = f" {text.lower()} "
            says_up = any(w in low_text for w in up_words)
            says_down = any(w in low_text for w in down_words)
            all_up = all(v > 0 for v in pct_signs)
            all_down = all(v < 0 for v in pct_signs)
            if says_up and not says_down and all_down:
                return False, "direction mismatch (says increased, computed decrease)"
            if says_down and not says_up and all_up:
                return False, "direction mismatch (says decreased, computed increase)"
        return True, ""

    def _plain_english_story(self, text):
        """Strip residual technical jargon an LLM polish may have reintroduced,
        so the story stays readable for a non-technical manager."""
        import re as _re
        subs = [
            (r"\bR\^2\b|\bR²\b|\bR-squared\b", "pattern strength"),
            (r"\bRMSE\b", "typical forecast error"),
            (r"\bMAE\b", "average error"),
            (r"\bp-?value\b", "reliability"),
            (r"\bcoefficient\b", "relationship"),
            (r"\bregression\b|\bOLS\b", "trend analysis"),
            (r"\bconfidence interval\b", "expected range"),
            (r"\bcorrelation\b", "relationship"),
            (r"\banomaly score\b", "unusual activity"),
            (r"\bfeature importance\b", "most important factors"),
            (r"\bstandard deviation\b", "typical variation"),
            (r"\bstandard deviations\b", "typical variations"),
            (r"\bmedian\b", "middle value"),
            (r"\bcoefficient of variation\b|\bCV\b", "relative variation"),
            (r"\bdispersion\b", "spread"),
            (r"\bz-scores?\b|\bz score\b", "unusual pattern"),
            (r"\bIQR\b|\binterquartile\b", "usual range"),
            (r"\bSpearman\b|\bPearson\b|\bKruskal\b", "checked pattern"),
            (r"\bFK\b|\bforeign key\b", "linked records"),
            (r"\br-squared\b", "pattern strength"),
            (r"\bstd\.?\b", "typical variation"),
            (r"\bstats\b", "statistics"),
        ]
        out = text
        for pat, repl in subs:
            out = _re.sub(pat, repl, out, flags=_re.IGNORECASE)
        # Collapse excessive decimal precision in prose (keep 1 decimal max).
        out = _re.sub(r"(\d+)\.(\d{3,})", lambda m: f"{m.group(1)}.{m.group(2)[0]}", out)
        return out

    def _narrative_story(self, processed_data, kpis, trends, anomalies, hypotheses,
                         predictions=None):
        """Deterministic, plain-English narrative assembled from the findings.
        Always available and identical for every provider. Conversational:
        short sentences, human-scale numbers, humanized names — every number
        traceable to the computed evidence, never invented."""
        def _hz(name):
            try:
                from core.semantic import humanize as _h
                return _h(str(name or ""))
            except Exception:
                return str(name or "").replace("_", " ").strip() or str(name or "")
        goal = (processed_data.get("user_goal") or "the business question").strip()
        rows = processed_data.get("row_count", 0)
        parts = []
        if goal.lower() != "unknown goal":
            parts.append(f"Looking at your question - {goal} - across {rows} rows of data.")
        else:
            parts.append(f"This covers {rows} rows of data.")

        if hypotheses:
            parts.append(f"The headline: {hypotheses[0]['hypothesis']}")
        if kpis:
            top = max(kpis, key=lambda k: k.get("mean") or 0)
            def _biz(v):
                try:
                    n = float(v)
                except (TypeError, ValueError):
                    return str(v)
                a = abs(n)
                if a >= 1_000_000:
                    return f"{n / 1_000_000:.1f}M"
                if a >= 10_000:
                    return f"{n / 1_000:.1f}K"
                if a >= 100:
                    return f"{round(n):,}"
                if a >= 1:
                    return f"{round(n, 1)}"
                return f"{round(n, 2)}"
            parts.append(
                f"On average, {_hz(top['column']).lower()} is about {_biz(top['mean'])} - "
                f"usually between {_biz(top.get('min'))} and {_biz(top.get('max'))}."
            )
        if trends:
            t = trends[0]
            _cols = [_hz(c).lower() for c in (t.get("columns") or ([t["column"]] if t.get("column") else ["the measure"]))]
            _direction = str(t.get("direction", "mixed")).lower()
            _plain_dir = {"increasing": "up", "decreasing": "down"}.get(_direction, _direction)
            try:
                _pct = float(str(t["pct_change"]).replace("%", ""))
                _pct_s = f"{_pct:+.1f}%"
            except (TypeError, ValueError, KeyError):
                _pct_s = str(t.get("pct_change", ""))
            parts.append(
                f"{', '.join(_cols)} is heading {_plain_dir} over time - about {_pct_s}."
            )
        for h in hypotheses[1:3]:
            parts.append(f"Also: {h['hypothesis']}")
        if anomalies:
            a = anomalies[0]
            parts.append(
                f"One value stands out: {_hz(a['column']).lower()} hit {a['value']} - "
                f"well outside the usual range. Worth a closer look."
            )
        if predictions and predictions.get("valid") and predictions.get("points"):
            period_word = predictions.get("period") or "periods"
            if period_word in ("sequence",):
                period_word = "periods"
            try:
                _pp = float(str(predictions['projected_pct_change']).replace("%", ""))
                _pp_s = f"{_pp:+.1f}%"
            except (TypeError, ValueError, KeyError):
                _pp_s = str(predictions.get("projected_pct_change", ""))
            parts.append(
                f"Looking ahead, {_hz(predictions['column']).lower()} is expected to move "
                f"about {_pp_s} over the next "
                f"{predictions['horizon']} {period_word}."
            )
        story = " ".join(parts)
        if story and (kpis or trends or anomalies or hypotheses):
            return story
        # Insufficient evidence: honest state, or a plain evidence sentence
        # when a single safe fact exists (never invent).
        if trends:
            t = trends[0]
            metric = (t.get("columns") or ["metric"])[0]
            return (f"{metric} {t.get('direction', 'changed')} "
                    f"{t.get('pct_change', '')}% compared with the previous period.")
        if kpis:
            top = max(kpis, key=lambda k: k.get("mean") or 0)
            return (f"ARIA reviewed {rows} rows; {top.get('column')} averages "
                    f"{top.get('mean')} with no strong pattern to report.")
        return (f"ARIA reviewed {rows} rows but found no strong pattern to report. "
                f"Review the KPIs and charts below.")

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self, processed_data_path="processed_data.json", output_path="insights.json"):
        warnings = []
        try:
            processed_data = self.load_processed_data(processed_data_path)
            rows = processed_data.get("data", [])
            df = pd.DataFrame(rows)
        except Exception as exc:
            logging.warning("Insight Agent could not load processed data: %s", exc)
            processed_data = {}
            df = pd.DataFrame()
            warnings.append(f"load_processed_data: {exc}")

        print(f"Insight Agent analysing {processed_data_path} "
              f"({processed_data.get('row_count', len(df))} rows)...")

        def safe(step_name, fn, default):
            try:
                return fn()
            except Exception as exc:
                logging.warning("Insight step '%s' failed: %s", step_name, exc)
                warnings.append(f"{step_name}: {exc}")
                return default

        numeric_cols, category_cols, datetime_cols = safe(
            "classify_columns", lambda: self._classify_columns(df), ([], [], [])
        )
        category_cardinality = 0
        if category_cols:
            category_cardinality = int(df[category_cols[0]].nunique(dropna=True))
        chart_options = compatible_chart_options(
            len(numeric_cols), len(category_cols), len(datetime_cols), category_cardinality
        )
        kpis = safe("compute_kpis", lambda: self.compute_kpis(df, numeric_cols), [])
        trends = safe("detect_trends", lambda: self.detect_trends(df, datetime_cols, numeric_cols), [])
        anomalies = safe("detect_anomalies", lambda: self.detect_anomalies(df, numeric_cols), [])
        chart_suggestions = None
        dashboard_spec = safe(
            "build_dashboard_spec",
            lambda: self.build_dashboard_spec(
                df, numeric_cols, category_cols, datetime_cols, chart_suggestions
            ),
            [],
        )
        hypotheses, ranking_score = safe(
            "generate_hypotheses",
            lambda: self.generate_hypotheses(
                df, numeric_cols, category_cols, datetime_cols, kpis, anomalies
            ),
            ([], 0.0),
        )

        predictions = None
        if numeric_cols:
            try:
                predictions = self.prediction.forecast(
                    processed_data,
                    datetime_col=datetime_cols[0] if datetime_cols else None,
                )
            except Exception as exc:
                logging.warning("Prediction Agent failed: %s", exc)
                warnings.append(f"prediction: {exc}")

        prescriptions = None
        try:
            prescriptions = self.prescription.prescribe(
                processed_data, kpis, trends, anomalies, hypotheses, predictions
            )
        except Exception as exc:
            logging.warning("Prescription Agent failed: %s", exc)
            warnings.append(f"prescription: {exc}")

        story = safe(
            "generate_business_story",
            lambda: self.generate_business_story(
                processed_data, kpis, trends, anomalies, hypotheses, predictions
            ),
            None,
        )
        if not story:
            try:
                story = self._narrative_story(processed_data, kpis, trends, anomalies,
                                              hypotheses, predictions)
            except Exception as exc:
                logging.warning("Template story fallback failed: %s", exc)
                story = (
                    "ARIA analysed the data, but the narrative could not be "
                    "generated. Review the KPIs, charts, and recommendations below."
                )

        insights = {
            "analysis_version": self.VERSION,
            "anomaly_method": self.ANOMALY_METHOD,
            "generated_at": datetime.now().isoformat(),
            "source_processed_data": processed_data_path,
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "summary": {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": category_cols,
                "datetime_columns": datetime_cols,
            },
            "chart_options": chart_options,
            "kpis": kpis,
            "trends": trends,
            "anomalies": anomalies,
            "hypotheses": hypotheses,
            "predictions": predictions,
            "prescriptions": prescriptions,
            "hypothesis_ranking_score": round(ranking_score, 2),
            "dashboard": dashboard_spec,
            "business_story": story,
            "business_story_status": getattr(self, "last_story_status", "SUCCESS"),
            "warning": "; ".join(warnings) if warnings else None,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(sanitize_json(insights), indent=2, default=str), encoding="utf-8"
        )
        print(f"Insight Agent done! Saved to {output_path}")
        return output_path

