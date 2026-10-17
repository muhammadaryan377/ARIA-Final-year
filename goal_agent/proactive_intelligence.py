"""Proactive Intelligence layer — runs after Goal Agent to surface
anomalies, data quality warnings, and 'ARIA noticed' insights."""

import statistics
from typing import Any


def analyze_result_intelligence(data: list[dict], goal: str, message: str = "") -> dict:
    """Lightweight post-processing of Goal Agent results to surface
    proactive insights, anomalies, and data quality warnings."""
    result = {
        "proactive_insights": [],
        "anomalies": [],
        "data_quality_warnings": [],
        "explanation": None,
        "confidence": "high",
    }

    if not data or len(data) < 2:
        return result

    columns = list(data[0].keys())
    numeric_cols = _find_numeric_columns(data, columns)
    categorical_cols = [c for c in columns if c not in numeric_cols]

    # --- Anomaly detection (z-score based) ---
    for col in numeric_cols:
        values = [row[col] for row in data if row.get(col) is not None and isinstance(row[col], (int, float))]
        if len(values) < 4:
            continue
        try:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            if stdev == 0:
                continue
            for i, row in enumerate(data):
                val = row.get(col)
                if val is not None and isinstance(val, (int, float)):
                    z = abs((val - mean) / stdev)
                    if z > 2.0:
                        dim = categorical_cols[0] if categorical_cols else f"Row {i + 1}"
                        dim_val = row.get(dim, f"Row {i + 1}")
                        direction = "above" if val > mean else "below"
                        result["anomalies"].append({
                            "column": col,
                            "dimension": dim,
                            "dimension_value": str(dim_val),
                            "value": val,
                            "expected_range": f"{mean - stdev:.1f} - {mean + stdev:.1f}",
                            "severity": "high" if z > 3.0 else "medium",
                            "description": f"{dim_val} has {col} of {val:.1f}, which is {direction} the typical range ({mean - stdev:.1f} - {mean + stdev:.1f}).",
                        })
        except (statistics.StatisticsError, TypeError):
            continue

    # --- Data quality checks ---
    total_rows = len(data)
    for col in columns:
        null_count = sum(1 for row in data if row.get(col) is None or row.get(col) == "" or row.get(col) == "NULL")
        if null_count > 0:
            pct = (null_count / total_rows) * 100
            if pct > 10:
                result["data_quality_warnings"].append({
                    "column": col,
                    "null_percentage": round(pct, 1),
                    "description": f"{pct:.0f}% of {col} values are missing.",
                })

        # Check for single-value columns (low cardinality)
        distinct = set(str(row.get(col, "")) for row in data)
        if len(distinct) == 1 and total_rows > 3:
            result["data_quality_warnings"].append({
                "column": col,
                "issue": "constant",
                "description": f"{col} has only one distinct value across all rows.",
            })

    # --- Proactive insights ---
    if categorical_cols and numeric_cols:
        dim = categorical_cols[0]
        metric = numeric_cols[0]
        groups = {}
        for row in data:
            key = str(row.get(dim, "Unknown"))
            val = row.get(metric)
            if val is not None and isinstance(val, (int, float)):
                groups.setdefault(key, []).append(val)

        if len(groups) >= 2:
            totals = {k: sum(v) for k, v in groups.items()}
            sorted_groups = sorted(totals.items(), key=lambda x: x[1], reverse=True)
            total = sum(totals.values())
            if total > 0:
                top_name, top_val = sorted_groups[0]
                top_pct = (top_val / total) * 100
                if top_pct >= 40:
                    result["proactive_insights"].append({
                        "type": "dominance",
                        "description": f"{top_name} dominates {metric.replace('_', ' ')} at {top_pct:.0f}% of total.",
                    })

                # Check for underperformers
                if len(sorted_groups) >= 2:
                    bottom_name, bottom_val = sorted_groups[-1]
                    bottom_pct = (bottom_val / total) * 100
                    if top_pct > 0 and bottom_pct > 0 and (top_pct / bottom_pct) > 5:
                        result["proactive_insights"].append({
                            "type": "imbalance",
                            "description": f"{top_name} is {top_pct / bottom_pct:.0f}x larger than {bottom_name} in {metric.replace('_', ' ')}.",
                        })

    # --- Confidence assessment ---
    null_total = sum(
        sum(1 for row in data if row.get(c) is None or row.get(c) == "")
        for c in columns
    )
    total_cells = total_rows * len(columns)
    if total_cells > 0:
        null_ratio = null_total / total_cells
        if null_ratio > 0.3:
            result["confidence"] = "low"
            result["data_quality_warnings"].append({
                "issue": "high_null_ratio",
                "description": f"{null_ratio * 100:.0f}% of all data cells are empty or null.",
            })
        elif null_ratio > 0.1:
            result["confidence"] = "moderate"

    # --- Explanation ---
    if message:
        result["explanation"] = {
            "answer": message,
            "tables_used": list(set(
                str(row.get("_table", "")) for row in data if row.get("_table")
            )) or ["See data"],
            "row_count": total_rows,
            "columns_analyzed": columns,
        }

    return result


def _find_numeric_columns(data: list[dict], columns: list[str]) -> list[str]:
    """Find columns that contain numeric values."""
    numeric = []
    for col in columns:
        vals = [row.get(col) for row in data[:20] if row.get(col) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            numeric.append(col)
    return numeric
