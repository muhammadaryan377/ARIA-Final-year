"""Data-driven chart option catalog for the Insight Agent.

Each option describes the data shape it can represent and the renderer family
used by the current dashboard contract. Selection is deterministic; this
catalog never asks an LLM to choose an incompatible visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ChartOption:
    chart_type: str
    label: str
    family: str
    shapes: tuple[str, ...]
    render_type: str
    analyst_use: str


_OPTIONS = (
    ("bar", "Bar", "comparison", ("category_numeric", "category_counts"), "bar", "Compare groups"),
    ("bar_horizontal", "Horizontal Bar", "comparison", ("category_numeric", "category_counts"), "horizontal-bar", "Compare groups with long labels"),
    ("bar_stacked", "Stacked Bar", "composition", ("category_numeric_multi",), "bar", "Show parts of a category total"),
    ("bar_grouped", "Grouped Bar", "comparison", ("category_numeric_multi",), "bar", "Compare several measures by group"),
    ("bar_normalized", "100% Stacked Bar", "composition", ("category_numeric_multi",), "bar", "Compare composition independent of size"),
    ("bar_diverging", "Diverging Bar", "comparison", ("category_numeric",), "bar", "Show positive and negative variance"),
    ("lollipop", "Lollipop", "comparison", ("category_numeric", "category_counts"), "bar", "Rank a small number of groups"),
    ("pareto", "Pareto", "comparison", ("category_numeric", "category_counts"), "bar", "Find the vital few contributors"),
    ("waterfall", "Waterfall", "composition", ("category_numeric",), "bar", "Explain sequential gains and losses"),
    ("line", "Line", "trend", ("datetime_numeric", "category_numeric"), "line", "Show change over an ordered axis"),
    ("multi_line", "Multi-line", "trend", ("datetime_numeric_multi",), "line", "Compare several trends"),
    ("area", "Area", "trend", ("datetime_numeric",), "area", "Emphasize magnitude over time"),
    ("stacked_area", "Stacked Area", "composition", ("datetime_numeric_multi",), "area", "Show changing composition over time"),
    ("step_line", "Step Line", "trend", ("datetime_numeric",), "line", "Show discrete period changes"),
    ("slope", "Slope", "trend", ("datetime_numeric_multi",), "line", "Compare start and end positions"),
    ("sparkline", "Sparkline", "trend", ("datetime_numeric",), "line", "Show a compact trend"),
    ("combo", "Combo Line + Bar", "comparison", ("datetime_numeric_multi", "datetime_numeric"), "bar", "Compare measures with different scales"),
    ("scatter", "Scatter", "relationship", ("numeric_numeric",), "scatter", "Test a relationship between measures"),
    ("bubble", "Bubble", "relationship", ("numeric_numeric_numeric",), "scatter", "Add a third measure as size"),
    ("hexbin", "Hexbin", "relationship", ("numeric_numeric",), "scatter", "Show dense point concentration"),
    ("contour", "Contour", "relationship", ("numeric_numeric",), "scatter", "Show relationship density"),
    ("pie", "Pie", "composition", ("share", "category_low_card"), "pie", "Show shares for a few groups"),
    ("doughnut", "Doughnut", "composition", ("share", "category_low_card"), "donut", "Show shares with a central total"),
    ("polar_area", "Polar Area", "composition", ("share",), "pie", "Compare category shares and magnitude"),
    ("radar", "Radar", "comparison", ("category_numeric_multi",), "line", "Compare several measures across few groups"),
    ("histogram", "Histogram", "distribution", ("distribution",), "bar", "Show the distribution of a measure"),
    ("density", "Density Curve", "distribution", ("distribution",), "line", "Show a smoothed distribution"),
    ("box_plot", "Box Plot", "distribution", ("category_numeric", "distribution"), "bar", "Compare spread and outliers"),
    ("violin", "Violin", "distribution", ("category_numeric",), "area", "Compare distribution shape by group"),
    ("ecdf", "ECDF", "distribution", ("distribution",), "line", "Compare cumulative distributions"),
    ("heatmap", "Heatmap", "matrix", ("category_numeric_multi", "datetime_numeric_multi"), "bar", "Spot intensity across two dimensions"),
    ("calendar_heatmap", "Calendar Heatmap", "matrix", ("datetime_numeric",), "bar", "Spot daily or weekly seasonality"),
    ("control_chart", "Control Chart", "quality", ("datetime_numeric",), "line", "Monitor process stability"),
    ("bullet", "Bullet", "comparison", ("category_numeric",), "bar", "Compare actuals with a target"),
    ("gauge", "Gauge", "kpi", ("category_numeric", "distribution"), "pie", "Show one value against a range"),
    ("funnel", "Funnel", "composition", ("category_numeric",), "bar", "Show sequential conversion stages"),
    ("kpi_card", "KPI Card", "kpi", ("kpi",), "kpi_card", "Show a headline metric"),
)

CHART_OPTIONS = tuple(ChartOption(*option) for option in _OPTIONS)
CHART_OPTION_INDEX = {option.chart_type: option for option in CHART_OPTIONS}


def compatible_chart_options(
    numeric_count: int,
    category_count: int,
    datetime_count: int,
    category_cardinality: int = 0,
) -> list[dict]:
    """Return compatible chart options in analyst-oriented priority order."""
    if datetime_count and numeric_count >= 2:
        shape = "datetime_numeric_multi"
    elif datetime_count and numeric_count:
        shape = "datetime_numeric"
    elif category_count and numeric_count >= 2:
        shape = "category_numeric_multi"
    elif category_count and numeric_count:
        shape = "category_numeric"
    elif category_count:
        shape = "category_counts"
    elif numeric_count >= 3:
        shape = "numeric_numeric_numeric"
    elif numeric_count == 2:
        shape = "numeric_numeric"
    elif numeric_count == 1:
        shape = "distribution"
    else:
        shape = "kpi"

    compatible = [option for option in CHART_OPTIONS if shape in option.shapes]
    if category_cardinality and category_cardinality > 12:
        compatible.sort(key=lambda option: option.chart_type not in {"bar_horizontal", "bar", "pareto"})
    elif 1 < category_cardinality <= 6:
        compatible.sort(key=lambda option: option.chart_type not in {"bar", "pie", "doughnut", "polar_area"})
    return [asdict(option) for option in compatible]
