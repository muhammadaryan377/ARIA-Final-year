"""Command-line entry point for ``python -m insight_agent``."""

import argparse

from .agent import InsightAgent


def main():
    parser = argparse.ArgumentParser(
        description="ARIA Insight Agent - KPI, trend, anomaly, and dashboard analysis"
    )
    parser.add_argument("processed_data", help="Path to processed data JSON")
    parser.add_argument(
        "--output",
        default="insights.json",
        help="Path for the generated insights JSON (default: insights.json)",
    )
    args = parser.parse_args()
    InsightAgent().analyze(args.processed_data, args.output)


if __name__ == "__main__":
    main()
