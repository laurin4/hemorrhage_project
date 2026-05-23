"""
CLI: construct case-level structures from flat report CSV (Phase 0).

Usage:
  python3 -m src.tasks.hemorrhage.build_cases
  python3 -m src.tasks.hemorrhage.build_cases --input data/raw/reports.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.paths import (
    CASES_CONSTRUCTION_REPORT_PATH,
    CASES_EXPORT_PATH,
    FLAT_REPORTS_INPUT_PATH,
)
from src.tasks.hemorrhage.export.case_export_schema import cases_to_export_dataframe
from src.tasks.hemorrhage.preprocessing.case_builder import (
    build_cases_from_dataframe,
    load_flat_reports_dataframe,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build case-centric structures from flat reports.")
    parser.add_argument(
        "--input",
        type=Path,
        default=FLAT_REPORTS_INPUT_PATH,
        help="Flat report CSV (semicolon-separated)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CASES_EXPORT_PATH,
        help="Case-level export CSV (one row per case)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=CASES_CONSTRUCTION_REPORT_PATH,
        help="Construction log / summary text file",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        LOGGER.error("Input not found: %s", args.input)
        return 1

    df = load_flat_reports_dataframe(args.input)
    cases, stats = build_cases_from_dataframe(df)
    export_df = cases_to_export_dataframe(cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(args.output, index=False, sep=";", encoding="utf-8")
    LOGGER.info("Wrote %d cases to %s", len(export_df), args.output)

    lines = stats.to_summary_lines() + ["", f"export_path={args.output}"]
    if stats.anomaly_messages:
        lines.append("")
        lines.append("Anomalies (first 50):")
        for msg in stats.anomaly_messages[:50]:
            lines.append(f"  - {msg}")
        if len(stats.anomaly_messages) > 50:
            lines.append(f"  ... and {len(stats.anomaly_messages) - 50} more")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote construction report to %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
