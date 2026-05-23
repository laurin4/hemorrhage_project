"""
CLI: descriptive analytics on CCM DAVF reference labels (no NLP).

  python3 -m src.tasks.hemorrhage.analyze_reference_labels
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.paths import INSPECTION_DIR
from src.tasks.hemorrhage.analysis.reference_labels import run_reference_label_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze hemorrhage reference spreadsheet labels (descriptive only)."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference Excel (default: data/raw/260507_CCM_DAVF.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INSPECTION_DIR,
        help="Output directory (default: data/inspection/)",
    )
    args = parser.parse_args(argv)

    result = run_reference_label_analysis(
        reference_path=args.reference,
        output_dir=args.output_dir,
    )

    for line in result.summary_lines:
        print(line)

    if result.errors:
        for err in result.errors:
            logging.warning("%s", err)

    if result.total_rows == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
