"""
CLI: merge hemorrhage case classifications into the patient/case spreadsheet.

Reads the predictions (``data/outputs/hemorrhage_case_predictions.csv``) and a
template Excel under ``data/raw/`` (default ``NCH_cavernom_eingeblutet.xlsx``),
fills one-hot final-class columns per report row, and writes a merged copy to
``data/outputs/`` (the raw template is never modified).

  python3 -m src.tasks.hemorrhage.merge_classifications
  python3 -m src.tasks.hemorrhage.merge_classifications --template path/to/sheet.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.tasks.hemorrhage.export.classification_merge import run_merge_classifications

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge hemorrhage case classifications into a patient/case spreadsheet."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Predictions CSV (default: data/outputs/hemorrhage_case_predictions.csv)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Template Excel under data/raw/ (default: NCH_cavernom_eingeblutet.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Merged Excel output (default: data/outputs/NCH_cavernom_eingeblutet_classified.xlsx)",
    )
    args = parser.parse_args(argv)

    result = run_merge_classifications(
        predictions_path=args.predictions,
        template_path=args.template,
        output_path=args.output,
    )

    for line in result.summary_lines:
        print(line)

    if result.template_rows == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
