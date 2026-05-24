"""
CLI: build unified hemorrhage prediction review export for qualitative inspection.

  python3 -m src.tasks.hemorrhage.build_prediction_review
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.paths import (
    HEMORRHAGE_CASE_PREDICTIONS_PATH,
    HEMORRHAGE_CONFUSION_REVIEW_PATH,
    HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH,
    HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH,
)
from src.tasks.hemorrhage.export.prediction_review import run_build_prediction_review

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build hemorrhage prediction review table "
            "(preliminary comparison / qualitative review — not final evaluation)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=HEMORRHAGE_CASE_PREDICTIONS_PATH,
        help="Predictions CSV (default: data/outputs/hemorrhage_case_predictions.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HEMORRHAGE_PREDICTION_REVIEW_PATH,
        help="Review CSV output path",
    )
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=HEMORRHAGE_CONFUSION_REVIEW_PATH,
        help="Compact confusion review CSV output path",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH,
        help="Summary text output path",
    )
    parser.add_argument("--reports", type=Path, default=None, help="Reports Excel for case previews")
    parser.add_argument("--reference", type=Path, default=None, help="Reference Excel override")
    parser.add_argument("--limit", type=int, default=None, help="Write only first N review rows")
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Include only FP/FN and parse/LLM failures",
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help="Include only hemorrhagic/non_hemorrhagic reference labels",
    )
    parser.add_argument(
        "--only-fn",
        action="store_true",
        help="Write only FN rows to main review CSV (FN/FP detailed exports still created)",
    )
    parser.add_argument(
        "--only-fp",
        action="store_true",
        help="Write only FP rows to main review CSV (FN/FP detailed exports still created)",
    )
    parser.add_argument(
        "--fn-output",
        type=Path,
        default=HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH,
        help="Detailed false-negative review CSV path",
    )
    parser.add_argument(
        "--fp-output",
        type=Path,
        default=HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH,
        help="Detailed false-positive review CSV path",
    )
    args = parser.parse_args(argv)

    if args.only_fn and args.only_fp:
        parser.error("Use only one of --only-fn or --only-fp")

    result = run_build_prediction_review(
        predictions_path=args.input,
        review_path=args.output,
        confusion_path=args.confusion_output,
        summary_path=args.summary,
        reports_path=args.reports,
        reference_path=args.reference,
        limit=args.limit,
        only_mismatches=args.only_mismatches,
        only_labeled=args.only_labeled,
        only_fn=args.only_fn,
        only_fp=args.only_fp,
        false_negative_path=args.fn_output,
        false_positive_path=args.fp_output,
    )

    for line in result.summary_lines:
        print(line)

    if result.rows_written == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
