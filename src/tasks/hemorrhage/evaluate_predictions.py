"""
CLI: preliminary evaluation of hemorrhage case-level predictions.

  python3 -m src.tasks.hemorrhage.evaluate_predictions
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.paths import (
    HEMORRHAGE_CONFUSION_REVIEW_PATH,
    HEMORRHAGE_EVALUATION_DIR,
    HEMORRHAGE_PREDICTION_REVIEW_PATH,
)
from src.tasks.hemorrhage.evaluation.runner import run_evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preliminary evaluation on labeled subset "
            "(NOT final validation — Verify_Vaskulär excluded by default)."
        )
    )
    parser.add_argument(
        "--input-review",
        type=Path,
        default=HEMORRHAGE_PREDICTION_REVIEW_PATH,
        help="Prediction review CSV (default: data/outputs/hemorrhage_prediction_review.csv)",
    )
    parser.add_argument(
        "--input-confusion",
        type=Path,
        default=HEMORRHAGE_CONFUSION_REVIEW_PATH,
        help="Confusion review CSV (default: data/outputs/hemorrhage_confusion_review.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HEMORRHAGE_EVALUATION_DIR,
        help="Evaluation output directory (default: data/evaluation/)",
    )
    parser.add_argument(
        "--include-verify-as-negative",
        action="store_true",
        help=(
            "Also run sensitivity analysis treating verify_only as non_hemorrhagic "
            "(exploratory; default is conservative exclusion)."
        ),
    )
    args = parser.parse_args(argv)

    result = run_evaluate_predictions(
        review_path=args.input_review,
        confusion_path=args.input_confusion,
        output_dir=args.output_dir,
        include_verify_as_negative=args.include_verify_as_negative,
    )

    for line in result.summary_lines:
        print(line)

    if result.sensitivity_summary_lines:
        print("")
        print("--- Sensitivity analysis summary ---")
        for line in result.sensitivity_summary_lines:
            print(line)

    if result.warnings:
        print("")
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
