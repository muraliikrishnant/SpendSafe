#!/usr/bin/env python3
"""Evaluation workflow.

Runs the full pipeline against `dataset/sample_requests.csv` (25 solved
examples) and scores predictions against their given expected columns.

NOTE on token usage: `evaluation/usage_report.md` is written by
`code/main.py` itself at the end of its full-dataset run, NOT by this
script — that's the run that actually produces `output.csv`, and the
submission spec requires the report to reflect exactly that run. Scoring
here mostly hits the on-disk extraction cache (evidence overlaps with the
full dataset), so any usage this script would record is not representative
and is intentionally not written to the shared report. Use --usage-only
only to manually regenerate it from whatever calls happen to occur in this
process (rare, cache permitting) — prefer re-running code/main.py instead.

Usage:
    python3 evaluation/main.py                 # score against sample_requests.csv
    python3 evaluation/main.py --usage-only     # manually (re)write usage_report.md from this process
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from dotenv import load_dotenv

import extraction
from data_loader import DataStore
from main import process_request


def score_against_samples(dataset_dir: Path) -> dict:
    ds = DataStore.load(str(dataset_dir))
    sample_path = dataset_dir / "sample_requests.csv"

    total = 0
    exact_matches = {
        "affordability_status": 0,
        "recommended_payment_method": 0,
        "payment_plan": 0,
        "earliest_date_for_full_payment": 0,
        "spending_changes_needed": 0,
    }
    amount_abs_errors = []

    with open(sample_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            request_row = {k: row[k] for k in [
                "request_id", "user_id", "request_date", "request_type",
                "requested_amount", "desired_completion_date",
                "allows_partial_payment", "request_text",
            ]}
            predicted = process_request(request_row, ds)
            predicted_csv = predicted.to_csv_row()

            for col in exact_matches:
                if predicted_csv[col] == row[col]:
                    exact_matches[col] += 1

            try:
                amount_abs_errors.append(abs(float(predicted_csv["amount_safe_to_pay"]) - float(row["amount_safe_to_pay"])))
            except ValueError:
                pass

    report = {"total": total, "exact_matches": exact_matches}
    if amount_abs_errors:
        report["mean_abs_amount_error"] = sum(amount_abs_errors) / len(amount_abs_errors)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--usage-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if args.usage_only:
        extraction.write_usage_report(REPO_ROOT / "evaluation" / "usage_report.md")
        print("Wrote evaluation/usage_report.md from calls made in this process only.")
        print("For the authoritative report, re-run code/main.py — it writes this file itself.")
        return

    report = score_against_samples(Path(args.dataset_dir))
    print("Scored against dataset/sample_requests.csv:")
    print(f"  total examples: {report['total']}")
    for col, n in report["exact_matches"].items():
        print(f"  {col}: {n}/{report['total']} exact match")
    if "mean_abs_amount_error" in report:
        print(f"  amount_safe_to_pay mean abs error: {report['mean_abs_amount_error']:.2f}")


if __name__ == "__main__":
    main()
