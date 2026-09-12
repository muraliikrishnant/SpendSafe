#!/usr/bin/env python3
"""Evaluation workflow.

1. Runs the full pipeline against `dataset/sample_requests.csv` (25 solved
   examples) and scores predictions against their given expected columns.
2. Writes `evaluation/usage_report.md` summarizing LLM token usage/cost for
   the run that produced the current `output.csv` (call this after
   `code/main.py` has run against the full `dataset/requests.csv`).

Usage:
    python3 evaluation/main.py                 # score against sample_requests.csv
    python3 evaluation/main.py --usage-only     # only (re)write usage_report.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from dotenv import load_dotenv

import extraction
from data_loader import DataStore
from main import process_request

PRICE_PER_1K_TOKENS = {
    # Indicative public per-1K-token pricing for cost estimation only. Ollama
    # is local/free. Update if actual provider pricing differs.
    "nvidia": {"prompt": 0.0002, "completion": 0.0006},
    "ollama": {"prompt": 0.0, "completion": 0.0},
}


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


def write_usage_report(path: Path) -> None:
    usage = extraction.get_usage_log()
    by_provider: dict[str, dict] = {}
    for u in usage:
        key = (u["provider"], u["model"])
        agg = by_provider.setdefault(key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        agg["calls"] += 1
        agg["prompt_tokens"] += u["prompt_tokens"]
        agg["completion_tokens"] += u["completion_tokens"]

    lines = [
        "# Token Usage and Cost Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "This report summarizes model calls made by `code/main.py` for the run",
        "that produced the submitted `output.csv`.",
        "",
        "| Provider | Model | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Est. Cost (USD) |",
        "|---|---|---|---|---|---|---|",
    ]

    total_calls = total_prompt = total_completion = 0
    total_cost = 0.0
    for (provider, model), agg in by_provider.items():
        prices = PRICE_PER_1K_TOKENS.get(provider, {"prompt": 0.0, "completion": 0.0})
        cost = (agg["prompt_tokens"] / 1000) * prices["prompt"] + (agg["completion_tokens"] / 1000) * prices["completion"]
        total_tokens = agg["prompt_tokens"] + agg["completion_tokens"]
        lines.append(
            f"| {provider} | {model} | {agg['calls']} | {agg['prompt_tokens']} | "
            f"{agg['completion_tokens']} | {total_tokens} | ${cost:.4f} |"
        )
        total_calls += agg["calls"]
        total_prompt += agg["prompt_tokens"]
        total_completion += agg["completion_tokens"]
        total_cost += cost

    lines += [
        "",
        "## Overall",
        "",
        f"- Total model calls: {total_calls}",
        f"- Total prompt tokens: {total_prompt}",
        f"- Total completion tokens: {total_completion}",
        f"- Total tokens: {total_prompt + total_completion}",
        f"- Average tokens per request: {((total_prompt + total_completion) / total_calls):.1f}" if total_calls else "- Average tokens per request: n/a",
        f"- Estimated total cost: ${total_cost:.4f}",
        f"- Estimated cost per request: ${(total_cost / total_calls):.6f}" if total_calls else "- Estimated cost per request: n/a",
        "",
        "Note: Ollama calls are local and free (cost $0); NVIDIA NIM pricing above",
        "is indicative and should be replaced with actual invoiced rates if available.",
    ]

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--usage-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not args.usage_only:
        report = score_against_samples(Path(args.dataset_dir))
        print("Scored against dataset/sample_requests.csv:")
        print(f"  total examples: {report['total']}")
        for col, n in report["exact_matches"].items():
            print(f"  {col}: {n}/{report['total']} exact match")
        if "mean_abs_amount_error" in report:
            print(f"  amount_safe_to_pay mean abs error: {report['mean_abs_amount_error']:.2f}")

    write_usage_report(REPO_ROOT / "evaluation" / "usage_report.md")
    print("Wrote evaluation/usage_report.md")


if __name__ == "__main__":
    main()
