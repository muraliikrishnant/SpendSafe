#!/usr/bin/env python3
"""Hard-constraint validator for output.csv.

Checks every rule the submission spec states as a MUST, so a violation is
caught here rather than by a grader. Deliberately independent of the
pipeline code: it re-derives everything from dataset/*.csv and output.csv,
so a bug in the engine cannot also hide itself from the check.

Usage:
    python3 evaluation/validate_output.py [--output output.csv]

Exits non-zero if anything fails, so it can gate a run in CI or a script.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from schemas import OUTPUT_COLUMNS

ALLOWED_STATUS = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
ALLOWED_METHOD = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate(output_path: Path, dataset_dir: Path) -> list[str]:
    errors: list[str] = []

    requests = {r["request_id"]: r for r in csv.DictReader(open(dataset_dir / "requests.csv", newline=""))}
    events = {r["event_id"]: r for r in csv.DictReader(open(dataset_dir / "financial_events.csv", newline=""))}
    options: dict[str, list[dict]] = {}
    for row in csv.DictReader(open(dataset_dir / "request_payment_options.csv", newline="")):
        options.setdefault(row["request_id"], []).append(row)

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != OUTPUT_COLUMNS:
            errors.append(f"column mismatch: {reader.fieldnames} != {OUTPUT_COLUMNS}")
        rows = list(reader)

    seen: set[str] = set()
    for row in rows:
        rid = row["request_id"]
        if rid in seen:
            errors.append(f"{rid}: duplicate row")
        seen.add(rid)
        req = requests.get(rid)
        if req is None:
            errors.append(f"{rid}: not present in requests.csv")
            continue

        requested = float(req["requested_amount"])
        try:
            amount = float(row["amount_safe_to_pay"])
        except ValueError:
            errors.append(f"{rid}: amount_safe_to_pay is not numeric")
            continue
        if not (0 <= amount <= requested + 0.01):
            errors.append(f"{rid}: amount_safe_to_pay {amount} outside [0, {requested}]")

        status, method = row["affordability_status"], row["recommended_payment_method"]
        if status not in ALLOWED_STATUS:
            errors.append(f"{rid}: invalid affordability_status '{status}'")
        if method not in ALLOWED_METHOD:
            errors.append(f"{rid}: invalid recommended_payment_method '{method}'")
        if "Unable to safely evaluate" in row["decision_explanation"]:
            errors.append(f"{rid}: internal-error fallback row")

        payments: list[tuple[str, float]] = []
        if row["payment_plan"] != "none":
            for part in row["payment_plan"].split("|"):
                bits = part.split(":")
                if len(bits) != 2:
                    errors.append(f"{rid}: malformed payment '{part}'")
                    continue
                try:
                    datetime.strptime(bits[0], "%Y-%m-%d")
                    payments.append((bits[0], float(bits[1])))
                except ValueError:
                    errors.append(f"{rid}: malformed payment '{part}'")
            if [p[0] for p in payments] != sorted(p[0] for p in payments):
                errors.append(f"{rid}: payment_plan is not chronological")

        if method == "partial_payment":
            if status != "affordable_with_plan":
                errors.append(f"{rid}: partial_payment requires affordable_with_plan, got '{status}'")
            if len(payments) != 2:
                errors.append(f"{rid}: partial_payment needs exactly 2 payments, got {len(payments)}")
            else:
                if abs(payments[0][1] + payments[1][1] - requested) > 0.02:
                    errors.append(f"{rid}: partial payments do not sum to requested_amount")
                if payments[0][0] != req["request_date"]:
                    errors.append(f"{rid}: first partial payment is not on request_date")
                if payments[1][0] > req["desired_completion_date"]:
                    errors.append(f"{rid}: second partial payment is after desired_completion_date")

        if method == "installments":
            supplied = [o for o in options.get(rid, []) if o["payment_method"] == "installments"]
            if supplied and not any(
                len(payments) == int(o["number_of_payments"])
                and all(abs(p[1] - float(o["payment_amount"])) < 0.02 for p in payments)
                for o in supplied
            ):
                errors.append(f"{rid}: installment plan matches no supplied payment option")

        if status == "affordable_now" and row["earliest_date_for_full_payment"] != req["request_date"]:
            errors.append(f"{rid}: affordable_now but earliest_date_for_full_payment != request_date")

        if row["spending_changes_needed"] != "none":
            parts = row["spending_changes_needed"].split("|")
            if len(parts) > 3:
                errors.append(f"{rid}: more than three spending changes")
            stopped, reduced = set(), set()
            for change in parts:
                bits = change.split(":")
                if change.startswith("stop:") and len(bits) == 2:
                    event_id, new_amount = bits[1], None
                    stopped.add(event_id)
                elif change.startswith("reduce_to:") and len(bits) == 3:
                    event_id = bits[1]
                    reduced.add(event_id)
                    try:
                        new_amount = float(bits[2])
                    except ValueError:
                        errors.append(f"{rid}: non-numeric reduce_to amount in '{change}'")
                        continue
                else:
                    errors.append(f"{rid}: malformed spending change '{change}'")
                    continue

                event = events.get(event_id)
                if event is None:
                    errors.append(f"{rid}: spending change references unknown event '{event_id}'")
                    continue
                if event["flexibility"] == "fixed":
                    errors.append(f"{rid}: spending change targets non-flexible event '{event_id}'")
                floor = event["minimum_allowed_amount"].strip()
                if new_amount is not None and floor and new_amount < float(floor) - 0.01:
                    errors.append(
                        f"{rid}: reduce_to {new_amount} is below the {floor} floor for '{event_id}'"
                    )
            if stopped & reduced:
                errors.append(f"{rid}: event(s) both stopped and reduced: {sorted(stopped & reduced)}")

    missing = set(requests) - seen
    if missing:
        errors.append(f"missing rows for {len(missing)} request_ids, e.g. {sorted(missing)[:5]}")

    print(f"Checked {len(rows)} rows against {len(requests)} requests.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    args = parser.parse_args()

    errors = validate(Path(args.output), Path(args.dataset_dir))
    if errors:
        print(f"\nFAIL — {len(errors)} problem(s):")
        for e in errors[:50]:
            print("  -", e)
        sys.exit(1)
    print("PASS — every hard constraint satisfied.")


if __name__ == "__main__":
    main()
