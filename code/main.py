#!/usr/bin/env python3
"""Entry point: streams dataset/requests.csv, runs the full deterministic
pipeline (with constrained LLM extraction/explanation) per request, and
writes output.csv at the repo root — incrementally and resumably so this
scales from 250 requests to millions without a rewrite.

Usage:
    python3 code/main.py [--dataset-dir dataset] [--output ../output.csv]
                         [--workers 8] [--batch-size 50] [--no-resume]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agents_log
import extraction
import payment_options as po
import reconstruction
import safety_engine as se
import spending_changes as sc
from data_loader import DataStore, _parse_date, iter_requests
from schemas import AffordabilityStatus, OutputRow, Payment, PaymentMethod, SpendingChange

REPO_ROOT = Path(__file__).resolve().parent.parent
HORIZON_DAYS = 90

_resolved_users: set[str] = set()
_resolve_lock = threading.Lock()


def _to_bool(s: str) -> bool:
    return (s or "").strip().lower() == "true"


def resolve_blank_amounts(ds: DataStore, user_id: str) -> None:
    """For any event with a blank amount, pull the linked image (if any) and
    fill amount/currency via vision extraction. Never treats a blank as zero."""
    for event in ds.events_by_user.get(user_id, []):
        if event.get("amount", "").strip():
            continue
        img = ds.images_by_event.get(event["event_id"])
        if not img:
            continue
        context = f"{event.get('event_type')} / {event.get('category')} for a financial event on {event.get('event_date')}"
        # Debit (expense) amounts: an unresolved contradiction is safer
        # resolved toward the higher figure (assume the larger liability).
        # Credit (income) amounts: resolved toward the lower figure.
        conservative = "max" if event.get("direction") == "debit" else "min"
        fact = extraction.extract_amount_from_image(
            ds.image_path(img["image_id"]), img["image_id"], context, conservative=conservative
        )
        if fact and fact.amount is not None:
            event["amount"] = str(fact.amount)
            if fact.currency:
                event["currency"] = fact.currency


def apply_messages(ds: DataStore, request_id: str, user_id: str, request_date: date, home_ccy: str):
    """Apply message-derived facts conservatively:
    - a message linked to a specific event (related_event_id) can cancel or
      amend that event's amount if it explicitly says so;
    - an unlinked income-change fact is only applied when it comes from an
      authoritative third-party source (employer/bank/payroll), never from
      the user's own speculative statements (CLAUDE.md §6).
    Returns (extra_ledger_items, event_overrides) where event_overrides maps
    event_id -> "cancel" or a new amount (float, already in home currency).
    """
    extra_items: list[reconstruction.LedgerItem] = []
    overrides: dict[str, object] = {}

    relevant = list(ds.messages_by_request.get(request_id, [])) + [
        m for m in ds.messages_by_user.get(user_id, []) if not m.get("request_id")
    ]
    for msg in relevant:
        facts = extraction.extract_facts_from_message(msg)
        related_event_id = msg.get("related_event_id")
        for fact in facts:
            if related_event_id:
                if fact.fact_type == "cancellation":
                    overrides[related_event_id] = "cancel"
                elif fact.fact_type in {"amendment", "amount"} and fact.amount is not None:
                    overrides[related_event_id] = fact.amount
                continue

            if fact.fact_type == "income_change" and fact.amount and msg.get("source_type") in {
                "employer",
                "bank",
                "payroll",
            }:
                eff = fact.effective_date or request_date
                if request_date <= eff <= request_date + __import__("datetime").timedelta(days=HORIZON_DAYS):
                    amount_home = ds.convert(fact.amount, fact.currency or home_ccy, home_ccy, eff)
                    extra_items.append(
                        reconstruction.LedgerItem(
                            on_date=eff,
                            amount_home_ccy=amount_home,
                            event_id=f"message:{msg['message_id']}",
                            category="salary",
                            flexibility="fixed",
                            projected=False,
                        )
                    )
    return extra_items, overrides


def _fallback_row(request_id: str, reason: str) -> OutputRow:
    return OutputRow(
        request_id=request_id,
        amount_safe_to_pay=0.0,
        affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
        recommended_payment_method=PaymentMethod.NOT_RECOMMENDED,
        payment_plan=[],
        earliest_date_for_full_payment=None,
        spending_changes_needed=[],
        decision_explanation=f"Unable to safely evaluate this request: {reason}.",
    )


def process_request(row: dict, ds: DataStore) -> OutputRow:
    request_id = row["request_id"]
    user_id = row["user_id"]
    profile = ds.profiles_by_user.get(user_id)
    if not profile:
        return _fallback_row(request_id, "no financial profile found for this user")

    try:
        home_ccy = profile["home_currency"]
        balance = float(profile["current_available_balance"])
        min_balance = float(profile["minimum_balance_to_keep"])
        request_date = _parse_date(row["request_date"])
        desired_completion = _parse_date(row["desired_completion_date"])
        requested_amount = float(row["requested_amount"])
        allows_partial = _to_bool(row.get("allows_partial_payment", ""))

        with _resolve_lock:
            if user_id not in _resolved_users:
                resolve_blank_amounts(ds, user_id)
                _resolved_users.add(user_id)

        ledger = reconstruction.build_ledger(ds, user_id, home_ccy, request_date, HORIZON_DAYS)
        extra_items, overrides = apply_messages(ds, request_id, user_id, request_date, home_ccy)
        ledger = [it for it in ledger if overrides.get(it.event_id) != "cancel"]
        for it in ledger:
            if it.event_id in overrides and isinstance(overrides[it.event_id], (int, float)):
                delta_sign = 1 if it.amount_home_ccy >= 0 else -1
                it.amount_home_ccy = delta_sign * abs(float(overrides[it.event_id]))
        ledger += extra_items

        forecast = se.build_forecast(balance, ledger, request_date, HORIZON_DAYS)
        amt_safe = se.amount_safe_to_pay(forecast, min_balance, requested_amount)
        earliest_full = se.earliest_date_for_full_payment(forecast, min_balance, requested_amount)

        candidates: list[po.CandidatePlan] = []

        if po.user_accepts_method(profile, "full_payment"):
            payments = [(request_date, requested_amount)]
            safe, _ = se.verify_plan(forecast, payments, min_balance)
            if safe:
                candidates.append(
                    po.CandidatePlan(
                        "full_payment", payments, request_date <= desired_completion, False, requested_amount, request_date, 1
                    )
                )

        if (
            allows_partial
            and po.user_accepts_method(profile, "partial_payment")
            and 0 < amt_safe < requested_amount
            and earliest_full
            and earliest_full <= desired_completion
        ):
            payments = [(request_date, round(amt_safe, 2)), (earliest_full, round(requested_amount - amt_safe, 2))]
            safe, _ = se.verify_plan(forecast, payments, min_balance)
            if safe:
                candidates.append(po.CandidatePlan("partial_payment", payments, True, False, requested_amount, request_date, 2))

        if po.user_accepts_method(profile, "installments"):
            for opt in po.build_installment_options(ds.payment_options_by_request.get(request_id, [])):
                if not po.installment_within_user_limit(profile, opt):
                    continue
                safe, _ = se.verify_plan(forecast, opt.payments, min_balance)
                if safe:
                    candidates.append(
                        po.CandidatePlan(
                            "installments",
                            opt.payments,
                            opt.payments[-1][0] <= desired_completion,
                            False,
                            opt.total_payable_amount,
                            opt.payments[0][0],
                            opt.number_of_payments,
                            opt.payment_option_id,
                        )
                    )

        if po.user_accepts_method(profile, "full_payment") and earliest_full and earliest_full > request_date:
            candidates.append(
                po.CandidatePlan(
                    "wait",
                    [(earliest_full, requested_amount)],
                    earliest_full <= desired_completion,
                    False,
                    requested_amount,
                    earliest_full,
                    1,
                )
            )

        spending_changes_used: list[SpendingChange] = []
        chosen: po.CandidatePlan | None = None
        chosen_forecast = forecast

        if candidates:
            chosen = po.rank_candidates(candidates)[0]
        else:
            attempts: list[po.CandidatePlan] = []
            if po.user_accepts_method(profile, "full_payment"):
                attempts.append(
                    po.CandidatePlan(
                        "full_payment", [(request_date, requested_amount)], request_date <= desired_completion, True, requested_amount, request_date, 1
                    )
                )
            if po.user_accepts_method(profile, "installments"):
                for opt in po.build_installment_options(ds.payment_options_by_request.get(request_id, [])):
                    if po.installment_within_user_limit(profile, opt):
                        attempts.append(
                            po.CandidatePlan(
                                "installments",
                                opt.payments,
                                opt.payments[-1][0] <= desired_completion,
                                True,
                                opt.total_payable_amount,
                                opt.payments[0][0],
                                opt.number_of_payments,
                                opt.payment_option_id,
                            )
                        )

            rescued = []
            for attempt in attempts:
                result = sc.try_resolve_with_spending_changes(forecast, ledger, attempt.payments, min_balance, profile)
                if result.is_safe:
                    rescued.append((attempt, result))
            if rescued:
                best_attempt, best_result = po.rank_candidates([a for a, _ in rescued])[0], None
                for a, r in rescued:
                    if a is best_attempt:
                        best_result = r
                        break
                chosen = best_attempt
                spending_changes_used = best_result.changes
                chosen_forecast = best_result.adjusted_forecast

        worst_case = min(chosen_forecast.balance_no_purchase)

        if chosen is None:
            status = AffordabilityStatus.NOT_AFFORDABLE
            method = PaymentMethod.NOT_RECOMMENDED
            payments_out: list[Payment] = []
        else:
            if chosen.method == "full_payment" and chosen.start_date == request_date and not spending_changes_used:
                status = AffordabilityStatus.AFFORDABLE_NOW
            elif chosen.method in {"full_payment", "partial_payment", "installments"}:
                status = AffordabilityStatus.AFFORDABLE_WITH_PLAN
            else:
                status = AffordabilityStatus.AFFORDABLE_LATER
            method = PaymentMethod(chosen.method)
            payments_out = [Payment(pay_date=d, amount=round(a, 2)) for d, a in chosen.payments]

        from explanation import ExplanationFacts, generate_explanation

        facts = ExplanationFacts(
            currency=home_ccy,
            requested_amount=requested_amount,
            amount_safe_to_pay=amt_safe,
            min_balance=min_balance,
            worst_case_balance=worst_case,
            status=status.value,
            method=method.value,
            payments=[(p.pay_date.isoformat(), p.amount) for p in payments_out],
            earliest_full_date=earliest_full.isoformat() if earliest_full else None,
            spending_changes=[c.render() for c in spending_changes_used],
            request_type=row.get("request_type", ""),
        )
        explanation_text = generate_explanation(facts)

        return OutputRow(
            request_id=request_id,
            amount_safe_to_pay=round(amt_safe, 2),
            affordability_status=status,
            recommended_payment_method=method,
            payment_plan=payments_out,
            earliest_date_for_full_payment=earliest_full,
            spending_changes_needed=spending_changes_used,
            decision_explanation=explanation_text,
        )
    except Exception as exc:  # noqa: BLE001 - never let one bad row kill the run
        traceback.print_exc()
        return _fallback_row(request_id, f"internal error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("MAX_WORKERS", "8")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "50")))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    agents_log.session_start()

    from output_writer import OutputWriter

    ds = DataStore.load(args.dataset_dir)
    writer = OutputWriter(args.output, resume=not args.no_resume)

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in iter_requests(args.dataset_dir, chunk_size=args.batch_size):
            pending = [row for row in batch if not writer.is_done(row["request_id"])]
            futures = {pool.submit(process_request, row, ds): row for row in pending}
            for fut in as_completed(futures):
                out_row = fut.result()
                writer.write(out_row)
                total += 1
                if total % 25 == 0:
                    print(f"...{total} requests written")

    writer.close()
    print(f"Done. Wrote predictions to {args.output}")

    summary = f"Processed a batch of requests from {args.dataset_dir}; wrote {args.output}."
    agents_log.log_turn(
        title="Run full-dataset affordability pipeline",
        user_prompt="(automated run — see chat transcript for the originating request)",
        summary=summary,
        actions=[f"python3 code/main.py -> {args.output}"],
    )


if __name__ == "__main__":
    main()
