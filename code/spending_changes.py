"""Greedy flexible-spending adjustment: only used when a plan is otherwise
unsafe. Only ever touches events already marked flexible in the data AND in
a category the user has explicitly said they're willing to reduce/stop
(CLAUDE.md §9 — essential spending is never touched, and the user stays in
control of which categories are adjustable).
"""
from __future__ import annotations

from dataclasses import dataclass

from reconstruction import LedgerItem
from safety_engine import Forecast, verify_plan
from schemas import SpendingChange

MAX_CHANGES = 3


@dataclass
class AdjustmentResult:
    is_safe: bool
    min_balance_reached: float
    changes: list[SpendingChange]
    adjusted_forecast: Forecast


def _adjustable_items(
    ledger: list[LedgerItem], profile: dict
) -> list[LedgerItem]:
    stoppable_categories = set((profile.get("expense_categories_user_is_willing_to_stop") or "").split("|"))
    reducible_categories = set((profile.get("expense_categories_user_is_willing_to_reduce") or "").split("|"))
    out = []
    for item in ledger:
        if item.amount_home_ccy >= 0:
            continue  # only expenses can be reduced/stopped
        if item.flexibility == "stoppable" and item.category in stoppable_categories:
            out.append(item)
        elif item.flexibility == "reducible" and item.category in reducible_categories:
            out.append(item)
        elif item.flexibility == "reducible_or_stoppable" and (
            item.category in stoppable_categories or item.category in reducible_categories
        ):
            out.append(item)
    # Largest expenses first — resolves the deficit with the fewest changes.
    out.sort(key=lambda i: i.amount_home_ccy)
    return out


def try_resolve_with_spending_changes(
    base_forecast: Forecast,
    ledger: list[LedgerItem],
    payments: list[tuple],
    min_balance: float,
    profile: dict,
) -> AdjustmentResult:
    changes: list[SpendingChange] = []
    used_event_ids: set[str] = set()
    adjusted = base_forecast

    candidates = _adjustable_items(ledger, profile)

    for item in candidates:
        if len(changes) >= MAX_CHANGES:
            break
        is_safe, min_reached = verify_plan(adjusted, payments, min_balance)
        if is_safe:
            break
        if item.event_id in used_event_ids:
            continue

        stoppable_categories = set((profile.get("expense_categories_user_is_willing_to_stop") or "").split("|"))
        can_stop = item.flexibility in {"stoppable", "reducible_or_stoppable"} and item.category in stoppable_categories

        offset = (item.on_date - adjusted.as_of).days
        if offset < 0 or offset >= len(adjusted.dates):
            continue

        if can_stop:
            recovered = -item.amount_home_ccy  # remove the whole expense
            new_deltas = _shift(adjusted.balance_no_purchase, offset, recovered)
            changes.append(SpendingChange(action="stop", event_id=item.event_id))
        else:
            recovered = -item.amount_home_ccy * 0.5  # reduce by half as a conservative default
            new_deltas = _shift(adjusted.balance_no_purchase, offset, recovered)
            new_amount = round(-item.amount_home_ccy - recovered, 2)
            changes.append(SpendingChange(action="reduce_to", event_id=item.event_id, new_amount=new_amount))

        used_event_ids.add(item.event_id)
        adjusted = Forecast(
            as_of=adjusted.as_of,
            horizon_days=adjusted.horizon_days,
            dates=adjusted.dates,
            balance_no_purchase=new_deltas,
        )

    is_safe, min_reached = verify_plan(adjusted, payments, min_balance)
    return AdjustmentResult(is_safe=is_safe, min_balance_reached=min_reached, changes=changes, adjusted_forecast=adjusted)


def _shift(balances: list[float], from_offset: int, amount: float) -> list[float]:
    return [b + amount if i >= from_offset else b for i, b in enumerate(balances)]
