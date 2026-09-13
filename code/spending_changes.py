"""Greedy flexible-spending adjustment: only used when a plan is otherwise
unsafe. Only ever touches events already marked flexible in the data AND in
a category the user has explicitly said they're willing to reduce/stop
(CLAUDE.md §9 — essential spending is never touched, and the user stays in
control of which categories are adjustable).

A recurring expense can appear as several LedgerItem occurrences within the
90-day window (one real row plus several forecasted ones, or several
forecasted ones alone). All occurrences sharing the same `real_event_id`
are treated as ONE adjustable commitment: stopping/reducing it affects
every future occurrence's dollar impact, and emits exactly one spending
change referencing that real, gradeable event_id — never a fabricated
per-occurrence id.
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


def _adjustable_groups(ledger: list[LedgerItem], profile: dict) -> list[list[LedgerItem]]:
    stoppable_categories = set((profile.get("expense_categories_user_is_willing_to_stop") or "").split("|"))
    reducible_categories = set((profile.get("expense_categories_user_is_willing_to_reduce") or "").split("|"))

    groups: dict[str, list[LedgerItem]] = {}
    for item in ledger:
        if item.amount_home_ccy >= 0:
            continue  # only expenses can be reduced/stopped
        eligible = (
            (item.flexibility == "stoppable" and item.category in stoppable_categories)
            or (item.flexibility == "reducible" and item.category in reducible_categories)
            or (
                item.flexibility == "reducible_or_stoppable"
                and (item.category in stoppable_categories or item.category in reducible_categories)
            )
        )
        if not eligible:
            continue
        groups.setdefault(item.real_event_id, []).append(item)

    # Largest total future impact first — resolves the deficit with the
    # fewest distinct spending changes.
    return sorted(groups.values(), key=lambda items: sum(i.amount_home_ccy for i in items))


def try_resolve_with_spending_changes(
    base_forecast: Forecast,
    ledger: list[LedgerItem],
    payments: list[tuple],
    min_balance: float,
    profile: dict,
) -> AdjustmentResult:
    changes: list[SpendingChange] = []
    used_real_ids: set[str] = set()
    adjusted = base_forecast

    stoppable_categories = set((profile.get("expense_categories_user_is_willing_to_stop") or "").split("|"))
    groups = _adjustable_groups(ledger, profile)

    for group in groups:
        if len(changes) >= MAX_CHANGES:
            break
        is_safe, _ = verify_plan(adjusted, payments, min_balance)
        if is_safe:
            break

        real_id = group[0].real_event_id
        if real_id in used_real_ids:
            continue

        category = group[0].category
        flexibility = group[0].flexibility
        can_stop = flexibility in {"stoppable", "reducible_or_stoppable"} and category in stoppable_categories

        new_deltas = adjusted.balance_no_purchase
        representative_new_amount = None
        affected_any = False

        for item in group:
            offset = (item.on_date - adjusted.as_of).days
            if offset < 0 or offset >= len(adjusted.dates):
                continue
            spend = -item.amount_home_ccy  # positive magnitude of the expense
            if can_stop:
                recovered = spend  # remove the whole expense
            else:
                # Halve it, but never below the event's own
                # minimum_allowed_amount — the dataset states a floor for every
                # reducible event, and a naive 50% cut breaches it often.
                target = spend * 0.5
                if item.min_allowed is not None:
                    target = max(target, item.min_allowed)
                if target >= spend - 0.01:
                    continue  # already at its floor — no headroom to reduce
                recovered = spend - target
                representative_new_amount = round(target, 2)
            affected_any = True
            new_deltas = _shift(new_deltas, offset, recovered)

        if not affected_any:
            continue  # no occurrence in-window with any headroom to change

        if can_stop:
            changes.append(SpendingChange(action="stop", event_id=real_id))
        else:
            if representative_new_amount is None:
                continue
            changes.append(SpendingChange(action="reduce_to", event_id=real_id, new_amount=representative_new_amount))

        used_real_ids.add(real_id)
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
