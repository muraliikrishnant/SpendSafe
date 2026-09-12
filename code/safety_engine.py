"""The deterministic safety engine.

This is the sole authority on affordability math (CLAUDE.md §2). It never
calls an LLM. Given a starting balance, a resolved 90-day ledger, and the
user's minimum balance, it computes:

- the day-by-day projected balance with no purchase,
- `amount_safe_to_pay` (max payable today without breaching the minimum),
- `earliest_date_for_full_payment` (first date a single full payment stays safe),
- and a generic `verify_plan` used to check any candidate payment schedule
  (partial payment, installments, or a plan with spending changes applied).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from reconstruction import LedgerItem


@dataclass
class Forecast:
    as_of: date
    horizon_days: int
    dates: list  # list[date], as_of .. as_of+horizon_days inclusive
    balance_no_purchase: list  # list[float], same length as dates

    def suffix_min(self) -> list:
        out = [0.0] * len(self.balance_no_purchase)
        running = float("inf")
        for i in range(len(self.balance_no_purchase) - 1, -1, -1):
            running = min(running, self.balance_no_purchase[i])
            out[i] = running
        return out


def build_forecast(
    starting_balance: float,
    ledger: list[LedgerItem],
    as_of: date,
    horizon_days: int = 90,
) -> Forecast:
    n_days = horizon_days + 1
    dates = [as_of + timedelta(days=i) for i in range(n_days)]
    deltas = [0.0] * n_days

    for item in ledger:
        offset = (item.on_date - as_of).days
        if 0 <= offset < n_days:
            deltas[offset] += item.amount_home_ccy

    balances = []
    running = starting_balance
    for d in deltas:
        running += d
        balances.append(running)

    return Forecast(as_of=as_of, horizon_days=horizon_days, dates=dates, balance_no_purchase=balances)


def amount_safe_to_pay(forecast: Forecast, min_balance: float, requested_amount: float) -> float:
    worst_case = min(forecast.balance_no_purchase)
    safe = worst_case - min_balance
    return max(0.0, min(safe, requested_amount))


def earliest_date_for_full_payment(
    forecast: Forecast, min_balance: float, requested_amount: float
) -> date | None:
    suffix_min = forecast.suffix_min()
    threshold = min_balance + requested_amount
    for i, d in enumerate(forecast.dates):
        if suffix_min[i] >= threshold:
            return d
    return None


def verify_plan(
    forecast: Forecast,
    payments: list[tuple[date, float]],
    min_balance: float,
) -> tuple[bool, float]:
    """Apply a candidate payment schedule on top of the no-purchase forecast
    and confirm the balance never breaches `min_balance` at any point in the
    forecast window. Returns (is_safe, minimum_balance_reached)."""
    n = len(forecast.dates)
    extra_delta = [0.0] * n
    for pay_date, amount in payments:
        offset = (pay_date - forecast.as_of).days
        if offset < 0:
            return False, float("-inf")
        if offset >= n:
            # Payment falls after the forecast window; cannot be verified safe.
            continue
        extra_delta[offset] -= amount

    running_extra = 0.0
    min_balance_reached = float("inf")
    for i in range(n):
        running_extra += extra_delta[i]
        total = forecast.balance_no_purchase[i] + running_extra
        min_balance_reached = min(min_balance_reached, total)

    return min_balance_reached >= min_balance, min_balance_reached
