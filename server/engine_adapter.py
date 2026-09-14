"""Adapter: a manually-entered profile -> the real SpendSafe safety engine.

The hackathon pipeline reconstructs a ledger from financial_events.csv. A
person using the web app types their bills in instead. Only that front half
differs — everything that decides anything (the 90-day forecast, the safe
amount, the earliest safe date, plan verification, flexible-spending trims
with their floors) is imported from code/ and runs unmodified, so the web
app and the batch pipeline cannot drift apart on the math.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import safety_engine as se          # noqa: E402  the audited engine, untouched
import spending_changes as sc       # noqa: E402
from reconstruction import LedgerItem  # noqa: E402

HORIZON_DAYS = 90
INSTALLMENT_SPLITS = (3, 6, 12)
INSTALLMENT_GAP_DAYS = 30

FLEX_STOPPABLE = "stoppable"
FLEX_REDUCIBLE = "reducible"


@dataclass
class Decision:
    status: str
    method: str | None
    payments: list[tuple[date, float]]
    amount_safe_today: float
    earliest_full_date: date | None
    spending_changes: list[dict]
    baseline_balances: list[float]
    plan_balances: list[float]
    dates: list[date]
    min_balance: float
    requested_amount: float


def _occurrences(first: date, every_days: int, as_of: date, end: date) -> list[date]:
    """Every occurrence of a repeating item inside the window."""
    every_days = max(1, int(every_days or 30))
    out, cursor, guard = [], first, 0
    while cursor <= end and guard < 500:
        if cursor >= as_of:
            out.append(cursor)
        cursor += timedelta(days=every_days)
        guard += 1
    return out


def build_ledger(profile: dict, as_of: date) -> list[LedgerItem]:
    """Manual profile -> the same LedgerItem list the batch pipeline builds."""
    end = as_of + timedelta(days=HORIZON_DAYS)
    items: list[LedgerItem] = []

    for inc in profile.get("income", []):
        for when in _occurrences(_as_date(inc["nextDate"]), inc.get("everyDays", 30), as_of, end):
            items.append(LedgerItem(
                on_date=when, amount_home_ccy=abs(float(inc.get("amount") or 0)),
                event_id=str(inc.get("id") or inc.get("label")), category=str(inc.get("label") or "income"),
                flexibility="fixed", projected=False,
                real_event_id=str(inc.get("id") or inc.get("label")),
            ))

    for bill in profile.get("expenses", []):
        flex = bill.get("flexibility", "fixed")
        floor = float(bill.get("floor") or 0) or None
        ident = str(bill.get("id") or bill.get("label"))
        for when in _occurrences(_as_date(bill["nextDate"]), bill.get("everyDays", 30), as_of, end):
            items.append(LedgerItem(
                on_date=when, amount_home_ccy=-abs(float(bill.get("amount") or 0)),
                event_id=ident, category=ident, flexibility=flex, projected=False,
                real_event_id=ident, min_allowed=floor,
            ))

    for one in profile.get("oneOffs", []):
        when = _as_date(one["date"])
        if as_of <= when <= end:
            ident = str(one.get("id") or one.get("label"))
            items.append(LedgerItem(
                on_date=when, amount_home_ccy=-abs(float(one.get("amount") or 0)),
                event_id=ident, category=ident, flexibility="fixed", projected=False,
                real_event_id=ident,
            ))

    items.sort(key=lambda i: i.on_date)
    return items


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _flex_profile(profile: dict) -> dict:
    """spending_changes reads the user's willingness from pipe-joined category
    lists (the dataset's shape). Each manual bill is its own category, so a
    bill is adjustable exactly when the person marked it so."""
    stop, reduce = [], []
    for bill in profile.get("expenses", []):
        ident = str(bill.get("id") or bill.get("label"))
        if bill.get("flexibility") == FLEX_STOPPABLE:
            stop.append(ident)
        elif bill.get("flexibility") == FLEX_REDUCIBLE:
            reduce.append(ident)
    return {
        "expense_categories_user_is_willing_to_stop": "|".join(stop),
        "expense_categories_user_is_willing_to_reduce": "|".join(reduce),
    }


def _label_for(profile: dict, event_id: str) -> str:
    for bill in profile.get("expenses", []):
        if str(bill.get("id") or bill.get("label")) == event_id:
            return str(bill.get("label") or event_id)
    return event_id


def decide(profile: dict, requested_amount: float, by_date: date | None = None,
           as_of: date | None = None) -> Decision:
    as_of = as_of or date.today()
    requested_amount = max(0.0, float(requested_amount))
    min_balance = float(profile.get("minBalance") or 0)
    starting = float(profile.get("balance") or 0)

    ledger = build_ledger(profile, as_of)
    forecast = se.build_forecast(starting, ledger, as_of, HORIZON_DAYS)

    safe_today = se.amount_safe_to_pay(forecast, min_balance, requested_amount)
    earliest = se.earliest_date_for_full_payment(forecast, min_balance, requested_amount)
    deadline_offset = (by_date - as_of).days if by_date else None

    def within_deadline(offset: int) -> bool:
        return deadline_offset is None or offset <= deadline_offset

    candidates = []

    full = [(as_of, requested_amount)]
    if se.verify_plan(forecast, full, min_balance)[0]:
        candidates.append(dict(method="full_payment", payments=full, total=requested_amount,
                               start=0, n=1, ok=within_deadline(0)))

    for n in INSTALLMENT_SPLITS:
        each = round(requested_amount / n, 2)
        pays = [(as_of + timedelta(days=k * INSTALLMENT_GAP_DAYS), each) for k in range(n)]
        last = (pays[-1][0] - as_of).days
        if last > HORIZON_DAYS:
            continue
        if se.verify_plan(forecast, pays, min_balance)[0]:
            candidates.append(dict(method="installments", payments=pays, total=round(each * n, 2),
                                   start=0, n=n, ok=within_deadline(last)))

    if 0 < safe_today < requested_amount and earliest is not None:
        rest = round(requested_amount - safe_today, 2)
        pays = [(as_of, round(safe_today, 2)), (earliest, rest)]
        if within_deadline((earliest - as_of).days) and se.verify_plan(forecast, pays, min_balance)[0]:
            candidates.append(dict(method="partial_payment", payments=pays, total=requested_amount,
                                   start=0, n=2, ok=True))

    if earliest is not None and earliest > as_of:
        off = (earliest - as_of).days
        candidates.append(dict(method="wait", payments=[(earliest, requested_amount)],
                               total=requested_amount, start=off, n=1, ok=within_deadline(off)))

    candidates.sort(key=lambda c: (not c["ok"], c["method"] == "wait", c["total"], c["start"], c["n"]))

    chosen = candidates[0] if candidates else None
    changes: list[dict] = []
    plan_forecast = forecast

    # Nothing safe outright (or nothing that meets the deadline): see whether
    # trimming flexible bills rescues it — same routine the pipeline uses.
    if chosen is None or not chosen["ok"]:
        attempts = [full] + [
            [(as_of + timedelta(days=k * INSTALLMENT_GAP_DAYS), round(requested_amount / n, 2))
             for k in range(n)]
            for n in INSTALLMENT_SPLITS
        ]
        flex_profile = _flex_profile(profile)
        for attempt in attempts:
            if (attempt[-1][0] - as_of).days > HORIZON_DAYS:
                continue
            result = sc.try_resolve_with_spending_changes(
                forecast, ledger, attempt, min_balance, flex_profile)
            if result.is_safe:
                chosen = dict(method="full_payment" if len(attempt) == 1 else "installments",
                              payments=attempt, total=round(sum(a for _, a in attempt), 2),
                              start=0, n=len(attempt),
                              ok=within_deadline((attempt[-1][0] - as_of).days))
                changes = [
                    {"action": c.action, "event_id": c.event_id,
                     "label": _label_for(profile, c.event_id), "new_amount": c.new_amount}
                    for c in result.changes
                ]
                plan_forecast = result.adjusted_forecast
                break

    # Final gate: re-verify whatever won, against the forecast actually in force.
    if chosen is not None and chosen["method"] != "wait":
        if not se.verify_plan(plan_forecast, chosen["payments"], min_balance)[0]:
            chosen, changes, plan_forecast = None, [], forecast

    if chosen is None:
        status = "not_affordable"
    elif chosen["method"] == "full_payment" and not changes:
        status = "affordable_now"
    elif chosen["method"] == "wait":
        status = "affordable_later"
    else:
        status = "affordable_with_plan"

    plan_balances = _apply(plan_forecast, chosen["payments"] if chosen else [], as_of)

    return Decision(
        status=status,
        method=chosen["method"] if chosen else "not_recommended",
        payments=chosen["payments"] if chosen else [],
        amount_safe_today=round(safe_today, 2),
        earliest_full_date=earliest,
        spending_changes=changes,
        baseline_balances=list(forecast.balance_no_purchase),
        plan_balances=plan_balances,
        dates=list(forecast.dates),
        min_balance=min_balance,
        requested_amount=requested_amount,
    )


def _apply(forecast, payments, as_of: date) -> list[float]:
    balances = list(forecast.balance_no_purchase)
    running = 0.0
    deltas = [0.0] * len(balances)
    for when, amount in payments:
        off = (when - as_of).days
        if 0 <= off < len(deltas):
            deltas[off] -= amount
    out = []
    for i, base in enumerate(balances):
        running += deltas[i]
        out.append(base + running)
    return out
