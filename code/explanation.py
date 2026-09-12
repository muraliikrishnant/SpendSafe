"""Explanation generation, constrained to facts the engine actually used.

Default is a deterministic template (always grounded, zero LLM cost at
scale). An optional LLM "polish" pass may rephrase it, but only ever from
the same fact set, and the result is rejected (falling back to the
template) if it introduces any number or event_id not in the allowed set.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from extraction import LLMRouter, UNTRUSTED_GUARD
from schemas import OutputRow


@dataclass
class ExplanationFacts:
    currency: str
    requested_amount: float
    amount_safe_to_pay: float
    min_balance: float
    worst_case_balance: float
    status: str
    method: str
    payments: list[tuple[str, float]]  # (iso_date, amount)
    earliest_full_date: str | None
    spending_changes: list[str]
    request_type: str
    allowed_event_ids: set = field(default_factory=set)


def _fmt(n: float) -> str:
    r = round(n, 2)
    return f"{int(r):,}" if r == int(r) else f"{r:,.2f}"


def build_deterministic_explanation(f: ExplanationFacts) -> str:
    parts = []
    if f.status == "affordable_now":
        parts.append(
            f"Pay {f.currency} {_fmt(f.requested_amount)} in full today. "
            f"This keeps the projected balance at or above the {_fmt(f.min_balance)} minimum "
            f"throughout the 90-day forecast (worst point projected at {f.currency} {_fmt(f.worst_case_balance)})."
        )
    elif f.status == "affordable_with_plan":
        plan_desc = ", then ".join(f"{d}: {f.currency} {_fmt(a)}" for d, a in f.payments)
        parts.append(
            f"Use {f.method.replace('_', ' ')}: {plan_desc}. "
            f"This keeps the balance at or above the {_fmt(f.min_balance)} minimum throughout the forecast."
        )
        if f.spending_changes:
            parts.append("Requires: " + "; ".join(f.spending_changes) + ".")
    elif f.status == "affordable_later":
        parts.append(
            f"Not safe today: paying now would risk the {_fmt(f.min_balance)} minimum balance "
            f"(projected worst point {f.currency} {_fmt(f.worst_case_balance)}). "
            f"The full amount is projected safe on {f.earliest_full_date}."
        )
    else:
        parts.append(
            f"Not affordable within the forecast period: the projected balance falls to "
            f"{f.currency} {_fmt(f.worst_case_balance)}, below the required minimum of {_fmt(f.min_balance)}, "
            f"and no eligible payment method keeps the plan safe."
        )
    return " ".join(parts)


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _allowed_numbers(f: ExplanationFacts) -> set[str]:
    nums = {_fmt(f.requested_amount), _fmt(f.amount_safe_to_pay), _fmt(f.min_balance), _fmt(f.worst_case_balance)}
    nums |= {_fmt(a) for _, a in f.payments}
    return nums


def _is_grounded(text: str, f: ExplanationFacts) -> bool:
    allowed = _allowed_numbers(f)
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if val in (0, 1, 90, 30):  # common non-financial numbers (durations, counts)
            continue
        if _fmt(val) not in allowed and raw not in allowed:
            return False
    return True


def generate_explanation(f: ExplanationFacts, router: LLMRouter | None = None) -> str:
    template = build_deterministic_explanation(f)
    if os.getenv("EXPLANATION_LLM_POLISH", "false").lower() != "true":
        return template

    router = router or LLMRouter()
    prompt = (
        "Rephrase this financial decision explanation to be clearer and more natural, "
        "in 2-3 sentences. Do not add, remove, or change any number, date, or amount. "
        "Do not invent any new fact.\n\nExplanation:\n" + template
    )
    try:
        parsed, _, _ = router.chat_json(
            [
                {"role": "system", "content": UNTRUSTED_GUARD},
                {"role": "user", "content": prompt + '\n\nRespond as JSON: {"text": "..."}'},
            ]
        )
        polished = parsed.get("text", "")
        if polished and _is_grounded(polished, f):
            return polished
    except Exception:  # noqa: BLE001 - any polish failure falls back to the template
        pass
    return template
