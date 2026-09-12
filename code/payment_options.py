"""Payment-option eligibility and the "choosing between safe plans" ranking
from the problem statement's Selection Hierarchy. Pure deterministic logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from data_loader import _parse_date


@dataclass
class InstallmentOption:
    payment_option_id: str
    request_id: str
    payments: list[tuple[date, float]]
    total_payable_amount: float
    number_of_payments: int

    def span_days(self) -> int:
        if len(self.payments) < 2:
            return 0
        return (self.payments[-1][0] - self.payments[0][0]).days


def build_installment_options(raw_options: list[dict]) -> list[InstallmentOption]:
    out = []
    for opt in raw_options:
        if opt["payment_method"] != "installments":
            continue
        n = int(opt["number_of_payments"])
        first = _parse_date(opt["first_payment_date"])
        freq = int(opt["payment_frequency_days"]) if opt.get("payment_frequency_days") else 0
        amount = float(opt["payment_amount"])
        payments = [(first + timedelta(days=freq * i), amount) for i in range(n)]
        out.append(
            InstallmentOption(
                payment_option_id=opt["payment_option_id"],
                request_id=opt["request_id"],
                payments=payments,
                total_payable_amount=float(opt["total_payable_amount"]),
                number_of_payments=n,
            )
        )
    return out


def user_accepts_method(profile: dict, method: str) -> bool:
    allowed = (profile.get("payment_methods_user_will_consider") or "").split("|")
    return method in allowed


def installment_within_user_limit(profile: dict, option: InstallmentOption) -> bool:
    max_months = profile.get("max_installment_months")
    if not max_months:
        return False  # blank => user will not consider installments at all
    max_months = int(max_months)
    span_months = max(1, round(option.span_days() / 30))
    return span_months <= max_months


def _option_id_sort_key(payment_option_id: str) -> tuple:
    # Sort "payment_option_01" < "payment_option_02" numerically, not lexically.
    digits = "".join(ch for ch in payment_option_id if ch.isdigit())
    return (int(digits) if digits else 0, payment_option_id)


@dataclass
class CandidatePlan:
    method: str  # full_payment | partial_payment | installments
    payments: list[tuple[date, float]]
    completes_by_deadline: bool
    requires_spending_changes: bool
    total_paid: float
    start_date: date
    n_payments: int
    payment_option_id: str = ""  # only for installments, used as final tiebreaker


def rank_candidates(candidates: list[CandidatePlan]) -> list[CandidatePlan]:
    """Selection Hierarchy: complete by deadline > no spending changes >
    min total paid > earlier start > fewer payments > lowest payment_option_id."""
    return sorted(
        candidates,
        key=lambda c: (
            0 if c.completes_by_deadline else 1,
            0 if not c.requires_spending_changes else 1,
            c.total_paid,
            c.start_date,
            c.n_payments,
            _option_id_sort_key(c.payment_option_id) if c.payment_option_id else (0, ""),
        ),
    )
