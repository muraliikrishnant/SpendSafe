"""Typed models shared across the pipeline.

Per CLAUDE.md, every financial fact carries provenance and a confidence/
verification state. AI-inferred facts (from messages or images) must never
silently become "confirmed" — only settled/explicit records from the
structured dataset are confirmed by default.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FactSource(str, Enum):
    USER_INPUT = "user_input"
    TRANSACTION_HISTORY = "transaction_history"
    UPLOADED_IMAGE = "uploaded_image"
    UPLOADED_DOCUMENT = "uploaded_document"
    USER_MESSAGE = "user_message"
    SYSTEM_CALCULATION = "system_calculation"
    AI_INFERENCE = "ai_inference"


class ExtractedFact(BaseModel):
    """One structured, provenance-tagged fact pulled from untrusted evidence
    (a message or an image) by the LLM extraction layer. Never treated as
    ground truth until validated against the rules in reconstruction.py.
    """

    fact_type: str  # e.g. "amount", "income_change", "cancellation", "confirmation"
    event_id: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    effective_date: Optional[date] = None
    description: str = ""
    source: FactSource
    confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = False
    raw_evidence_id: str = ""  # message_id or image_id this came from

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class PaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class Payment(BaseModel):
    pay_date: date
    amount: float


class SpendingChange(BaseModel):
    action: str  # "stop" | "reduce_to"
    event_id: str
    new_amount: Optional[float] = None

    def render(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{self.new_amount}"


class OutputRow(BaseModel):
    request_id: str
    amount_safe_to_pay: float
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: list[Payment] = Field(default_factory=list)
    earliest_date_for_full_payment: Optional[date] = None
    spending_changes_needed: list[SpendingChange] = Field(default_factory=list)
    decision_explanation: str

    def to_csv_row(self) -> dict:
        plan = (
            "|".join(f"{p.pay_date.isoformat()}:{_fmt_amount(p.amount)}" for p in self.payment_plan)
            if self.payment_plan
            else "none"
        )
        changes = (
            "|".join(c.render() for c in self.spending_changes_needed[:3])
            if self.spending_changes_needed
            else "none"
        )
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": _fmt_amount(self.amount_safe_to_pay),
            "affordability_status": self.affordability_status.value,
            "recommended_payment_method": self.recommended_payment_method.value,
            "payment_plan": plan,
            "earliest_date_for_full_payment": (
                self.earliest_date_for_full_payment.isoformat()
                if self.earliest_date_for_full_payment
                else ""
            ),
            "spending_changes_needed": changes,
            "decision_explanation": self.decision_explanation,
        }


def _fmt_amount(a: float) -> str:
    r = round(a, 2)
    return str(int(r)) if r == int(r) else f"{r:.2f}"


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
