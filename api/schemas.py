"""Pydantic v2 request/response schemas for AML-Shield API."""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str
    amount: float = Field(gt=0)
    transaction_type: Literal[
        "wire_transfer",
        "cash_deposit",
        "cash_withdrawal",
        "crypto_exchange",
        "internal_transfer",
        "card_payment",
    ]
    sender_account: str
    receiver_account: str
    sender_country: str = Field(min_length=2, max_length=2)
    receiver_country: str = Field(min_length=2, max_length=2)
    is_cross_border: bool
    timestamp: datetime
    reference: str = ""

    def model_post_init(self, __context):
        # Normalize country codes to uppercase
        object.__setattr__(self, "sender_country", self.sender_country.upper())
        object.__setattr__(self, "receiver_country", self.receiver_country.upper())


class AgentResponse(BaseModel):
    transaction_id: str
    final_decision: str
    risk_score: Optional[float]
    reasoning_chain: list[dict]
    tool_calls_count: int
    total_tokens: int
    duration_ms: int
    sar_report: Optional[dict]
    error: Optional[str]


class CaseSummary(BaseModel):
    transaction_id: str
    final_decision: str
    risk_score: Optional[float]
    analyzed_at: datetime
    duration_ms: int
