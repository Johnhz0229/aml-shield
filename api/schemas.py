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
        object.__setattr__(self, "sender_country", self.sender_country.upper())
        object.__setattr__(self, "receiver_country", self.receiver_country.upper())


class AnalysisResponse(BaseModel):
    """Unified response for all three routing tiers."""

    transaction_id: str

    # ── ML triage layer ────────────────────────────────────────────────────────
    routing: str            # AUTO_CLEAR | AGENT_REVIEW | AUTO_SAR
    ml_score: float
    ml_risk_level: str
    ml_routing_reason: str
    shap_attributions: list[dict]
    model_version: str

    # ── Final decision ─────────────────────────────────────────────────────────
    # AUTO_CLEAR  → CLEAR
    # AUTO_SAR    → SAR_REQUIRED
    # AGENT_REVIEW → whatever the agent decides
    final_decision: str

    # ── Agent output (only populated for AGENT_REVIEW) ─────────────────────────
    reasoning_chain: list[dict] = Field(default_factory=list)
    tool_calls_count: int = 0
    total_tokens: int = 0
    sar_report: Optional[dict] = None

    duration_ms: int
    error: Optional[str] = None


class CaseSummary(BaseModel):
    transaction_id: str
    routing: Optional[str] = None
    final_decision: str
    risk_score: Optional[float]
    analyzed_at: datetime
    duration_ms: int


# Legacy alias — kept for any code that still imports AgentResponse
AgentResponse = AnalysisResponse
