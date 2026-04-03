# AML Agent core loop
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AgentResult:
    transaction_id: str
    final_decision: str
    risk_score: Optional[float]
    reasoning_chain: list
    tool_calls: list
    tool_results: list
    final_text: str
    total_tokens: int
    duration_ms: int
    error: Optional[str]

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "final_decision": self.final_decision,
            "risk_score": self.risk_score,
            "reasoning_chain": self.reasoning_chain,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "final_text": self.final_text,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class AMLAgent:
    def analyze(self, transaction: dict) -> AgentResult:
        raise NotImplementedError
