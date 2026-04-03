"""POST /api/v1/analyze — run AML agent on a transaction."""

import logging
from fastapi import APIRouter, HTTPException

from api.schemas import TransactionInput, AgentResponse
from api.database import save_case
from agent.core import AMLAgent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AgentResponse)
def analyze_transaction(transaction: TransactionInput):
    try:
        agent = AMLAgent()
        tx_dict = transaction.model_dump()
        # Convert datetime to ISO string for the agent
        tx_dict["timestamp"] = tx_dict["timestamp"].isoformat()

        result = agent.analyze(tx_dict)

        # Extract SAR report from tool results if present
        sar_report = None
        for tr in result.tool_results:
            if tr.get("tool_name") == "sar_report_generator":
                raw = tr.get("result", {})
                sar_report = raw.get("sar_report")
                break

        # Persist to SQLite
        save_case(
            transaction_id=result.transaction_id,
            decision=result.final_decision,
            risk_score=result.risk_score,
            result=result.to_dict(),
        )

        return AgentResponse(
            transaction_id=result.transaction_id,
            final_decision=result.final_decision,
            risk_score=result.risk_score,
            reasoning_chain=result.reasoning_chain,
            tool_calls_count=len(result.tool_calls),
            total_tokens=result.total_tokens,
            duration_ms=result.duration_ms,
            sar_report=sar_report,
            error=result.error,
        )

    except ValueError as e:
        # e.g. missing API key
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error in analyze endpoint")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
