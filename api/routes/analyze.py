"""POST /api/v1/analyze — ML triage gate + conditional AI agent investigation."""

import time
import logging
from fastapi import APIRouter, HTTPException

from api.schemas import TransactionInput, AnalysisResponse
from api.database import save_case
from api.triage import triage, LOW_THRESHOLD, HIGH_THRESHOLD

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_transaction(transaction: TransactionInput):
    start_ms = time.time()

    tx_dict = transaction.model_dump()
    tx_dict["timestamp"] = tx_dict["timestamp"].isoformat()

    try:
        # ── Step 1: ML triage ──────────────────────────────────────────────────
        triage_result = triage(tx_dict)

        routing        = triage_result.routing
        ml_score       = triage_result.ml_score
        ml_risk_level  = triage_result.ml_risk_level
        shap_attrs     = triage_result.shap_attributions
        model_version  = triage_result.model_version
        routing_reason = triage_result.routing_reason

        reasoning_chain  = []
        tool_calls_count = 0
        total_tokens     = 0
        sar_report       = None
        error            = None

        # ── Step 2: Route ──────────────────────────────────────────────────────
        if routing == "AUTO_CLEAR":
            # Below low threshold — pass through without agent
            final_decision = "CLEAR"
            logger.info(
                f"[{triage_result.transaction_id}] AUTO_CLEAR "
                f"(ml_score={ml_score:.1f} < {LOW_THRESHOLD})"
            )

        elif routing == "AUTO_SAR":
            # Above high threshold — auto-escalate without agent
            final_decision = "SAR_REQUIRED"
            logger.info(
                f"[{triage_result.transaction_id}] AUTO_SAR "
                f"(ml_score={ml_score:.1f} ≥ {HIGH_THRESHOLD})"
            )

        else:
            # ── Step 3: Medium risk → AI Agent investigation ───────────────────
            logger.info(
                f"[{triage_result.transaction_id}] AGENT_REVIEW "
                f"(ml_score={ml_score:.1f} in [{LOW_THRESHOLD}, {HIGH_THRESHOLD}))"
            )

            from agent.core import AMLAgent
            agent = AMLAgent()

            ml_context = {
                "score":            ml_score,
                "risk_level":       ml_risk_level,
                "shap_attributions": shap_attrs,
                "routing_reason":   routing_reason,
                "low_threshold":    LOW_THRESHOLD,
                "high_threshold":   HIGH_THRESHOLD,
            }

            agent_result = agent.analyze(tx_dict, ml_context=ml_context)

            final_decision   = agent_result.final_decision
            reasoning_chain  = agent_result.reasoning_chain
            tool_calls_count = len(agent_result.tool_calls)
            total_tokens     = agent_result.total_tokens
            error            = agent_result.error

            # Extract SAR report if present
            for tr in agent_result.tool_results:
                if tr.get("tool_name") == "sar_report_generator":
                    sar_report = tr.get("result", {}).get("sar_report")
                    break

        duration_ms = int((time.time() - start_ms) * 1000)

        response = AnalysisResponse(
            transaction_id   = triage_result.transaction_id,
            routing          = routing,
            ml_score         = ml_score,
            ml_risk_level    = ml_risk_level,
            ml_routing_reason = routing_reason,
            shap_attributions = shap_attrs,
            model_version    = model_version,
            final_decision   = final_decision,
            reasoning_chain  = reasoning_chain,
            tool_calls_count = tool_calls_count,
            total_tokens     = total_tokens,
            sar_report       = sar_report,
            duration_ms      = duration_ms,
            error            = error,
        )

        save_case(
            transaction_id = triage_result.transaction_id,
            decision       = final_decision,
            routing        = routing,
            risk_score     = ml_score,
            result         = response.model_dump(),
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error in analyze endpoint")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
