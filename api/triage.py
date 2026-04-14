"""
ML Triage Gate — routes transactions before AI Agent involvement.

Three tiers:
  AUTO_CLEAR   (score < LOW_THRESHOLD)   — pass through, no agent
  AGENT_REVIEW (LOW <= score < HIGH)     — enter AI agent investigation queue
  AUTO_SAR     (score >= HIGH_THRESHOLD) — auto-freeze and SAR escalation

Thresholds are configurable via environment variables.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Configurable thresholds — tune based on business precision/recall requirements.
#
# AML cost asymmetry:
#   FN (auto-cleared criminal)  → high cost, no recovery
#   FP (sent to AI review)      → low cost, AI clears it in seconds
#
# LOW_THRESHOLD is deliberately conservative (10 not 25): only transactions the
# model is very confident are clean bypass the review queue entirely.
# Raising this threshold increases throughput but also increases FN risk.
LOW_THRESHOLD  = float(os.environ.get("TRIAGE_LOW_THRESHOLD",  "10"))
HIGH_THRESHOLD = float(os.environ.get("TRIAGE_HIGH_THRESHOLD", "75"))


@dataclass
class TriageResult:
    transaction_id: str
    ml_score: float
    ml_risk_level: str
    routing: str            # AUTO_CLEAR | AGENT_REVIEW | AUTO_SAR
    routing_reason: str
    shap_attributions: list
    model_version: str
    scored_at: str

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "ml_score": self.ml_score,
            "ml_risk_level": self.ml_risk_level,
            "routing": self.routing,
            "routing_reason": self.routing_reason,
            "shap_attributions": self.shap_attributions,
            "model_version": self.model_version,
            "scored_at": self.scored_at,
        }


def triage(transaction: dict) -> TriageResult:
    """
    Score the transaction with XGBoost and return a routing decision.
    Does NOT call the AI Agent — pure ML inference.
    """
    transaction_id = transaction.get("transaction_id", "UNKNOWN")

    # Try trained XGBoost model first; fall back to heuristic scorer
    ml_result = _score(transaction, transaction_id)

    score        = float(ml_result["risk_score"])
    risk_level   = ml_result.get("risk_level", _level(score))
    shap_attrs   = ml_result.get("shap_attributions", [])
    model_ver    = ml_result.get("model_version", "heuristic")

    routing, reason = _route(score)

    return TriageResult(
        transaction_id  = transaction_id,
        ml_score        = round(score, 2),
        ml_risk_level   = risk_level,
        routing         = routing,
        routing_reason  = reason,
        shap_attributions = shap_attrs,
        model_version   = model_ver,
        scored_at       = datetime.now(timezone.utc).isoformat(),
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _score(transaction: dict, transaction_id: str) -> dict:
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_aml.pkl")
    if os.path.exists(model_path):
        try:
            from models.predict import predict_with_shap
            return predict_with_shap(transaction)
        except Exception as exc:
            logger.warning(f"[{transaction_id}] XGBoost inference failed, using heuristic: {exc}")

    # Heuristic fallback (same logic as agent/executors.py _run_risk_scorer)
    from agent.executors import _run_risk_scorer
    return _run_risk_scorer(transaction)


def _route(score: float) -> tuple[str, str]:
    if score < LOW_THRESHOLD:
        return (
            "AUTO_CLEAR",
            f"ML score {score:.1f} < {LOW_THRESHOLD} — below low-risk threshold, no suspicious indicators detected",
        )
    elif score >= HIGH_THRESHOLD:
        return (
            "AUTO_SAR",
            f"ML score {score:.1f} ≥ {HIGH_THRESHOLD} — exceeds high-risk threshold, automatic SAR escalation required",
        )
    else:
        return (
            "AGENT_REVIEW",
            (
                f"ML score {score:.1f} in medium-risk band [{LOW_THRESHOLD}, {HIGH_THRESHOLD}) "
                f"— ML model is uncertain; AI agent investigation required for multi-source reasoning"
            ),
        )


def _level(score: float) -> str:
    if score < 30:
        return "LOW"
    elif score < 60:
        return "MEDIUM"
    elif score < 80:
        return "HIGH"
    return "CRITICAL"
