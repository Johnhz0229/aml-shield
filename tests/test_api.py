"""
API tests for AML-Shield FastAPI application.
Tests cover all three routing tiers: AUTO_CLEAR, AGENT_REVIEW, AUTO_SAR.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import app
from api.database import init_db
from agent.core import AgentResult

init_db()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _mock_triage(routing: str, score: float):
    """Return a TriageResult mock for a given routing tier."""
    from api.triage import TriageResult
    return TriageResult(
        transaction_id   = "TX-API-TEST-001",
        ml_score         = score,
        ml_risk_level    = "MEDIUM",
        routing          = routing,
        routing_reason   = f"Test routing: {routing}",
        shap_attributions = [],
        model_version    = "test-heuristic",
        scored_at        = datetime.now(timezone.utc).isoformat(),
    )


def _mock_agent_result(decision: str = "WATCHLIST", risk_score: float = 45.0) -> AgentResult:
    return AgentResult(
        transaction_id  = "TX-API-TEST-001",
        final_decision  = decision,
        risk_score      = risk_score,
        reasoning_chain = [
            {"step": 1, "type": "reasoning", "content": "Investigating medium-risk case..."},
            {"step": 1, "type": "tool_call", "tool_name": "entity_network_analyzer", "input": {}},
            {"step": 1, "type": "tool_result", "tool_name": "entity_network_analyzer",
             "result": {"network_risk_level": "medium"}},
        ],
        tool_calls  = [
            {"step": 1, "tool_name": "entity_network_analyzer", "tool_use_id": "tu_001", "input": {}}
        ],
        tool_results = [
            {"step": 1, "tool_name": "entity_network_analyzer",
             "tool_use_id": "tu_001", "result": {"network_risk_level": "medium"}}
        ],
        final_text   = f"DECISION: {decision}\nML Score: {risk_score}",
        total_tokens = 450,
        duration_ms  = 1200,
        error        = None,
    )


VALID_TRANSACTION = {
    "transaction_id":   "TX-API-TEST-001",
    "amount":           5000.00,
    "transaction_type": "wire_transfer",
    "sender_account":   "DE89370400440532013000",
    "receiver_account": "PK-BANK-123456",
    "sender_country":   "DE",
    "receiver_country": "PK",
    "is_cross_border":  True,
    "timestamp":        "2024-03-15T14:00:00Z",
    "reference":        "invoice payment",
}


# ---------------------------------------------------------------------------
# Test 1: Health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "uptime_seconds" in data
    assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Test 2: AUTO_CLEAR path — agent must NOT be called
# ---------------------------------------------------------------------------

def test_auto_clear_skips_agent(client):
    """Low ML score → AUTO_CLEAR, agent is never invoked."""
    mock_triage_result = _mock_triage("AUTO_CLEAR", score=12.0)

    with patch("api.routes.analyze.triage", return_value=mock_triage_result), \
         patch("agent.core.AMLAgent") as MockAgent:

        response = client.post("/api/v1/analyze", json=VALID_TRANSACTION)

    assert response.status_code == 200
    data = response.json()
    assert data["routing"] == "AUTO_CLEAR"
    assert data["final_decision"] == "CLEAR"
    assert data["tool_calls_count"] == 0
    assert data["total_tokens"] == 0

    # Agent must not have been instantiated
    MockAgent.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: AUTO_SAR path — agent must NOT be called
# ---------------------------------------------------------------------------

def test_auto_sar_skips_agent(client):
    """High ML score → AUTO_SAR, agent is never invoked."""
    mock_triage_result = _mock_triage("AUTO_SAR", score=88.0)

    with patch("api.routes.analyze.triage", return_value=mock_triage_result), \
         patch("agent.core.AMLAgent") as MockAgent:

        tx = dict(VALID_TRANSACTION)
        tx["transaction_id"] = "TX-AUTO-SAR-001"
        response = client.post("/api/v1/analyze", json=tx)

    assert response.status_code == 200
    data = response.json()
    assert data["routing"] == "AUTO_SAR"
    assert data["final_decision"] == "SAR_REQUIRED"
    assert data["tool_calls_count"] == 0

    MockAgent.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: AGENT_REVIEW path — agent IS called with ml_context
# ---------------------------------------------------------------------------

def test_agent_review_calls_agent_with_ml_context(client):
    """Medium ML score → AGENT_REVIEW, agent receives ml_context."""
    mock_triage_result = _mock_triage("AGENT_REVIEW", score=52.0)
    mock_agent_result  = _mock_agent_result("WATCHLIST", 52.0)

    captured_ml_context = {}

    def fake_analyze(tx, ml_context=None):
        captured_ml_context.update(ml_context or {})
        return mock_agent_result

    with patch("api.routes.analyze.triage", return_value=mock_triage_result), \
         patch("agent.core.AMLAgent") as MockAgent:

        mock_instance = MagicMock()
        MockAgent.return_value = mock_instance
        mock_instance.analyze.side_effect = fake_analyze

        tx = dict(VALID_TRANSACTION)
        tx["transaction_id"] = "TX-AGENT-REVIEW-001"
        response = client.post("/api/v1/analyze", json=tx)

    assert response.status_code == 200
    data = response.json()
    assert data["routing"] == "AGENT_REVIEW"
    assert data["final_decision"] == "WATCHLIST"
    assert data["tool_calls_count"] > 0

    # Verify agent received ml_context with the score
    assert "score" in captured_ml_context
    assert captured_ml_context["score"] == 52.0


# ---------------------------------------------------------------------------
# Test 5: Missing required field → 422
# ---------------------------------------------------------------------------

def test_analyze_missing_field(client):
    incomplete = {k: v for k, v in VALID_TRANSACTION.items() if k != "amount"}
    response = client.post("/api/v1/analyze", json=incomplete)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 6: Cases endpoint — routing field is returned
# ---------------------------------------------------------------------------

def test_cases_endpoint_includes_routing(client):
    mock_triage_result = _mock_triage("AUTO_SAR", score=90.0)

    with patch("api.routes.analyze.triage", return_value=mock_triage_result), \
         patch("agent.core.AMLAgent"):

        tx = dict(VALID_TRANSACTION)
        tx["transaction_id"] = "TX-CASES-TEST-001"
        mock_triage_result.transaction_id = "TX-CASES-TEST-001"
        client.post("/api/v1/analyze", json=tx)

    response = client.get("/api/v1/cases")
    assert response.status_code == 200
    data = response.json()
    ids = [c["transaction_id"] for c in data]
    assert "TX-CASES-TEST-001" in ids

    case = next(c for c in data if c["transaction_id"] == "TX-CASES-TEST-001")
    assert "routing" in case
