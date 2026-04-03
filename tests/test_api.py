"""
API tests for AML-Shield FastAPI application.
Uses httpx.AsyncClient with ASGI transport to avoid spawning a real server.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import httpx
from fastapi.testclient import TestClient

from api.main import app
from api.database import init_db
from agent.core import AgentResult


# Initialize DB for tests
init_db()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Synchronous test client — avoids asyncio complexity for most tests."""
    with TestClient(app) as c:
        yield c


def _mock_agent_result(decision: str = "WATCHLIST", risk_score: float = 45.0) -> AgentResult:
    """Return a realistic AgentResult for mocking AMLAgent.analyze."""
    return AgentResult(
        transaction_id="TX-API-TEST-001",
        final_decision=decision,
        risk_score=risk_score,
        reasoning_chain=[
            {"step": 1, "type": "reasoning", "content": "Analyzing transaction..."},
            {"step": 1, "type": "tool_call", "tool_name": "transaction_risk_scorer", "input": {}},
            {"step": 1, "type": "tool_result", "tool_name": "transaction_risk_scorer", "result": {"risk_score": risk_score}},
        ],
        tool_calls=[
            {"step": 1, "tool_name": "transaction_risk_scorer", "tool_use_id": "tu_001", "input": {}}
        ],
        tool_results=[
            {"step": 1, "tool_name": "transaction_risk_scorer", "tool_use_id": "tu_001", "result": {"risk_score": risk_score}}
        ],
        final_text=f"DECISION: {decision}\nRisk Score: {risk_score}",
        total_tokens=450,
        duration_ms=1200,
        error=None,
    )


VALID_TRANSACTION = {
    "transaction_id": "TX-API-TEST-001",
    "amount": 5000.00,
    "transaction_type": "wire_transfer",
    "sender_account": "DE89370400440532013000",
    "receiver_account": "PK-BANK-123456",
    "sender_country": "DE",
    "receiver_country": "PK",
    "is_cross_border": True,
    "timestamp": "2024-03-15T14:00:00Z",
    "reference": "invoice payment",
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
# Test 2: Analyze valid transaction
# ---------------------------------------------------------------------------

def test_analyze_valid_transaction(client):
    mock_result = _mock_agent_result("WATCHLIST", 45.0)

    with patch("api.routes.analyze.AMLAgent") as MockAgent:
        mock_instance = MagicMock()
        MockAgent.return_value = mock_instance
        mock_instance.analyze.return_value = mock_result

        response = client.post("/api/v1/analyze", json=VALID_TRANSACTION)

    assert response.status_code == 200
    data = response.json()
    assert "final_decision" in data
    assert data["final_decision"] == "WATCHLIST"
    assert data["transaction_id"] == "TX-API-TEST-001"


# ---------------------------------------------------------------------------
# Test 3: Missing required field → 422 validation error
# ---------------------------------------------------------------------------

def test_analyze_missing_field(client):
    incomplete = {k: v for k, v in VALID_TRANSACTION.items() if k != "amount"}
    response = client.post("/api/v1/analyze", json=incomplete)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4: Cases endpoint returns list after an analyze
# ---------------------------------------------------------------------------

def test_cases_endpoint(client):
    mock_result = _mock_agent_result("SAR_REQUIRED", 88.0)
    mock_result.transaction_id = "TX-CASES-TEST-001"

    tx = dict(VALID_TRANSACTION)
    tx["transaction_id"] = "TX-CASES-TEST-001"

    with patch("api.routes.analyze.AMLAgent") as MockAgent:
        mock_instance = MagicMock()
        MockAgent.return_value = mock_instance
        mock_instance.analyze.return_value = mock_result
        client.post("/api/v1/analyze", json=tx)

    response = client.get("/api/v1/cases")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # At least the case we just created should be there
    ids = [c["transaction_id"] for c in data]
    assert "TX-CASES-TEST-001" in ids
