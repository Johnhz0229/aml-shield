"""
Agent tests for AML-Shield.
Mocks the Anthropic API to avoid real API calls and ensure deterministic results.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.core import AMLAgent, AgentResult
from agent.executors import execute_tool


# ---------------------------------------------------------------------------
# Helpers: build fake Anthropic API responses
# ---------------------------------------------------------------------------

def _make_tool_use_block(tool_name: str, tool_input: dict, tool_id: str = "tu_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    return block


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(content: list, stop_reason: str = "tool_use"):
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 80
    return resp


def _build_full_mock_sequence(risk_score: float, decision: str):
    """
    Build a sequence of mock Anthropic responses that simulate a full ReAct loop:
    1. Call risk_scorer
    2. Call network_analyzer
    3. Call rule_checker
    4. Call escalation_decider
    5. Final text response (end_turn)
    """
    tx_id = "TX-TEST-001"

    # Round 1: call risk scorer
    round1 = _make_response(
        [_make_tool_use_block("transaction_risk_scorer", {
            "transaction_id": tx_id,
            "amount": 500.0,
            "sender_account": "DE001",
            "receiver_account": "DE002",
            "transaction_type": "card_payment",
            "timestamp": "2024-03-15T14:00:00Z",
            "sender_country": "DE",
            "receiver_country": "DE",
            "is_cross_border": False,
        }, "tu_001")],
        stop_reason="tool_use",
    )

    # Round 2: call escalation decider (low risk → skip network analysis)
    round2 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id,
            "risk_score": risk_score,
            "network_risk": "low",
            "rules_triggered": 0,
            "agent_reasoning": f"Low risk transaction, decision: {decision}",
            "sar_generated": False,
        }, "tu_002")],
        stop_reason="tool_use",
    )

    # Round 3: final text
    round3 = _make_response(
        [_make_text_block(f"DECISION: {decision}\nRisk Score: {risk_score}\nConfidence: High\n\nKey Findings:\n• Low risk domestic transaction\n\nRegulatory Basis: N/A\n\nReasoning: Transaction poses minimal AML risk.")],
        stop_reason="end_turn",
    )

    return [round1, round2, round3]


# ---------------------------------------------------------------------------
# Test 1: CLEAR transaction (low risk domestic)
# ---------------------------------------------------------------------------

def test_clear_transaction():
    """€500 domestic card payment should result in CLEAR decision."""
    tx_id = "TX-CLEAR-001"

    # Sequence: risk_scorer → escalation_decider → end_turn
    round1 = _make_response(
        [_make_tool_use_block("transaction_risk_scorer", {
            "transaction_id": tx_id,
            "amount": 500.0,
            "sender_account": "DE001",
            "receiver_account": "DE002",
            "transaction_type": "card_payment",
            "timestamp": "2024-03-15T14:00:00Z",
            "sender_country": "DE",
            "receiver_country": "DE",
            "is_cross_border": False,
        }, "tu_001")],
        stop_reason="tool_use",
    )

    round2 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id,
            "risk_score": 15.0,
            "network_risk": "low",
            "rules_triggered": 0,
            "agent_reasoning": "Low risk domestic transaction, no suspicious indicators",
        }, "tu_002")],
        stop_reason="tool_use",
    )

    round3 = _make_response(
        [_make_text_block("DECISION: CLEAR\nRisk Score: 15\nConfidence: High\n\nKey Findings:\n• Domestic card payment with no suspicious features\n\nRegulatory Basis: N/A\n\nReasoning: Standard domestic card payment within normal parameters.")],
        stop_reason="end_turn",
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze({
            "transaction_id": tx_id,
            "amount": 500.0,
            "transaction_type": "card_payment",
            "sender_account": "DE001",
            "receiver_account": "DE002",
            "sender_country": "DE",
            "receiver_country": "DE",
            "is_cross_border": False,
            "timestamp": "2024-03-15T14:00:00Z",
        })

    assert result.final_decision == "CLEAR"

    # Rule A: Low risk + domestic → entity_network_analyzer must NOT be called
    called_tools = [tc["tool_name"] for tc in result.tool_calls]
    assert "entity_network_analyzer" not in called_tools, (
        "entity_network_analyzer should be skipped for low-risk domestic transactions"
    )


# ---------------------------------------------------------------------------
# Test 2: SAR_REQUIRED transaction (€9,750 DE→IR at 02:34 AM)
# ---------------------------------------------------------------------------

def test_sar_transaction():
    """€9,750 wire transfer DE→IR at 02:34 AM should result in SAR_REQUIRED."""
    tx_id = "TX-SAR-001"

    round1 = _make_response(
        [_make_tool_use_block("transaction_risk_scorer", {
            "transaction_id": tx_id,
            "amount": 9750.0,
            "sender_account": "DE-SENDER",
            "receiver_account": "IR-RECEIVER",
            "transaction_type": "wire_transfer",
            "timestamp": "2024-03-15T02:34:00Z",
            "sender_country": "DE",
            "receiver_country": "IR",
            "is_cross_border": True,
        }, "tu_001")],
        stop_reason="tool_use",
    )

    round2 = _make_response(
        [_make_tool_use_block("entity_network_analyzer", {
            "account_id": "DE-SENDER",
            "depth": 3,
            "lookback_days": 90,
        }, "tu_002")],
        stop_reason="tool_use",
    )

    round3 = _make_response(
        [_make_tool_use_block("regulatory_rule_checker", {
            "transaction_id": tx_id,
            "amount": 9750.0,
            "transaction_type": "wire_transfer",
            "sender_country": "DE",
            "receiver_country": "IR",
        }, "tu_003")],
        stop_reason="tool_use",
    )

    round4 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id,
            "risk_score": 95.0,
            "network_risk": "high",
            "rules_triggered": 3,
            "agent_reasoning": "Structuring + blacklisted country + night transaction",
            "sar_generated": True,
        }, "tu_004")],
        stop_reason="tool_use",
    )

    round5 = _make_response(
        [_make_text_block("DECISION: SAR_REQUIRED\nRisk Score: 95\nConfidence: High\n\nKey Findings:\n• Amount €9,750 in structuring band — GwG §43 Abs. 1\n• Iran blacklisted — EU 6AMLD Art. 18\n• Night transaction\n\nRegulatory Basis: GwG §43 Abs. 1, FATF Recommendation 20, EU 6AMLD Art. 18\n\nReasoning: Multiple critical risk indicators.")],
        stop_reason="end_turn",
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3, round4, round5]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze({
            "transaction_id": tx_id,
            "amount": 9750.0,
            "transaction_type": "wire_transfer",
            "sender_account": "DE-SENDER",
            "receiver_account": "IR-RECEIVER",
            "sender_country": "DE",
            "receiver_country": "IR",
            "is_cross_border": True,
            "timestamp": "2024-03-15T02:34:00Z",
        })

    assert result.final_decision == "SAR_REQUIRED"
    assert len(result.reasoning_chain) >= 3

    called_tools = [tc["tool_name"] for tc in result.tool_calls]
    assert "regulatory_rule_checker" in called_tools


# ---------------------------------------------------------------------------
# Test 3: Grey zone — reasoning should mention evidence for/against
# ---------------------------------------------------------------------------

def test_grey_zone_reasoning():
    """€5,000 DE→PK wire at 14:00 should produce reasoning with evidence for/against."""
    tx_id = "TX-GREY-001"

    round1 = _make_response(
        [_make_tool_use_block("transaction_risk_scorer", {
            "transaction_id": tx_id,
            "amount": 5000.0,
            "sender_account": "DE001",
            "receiver_account": "PK001",
            "transaction_type": "wire_transfer",
            "timestamp": "2024-03-15T14:00:00Z",
            "sender_country": "DE",
            "receiver_country": "PK",
            "is_cross_border": True,
        }, "tu_001")],
        stop_reason="tool_use",
    )

    round2 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id,
            "risk_score": 50.0,
            "network_risk": "medium",
            "rules_triggered": 1,
            "agent_reasoning": (
                "Grey zone analysis:\n"
                "Evidence FOR suspicious: Pakistan is FATF grey-listed, cross-border transfer.\n"
                "Evidence AGAINST suspicious: Amount within normal range, business hours, no structuring.\n"
                "Decision: WATCHLIST given the grey-listed country and cross-border nature."
            ),
        }, "tu_002")],
        stop_reason="tool_use",
    )

    round3 = _make_response(
        [_make_text_block(
            "DECISION: WATCHLIST\nRisk Score: 50\nConfidence: Medium\n\n"
            "Key Findings:\n• Pakistan is FATF grey-listed\n\n"
            "Evidence for suspicious: Cross-border to FATF grey-listed jurisdiction.\n"
            "Evidence against suspicious: Normal business hours, amount below threshold.\n\n"
            "Regulatory Basis: FATF Recommendation 19\n\n"
            "Reasoning: Grey zone case requiring enhanced monitoring."
        )],
        stop_reason="end_turn",
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze({
            "transaction_id": tx_id,
            "amount": 5000.0,
            "transaction_type": "wire_transfer",
            "sender_account": "DE001",
            "receiver_account": "PK001",
            "sender_country": "DE",
            "receiver_country": "PK",
            "is_cross_border": True,
            "timestamp": "2024-03-15T14:00:00Z",
        })

    # Check that reasoning chain contains grey-zone evidence discussion
    full_reasoning = " ".join(
        step.get("content", "") + str(step.get("input", "")) + str(step.get("result", ""))
        for step in result.reasoning_chain
    ).lower()

    assert "evidence" in full_reasoning and (
        "for" in full_reasoning or "against" in full_reasoning
    ), "Grey zone reasoning should contain evidence for/against analysis"


# ---------------------------------------------------------------------------
# Test 4: Tool executor — risk scorer
# ---------------------------------------------------------------------------

def test_tool_executor_risk_scorer():
    result = execute_tool("transaction_risk_scorer", {
        "transaction_id": "TX-EXEC-001",
        "amount": 9750.0,
        "sender_account": "DE001",
        "receiver_account": "IR001",
        "transaction_type": "wire_transfer",
        "timestamp": "2024-03-15T02:34:00Z",
        "sender_country": "DE",
        "receiver_country": "IR",
        "is_cross_border": True,
    })

    assert "risk_score" in result
    assert "risk_level" in result
    assert "shap_attributions" in result
    assert 0 <= result["risk_score"] <= 100
    assert len(result["shap_attributions"]) <= 6
    assert all("feature" in a and "shap" in a for a in result["shap_attributions"])


# ---------------------------------------------------------------------------
# Test 5: Tool executor — rule checker with Iran + structuring
# ---------------------------------------------------------------------------

def test_tool_executor_rule_checker():
    result = execute_tool("regulatory_rule_checker", {
        "transaction_id": "TX-RULE-001",
        "amount": 9750.0,
        "transaction_type": "wire_transfer",
        "sender_country": "DE",
        "receiver_country": "IR",
    })

    triggered_ids = [r["rule_id"] for r in result["rules_triggered"]]

    assert "HIGH_RISK_JURISDICTION_BLACKLIST" in triggered_ids, (
        "Iran (IR) must trigger HIGH_RISK_JURISDICTION_BLACKLIST"
    )
    assert "STRUCTURING_BELOW_THRESHOLD" in triggered_ids, (
        "€9,750 must trigger STRUCTURING_BELOW_THRESHOLD"
    )
    assert result["sar_obligation"] is True
