"""
Agent tests for AML-Shield.
Tests focus on the AGENT_REVIEW path: medium-risk cases where the ML model
has already scored the transaction and the agent investigates further.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.core import AMLAgent, AgentResult
from agent.executors import execute_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_use_block(tool_name: str, tool_input: dict, tool_id: str = "tu_001"):
    block = MagicMock()
    block.type  = "tool_use"
    block.name  = tool_name
    block.input = tool_input
    block.id    = tool_id
    return block


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(content: list, stop_reason: str = "tool_use"):
    resp = MagicMock()
    resp.content    = content
    resp.stop_reason = stop_reason
    resp.usage.input_tokens  = 100
    resp.usage.output_tokens = 80
    return resp


# ---------------------------------------------------------------------------
# Test 1: ML context is passed into the prompt (no re-scoring)
# ---------------------------------------------------------------------------

def test_agent_receives_ml_context_in_prompt():
    """Agent prompt must include ML pre-screening block when ml_context is provided."""
    tx_id = "TX-ML-CTX-001"

    # Agent calls: network_analyzer → escalation_decider → end_turn
    round1 = _make_response(
        [_make_tool_use_block("entity_network_analyzer", {
            "account_id": "DE001",
            "depth": 2,
        }, "tu_001")],
        stop_reason="tool_use",
    )
    round2 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id,
            "risk_score": 48.0,
            "network_risk": "medium",
            "rules_triggered": 1,
            "agent_reasoning": "Medium risk, grey-listed country, monitoring recommended.",
        }, "tu_002")],
        stop_reason="tool_use",
    )
    round3 = _make_response(
        [_make_text_block(
            "DECISION: WATCHLIST\nML Score: 48.0\nAgent Confidence: Medium\n\n"
            "Key Findings:\n• Pakistan is FATF grey-listed — FATF Recommendation 19\n\n"
            "ML Score Verdict: CONFIRMED\n  Reason: Network analysis supports ML assessment.\n\n"
            "Regulatory Basis: FATF Recommendation 19\n\n"
            "Reasoning: Transaction to FATF grey-listed jurisdiction warrants monitoring."
        )],
        stop_reason="end_turn",
    )

    captured_messages = []

    def fake_create(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        side = [round1, round2, round3]
        return side.pop(0)

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze(
            {
                "transaction_id":   tx_id,
                "amount":           5000.0,
                "transaction_type": "wire_transfer",
                "sender_account":   "DE001",
                "receiver_account": "PK001",
                "sender_country":   "DE",
                "receiver_country": "PK",
                "is_cross_border":  True,
                "timestamp":        "2024-03-15T14:00:00Z",
            },
            ml_context={
                "score":             48.0,
                "risk_level":        "MEDIUM",
                "shap_attributions": [
                    {"feature": "is_cross_border", "value": 1, "shap": 0.31, "direction": "increases_risk"},
                    {"feature": "amount_log",       "value": 8.5, "shap": 0.12, "direction": "increases_risk"},
                ],
                "routing_reason": "Score 48.0 in medium-risk band",
                "low_threshold":  25,
                "high_threshold": 75,
            },
        )

    assert result.final_decision == "WATCHLIST"

    # The first user message must contain the ML pre-screening block
    first_msg = mock_client.messages.create.call_args_list[0]
    first_user_content = first_msg.kwargs.get("messages", first_msg.args[0] if first_msg.args else [])[0]["content"]
    assert "ML Pre-Screening Result" in first_user_content
    assert "48.0" in first_user_content
    assert "AGENT_REVIEW" in first_user_content


# ---------------------------------------------------------------------------
# Test 2: Agent does NOT call transaction_risk_scorer when ml_context provided
# ---------------------------------------------------------------------------

def test_agent_skips_risk_scorer_with_ml_context():
    """When ml_context is given, the agent should NOT call transaction_risk_scorer."""
    tx_id = "TX-NO-RESCORE-001"

    round1 = _make_response(
        [_make_tool_use_block("entity_network_analyzer", {"account_id": "DE001"}, "tu_001")],
        stop_reason="tool_use",
    )
    round2 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id,
            "risk_score": 55.0,
            "network_risk": "medium",
            "rules_triggered": 1,
            "agent_reasoning": "Monitoring recommended.",
        }, "tu_002")],
        stop_reason="tool_use",
    )
    round3 = _make_response(
        [_make_text_block("DECISION: WATCHLIST\nML Score: 55.0\nAgent Confidence: Medium\n\n"
                          "Key Findings:\n• Cross-border to FATF grey-listed jurisdiction\n\n"
                          "ML Score Verdict: CONFIRMED\n  Reason: Investigation supports ML.\n\n"
                          "Regulatory Basis: FATF Recommendation 19\n\n"
                          "Reasoning: Medium risk monitoring case.")],
        stop_reason="end_turn",
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze(
            {
                "transaction_id": tx_id, "amount": 5000.0,
                "transaction_type": "wire_transfer",
                "sender_account": "DE001", "receiver_account": "PK001",
                "sender_country": "DE", "receiver_country": "PK",
                "is_cross_border": True, "timestamp": "2024-03-15T14:00:00Z",
            },
            ml_context={"score": 55.0, "risk_level": "MEDIUM",
                        "shap_attributions": [], "routing_reason": "test"},
        )

    called_tools = [tc["tool_name"] for tc in result.tool_calls]
    assert "transaction_risk_scorer" not in called_tools, (
        "Agent must not re-call transaction_risk_scorer when ml_context is provided"
    )
    assert "entity_network_analyzer" in called_tools


# ---------------------------------------------------------------------------
# Test 3: SAR_REQUIRED — investigation escalates ML verdict
# ---------------------------------------------------------------------------

def test_agent_escalates_to_sar_on_network_hit():
    """Agent escalates to SAR_REQUIRED when network analysis finds a sanctions match."""
    tx_id = "TX-SAR-ESCALATE-001"

    round1 = _make_response(
        [_make_tool_use_block("entity_network_analyzer", {
            "account_id": "DE-SENDER", "depth": 3, "lookback_days": 90,
        }, "tu_001")],
        stop_reason="tool_use",
    )
    round2 = _make_response(
        [_make_tool_use_block("regulatory_rule_checker", {
            "transaction_id": tx_id, "amount": 9750.0,
            "transaction_type": "wire_transfer",
            "sender_country": "DE", "receiver_country": "IR",
        }, "tu_002")],
        stop_reason="tool_use",
    )
    round3 = _make_response(
        [_make_tool_use_block("sar_report_generator", {
            "transaction_id": tx_id, "risk_score": 72.0,
            "triggered_rules": ["STRUCTURING_BELOW_THRESHOLD", "HIGH_RISK_JURISDICTION_BLACKLIST"],
            "suspicious_patterns": ["structuring", "sanctions_proximity"],
            "narrative": "Transaction to Iran with structuring amount; network shows sanctions-adjacent account.",
        }, "tu_003")],
        stop_reason="tool_use",
    )
    round4 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id, "risk_score": 72.0,
            "network_risk": "critical", "rules_triggered": 2,
            "agent_reasoning": "Sanctions-adjacent network + structuring + Iran = SAR_REQUIRED.",
            "sar_generated": True,
        }, "tu_004")],
        stop_reason="tool_use",
    )
    round5 = _make_response(
        [_make_text_block(
            "DECISION: SAR_REQUIRED\nML Score: 72.0\nAgent Confidence: High\n\n"
            "Key Findings:\n• Structuring band — GwG §43 Abs. 1\n"
            "• Iran blacklist — EU 6AMLD Art. 18\n• Network sanctions proximity\n\n"
            "ML Score Verdict: ESCALATED\n  Reason: Network hit confirms escalation beyond ML score.\n\n"
            "Regulatory Basis: GwG §43 Abs. 1, EU 6AMLD Art. 18, FATF Recommendation 20\n\n"
            "Reasoning: Combination of structuring, blacklisted jurisdiction, and network sanctions proximity."
        )],
        stop_reason="end_turn",
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3, round4, round5]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze(
            {
                "transaction_id": tx_id, "amount": 9750.0,
                "transaction_type": "wire_transfer",
                "sender_account": "DE-SENDER", "receiver_account": "IR-RECEIVER",
                "sender_country": "DE", "receiver_country": "IR",
                "is_cross_border": True, "timestamp": "2024-03-15T02:34:00Z",
            },
            ml_context={
                "score": 72.0, "risk_level": "HIGH",
                "shap_attributions": [
                    {"feature": "is_near_ctr_threshold", "value": 1, "shap": 0.45, "direction": "increases_risk"},
                ],
                "routing_reason": "Score 72.0 in AGENT_REVIEW band",
                "low_threshold": 25, "high_threshold": 75,
            },
        )

    assert result.final_decision == "SAR_REQUIRED"

    called_tools = [tc["tool_name"] for tc in result.tool_calls]
    assert "regulatory_rule_checker" in called_tools
    assert "sar_report_generator" in called_tools


# ---------------------------------------------------------------------------
# Test 4: Grey zone reasoning — evidence for/against documented
# ---------------------------------------------------------------------------

def test_grey_zone_reasoning():
    """Score in 40-60 band — reasoning must contain evidence for/against."""
    tx_id = "TX-GREY-001"

    round1 = _make_response(
        [_make_tool_use_block("entity_network_analyzer", {"account_id": "DE001"}, "tu_001")],
        stop_reason="tool_use",
    )
    round2 = _make_response(
        [_make_tool_use_block("case_escalation_decider", {
            "transaction_id": tx_id, "risk_score": 50.0,
            "network_risk": "medium", "rules_triggered": 1,
            "agent_reasoning": (
                "Grey zone analysis:\n"
                "Evidence FOR suspicious: Pakistan is FATF grey-listed, cross-border.\n"
                "Evidence AGAINST suspicious: Business hours, amount within normal range.\n"
                "Decision: WATCHLIST — grey-listed country and cross-border nature."
            ),
        }, "tu_002")],
        stop_reason="tool_use",
    )
    round3 = _make_response(
        [_make_text_block(
            "DECISION: WATCHLIST\nML Score: 50.0\nAgent Confidence: Medium\n\n"
            "Key Findings:\n• Pakistan is FATF grey-listed — FATF Recommendation 19\n\n"
            "Evidence for suspicious: Cross-border to FATF grey-listed jurisdiction.\n"
            "Evidence against suspicious: Normal business hours, amount below threshold.\n\n"
            "ML Score Verdict: CONFIRMED\n  Reason: Investigation aligns with ML assessment.\n\n"
            "Regulatory Basis: FATF Recommendation 19\n\n"
            "Reasoning: Grey zone case with grey-listed jurisdiction."
        )],
        stop_reason="end_turn",
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.side_effect = [round1, round2, round3]

        agent = AMLAgent(api_key="test-key")
        result = agent.analyze(
            {
                "transaction_id": tx_id, "amount": 5000.0,
                "transaction_type": "wire_transfer",
                "sender_account": "DE001", "receiver_account": "PK001",
                "sender_country": "DE", "receiver_country": "PK",
                "is_cross_border": True, "timestamp": "2024-03-15T14:00:00Z",
            },
            ml_context={"score": 50.0, "risk_level": "MEDIUM",
                        "shap_attributions": [], "routing_reason": "test"},
        )

    full_reasoning = " ".join(
        step.get("content", "") + str(step.get("input", "")) + str(step.get("result", ""))
        for step in result.reasoning_chain
    ).lower()

    assert "evidence" in full_reasoning and (
        "for" in full_reasoning or "against" in full_reasoning
    ), "Grey zone reasoning must contain evidence for/against analysis"


# ---------------------------------------------------------------------------
# Test 5: Tool executor — rule checker with Iran + structuring
# ---------------------------------------------------------------------------

def test_tool_executor_rule_checker():
    result = execute_tool("regulatory_rule_checker", {
        "transaction_id":   "TX-RULE-001",
        "amount":           9750.0,
        "transaction_type": "wire_transfer",
        "sender_country":   "DE",
        "receiver_country": "IR",
    })

    triggered_ids = [r["rule_id"] for r in result["rules_triggered"]]
    assert "HIGH_RISK_JURISDICTION_BLACKLIST" in triggered_ids
    assert "STRUCTURING_BELOW_THRESHOLD" in triggered_ids
    assert result["sar_obligation"] is True


# ---------------------------------------------------------------------------
# Test 6: Tool executor — risk scorer still works standalone
# ---------------------------------------------------------------------------

def test_tool_executor_risk_scorer():
    result = execute_tool("transaction_risk_scorer", {
        "transaction_id":   "TX-EXEC-001",
        "amount":           9750.0,
        "sender_account":   "DE001",
        "receiver_account": "IR001",
        "transaction_type": "wire_transfer",
        "timestamp":        "2024-03-15T02:34:00Z",
        "sender_country":   "DE",
        "receiver_country": "IR",
        "is_cross_border":  True,
    })

    assert "risk_score" in result
    assert "risk_level" in result
    assert "shap_attributions" in result
    assert 0 <= result["risk_score"] <= 100
    assert len(result["shap_attributions"]) <= 6
    assert all("feature" in a and "shap" in a for a in result["shap_attributions"])
