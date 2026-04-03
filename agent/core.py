"""
AML-Shield Agent Core Loop
Implements ReAct (Reasoning + Acting) pattern using Anthropic Tool-Use API.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

import anthropic

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS
from agent.executors import execute_tool

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
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
    MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    MAX_ITERATIONS = 10
    MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=key)

    def analyze(self, transaction: dict) -> AgentResult:
        """
        Run the ReAct agent loop to analyze a transaction for AML risk.
        Returns a fully populated AgentResult with reasoning chain and final decision.
        """
        start_ms = time.time()
        transaction_id = transaction.get("transaction_id", "UNKNOWN")

        reasoning_chain = []
        tool_calls_log = []
        tool_results_log = []
        final_text = ""
        total_tokens = 0
        final_decision = "UNKNOWN"
        risk_score = None
        error = None

        messages = [
            {
                "role": "user",
                "content": self._format_transaction_prompt(transaction),
            }
        ]

        try:
            iteration = 0
            while iteration < self.MAX_ITERATIONS:
                iteration += 1
                logger.debug(f"[{transaction_id}] ReAct iteration {iteration}")

                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                total_tokens += response.usage.input_tokens + response.usage.output_tokens

                # Process each content block in the response
                tool_use_blocks = []
                tool_result_blocks = []

                for block in response.content:
                    if block.type == "text":
                        reasoning_chain.append({
                            "step": iteration,
                            "type": "reasoning",
                            "content": block.text,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        if response.stop_reason == "end_turn":
                            final_text = block.text

                    elif block.type == "tool_use":
                        tool_call_record = {
                            "step": iteration,
                            "tool_name": block.name,
                            "tool_use_id": block.id,
                            "input": block.input,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        tool_calls_log.append(tool_call_record)
                        reasoning_chain.append({
                            "step": iteration,
                            "type": "tool_call",
                            "tool_name": block.name,
                            "input": block.input,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                        # Execute the tool
                        logger.debug(f"[{transaction_id}] Executing tool: {block.name}")
                        tool_result = execute_tool(block.name, block.input)

                        tool_result_record = {
                            "step": iteration,
                            "tool_name": block.name,
                            "tool_use_id": block.id,
                            "result": tool_result,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        tool_results_log.append(tool_result_record)
                        reasoning_chain.append({
                            "step": iteration,
                            "type": "tool_result",
                            "tool_name": block.name,
                            "result": tool_result,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                        # Extract key values from tool results
                        if block.name == "transaction_risk_scorer":
                            risk_score = tool_result.get("risk_score")
                        elif block.name == "case_escalation_decider":
                            final_decision = tool_result.get("final_decision", final_decision)

                        tool_use_blocks.append(block)
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(tool_result),
                        })

                # Check stop condition
                if response.stop_reason == "end_turn":
                    logger.debug(f"[{transaction_id}] Agent reached end_turn after {iteration} iterations")
                    break

                if response.stop_reason == "tool_use" and tool_use_blocks:
                    # Append assistant message and tool results, continue loop
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_result_blocks})
                else:
                    # Unexpected stop reason
                    logger.warning(f"[{transaction_id}] Unexpected stop_reason: {response.stop_reason}")
                    break

        except anthropic.APIError as e:
            error = f"Anthropic API error: {str(e)}"
            logger.error(f"[{transaction_id}] {error}")
        except Exception as e:
            error = f"Agent error: {str(e)}"
            logger.exception(f"[{transaction_id}] Unexpected error in agent loop")

        duration_ms = int((time.time() - start_ms) * 1000)

        return AgentResult(
            transaction_id=transaction_id,
            final_decision=final_decision,
            risk_score=risk_score,
            reasoning_chain=reasoning_chain,
            tool_calls=tool_calls_log,
            tool_results=tool_results_log,
            final_text=final_text,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            error=error,
        )

    def _format_transaction_prompt(self, tx: dict) -> str:
        """Format transaction as a clean markdown table for the agent."""
        lines = ["## Transaction Under Review", ""]
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")

        field_labels = {
            "transaction_id": "Transaction ID",
            "amount": "Amount (EUR)",
            "transaction_type": "Transaction Type",
            "sender_account": "Sender Account",
            "receiver_account": "Receiver Account",
            "sender_country": "Sender Country",
            "receiver_country": "Receiver Country",
            "is_cross_border": "Cross-Border",
            "timestamp": "Timestamp",
            "reference": "Reference",
        }

        for key, label in field_labels.items():
            value = tx.get(key)
            if value is not None:
                if key == "amount":
                    value = f"€{float(value):,.2f}"
                lines.append(f"| {label} | {value} |")

        lines.append("")
        lines.append("Begin your analysis. Follow the ReAct protocol.")
        return "\n".join(lines)


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    load_dotenv()

    # Test transaction: €9,750 wire transfer DE→IR at 02:34 AM
    test_transaction = {
        "transaction_id": "TX-TEST-IRAN-001",
        "amount": 9750.00,
        "transaction_type": "wire_transfer",
        "sender_account": "DE89370400440532013000",
        "receiver_account": "IR-BANK-TEHRAN-44821",
        "sender_country": "DE",
        "receiver_country": "IR",
        "is_cross_border": True,
        "timestamp": "2024-03-15T02:34:00Z",
        "reference": "invoice payment",
    }

    print("=" * 60)
    print("AML-Shield Agent — Smoke Test")
    print("=" * 60)
    print(f"Transaction: €9,750 wire transfer DE→IR at 02:34 AM")
    print()

    agent = AMLAgent()
    result = agent.analyze(test_transaction)

    print(f"Decision:     {result.final_decision}")
    print(f"Risk Score:   {result.risk_score}")
    print(f"Iterations:   {len([s for s in result.reasoning_chain if s['type'] == 'tool_call'])}")
    print(f"Tool Calls:   {len(result.tool_calls)}")
    print(f"Tokens:       {result.total_tokens}")
    print(f"Duration:     {result.duration_ms}ms")
    if result.error:
        print(f"Error:        {result.error}")
    print()
    print("--- Final Text ---")
    print(result.final_text[:1000] if result.final_text else "(no final text)")
