"""
AML-Shield — Three-Tier Architecture Demo
==========================================
Demonstrates the ML pre-screening → conditional AI agent pipeline
across all three routing tiers with realistic transactions.

Usage:
    python demo.py              # uses heuristic scorer (no model file needed)
    python demo.py --live       # calls real Anthropic API for AGENT_REVIEW case
                                # requires ANTHROPIC_API_KEY in .env
"""

import os
import sys
import argparse
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

# Force heuristic scorer for demo: the trained XGBoost model was trained on IBM
# AMLworld data which has no country columns (all defaulted to "DE"), so it never
# learned country-based risk signals.  The heuristic explicitly encodes FATF
# blacklists / greylists and the structuring band — exactly what the demo needs.
os.environ["MODEL_PATH"] = "__demo_force_heuristic__"

# Demo triage thresholds — widened from production defaults to illustrate all
# three tiers clearly with the heuristic scorer (base_score=20, nothing < 10).
# Production: TRIAGE_LOW_THRESHOLD=10  (strict; only very confident clears pass)
# Demo:       TRIAGE_LOW_THRESHOLD=20  (shows AUTO_CLEAR tier with heuristic)
os.environ["TRIAGE_LOW_THRESHOLD"]  = "25"
os.environ["TRIAGE_HIGH_THRESHOLD"] = "75"

console = Console(width=100)

# ── Demo transactions — one per routing tier ──────────────────────────────────
DEMO_CASES = [
    {
        "label":       "Tier 1 — AUTO_CLEAR",
        "description": "€500 card payment, domestic Germany, 14:00",
        "color":       "green",
        "transaction": {
            "transaction_id":   "DEMO-TX-CLEAR-001",
            "amount":           500.00,
            "transaction_type": "card_payment",
            "sender_account":   "DE89370400440532013000",
            "receiver_account": "DE91100000000123456789",
            "sender_country":   "DE",
            "receiver_country": "DE",
            "is_cross_border":  False,
            "timestamp":        "2024-06-15T14:00:00Z",
            "reference":        "online purchase",
        },
    },
    {
        "label":       "Tier 2 — AGENT_REVIEW",
        "description": "€5 000 wire transfer DE→PK (FATF grey-listed), 14:00",
        "color":       "yellow",
        "transaction": {
            "transaction_id":   "DEMO-TX-REVIEW-002",
            "amount":           5_000.00,
            "transaction_type": "wire_transfer",
            "sender_account":   "DE89370400440532013000",
            "receiver_account": "PK-UBL-KARACHI-881234",
            "sender_country":   "DE",
            "receiver_country": "PK",
            "is_cross_border":  True,
            "timestamp":        "2024-06-15T14:00:00Z",
            "reference":        "supplier payment",
        },
    },
    {
        "label":       "Tier 3 — AUTO_SAR",
        "description": "€9 750 wire transfer DE→IR (blacklisted), 02:34 AM",
        "color":       "red",
        "transaction": {
            "transaction_id":   "DEMO-TX-SAR-003",
            "amount":           9_750.00,
            "transaction_type": "wire_transfer",
            "sender_account":   "DE89370400440532013000",
            "receiver_account": "IR-BANK-TEHRAN-44821",
            "sender_country":   "DE",
            "receiver_country": "IR",
            "is_cross_border":  True,
            "timestamp":        "2024-06-15T02:34:00Z",
            "reference":        "invoice payment",
        },
    },
]


# ── Simulated agent reasoning chain (used when --live is not set) ─────────────
SIMULATED_AGENT = {
    "final_decision": "ESCALATE_TO_HUMAN",
    "reasoning_chain": [
        {
            "step": 1, "type": "reasoning",
            "content": (
                "ML pre-screening scored this transaction at 45.0 (MEDIUM). "
                "SHAP top factor: is_cross_border (+0.31). "
                "Pakistan is FATF grey-listed — I will start with network analysis."
            ),
        },
        {
            "step": 1, "type": "tool_call",
            "tool_name": "entity_network_analyzer",
            "input": {"account_id": "DE89370400440532013000", "depth": 2, "lookback_days": 90},
        },
        {
            "step": 1, "type": "tool_result",
            "tool_name": "entity_network_analyzer",
            "result": {
                "network_risk_level": "medium",
                "pattern_detected":   "fan_out",
                "total_connected_accounts": 11,
                "flagged_connections": [
                    {
                        "account_id": "ACC-77412",
                        "relationship": "prior_structuring_detected",
                        "risk_indicator": "Multiple sub-threshold deposits within 24h",
                    }
                ],
            },
        },
        {
            "step": 2, "type": "reasoning",
            "content": (
                "Network shows fan_out pattern with 11 connected accounts and a prior "
                "structuring flag on ACC-77412. I will now run regulatory checks."
            ),
        },
        {
            "step": 2, "type": "tool_call",
            "tool_name": "regulatory_rule_checker",
            "input": {
                "transaction_id":   "DEMO-TX-REVIEW-002",
                "amount":           5000.0,
                "transaction_type": "wire_transfer",
                "sender_country":   "DE",
                "receiver_country": "PK",
            },
        },
        {
            "step": 2, "type": "tool_result",
            "tool_name": "regulatory_rule_checker",
            "result": {
                "rules_triggered_count": 2,
                "overall_severity": "medium",
                "sar_obligation": False,
                "rules_triggered": [
                    {
                        "rule_id":     "HIGH_RISK_JURISDICTION_GREYLIST",
                        "severity":    "medium",
                        "citation":    "FATF Recommendation 19 / EU 6AMLD Art. 18a",
                        "description": "Pakistan is FATF grey-listed — enhanced monitoring required",
                    },
                    {
                        "rule_id":     "WIRE_TRANSFER_FATF16",
                        "severity":    "medium",
                        "citation":    "FATF Recommendation 16 / EU Wire Transfer Regulation 2015/847",
                        "description": "Cross-border wire requires full originator/beneficiary info",
                    },
                ],
            },
        },
        {
            "step": 3, "type": "reasoning",
            "content": (
                "Grey zone analysis (ML score 45.0, between 40–60):\n"
                "FOR suspicious: Pakistan FATF grey-listed; fan_out network pattern; "
                "linked account with prior structuring flag.\n"
                "AGAINST suspicious: Amount within normal commercial range; business hours; "
                "no direct sanctions match; sar_obligation=False.\n"
                "Decision: ESCALATE_TO_HUMAN — network flag requires senior review, "
                "but evidence does not yet meet SAR threshold under GwG §43 Abs. 1."
            ),
        },
        {
            "step": 3, "type": "tool_call",
            "tool_name": "case_escalation_decider",
            "input": {
                "transaction_id": "DEMO-TX-REVIEW-002",
                "risk_score":     45.0,
                "network_risk":   "medium",
                "rules_triggered": 2,
                "agent_reasoning": (
                    "Grey zone: fan_out network + FATF grey-listed country + prior structuring link. "
                    "Amount and timing are benign. Escalate for senior compliance review."
                ),
            },
        },
        {
            "step": 3, "type": "tool_result",
            "tool_name": "case_escalation_decider",
            "result": {
                "final_decision": "ESCALATE_TO_HUMAN",
                "case_priority":  "P2",
                "sla_hours":      48,
                "assigned_queue": "SENIOR_COMPLIANCE_REVIEW",
            },
        },
    ],
    "tool_calls": [
        {"tool_name": "entity_network_analyzer"},
        {"tool_name": "regulatory_rule_checker"},
        {"tool_name": "case_escalation_decider"},
    ],
    "final_text": (
        "DECISION: ESCALATE_TO_HUMAN\n"
        "ML Score: 45.0\n"
        "Agent Confidence: Medium\n\n"
        "Key Findings:\n"
        "• Pakistan FATF grey-listed — FATF Recommendation 19 / EU 6AMLD Art. 18a\n"
        "• Fan-out network pattern with 11 connected accounts\n"
        "• Connected account ACC-77412 has prior structuring flag\n"
        "• Cross-border wire missing full originator info — FATF Recommendation 16\n\n"
        "ML Score Verdict: CONFIRMED\n"
        "  Reason: Network investigation confirms medium-risk assessment.\n\n"
        "Regulatory Basis: FATF Recommendation 19, FATF Recommendation 16, EU 6AMLD Art. 18a\n\n"
        "Reasoning: Grey zone case with FATF grey-listed destination, fan-out network pattern, "
        "and prior structuring link. Amount and timing are non-suspicious. "
        "Escalated to senior compliance for 48h review per P2 SLA."
    ),
    "total_tokens": 0,
    "duration_ms":  0,
}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _header():
    console.print()
    console.print(Panel(
        "[bold white]AML-Shield — Three-Tier Architecture Demo[/bold white]\n"
        "[dim]XGBoost pre-screening  →  conditional AI agent investigation[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
        padding=(0, 4),
    ))
    console.print()

    threshold_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    threshold_table.add_column(style="bold")
    threshold_table.add_column()
    threshold_table.add_row("[green]AUTO_CLEAR[/green]",
        "ML score < 25   →  pass through  [dim](prod: < 10, stricter)[/dim]")
    threshold_table.add_row("[yellow]AGENT_REVIEW[/yellow]",
        "ML score 20–74  →  AI agent investigates")
    threshold_table.add_row("[red]AUTO_SAR[/red]",
        "ML score ≥ 75   →  automatic SAR escalation")
    console.print(threshold_table)
    console.print()


def _case_divider(idx: int, case: dict):
    color = case["color"]
    console.rule(
        f"[bold {color}]CASE {idx}/3  ·  {case['label']}[/bold {color}]"
        f"  [dim]{case['description']}[/dim]",
        style=color,
    )
    console.print()


def _print_transaction(tx: dict):
    t = Table(title="Transaction", box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    t.add_column("Field",  style="dim")
    t.add_column("Value",  style="white")

    labels = {
        "transaction_id":   "ID",
        "amount":           "Amount",
        "transaction_type": "Type",
        "sender_country":   "Sender Country",
        "receiver_country": "Receiver Country",
        "is_cross_border":  "Cross-border",
        "timestamp":        "Timestamp",
    }
    for key, label in labels.items():
        val = tx.get(key, "")
        if key == "amount":
            val = f"€{float(val):,.2f}"
        t.add_row(label, str(val))

    console.print(t)
    console.print()


def _print_triage(triage_result, color: str):
    routing_color = {"AUTO_CLEAR": "green", "AGENT_REVIEW": "yellow", "AUTO_SAR": "red"}
    rc = routing_color.get(triage_result.routing, "white")

    t = Table(title="ML Triage Result", box=box.SIMPLE_HEAD, padding=(0, 2))
    t.add_column("Metric", style="dim")
    t.add_column("Value")

    t.add_row("ML Score",   f"[bold]{triage_result.ml_score:.1f}[/bold] / 100")
    t.add_row("Risk Level", triage_result.ml_risk_level)
    t.add_row("Routing",    f"[bold {rc}]{triage_result.routing}[/bold {rc}]")
    t.add_row("Reason",     f"[dim]{triage_result.routing_reason}[/dim]")
    t.add_row("Model",      triage_result.model_version)
    console.print(t)

    if triage_result.shap_attributions:
        st = Table(title="Top SHAP Attributions", box=box.SIMPLE_HEAD, padding=(0, 2))
        st.add_column("Feature",   style="dim")
        st.add_column("Value",     justify="right")
        st.add_column("SHAP",      justify="right")
        st.add_column("Direction")

        for attr in triage_result.shap_attributions[:5]:
            shap_val = attr.get("shap", 0)
            direction = attr.get("direction", "")
            color_d = "red" if "increases" in direction else "green"
            st.add_row(
                attr.get("feature", ""),
                str(attr.get("value", "")),
                f"{shap_val:+.3f}",
                f"[{color_d}]{direction}[/{color_d}]",
            )
        console.print(st)

    console.print()


def _print_auto_result(routing: str, final_decision: str, duration_ms: int):
    color = "green" if routing == "AUTO_CLEAR" else "red"
    icon  = "✓" if routing == "AUTO_CLEAR" else "⚠"
    console.print(Panel(
        f"[bold {color}]{icon}  {routing}  →  {final_decision}[/bold {color}]\n"
        f"[dim]No AI agent involvement  ·  {duration_ms}ms  ·  0 tokens[/dim]",
        border_style=color,
        padding=(0, 4),
    ))
    console.print()


def _print_agent_result(agent_data: dict, duration_ms: int):
    decision = agent_data["final_decision"]
    tokens   = agent_data.get("total_tokens", 0)
    tools    = [tc["tool_name"] for tc in agent_data.get("tool_calls", [])]

    console.print(Panel(
        f"[bold yellow]🤖  AGENT DECISION:  {decision}[/bold yellow]",
        border_style="yellow",
        padding=(0, 4),
    ))
    console.print()

    # Reasoning chain
    console.print("[bold]Agent Reasoning Chain:[/bold]")
    console.print()

    for step in agent_data.get("reasoning_chain", []):
        stype = step.get("type", "")
        if stype == "reasoning":
            console.print(
                f"  [cyan]💭 Think:[/cyan] [dim]{step['content'][:300]}[/dim]"
            )
        elif stype == "tool_call":
            tool = step.get("tool_name", "")
            inp  = step.get("input", {})
            console.print(
                f"  [blue]🔧 Call:[/blue] [bold]{tool}[/bold]"
                f"  [dim]{_fmt_input(inp)}[/dim]"
            )
        elif stype == "tool_result":
            tool   = step.get("tool_name", "")
            result = step.get("result", {})
            console.print(
                f"  [green]📋 Result:[/green] [bold]{tool}[/bold]"
                f"  [dim]{_fmt_result(tool, result)}[/dim]"
            )
        console.print()

    # Final report
    final_text = agent_data.get("final_text", "")
    if final_text:
        console.print(Panel(
            final_text,
            title="[bold]Final Compliance Report[/bold]",
            border_style="yellow",
            padding=(0, 2),
        ))
        console.print()

    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    meta.add_row("Tools called",  " → ".join(tools))
    meta.add_row("Tokens used",   str(tokens) if tokens else "[dim]simulated[/dim]")
    meta.add_row("Duration",      f"{duration_ms}ms" if duration_ms else "[dim]simulated[/dim]")
    console.print(meta)
    console.print()


def _fmt_input(inp: dict) -> str:
    items = []
    for k, v in list(inp.items())[:3]:
        items.append(f"{k}={repr(v)[:30]}")
    return "  ".join(items)


def _fmt_result(tool: str, result: dict) -> str:
    if tool == "entity_network_analyzer":
        risk  = result.get("network_risk_level", "?")
        pat   = result.get("pattern_detected", "none")
        flags = len(result.get("flagged_connections", []))
        return f"network_risk={risk}  pattern={pat}  flagged_connections={flags}"
    if tool == "regulatory_rule_checker":
        count = result.get("rules_triggered_count", 0)
        sev   = result.get("overall_severity", "?")
        sar   = result.get("sar_obligation", False)
        return f"rules_triggered={count}  severity={sev}  sar_obligation={sar}"
    if tool == "case_escalation_decider":
        d   = result.get("final_decision", "?")
        pri = result.get("case_priority", "?")
        sla = result.get("sla_hours", "?")
        return f"decision={d}  priority={pri}  sla={sla}h"
    return str(result)[:120]


def _summary_table(results: list):
    console.rule("[bold]Summary[/bold]", style="cyan")
    console.print()

    t = Table(box=box.ROUNDED, show_header=True, padding=(0, 2), border_style="cyan")
    t.add_column("Case",           style="bold")
    t.add_column("Amount",         justify="right")
    t.add_column("Route",          justify="center")
    t.add_column("ML Score",       justify="right")
    t.add_column("Final Decision", justify="center")
    t.add_column("Agent Calls",    justify="center")
    t.add_column("Tokens",         justify="right")

    routing_color = {"AUTO_CLEAR": "green", "AGENT_REVIEW": "yellow", "AUTO_SAR": "red"}
    decision_color = {
        "CLEAR": "green", "WATCHLIST": "cyan",
        "ESCALATE_TO_HUMAN": "yellow", "SAR_REQUIRED": "red",
    }

    for r in results:
        rc = routing_color.get(r["routing"], "white")
        dc = decision_color.get(r["decision"], "white")
        t.add_row(
            r["label"],
            f"€{r['amount']:,.0f}",
            f"[{rc}]{r['routing']}[/{rc}]",
            f"{r['ml_score']:.1f}",
            f"[{dc}]{r['decision']}[/{dc}]",
            str(r["agent_calls"]),
            str(r["tokens"]) if r["tokens"] else "—",
        )

    console.print(t)
    console.print()

    agent_cases = sum(1 for r in results if r["routing"] == "AGENT_REVIEW")
    auto_cases  = len(results) - agent_cases
    console.print(
        f"  [dim]Agent invoked for {agent_cases}/{len(results)} cases  "
        f"({auto_cases} transactions auto-routed without LLM cost)[/dim]"
    )
    console.print()


# ── Main demo logic ───────────────────────────────────────────────────────────

def run_demo(live_agent: bool = False):
    from api.triage import triage

    _header()

    summary = []

    for idx, case in enumerate(DEMO_CASES, start=1):
        _case_divider(idx, case)
        tx    = case["transaction"]
        color = case["color"]

        _print_transaction(tx)

        # ML triage
        console.print("[dim]Running ML triage...[/dim]")
        t0           = time.time()
        triage_result = triage(tx)
        triage_ms    = int((time.time() - t0) * 1000)
        console.print(f"[dim]  scored in {triage_ms}ms[/dim]")
        console.print()

        _print_triage(triage_result, color)

        if triage_result.routing == "AUTO_CLEAR":
            _print_auto_result("AUTO_CLEAR", "CLEAR", triage_ms)
            summary.append({
                "label": case["label"], "amount": tx["amount"],
                "routing": "AUTO_CLEAR", "ml_score": triage_result.ml_score,
                "decision": "CLEAR", "agent_calls": 0, "tokens": 0,
            })

        elif triage_result.routing == "AUTO_SAR":
            _print_auto_result("AUTO_SAR", "SAR_REQUIRED", triage_ms)
            summary.append({
                "label": case["label"], "amount": tx["amount"],
                "routing": "AUTO_SAR", "ml_score": triage_result.ml_score,
                "decision": "SAR_REQUIRED", "agent_calls": 0, "tokens": 0,
            })

        else:  # AGENT_REVIEW
            console.print(
                "[yellow]⟶  Entering AI Agent investigation queue...[/yellow]"
            )
            console.print()

            ml_context = {
                "score":             triage_result.ml_score,
                "risk_level":        triage_result.ml_risk_level,
                "shap_attributions": triage_result.shap_attributions,
                "routing_reason":    triage_result.routing_reason,
                "low_threshold":     25,
                "high_threshold":    75,
            }

            if live_agent and os.environ.get("ANTHROPIC_API_KEY"):
                console.print("[dim]Calling Anthropic API (live mode)...[/dim]")
                from agent.core import AMLAgent
                agent = AMLAgent()
                t1 = time.time()
                agent_result = agent.analyze(tx, ml_context=ml_context)
                agent_ms = int((time.time() - t1) * 1000)
                agent_data = {
                    "final_decision": agent_result.final_decision,
                    "reasoning_chain": agent_result.reasoning_chain,
                    "tool_calls": agent_result.tool_calls,
                    "final_text": agent_result.final_text,
                    "total_tokens": agent_result.total_tokens,
                    "duration_ms": agent_ms,
                }
            else:
                console.print(
                    "[dim]Using simulated agent response "
                    "(pass --live with ANTHROPIC_API_KEY for real API call)[/dim]"
                )
                console.print()
                agent_data = SIMULATED_AGENT
                agent_ms   = 0

            _print_agent_result(agent_data, agent_ms)

            tools_called = len(agent_data.get("tool_calls", []))
            summary.append({
                "label": case["label"], "amount": tx["amount"],
                "routing": "AGENT_REVIEW", "ml_score": triage_result.ml_score,
                "decision": agent_data["final_decision"],
                "agent_calls": tools_called,
                "tokens": agent_data.get("total_tokens", 0),
            })

    _summary_table(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML-Shield architecture demo")
    parser.add_argument(
        "--live", action="store_true",
        help="Call real Anthropic API for AGENT_REVIEW case (requires ANTHROPIC_API_KEY)"
    )
    args = parser.parse_args()

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]--live requires ANTHROPIC_API_KEY set in .env[/red]")
        sys.exit(1)

    run_demo(live_agent=args.live)
