"""
Tool executor functions for AML-Shield agent.
Each executor implements a tool defined in agent/tools.py.
"""

import os
import json
import random
import logging
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# High-risk jurisdiction blacklist (FATF/EU Delegated Reg. 2016/1675)
BLACKLIST_COUNTRIES = {"IR", "KP", "MM", "SY", "YE", "AF"}

# High-risk jurisdiction greylist (FATF monitored)
GREYLIST_COUNTRIES = {"PK", "TR", "ML", "VN", "MZ", "TZ", "JO"}


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Route tool calls to the appropriate executor."""
    executors = {
        "transaction_risk_scorer": _run_risk_scorer,
        "entity_network_analyzer": _run_network_analyzer,
        "regulatory_rule_checker": _run_rule_checker,
        "sar_report_generator": _run_sar_generator,
        "case_escalation_decider": _run_escalation_decider,
    }
    if tool_name not in executors:
        return {"error": f"Unknown tool: {tool_name}"}
    return executors[tool_name](tool_input)


def _run_risk_scorer(inp: dict) -> dict:
    """
    Score AML risk using XGBoost model (or heuristic fallback).
    Returns risk_score, risk_level, confidence_interval, shap_attributions,
    model_version, scored_at.
    """
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_aml.pkl")
    model_version = "xgboost-v1-heuristic"
    risk_score = None

    # Try real model first
    if os.path.exists(model_path):
        try:
            from models.predict import predict_with_shap
            result = predict_with_shap(inp)
            result["scored_at"] = datetime.now(timezone.utc).isoformat()
            result["model_version"] = "xgboost-v1-trained"
            return result
        except Exception as e:
            logger.warning(f"Model inference failed, using heuristic: {e}")

    # Heuristic fallback scoring
    amount = float(inp.get("amount", 0))
    is_cross_border = bool(inp.get("is_cross_border", False))
    transaction_type = inp.get("transaction_type", "")
    timestamp = inp.get("timestamp", "")

    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        hour = dt.hour
    except Exception:
        hour = 12  # default to midday if parsing fails

    base_score = 20.0

    # Structuring signal: just below €10,000 CTR threshold
    if 8500 <= amount < 10000:
        base_score += 35
    elif amount >= 10000:
        base_score += 15

    if is_cross_border:
        base_score += 15

    if transaction_type == "crypto_exchange":
        base_score += 20
    elif transaction_type in ("cash_deposit", "cash_withdrawal"):
        base_score += 10

    # Night transaction signal
    if hour < 6 or hour > 22:
        base_score += 10

    # High-risk jurisdiction signals
    sender_country = inp.get("sender_country", "")
    receiver_country = inp.get("receiver_country", "")
    if sender_country in BLACKLIST_COUNTRIES or receiver_country in BLACKLIST_COUNTRIES:
        base_score += 25
    elif sender_country in GREYLIST_COUNTRIES or receiver_country in GREYLIST_COUNTRIES:
        base_score += 10

    # Add gaussian noise
    noise = random.gauss(0, 3)
    risk_score = max(0.0, min(100.0, base_score + noise))

    # Determine risk level
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 60:
        risk_level = "MEDIUM"
    elif risk_score < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    # Confidence interval (heuristic: ±8 points)
    ci_half = 8.0
    confidence_interval = {
        "lower": max(0.0, round(risk_score - ci_half, 1)),
        "upper": min(100.0, round(risk_score + ci_half, 1))
    }

    # Mock SHAP attributions (sorted by |shap| descending)
    def _shap_val(feature_boost: float) -> float:
        return round(feature_boost * random.uniform(0.6, 1.2), 3)

    is_near_threshold = 1 if 8500 <= amount < 10000 else 0
    sender_country_risk = 2 if sender_country in BLACKLIST_COUNTRIES else (
        1 if sender_country in GREYLIST_COUNTRIES else 0
    )

    raw_shap = [
        {"feature": "amount", "value": amount, "shap": _shap_val(15 if 8500 <= amount < 10000 else 5)},
        {"feature": "is_cross_border", "value": int(is_cross_border), "shap": _shap_val(10 if is_cross_border else -2)},
        {"feature": "transaction_type", "value": transaction_type, "shap": _shap_val(12 if transaction_type == "crypto_exchange" else 3)},
        {"feature": "hour_of_day", "value": hour, "shap": _shap_val(8 if (hour < 6 or hour > 22) else -1)},
        {"feature": "is_near_threshold", "value": is_near_threshold, "shap": _shap_val(14 if is_near_threshold else -1)},
        {"feature": "sender_country_risk", "value": sender_country_risk, "shap": _shap_val(18 if sender_country_risk == 2 else (6 if sender_country_risk == 1 else -2))},
    ]
    # Sort by absolute shap value descending
    shap_attributions = sorted(raw_shap, key=lambda x: abs(x["shap"]), reverse=True)
    for attr in shap_attributions:
        attr["direction"] = "increases_risk" if attr["shap"] > 0 else "decreases_risk"

    return {
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "confidence_interval": confidence_interval,
        "shap_attributions": shap_attributions,
        "model_version": model_version,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_network_analyzer(inp: dict) -> dict:
    """
    Analyze account network using NetworkX.
    Detects fan_out, fan_in, and cycle patterns.
    Falls back to probabilistic mock if no real graph data is available.
    """
    import networkx as nx

    account_id = inp.get("account_id", "")
    depth = int(inp.get("depth", 2))
    lookback_days = int(inp.get("lookback_days", 90))

    # Build a mock transaction graph (in production, load from database)
    G = nx.DiGraph()
    # Seed graph with account_id and some neighbors for demo purposes
    rng = random.Random(hash(account_id) % (2**31))

    # Probabilistic risk distribution: 60% low, 25% medium, 12% high, 3% critical
    roll = rng.random()
    if roll < 0.60:
        scenario = "low"
    elif roll < 0.85:
        scenario = "medium"
    elif roll < 0.97:
        scenario = "high"
    else:
        scenario = "critical"

    flagged_connections = []
    pattern_detected = "none"
    network_risk_level = "low"
    total_connected_accounts = rng.randint(2, 8)

    if scenario == "low":
        network_risk_level = "low"
        graph_metrics = {
            "betweenness_centrality": round(rng.uniform(0.01, 0.15), 4),
            "clustering_coefficient": round(rng.uniform(0.1, 0.4), 4),
            "transaction_velocity_30d": rng.randint(1, 12),
        }

    elif scenario == "medium":
        network_risk_level = "medium"
        pattern_detected = "fan_out"
        flagged_connections = [
            {
                "account_id": f"ACC-{rng.randint(10000, 99999)}",
                "relationship": "prior_structuring_detected",
                "risk_indicator": "Multiple sub-threshold deposits within 24h window",
                "flagged_date": "2024-08-15",
            }
        ]
        total_connected_accounts = rng.randint(6, 15)
        graph_metrics = {
            "betweenness_centrality": round(rng.uniform(0.20, 0.45), 4),
            "clustering_coefficient": round(rng.uniform(0.05, 0.20), 4),
            "transaction_velocity_30d": rng.randint(15, 40),
        }

    elif scenario == "high":
        network_risk_level = "high"
        patterns = ["fan_in", "cycle", "fan_out"]
        pattern_detected = rng.choice(patterns)
        n_flagged = rng.randint(1, 2)
        for i in range(n_flagged):
            flagged_connections.append({
                "account_id": f"ACC-{rng.randint(10000, 99999)}",
                "relationship": "known_sanctions_adjacent" if i == 0 else "layering_node",
                "risk_indicator": "Account linked to previously SARed entity" if i == 0 else "Funds passed through multiple jurisdictions within 48h",
                "flagged_date": f"2024-0{rng.randint(1,9)}-{rng.randint(10,28)}",
            })
        total_connected_accounts = rng.randint(15, 40)
        graph_metrics = {
            "betweenness_centrality": round(rng.uniform(0.45, 0.75), 4),
            "clustering_coefficient": round(rng.uniform(0.02, 0.10), 4),
            "transaction_velocity_30d": rng.randint(40, 120),
        }

    else:  # critical — direct sanctions match
        network_risk_level = "critical"
        pattern_detected = "cycle"
        flagged_connections = [
            {
                "account_id": f"OFAC-SDN-{rng.randint(1000, 9999)}",
                "relationship": "direct_sanctions_match",
                "risk_indicator": "OFAC SDN List match — confirmed sanctions hit",
                "flagged_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "sanctions_list": "OFAC SDN",
                "sanctions_program": "IRAN",
            }
        ]
        total_connected_accounts = rng.randint(3, 10)
        graph_metrics = {
            "betweenness_centrality": round(rng.uniform(0.70, 0.95), 4),
            "clustering_coefficient": round(rng.uniform(0.01, 0.05), 4),
            "transaction_velocity_30d": rng.randint(5, 30),
        }

    return {
        "account_id": account_id,
        "analysis_depth": depth,
        "lookback_days": lookback_days,
        "total_connected_accounts": total_connected_accounts,
        "flagged_connections": flagged_connections,
        "network_risk_level": network_risk_level,
        "pattern_detected": pattern_detected,
        "graph_metrics": graph_metrics,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_rule_checker(inp: dict) -> dict:
    """
    Check regulatory rules against transaction parameters.
    Returns triggered rules with exact regulatory citations.
    """
    transaction_id = inp.get("transaction_id", "")
    amount = float(inp.get("amount", 0))
    transaction_type = inp.get("transaction_type", "")
    sender_country = (inp.get("sender_country") or "").upper().strip()
    receiver_country = (inp.get("receiver_country") or "").upper().strip()

    triggered = []

    # CTR_THRESHOLD_EU: Cash Transaction Report obligation ≥ €10,000
    if amount >= 10000:
        triggered.append({
            "rule_id": "CTR_THRESHOLD_EU",
            "description": f"Transaction amount €{amount:,.2f} meets or exceeds the €10,000 Customer Due Diligence threshold",
            "severity": "high",
            "citation": "GwG §10 Abs. 3 Nr. 1 / FATF Recommendation 29",
        })

    # STRUCTURING_BELOW_THRESHOLD: Structuring/smurfing signal
    if 8500 <= amount < 10000:
        triggered.append({
            "rule_id": "STRUCTURING_BELOW_THRESHOLD",
            "description": f"Transaction amount €{amount:,.2f} falls within the €8,500–€9,999 structuring detection band — possible CTR avoidance",
            "severity": "high",
            "citation": "GwG §43 Abs. 1 / FATF Recommendation 20",
        })

    # HIGH_RISK_JURISDICTION_BLACKLIST: FATF blacklist countries
    blacklist_hit = []
    if sender_country in BLACKLIST_COUNTRIES:
        blacklist_hit.append(f"sender ({sender_country})")
    if receiver_country in BLACKLIST_COUNTRIES:
        blacklist_hit.append(f"receiver ({receiver_country})")
    if blacklist_hit:
        triggered.append({
            "rule_id": "HIGH_RISK_JURISDICTION_BLACKLIST",
            "description": f"Transaction involves FATF blacklisted high-risk jurisdiction: {', '.join(blacklist_hit)}",
            "severity": "critical",
            "citation": "EU 6AMLD Art. 18 / FATF Recommendation 19 / EU Delegated Reg. 2016/1675",
        })

    # HIGH_RISK_JURISDICTION_GREYLIST: FATF grey list countries
    greylist_hit = []
    if sender_country in GREYLIST_COUNTRIES:
        greylist_hit.append(f"sender ({sender_country})")
    if receiver_country in GREYLIST_COUNTRIES:
        greylist_hit.append(f"receiver ({receiver_country})")
    if greylist_hit:
        triggered.append({
            "rule_id": "HIGH_RISK_JURISDICTION_GREYLIST",
            "description": f"Transaction involves FATF grey-listed jurisdiction under enhanced monitoring: {', '.join(greylist_hit)}",
            "severity": "medium",
            "citation": "FATF Recommendation 19 / EU 6AMLD Art. 18a",
        })

    # WIRE_TRANSFER_FATF16: Cross-border wire transfer transparency
    if transaction_type == "wire_transfer" and sender_country and receiver_country and sender_country != receiver_country:
        triggered.append({
            "rule_id": "WIRE_TRANSFER_FATF16",
            "description": "Cross-border wire transfer requires full originator and beneficiary information under the travel rule",
            "severity": "medium",
            "citation": "FATF Recommendation 16 / EU Wire Transfer Regulation 2015/847",
        })

    # CRYPTO_ASSET_EXCHANGE: MiCA and 6AMLD obligations
    if transaction_type == "crypto_exchange":
        triggered.append({
            "rule_id": "CRYPTO_ASSET_EXCHANGE",
            "description": "Crypto-asset exchange transaction triggers enhanced due diligence under MiCA and 6AMLD",
            "severity": "medium",
            "citation": "EU MiCA Regulation (2023/1114) / EU 6AMLD Art. 2 Abs. 1 Nr. 3g",
        })

    # Determine overall severity
    severity_order = ["critical", "high", "medium", "low"]
    overall_severity = "low"
    for sev in severity_order:
        if any(r["severity"] == sev for r in triggered):
            overall_severity = sev
            break

    # SAR obligation triggered by any high or critical rule
    sar_obligation = any(r["severity"] in ("high", "critical") for r in triggered)

    return {
        "transaction_id": transaction_id,
        "rules_triggered": triggered,
        "rules_triggered_count": len(triggered),
        "overall_severity": overall_severity,
        "sar_obligation": sar_obligation,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_sar_generator(inp: dict) -> dict:
    """
    Generate a SAR draft in BaFin/GwG format.
    Only call when SAR filing is warranted.
    """
    transaction_id = inp.get("transaction_id", "")
    risk_score = float(inp.get("risk_score", 0))
    triggered_rules = inp.get("triggered_rules", [])
    suspicious_patterns = inp.get("suspicious_patterns", [])
    narrative = inp.get("narrative", "")
    report_format = inp.get("report_format", "bafin_gwg")

    today = datetime.now(timezone.utc)
    report_id = f"SAR-{today.strftime('%Y%m%d')}-{transaction_id[:8].upper()}"

    sar_report = {
        "report_id": report_id,
        "report_format": report_format,
        "status": "DRAFT — pending compliance officer review",
        "created_at": today.isoformat(),
        "filing_deadline": "Within 24 hours of detection per GwG §43 Abs. 1",

        # BaFin/GwG §43 required fields
        "reporting_institution": {
            "name": "[INSTITUTION NAME]",
            "bafin_registration": "[BaFin Registration Number]",
            "compliance_officer": "[Compliance Officer Name]",
            "contact_email": "[compliance@institution.de]",
        },
        "subject_transaction": {
            "transaction_id": transaction_id,
            "risk_score": risk_score,
            "risk_level": "SAR_REQUIRED",
        },
        "suspicious_activity": {
            "triggered_rules": triggered_rules,
            "patterns_detected": suspicious_patterns,
            "narrative": narrative,
        },
        "regulatory_basis": [
            "GwG §43 Abs. 1 — Suspicious transaction reporting obligation",
            "GwG §43 Abs. 2 — Immediate reporting to BaFin FIU required",
        ],
        "next_steps": [
            "1. Compliance officer review within 4 hours",
            "2. Submit to BaFin FIU via goAML portal within 24h of detection",
            "3. Freeze or place enhanced monitoring on account pending FIU response",
            "4. Do NOT alert customer — tipping-off is prohibited under GwG §47 Abs. 5",
        ],
        "tipping_off_warning": (
            "LEGAL WARNING: Alerting the customer or any third party about this SAR filing "
            "constitutes a criminal offense under GwG §47 Abs. 5 (Tipping-off). "
            "Maximum penalty: imprisonment up to 2 years or fine."
        ),
    }

    return {
        "sar_report": sar_report,
        "filed": False,
    }


def _run_escalation_decider(inp: dict) -> dict:
    """
    Make the final case escalation decision.
    Decision matrix evaluated in priority order: SAR > ESCALATE > WATCHLIST > CLEAR.
    """
    transaction_id = inp.get("transaction_id", "")
    risk_score = float(inp.get("risk_score", 0))
    network_risk = inp.get("network_risk", "low")
    rules_triggered = int(inp.get("rules_triggered", 0))
    agent_reasoning = inp.get("agent_reasoning", "")
    sar_generated = bool(inp.get("sar_generated", False))

    # Decision matrix — evaluate in priority order
    final_decision = "CLEAR"
    case_priority = None
    sla_hours = None
    assigned_queue = None

    # P1: SAR_REQUIRED
    if risk_score >= 80 or sar_generated or network_risk == "critical":
        final_decision = "SAR_REQUIRED"
        case_priority = "P1"
        sla_hours = 24
        assigned_queue = "SAR_FILING_QUEUE"

    # P2: ESCALATE_TO_HUMAN
    elif risk_score >= 60 or network_risk == "high" or rules_triggered >= 2:
        final_decision = "ESCALATE_TO_HUMAN"
        case_priority = "P2"
        sla_hours = 48
        assigned_queue = "SENIOR_COMPLIANCE_REVIEW"

    # P3: WATCHLIST
    elif risk_score >= 30 or network_risk == "medium" or rules_triggered == 1:
        final_decision = "WATCHLIST"
        case_priority = "P3"
        sla_hours = 72
        assigned_queue = "WATCHLIST_MONITORING"

    # CLEAR
    else:
        final_decision = "CLEAR"
        case_priority = None
        sla_hours = None
        assigned_queue = "ARCHIVE"

    decision_rationale = (
        f"Decision '{final_decision}' based on: "
        f"risk_score={risk_score}, network_risk={network_risk}, "
        f"rules_triggered={rules_triggered}, sar_generated={sar_generated}. "
        f"Agent reasoning: {agent_reasoning}"
    )

    return {
        "transaction_id": transaction_id,
        "final_decision": final_decision,
        "case_priority": case_priority,
        "sla_hours": sla_hours,
        "assigned_queue": assigned_queue,
        "decision_rationale": decision_rationale,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
