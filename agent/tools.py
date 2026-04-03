# Anthropic Tool-Use API tool definitions for AML-Shield agent

TOOLS = [
    {
        "name": "transaction_risk_scorer",
        "description": (
            "Score AML risk for a financial transaction using XGBoost model. "
            "Returns risk_score (0-100), confidence interval, and SHAP feature attributions "
            "explaining which features drove the score. "
            "Always call this tool first before any other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique identifier for the transaction"
                },
                "amount": {
                    "type": "number",
                    "description": "Transaction amount in EUR"
                },
                "sender_account": {
                    "type": "string",
                    "description": "Sender account identifier"
                },
                "receiver_account": {
                    "type": "string",
                    "description": "Receiver account identifier"
                },
                "transaction_type": {
                    "type": "string",
                    "enum": [
                        "wire_transfer",
                        "cash_deposit",
                        "cash_withdrawal",
                        "crypto_exchange",
                        "internal_transfer",
                        "card_payment"
                    ],
                    "description": "Type of financial transaction"
                },
                "timestamp": {
                    "type": "string",
                    "description": "Transaction timestamp in ISO 8601 format"
                },
                "sender_country": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code of sender (optional)"
                },
                "receiver_country": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code of receiver (optional)"
                },
                "is_cross_border": {
                    "type": "boolean",
                    "description": "Whether the transaction crosses national borders (optional)"
                }
            },
            "required": [
                "transaction_id",
                "amount",
                "sender_account",
                "receiver_account",
                "transaction_type",
                "timestamp"
            ]
        }
    },
    {
        "name": "entity_network_analyzer",
        "description": (
            "Analyze the network of accounts connected to a given account. "
            "Detects structuring (smurfing), layering, fan-out, fan-in, and cycle patterns. "
            "Returns graph metrics, flagged connections, and suspicious network paths. "
            "If any flagged_connections are returned, call this tool again with the flagged account as center."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier to use as the center node for network analysis"
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Network traversal depth (default: 2). Use depth=3 when risk_score > 80."
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days to look back for transaction history (default: 90)"
                }
            },
            "required": ["account_id"]
        }
    },
    {
        "name": "regulatory_rule_checker",
        "description": (
            "Check whether a transaction triggers regulatory reporting obligations. "
            "Covers FATF 40 Recommendations, EU 6AMLD, GwG (German Money Laundering Act), "
            "and Wire Transfer Regulation 2015/847. "
            "Returns triggered rules with severity levels and exact regulatory citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique identifier for the transaction"
                },
                "amount": {
                    "type": "number",
                    "description": "Transaction amount in EUR"
                },
                "transaction_type": {
                    "type": "string",
                    "enum": [
                        "wire_transfer",
                        "cash_deposit",
                        "cash_withdrawal",
                        "crypto_exchange",
                        "internal_transfer",
                        "card_payment"
                    ],
                    "description": "Type of financial transaction"
                },
                "sender_country": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code of sender (optional)"
                },
                "receiver_country": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code of receiver (optional)"
                },
                "account_id": {
                    "type": "string",
                    "description": "Account identifier for account-level rule checks (optional)"
                },
                "check_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific rule types to check (optional, defaults to all)"
                }
            },
            "required": ["transaction_id", "amount", "transaction_type"]
        }
    },
    {
        "name": "sar_report_generator",
        "description": (
            "Generate a Suspicious Activity Report (SAR) draft in BaFin/GwG format. "
            "Only call this tool when SAR filing is warranted (risk_score >= 80 or sanctions match). "
            "Synthesizes all prior analysis into a structured report for BaFin FIU submission via goAML portal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique identifier for the transaction"
                },
                "risk_score": {
                    "type": "number",
                    "description": "Risk score from transaction_risk_scorer (0-100)"
                },
                "triggered_rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of rule IDs triggered by regulatory_rule_checker"
                },
                "suspicious_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of suspicious patterns detected (e.g., structuring, layering, sanctions_match)"
                },
                "narrative": {
                    "type": "string",
                    "description": "Human-readable narrative describing the suspicious activity and evidence"
                },
                "report_format": {
                    "type": "string",
                    "enum": ["bafin_gwg", "fincen_sar", "eu_amld"],
                    "description": "Report format to generate (optional, default: bafin_gwg)"
                }
            },
            "required": [
                "transaction_id",
                "risk_score",
                "triggered_rules",
                "suspicious_patterns",
                "narrative"
            ]
        }
    },
    {
        "name": "case_escalation_decider",
        "description": (
            "Make the final case escalation decision based on all gathered evidence. "
            "Always call this tool last, after all other relevant tools have been called. "
            "Returns final decision (CLEAR/WATCHLIST/ESCALATE_TO_HUMAN/SAR_REQUIRED), "
            "case priority, SLA hours, and assigned queue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique identifier for the transaction"
                },
                "risk_score": {
                    "type": "number",
                    "description": "Risk score from transaction_risk_scorer (0-100)"
                },
                "network_risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Network risk level from entity_network_analyzer"
                },
                "rules_triggered": {
                    "type": "integer",
                    "description": "Number of regulatory rules triggered by regulatory_rule_checker"
                },
                "agent_reasoning": {
                    "type": "string",
                    "description": "Agent's reasoning summary for the final decision"
                },
                "sar_generated": {
                    "type": "boolean",
                    "description": "Whether a SAR report was generated in this analysis (optional)"
                }
            },
            "required": [
                "transaction_id",
                "risk_score",
                "network_risk",
                "rules_triggered",
                "agent_reasoning"
            ]
        }
    }
]
