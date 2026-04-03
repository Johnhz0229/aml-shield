# AML-Shield

AI-powered Anti-Money Laundering compliance agent for BaFin-regulated financial institutions.

Unlike rule-based systems, AML-Shield uses a **conditional ReAct agent** (Claude + XGBoost) that reasons step-by-step, cites exact regulatory articles (GwG §43, FATF Rec.16/19/20, EU 6AMLD), and produces a full audit trail with every decision. Every output is explainable via SHAP feature attributions — meeting the interpretability expectations of compliance officers and regulators.

---

## Architecture

```mermaid
flowchart LR
    A[Transaction Input] --> B[FastAPI /api/v1/analyze]
    B --> C{AML Agent\nClaude Sonnet}

    C -->|1 Always first| D[transaction_risk_scorer\nXGBoost + SHAP]
    C -->|2 Conditional| E[entity_network_analyzer\nNetworkX graph]
    C -->|3 Regulatory check| F[regulatory_rule_checker\nGwG / FATF / 6AMLD]
    C -->|4 If SAR warranted| G[sar_report_generator\nBaFin GwG format]
    C -->|5 Always last| H[case_escalation_decider\nDecision matrix]

    H --> I{Final Decision}
    I --> J[CLEAR]
    I --> K[WATCHLIST]
    I --> L[ESCALATE_TO_HUMAN]
    I --> M[SAR_REQUIRED]

    M --> N[SAR → BaFin FIU\ngoAML portal]
```

---

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url>
cd aml-shield
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 2. Start with Docker
docker compose up

# 3. Analyze a transaction
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX-DEMO-001",
    "amount": 9750.00,
    "transaction_type": "wire_transfer",
    "sender_account": "DE89370400440532013000",
    "receiver_account": "IR-BANK-TEHRAN-44821",
    "sender_country": "DE",
    "receiver_country": "IR",
    "is_cross_border": true,
    "timestamp": "2024-03-15T02:34:00Z"
  }'
```

---

## Example: Iran Wire Transfer Case

**Input**

```json
{
  "transaction_id": "TX-IRAN-001",
  "amount": 9750.00,
  "transaction_type": "wire_transfer",
  "sender_country": "DE",
  "receiver_country": "IR",
  "is_cross_border": true,
  "timestamp": "2024-03-15T02:34:00Z"
}
```

**Output** (excerpt)

```json
{
  "transaction_id": "TX-IRAN-001",
  "final_decision": "SAR_REQUIRED",
  "risk_score": 97.4,
  "tool_calls_count": 5,
  "reasoning_chain": [
    {
      "type": "reasoning",
      "content": "THINK: Amount €9,750 is in the €8,500–€9,999 structuring band. Receiver is Iran (IR) — FATF blacklisted. Transaction at 02:34 AM. Risk score will be very high — Rule B: I must use depth=3 for network analysis."
    },
    {
      "type": "tool_call",
      "tool_name": "transaction_risk_scorer",
      "input": { "amount": 9750, "receiver_country": "IR", "timestamp": "2024-03-15T02:34:00Z" }
    },
    {
      "type": "tool_result",
      "tool_name": "transaction_risk_scorer",
      "result": {
        "risk_score": 97.4,
        "shap_attributions": [
          { "feature": "is_high_risk_country", "shap": 0.821, "direction": "increases_risk" },
          { "feature": "is_near_ctr_threshold", "shap": 0.614, "direction": "increases_risk" },
          { "feature": "is_night", "shap": 0.392, "direction": "increases_risk" }
        ]
      }
    }
  ],
  "sar_report": {
    "report_id": "SAR-20240315-TX-IRAN0",
    "status": "DRAFT — pending compliance officer review",
    "tipping_off_warning": "LEGAL WARNING: Alerting the customer about this SAR filing constitutes a criminal offense under GwG §47 Abs. 5."
  }
}
```

---

## Technical Highlights

- **Conditional ReAct agent** — not a fixed pipeline. The agent branches based on risk score (Rule A: skip network analysis for low-risk domestic; Rule B: depth=3 for score >80; Rule E: sanctions fast-track)
- **XGBoost + SHAP explainability** — every risk score is backed by ranked feature attributions showing which factors drove the result, meeting EU AI Act interpretability requirements
- **Regulatory grounding** — all rules cite exact articles: GwG §43 Abs. 1, FATF Rec.16/19/20, EU 6AMLD Art. 18, Wire Transfer Regulation 2015/847
- **SAR draft generation** — auto-generates BaFin/GwG-format SAR reports with tipping-off warnings (GwG §47 Abs. 5) and goAML submission guidance
- **Full reasoning chain** — every tool call input, output, and agent reasoning step is logged and returned to the compliance officer for audit trail

---

## Dataset: IBM AMLworld (NeurIPS 2023)

This project uses the IBM AMLworld dataset, not PaySim.

**Download:** https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Place the CSV files in `data/`:
```
data/HI-Small_Trans.csv
data/HI-Medium_Trans.csv
data/HI-Large_Trans.csv
data/LI-Small_Trans.csv
data/LI-Medium_Trans.csv
data/LI-Large_Trans.csv
```

If no data is present, `python -m models.train` automatically generates 2,000 synthetic transactions (70% legitimate / 30% suspicious) for development.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
aml-shield/
├── agent/
│   ├── core.py          # AMLAgent class, ReAct loop, AgentResult
│   ├── executors.py     # Tool executor functions (5 tools)
│   ├── prompts.py       # SYSTEM_PROMPT with regulatory citations
│   └── tools.py         # Anthropic Tool-Use API definitions
├── api/
│   ├── main.py          # FastAPI app, CORS, lifespan
│   ├── schemas.py       # Pydantic v2 request/response models
│   ├── database.py      # SQLite case persistence
│   └── routes/
│       ├── analyze.py   # POST /api/v1/analyze
│       ├── cases.py     # GET /api/v1/cases, GET /api/v1/cases/{id}
│       └── health.py    # GET /api/v1/health
├── models/
│   ├── features.py      # Feature engineering (14 features)
│   ├── train.py         # XGBoost training pipeline
│   └── predict.py       # SHAP inference
├── tests/
│   ├── test_agent.py    # Agent ReAct loop + executor unit tests
│   └── test_api.py      # FastAPI endpoint tests
├── data/                # IBM AMLworld CSVs (not committed)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
