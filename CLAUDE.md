# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose
AI-powered Anti-Money Laundering (AML) compliance agent. Portfolio project targeting Deutsche Bank AFC, Big4 FSA, and RegTech interviews. Core differentiator: every decision includes a full visible reasoning chain with regulatory citations.

## Tech Stack
- Python 3.11, FastAPI, uvicorn
- XGBoost + SHAP for ML scoring
- Anthropic Claude API — Tool-Use, ReAct pattern
- SQLite (dev), PostgreSQL (prod)
- Redis for caching
- React + Tailwind for frontend
- Docker Compose for deployment

## Project Structure
- `agent/` — Claude agent: `core.py`, `tools.py`, `executors.py`, `prompts.py`
- `api/` — FastAPI routes and Pydantic schemas
- `models/` — XGBoost training, SHAP inference, feature engineering
- `data/` — IBM AMLworld CSVs (never commit to git)
- `frontend/` — React app
- `tests/` — pytest suite

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env  # then fill in required vars

# Train ML model
python models/train.py

# Run API server
uvicorn api.main:app --reload

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/path/to/test_file.py::test_function_name -v
```

## Environment Variables
Required in `.env`: `ANTHROPIC_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `MODEL_PATH`

## Coding Rules
- All code comments and commit messages in English
- Commit message format: `feat: Phase N - description`
- Use IBM AMLworld dataset only — do not use PaySim
- Do not modify regulatory citations in `agent/prompts.py`
- After each phase: run tests, fix failures, then commit

## Agent Architecture Rules
- Agent must use conditional branching — not a fixed pipeline
- Every tool call input and output must be logged in `reasoning_chain`
- Legal citations must be exact: GwG §43, FATF Rec.16/19/20, EU 6AMLD
- When uncertain between WATCHLIST and SAR_REQUIRED: default to SAR_REQUIRED
