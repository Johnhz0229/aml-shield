"""Simple SQLite helper for persisting case results."""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./aml_shield.db").replace("sqlite:///", "")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the cases table if it does not exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                transaction_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                risk_score REAL,
                result_json TEXT NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """)
        conn.commit()


def save_case(transaction_id: str, decision: str, risk_score: Optional[float], result: dict):
    """Insert or replace a case result."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cases
                (transaction_id, decision, risk_score, result_json, analyzed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                decision,
                risk_score,
                json.dumps(result),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def get_cases(decision_filter: Optional[str] = None, min_risk: Optional[float] = None) -> list[dict]:
    """Return last 50 cases, optionally filtered."""
    query = "SELECT * FROM cases WHERE 1=1"
    params = []

    if decision_filter:
        query += " AND decision = ?"
        params.append(decision_filter)
    if min_risk is not None:
        query += " AND risk_score >= ?"
        params.append(min_risk)

    query += " ORDER BY analyzed_at DESC LIMIT 50"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_case_by_id(transaction_id: str) -> Optional[dict]:
    """Return a single case by transaction_id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
    return dict(row) if row else None
