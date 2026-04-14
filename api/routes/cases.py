"""GET /api/v1/cases — list and retrieve case results."""

import json
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query

from api.schemas import CaseSummary
from api.database import get_cases, get_case_by_id

router = APIRouter()


@router.get("/cases", response_model=list[CaseSummary])
def list_cases(
    decision: Optional[str] = Query(default=None),
    min_risk: Optional[float] = Query(default=None),
):
    rows = get_cases(decision_filter=decision, min_risk=min_risk)
    summaries = []
    for row in rows:
        result = json.loads(row["result_json"])
        summaries.append(
            CaseSummary(
                transaction_id=row["transaction_id"],
                routing=row.get("routing"),
                final_decision=row["decision"],
                risk_score=row["risk_score"],
                analyzed_at=datetime.fromisoformat(row["analyzed_at"]),
                duration_ms=result.get("duration_ms", 0),
            )
        )
    return summaries


@router.get("/cases/{transaction_id}")
def get_case(transaction_id: str):
    row = get_case_by_id(transaction_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case '{transaction_id}' not found")
    return json.loads(row["result_json"])
