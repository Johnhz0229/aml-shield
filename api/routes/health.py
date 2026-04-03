"""GET /api/v1/health — service health check."""

import os
import time
from fastapi import APIRouter

router = APIRouter()
_start_time = time.time()


@router.get("/health")
def health_check():
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_aml.pkl")
    return {
        "status": "ok",
        "model_loaded": os.path.exists(model_path),
        "uptime_seconds": round(time.time() - _start_time, 1),
        "version": "1.0.0",
    }
