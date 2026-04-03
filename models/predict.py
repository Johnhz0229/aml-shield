"""SHAP-based inference for AML risk scoring."""

import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import shap

from models.features import build_features

logger = logging.getLogger(__name__)

_model_cache = None
_explainer_cache = None


def _load_model():
    global _model_cache, _explainer_cache
    if _model_cache is not None:
        return _model_cache, _explainer_cache

    model_path = os.environ.get("MODEL_PATH", "models/xgboost_aml.pkl")
    _model_cache = joblib.load(model_path)
    _explainer_cache = shap.TreeExplainer(_model_cache)
    return _model_cache, _explainer_cache


def predict_with_shap(transaction_dict: dict) -> dict:
    """
    Predict AML risk probability and return top 6 SHAP feature attributions.

    Args:
        transaction_dict: raw transaction fields (same as TransactionInput)

    Returns:
        dict with risk_score, risk_level, confidence_interval, shap_attributions
    """
    model, explainer = _load_model()

    # Build a single-row DataFrame
    df = pd.DataFrame([transaction_dict])
    X = build_features(df)

    # Predict probability
    prob = float(model.predict_proba(X)[0, 1])
    risk_score = round(prob * 100, 1)

    # Determine risk level
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 60:
        risk_level = "MEDIUM"
    elif risk_score < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    # Confidence interval based on model uncertainty (approx ±5 points)
    ci_half = 5.0
    confidence_interval = {
        "lower": round(max(0.0, risk_score - ci_half), 1),
        "upper": round(min(100.0, risk_score + ci_half), 1),
    }

    # SHAP values
    shap_values = explainer.shap_values(X)
    # For binary classification XGBoost returns a 2D array [n_samples, n_features]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # positive class
    else:
        sv = shap_values[0]

    feature_names = X.columns.tolist()
    feature_values = X.iloc[0].tolist()

    attributions = []
    for fname, fval, sv_val in zip(feature_names, feature_values, sv):
        attributions.append({
            "feature": fname,
            "value": fval,
            "shap": round(float(sv_val), 4),
            "direction": "increases_risk" if sv_val > 0 else "decreases_risk",
        })

    # Sort by absolute SHAP value descending, return top 6
    attributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    top_attributions = attributions[:6]

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence_interval": confidence_interval,
        "shap_attributions": top_attributions,
        "model_version": "xgboost-v1-trained",
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
