"""
XGBoost AML model training pipeline.
Requires IBM AMLworld data in data/ — no synthetic fallback.

AML cost asymmetry:
  FN (missed laundering) → criminal escapes, regulatory fine, reputational damage — HIGH cost
  FP (false alarm)       → enters AI agent review queue, O(seconds) to clear — LOW cost

Objective: maximise recall subject to recall >= TARGET_RECALL.
We use the full imbalance ratio as scale_pos_weight (not sqrt) so the model
genuinely penalises false negatives during training.

For an interactive, documented version see models/train.ipynb.
"""

import os
import json
import glob
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
)
import xgboost as xgb
import joblib

from models.features import build_features, FEATURE_COLUMNS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT         = "models/xgboost_aml.pkl"
FEATURE_NAMES_OUTPUT = "models/feature_names.json"

# ── AML business objective ────────────────────────────────────────────────────
# Minimum acceptable recall.  Any threshold that drops below this is rejected
# regardless of how well it improves precision.
# Tune upward (e.g. 0.90) for stricter compliance environments.
TARGET_RECALL = float(os.environ.get("AML_TARGET_RECALL", "0.80"))

PAYMENT_FORMAT_MAP = {
    "Wire":         "wire_transfer",
    "Cheque":       "wire_transfer",
    "Credit Card":  "card_payment",
    "ACH":          "internal_transfer",
    "Cash":         "cash_deposit",
    "Bitcoin":      "crypto_exchange",
    "Reinvestment": "internal_transfer",
}


def _find_trans_csvs(data_dir: str) -> list[str]:
    all_files = glob.glob(os.path.join(data_dir, "*Trans*.csv"))
    return sorted(f for f in all_files if "paysim" not in f.lower())


def _load_csv(path: str, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    if "Amount Paid" in df.columns:
        out["amount"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0)
    else:
        out["amount"] = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0.0)

    out["timestamp"]       = pd.to_datetime(df["Timestamp"], errors="coerce").dt.tz_localize("UTC")
    out["is_laundering"]   = df["Is Laundering"].astype(int)
    out["transaction_type"]= df["Payment Format"].map(PAYMENT_FORMAT_MAP).fillna("wire_transfer")
    out["is_cross_border"] = (df["From Bank"].astype(str) != df["To Bank"].astype(str)).astype(int)
    out["sender_country"]  = "DE"
    out["receiver_country"]= "DE"
    return out.reset_index(drop=True)


def _recall_first_threshold(
    proba: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> tuple[float, float, float]:
    """
    Pick the threshold that maximises precision while keeping recall >= target_recall.

    Returns (threshold, precision_at_threshold, recall_at_threshold).

    AML rationale:
    - We fix a recall floor (criminals captured rate).
    - Within that constraint, pick the highest precision to limit the AI review queue.
    - If no threshold meets the recall floor, fall back to the highest-recall point.
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels, proba)

    # precision_recall_curve returns one extra point at the end (threshold=1.0
    # implied), so prec_arr and rec_arr have len = len(thresh_arr) + 1.
    # Align: iterate over the interior points that have explicit thresholds.
    candidates = [
        (float(p), float(r), float(t))
        for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thresh_arr)
        if r >= target_recall
    ]

    if candidates:
        # Maximise precision within the recall floor
        best_prec, best_rec, best_thresh = max(candidates, key=lambda x: x[0])
    else:
        # No threshold achieves target recall — take the best recall we can get
        logger.warning(
            f"⚠  No threshold achieves recall ≥ {target_recall:.0%}. "
            f"Using highest-recall point. Consider increasing sample_size or target_recall."
        )
        best_idx = int(np.argmax(rec_arr[:-1]))
        best_prec  = float(prec_arr[best_idx])
        best_rec   = float(rec_arr[best_idx])
        best_thresh= float(thresh_arr[best_idx])

    return best_thresh, best_prec, best_rec


def train(data_dir: str = "data/", sample_size: int = 500_000) -> None:
    """Main training entry point — recall-first AML objective."""
    logger.info("Starting AML-Shield XGBoost training  [TARGET_RECALL=%.0f%%]", TARGET_RECALL * 100)

    csv_files = _find_trans_csvs(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            "\n❌ No IBM AMLworld CSV files found in data/\n"
            "   Files must match: *Trans*.csv (excluding PaySim)\n"
        )

    frames = []
    for path in csv_files:
        try:
            raw = _load_csv(path, nrows=sample_size)
            frames.append(raw)
            pos = int(raw["Is Laundering"].sum())
            logger.info(f"  {os.path.basename(path):35s} → {len(raw):>8,} rows | {pos} illicit")
        except Exception as exc:
            logger.warning(f"Could not load {path}: {exc}")

    if not frames:
        raise ValueError("❌ All CSV files failed to load.")

    raw_all = pd.concat(frames, ignore_index=True)
    df      = _preprocess(raw_all)
    df      = df.dropna(subset=["amount", "is_laundering"]).reset_index(drop=True)

    positive_rate = df["is_laundering"].mean()
    logger.info(f"Dataset: {len(df):,} rows | {positive_rate:.4%} illicit")

    X = build_features(df)
    y = df["is_laundering"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    # ── scale_pos_weight: full imbalance ratio ────────────────────────────────
    # AML cost asymmetry: FN (missed crime) >> FP (AI review queue entry).
    # Using the full ratio tells XGBoost that every positive is worth as much
    # as ~N negatives.  Previously we used sqrt(ratio) to protect precision,
    # but with a three-tier architecture where FP merely enter an AI queue
    # (low cost), the correct objective is to minimise FN — so full ratio it is.
    # The recall-first threshold selection below controls the precision/recall
    # operating point without retraining.
    scale_pos_weight = float(n_neg) / max(float(n_pos), 1.0)
    logger.info(
        f"Imbalance: {n_neg:,} neg / {n_pos:,} pos  →  "
        f"scale_pos_weight={scale_pos_weight:.0f}  (full ratio, FN-penalising)"
    )

    model = xgb.XGBClassifier(
        n_estimators         = 300,
        max_depth            = 6,
        learning_rate        = 0.05,
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        min_child_weight     = 1,
        scale_pos_weight     = scale_pos_weight,
        eval_metric          = "aucpr",   # PR-AUC: more informative than ROC under imbalance
        early_stopping_rounds= 20,
        random_state         = 42,
        verbosity            = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Recall-first threshold selection ─────────────────────────────────────
    best_threshold, thr_precision, thr_recall = _recall_first_threshold(
        y_proba, y_test.values, TARGET_RECALL
    )
    y_pred = (y_proba >= best_threshold).astype(int)

    auc      = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    rec      = recall_score(y_test, y_pred, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)

    fn_count = int(cm[1][0])
    fp_count = int(cm[0][1])

    print("\n" + "=" * 60)
    print("AML-Shield XGBoost — Training Results")
    print("=" * 60)
    print(f"  AUC-ROC              : {auc:.4f}")
    print(f"  PR-AUC               : {avg_prec:.4f}")
    print(f"  Target recall        : {TARGET_RECALL:.0%}")
    print(f"  Threshold selected   : {best_threshold:.4f}")
    print(f"  Recall (achieved)    : {rec:.4f}  ← criminals caught")
    print(f"  Precision            : {prec:.4f}  ← proportion of flags that are real")
    print(f"  F1 Score             : {f1:.4f}")
    print(f"  False Negatives      : {fn_count}   ← escaped (minimise this)")
    print(f"  False Positives      : {fp_count}  ← enter AI review queue")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}   TP={cm[1][1]}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    meta = {
        "feature_columns":  FEATURE_COLUMNS,
        "threshold":        best_threshold,
        "target_recall":    TARGET_RECALL,
        "scale_pos_weight": scale_pos_weight,
        "achieved_recall":  round(rec, 4),
        "achieved_precision": round(prec, 4),
    }
    json.dump(meta, open(FEATURE_NAMES_OUTPUT, "w"), indent=2)
    logger.info(f"Model saved → {MODEL_OUTPUT}")
    logger.info(f"Meta  saved → {FEATURE_NAMES_OUTPUT}  (threshold={best_threshold:.4f}, recall={rec:.4f})")


if __name__ == "__main__":
    train()
