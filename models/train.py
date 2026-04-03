"""
XGBoost AML model training pipeline.
Tries to load IBM AMLworld data; falls back to 2000 synthetic transactions.
"""

import os
import json
import glob
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import joblib

from models.features import build_features, FEATURE_COLUMNS, BLACKLIST_COUNTRIES, GREYLIST_COUNTRIES

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT = "models/xgboost_aml.pkl"
FEATURE_NAMES_OUTPUT = "models/feature_names.json"

HIGH_RISK_COUNTRIES = list(BLACKLIST_COUNTRIES)
GREYLIST = list(GREYLIST_COUNTRIES)
SAFE_COUNTRIES = ["DE", "FR", "GB", "NL", "AT", "CH", "US", "CA", "AU", "JP"]
TX_TYPES = ["wire_transfer", "cash_deposit", "cash_withdrawal", "crypto_exchange", "internal_transfer", "card_payment"]


def _load_ibm_amlworld(data_dir: str = "data/") -> pd.DataFrame | None:
    """Try to load IBM AMLworld transaction CSV from data/."""
    # Do not use paysim.csv
    candidates = [
        f for f in glob.glob(os.path.join(data_dir, "*Trans*.csv"))
        if "paysim" not in f.lower()
    ]
    if not candidates:
        return None

    dfs = []
    for path in candidates[:3]:  # limit to first 3 files to keep training fast
        try:
            df = pd.read_csv(path, nrows=5000)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {path}")
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic transaction data for training when IBM AMLworld is unavailable."""
    rng = np.random.default_rng(seed)
    n_legit = int(n * 0.70)
    n_suspicious = n - n_legit

    # Legitimate transactions (low risk profile)
    legit = pd.DataFrame({
        "amount": rng.uniform(10, 5000, n_legit),
        "transaction_type": rng.choice(["card_payment", "internal_transfer", "wire_transfer"], n_legit, p=[0.5, 0.3, 0.2]),
        "sender_country": rng.choice(SAFE_COUNTRIES, n_legit),
        "receiver_country": rng.choice(SAFE_COUNTRIES, n_legit),
        "is_cross_border": rng.choice([0, 1], n_legit, p=[0.7, 0.3]),
        "hour_of_day": rng.integers(7, 20, n_legit),
        "day_of_week": rng.integers(0, 5, n_legit),
        "is_laundering": 0,
    })

    # Suspicious transactions (high risk profile)
    # Mix of structuring, high-risk countries, night transactions, crypto
    susp_amounts = np.concatenate([
        rng.uniform(8500, 9999, int(n_suspicious * 0.35)),  # structuring band
        rng.uniform(10000, 50000, int(n_suspicious * 0.25)),  # above CTR
        rng.uniform(100, 5000, int(n_suspicious * 0.40)),    # small suspicious
    ])
    susp_amounts = susp_amounts[:n_suspicious]

    sender_countries = np.where(
        rng.random(n_suspicious) < 0.5,
        rng.choice(HIGH_RISK_COUNTRIES + GREYLIST, n_suspicious),
        rng.choice(SAFE_COUNTRIES, n_suspicious),
    )
    receiver_countries = np.where(
        rng.random(n_suspicious) < 0.5,
        rng.choice(HIGH_RISK_COUNTRIES + GREYLIST, n_suspicious),
        rng.choice(SAFE_COUNTRIES, n_suspicious),
    )

    susp = pd.DataFrame({
        "amount": susp_amounts,
        "transaction_type": rng.choice(
            ["wire_transfer", "cash_deposit", "crypto_exchange"], n_suspicious, p=[0.4, 0.35, 0.25]
        ),
        "sender_country": sender_countries,
        "receiver_country": receiver_countries,
        "is_cross_border": rng.choice([0, 1], n_suspicious, p=[0.2, 0.8]),
        "hour_of_day": np.concatenate([
            rng.integers(0, 6, int(n_suspicious * 0.4)),   # night
            rng.integers(22, 24, int(n_suspicious * 0.2)),  # late night
            rng.integers(6, 22, n_suspicious - int(n_suspicious * 0.6)),
        ])[:n_suspicious],
        "day_of_week": rng.integers(0, 7, n_suspicious),
        "is_laundering": 1,
    })

    df = pd.concat([legit, susp], ignore_index=True).sample(frac=1, random_state=seed)
    df["timestamp"] = pd.Timestamp("2024-01-01", tz="UTC")
    return df


def _prepare_ibm_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map IBM AMLworld columns to our feature schema."""
    col_map = {
        "Amount Paid": "amount",
        "Amount Received": "amount",   # fallback
        "Timestamp": "timestamp",
        "Is Laundering": "is_laundering",
        "Payment Format": "transaction_type",
        "Account": "sender_account",
        "Account.1": "receiver_account",
    }
    # Rename columns that exist — prefer Amount Paid over Amount Received
    df = df.copy()
    if "Amount Paid" in df.columns:
        df = df.rename(columns={"Amount Paid": "amount"})
    elif "Amount Received" in df.columns:
        df = df.rename(columns={"Amount Received": "amount"})

    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "amount" not in df.columns or "is_laundering" not in df.columns:
        return None

    # Map IBM payment formats to our transaction types
    type_map = {
        "Wire": "wire_transfer",
        "Cheque": "wire_transfer",
        "Credit Card": "card_payment",
        "ACH": "internal_transfer",
        "Cash": "cash_deposit",
        "Bitcoin": "crypto_exchange",
        "Reinvestment": "internal_transfer",
    }
    if "transaction_type" in df.columns:
        df["transaction_type"] = df["transaction_type"].map(type_map).fillna("wire_transfer")
    else:
        df["transaction_type"] = "wire_transfer"

    # IBM AMLworld doesn't always have country — use defaults
    if "sender_country" not in df.columns:
        df["sender_country"] = "DE"
    if "receiver_country" not in df.columns:
        df["receiver_country"] = "DE"

    df["is_cross_border"] = (df.get("sender_country", "DE") != df.get("receiver_country", "DE")).astype(int)

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp("2024-01-01", tz="UTC")

    return df


def train():
    """Main training entry point."""
    logger.info("Starting AML-Shield XGBoost training pipeline")

    # Try IBM AMLworld data first
    raw_df = _load_ibm_amlworld("data/")
    if raw_df is not None:
        df = _prepare_ibm_dataframe(raw_df)
        positive_rate = df["is_laundering"].mean() if df is not None and "is_laundering" in df.columns else 0
        if df is None or "is_laundering" not in df.columns or positive_rate < 0.01:
            logger.warning(
                f"IBM AMLworld data insufficient for training "
                f"(positive rate {positive_rate:.1%} < 1%) — falling back to synthetic"
            )
            df = _generate_synthetic_data(2000)
        else:
            logger.info(f"Using IBM AMLworld data: {len(df)} rows, {df['is_laundering'].mean():.1%} suspicious")
    else:
        logger.info("No IBM AMLworld data found — generating 2000 synthetic transactions")
        df = _generate_synthetic_data(2000)

    # Build features
    X = build_features(df)
    y = df["is_laundering"].astype(int)

    logger.info(f"Dataset: {len(X)} rows | Features: {X.shape[1]} | Positive rate: {y.mean():.1%}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("AML-Shield XGBoost Training Results")
    print("=" * 50)
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 50)

    # Save model and feature names
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    json.dump(FEATURE_COLUMNS, open(FEATURE_NAMES_OUTPUT, "w"))
    logger.info(f"Model saved to {MODEL_OUTPUT}")
    logger.info(f"Feature names saved to {FEATURE_NAMES_OUTPUT}")


if __name__ == "__main__":
    train()
