"""Feature engineering for AML transaction risk scoring."""

import numpy as np
import pandas as pd

# High-risk jurisdiction blacklist (FATF/EU Delegated Reg. 2016/1675)
BLACKLIST_COUNTRIES = {"IR", "KP", "MM", "SY", "YE", "AF"}

# High-risk jurisdiction greylist (FATF monitored)
GREYLIST_COUNTRIES = {"PK", "TR", "ML", "VN", "MZ", "TZ", "JO"}

FEATURE_COLUMNS = [
    "amount_log",
    "amount_eur",
    "is_near_ctr_threshold",
    "is_above_ctr",
    "is_round_number",
    "hour_of_day",
    "is_night",
    "is_weekend",
    "is_cross_border",
    "is_high_risk_country",
    "is_greylist_country",
    "transaction_type_wire",
    "transaction_type_crypto",
    "transaction_type_cash",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw transaction DataFrame.

    Input columns expected:
        amount, timestamp (or hour_of_day, day_of_week),
        is_cross_border, sender_country, receiver_country, transaction_type

    Returns a DataFrame with FEATURE_COLUMNS only.
    """
    feat = pd.DataFrame(index=df.index)

    # Amount features
    feat["amount_log"] = np.log1p(df["amount"].astype(float))
    feat["amount_eur"] = df["amount"].astype(float)
    feat["is_near_ctr_threshold"] = ((df["amount"] >= 8500) & (df["amount"] < 10000)).astype(int)
    feat["is_above_ctr"] = (df["amount"] >= 10000).astype(int)
    feat["is_round_number"] = (df["amount"] % 100 == 0).astype(int)

    # Time features
    if "hour_of_day" in df.columns:
        hour = df["hour_of_day"].astype(int)
        day_of_week = df.get("day_of_week", pd.Series(2, index=df.index)).astype(int)
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        hour = ts.dt.hour
        day_of_week = ts.dt.dayofweek

    feat["hour_of_day"] = hour
    feat["is_night"] = ((hour < 6) | (hour > 22)).astype(int)
    feat["is_weekend"] = (day_of_week >= 5).astype(int)

    # Geographic features
    feat["is_cross_border"] = df["is_cross_border"].astype(int)

    sender = df.get("sender_country", pd.Series("", index=df.index)).fillna("").str.upper()
    receiver = df.get("receiver_country", pd.Series("", index=df.index)).fillna("").str.upper()

    feat["is_high_risk_country"] = (
        sender.isin(BLACKLIST_COUNTRIES) | receiver.isin(BLACKLIST_COUNTRIES)
    ).astype(int)

    feat["is_greylist_country"] = (
        sender.isin(GREYLIST_COUNTRIES) | receiver.isin(GREYLIST_COUNTRIES)
    ).astype(int)

    # Transaction type one-hot
    tx_type = df.get("transaction_type", pd.Series("", index=df.index)).fillna("")
    feat["transaction_type_wire"] = (tx_type == "wire_transfer").astype(int)
    feat["transaction_type_crypto"] = (tx_type == "crypto_exchange").astype(int)
    feat["transaction_type_cash"] = (
        tx_type.isin(["cash_deposit", "cash_withdrawal"])
    ).astype(int)

    return feat[FEATURE_COLUMNS]
