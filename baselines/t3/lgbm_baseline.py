#!/usr/bin/env python3
"""T3 LightGBM Baseline -- Evidence Grading (Inference-Only).

Loads precomputed tweet and market embeddings and trains LightGBM classifier.
Reports Cohen's Kappa and Macro F1.

Usage:
    python t3_lgbm_inference.py
    python t3_lgbm_inference.py --local-dir /path/to/data
    python t3_lgbm_inference.py --tweet-emb tweet_embeddings.npy --market-emb market_embeddings.npy
"""
from __future__ import annotations

import argparse
from collections import Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import eventxbench


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
RULE_COLS = [
    "check_source",
    "check_time",
    "check_threshold",
    "check_predicate",
    "candidate_grade",
    "requires_official",
    "needs_llm",
]

CAT_COLS = ["check_source", "check_time", "check_threshold", "check_predicate"]


def build_features(
    df: pd.DataFrame,
    tweet_embeddings: np.ndarray,
    market_embeddings: np.ndarray,
) -> np.ndarray:
    rule_df = df[RULE_COLS].copy()

    le = LabelEncoder()
    for col in CAT_COLS:
        rule_df[col] = le.fit_transform(rule_df[col].astype(str))

    rule_df["requires_official"] = rule_df["requires_official"].astype(int)
    rule_df["needs_llm"] = rule_df["needs_llm"].astype(int)
    rule_df["candidate_grade"] = rule_df["candidate_grade"].fillna(3)

    return np.hstack([rule_df.values, tweet_embeddings, market_embeddings])


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
def split_by_market(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    markets = df["condition_id"].unique()
    train_markets, test_markets = train_test_split(
        markets, test_size=test_size, random_state=random_state
    )
    train_idx = df["condition_id"].isin(train_markets).values
    test_idx = df["condition_id"].isin(test_markets).values
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _run_lgbm(X_train, y_train, X_test, y_test) -> dict:
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "lgbm",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "kappa": kappa,
        "macro_f1": f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 LightGBM baseline (Inference Only)")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--tweet-emb", default="tweet_embeddings.npy")
    parser.add_argument("--market-emb", default="market_embeddings.npy")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    df["predicate_text"] = df["predicate"].fillna(df["market"])

    print(f"T3 samples: {len(df)}")
    print(f"Grade distribution: {dict(sorted(Counter(df['final_grade'].tolist()).items()))}")

    # Load precomputed embeddings
    print(f"\nLoading embeddings from disk...")
    tweet_embeddings = np.load(args.tweet_emb)
    market_embeddings = np.load(args.market_emb)
    print(f"Tweet embeddings shape:  {tweet_embeddings.shape}")
    print(f"Market embeddings shape: {market_embeddings.shape}")

    # Features
    X = build_features(df, tweet_embeddings, market_embeddings)
    y = df["final_grade"].values
    print(f"Feature matrix shape: {X.shape}")

    # Split
    train_idx, test_idx = split_by_market(df, args.test_size, args.random_state)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train & evaluate
    result = _run_lgbm(X_train, y_train, X_test, y_test)
    print(f"\n[LightGBM Results]")
    print(f"  Kappa={result['kappa']:.4f}, Macro F1={result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()