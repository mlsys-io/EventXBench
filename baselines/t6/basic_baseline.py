#!/usr/bin/env python3
"""T6 Basic Baselines -- Cross-Market Propagation.

Evaluates majority-class and random-prior baselines for T6 cross-market
propagation classification.  Reports Macro-F1.

Usage:
    python -m baselines.t6.basic_baseline
    python -m baselines.t6.basic_baseline --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

try:
    from .data_utils import LABEL_ORDER, load_t6_dataframe, train_eval_frames
except ImportError:
    from data_utils import LABEL_ORDER, load_t6_dataframe, train_eval_frames


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p == lab)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != lab and p == lab)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _majority_baseline(y_true: list[str], labels: list[str]) -> dict:
    counts = Counter(y_true)
    majority = counts.most_common(1)[0][0]
    y_pred = [majority] * len(y_true)
    mf1 = _macro_f1(y_true, y_pred, labels)
    return {
        "baseline": "majority",
        "majority_label": majority,
        "n": len(y_true),
        "macro_f1": mf1,
    }


def _random_baseline(y_true: list[str], labels: list[str], seeds: list[int] | None = None, train_labels: list[str] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    prior_source = train_labels if train_labels is not None else y_true
    counts = Counter(prior_source)
    total = len(y_true)
    priors = np.array([counts.get(lab, 0) / len(prior_source) for lab in labels])

    f1_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(labels, size=total, p=priors).tolist()
        f1_scores.append(_macro_f1(y_true, y_pred, labels))

    return {
        "baseline": "random_prior",
        "seeds": seeds,
        "n": total,
        "mean_macro_f1": float(np.mean(f1_scores)),
        "per_seed_macro_f1": [round(f, 4) for f in f1_scores],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T6 basic baselines")
    parser.add_argument("--repo", default="mlsys-io/EventXBench")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--feature-file", default=None,
                        help="Path to feature JSONL with split column (e.g. t6_db_features.jsonl)")
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "all"],
        default="test",
        help="Evaluation split. Use 'all' only for reproducing all-clean LLM-style runs.",
    )
    parser.add_argument("--include-confounded-eval", action="store_true")
    parser.add_argument("--include-insufficient", action="store_true")
    args = parser.parse_args()

    full_df = load_t6_dataframe(args.feature_file, args.local_dir, repo=args.repo)
    train_df, test_df = train_eval_frames(
        full_df,
        eval_split=args.eval_split,
        include_confounded_eval=args.include_confounded_eval,
        include_insufficient=args.include_insufficient,
    )

    eval_labels = LABEL_ORDER
    y_true = test_df["label"].tolist()

    print(f"T6 train: {len(train_df)}, test: {len(test_df)}")
    print(f"Test class distribution: {dict(Counter(y_true))}")

    # Majority baseline (majority from train)
    majority_label = train_df["label"].value_counts().idxmax()
    y_pred_maj = [majority_label] * len(y_true)
    mf1_maj = _macro_f1(y_true, y_pred_maj, eval_labels)
    print(f"\n[Majority] always predict '{majority_label}'")
    print(f"  Macro-F1: {mf1_maj:.4f}")

    # Random baseline (priors from train)
    rand = _random_baseline(y_true, eval_labels, train_labels=train_df["label"].tolist())
    print(f"\n[Random Prior] sample from training distribution")
    print(f"  Mean Macro-F1: {rand['mean_macro_f1']:.4f}")
    print(f"  Per-seed: {rand['per_seed_macro_f1']}")


if __name__ == "__main__":
    main()
