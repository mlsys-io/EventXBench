#!/usr/bin/env python3
"""T5/T7 Basic Baselines -- Impact Persistence (Decay Classification).

Evaluates majority-class and random-prior baselines for decay_class,
aligned with the Task7-style dummy baseline report format.

Note: In the original codebase this task is referred to as T7 / task5+7,
but in the paper and public release it is T5.

Usage:
    python -m baselines.t5.basic_baseline
    python -m baselines.t5.basic_baseline --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

import eventxbench

DECAY_LABELS = ["transient", "sustained", "reversal"]


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
def compute_majority_macro_f1(label_counts: dict[str, int]) -> tuple[float, str | None]:
    total = sum(label_counts.values())
    if total == 0 or not label_counts:
        return 0.0, None

    majority_label, majority_count = sorted(
        label_counts.items(), key=lambda x: x[1], reverse=True
    )[0]
    p_majority = majority_count / total
    num_classes = len(label_counts)

    # Majority predicts one class for all samples.
    macro_f1 = (2.0 * p_majority / (1.0 + p_majority)) / num_classes
    return macro_f1 * 100.0, majority_label


def compute_random_prior_macro_f1(label_counts: dict[str, int]) -> float:
    total = sum(label_counts.values())
    if total == 0 or not label_counts:
        return 0.0

    # Under random-prior prediction, expected F1 for class c equals p(c).
    priors = [cnt / total for cnt in label_counts.values()]
    return (sum(priors) / len(priors)) * 100.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T5 basic baselines")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    data = eventxbench.load_task("t5", local_dir=args.local_dir)
    if isinstance(data, tuple):
        df = pd.concat(data, ignore_index=True)
    else:
        df = data

    df = df[df["decay_class"].isin(DECAY_LABELS)].reset_index(drop=True)

    tier_frames: list[tuple[str, object]] = [("1. All Data", df)]
    if "confound_flag" in df.columns:
        tier_frames.append(("2. Non-confounded", df[df["confound_flag"] == False].reset_index(drop=True)))

    print("\n" + "=" * 84)
    print("TASK 7 DUMMY BASELINES REPORT (TARGET: decay_class)")
    print("=" * 84)

    for tier_name, tier_df in tier_frames:
        labels = tier_df["decay_class"].dropna().astype(str).tolist()
        label_counts = dict(Counter(labels))
        total_samples = sum(label_counts.values())

        print(f"\n--- Tier: {tier_name} ---")
        if total_samples == 0:
            print("No data.")
            continue

        majority_macro_f1, majority_label = compute_majority_macro_f1(label_counts)
        random_macro_f1 = compute_random_prior_macro_f1(label_counts)

        print(f"Total Samples: {total_samples}")
        dist_items = []
        for label, cnt in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            pct = cnt / total_samples * 100.0
            dist_items.append(f"{label}({cnt}/{total_samples}={pct:.1f}%)")
        print("Class Distribution: " + ", ".join(dist_items))
        print(f"[*] Majority Baseline Macro-F1 (always predict '{majority_label}'): {majority_macro_f1:.2f}%")
        print(f"[*] Random Prior Baseline Expected Macro-F1:               {random_macro_f1:.2f}%")


if __name__ == "__main__":
    main()
