#!/usr/bin/env python3
"""T4 Basic Baselines -- Majority Class and Random Walk.

Computes trivial baselines for Market Movement Prediction:

1. **Majority class**: always predict the most common label.
2. **Random prior**: sample from empirical class distribution.
3. **Random walk**: report three zero-drift views for direction, magnitude,
   and continuous-delta correlation.

All baselines are evaluated across three tiers:
  - Tier 1: All data
  - Tier 2: Non-confounded only
  - Tier 3: Active signals (non-confounded + non-flat)

Usage:
    python baselines/t4/basic_baseline.py
    python baselines/t4/basic_baseline.py --local-dir ./data
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["up", "down", "flat"]
MAGNITUDE_LABELS = ["small", "medium", "large"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 basic baselines")
    parser.add_argument("--local-dir", default=None, help="Local data directory")
    parser.add_argument(
        "--rw-backend",
        choices=["hf_pre30m", "label_zero"],
        default="hf_pre30m",
        help="Random-walk backend: hf_pre30m matches pre-30m momentum logic; label_zero keeps old zero-drift baseline.",
    )
    parser.add_argument(
        "--ohlcv-path",
        default=None,
        help="Optional local path to raw market_ohlcv.json. If omitted, download from HF raw/market_ohlcv.json.",
    )
    parser.add_argument(
        "--posts-path",
        default=None,
        help="Optional local path to raw posts_no_text.jsonl for exact tweet created_at alignment.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(local_dir: Optional[str]) -> pd.DataFrame:
    import eventxbench

    if local_dir:
        result = eventxbench.load_task("t4", local_dir=local_dir)
    else:
        result = eventxbench.load_task("t4")
    if isinstance(result, tuple):
        return pd.concat(result, ignore_index=True)
    return result


def load_ohlcv_data(ohlcv_path: Optional[str]) -> pd.DataFrame:
    if ohlcv_path:
        path = Path(ohlcv_path)
    else:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="mlsys-io/EventXBench",
            filename="raw/market_ohlcv.json",
            repo_type="dataset",
        )
        path = Path(downloaded)

    df = pd.read_json(path)
    expected_cols = {"condition_id", "side", "timestamp", "close"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV file missing columns: {sorted(missing)}")

    df = df.copy()
    df["side"] = df["side"].astype(str).str.lower()
    df = df[df["side"] == "yes"].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df["close_price"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["condition_id", "ts", "close_price"]].dropna().sort_values(["condition_id", "ts"])
    return df


def load_post_times(posts_path: Optional[str], tweet_ids: pd.Series) -> pd.DataFrame:
    """Load tweet_id -> post_time mapping from raw posts metadata.

    Reads posts_no_text.jsonl in chunks and keeps only required tweet_ids.
    """
    if posts_path:
        path = Path(posts_path)
    else:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="mlsys-io/EventXBench",
            filename="raw/posts_no_text.jsonl",
            repo_type="dataset",
        )
        path = Path(downloaded)

    target_ids = set(pd.to_numeric(tweet_ids, errors="coerce").dropna().astype("int64").tolist())
    if not target_ids:
        return pd.DataFrame(columns=["tweet_id", "post_time"])

    matches = []
    for chunk in pd.read_json(path, lines=True, chunksize=200000):
        if "tweet_id" not in chunk.columns or "created_at" not in chunk.columns:
            continue
        chunk = chunk[["tweet_id", "created_at"]].copy()
        chunk["tweet_id"] = pd.to_numeric(chunk["tweet_id"], errors="coerce")
        hit = chunk[chunk["tweet_id"].isin(target_ids)]
        if not hit.empty:
            matches.append(hit)

    if not matches:
        return pd.DataFrame(columns=["tweet_id", "post_time"])

    out = pd.concat(matches, ignore_index=True).drop_duplicates(subset=["tweet_id"])
    out["post_time"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
    out["tweet_id"] = out["tweet_id"].astype("int64")
    return out[["tweet_id", "post_time"]].dropna(subset=["post_time"])


def build_pre30_rows_from_hf(
    t4_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    post_times_df: pd.DataFrame,
) -> pd.DataFrame:
    t4 = t4_df.copy()
    t4["tweet_id"] = pd.to_numeric(t4["tweet_id"], errors="coerce")
    if not post_times_df.empty:
        t4 = t4.merge(post_times_df, on="tweet_id", how="left")
    else:
        t4["post_time"] = pd.NaT

    # Fallback only if posts file mapping is unavailable for a row.
    fallback_post_time = pd.to_datetime(t4["created_at"], utc=True, errors="coerce")
    t4["post_time"] = t4["post_time"].fillna(fallback_post_time)
    t4["target_pre30_time"] = t4["post_time"] - pd.Timedelta(minutes=30)

    left = t4.sort_values(["target_pre30_time", "condition_id"]).reset_index(drop=True)
    right = ohlcv_df.sort_values(["ts", "condition_id"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="target_pre30_time",
        right_on="ts",
        by="condition_id",
        direction="backward",
    )

    merged = merged.rename(columns={"close_price": "price_pre_30m"})
    merged = merged[merged["price_pre_30m"].notna()].copy()
    merged["pre_delta"] = merged["price_t0"].astype(float) - merged["price_pre_30m"].astype(float)

    required = [
        "confound_flag",
        "direction_label",
        "magnitude_bucket",
        "pre_delta",
        "delta_30m",
        "delta_2h",
        "delta_6h",
    ]
    return merged[required].copy()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def majority_macro_f1(label_counts: dict[str, int], all_labels: list[str]) -> dict:
    """Compute majority-class accuracy and macro-F1."""
    total = sum(label_counts.values())
    if total == 0:
        return {"majority_label": None, "accuracy": 0.0, "macro_f1": 0.0}

    majority_label = max(label_counts.items(), key=lambda item: item[1])[0]
    majority_count = label_counts[majority_label]

    acc = majority_count / total
    # Majority predicts one class: F1 = 2p/(1+p) for that class, 0 for others
    p = majority_count / total
    f1_majority = 2 * p / (1 + p)
    macro_f1 = f1_majority / len(all_labels)

    return {"majority_label": majority_label, "accuracy": acc, "macro_f1": macro_f1}


def random_prior_f1(label_counts: dict[str, int]) -> dict:
    """Expected macro-F1 under random-prior sampling."""
    total = sum(label_counts.values())
    if total == 0:
        return {"expected_accuracy": 0.0, "expected_macro_f1": 0.0}
    priors = {k: v / total for k, v in label_counts.items()}
    exp_acc = sum(p ** 2 for p in priors.values())
    exp_f1 = sum(priors.values()) / len(priors)
    return {"expected_accuracy": exp_acc, "expected_macro_f1": exp_f1}


# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------


def build_tiers(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("Tier 1: All Data", df.copy()),
        ("Tier 2: Non-confounded", df[~df["confound_flag"].astype(bool)].copy()),
        (
            "Tier 3: Active (non-confounded + non-flat)",
            df[
                (~df["confound_flag"].astype(bool)) & (df["direction_label"] != "flat")
            ].copy(),
        ),
    ]


def evaluate_random_walk_direction_tier(tier_df: pd.DataFrame) -> float:
    """Random walk direction view: always predict flat."""
    if len(tier_df) == 0:
        return 0.0

    dir_true = tier_df["direction_label"].tolist()
    dir_pred = ["flat"] * len(tier_df)
    return float(accuracy_score(dir_true, dir_pred))


def evaluate_random_walk_magnitude_tier(tier_df: pd.DataFrame) -> float:
    """Random walk magnitude view: always predict small."""
    if len(tier_df) == 0:
        return 0.0

    mag_true = tier_df["magnitude_bucket"].tolist()
    mag_pred = ["small"] * len(tier_df)
    return float(f1_score(
        mag_true,
        mag_pred,
        labels=MAGNITUDE_LABELS,
        average="macro",
        zero_division=0,
    ))


def evaluate_random_walk_spearman_tier(tier_df: pd.DataFrame) -> Optional[float]:
    """Random walk continuous view: predict zero delta for all horizons."""
    if len(tier_df) == 0:
        return None

    actual = []
    for horizon in ("delta_30m", "delta_2h", "delta_6h"):
        values = tier_df[horizon].dropna().tolist()
        actual.extend([float(value) for value in values])

    predicted = [0.0] * len(actual)
    return spearman(predicted, actual)


def _macro_f1_from_pred(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def run_hf_pre30m_random_walk(pre30_df: pd.DataFrame) -> None:
    # Direction: sign(pre_delta)
    dir_df = pre30_df[pre30_df["direction_label"].notna()].copy()
    dir_df["pred_direction"] = dir_df["pre_delta"].apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
    )

    print_header("RANDOM WALK BASELINE -- Direction (pre-30m momentum)")
    dir_tiers = [
        ("Tier 1: All Data", dir_df),
        ("Tier 2: Non-confounded", dir_df[~dir_df["confound_flag"].astype(bool)]),
        (
            "Tier 3: Active (non-confounded + non-flat)",
            dir_df[
                (~dir_df["confound_flag"].astype(bool))
                & (dir_df["direction_label"].isin(["up", "down"]))
            ],
        ),
    ]
    for tier_name, tier in dir_tiers:
        n = len(tier)
        acc = (
            float((tier["direction_label"] == tier["pred_direction"]).mean())
            if n > 0
            else 0.0
        )
        print(f"  {tier_name} (n={n}):  Accuracy = {acc*100:.2f}%")

    # Magnitude: bucket(abs(pre_delta))
    mag_df = pre30_df[pre30_df["magnitude_bucket"].notna()].copy()

    def _pred_mag(x: float) -> str:
        a = abs(float(x))
        if a <= 0.02:
            return "small"
        if a <= 0.08:
            return "medium"
        return "large"

    mag_df["pred_magnitude"] = mag_df["pre_delta"].apply(_pred_mag)
    print_header("BASELINE PERFORMANCE REPORT (MAGNITUDE METRICS)")

    mag_tiers = [
        ("Tier 1: All Data", mag_df, ["small", "medium", "large"]),
        (
            "Tier 2: Non-confounded",
            mag_df[~mag_df["confound_flag"].astype(bool)],
            ["small", "medium", "large"],
        ),
        (
            "Tier 3: Active (non-confounded + medium/large)",
            mag_df[
                (~mag_df["confound_flag"].astype(bool))
                & (mag_df["magnitude_bucket"].isin(["medium", "large"]))
            ],
            ["medium", "large"],
        ),
    ]
    mag_results = []
    for _, tier, labels in mag_tiers:
        n = len(tier)
        macro_f1 = (
            _macro_f1_from_pred(
                tier["magnitude_bucket"].astype(str).tolist(),
                tier["pred_magnitude"].astype(str).tolist(),
                labels,
            )
            if n > 0
            else 0.0
        )
        mag_results.append((macro_f1, n))

    print(f"1. Macro-F1 (All Data):                            {mag_results[0][0]*100:.2f}%  (N={mag_results[0][1]})")
    print(f"2. Macro-F1 (Non-confounded):                      {mag_results[1][0]*100:.2f}%  (N={mag_results[1][1]})")
    print(f"3. Active Magnitudes Macro-F1 (Non-confounded):    {mag_results[2][0]*100:.2f}%  (N={mag_results[2][1]})")
    print("   *(Subset: actual magnitude in {medium, large})*")

    # Spearman: d30m=1x, d2h=4x, d6h=12x pre_delta
    print_header("RANDOM WALK SPEARMAN RHO REPORT (CONTINUOUS DELTA EVALUATION)")
    print("Prediction rule: d30m = pre_delta, d2h = 4*pre_delta, d6h = 12*pre_delta")
    spr_base = pre30_df.dropna(subset=["delta_30m", "delta_2h", "delta_6h", "direction_label"]).copy()
    horizon_scales = [("30m", "delta_30m", 1.0), ("2h", "delta_2h", 4.0), ("6h", "delta_6h", 12.0)]
    spr_tiers = [
        ("Tier 1: All Data", spr_base),
        ("Tier 2: Non-confounded", spr_base[~spr_base["confound_flag"].astype(bool)]),
        (
            "Tier 3: Active (non-confounded + non-flat)",
            spr_base[
                (~spr_base["confound_flag"].astype(bool))
                & (spr_base["direction_label"] != "flat")
            ],
        ),
    ]
    for tier_name, tier in spr_tiers:
        print("\n" + "-" * 84)
        print(f"{tier_name}")
        print(f"Samples: {len(tier)}")
        print("-" * 84)

        pred_all = []
        actual_all = []
        for horizon_name, col, scale in horizon_scales:
            actual = tier[col].astype(float).tolist()
            pred = (tier["pre_delta"].astype(float) * scale).tolist()

            spr_h = spearman(pred, actual)
            if spr_h is None:
                print(f"Spearman rho @ {horizon_name}: N/A (insufficient variance or samples)")
            else:
                print(f"Spearman rho @ {horizon_name}: {spr_h:.4f} (n={len(actual)})")

            actual_all.extend(actual)
            pred_all.extend(pred)

        spr_flat = spearman(pred_all, actual_all)
        if spr_flat is None:
            print("Spearman rho @ ALL horizons (flatten): N/A (insufficient variance or samples)")
        else:
            print(f"Spearman rho @ ALL horizons (flatten): {spr_flat:.4f} (n={len(actual_all)})")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_row(tier_name: str, n: int, metrics: dict, target: str) -> None:
    if target == "direction":
        val = metrics.get("accuracy", metrics.get("dir_acc", 0.0))
        print(f"  {tier_name} (n={n}):  Accuracy = {val*100:.2f}%")
    elif target == "magnitude":
        val = metrics.get("macro_f1", metrics.get("mag_f1", 0.0))
        print(f"  {tier_name} (n={n}):  Macro-F1 = {val*100:.2f}%")
    elif target == "spearman":
        spr = metrics.get("spearman")
        txt = "N/A" if spr is None else f"{spr:.4f}"
        print(f"  {tier_name} (n={n}):  Spearman = {txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print("Loading T4 data...")
    df = load_data(args.local_dir)
    print(f"Loaded {len(df)} rows")

    tiers = build_tiers(df)

    # === Majority Baseline ===
    print_header("MAJORITY BASELINE -- Direction")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["direction_label"].value_counts())
        if "non-flat" in tier_name:
            labels = ["up", "down"]
        else:
            labels = DIRECTION_LABELS
        m = majority_macro_f1(counts, labels)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"Always predict '{m['majority_label']}'  "
              f"Acc={m['accuracy']*100:.2f}%  Macro-F1={m['macro_f1']*100:.2f}%")

    print_header("MAJORITY BASELINE -- Magnitude")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["magnitude_bucket"].value_counts())
        m = majority_macro_f1(counts, MAGNITUDE_LABELS)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"Always predict '{m['majority_label']}'  "
              f"Acc={m['accuracy']*100:.2f}%  Macro-F1={m['macro_f1']*100:.2f}%")

    # === Random Prior Baseline ===
    print_header("RANDOM PRIOR BASELINE -- Direction")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["direction_label"].value_counts())
        r = random_prior_f1(counts)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"E[Acc]={r['expected_accuracy']*100:.2f}%  E[F1]={r['expected_macro_f1']*100:.2f}%")

    print_header("RANDOM PRIOR BASELINE -- Magnitude")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["magnitude_bucket"].value_counts())
        r = random_prior_f1(counts)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"E[Acc]={r['expected_accuracy']*100:.2f}%  E[F1]={r['expected_macro_f1']*100:.2f}%")

    # === Random Walk Baselines ===
    if args.rw_backend == "hf_pre30m":
        ohlcv_df = load_ohlcv_data(args.ohlcv_path)
        post_times_df = load_post_times(args.posts_path, df["tweet_id"])
        pre30_df = build_pre30_rows_from_hf(df, ohlcv_df, post_times_df)
        run_hf_pre30m_random_walk(pre30_df)
    else:
        print_header("RANDOM WALK BASELINE -- Direction (predict flat)")
        for tier_name, tier_df in tiers:
            dir_acc = evaluate_random_walk_direction_tier(tier_df)
            print(f"  {tier_name} (n={len(tier_df)}):  Accuracy = {dir_acc*100:.2f}%")

        print_header("RANDOM WALK BASELINE -- Magnitude (predict small)")
        for tier_name, tier_df in tiers:
            mag_f1 = evaluate_random_walk_magnitude_tier(tier_df)
            print(f"  {tier_name} (n={len(tier_df)}):  Macro-F1 = {mag_f1*100:.2f}%")

        print_header("RANDOM WALK BASELINE -- Spearman (predict delta=0)")
        for tier_name, tier_df in tiers:
            spr = evaluate_random_walk_spearman_tier(tier_df)
            spr_txt = "N/A" if spr is None else f"{spr:.4f}"
            print(f"  {tier_name} (n={len(tier_df)}):  Spearman = {spr_txt}")

if __name__ == "__main__":
    main()
