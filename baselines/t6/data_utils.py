from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import eventxbench

LABEL_ORDER = ["no_cross_market_effect", "primary_mover", "propagated_signal"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

FEATURE_COLS = [
    "like_count",
    "reply_count",
    "view_count",
    "follower_count",
    "price_t0",
    "primary_sigma_24h",
    "primary_baseline_points",
    "volume_24h_baseline",
    "category_sports",
    "category_crypto / digital assets",
    "category_elections / politics",
    "category_entertainment / awards",
    "category_company / product announcements",
    "finbert_pos_prob",
    "finbert_question_pos_prob",
    "sibling_count_graph",
    "sibling_mean_pairwise_bge_cosine",
    "primary_sibling_max_bge_cosine",
    "primary_sibling_mean_bge_cosine",
    "primary_sibling_top12_gap",
    "sibling_score_top1",
    "sibling_score_mean",
    "tweet_primary_bge_cosine",
    "tweet_sibling_max_bge_cosine",
    "tweet_sibling_mean_bge_cosine",
    "tweet_primary_minus_sibling_max_bge_gap",
]


def load_t6_dataframe(
    feature_file: Optional[str] = None,
    local_dir: Optional[str] = None,
    repo: str = "mlsys-io/EventXBench",
) -> pd.DataFrame:
    if feature_file:
        return pd.read_json(Path(feature_file), lines=True)

    frames = []
    for split_name in ("train", "validation", "test"):
        try:
            frame = eventxbench.load_task(
                "t6",
                repo=repo,
                local_dir=local_dir,
                split=split_name,
            )
        except Exception:
            if split_name == "validation":
                continue
            frames = []
            break
        frame = frame.copy()
        if "split" not in frame.columns:
            frame["split"] = "val" if split_name == "validation" else split_name
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)

    data = eventxbench.load_task("t6", repo=repo, local_dir=local_dir)
    if isinstance(data, tuple):
        train_df, test_df = data
        frames = []
        for split_name, frame in (("train", train_df), ("test", test_df)):
            frame = frame.copy()
            if "split" not in frame.columns:
                frame["split"] = split_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    return data.copy()


def clean_t6_dataframe(
    df: pd.DataFrame,
    *,
    include_insufficient: bool = False,
    include_confounded: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if "label" in out.columns:
        out = out[out["label"].isin(LABEL_ORDER)].copy()
    if not include_insufficient and "insufficient_data_flag" in out.columns:
        out = out[out["insufficient_data_flag"] == False].copy()
    if not include_confounded and "confound_flag" in out.columns:
        out = out[out["confound_flag"] == False].copy()
    return out.reset_index(drop=True)


def select_eval_split(df: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    if eval_split == "all":
        return df.copy().reset_index(drop=True)
    if "split" not in df.columns:
        raise ValueError(
            f"Cannot select eval split '{eval_split}' because T6 data has no split column."
        )
    return df[df["split"] == eval_split].copy().reset_index(drop=True)


def train_eval_frames(
    df: pd.DataFrame,
    *,
    eval_split: str = "test",
    include_confounded_eval: bool = False,
    include_insufficient: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = clean_t6_dataframe(
        df,
        include_insufficient=include_insufficient,
        include_confounded=True,
    )
    if "split" in filtered.columns:
        train_df = filtered[filtered["split"] == "train"].copy()
        eval_df = select_eval_split(filtered, eval_split)
    else:
        split_idx = int(len(filtered) * 0.8)
        train_df = filtered.iloc[:split_idx].copy()
        eval_df = filtered.iloc[split_idx:].copy()

    if not include_confounded_eval and "confound_flag" in eval_df.columns:
        eval_df = eval_df[eval_df["confound_flag"] == False].copy()

    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def available_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLS if col in df.columns]
