"""Load EventX data from Hugging Face or local files.

Supports two local directory layouts:
  1. Prepared HF layout: data/t1/train.jsonl, data/t1/test.jsonl, ...
  2. Original raw layout: task1/groundtruth/..., task4/t4_labels.jsonl, ...

The loader auto-detects which layout is present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

# Default HF repo -- update after publishing
HF_REPO = "mlsys-io/EventXBench"

VALID_TASKS = {"t1", "t2", "t3", "t4", "t5", "t6", "t7"}


def load_task(
    task: str,
    repo: str = HF_REPO,
    local_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load a task dataset.

    Args:
        task: Task name (t1-t6).
        repo: Hugging Face dataset repo ID.
        local_dir: If set, load from local directory instead of HF.
        split: If set, return only this split ("train" or "test").
               If None, returns (train, test) tuple for tasks with both splits,
               or just test DataFrame for tasks with only test.

    Returns:
        DataFrame or (train_df, test_df) tuple.
    """
    task = task.lower().strip()
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid: {sorted(VALID_TASKS)}")

    if local_dir:
        return _load_local(task, Path(local_dir), split)
    return _load_hf(task, repo, split)


def _load_hf(task: str, repo: str, split: Optional[str]):
    from datasets import load_dataset

    ds = load_dataset(repo, task, trust_remote_code=True)

    if split:
        return ds[split].to_pandas()

    splits = list(ds.keys())
    if "train" in splits and "test" in splits:
        return ds["train"].to_pandas(), ds["test"].to_pandas()
    elif "test" in splits:
        return ds["test"].to_pandas()
    else:
        return ds[splits[0]].to_pandas()


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# Prepared HF layout: data/t1/train.jsonl, data/t1/test.jsonl
_HF_LAYOUT = {
    "t1": {"train": "t1/train.jsonl", "test": "t1/test.jsonl"},
    "t2": {"test": "t2/test.jsonl"},
    "t3": {"test": "t3/test.jsonl"},
    "t4": {"train": "t4/train.jsonl", "test": "t4/test.jsonl"},
    "t5": {"train": "t5/train.jsonl", "test": "t5/test.jsonl"},
    "t6": {"train": "t6/train.jsonl", "test": "t6/test.jsonl"},
    "t7": {"train": "t7/train.jsonl", "test": "t7/test.jsonl"},
}

# Original raw layout from EventX/ directory
_RAW_LAYOUT = {
    "t1": {
        "train": "task1/groundtruth/t1_market_level_train_premarket_only_new.jsonl",
        "test": "task1/groundtruth/t1_market_level_test_premarket_only_new.jsonl",
    },
    "t2": {"test": "task2/t2_groundtruth.jsonl"},
    "t3": {"test": "task3/t3_final_graded.json"},
    "t4": {"full": "task4/t4_labels.jsonl"},
    "t5": {"full": "task5+7/t5(7)_label.jsonl"},
    "t6": {"full": "task6/task6_labels_v2_tuned_t35confound_full.jsonl"},
    "t7": {"full": "task5+7/t5(7)_label.jsonl"},
}


def _detect_layout(data_dir: Path, task: str) -> str:
    """Auto-detect which directory layout is present."""
    hf_path = data_dir / _HF_LAYOUT[task].get("test", _HF_LAYOUT[task].get("train", ""))
    if hf_path.exists():
        return "hf"
    return "raw"


def _load_local(task: str, data_dir: Path, split: Optional[str]):
    layout = _detect_layout(data_dir, task)

    if layout == "hf":
        return _load_hf_layout(task, data_dir, split)
    return _load_raw_layout(task, data_dir, split)


def _load_hf_layout(task: str, data_dir: Path, split: Optional[str]):
    """Load from prepared HF directory structure."""
    files = _HF_LAYOUT[task]

    if split:
        if split not in files:
            available = sorted(files.keys())
            raise ValueError(f"Task {task} has no '{split}' split. Available: {available}")
        return _load_jsonl(data_dir / files[split])

    if "train" in files and "test" in files:
        return _load_jsonl(data_dir / files["train"]), _load_jsonl(data_dir / files["test"])

    return _load_jsonl(data_dir / files["test"])


def _load_raw_layout(task: str, data_dir: Path, split: Optional[str]):
    """Load from original EventX/ raw data directory."""
    files = _RAW_LAYOUT[task]

    if "full" in files:
        path = data_dir / files["full"]
        if path.suffix == ".json":
            df = pd.read_json(path)
        else:
            df = _load_jsonl(path)

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        if split == "train":
            return train_df
        elif split == "test":
            return test_df
        return train_df, test_df

    if split:
        if split not in files:
            available = sorted(files.keys())
            raise ValueError(f"Task {task} has no '{split}' split. Available: {available}")
        path = data_dir / files[split]
        if path.suffix == ".json":
            return pd.read_json(path)
        return _load_jsonl(path)

    if "train" in files and "test" in files:
        return _load_jsonl(data_dir / files["train"]), _load_jsonl(data_dir / files["test"])

    test_path = data_dir / files["test"]
    if test_path.suffix == ".json":
        return pd.read_json(test_path)
    return _load_jsonl(test_path)


def load_markets(repo: str = HF_REPO, local_path: Optional[str] = None) -> pd.DataFrame:
    """Load market metadata."""
    if local_path:
        return pd.read_json(local_path)
    from datasets import load_dataset
    ds = load_dataset(repo, "markets")
    return ds[list(ds.keys())[0]].to_pandas()


def load_ohlcv(repo: str = HF_REPO, local_path: Optional[str] = None) -> pd.DataFrame:
    """Load market OHLCV time series."""
    if local_path:
        return pd.read_json(local_path)
    from datasets import load_dataset
    ds = load_dataset(repo, "ohlcv")
    return ds[list(ds.keys())[0]].to_pandas()
