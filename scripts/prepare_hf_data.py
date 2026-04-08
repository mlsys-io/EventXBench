#!/usr/bin/env python3
"""Reorganize EventX raw data into a clean HF-ready directory structure.

This script reads the original data files from the EventX/ directory and writes
clean, standardized files into EventXBench/data/ ready for HF upload.

Usage:
    python scripts/prepare_hf_data.py --source-dir ../EventX --output-dir data/

What it does:
  1. Copies and renames task label files with consistent naming
  2. Splits T4/T5 into train/test (80/20, stratified, seed=42)
     and writes T6's fixed train/validation/test split files from its unified source
  3. Normalizes field names (e.g., T6 label shorthand)
  4. Creates a manifest.json listing all files and their metadata
  5. Skips large files (posts, OHLCV, market metadata) — upload those separately
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows -> {path}")


def prepare_t1(source: Path, output: Path):
    """T1: Already has train/test splits."""
    print("\n[T1] Market Volume Prediction")
    gt = source / "task1" / "groundtruth"

    train = load_jsonl(gt / "t1_market_level_train_premarket_only_new.jsonl")
    test = load_jsonl(gt / "t1_market_level_test_premarket_only_new.jsonl")

    write_jsonl(train, output / "t1" / "train.jsonl")
    write_jsonl(test, output / "t1" / "test.jsonl")

    labels = Counter(r.get("interest_label") for r in train + test)
    print(f"  Distribution: {dict(labels)}")
    return {"task": "t1", "train": len(train), "test": len(test), "labels": dict(labels)}


def prepare_t2(source: Path, output: Path):
    """T2: Test-only (resolution tier)."""
    print("\n[T2] Post-to-Market Linking")
    rows = load_jsonl(source / "task2" / "t2_groundtruth.jsonl")
    write_jsonl(rows, output / "t2" / "test.jsonl")
    return {"task": "t2", "test": len(rows)}


def prepare_t3(source: Path, output: Path):
    """T3: Test-only (resolution tier). Large JSON array."""
    print("\n[T3] Evidence Grading")
    with open(source / "task3" / "t3_final_graded.json", "r") as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = list(rows.values())

    # Write as JSONL for streaming compatibility
    write_jsonl(rows, output / "t3" / "test.jsonl")

    grades = Counter(r.get("final_grade") for r in rows)
    print(f"  Grade distribution: {dict(sorted(grades.items()))}")
    return {"task": "t3", "test": len(rows), "grades": dict(sorted(grades.items()))}


def prepare_t4(source: Path, output: Path):
    """T4: Stratified 80/20 split on direction_label."""
    print("\n[T4] Market Movement Prediction")
    rows = load_jsonl(source / "task4" / "t4_labels.jsonl")
    df = pd.DataFrame(rows)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df["direction_label"],
    )

    write_jsonl(train_df.to_dict("records"), output / "t4" / "train.jsonl")
    write_jsonl(test_df.to_dict("records"), output / "t4" / "test.jsonl")

    labels = Counter(df["direction_label"])
    print(f"  Direction distribution: {dict(labels)}")
    return {"task": "t4", "train": len(train_df), "test": len(test_df), "labels": dict(labels)}


def prepare_t5(source: Path, output: Path):
    """T5: Stratified 80/20 split on decay_class. Source: task5+7/t5(7)_label.jsonl."""
    print("\n[T5] Volume & Price Impact (Decay)")
    rows = load_jsonl(source / "task5+7" / "t5(7)_label.jsonl")
    df = pd.DataFrame(rows)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df["decay_class"],
    )

    write_jsonl(train_df.to_dict("records"), output / "t5" / "train.jsonl")
    write_jsonl(test_df.to_dict("records"), output / "t5" / "test.jsonl")

    labels = Counter(df["decay_class"])
    print(f"  Decay distribution: {dict(labels)}")
    return {"task": "t5", "train": len(train_df), "test": len(test_df), "labels": dict(labels)}


def prepare_t6(source: Path, output: Path):
    """T6: Fixed split from the unified t6_full_with_split.jsonl file."""
    print("\n[T6] Cross-Market Propagation")
    full_path = source / "task6" / "t6_full_with_split.jsonl"
    if not full_path.exists():
        raise FileNotFoundError(
            f"T6 requires the fixed unified file with split/features: {full_path}"
        )
    rows = load_jsonl(full_path)
    df = pd.DataFrame(rows)
    if "split" not in df.columns:
        raise ValueError("T6 unified file must contain a split column.")

    split_paths = {
        "train": output / "t6" / "train.jsonl",
        "val": output / "t6" / "validation.jsonl",
        "test": output / "t6" / "test.jsonl",
    }
    for split_name, path in split_paths.items():
        split_rows = df[df["split"] == split_name].to_dict("records")
        write_jsonl(split_rows, path)

    labels = Counter(df["label"])
    split_counts = Counter(df["split"])
    print(f"  Label distribution: {dict(labels)}")
    print(f"  Split distribution: {dict(split_counts)}")
    return {
        "task": "t6",
        "train": int(split_counts.get("train", 0)),
        "val": int(split_counts.get("val", 0)),
        "test": int(split_counts.get("test", 0)),
        "labels": dict(labels),
    }


def prepare_market_metadata(source: Path, output: Path):
    """Copy market metadata (fundamental info)."""
    print("\n[Markets] Market metadata")
    src = source / "market_foundamental.json"
    if src.exists():
        dst = output / "markets" / "market_fundamental.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Symlink/copy: {src} -> {dst}")
        print(f"  NOTE: Large file ({src.stat().st_size / 1024**2:.0f} MB). Upload separately to HF.")
    else:
        print("  SKIP: market_foundamental.json not found")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare EventX data for HF release")
    parser.add_argument("--source-dir", required=True, help="Path to EventX/ data directory")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()

    if not source.exists():
        raise SystemExit(f"Source directory not found: {source}")

    print(f"Source: {source}")
    print(f"Output: {output}")

    manifest = []
    manifest.append(prepare_t1(source, output))
    manifest.append(prepare_t2(source, output))
    manifest.append(prepare_t3(source, output))
    manifest.append(prepare_t4(source, output))
    manifest.append(prepare_t5(source, output))
    manifest.append(prepare_t6(source, output))

    prepare_market_metadata(source, output)

    # Write manifest
    manifest_path = output / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for m in manifest:
        train_n = m.get("train", "--")
        test_n = m.get("test", "--")
        print(f"  {m['task']}: train={train_n}, test={test_n}")

    print("\nLarge files to upload separately:")
    print("  - posts_no_text.jsonl (~9M rows, ~5.3 GB)")
    print("  - market_foundamental.json (market metadata, ~483 MB)")
    print("  - market_ohlcv.json (price/volume OHLCV, ~1.8 GB)")
    print("\nUse scripts/upload_to_hf.py to push everything to Hugging Face.")


if __name__ == "__main__":
    main()
