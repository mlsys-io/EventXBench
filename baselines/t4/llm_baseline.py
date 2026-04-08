#!/usr/bin/env python3
"""T4 LLM Baseline -- Market Movement Prediction.

Asks a hosted LLM to predict price deltas (30m, 2h, 6h) for each tweet-market
pair, then derives direction (up/down/flat) and magnitude (small/medium/large)
from the predicted 2h delta.

NOTE: Tweet text is NOT included in the public label file. This baseline uses
only the fields available in t4_labels.jsonl (price_t0, condition_id). For
full-text features, users must rehydrate tweets via the Twitter/X API.

Evaluation tiers:
  - Tier 1: All data
  - Tier 2: Non-confounded only (confound_flag == False)
  - Tier 3: Active signals (non-confounded + non-flat direction)

Usage:
    python baselines/t4/llm_baseline.py --provider openai --model gpt-4o --shots 0
    python baselines/t4/llm_baseline.py --provider anthropic --shots 0 --limit 50
    python baselines/t4/llm_baseline.py --provider openai --dry-run --limit 5

API keys are read from environment variables (OPENAI_API_KEY / ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["up", "down", "flat"]
MAGNITUDE_LABELS = ["small", "medium", "large"]

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

SYSTEM_PROMPT = """\
You are a careful market-microreaction forecaster.
Use only the provided fields and output strict JSON only.

Task:
Predict YES-price deltas at three horizons: 30m, 2h, 6h.

Definition:
- delta_h means ABSOLUTE YES-price change from current_price to horizon h.
- future_price_h = current_price + delta_h
- future_price_h must stay in [0, 1]

Output schema (exact keys, no extras):
{
    "delta_30m": number,
    "delta_2h": number,
    "delta_6h": number
}
"""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 LLM baseline: market movement prediction")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--shots", type=int, default=0, help="Few-shot examples (0 = zero-shot)")
    parser.add_argument("--local-dir", default=None, help="Local data directory")
    parser.add_argument("--output", default="t4_llm_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max samples to evaluate")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=480,
        help="Randomly sample this many rows before inference (0 disables sampling)",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(local_dir: Optional[str]) -> pd.DataFrame:
    """Load T4 data. Returns a single DataFrame (no predefined train/test split)."""
    import eventxbench

    if local_dir:
        result = eventxbench.load_task("t4", local_dir=local_dir)
    else:
        result = eventxbench.load_task("t4")

    # load_task may return (train, test) tuple or single df
    if isinstance(result, tuple):
        return pd.concat(result, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


def clamp_delta(delta: float, price_t0: float) -> float:
    """Ensure price_t0 + delta stays in [0, 1]."""
    return max(-price_t0, min(1.0 - price_t0, delta))


def derive_direction(delta_2h: float) -> str:
    if delta_2h > 0.02:
        return "up"
    elif delta_2h < -0.02:
        return "down"
    return "flat"


def derive_magnitude(delta_2h: float) -> str:
    a = abs(delta_2h)
    if a <= 0.02:
        return "small"
    elif a <= 0.08:
        return "medium"
    return "large"


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


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p == lab)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != lab and p == lab)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_user_prompt(
    row: dict[str, Any],
    few_shot: list[dict[str, Any]],
) -> str:
    price = float(row.get("price_t0", 0.5))
    cid = row.get("condition_id", "unknown")
    price_low = -price
    price_high = 1.0 - price

    parts = ["Predict market movement after a tweet."]

    post_text = row.get("post_text")
    market_question = row.get("market_question")
    if post_text:
        parts.append(f"- post_text: {post_text}")
    if market_question:
        parts.append(f"- market_question: {market_question}")

    parts.append(f"- condition_id: {cid}")
    parts.append(f"- current_price: {price:.4f}")

    # Optional side features if upstream data contains them.
    like_count = row.get("like_count")
    reply_count = row.get("reply_count")
    view_count = row.get("view_count")
    follower_count = row.get("follower_count")
    volume_24h_baseline = row.get("volume_24h_baseline")
    category = row.get("category")
    if like_count is not None and reply_count is not None and view_count is not None:
        parts.append(
            f"- Engagement: {like_count} likes, {reply_count} replies, {view_count} views"
        )
    if follower_count is not None:
        parts.append(f"- Author followers: {follower_count}")
    if volume_24h_baseline is not None:
        parts.append(f"- Market volume (24h baseline): {float(volume_24h_baseline):.2f}")
    if category:
        parts.append(f"- Market category: {category}")

    parts.append("")
    parts.append("Definition of delta:")
    parts.append("- delta_h means ABSOLUTE YES-price change from current_price to horizon h.")
    parts.append("- future_price_h = current_price + delta_h.")
    parts.append(
        f"- future_price_h must be in [0,1], so each delta_h must be in [{price_low:.6f}, {price_high:.6f}]."
    )

    if few_shot:
        parts.append("")
        parts.append("Labeled examples:")
        for i, ex in enumerate(few_shot, 1):
            ep = float(ex.get("price_t0", 0.5))
            parts.append(
                f"Example {i}:\n"
                f"  condition_id: {ex.get('condition_id')}\n"
                f"  current_price: {ep:.4f}\n"
                f"  valid delta range: [{-ep:.4f}, {1-ep:.4f}]\n"
                f"  answer: {{\"delta_30m\": {ex.get('delta_30m', 0):.4f}, "
                f"\"delta_2h\": {ex.get('delta_2h', 0):.4f}, "
                f"\"delta_6h\": {ex.get('delta_6h', 0):.4f}}}"
            )

    parts.append("")
    parts.append("Output requirements:")
    parts.append("- delta_30m, delta_2h, delta_6h must be decimal numbers in [-1, 1].")
    parts.append("- Do not add extra keys.")
    parts.append("- Return strict JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai(api_key: str, model: str, prompt: str, timeout: float) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    resp = _post_json(OPENAI_API_URL, headers, body, timeout)
    return resp["choices"][0]["message"]["content"].strip()


def call_anthropic(api_key: str, model: str, prompt: str, timeout: float) -> str:
    body = {
        "model": model,
        "max_tokens": 200,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    resp = _post_json(ANTHROPIC_API_URL, headers, body, timeout)
    parts = [c["text"] for c in resp.get("content", []) if c.get("type") == "text"]
    return "\n".join(parts).strip()


def call_llm(provider: str, api_key: str, model: str, prompt: str, timeout: float) -> str:
    if provider == "anthropic":
        return call_anthropic(api_key, model, prompt, timeout)
    return call_openai(api_key, model, prompt, timeout)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_prediction(text: str, price_t0: float) -> dict[str, float]:
    candidate = text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])

    result = {}
    for key in ("delta_30m", "delta_2h", "delta_6h"):
        val = float(payload[key])
        result[key] = clamp_delta(val, price_t0)
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_tier(
    records: list[dict[str, Any]],
    pred_map: dict[str, dict],
    tier_name: str,
) -> dict[str, Any]:
    """Evaluate direction accuracy, magnitude macro-F1, and Spearman rho."""
    dir_true, dir_pred = [], []
    mag_true, mag_pred = [], []
    pred_deltas, actual_deltas = [], []

    for r in records:
        key = f"{r['tweet_id']}_{r['condition_id']}"
        p = pred_map.get(key)
        if p is None:
            continue

        # Direction / magnitude
        dir_true.append(r["direction_label"])
        dir_pred.append(derive_direction(p["delta_2h"]))
        mag_true.append(r["magnitude_bucket"])
        mag_pred.append(derive_magnitude(p["delta_2h"]))

        # Spearman on flattened deltas
        for horizon in ("delta_30m", "delta_2h", "delta_6h"):
            gold = r.get(horizon)
            pred_val = p.get(horizon)
            if gold is not None and pred_val is not None:
                actual_deltas.append(float(gold))
                pred_deltas.append(float(pred_val))

    n = len(dir_true)
    dir_acc = sum(a == b for a, b in zip(dir_true, dir_pred)) / n if n else 0.0
    mag_f1 = macro_f1(mag_true, mag_pred, MAGNITUDE_LABELS) if n else 0.0
    spr = spearman(pred_deltas, actual_deltas)

    return {
        "tier": tier_name,
        "n": n,
        "dir_acc": dir_acc,
        "mag_macro_f1": mag_f1,
        "spearman": spr,
    }


def print_tier_results(results: list[dict[str, Any]]) -> None:
    print(f"\n{'Tier':<45} {'N':>5} {'DirAcc':>8} {'MagF1':>8} {'Spearman':>10}")
    print("-" * 80)
    for r in results:
        spr = "N/A" if r["spearman"] is None else f"{r['spearman']:.4f}"
        print(
            f"{r['tier']:<45} {r['n']:>5} "
            f"{r['dir_acc']*100:>7.2f}% {r['mag_macro_f1']*100:>7.2f}% {spr:>10}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    model = args.model or (
        DEFAULT_ANTHROPIC_MODEL if args.provider == "anthropic" else DEFAULT_OPENAI_MODEL
    )

    api_key = ""
    if not args.dry_run:
        env_var = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_var, "")
        if not api_key:
            print(f"ERROR: Set {env_var} environment variable.", file=sys.stderr)
            sys.exit(1)

    print(f"Provider: {args.provider}  Model: {model}  Shots: {args.shots}")
    df = load_data(args.local_dir)
    print(f"Loaded {len(df)} rows")

    records = df.to_dict("records")

    # Default behavior: run on a reproducible random subset to control API cost.
    if args.sample_size > 0 and len(records) > args.sample_size:
        sampled_df = df.sample(n=args.sample_size, random_state=args.sample_seed).reset_index(drop=True)
        records = sampled_df.to_dict("records")

    if args.limit > 0:
        records = records[: args.limit]

    # Build few-shot pool from first N records with non-flat direction
    few_shot: list[dict] = []
    if args.shots > 0:
        candidates = [r for r in records if r.get("direction_label") != "flat"]
        few_shot = candidates[: args.shots]

    # Resume
    output_path = Path(args.output)
    completed: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            completed.add(row.get("key", ""))
        print(f"Resuming: {len(completed)} cached.")

    pred_map: dict[str, dict] = {}
    errors = 0

    for i, row in enumerate(records):
        key = f"{row['tweet_id']}_{row['condition_id']}"
        price_t0 = float(row.get("price_t0", 0.5))

        if key in completed:
            continue

        prompt = build_user_prompt(row, few_shot)
        result: dict[str, Any] = {
            "key": key,
            "tweet_id": row["tweet_id"],
            "condition_id": row["condition_id"],
            "gold_direction": row.get("direction_label"),
            "gold_magnitude": row.get("magnitude_bucket"),
        }

        if args.dry_run:
            result["prompt"] = prompt
            append_jsonl(output_path, result)
            continue

        try:
            raw = call_llm(args.provider, api_key, model, prompt, args.timeout)
            parsed = parse_prediction(raw, price_t0)
            result["prediction"] = parsed
            result["raw_output"] = raw
            pred_map[key] = parsed
        except Exception as exc:
            result["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            errors += 1

        append_jsonl(output_path, result)

        if (i + 1) % 20 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] errors={errors}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\nPredictions written to {output_path}")

    if args.dry_run:
        return

    # Also load any previously saved predictions for evaluation
    for row in read_jsonl(output_path):
        key = row.get("key", "")
        if "prediction" in row and key not in pred_map:
            pred_map[key] = row["prediction"]

    # Build tier subsets
    tier1 = records
    tier2 = [r for r in records if not r.get("confound_flag", False)]
    tier3 = [r for r in records if not r.get("confound_flag", False) and r.get("direction_label") != "flat"]

    results = [
        evaluate_tier(tier1, pred_map, "Tier 1: All Data"),
        evaluate_tier(tier2, pred_map, "Tier 2: Non-confounded"),
        evaluate_tier(tier3, pred_map, "Tier 3: Active (non-confounded + non-flat)"),
    ]
    print_tier_results(results)


if __name__ == "__main__":
    main()
