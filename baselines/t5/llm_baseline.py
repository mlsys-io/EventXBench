#!/usr/bin/env python3
"""T5 LLM Baseline -- Impact Persistence (Decay Classification).

Prompts an LLM with price impact data at multiple horizons to predict the
decay class (transient / sustained / reversal).  Evaluates Macro-F1.

Note: In the original codebase this task is referred to as T7 / task5+7,
but in the paper and public release it is T5.

Usage:
    python -m baselines.t5.llm_baseline --provider openai --model gpt-4o --shots 0
    python -m baselines.t5.llm_baseline --provider anthropic --model claude-sonnet-4-20250514 --shots 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import pandas as pd

import eventxbench

DECAY_LABELS = ["transient", "sustained", "reversal"]
HORIZONS = ["15m", "30m", "1h", "2h", "6h"]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "price_impacts": {"15m": 0.05, "30m": 0.04, "1h": 0.02, "2h": 0.005, "6h": 0.001},
        "volume_multipliers": {"15m": 3.2, "30m": 2.1, "1h": 1.3, "2h": 1.0, "6h": 0.9},
        "decay_class": "transient",
    },
    {
        "price_impacts": {"15m": 0.03, "30m": 0.04, "1h": 0.05, "2h": 0.06, "6h": 0.07},
        "volume_multipliers": {"15m": 2.0, "30m": 2.5, "1h": 2.8, "2h": 3.0, "6h": 2.5},
        "decay_class": "sustained",
    },
    {
        "price_impacts": {"15m": 0.06, "30m": 0.03, "1h": -0.01, "2h": -0.04, "6h": -0.05},
        "volume_multipliers": {"15m": 4.0, "30m": 2.5, "1h": 1.5, "2h": 2.0, "6h": 1.8},
        "decay_class": "reversal",
    },
]

TASK_DESCRIPTION = """\
Task7 targets and definitions:
1) price_impact_h: max absolute YES-price move from post time to horizon h (non-negative).
2) volume_multiplier_h: horizon volume divided by baseline volume (non-negative).
3) decay_class in {transient, sustained, reversal}.

Decay class definitions (important):
- transient: impact spikes early but clearly fades by later horizons; information effect does not persist.
- sustained: impact remains meaningfully elevated through later horizons (2h/6h), indicating persistent repricing.
- reversal: initial impact is later unwound/opposed, i.e., net effect weakens sharply and contradicts early move semantics.

Privacy/data rule:
- Do not rely on tweet text or market text. Use only the provided numeric horizon signals.
"""


def _format_impacts(price_impacts: dict, volume_multipliers: dict) -> str:
    lines = ["Price impacts by horizon:"]
    for h in HORIZONS:
        pi = price_impacts.get(h, "N/A")
        vm = volume_multipliers.get(h, "N/A")
        lines.append(f"  {h}: price_impact={pi}, volume_multiplier={vm}")
    return "\n".join(lines)


def _build_prompt_0shot(price_impacts: dict, volume_multipliers: dict) -> str:
    return (
        "You are a careful forecasting assistant for Task7 decay classification.\n\n"
        f"{TASK_DESCRIPTION}\n\n"
        f"{_format_impacts(price_impacts, volume_multipliers)}\n\n"
        "Classify decay_class from the numeric trajectory only.\n"
        "Output requirements:\n"
        "- Return strict JSON only.\n"
        "- Exactly one key: decay_class.\n"
        "- decay_class must be one of transient/sustained/reversal.\n"
        "Reply with ONLY: {\"decay_class\": \"transient\" | \"sustained\" | \"reversal\"}"
    )


def _build_prompt_3shot(price_impacts: dict, volume_multipliers: dict) -> str:
    lines = [
        "You are a careful forecasting assistant for Task7 decay classification.",
        "",
        TASK_DESCRIPTION,
        "",
        "=== Examples ===",
    ]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"\n{_format_impacts(ex['price_impacts'], ex['volume_multipliers'])}")
        lines.append(f"Decay class: {{\"decay_class\": \"{ex['decay_class']}\"}}")
    lines += [
        "",
        "=== Now classify ===",
        "",
        _format_impacts(price_impacts, volume_multipliers),
        "",
        "Use numeric signals only (no text assumptions).",
        "Reply with ONLY: {\"decay_class\": \"transient\" | \"sustained\" | \"reversal\"}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _make_client(provider: str):
    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key)
    elif provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "xai":
        from openai import OpenAI
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the XAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_llm(client, provider: str, model: str, prompt: str, max_tokens: int = 64) -> str:
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            timeout=60,
        )
        return resp.choices[0].message.content.strip()


def _parse_decay_class(raw: str) -> str | None:
    try:
        obj = json.loads(raw)
        dc = obj.get("decay_class", "").lower().strip()
        if dc in DECAY_LABELS:
            return dc
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    for label in DECAY_LABELS:
        if label in raw.lower():
            return label
    return None


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
# Data helpers
# ---------------------------------------------------------------------------
def _extract_impact_dict(row, prefix: str) -> dict:
    """Extract horizon dict from either a JSON column or flat columns."""
    json_col = f"{prefix}_json"
    if json_col in row.index:
        val = row[json_col]
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    # Try flat columns
    result = {}
    for h in HORIZONS:
        col = f"{prefix}_{h}"
        if col in row.index and pd.notna(row[col]):
            result[h] = float(row[col])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T5 LLM decay classification baseline")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        default="openai",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 3])
    parser.add_argument("--output", default="t5_llm_results.jsonl")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    data = eventxbench.load_task("t5", local_dir=args.local_dir)
    if isinstance(data, tuple):
        _, df = data  # use test split
    else:
        df = data

    if "decay_class" not in df.columns:
        raise ValueError("Missing 'decay_class' column in T5 data.")

    # Filter to valid labels
    df = df[df["decay_class"].isin(DECAY_LABELS)].reset_index(drop=True)
    print(f"T5 samples: {len(df)}, model: {args.model}, shots: {args.shots}")

    if not args.dry_run:
        client = _make_client(args.provider)

    results: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    parse_errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        price_impacts = _extract_impact_dict(row, "price_impact")
        volume_multipliers = _extract_impact_dict(row, "volume_multiplier")
        gold = str(row["decay_class"])

        prompt = (
            _build_prompt_0shot(price_impacts, volume_multipliers)
            if args.shots == 0
            else _build_prompt_3shot(price_impacts, volume_multipliers)
        )

        if args.dry_run:
            print("=== SAMPLE PROMPT ===")
            print(prompt)
            print(f"\nGold: {gold}")
            return

        try:
            raw = _call_llm(client, args.provider, args.model, prompt)
        except Exception as e:
            print(f"  [API ERROR] row {i}: {e}")
            time.sleep(5)
            continue

        pred = _parse_decay_class(raw)
        if pred is None:
            parse_errors += 1
            pred = "transient"  # fallback to majority

        y_true.append(gold)
        y_pred.append(pred)
        results.append(
            {
                "tweet_id": str(row.get("tweet_id", i)),
                "gold": gold,
                "predicted": pred,
                "llm_raw": raw,
            }
        )

        n = len(results)
        if n % 50 == 0:
            mf1 = _macro_f1(y_true, y_pred, DECAY_LABELS)
            print(f"  [{n}/{len(df)}] Macro-F1={mf1:.4f}  parse_errors={parse_errors}")

        time.sleep(args.delay)

    # -- Report -------------------------------------------------------------
    if results:
        mf1 = _macro_f1(y_true, y_pred, DECAY_LABELS)
        print(f"\n=== Results ({args.model}, {args.shots}-shot) ===")
        print(f"  N={len(results)}, Macro-F1={mf1:.4f}, parse_errors={parse_errors}")

        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved predictions -> {args.output}")


if __name__ == "__main__":
    main()
