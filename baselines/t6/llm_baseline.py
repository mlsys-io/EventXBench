#!/usr/bin/env python3
"""T6 LLM Baseline -- Cross-Market Propagation.

Prompts an LLM with market propagation features (sibling count, moved
siblings, primary price delta, etc.) to predict cross-market effect.
Evaluates Macro-F1.

Usage:
    python -m baselines.t6.llm_baseline --provider openai --model gpt-4o --shots 0
    python -m baselines.t6.llm_baseline --provider anthropic --model claude-sonnet-4-20250514 --shots 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

try:
    from .data_utils import LABEL_ORDER, clean_t6_dataframe, load_t6_dataframe, select_eval_split
except ImportError:
    from data_utils import LABEL_ORDER, clean_t6_dataframe, load_t6_dataframe, select_eval_split

LABELS = LABEL_ORDER

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
TASK_DESCRIPTION = """\
Cross-market propagation classification:
- no_effect: The tweet affected only the primary market; sibling markets were unaffected.
- primary_mover: The tweet's impact on the primary market propagated to sibling markets.
- propagated_signal: The market movement was a propagated signal from another market."""

FEW_SHOT_EXAMPLES = [
    {
        "sibling_count": 5,
        "moved_sibling_count": 4,
        "primary_delta_h": 0.08,
        "label": "primary_mover",
    },
    {
        "sibling_count": 3,
        "moved_sibling_count": 0,
        "primary_delta_h": 0.05,
        "label": "no_cross_market_effect",
    },
    {
        "sibling_count": 1,
        "moved_sibling_count": 0,
        "primary_delta_h": 0.02,
        "label": "propagated_signal",
    },
]


def _format_features(row) -> str:
    fields = [
        ("sibling_count", "Number of sibling markets"),
        ("moved_sibling_count", "Number of siblings that moved"),
        ("primary_delta_h", "Primary market price change"),
        ("confound_flag", "Confounding events detected"),
    ]
    lines = []
    for col, desc in fields:
        val = row.get(col, "N/A")
        if val is not None and str(val) != "nan":
            lines.append(f"  {desc}: {val}")
    return "\n".join(lines)


def _build_prompt_0shot(row) -> str:
    return (
        "You are classifying whether a tweet's market impact propagated to sibling markets.\n\n"
        f"{TASK_DESCRIPTION}\n\n"
        f"Features:\n{_format_features(row)}\n\n"
        "Based on these features, classify the cross-market propagation pattern.\n"
        "Reply with ONLY a JSON object: "
        "{\"label\": \"no_cross_market_effect\" | \"primary_mover\" | \"propagated_signal\"}"
    )


def _build_prompt_3shot(row) -> str:
    lines = [
        "You are classifying whether a tweet's market impact propagated to sibling markets.",
        "",
        TASK_DESCRIPTION,
        "",
        "=== Examples ===",
    ]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"\nFeatures:")
        lines.append(f"  Number of sibling markets: {ex['sibling_count']}")
        lines.append(f"  Number of siblings that moved: {ex['moved_sibling_count']}")
        lines.append(f"  Primary market price change: {ex['primary_delta_h']}")
        lines.append(f"Label: {{\"label\": \"{ex['label']}\"}}")
    lines += [
        "",
        "=== Now classify ===",
        "",
        f"Features:\n{_format_features(row)}",
        "",
        "Reply with ONLY a JSON object: "
        "{\"label\": \"no_cross_market_effect\" | \"primary_mover\" | \"propagated_signal\"}",
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


def _parse_label(raw: str) -> str | None:
    try:
        obj = json.loads(raw)
        lab = obj.get("label", "").lower().strip()
        if lab in LABELS:
            return lab
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    for label in LABELS:
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T6 LLM cross-market propagation baseline")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        default="openai",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 3])
    parser.add_argument("--output", default="t6_llm_results.jsonl")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", default="mlsys-io/EventXBench")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument(
        "--feature-file",
        default=None,
        help="Path to unified T6 feature JSONL with split column.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "all"],
        default="all",
        help="Default is 'all' to reproduce the existing all-clean LLM leaderboard rows.",
    )
    parser.add_argument("--include-confounded", action="store_true")
    parser.add_argument("--include-insufficient", action="store_true")
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    df = load_t6_dataframe(args.feature_file, args.local_dir, repo=args.repo)

    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in T6 data.")

    df = clean_t6_dataframe(
        df,
        include_insufficient=args.include_insufficient,
        include_confounded=args.include_confounded,
    )
    df = select_eval_split(df, args.eval_split)
    eval_labels = LABELS

    print(f"T6 samples: {len(df)}, model: {args.model}, shots: {args.shots}")

    if not args.dry_run:
        client = _make_client(args.provider)

    results: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    parse_errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        gold = str(row["label"])

        prompt = (
            _build_prompt_0shot(row)
            if args.shots == 0
            else _build_prompt_3shot(row)
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

        pred = _parse_label(raw)
        if pred is None:
            parse_errors += 1
            pred = "no_cross_market_effect"  # fallback

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
            mf1 = _macro_f1(y_true, y_pred, eval_labels)
            print(f"  [{n}/{len(df)}] Macro-F1={mf1:.4f}  parse_errors={parse_errors}")

        time.sleep(args.delay)

    # -- Report -------------------------------------------------------------
    if results:
        mf1 = _macro_f1(y_true, y_pred, eval_labels)
        print(f"\n=== Results ({args.model}, {args.shots}-shot) ===")
        print(f"  N={len(results)}, Macro-F1={mf1:.4f}, parse_errors={parse_errors}")

        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved predictions -> {args.output}")


if __name__ == "__main__":
    main()
