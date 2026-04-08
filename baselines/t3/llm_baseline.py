#!/usr/bin/env python3
"""T3 LLM Grading Baseline -- Evidence Grading.

Prompts an LLM to assign an evidence grade (0-5) for each tweet-market
pair, then evaluates against the human-annotated final_grade using
Spearman correlation and Quadratic Weighted Kappa (QWK).

Usage:
    python -m baselines.t3.llm_baseline --provider openai --model gpt-4o --shots 0
    python -m baselines.t3.llm_baseline --provider anthropic --model claude-sonnet-4-20250514 --shots 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

import eventxbench

# ---------------------------------------------------------------------------
# 0-shot system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert evidence-grading assistant for prediction markets.

Objective:
Given a tweet, a market question, and a resolution rule, assign a grade from 0 to 5 that reflects how strongly the tweet supports the resolution of the market.

Grading Scale:

5 (resolving): Tweet directly and unambiguously confirms that the resolution condition occurred. Evidence is authoritative, verifiable, and meets all resolution criteria.
4 (strong_direct): Tweet clearly addresses the resolution condition but fails in one aspect such as source authority, timing, or threshold. Still strong evidence.
3 (indirect_report): Tweet provides credible second-hand information or reporting that suggests the resolution condition but does not directly confirm it.
2 (speculation): Tweet is relevant to the condition but contains rumors, predictions, or opinions without authoritative confirmation.
1 (reaction): Tweet is commentary, opinion, or a reaction to events; it does not provide factual evidence about the condition.
0 (noise): Tweet is off-topic, unrelated, or uninformative regarding the resolution condition.

Instructions:

Carefully evaluate the tweet against the resolution rule.
Consider authority, timing, threshold, and directness of evidence.
Return only a single integer (0-5).
Do not include any text, explanation, formatting, or markdown.
"""

# ---------------------------------------------------------------------------
# 6-shot examples and system prompt
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "tweet": "Elena Rybakina and Aryna Sabalenka will face each other in the Cincinnati Quarterfinals. \n\nAryna leads the head to head 7-4.\n\nThey've split their last 4 meetings. \n\nThe last time they played, Elena had 4 match points, but Aryna pulled off the comeback. \n\nThe two biggest servers. \n\nThe two biggest ball strikers. \n\nThe most powerful matchup in tennis. \n\nWho wins? 🐠🐯",
        "question": "Australian Open Women's: Elena Rybakina vs Tereza Valentova",
        "predicate": "Elena Rybakina advances against Tereza Valentova in theAustralian Open WTA match scheduled for January 23.",
        "grade": 0
    },
    {
        "tweet": "@LizSimmie Yeah it's the 12mo thing.. we notice this bc on the terminal all ETFs DES1 page has exp ratio so they get blank until year is up. Ppl seem to care, I got a ton of replies asking how much BTCC cost and there's no precise answer rn",
        "question": "Roaring Kitty posts on Reddit today?",
        "predicate": "u/DeepFightingValue posts, comments, or replies on Reddit on December 5 between 2:00 PM and 11:59 PM ET.",
        "grade": 1
    },
    {
        "tweet": "@routinetrader @danielegroff @fvareladv @TomHougaard I like Hidden Divergences and they are a massive component of what I do - regular divergences are far more trouble than they are worth since they coax traders into fading strength and I have found a lot of success by inverting them",
        "question": "Roaring Kitty posts on Reddit today?",
        "predicate": "u/DeepFightingValue posts, comments, or replies on Reddit on December 5 between 2:00 PM and 11:59 PM ET.",
        "grade": 2
    },
    {
        "tweet": ".@FURIA take down @GamerLegion 2:0 to secure a spot in the #ESLProLeague play-offs! https://t.co/VJAKOuOiMQ",
        "question": "Games Total: O/U 2.5",
        "predicate": "The series between largadosypelados and RED Canids Academy plays 3 or more games.",
        "grade": 3
    },
    {
        "tweet": "Palantir Technologies surged following a record breaking Q3 2025, posting $1.18 billion in revenue, surpassing analyst estimates, and raising full-year guidance to $4.4 billion. $PLTR",
        "question": "Will Palantir reach $204 in November?",
        "predicate": "During November 2025 (ET), any 1-minute candle for Palantir Technologies Inc. (PLTR) has a final 'High' price equal to or above $204 as recorded on Yahoo Finance.",
        "grade": 4
    },
    {
        "tweet": "Indianapolis adds a field goal.\n\n17-7 Colts | 2Q 2:12 #INDvsBUF",
        "question": "Colts Team Total: O/U 16.5",
        "predicate": "The Colts score 17 or more points in the upcoming NFL game against the Seahawks on December 14.",
        "grade": 5
    }
]

_shot_prompt_text = ""
for ex in FEW_SHOT_EXAMPLES:
    _shot_prompt_text += (
        f"Tweet: \"{ex['tweet']}\"\n"
        f"Market question: \"{ex['question']}\"\n"
        f"Resolution rule: \"{ex['predicate']}\"\n"
        f"Grade: {ex['grade']}\n\n"
    )

SYSTEM_PROMPT_3SHOT = f"""You are an expert evidence-grading assistant for prediction markets.

Objective:
Given a tweet, a market question, and a resolution rule, assign a grade from 0 to 5 that reflects how strongly the tweet supports the resolution of the market.

Grading Scale:

5 (resolving): Tweet directly and unambiguously confirms that the resolution condition occurred. Evidence is authoritative, verifiable, and meets all resolution criteria.
4 (strong_direct): Tweet clearly addresses the resolution condition but fails in one aspect such as source authority, timing, or threshold. Still strong evidence.
3 (indirect_report): Tweet provides credible second-hand information or reporting that suggests the resolution condition but does not directly confirm it.
2 (speculation): Tweet is relevant to the condition but contains rumors, predictions, or opinions without authoritative confirmation.
1 (reaction): Tweet is commentary, opinion, or a reaction to events; it does not provide factual evidence about the condition.
0 (noise): Tweet is off-topic, unrelated, or uninformative regarding the resolution condition.

Instructions:

Carefully evaluate the tweet against the resolution rule.
Consider authority, timing, threshold, and directness of evidence.
Return only a single integer (0-5).
Do not include any text, explanation, formatting, or markdown.

Use the following examples as guidance (6-shot):

{_shot_prompt_text}
Now grade the next tweet.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_prompt_0shot(tweet: str, question: str, predicate: str) -> str:
    return (
        f"Market question: \"{question}\"\n"
        f"Resolution rule: \"{predicate}\"\n"
        f"Tweet: \"{tweet[:]}\""
    )


def _build_prompt_3shot(tweet: str, question: str, predicate: str) -> str:
    return (
        f"Market question: \"{question}\"\n"
        f"Resolution rule: \"{predicate}\"\n"
        f"Tweet: \"{tweet[:]}\""
    )


# ---------------------------------------------------------------------------
# LLM client helpers
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


def _call_llm(client, provider: str, model: str, system: str, prompt: str, max_tokens: int = 32) -> str:
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=60,
        )
        return resp.choices[0].message.content.strip()


def _parse_grade(raw: str) -> int | None:
    """Extract grade 0-5 from LLM response."""
    # Try JSON parse first
    try:
        obj = json.loads(raw)
        g = int(obj.get("grade", -1))
        if 0 <= g <= 5:
            return g
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fallback: find first digit 0-5
    match = re.search(r"\b([0-5])\b", raw)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return None

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * len(vals)
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

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def _quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], num_classes: int = 6) -> float:
    """Compute QWK for ordinal grades 0..num_classes-1."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # Confusion matrix
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t][p] += 1

    # Weight matrix (quadratic)
    W = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i][j] = (i - j) ** 2 / (num_classes - 1) ** 2

    # Expected matrix under independence
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / n

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 LLM grading baseline")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        default="openai",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 3])
    parser.add_argument("--output", default="t3_llm_results.jsonl")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    required = {"tweet_id", "final_grade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in T3 data: {missing}")

    # Determine text column names (may vary between data versions)
    tweet_col = "tweet" if "tweet" in df.columns else "tweet_text"
    question_col = "question" if "question" in df.columns else "market_question"
    predicate_col = "predicate" if "predicate" in df.columns else "resolution_rule"
    if tweet_col not in df.columns or question_col not in df.columns:
        raise ValueError(
            f"Expected '{tweet_col}' and '{question_col}' columns in T3 data. "
            f"Found: {sorted(df.columns)}"
        )

    print(f"T3 samples: {len(df)}, model: {args.model}, shots: {args.shots}")

    if not args.dry_run:
        client = _make_client(args.provider)

    results: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    parse_errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        tweet = str(row[tweet_col])
        question = str(row[question_col])
        predicate = str(row[predicate_col])
        gold = int(row["final_grade"])

        if args.shots == 0:
            system = SYSTEM_PROMPT
            prompt = _build_prompt_0shot(tweet, question, predicate)
        else:
            system = SYSTEM_PROMPT_3SHOT
            prompt = _build_prompt_3shot(tweet, question, predicate)

        if args.dry_run:
            print("=== SAMPLE PROMPT ===")
            print("SYSTEM:", system)
            print("USER:", prompt)
            print(f"\nGold grade: {gold}")
            return

        try:
            raw = _call_llm(client, args.provider, args.model, system, prompt)
        except Exception as e:
            print(f"  [API ERROR] row {i}: {e}")
            time.sleep(5)
            continue

        grade = _parse_grade(raw)
        if grade is None:
            parse_errors += 1
            grade = 2  # fallback to median grade

        y_true.append(gold)
        y_pred.append(grade)
        results.append(
            {
                "tweet_id": str(row["tweet_id"]),
                "condition_id": str(row.get("condition_id", "")),
                "gold_grade": gold,
                "predicted_grade": grade,
                "llm_raw": raw,
            }
        )

        n = len(results)
        if n % 50 == 0:
            rho = _spearman([float(v) for v in y_true], [float(v) for v in y_pred])
            rho_str = f"{rho:.4f}" if rho is not None else "N/A"
            print(f"  [{n}/{len(df)}] Spearman={rho_str}  parse_errors={parse_errors}")

        time.sleep(args.delay)

    # -- Report -------------------------------------------------------------
    if results:
        rho = _spearman([float(v) for v in y_true], [float(v) for v in y_pred])
        qwk = _quadratic_weighted_kappa(y_true, y_pred, num_classes=6)
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"

        print(f"\n=== Results ({args.model}, {args.shots}-shot) ===")
        print(f"  N={len(results)}, Spearman={rho_str}, QWK={qwk:.4f}, parse_errors={parse_errors}")

        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved predictions -> {args.output}")


if __name__ == "__main__":
    main()