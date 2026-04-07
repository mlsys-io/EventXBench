#!/usr/bin/env python3
"""T1 LLM Baseline -- Pre-Market Interest Forecasting

Classifies prediction-market questions into interest levels
(high_interest / moderate_interest / low_interest) using either hosted LLMs
or a local Qwen model through vLLM.

The script is self-contained and follows the repo's baseline conventions:
- loads Task 1 data via ``eventxbench.load_task("t1")``
- writes JSONL rows keyed by ``condition_id``
- prints accuracy and macro-F1 at the end of the run

Usage examples:
    python baselines/t1/llm_baseline.py --provider openai --model gpt-4o --shots 0
    python baselines/t1/llm_baseline.py --provider anthropic --model claude-3-5-sonnet-20241022 --shots 3
    python baselines/t1/llm_baseline.py --provider xai --model grok-4-1-fast-non-reasoning
    python baselines/t1/llm_baseline.py --provider qwen-local --model Qwen/Qwen3.5-4B
    python baselines/t1/llm_baseline.py --provider openai --dry-run --limit 5
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
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

LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]
VALID_LABELS = set(LABEL_ORDER)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
XAI_API_URL = "https://api.x.ai/v1/responses"

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_XAI_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_OUTPUT = "t1_llm_predictions.jsonl"

SYSTEM_PROMPT = """\
You are evaluating a benchmark task: Pre-Market Interest Forecasting.

You will be given:
- A target prediction market question.
- Pre-market social signals extracted from tweets before market creation.
- Optionally, a few labeled examples from the training set.

Your task is to predict the market interest label:
- high_interest: very strong later market interest / trading volume
- moderate_interest: meaningful but not top-tier later interest
- low_interest: relatively weak later market interest

Rules:
- Use only the information explicitly provided in the prompt.
- Do not use external knowledge or future information.
- Focus on whether the pre-market signal suggests later market attention.

Return strict JSON only (no explanation) in exactly this format:
{
  "label": "high_interest | moderate_interest | low_interest",
  "confidence": 0.0,
  "scores": {
    "high_interest": 0.0,
    "moderate_interest": 0.0,
    "low_interest": 0.0
  }
}
"""

FEATURE_COLUMNS = [
    "score",
    "cluster_count",
    "linked_tweet_count",
    "avg_link_confidence",
    "max_link_confidence",
    "text_similarity",
    "tweet_count",
    "unique_user_count",
    "burst_duration_hours",
    "max_author_followers",
    "mean_author_followers",
    "median_author_followers",
    "high_follower_author_count",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T1 LLM baseline: classify market interest level"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai", "qwen", "qwen-local"],
        required=True,
        help="LLM provider",
    )
    parser.add_argument("--model", default="", help="Model name (defaults per provider)")
    parser.add_argument(
        "--shots",
        "--shots-per-class",
        dest="shots_per_class",
        type=int,
        default=0,
        help="Number of few-shot examples per class (0 = zero-shot)",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Path to local EventX data directory (skips HF download)",
    )
    parser.add_argument(
        "--output",
        "--output-file",
        dest="output",
        default=DEFAULT_OUTPUT,
        help="JSONL file to append predictions to",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max test samples to evaluate")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset into test split")
    parser.add_argument("--resume", action="store_true", help="Skip already-predicted IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument(
        "--sleep",
        "--sleep-seconds",
        dest="sleep_seconds",
        type=float,
        default=0.0,
        help="Seconds between processed requests",
    )
    parser.add_argument(
        "--timeout",
        "--timeout-seconds",
        dest="timeout_seconds",
        type=float,
        default=120.0,
        help="API request timeout",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for hosted APIs")
    parser.add_argument("--api-key", default="", help="Optional explicit API key override")
    parser.add_argument(
        "--include-question",
        dest="include_question",
        action="store_true",
        help="Include the market question in the prompt",
    )
    parser.add_argument(
        "--no-include-question",
        dest="include_question",
        action="store_false",
        help="Omit the market question from the prompt",
    )
    parser.add_argument(
        "--include-structured-features",
        dest="include_structured_features",
        action="store_true",
        help="Include numeric social-signal features in the prompt",
    )
    parser.add_argument(
        "--no-include-structured-features",
        dest="include_structured_features",
        action="store_false",
        help="Omit numeric social-signal features from the prompt",
    )
    parser.add_argument(
        "--max-event-text-chars",
        type=int,
        default=1200,
        help="Maximum characters of event_text to include",
    )
    parser.add_argument("--chunk-size", type=int, default=128, help="Batch size for local Qwen")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.set_defaults(
        include_question=True,
        include_structured_features=True,
        trust_remote_code=True,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(local_dir: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test splits via eventxbench loader."""
    import eventxbench

    if local_dir:
        train_df, test_df = eventxbench.load_task("t1", local_dir=local_dir)
    else:
        train_df, test_df = eventxbench.load_task("t1")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return re.sub(r"\s+", " ", str(value).strip())


def _trim(text: Any, max_chars: int) -> str:
    s = _fmt(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _instance_block(
    row: dict[str, Any],
    feature_cols: list[str],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    if args.include_question:
        lines.append(f"- question: {_fmt(row.get('question'))}")
    lines.append(f"- event_group_label: {_fmt(row.get('event_group_label'))}")
    lines.append(f"- event_text: {_trim(row.get('event_text'), args.max_event_text_chars)}")
    if args.include_structured_features and feature_cols:
        lines.append("- structured_features:")
        for col in feature_cols:
            if col in row:
                lines.append(f"    {col}: {_fmt(row.get(col))}")
    return "\n".join(lines)


def select_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    available_cols = set(train_df.columns).union(set(test_df.columns))
    return [col for col in FEATURE_COLUMNS if col in available_cols]


def select_few_shot_examples(
    train_df: pd.DataFrame,
    shots_per_class: int,
) -> list[dict[str, Any]]:
    if shots_per_class <= 0:
        return []
    examples: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        subset = train_df[train_df["interest_label"].astype(str) == label]
        examples.extend(subset.head(shots_per_class).to_dict("records"))
    return examples


def build_user_prompt(
    row: dict[str, Any],
    feature_cols: list[str],
    few_shot: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    parts = ["Task 1: Pre-Market Interest Forecasting\n"]
    if few_shot:
        parts.append("Labeled examples:")
        for index, example in enumerate(few_shot, 1):
            block = _instance_block(example, feature_cols, args)
            parts.append(f"Example {index}:\n{block}\n- label: {example['interest_label']}")
        parts.append("")
    parts.append("Target market to classify:")
    parts.append(_instance_block(row, feature_cols, args))
    parts.append("")
    parts.append("Return strict JSON only.")
    return "\n".join(parts)


def build_chat_prompt(user_prompt: str) -> str:
    return "\n".join(
        [
            "<|im_start|>system",
            SYSTEM_PROMPT,
            "<|im_end|>",
            "<|im_start|>user",
            user_prompt,
            "<|im_end|>",
            "<|im_start|>assistant",
            "<think>\n</think>",
        ]
    )


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------


def is_local_qwen_provider(provider: str) -> bool:
    return provider in {"qwen", "qwen-local"}


def default_model_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "xai":
        return DEFAULT_XAI_MODEL
    if is_local_qwen_provider(provider):
        return DEFAULT_QWEN_MODEL
    return DEFAULT_OPENAI_MODEL


def api_key_env_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "xai":
        return "XAI_API_KEY"
    return "OPENAI_API_KEY"


def api_key_for_provider(provider: str, explicit: str) -> str:
    if explicit:
        return explicit
    if is_local_qwen_provider(provider):
        return ""
    return os.getenv(api_key_env_for_provider(provider), "")


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai(
    api_key: str,
    model: str,
    user_prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response_json = _post_json(OPENAI_API_URL, headers, body, timeout_seconds)
    output_text = response_json["choices"][0]["message"]["content"].strip()
    return response_json, output_text


def call_anthropic(
    api_key: str,
    model: str,
    user_prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    body = {
        "model": model,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    response_json = _post_json(ANTHROPIC_API_URL, headers, body, timeout_seconds)
    parts = [
        chunk["text"]
        for chunk in response_json.get("content", [])
        if chunk.get("type") == "text"
    ]
    return response_json, "\n".join(parts).strip()


def extract_xai_output_text(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            item_type = content_item.get("type")
            if item_type in {"output_text", "text"}:
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    if parts:
        return "\n".join(parts).strip()

    try:
        return response_json["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to extract xAI response text: {exc}") from exc


def call_xai(
    api_key: str,
    model: str,
    user_prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response_json = _post_json(XAI_API_URL, headers, body, timeout_seconds)
    return response_json, extract_xai_output_text(response_json)


def call_provider(
    provider: str,
    api_key: str,
    model: str,
    user_prompt: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    if provider == "anthropic":
        return call_anthropic(api_key, model, user_prompt, timeout_seconds)
    if provider == "xai":
        return call_xai(api_key, model, user_prompt, timeout_seconds)
    return call_openai(api_key, model, user_prompt, timeout_seconds)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _clean_prediction_text(text: str) -> str:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    return candidate.strip()


def _extract_json_candidate(text: str) -> str:
    candidate = _clean_prediction_text(text)
    start = candidate.find("{")
    if start >= 0:
        candidate = candidate[start:]
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if open_braces > close_braces:
        candidate = candidate + ("}" * (open_braces - close_braces))
    end = candidate.rfind("}")
    if end >= 0:
        candidate = candidate[: end + 1]
    return candidate


def _regex_fallback_prediction(text: str) -> dict[str, Any]:
    candidate = _clean_prediction_text(text)
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', candidate)
    if not label_match:
        raise ValueError("unable to recover label from model output")
    label = label_match.group(1).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"invalid label: {label!r}")
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', candidate)
    confidence = float(confidence_match.group(1)) if confidence_match else None
    scores: dict[str, float] = {}
    for target_label in LABEL_ORDER:
        score_match = re.search(
            rf'"{re.escape(target_label)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            candidate,
        )
        scores[target_label] = float(score_match.group(1)) if score_match else 0.0
    return {"label": label, "confidence": confidence, "scores": scores}


def parse_prediction(text: str) -> dict[str, Any]:
    candidate = _clean_prediction_text(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            payload = json.loads(_extract_json_candidate(candidate))
        except json.JSONDecodeError:
            payload = _regex_fallback_prediction(candidate)

    label = payload.get("label")
    if label not in VALID_LABELS:
        raise ValueError(f"invalid label: {label!r}")

    scores = payload.get("scores") or {}
    parsed_scores = {
        target_label: max(0.0, float(scores.get(target_label, 0.0)))
        for target_label in LABEL_ORDER
    }
    score_sum = sum(parsed_scores.values())
    if score_sum > 0:
        parsed_scores = {
            key: value / score_sum for key, value in parsed_scores.items()
        }
    else:
        parsed_scores = {
            key: (1.0 if key == label else 0.0) for key in LABEL_ORDER
        }

    confidence = payload.get("confidence")
    if confidence is None:
        confidence = parsed_scores[label]

    return {
        "label": label,
        "confidence": float(confidence),
        "scores": parsed_scores,
    }


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(gold: list[str], pred: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(
            gold,
            pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
    }


def build_result_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    feature_cols: list[str],
    few_shot_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    user_prompt = build_user_prompt(row, feature_cols, few_shot_examples, args)
    result_row: dict[str, Any] = {
        "condition_id": str(row["condition_id"]),
        "provider": args.provider,
        "model": args.model,
        "gold_label": row.get("interest_label"),
    }

    if args.dry_run:
        result_row["user_prompt"] = user_prompt
        return result_row

    if is_local_qwen_provider(args.provider):
        result_row["prompt"] = build_chat_prompt(user_prompt)
        result_row["user_prompt"] = user_prompt
        return result_row

    try:
        response_json, output_text = call_provider(
            args.provider,
            api_key,
            args.model,
            user_prompt,
            args.timeout_seconds,
        )
        parsed = parse_prediction(output_text)
        result_row["prediction"] = parsed
        result_row["raw_output"] = output_text
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result_row["error"] = {"type": "http_error", "status": exc.code, "body": body}
    except Exception as exc:  # noqa: BLE001
        result_row["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
    return result_row


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def run_qwen_generation(
    llm: Any,
    sampling_params: Any,
    batch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompts = [row["prompt"] for row in batch_rows]
    outputs = llm.generate(prompts, sampling_params)
    result_rows: list[dict[str, Any]] = []
    for base_row, output in zip(batch_rows, outputs):
        result_row = {key: value for key, value in base_row.items() if key != "prompt"}
        try:
            output_text = output.outputs[0].text.strip() if output.outputs else ""
            if not output_text:
                raise ValueError("empty model output")
            parsed = parse_prediction(output_text)
            result_row["prediction"] = parsed
            result_row["raw_output"] = output_text
        except Exception as exc:  # noqa: BLE001
            result_row["raw_output"] = output.outputs[0].text if output.outputs else ""
            result_row["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        result_rows.append(result_row)
    return result_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.model = args.model or default_model_for_provider(args.provider)
    api_key = api_key_for_provider(args.provider, args.api_key)

    if not args.dry_run and not is_local_qwen_provider(args.provider) and not api_key:
        env_name = api_key_env_for_provider(args.provider)
        print(f"ERROR: Set {env_name} environment variable or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Provider: {args.provider}  Model: {args.model}  "
        f"Shots/class: {args.shots_per_class}"
    )
    train_df, test_df = load_data(args.local_dir)

    if args.start_index > 0:
        test_df = test_df.iloc[args.start_index:].reset_index(drop=True)
    if args.limit > 0:
        test_df = test_df.head(args.limit).reset_index(drop=True)

    feature_cols = select_feature_columns(train_df, test_df)
    few_shot_examples = select_few_shot_examples(train_df, args.shots_per_class)

    output_path = Path(args.output)
    completed_ids: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            completed_ids.add(str(row.get("condition_id", "")))
        print(f"Resuming: {len(completed_ids)} predictions already cached.")

    records = [
        row
        for row in test_df.to_dict("records")
        if str(row["condition_id"]) not in completed_ids
    ]

    processed = 0
    errors = 0
    gold_labels: list[str] = []
    pred_labels: list[str] = []

    if is_local_qwen_provider(args.provider):
        prompt_rows = [
            build_result_row(row, args, api_key, feature_cols, few_shot_examples)
            for row in records
        ]
        if args.dry_run:
            for index, result_row in enumerate(prompt_rows, 1):
                append_jsonl(output_path, result_row)
                processed += 1
                print(f"[{index}/{len(prompt_rows)}] {result_row['condition_id']} (dry-run)")
        elif prompt_rows:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise SystemExit("Install vllm to use --provider qwen or qwen-local") from exc

            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                quantization=args.quantization or None,
                trust_remote_code=args.trust_remote_code,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                enforce_eager=args.enforce_eager,
                enable_prefix_caching=args.enable_prefix_caching,
                hf_token=args.hf_token or None,
            )
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop=["<|im_end|>"],
            )
            for batch_rows in chunked(prompt_rows, max(1, args.chunk_size)):
                for result_row in run_qwen_generation(llm, sampling_params, batch_rows):
                    append_jsonl(output_path, result_row)
                    processed += 1
                    if "error" in result_row:
                        errors += 1
                    else:
                        prediction = result_row.get("prediction") or {}
                        label = prediction.get("label")
                        gold_label = str(result_row.get("gold_label", ""))
                        if label in VALID_LABELS and gold_label in VALID_LABELS:
                            gold_labels.append(gold_label)
                            pred_labels.append(label)
                    print(f"  [{processed}/{len(prompt_rows)}] errors={errors}", flush=True)
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
    elif args.workers <= 1:
        total = len(records)
        for index, row in enumerate(records, 1):
            result_row = build_result_row(
                row,
                args,
                api_key,
                feature_cols,
                few_shot_examples,
            )
            append_jsonl(output_path, result_row)
            processed += 1
            if "error" in result_row:
                errors += 1
            else:
                prediction = result_row.get("prediction") or {}
                label = prediction.get("label")
                gold_label = str(result_row.get("gold_label", ""))
                if label in VALID_LABELS and gold_label in VALID_LABELS:
                    gold_labels.append(gold_label)
                    pred_labels.append(label)
            if args.dry_run:
                print(f"[{index}/{total}] {result_row['condition_id']} (dry-run)")
            elif index % 20 == 0 or index == total:
                print(f"  [{index}/{total}] errors={errors}")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    build_result_row,
                    row,
                    args,
                    api_key,
                    feature_cols,
                    few_shot_examples,
                )
                for row in records
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result_row = future.result()
                append_jsonl(output_path, result_row)
                processed += 1
                if "error" in result_row:
                    errors += 1
                else:
                    prediction = result_row.get("prediction") or {}
                    label = prediction.get("label")
                    gold_label = str(result_row.get("gold_label", ""))
                    if label in VALID_LABELS and gold_label in VALID_LABELS:
                        gold_labels.append(gold_label)
                        pred_labels.append(label)
                if args.dry_run:
                    print(f"[{index}/{len(futures)}] {result_row['condition_id']} (dry-run)")
                elif index % 20 == 0 or index == len(futures):
                    print(f"  [{index}/{len(futures)}] errors={errors}")
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    print(f"\nPredictions written to {output_path}")
    print(f"Total processed this run: {processed}  Errors: {errors}")

    if gold_labels and not args.dry_run:
        metrics = evaluate(gold_labels, pred_labels)
        print("\n--- Results ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
