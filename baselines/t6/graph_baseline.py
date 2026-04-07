#!/usr/bin/env python3
"""T6 graph heuristic baseline.

This is the paper graph baseline ported into EventXBench.  It uses the unified
T6 label/feature JSONL for labels, split, confound filtering, and lag targets;
the graph inputs remain external files because they are derived artifacts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .data_utils import LABEL_ORDER, load_t6_dataframe
except ImportError:
    from data_utils import LABEL_ORDER, load_t6_dataframe

DEFAULT_SIBLINGS = "data/t6/task6_sibling_moves_v2_tuned_t35confound_full.jsonl"
DEFAULT_EMBEDDING_RECORDS = "data/t6/market_embedding_records_full.jsonl"
DEFAULT_EMBEDDINGS = "data/t6/question_only_bge_embeddings_full.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T6 graph heuristic baseline")
    parser.add_argument("--repo", default="mlsys-io/EventXBench")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional local override for unified T6 JSONL. Defaults to Hugging Face.",
    )
    parser.add_argument("--sibling-file", default=DEFAULT_SIBLINGS)
    parser.add_argument("--embedding-records", default=DEFAULT_EMBEDDING_RECORDS)
    parser.add_argument("--embedding-file", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-file", default="t6_graph_heuristic_predictions.jsonl")
    parser.add_argument("--summary-json", default="t6_graph_heuristic_summary.json")
    parser.add_argument("--threshold", type=float, default=0.9510023593902588)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--eval-split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--grid-start", type=float, default=0.90)
    parser.add_argument("--grid-stop", type=float, default=0.999)
    parser.add_argument("--grid-step", type=float, default=0.001)
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--include-confounded", action="store_true")
    parser.add_argument("--include-insufficient", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def filter_label_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    include_confounded: bool,
    include_insufficient: bool,
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("label") not in LABEL_ORDER:
            continue
        if not include_confounded and bool(row.get("confound_flag")):
            continue
        if not include_insufficient and bool(row.get("insufficient_data_flag")):
            continue
        out.append(row)
    return out


def build_siblings_by_tweet(
    rows: Sequence[Dict[str, Any]],
    allowed_tweet_ids: set[int],
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tweet_id = int(row["tweet_id"])
        if tweet_id in allowed_tweet_ids:
            grouped[tweet_id].append(row)
    return grouped


def load_embedding_index(
    record_path: Path,
    embedding_path: Path,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]], np.ndarray]:
    records = read_jsonl(record_path)
    embeddings = np.load(embedding_path).astype(np.float32)
    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Embedding row count {embeddings.shape[0]} does not match record count {len(records)}"
        )
    index = {str(record["condition_id"]): int(record["row_index"]) for record in records}
    lookup = {str(record["condition_id"]): record for record in records}
    return index, lookup, embeddings


def max_primary_sibling_cosine(
    primary_id: str,
    sibling_rows: Sequence[Dict[str, Any]],
    embedding_index: Dict[str, int],
    embedding_matrix: np.ndarray,
) -> Tuple[Optional[float], Optional[str]]:
    primary_idx = embedding_index.get(primary_id)
    if primary_idx is None:
        return None, None
    best_cosine: Optional[float] = None
    best_sibling_id: Optional[str] = None
    primary_vec = embedding_matrix[primary_idx]
    seen: set[str] = set()
    for row in sibling_rows:
        sibling_id = str(row["sibling_condition_id"])
        if sibling_id in seen:
            continue
        seen.add(sibling_id)
        sibling_idx = embedding_index.get(sibling_id)
        if sibling_idx is None:
            continue
        cosine = float(primary_vec @ embedding_matrix[sibling_idx])
        if best_cosine is None or cosine > best_cosine:
            best_cosine = cosine
            best_sibling_id = sibling_id
    return best_cosine, best_sibling_id


def predict_label(max_cosine: Optional[float], threshold: float) -> str:
    if max_cosine is not None and max_cosine >= threshold:
        return "primary_mover"
    return "no_cross_market_effect"


def macro_f1(rows: Sequence[Dict[str, Any]]) -> float:
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get("predicted_label")
        if gold and pred:
            confusion[gold][pred] += 1
    scores: List[float] = []
    for label in LABEL_ORDER:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABEL_ORDER if other != label)
        fn = sum(confusion[label][other] for other in LABEL_ORDER if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    compared = [row for row in rows if row.get("gold_label") and row.get("predicted_label")]
    if not compared:
        return 0.0
    return sum(row["gold_label"] == row["predicted_label"] for row in compared) / len(compared)


def evaluate_metric(rows: Sequence[Dict[str, Any]], metric: str) -> float:
    return macro_f1(rows) if metric == "macro_f1" else accuracy(rows)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def median_train_lag_minutes(label_rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    train_lags = [
        float(row["first_sibling_move_lag_min_bucket"])
        for row in label_rows
        if row.get("split") == "train"
        and not row.get("insufficient_data_flag")
        and not is_missing(row.get("first_sibling_move_lag_min_bucket"))
    ]
    if not train_lags:
        return None
    return float(np.median(np.asarray(train_lags, dtype=np.float32)))


def lag_mae(rows: Sequence[Dict[str, Any]], predicted_lag: Optional[float]) -> tuple[Optional[float], int]:
    if predicted_lag is None:
        return None, 0
    gold = [
        float(row["gold_first_sibling_move_lag_min_bucket"])
        for row in rows
        if not is_missing(row.get("gold_first_sibling_move_lag_min_bucket"))
    ]
    if not gold:
        return None, 0
    return float(np.mean(np.abs(np.asarray(gold, dtype=np.float32) - predicted_lag))), len(gold)


def build_prediction_rows(
    label_rows: Sequence[Dict[str, Any]],
    siblings_by_tweet: Dict[int, List[Dict[str, Any]]],
    embedding_index: Dict[str, int],
    embedding_matrix: np.ndarray,
    record_lookup: Dict[str, Dict[str, Any]],
    threshold: float,
    predicted_lag: Optional[float],
) -> List[Dict[str, Any]]:
    outputs = []
    for row in label_rows:
        tweet_id = int(row["tweet_id"])
        primary_id = str(row["primary_condition_id"])
        sibling_rows = siblings_by_tweet.get(tweet_id, [])
        max_cosine, argmax_sibling_id = max_primary_sibling_cosine(
            primary_id,
            sibling_rows,
            embedding_index,
            embedding_matrix,
        )
        outputs.append(
            {
                "tweet_id": tweet_id,
                "primary_condition_id": primary_id,
                "gold_label": row["label"],
                "predicted_label": predict_label(max_cosine, threshold),
                "threshold": threshold,
                "split": row.get("split"),
                "confound_flag": bool(row.get("confound_flag")),
                "insufficient_data_flag": bool(row.get("insufficient_data_flag")),
                "sibling_count": len({str(s["sibling_condition_id"]) for s in sibling_rows}),
                "max_primary_sibling_cosine": max_cosine,
                "gold_first_sibling_move_lag_min_bucket": row.get("first_sibling_move_lag_min_bucket"),
                "predicted_onset_lag_min_bucket": predicted_lag,
                "argmax_sibling_condition_id": argmax_sibling_id,
                "argmax_sibling_question": (record_lookup.get(argmax_sibling_id) or {}).get("question"),
                "baseline_name": "t6_graph_heuristic_bge_maxcos",
            }
        )
    return outputs


def tune_threshold(
    rows: Sequence[Dict[str, Any]],
    *,
    start: float,
    stop: float,
    step: float,
    metric: str,
) -> tuple[float, float]:
    best_threshold = start
    best_score = -1.0
    for threshold in np.arange(start, stop + 1e-9, step, dtype=np.float32):
        scored_rows = [
            {
                **row,
                "predicted_label": predict_label(row.get("max_primary_sibling_cosine"), float(threshold)),
            }
            for row in rows
        ]
        score = evaluate_metric(scored_rows, metric)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def main() -> None:
    args = parse_args()

    if args.labels_file:
        all_label_rows = read_jsonl(Path(args.labels_file))
    else:
        all_label_rows = load_t6_dataframe(local_dir=args.local_dir, repo=args.repo).to_dict("records")
    label_rows = filter_label_rows(
        all_label_rows,
        include_confounded=args.include_confounded,
        include_insufficient=args.include_insufficient,
    )
    allowed_ids = {int(row["tweet_id"]) for row in label_rows}
    sibling_rows = build_siblings_by_tweet(read_jsonl(Path(args.sibling_file)), allowed_ids)
    embedding_index, record_lookup, embedding_matrix = load_embedding_index(
        Path(args.embedding_records),
        Path(args.embedding_file),
    )
    predicted_lag = median_train_lag_minutes(all_label_rows)

    base_rows = build_prediction_rows(
        label_rows,
        sibling_rows,
        embedding_index,
        embedding_matrix,
        record_lookup,
        args.threshold,
        predicted_lag,
    )

    tuned_threshold = args.threshold
    tuned_score = None
    tuning_rows_count = 0
    if args.tune_threshold:
        tune_rows = [row for row in base_rows if row.get("split") == "val"]
        if not tune_rows:
            raise SystemExit("No validation rows available for threshold tuning.")
        tuned_threshold, tuned_score = tune_threshold(
            tune_rows,
            start=args.grid_start,
            stop=args.grid_stop,
            step=args.grid_step,
            metric=args.metric,
        )
        tuning_rows_count = len(tune_rows)
        base_rows = [
            {
                **row,
                "threshold": tuned_threshold,
                "predicted_label": predict_label(row.get("max_primary_sibling_cosine"), tuned_threshold),
            }
            for row in base_rows
        ]

    if args.eval_split != "all":
        base_rows = [row for row in base_rows if row.get("split") == args.eval_split]

    onset_lag_mae, onset_lag_eval_rows = lag_mae(base_rows, predicted_lag)
    summary = {
        "baseline_name": "t6_graph_heuristic_bge_maxcos",
        "rows": len(base_rows),
        "pred_counts": dict(Counter(row["predicted_label"] for row in base_rows)),
        "gold_counts": dict(Counter(row["gold_label"] for row in base_rows)),
        "accuracy": round(accuracy(base_rows), 4),
        "macro_f1": round(macro_f1(base_rows), 4),
        "lag_prediction_strategy": "median_train_first_sibling_move_lag_min_bucket",
        "predicted_onset_lag_min_bucket": round(predicted_lag, 4) if predicted_lag is not None else None,
        "lag_eval_rows": onset_lag_eval_rows,
        "lag_mae": round(onset_lag_mae, 4) if onset_lag_mae is not None else None,
        "threshold": tuned_threshold,
        "threshold_source": "validation_tuned" if args.tune_threshold else "fixed",
        "tuning_metric": args.metric if args.tune_threshold else None,
        "tuning_metric_value": round(tuned_score, 4) if tuned_score is not None else None,
        "tuning_rows": tuning_rows_count,
        "filters": {
            "include_confounded": args.include_confounded,
            "include_insufficient": args.include_insufficient,
            "eval_split": args.eval_split,
        },
        "outputs": {
            "predictions": str(Path(args.output_file)),
            "summary": str(Path(args.summary_json)),
        },
    }
    write_jsonl(Path(args.output_file), base_rows)
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
