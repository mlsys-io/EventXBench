
from __future__ import annotations

import argparse
import json
import re
import warnings
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]

DEFAULT_OUTPUT = "t1_lightgbm_predictions.jsonl"

NUMERIC_CANDIDATES = [
    "score",
    "cluster_count",
    "linked_tweet_count",
    "avg_link_confidence",
    "max_link_confidence",
    "text_similarity",
    "tweet_count",
    "unique_user_count",
    "burst_duration_hours",
    "lag_days",
    "temporal_fit",
    "time_to_market_days",
    "max_author_tweet_count",
    "mean_author_tweet_count",
    "median_author_tweet_count",
    "dominant_author_share",
    "repeat_author_count",
    "max_author_followers",
    "mean_author_followers",
    "median_author_followers",
    "high_follower_author_count",
]

CATEGORICAL_CANDIDATES = [
    "event_group_label",
    "has_tweet_link",
    "time_to_market_bucket",
    "pre_market_topic",
]

TEXT_CANDIDATES = [
    "question",
    "event_group_label",
    "event_text",
    "normalized_event_text",
]

FOLLOWER_CANDIDATES = [
    "max_author_followers",
    "mean_author_followers",
    "median_author_followers",
    "high_follower_author_count",
]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T1 LightGBM baseline with text, categorical, and threshold tuning"
    )
    parser.add_argument("--local-dir", default=None, help="Local data directory (skips HF)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=3)
    parser.add_argument("--question-svd-components", type=int, default=80)
    parser.add_argument("--event-svd-components", type=int, default=120)
    parser.add_argument("--use-threshold-tuning", action="store_true")
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="*",
        default=[0.7, 0.85, 1.0, 1.15, 1.3],
    )
    return parser.parse_args()


def load_data(local_dir: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import eventxbench

    if local_dir:
        return eventxbench.load_task("t1", local_dir=local_dir)
    return eventxbench.load_task("t1")


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def combine_text_fields(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        if column in row:
            text = normalize_text(row.get(column))
            if text:
                parts.append(text)
    return " ".join(parts)


def select_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    available = set(train_df.columns) & set(test_df.columns)
    numeric_features = [col for col in NUMERIC_CANDIDATES if col in available]
    categorical_features = [col for col in CATEGORICAL_CANDIDATES if col in available]
    return numeric_features, categorical_features


def select_text_fields(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    available = set(train_df.columns) & set(test_df.columns)
    return [col for col in TEXT_CANDIDATES if col in available]


def select_follower_features(numeric_features: list[str]) -> list[str]:
    return [col for col in FOLLOWER_CANDIDATES if col in numeric_features]


def build_text_pipeline(
    selector_fn: FunctionTransformer,
    max_features: int,
    min_df: int,
    ngram_max: int,
    n_components: int,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        [
            ("selector", selector_fn),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=(1, ngram_max),
                    sublinear_tf=True,
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
        ]
    )


def effective_svd_components(
    series: pd.Series,
    max_features: int,
    min_df: int,
    ngram_max: int,
    requested_components: int,
) -> int:
    if requested_components <= 0:
        return 0
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, ngram_max),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(series)
    except ValueError:
        return 0
    if matrix.shape[1] <= 1:
        return 0
    return min(requested_components, matrix.shape[1] - 1)


def predict_with_thresholds(
    prob_matrix: np.ndarray,
    classes: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    adjusted = np.zeros_like(prob_matrix, dtype=float)
    class_to_idx = {label: index for index, label in enumerate(classes)}
    for label in classes:
        idx = class_to_idx[label]
        adjusted[:, idx] = prob_matrix[:, idx] / max(thresholds[label], 1e-8)
    return classes[adjusted.argmax(axis=1)]


def tune_thresholds_oof(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    threshold_grid: list[float],
) -> tuple[dict[str, float], float]:
    oof_proba = cross_val_predict(
        estimator,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    estimator.fit(x_train, y_train)
    classes = estimator.named_steps["model"].classes_
    best_thresholds = {label: 1.0 for label in classes}
    best_score = -1.0
    for values in product(threshold_grid, repeat=len(classes)):
        thresholds = {label: value for label, value in zip(classes, values)}
        pred = predict_with_thresholds(oof_proba, classes, thresholds)
        score = f1_score(
            y_train,
            pred,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        )
        if score > best_score:
            best_score = score
            best_thresholds = thresholds.copy()
    return best_thresholds, best_score


def build_preprocessor(
    x_train: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str],
    args: argparse.Namespace,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    if "question" in text_features:
        question_series = x_train["question"].fillna("").map(normalize_text)
        question_components = effective_svd_components(
            question_series,
            args.tfidf_max_features,
            args.tfidf_min_df,
            args.tfidf_ngram_max,
            args.question_svd_components,
        )
        if question_components > 0:
            transformers.append(
                (
                    "question_txt",
                    build_text_pipeline(
                        FunctionTransformer(
                            lambda x: x["question"].fillna("").map(normalize_text),
                            validate=False,
                        ),
                        args.tfidf_max_features,
                        args.tfidf_min_df,
                        args.tfidf_ngram_max,
                        question_components,
                        args.random_state,
                    ),
                    ["question"],
                )
            )

    event_text_fields = [
        column
        for column in ["event_group_label", "event_text", "normalized_event_text"]
        if column in text_features
    ]
    if event_text_fields:
        event_series = x_train[event_text_fields].apply(
            lambda row: combine_text_fields(row, event_text_fields),
            axis=1,
        )
        event_components = effective_svd_components(
            event_series,
            args.tfidf_max_features,
            args.tfidf_min_df,
            args.tfidf_ngram_max,
            args.event_svd_components,
        )
        if event_components > 0:
            transformers.append(
                (
                    "event_txt",
                    build_text_pipeline(
                        FunctionTransformer(
                            lambda x, cols=event_text_fields: x.apply(
                                lambda row: combine_text_fields(row, cols),
                                axis=1,
                            ),
                            validate=False,
                        ),
                        args.tfidf_max_features,
                        args.tfidf_min_df,
                        args.tfidf_ngram_max,
                        event_components,
                        args.random_state,
                    ),
                    event_text_fields,
                )
            )

    if not transformers:
        raise SystemExit("No usable numeric, categorical, or text features found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def main() -> None:
    args = parse_args()

    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise RuntimeError("Please install lightgbm first: pip install lightgbm") from exc

    print("Loading T1 data...")
    train_df, test_df = load_data(args.local_dir)

    numeric_features, categorical_features = select_features(train_df, test_df)
    text_features = select_text_fields(train_df, test_df)
    follower_features = select_follower_features(numeric_features)

    used_cols = sorted(set(numeric_features + categorical_features + text_features))
    if not used_cols:
        raise SystemExit("No overlapping features found between train and test.")

    x_train = train_df[used_cols].copy()
    x_test = test_df[used_cols].copy()
    y_train = train_df["interest_label"].astype(str)
    y_test = test_df["interest_label"].astype(str)

    preprocessor = build_preprocessor(
        x_train,
        numeric_features,
        categorical_features,
        text_features,
        args,
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=len(LABEL_ORDER),
                    random_state=args.random_state,
                    verbosity=-1,
                    n_jobs=1,
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.03, 0.1],
        "model__num_leaves": [7, 15, 31],
        "model__min_child_samples": [3, 10],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__class_weight": ["balanced"],
    }

    cv = StratifiedKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    best_model: Pipeline = search.best_estimator_

    thresholds = None
    threshold_score = None
    if args.use_threshold_tuning:
        thresholds, threshold_score = tune_thresholds_oof(
            best_model,
            x_train,
            y_train,
            cv,
            args.threshold_grid,
        )

    pred_proba = best_model.predict_proba(x_test)
    classes = best_model.named_steps["model"].classes_
    if thresholds is None:
        pred_labels = best_model.predict(x_test)
    else:
        pred_labels = predict_with_thresholds(pred_proba, classes, thresholds)

    acc = accuracy_score(y_test, pred_labels)
    macro_f1 = f1_score(
        y_test,
        pred_labels,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    print(f"numeric_features: {numeric_features}")
    print(f"follower_features: {follower_features}")
    print(f"categorical_features: {categorical_features}")
    print(f"text_features: {text_features}")
    print(f"best_params: {search.best_params_}")
    print(f"best_cv_macro_f1: {search.best_score_:.4f}")
    if thresholds is not None and threshold_score is not None:
        print(f"thresholds: {thresholds}")
        print(f"threshold_tuned_oof_macro_f1: {threshold_score:.4f}")

    print("\n--- Test Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")

    class_to_index = {label: index for index, label in enumerate(classes)}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (_, row) in enumerate(test_df.iterrows()):
            rec = {
                "condition_id": str(row["condition_id"]),
                "gold_label": str(y_test.iloc[index]),
                "pred_label": str(pred_labels[index]),
                "confidence": float(pred_proba[index].max()),
                "scores": {
                    label: float(pred_proba[index][class_to_index[label]])
                    for label in LABEL_ORDER
                },
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
