#!/usr/bin/env python3
"""T6 Structured LightGBM Baseline -- Cross-Market Propagation.

Trains the paper-style two-stage LightGBM classifier on unified T6 feature
JSONL rows: first propagation vs no-effect, then primary_mover vs
propagated_signal among propagated rows.  Uses the train/val/test split in
the data when present and reports Macro-F1.

Usage:
    python -m baselines.t6.lightgbm_baseline
    python -m baselines.t6.lightgbm_baseline --n-trials 20 --local-dir /path/to/data
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

try:
    from .data_utils import (
        FEATURE_COLS,
        LABEL_ORDER,
        LABEL_TO_ID,
        available_feature_cols,
        clean_t6_dataframe,
        load_t6_dataframe,
        select_eval_split,
    )
except ImportError:
    from data_utils import (
        FEATURE_COLS,
        LABEL_ORDER,
        LABEL_TO_ID,
        available_feature_cols,
        clean_t6_dataframe,
        load_t6_dataframe,
        select_eval_split,
    )

RANDOM_STATE = 42
N_TRIALS = 20


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def _select_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of paper T6 feature columns that exist."""
    return available_feature_cols(df)


def _build_weights(y: pd.Series, weight_power: float) -> np.ndarray:
    return np.power(compute_sample_weight(class_weight="balanced", y=y), weight_power)


def _train_binary_model(
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weights,
    *,
    n_trials: int,
    random_state: int,
    objective_metric: str = "f1",
    num_boost_round: int = 400,
    early_stopping_rounds: int = 20,
):
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "random_state": random_state,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        }
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        pred = (model.predict(X_val) >= 0.5).astype(int)
        if objective_metric == "accuracy":
            return accuracy_score(y_val, pred)
        return f1_score(y_val, pred, average="binary", zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "random_state": random_state,
        }
    )
    model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    return model, study


def _decode_predictions(
    propagation_prob: np.ndarray,
    class_prob: np.ndarray,
    propagation_threshold: float,
    propagated_threshold: float,
) -> np.ndarray:
    pred = np.full(len(propagation_prob), LABEL_TO_ID["no_cross_market_effect"], dtype=int)
    active = propagation_prob >= propagation_threshold
    pred[active] = np.where(
        class_prob[active] >= propagated_threshold,
        LABEL_TO_ID["propagated_signal"],
        LABEL_TO_ID["primary_mover"],
    )
    return pred


def _tune_thresholds(
    propagation_prob: np.ndarray,
    class_prob: np.ndarray,
    y_val,
    *,
    min_primary_rate: float,
    min_propagated_rate: float,
):
    best = None
    grid = np.arange(0.2, 0.81, 0.05)
    min_primary = int(round(len(y_val) * min_primary_rate))
    min_propagated = int(round(len(y_val) * min_propagated_rate))

    for propagation_threshold in grid:
        for propagated_threshold in grid:
            pred = _decode_predictions(
                propagation_prob,
                class_prob,
                float(propagation_threshold),
                float(propagated_threshold),
            )
            primary_count = int(np.sum(pred == LABEL_TO_ID["primary_mover"]))
            propagated_count = int(np.sum(pred == LABEL_TO_ID["propagated_signal"]))
            if primary_count < min_primary or propagated_count < min_propagated:
                continue

            acc = accuracy_score(y_val, pred)
            mf1 = f1_score(y_val, pred, average="macro", zero_division=0)
            score = float(0.8 * mf1 + 0.2 * acc)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "propagation_threshold": float(propagation_threshold),
                    "propagated_threshold": float(propagated_threshold),
                    "val_accuracy": float(acc),
                    "val_macro_f1": float(mf1),
                    "val_primary_predictions": primary_count,
                    "val_propagated_predictions": propagated_count,
                }

    if best is None:
        for propagation_threshold in grid:
            for propagated_threshold in grid:
                pred = _decode_predictions(
                    propagation_prob,
                    class_prob,
                    float(propagation_threshold),
                    float(propagated_threshold),
                )
                acc = accuracy_score(y_val, pred)
                mf1 = f1_score(y_val, pred, average="macro", zero_division=0)
                score = float(0.8 * mf1 + 0.2 * acc)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "propagation_threshold": float(propagation_threshold),
                        "propagated_threshold": float(propagated_threshold),
                        "val_accuracy": float(acc),
                        "val_macro_f1": float(mf1),
                        "val_primary_predictions": int(np.sum(pred == LABEL_TO_ID["primary_mover"])),
                        "val_propagated_predictions": int(np.sum(pred == LABEL_TO_ID["propagated_signal"])),
                    }
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T6 LightGBM cross-market baseline")
    parser.add_argument("--repo", default="mlsys-io/EventXBench")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--feature-file", default=None,
                        help="Path to unified T6 feature JSONL with split column.")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--include-confounded-eval", action="store_true")
    parser.add_argument("--include-insufficient", action="store_true")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--min-primary-rate", type=float, default=0.03)
    parser.add_argument("--min-propagated-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    full_df = load_t6_dataframe(args.feature_file, args.local_dir, repo=args.repo)
    full_df = clean_t6_dataframe(
        full_df,
        include_insufficient=args.include_insufficient,
        include_confounded=True,
    )

    if "split" in full_df.columns:
        train_df = full_df[full_df["split"] == "train"].copy()
        val_df = full_df[full_df["split"] == "val"].copy()
        test_df = select_eval_split(full_df, args.eval_split)
    else:
        train_df, test_df = train_test_split(
            full_df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=full_df["label"],
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=train_df["label"],
        )

    if not args.include_confounded_eval:
        if "confound_flag" in val_df.columns:
            val_df = val_df[val_df["confound_flag"] == False].copy()
        if "confound_flag" in test_df.columns:
            test_df = test_df[test_df["confound_flag"] == False].copy()

    # Determine features
    feature_cols = _select_features(train_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in T6 data.")
    missing_features = [col for col in FEATURE_COLS if col not in feature_cols]

    for frame in (train_df, val_df, test_df):
        frame[feature_cols] = frame[feature_cols].fillna(0.0)

    y_train = train_df["label"].map(LABEL_TO_ID)
    y_val = val_df["label"].map(LABEL_TO_ID)
    y_test = test_df["label"].map(LABEL_TO_ID)
    X_train = train_df[feature_cols].astype(float)
    X_val = val_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    print(
        f"Train: {len(train_df)}, Val: {len(val_df)}, "
        f"Eval({args.eval_split}): {len(test_df)}, Features: {len(feature_cols)}"
    )
    print(f"Features: {feature_cols}")
    if missing_features:
        print(f"Missing paper feature columns: {missing_features}")
    print(f"Train class distribution:\n{train_df['label'].value_counts().to_string()}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_train_prop = (y_train != LABEL_TO_ID["no_cross_market_effect"]).astype(int)
    y_val_prop = (y_val != LABEL_TO_ID["no_cross_market_effect"]).astype(int)
    prop_weights = _build_weights(y_train_prop, args.weight_power)
    propagation_model, propagation_study = _train_binary_model(
        X_train,
        y_train_prop,
        X_val,
        y_val_prop,
        prop_weights,
        n_trials=args.n_trials,
        random_state=RANDOM_STATE,
        objective_metric="f1",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    train_pos = train_df[train_df["label"] != "no_cross_market_effect"].copy()
    val_pos = val_df[val_df["label"] != "no_cross_market_effect"].copy()
    X_train_pos = train_pos[feature_cols].astype(float)
    X_val_pos = val_pos[feature_cols].astype(float)
    y_train_pos = (train_pos["label"] == "propagated_signal").astype(int)
    y_val_pos = (val_pos["label"] == "propagated_signal").astype(int)
    pos_weights = _build_weights(y_train_pos, args.weight_power)
    class_model, class_study = _train_binary_model(
        X_train_pos,
        y_train_pos,
        X_val_pos,
        y_val_pos,
        pos_weights,
        n_trials=args.n_trials,
        random_state=RANDOM_STATE + 1,
        objective_metric="f1",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # -- Evaluate on test set -----------------------------------------------
    val_prop_prob = propagation_model.predict(X_val)
    val_class_prob = class_model.predict(X_val)
    threshold_info = _tune_thresholds(
        val_prop_prob,
        val_class_prob,
        y_val,
        min_primary_rate=args.min_primary_rate,
        min_propagated_rate=args.min_propagated_rate,
    )

    test_prop_prob = propagation_model.predict(X_test)
    test_class_prob = class_model.predict(X_test)
    pred_test = _decode_predictions(
        test_prop_prob,
        test_class_prob,
        threshold_info["propagation_threshold"],
        threshold_info["propagated_threshold"],
    )

    test_macro_f1 = f1_score(y_test, pred_test, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, pred_test)
    cm = confusion_matrix(y_test, pred_test, labels=list(range(len(LABEL_ORDER))))

    print(f"\n=== T6 Structured LightGBM Results ===")
    print(f"  Test Macro-F1: {test_macro_f1:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Threshold tuning: {threshold_info}")
    print(
        "  Best params: "
        f"propagation={propagation_study.best_params}, class={class_study.best_params}"
    )
    print(f"  Confusion matrix ({LABEL_ORDER}): {cm.tolist()}")

    # Feature importance
    feat_imp = []
    for stage_name, model in (
        ("propagation_stage", propagation_model),
        ("class_stage", class_model),
    ):
        feat_imp.extend(
            (f"{stage_name}:{name}", imp)
            for name, imp in zip(feature_cols, model.feature_importance(importance_type="gain"))
        )
    feat_imp = sorted(feat_imp, key=lambda x: x[1], reverse=True)
    print("  Top 5 features:")
    for name, imp in feat_imp[:5]:
        print(f"    {name}: {imp:.1f}")


if __name__ == "__main__":
    main()
