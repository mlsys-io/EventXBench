#!/usr/bin/env python3
"""T5 LightGBM baseline aligned with the Task7-style script.

This script evaluates:
1) Regression on price_impact and volume_multiplier across 5 horizons
2) Classification on decay_class

Both are reported under two scopes:
- all
- non_confounded
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import eventxbench
from huggingface_hub import hf_hub_download

try:
    from scipy.stats import spearmanr as scipy_spearmanr
except Exception:
    scipy_spearmanr = None


N_TRIALS = 10
RANDOM_STATE = 42
TARGET_METRICS = ["price_impact", "volume_multiplier"]
TARGET_HORIZONS = ["15m", "30m", "1h", "2h", "6h"]
DECAY_LABELS = ["transient", "sustained", "reversal"]

# Keep this feature set aligned with your reference script.
REFERENCE_FEATURE_COLS = [
    "like_count",
    "reply_count",
    "view_count",
    "follower_count",
    "price_t0",
    "volume_24h_baseline",
    "category_sports",
    "category_crypto / digital assets",
    "category_elections / politics",
    "category_entertainment / awards",
    "category_company / product announcements",
    "finbert_pos_prob",
]


def _parse_json_col(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def load_jsonl_df(file_path: Path) -> pd.DataFrame:
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def try_load_external_features(local_dir: str | None, feature_file: str | None) -> pd.DataFrame | None:
    candidates = []
    if feature_file:
        candidates.append(Path(feature_file))
    if local_dir:
        candidates.append(Path(local_dir) / "raw" / "t4_db_features.jsonl")
    candidates.append(Path("data") / "raw" / "t4_db_features.jsonl")

    for path in candidates:
        if path.exists():
            print(f"Loading external feature table from local file: {path}")
            return load_jsonl_df(path)

    try:
        hf_path = hf_hub_download(
            repo_id="mlsys-io/EventXBench",
            repo_type="dataset",
            filename="raw/t4_db_features.jsonl",
        )
        print(f"Loading external feature table from HF cache: {hf_path}")
        return load_jsonl_df(Path(hf_path))
    except Exception as exc:
        print(f"Warning: unable to load external feature table from HF: {exc}")
        return None


def resolve_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in REFERENCE_FEATURE_COLS if c in df.columns]
    if cols:
        return cols

    # Fallback only when reference feature table is unavailable.
    blocked = {
        "tweet_id",
        "condition_id",
        "confound_flag",
        "decay_class",
        "price_impact_json",
        "volume_multiplier_json",
    }
    blocked.update(f"price_impact_{h}" for h in TARGET_HORIZONS)
    blocked.update(f"volume_multiplier_{h}" for h in TARGET_HORIZONS)
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def pearson_corr(x, y):
    n = len(x)
    if n < 2:
        return None

    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    var_x = sum((a - mx) ** 2 for a in x)
    var_y = sum((b - my) ** 2 for b in y)

    if var_x == 0 or var_y == 0:
        return None

    return cov / math.sqrt(var_x * var_y)


def spearman_rho(x, y):
    if len(x) < 2 or len(y) < 2:
        return None

    if scipy_spearmanr is not None:
        rho = scipy_spearmanr(x, y).statistic
        if rho is None or math.isnan(rho):
            return None
        return float(rho)

    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def build_targets_from_json(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in TARGET_METRICS:
        json_col = f"{metric}_json"
        parsed = out[json_col].apply(_parse_json_col) if json_col in out.columns else pd.Series([{}] * len(out))

        for h in TARGET_HORIZONS:
            col = f"{metric}_{h}"
            if col not in out.columns:
                out[col] = parsed.apply(lambda d: d.get(h, np.nan))
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def run_decay_class_target(df, feature_cols, n_trials, seed):
    target_col = "decay_class"
    required_cols = feature_cols + [target_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Missing columns for {target_col}: {missing_cols}")
        return None

    work_df = df[required_cols].dropna().copy()
    work_df = work_df[work_df[target_col].isin(DECAY_LABELS)].copy()
    if len(work_df) < 30:
        print(f"Insufficient rows for {target_col}: {len(work_df)}")
        return None

    label_to_id = {lab: i for i, lab in enumerate(DECAY_LABELS)}
    y = work_df[target_col].map(label_to_id)
    X = work_df[feature_cols]
    sample_weights = compute_sample_weight("balanced", y)

    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        list(skf.split(X, y))
    except ValueError:
        print("Warning: classes too imbalanced for StratifiedKFold, falling back to KFold.")
        skf = KFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": len(DECAY_LABELS),
            "metric": "multi_logloss",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": seed,
        }

        oof_preds = np.zeros((len(X), len(DECAY_LABELS)))
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, y_tr, w_tr = X.iloc[tr_idx], y.iloc[tr_idx], sample_weights[tr_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            gbm = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            oof_preds[val_idx] = gbm.predict(X_val)

        pred_labels = np.argmax(oof_preds, axis=1)
        return f1_score(y, pred_labels, average="macro", zero_division=0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "multiclass",
            "num_class": len(DECAY_LABELS),
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": seed,
        }
    )

    oof_preds = np.zeros((len(X), len(DECAY_LABELS)))
    feat_imp_accum = np.zeros(len(feature_cols), dtype=float)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr, w_tr = X.iloc[tr_idx], y.iloc[tr_idx], sample_weights[tr_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        fold_model = lgb.train(
            best_params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        oof_preds[val_idx] = fold_model.predict(X_val)
        feat_imp_accum += fold_model.feature_importance(importance_type="gain")

    final_pred_labels = np.argmax(oof_preds, axis=1)
    macro_f1 = f1_score(y, final_pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, final_pred_labels, average="weighted", zero_division=0)
    acc = accuracy_score(y, final_pred_labels)

    feat_imp = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": feat_imp_accum / skf.n_splits,
        }
    ).sort_values(by="Importance", ascending=False)

    return {
        "target": target_col,
        "samples": len(work_df),
        "cv_folds": 5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "best_params": study.best_params,
        "top3_features": feat_imp.head(3).to_dict("records"),
    }


def run_one_target(df, target_col, feature_cols, n_trials, seed):
    required_cols = feature_cols + [target_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Missing columns for {target_col}: {missing_cols}")
        return None

    work_df = df[required_cols].dropna().copy()
    if len(work_df) < 30:
        print(f"Insufficient rows for {target_col}: {len(work_df)}")
        return None

    X = work_df[feature_cols]
    y_raw = work_df[target_col].astype(float)

    if (y_raw <= -1.0).any():
        print(f"Skip {target_col}: found values <= -1, cannot apply log1p safely")
        return None
    y_log = np.log1p(y_raw)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": seed,
        }

        oof_preds_raw = np.zeros(len(X), dtype=float)
        for tr_idx, val_idx in kf.split(X):
            X_tr = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr_log = y_log.iloc[tr_idx]
            y_val_log = y_log.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr_log)
            dval = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)
            gbm = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )

            pred_log = gbm.predict(X_val)
            oof_preds_raw[val_idx] = np.expm1(pred_log)

        rho = spearman_rho(y_raw.tolist(), oof_preds_raw.tolist())
        return -1.0 if rho is None else rho

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": seed,
        }
    )

    oof_preds_raw = np.zeros(len(X), dtype=float)
    feat_imp_accum = np.zeros(len(feature_cols), dtype=float)

    for tr_idx, val_idx in kf.split(X):
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr_log = y_log.iloc[tr_idx]
        y_val_log = y_log.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr_log)
        dval = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)
        fold_model = lgb.train(
            best_params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        pred_log = fold_model.predict(X_val)
        oof_preds_raw[val_idx] = np.expm1(pred_log)
        feat_imp_accum += fold_model.feature_importance(importance_type="gain")

    rho = spearman_rho(y_raw.tolist(), oof_preds_raw.tolist())
    mae = mean_absolute_error(y_raw, oof_preds_raw)
    rmse = np.sqrt(mean_squared_error(y_raw, oof_preds_raw))

    feat_imp = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": feat_imp_accum / 5.0,
        }
    ).sort_values(by="Importance", ascending=False)

    return {
        "target": target_col,
        "samples": len(work_df),
        "cv_folds": 5,
        "target_transform": "log1p/expm1",
        "rho": rho,
        "mae": mae,
        "rmse": rmse,
        "best_params": study.best_params,
        "top3_features": feat_imp.head(3).to_dict("records"),
        "oof_y_true": y_raw.tolist(),
        "oof_y_pred": oof_preds_raw.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 LightGBM (regression + decay classification)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--feature-file", default=None, help="Optional path to t4_db_features.jsonl")
    args = parser.parse_args()

    print("=" * 90)
    print("TASK5 LIGHTGBM (REGRESSION + DECAY CLASSIFICATION)")
    print("=" * 90)

    data = eventxbench.load_task("t5", local_dir=args.local_dir)
    if isinstance(data, tuple):
        train_df, test_df = data
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = data.copy()

    df = build_targets_from_json(df)

    ext_features = try_load_external_features(args.local_dir, args.feature_file)
    if ext_features is not None and "tweet_id" in ext_features.columns:
        keep_cols = ["tweet_id"] + [c for c in REFERENCE_FEATURE_COLS if c in ext_features.columns]
        ext_features = ext_features[keep_cols].drop_duplicates(subset=["tweet_id"])
        df = df.merge(ext_features, on="tweet_id", how="left")

    feature_cols = resolve_feature_cols(df)
    if not feature_cols:
        raise ValueError("No usable feature columns found for modeling.")
    print(f"Using {len(feature_cols)} features.")
    print(f"Feature columns: {feature_cols}")

    if "confound_flag" not in df.columns:
        raise ValueError("Missing required column: confound_flag")

    df_all = df.copy()
    df_nc = df[df["confound_flag"] == False].copy()

    print(f"All T5 rows: {len(df_all)}")
    print(f"Non-confounded T5 rows: {len(df_nc)}")

    summary_rows = []
    scopes = [("all", df_all), ("non_confounded", df_nc)]

    for scope_name, df_scope in scopes:
        print("\n" + "=" * 90)
        print(f"SCOPE: {scope_name} (rows={len(df_scope)})")
        print("=" * 90)

        for metric in TARGET_METRICS:
            print("\n" + "#" * 90)
            print(f"METRIC: {metric} | SCOPE: {scope_name}")
            print("#" * 90)

            metric_target_cols = [f"{metric}_{h}" for h in TARGET_HORIZONS]
            metric_required_cols = feature_cols + metric_target_cols
            metric_df = df_scope[metric_required_cols].dropna().copy()
            if len(metric_df) < 30:
                print(f"Skip metric {metric}: insufficient complete rows across horizons ({len(metric_df)})")
                continue

            flat_true = []
            flat_pred = []

            for horizon in TARGET_HORIZONS:
                target_col = f"{metric}_{horizon}"
                print("\n" + "-" * 90)
                print(f"Training target: {target_col}")
                print("-" * 90)

                result = run_one_target(metric_df, target_col, feature_cols, args.n_trials, args.seed)
                if result is None:
                    print(f"Skip {target_col}")
                    continue

                rho_display = "N/A" if result["rho"] is None else f"{result['rho']:.4f}"
                print(f"Samples: {result['samples']}")
                print(f"CV folds: {result['cv_folds']} | Target transform: {result['target_transform']}")
                print(f"OOF Spearman rho: {rho_display}")
                print(f"OOF MAE: {result['mae']:.4f}")
                print(f"OOF RMSE: {result['rmse']:.4f}")
                print(f"Best params: {result['best_params']}")
                print("Top 3 features:")
                for item in result["top3_features"]:
                    print(f"  - {item['Feature']}: {item['Importance']:.4f}")

                flat_true.extend(result["oof_y_true"])
                flat_pred.extend(result["oof_y_pred"])

            if flat_true and flat_pred:
                rho_flat = spearman_rho(flat_true, flat_pred)
                mae_flat = mean_absolute_error(flat_true, flat_pred)
                rmse_flat = np.sqrt(mean_squared_error(flat_true, flat_pred))
                n_events = len(metric_df)
                n_points = len(flat_true)

                rho_flat_display = "N/A" if rho_flat is None else f"{rho_flat:.4f}"
                print("\n" + "~" * 90)
                print(
                    f"FINAL FLATTENED ({metric} | {scope_name}) | "
                    f"events={n_events} points={n_points} | "
                    f"rho={rho_flat_display} mae={mae_flat:.4f} rmse={rmse_flat:.4f}"
                )
                print("~" * 90)

                summary_rows.append(
                    {
                        "scope": scope_name,
                        "metric": metric,
                        "horizon": "flat_all_horizons",
                        "target": f"{metric}_flat_all_horizons",
                        "task_type": "regression",
                        "samples": n_events,
                        "points": n_points,
                        "rho": rho_flat,
                        "mae": mae_flat,
                        "rmse": rmse_flat,
                    }
                )

        print("\n" + "#" * 90)
        print(f"TASK: decay_class (classification) | SCOPE: {scope_name}")
        print("#" * 90)
        decay_result = run_decay_class_target(df_scope, feature_cols, args.n_trials, args.seed)
        if decay_result is None:
            print("Skip decay_class")
        else:
            print(f"Samples: {decay_result['samples']}")
            print(f"Macro-F1: {decay_result['macro_f1']:.4f}")
            print(f"Weighted-F1: {decay_result['weighted_f1']:.4f}")
            print(f"Accuracy: {decay_result['accuracy']:.4f}")
            print(f"Best params: {decay_result['best_params']}")
            print("Top 3 features:")
            for item in decay_result["top3_features"]:
                print(f"  - {item['Feature']}: {item['Importance']:.4f}")

            summary_rows.append(
                {
                    "scope": scope_name,
                    "metric": "decay_class",
                    "horizon": "N/A",
                    "target": "decay_class",
                    "task_type": "classification",
                    "samples": decay_result["samples"],
                    "rho": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "macro_f1": decay_result["macro_f1"],
                    "weighted_f1": decay_result["weighted_f1"],
                    "accuracy": decay_result["accuracy"],
                }
            )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    if not summary_rows:
        print("No successful runs.")
        return

    summary_df = pd.DataFrame(summary_rows)
    if "rho" in summary_df.columns:
        summary_df["rho"] = pd.to_numeric(summary_df["rho"], errors="coerce")

    reg_df = summary_df[summary_df["task_type"] == "regression"].copy()
    cls_df = summary_df[summary_df["task_type"] == "classification"].copy()

    if not reg_df.empty:
        print("\n[Regression Summary]")
        print(reg_df[["scope", "metric", "horizon", "samples", "points", "rho", "mae", "rmse"]].to_string(index=False))

    if not cls_df.empty:
        print("\n[Classification Summary]")
        print(cls_df[["scope", "metric", "samples", "macro_f1", "weighted_f1", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
