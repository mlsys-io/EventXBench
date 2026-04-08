#!/usr/bin/env python3
"""T4 LightGBM Baseline -- Reproduction of classification + regression reports.

This script reproduces two workflows:
1) Classification for direction / magnitude tiers.
2) Regression for delta_30m / delta_2h / delta_6h with Spearman rho report.

Primary feature source is `t4_db_features.jsonl` (HF raw file or local path).
If split is not present in the feature file, split is inferred by joining against
the official T4 train/test split from `eventxbench.load_task("t4")`.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
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
    "finbert_question_pos_prob",
]

DELTA_COLS = ["delta_30m", "delta_2h", "delta_6h"]
HORIZON_NAMES = ["30m", "2h", "6h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 LightGBM baseline (classification + regression)")
    parser.add_argument("--local-dir", default=None, help="Local data directory for eventxbench.load_task")
    parser.add_argument("--feature-file", default=None, help="Path to t4_db_features.jsonl")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["official", "random"],
        default="official",
        help="official: prefer train/test split; random: force random split like original scripts",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Used only when split is unavailable")
    parser.add_argument("--output-cls", default="t4_lightgbm_classification.jsonl")
    parser.add_argument("--output-reg", default="t4_lightgbm_regression.jsonl")
    return parser.parse_args()


def _norm_id(v: object) -> str:
    return "" if v is None else str(v).strip()


def _build_key_df(df: pd.DataFrame) -> pd.Series:
    has_condition = "condition_id" in df.columns
    has_market = "market_id" in df.columns
    if has_condition:
        return df["tweet_id"].map(_norm_id) + "__" + df["condition_id"].map(_norm_id)
    if has_market:
        return df["tweet_id"].map(_norm_id) + "__" + df["market_id"].map(_norm_id)
    return df["tweet_id"].map(_norm_id)


def load_feature_table(feature_file: Optional[str]) -> pd.DataFrame:
    if feature_file:
        path = Path(feature_file)
        if not path.exists():
            raise SystemExit(f"Feature file not found: {path}")
        df = pd.read_json(path, lines=True)
        print(f"Loaded features from local file: {path}")
        return df

    # Try workspace-local raw file first.
    workspace_raw = Path(__file__).resolve().parents[2] / "raw" / "t4_db_features.jsonl"
    if workspace_raw.exists():
        df = pd.read_json(workspace_raw, lines=True)
        print(f"Loaded features from workspace raw file: {workspace_raw}")
        return df

    # Fall back to HF gated dataset.
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="mlsys-io/EventXBench",
            filename="raw/t4_db_features.jsonl",
            repo_type="dataset",
        )
        df = pd.read_json(downloaded, lines=True)
        print(f"Loaded features from HF: {downloaded}")
        return df
    except Exception as exc:
        raise SystemExit(
            "Unable to load t4_db_features.jsonl. Provide --feature-file, place it at raw/t4_db_features.jsonl, "
            "or login to HF if the dataset is gated."
        ) from exc


def attach_split_from_t4(df_feat: pd.DataFrame, local_dir: Optional[str]) -> pd.DataFrame:
    if "split" in df_feat.columns:
        split_vals = df_feat["split"].astype(str).str.lower().unique().tolist()
        if any(v in {"train", "test"} for v in split_vals):
            df = df_feat.copy()
            df["split"] = df["split"].astype(str).str.lower()
            return df

    import eventxbench

    loaded = eventxbench.load_task("t4", local_dir=local_dir)
    if not isinstance(loaded, tuple):
        # fallback: random split later
        return df_feat.copy()

    train_df, test_df = loaded

    # If the feature table has no market-level ID, align by tweet_id only.
    feat_has_market_level = ("condition_id" in df_feat.columns) or ("market_id" in df_feat.columns)
    task_has_market_level = "condition_id" in train_df.columns

    if feat_has_market_level and task_has_market_level:
        train_keys = set((_build_key_df(train_df)).tolist())
        test_keys = set((_build_key_df(test_df)).tolist())
    else:
        train_keys = set(train_df["tweet_id"].map(_norm_id).tolist())
        test_keys = set(test_df["tweet_id"].map(_norm_id).tolist())

    out = df_feat.copy()
    if feat_has_market_level and task_has_market_level:
        keys = _build_key_df(out)
    else:
        keys = out["tweet_id"].map(_norm_id)
    out["split"] = ""
    out.loc[keys.isin(train_keys), "split"] = "train"
    out.loc[keys.isin(test_keys), "split"] = "test"
    return out


def attach_deltas_from_t4(df_feat: pd.DataFrame, local_dir: Optional[str]) -> pd.DataFrame:
    """Attach delta_30m/delta_2h/delta_6h from official T4 labels when missing."""
    if all(col in df_feat.columns for col in DELTA_COLS):
        return df_feat

    import eventxbench

    loaded = eventxbench.load_task("t4", local_dir=local_dir)
    if isinstance(loaded, tuple):
        t4_full = pd.concat(loaded, ignore_index=True)
    else:
        t4_full = loaded

    needed = ["tweet_id", "condition_id"] + DELTA_COLS
    existing_needed = [c for c in needed if c in t4_full.columns]
    if not all(c in existing_needed for c in ["tweet_id"] + DELTA_COLS):
        return df_feat

    t4_small = t4_full[existing_needed].copy()

    out = df_feat.copy()
    if "condition_id" in out.columns and "condition_id" in t4_small.columns:
        out = out.merge(t4_small, on=["tweet_id", "condition_id"], how="left", suffixes=("", "_from_t4"))
    else:
        # Fallback to tweet-only join for feature tables without condition_id.
        t4_small = t4_small.drop_duplicates(subset=["tweet_id"])
        out = out.merge(t4_small, on=["tweet_id"], how="left", suffixes=("", "_from_t4"))

    # Consolidate potential suffix columns.
    for c in DELTA_COLS:
        if c not in out.columns and f"{c}_from_t4" in out.columns:
            out[c] = out[f"{c}_from_t4"]
        elif c in out.columns and f"{c}_from_t4" in out.columns:
            out[c] = out[c].fillna(out[f"{c}_from_t4"])
    drop_cols = [f"{c}_from_t4" for c in DELTA_COLS if f"{c}_from_t4" in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out


def rankdata(values: list[float]) -> list[float]:
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


def pearson_corr(x: list[float], y: list[float]) -> Optional[float]:
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


def spearman_rho(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    try:
        from scipy.stats import spearmanr

        raw: Any = spearmanr(x, y)
        if isinstance(raw, tuple):
            rho_val = raw[0]
        else:
            rho_val = getattr(raw, "statistic", None)
        if rho_val is None:
            return None
        rho = float(rho_val)
        if rho is None or math.isnan(rho):
            return None
        return float(rho)
    except Exception:
        return pearson_corr(rankdata(x), rankdata(y))


def get_active_mask(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col == "direction_label":
        return (df["confound_flag"] == False) & (df["direction_label"] != "flat")
    if target_col == "magnitude_bucket":
        return (df["confound_flag"] == False) & (df["direction_label"] != "flat")
    return df["confound_flag"] == False


def get_label_mapping(target_col: str, tier_name: str) -> dict[str, int]:
    if target_col == "direction_label":
        if tier_name == "Tier3 Active (Non-confounded + Non-flat)":
            return {"down": 0, "up": 1}
        return {"flat": 0, "up": 1, "down": 2}
    if target_col == "magnitude_bucket":
        if tier_name == "Tier3 Active (Non-confounded + Non-small)":
            return {"medium": 0, "large": 1}
        return {"small": 0, "medium": 1, "large": 2}
    raise ValueError(f"Unsupported target: {target_col}")


def predict_labels(model, x_test: pd.DataFrame, n_classes: int) -> np.ndarray:
    preds_prob = model.predict(x_test)
    if n_classes == 2:
        return (preds_prob >= 0.5).astype(int)
    return np.argmax(preds_prob, axis=1)


def split_train_test(
    df_tier: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float,
    split_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if split_mode == "official" and "split" in df_tier.columns:
        split_col = df_tier["split"].astype(str).str.lower()
        tr_mask = split_col == "train"
        te_mask = split_col == "test"
        if tr_mask.any() and te_mask.any():
            return (
                df_tier.loc[tr_mask],
                df_tier.loc[te_mask],
                y.loc[tr_mask],
                y.loc[te_mask],
            )

    x_train, x_test, y_train, y_test = train_test_split(
        df_tier,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def run_one_tier_classification(
    df_tier: pd.DataFrame,
    tier_name: str,
    target_col: str,
    features: list[str],
    n_trials: int,
    seed: int,
    test_size: float,
    split_mode: str,
) -> Optional[dict]:
    import lightgbm as lgb
    import optuna

    print("\n" + "=" * 70)
    print(f"{tier_name} | TARGET: {target_col}")
    print("=" * 70)

    label_mapping = get_label_mapping(target_col, tier_name)
    class_count = len(label_mapping)

    y = df_tier[target_col].map(label_mapping)
    valid_mask = y.notnull()
    df_tier = df_tier[valid_mask].copy()
    y = y[valid_mask].astype(int)

    if len(df_tier) < 10:
        print(f"Insufficient samples: {len(df_tier)}")
        return None

    X = df_tier[features].copy().astype(float)
    print(f"Samples: {len(df_tier)} | Classes: {class_count}")
    print("Class distribution:")
    print(df_tier[target_col].value_counts())

    x_train_df, x_test_df, y_train, y_test = split_train_test(df_tier, y, seed, test_size, split_mode)
    X_train = x_train_df[features].astype(float)
    X_test = x_test_df[features].astype(float)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "random_state": seed,
        }
        if class_count == 2:
            params.update({"objective": "binary", "metric": "binary_logloss"})
        else:
            params.update({"objective": "multiclass", "num_class": class_count, "metric": "multi_logloss"})

        x_tr, x_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)

        gbm = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        pred_labels = predict_labels(gbm, x_val, class_count)
        return float(accuracy_score(y_val, pred_labels))

    print(f"Starting Bayesian Optimization (n_trials={n_trials})...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("Best Hyperparameters:")
    print(study.best_params)

    best_params = study.best_params.copy()
    best_params.update({"verbosity": -1, "random_state": seed})
    if class_count == 2:
        best_params.update({"objective": "binary", "metric": "binary_logloss"})
    else:
        best_params.update({"objective": "multiclass", "num_class": class_count, "metric": "multi_logloss"})

    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(best_params, dtrain_full, num_boost_round=150)

    test_preds = predict_labels(final_model, X_test, class_count)
    acc = float(accuracy_score(y_test, test_preds))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    print("\nFinal Test Results")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1: {macro_f1 * 100:.2f}%")

    feat_imp = pd.DataFrame(
        {
            "Feature": features,
            "Importance": final_model.feature_importance(importance_type="gain"),
        }
    ).sort_values(by="Importance", ascending=False)
    print("Top 5 Feature Importances (Gain):")
    print(feat_imp.head(5))

    return {
        "tier": tier_name,
        "target": target_col,
        "samples": int(len(df_tier)),
        "classes": int(class_count),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def run_regression_tier(
    df_tier: pd.DataFrame,
    tier_name: str,
    features: list[str],
    delta_cols: list[str],
    horizon_names: list[str],
    n_trials: int,
    seed: int,
    test_size: float,
    split_mode: str,
) -> Optional[dict]:
    import lightgbm as lgb
    import optuna

    print("\n" + "=" * 84)
    print(f"{tier_name} | REGRESSION: Predicting Delta Values")
    print("=" * 84)

    required_cols = features + delta_cols
    missing_cols = [col for col in required_cols if col not in df_tier.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, skipping tier")
        return None

    valid_mask = df_tier[delta_cols].notnull().all(axis=1)
    df_tier_clean = df_tier[valid_mask].copy()

    if len(df_tier_clean) < 10:
        print(f"Insufficient samples: {len(df_tier_clean)}")
        return None

    X = df_tier_clean[features].astype(float)
    print(f"Samples: {len(df_tier_clean)}")

    if split_mode == "official" and "split" in df_tier_clean.columns:
        split_col = df_tier_clean["split"].astype(str).str.lower()
        tr_mask = split_col == "train"
        te_mask = split_col == "test"
        if tr_mask.any() and te_mask.any():
            X_train = X.loc[tr_mask]
            X_test = X.loc[te_mask]
            idx_train = X_train.index
            idx_test = X_test.index
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)
            idx_train = X_train.index
            idx_test = X_test.index
    else:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)
        idx_train = X_train.index
        idx_test = X_test.index

    all_results = {
        "tier": tier_name,
        "samples": int(len(df_tier_clean)),
        "spearman_by_horizon": {},
        "spearman_flat": None,
        "mae_by_horizon": {},
    }

    all_pred: list[float] = []
    all_actual: list[float] = []

    for delta_col, horizon_name in zip(delta_cols, horizon_names):
        print(f"\nTraining regression for {horizon_name} (target: {delta_col})...")

        y_train = df_tier_clean.loc[idx_train, delta_col].astype(float)
        y_test = df_tier_clean.loc[idx_test, delta_col].astype(float)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "verbosity": -1,
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "mae",
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "random_state": seed,
            }

            x_tr, x_val, y_tr, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=seed,
            )
            dtrain = lgb.Dataset(x_tr, label=y_tr)
            dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)

            gbm = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )

            pred_val = np.asarray(gbm.predict(x_val), dtype=float)
            mae = mean_absolute_error(y_val, pred_val)
            return float(-mae)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        print(f"Best MAE: {-study.best_value:.4f}")
        print(f"Best Hyperparameters: {study.best_params}")

        best_params = study.best_params.copy()
        best_params.update(
            {
                "verbosity": -1,
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "mae",
                "random_state": seed,
            }
        )

        dtrain_full = lgb.Dataset(X_train, label=y_train)
        final_model = lgb.train(best_params, dtrain_full, num_boost_round=150)

        y_pred = np.asarray(final_model.predict(X_test), dtype=float)

        rho = spearman_rho(y_pred.tolist(), y_test.tolist())
        mae_test = float(mean_absolute_error(y_test, y_pred))
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"Test MAE: {mae_test:.4f}")
        print(f"Test RMSE: {rmse_test:.4f}")
        if rho is not None:
            print(f"Spearman rho @ {horizon_name}: {rho:.4f}")
        else:
            print(f"Spearman rho @ {horizon_name}: N/A")

        all_results["spearman_by_horizon"][horizon_name] = rho
        all_results["mae_by_horizon"][horizon_name] = mae_test

        all_pred.extend(y_pred.tolist())
        all_actual.extend(y_test.tolist())

        feat_imp = pd.DataFrame(
            {"Feature": features, "Importance": final_model.feature_importance(importance_type="gain")}
        ).sort_values(by="Importance", ascending=False)
        print(f"Top 3 Features for {horizon_name}:")
        print(feat_imp.head(3).to_string(index=False))

    rho_flat = spearman_rho(all_pred, all_actual)
    if rho_flat is not None:
        print(f"\nSpearman rho @ ALL horizons (flatten): {rho_flat:.4f}")
    else:
        print("\nSpearman rho @ ALL horizons (flatten): N/A")

    all_results["spearman_flat"] = rho_flat
    return all_results


def main() -> None:
    args = parse_args()

    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        raise SystemExit("Install lightgbm: pip install lightgbm")
    try:
        import optuna  # noqa: F401
    except ImportError:
        raise SystemExit("Install optuna: pip install optuna")

    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading feature data...")
    df = load_feature_table(args.feature_file)
    print(f"Total records in JSONL: {len(df)}")

    # Attach train/test split if missing.
    df = attach_split_from_t4(df, args.local_dir)
    df = attach_deltas_from_t4(df, args.local_dir)

    # Numeric cleanup for features/targets.
    for col in FEATURE_COLS + DELTA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if not available_features:
        raise SystemExit(f"No feature columns found. Expected any of: {FEATURE_COLS}")

    missing_feature_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feature_cols:
        print(f"Warning: Missing feature columns, using subset: {missing_feature_cols}")

    print(f"Using feature columns ({len(available_features)}): {available_features}")
    print(f"Split mode: {args.split_mode}")

    # ---------------------- Classification reproduction ----------------------
    cls_summary_rows: list[dict] = []
    for target_col in ["direction_label", "magnitude_bucket"]:
        if target_col not in df.columns:
            print(f"Skipping classification target (missing): {target_col}")
            continue

        if target_col == "direction_label":
            tier3_name = "Tier3 Active (Non-confounded + Non-flat)"
        else:
            tier3_name = "Tier3 Active (Non-confounded + Non-small)"

        tier_configs = [
            ("Tier1 All Data", df.copy()),
            ("Tier2 Non-confounded", df[df["confound_flag"] == False].copy()),
            (tier3_name, df[get_active_mask(df, target_col)].copy()),
        ]

        for tier_name, tier_df in tier_configs:
            result = run_one_tier_classification(
                tier_df,
                tier_name,
                target_col,
                available_features,
                args.trials,
                args.seed,
                args.test_size,
                args.split_mode,
            )
            if result is not None:
                cls_summary_rows.append(result)

    cls_summary_df = pd.DataFrame(cls_summary_rows)
    if not cls_summary_df.empty:
        cls_summary_df["accuracy_pct"] = cls_summary_df["accuracy"] * 100
        cls_summary_df["macro_f1_pct"] = cls_summary_df["macro_f1"] * 100

        out_cls = Path(args.output_cls)
        out_cls.parent.mkdir(parents=True, exist_ok=True)
        with out_cls.open("w", encoding="utf-8") as f:
            for rec in cls_summary_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Classification results saved to {out_cls}")

    # ---------------------- Regression reproduction -------------------------
    reg_results: list[dict] = []
    if all(col in df.columns for col in DELTA_COLS):
        reg_tiers = [
            ("Tier1 All Data", df.copy()),
            ("Tier2 Non-confounded", df[df["confound_flag"] == False].copy()),
            (
                "Tier3 Active (Non-confounded + Non-flat)",
                df[(df["confound_flag"] == False) & (df["direction_label"] != "flat")].copy(),
            ),
        ]

        for tier_name, tier_df in reg_tiers:
            result = run_regression_tier(
                tier_df,
                tier_name,
                available_features,
                DELTA_COLS,
                HORIZON_NAMES,
                args.trials,
                args.seed,
                args.test_size,
                args.split_mode,
            )
            if result is not None:
                reg_results.append(result)

        out_reg = Path(args.output_reg)
        out_reg.parent.mkdir(parents=True, exist_ok=True)
        with out_reg.open("w", encoding="utf-8") as f:
            for rec in reg_results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Regression results saved to {out_reg}")
    else:
        print(f"Skipping regression: missing one of {DELTA_COLS}")

    print("\n" + "=" * 90)
    print("FINAL SUMMARY BLOCK")
    print("=" * 90)

    if not cls_summary_df.empty:
        print("\n[Classification Summary: direction + magnitude]")
        print(
            cls_summary_df[["target", "tier", "samples", "classes", "accuracy_pct", "macro_f1_pct"]].to_string(
                index=False
            )
        )
    else:
        print("\n[Classification Summary: direction + magnitude]")
        print("No classification results.")

    if reg_results:
        print("\n[Regression Spearman Summary]")
        for result in reg_results:
            print(f"\n{result['tier']} (n={result['samples']})")
            print("-" * 84)
            for horizon in HORIZON_NAMES:
                rho = result["spearman_by_horizon"].get(horizon)
                mae = result["mae_by_horizon"].get(horizon)
                if rho is not None:
                    print(f"  {horizon}: rho={rho:.4f} | MAE={mae:.4f}")
                else:
                    print(f"  {horizon}: rho=N/A | MAE={mae:.4f}")

            if result["spearman_flat"] is not None:
                print(f"  ALL horizons (flatten): rho={result['spearman_flat']:.4f}")
            else:
                print("  ALL horizons (flatten): rho=N/A")
    else:
        print("\n[Regression Spearman Summary]")
        print("No regression results.")


if __name__ == "__main__":
    main()
