"""
Model improvement pipeline for PCSE yield prediction.

This script addresses:
1) CV/Test gap by enforcing district+crop grouped CV
2) Safe MAPE computation
3) Train-inference categorical encoding alignment
4) Rolling feature risk analysis and optional removal

Outputs are written under models/.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "twso_final"
COMBO = ["district_name", "crop_name", "variety_name", "year"]
LEAKY_COLS = ["TWSO", "TAGP", "TWLV", "TWST", "TWRT"]
ROLLING_PATTERNS = ("_roll7", "_roll30")


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("model_improvements")


@dataclass
class EvalResult:
    rmse: float
    mae: float
    r2: float
    mape_raw: float
    mape_safe_100: float


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 100) -> float:
    """Compute MAPE excluding low target values.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        threshold: Include only y_true > threshold.

    Returns:
        Safe MAPE in percent. Returns NaN if no value passes threshold.
    """
    mask = y_true > threshold
    if mask.sum() == 0:
        LOGGER.warning("safe_mape mask is empty for threshold=%s", threshold)
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def raw_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classic MAPE with epsilon for division safety."""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    """Compute regression metrics used in this project."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = raw_mape(y_true, y_pred)
    mape_safe = safe_mape(y_true, y_pred, threshold=100)
    return EvalResult(rmse=rmse, mae=mae, r2=r2, mape_raw=mape, mape_safe_100=mape_safe)


def fit_category_maps(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Build deterministic category-to-code maps for training and inference.

    Args:
        df: Training dataframe containing categorical columns.

    Returns:
        Mapping dict keyed by column name.
    """
    maps: Dict[str, Dict[str, int]] = {}
    for col in ["crop_name", "variety_name", "district_name", "soil_type", "growth_stage"]:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        maps[col] = {v: i for i, v in enumerate(vals)}
    return maps


def apply_category_maps(df: pd.DataFrame, maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Apply precomputed category maps to dataframe and populate *_enc columns.

    Args:
        df: Input dataframe with raw categorical columns.
        maps: Category maps from fit_category_maps.

    Returns:
        Dataframe with encoded columns updated.
    """
    out = df.copy()
    col_map = {
        "crop_name": "crop_name_enc",
        "variety_name": "variety_name_enc",
        "district_name": "district_name_enc",
        "soil_type": "soil_type_enc",
        "growth_stage": "growth_stage_enc",
    }
    for src_col, enc_col in col_map.items():
        out[enc_col] = out[src_col].astype(str).map(maps[src_col]).fillna(-1).astype(int)
    return out


def build_feature_list(df: pd.DataFrame, remove_rolling: bool) -> List[str]:
    """Create model feature list while excluding leakage and identifiers.

    Args:
        df: Input dataframe.
        remove_rolling: Whether to remove rolling columns.

    Returns:
        Ordered feature list.
    """
    drop_cols = {
        TARGET,
        "date",
        "cv_group",
        "district_name",
        "crop_name",
        "variety_name",
        "soil_type",
        "growth_stage",
    }
    drop_cols.update(LEAKY_COLS)
    features = [c for c in df.columns if c not in drop_cols]
    if remove_rolling:
        features = [c for c in features if not c.endswith(ROLLING_PATTERNS)]
    return features


def validate_leakage(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    combo: List[str],
) -> pd.DataFrame:
    """Estimate leakage risk via per-combination feature-target correlation.

    Args:
        df: Input dataframe containing feature and target columns.
        target: Target column name.
        features: Feature columns to inspect.
        combo: Grouping columns representing combination granularity.

    Returns:
        Dataframe with feature, mean_corr, max_corr, risk_flag.
    """
    rows = []
    grouped = df.groupby(combo, dropna=False)
    for feat in features:
        corr_vals = []
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        for _, g in grouped:
            if len(g) < 5:
                continue
            x = g[feat]
            y = g[target]
            if x.nunique() <= 1 or y.nunique() <= 1:
                continue
            corr = x.corr(y)
            if pd.notna(corr):
                corr_vals.append(abs(float(corr)))
        if not corr_vals:
            continue
        mean_corr = float(np.mean(corr_vals))
        max_corr = float(np.max(corr_vals))
        rows.append(
            {
                "feature": feat,
                "mean_corr": round(mean_corr, 4),
                "max_corr": round(max_corr, 4),
                "risk_flag": bool(max_corr > 0.95),
            }
        )
    risk_df = pd.DataFrame(rows)
    if risk_df.empty:
        return pd.DataFrame(columns=["feature", "mean_corr", "max_corr", "risk_flag"])
    return risk_df.sort_values(["risk_flag", "max_corr"], ascending=[False, False])


def evaluate_per_group(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Compute per-group regression metrics.

    Args:
        model: Trained model with predict method.
        X_test: Test feature matrix.
        y_test: Ground-truth values for test set.
        groups: Group labels aligned with y_true.

    Returns:
        Dataframe with group, RMSE, MAE, R2, n sorted by RMSE desc.
    """
    y_pred = model.predict(X_test)
    df_eval = pd.DataFrame({"group": groups.values, "y_true": y_test.values, "y_pred": y_pred})
    rows = []
    for grp, g in df_eval.groupby("group"):
        if len(g) < 2:
            continue
        rows.append(
            {
                "group": grp,
                "RMSE": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
                "MAE": float(mean_absolute_error(g["y_true"], g["y_pred"])),
                "R2": float(r2_score(g["y_true"], g["y_pred"])),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE", ascending=False).reset_index(drop=True)


def check_inference_alignment(train_df: pd.DataFrame, predictor) -> pd.DataFrame:
    """Compare training feature vectors with inference transformation.

    Args:
        train_df: Dataframe containing both raw columns and final feature columns.
        predictor: YieldPredictor instance.

    Returns:
        Dataframe of mismatched rows/features where absolute difference > 0.1.
    """
    sample = train_df.sample(n=min(10, len(train_df)), random_state=42).copy()
    mismatches = []

    for row_idx, row in sample.iterrows():
        row_dict = row.to_dict()
        x_inf = predictor._build_row(row_dict).iloc[0]
        x_train = row[predictor.feature_cols]
        diffs = (x_train - x_inf).abs()
        bad = diffs[diffs > 0.1]
        for feat_name, diff_val in bad.items():
            mismatches.append(
                {
                    "row_index": int(row_idx),
                    "feature": feat_name,
                    "train_value": float(x_train[feat_name]),
                    "inference_value": float(x_inf[feat_name]),
                    "abs_diff": float(diff_val),
                }
            )

    out = pd.DataFrame(mismatches)
    if out.empty:
        return pd.DataFrame(columns=["row_index", "feature", "train_value", "inference_value", "abs_diff"])
    return out.sort_values("abs_diff", ascending=False)


def run_group_cv(
    df: pd.DataFrame,
    features: List[str],
    group_col: str,
    model_params: Dict,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run 5-fold GroupKFold and return fold metrics + aggregate metrics."""
    X = df[features]
    y = df[TARGET].values
    groups = df[group_col].values

    gkf = GroupKFold(n_splits=5)
    fold_rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        model = LGBMRegressor(**model_params)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        fold_rows.append(
            {
                "fold": fold_id,
                "RMSE": float(np.sqrt(mean_squared_error(y[va_idx], pred))),
                "MAE": float(mean_absolute_error(y[va_idx], pred)),
                "R2": float(r2_score(y[va_idx], pred)),
                "n_val": int(len(va_idx)),
            }
        )

    folds = pd.DataFrame(fold_rows)
    agg = {
        "cv_rmse_mean": float(folds["RMSE"].mean()),
        "cv_rmse_std": float(folds["RMSE"].std(ddof=0)),
        "cv_mae_mean": float(folds["MAE"].mean()),
        "cv_r2_mean": float(folds["R2"].mean()),
    }
    return folds, agg


def update_scoreboards(summary_row: Dict[str, float]) -> None:
    """Update models/leaderboard.csv and models/metrics.csv with new columns."""
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    metrics_path = MODEL_DIR / "metrics.csv"

    if leaderboard_path.exists():
        lb = pd.read_csv(leaderboard_path)
    else:
        lb = pd.DataFrame({"Model": ["LightGBM"]})

    if "Model" not in lb.columns:
        lb = lb.reset_index().rename(columns={"index": "Rank"})
        if "Model" not in lb.columns and len(lb) == 1:
            lb["Model"] = "LightGBM"

    for key, value in summary_row.items():
        lb[key] = lb.get(key, np.nan)
        if isinstance(value, (str, bool, np.bool_)):
            lb[key] = lb[key].astype(object)
        lb.loc[lb["Model"] == "LightGBM", key] = value

    if "Rank" not in lb.columns:
        lb.insert(0, "Rank", np.arange(1, len(lb) + 1))

    lb.to_csv(leaderboard_path, index=False)
    LOGGER.info("Updated leaderboard: %s", leaderboard_path)

    if metrics_path.exists():
        mt = pd.read_csv(metrics_path)
    else:
        mt = pd.DataFrame()

    if mt.empty:
        mt = pd.DataFrame([{"model": "LightGBM"}])

    if "model" not in mt.columns:
        mt["model"] = "LightGBM"

    for key, value in summary_row.items():
        mt[key] = mt.get(key, np.nan)
        if isinstance(value, (str, bool, np.bool_)):
            mt[key] = mt[key].astype(object)
        mt.loc[mt["model"].str.lower() == "lightgbm", key] = value

    mt.to_csv(metrics_path, index=False)
    LOGGER.info("Updated metrics: %s", metrics_path)


def main() -> None:
    """Run the full improvement workflow and persist artifacts and reports."""
    data_path = DATA_DIR / "ml_dataset_multiyear.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df_raw = pd.read_parquet(data_path)
    LOGGER.info("Loaded dataset: %s | shape=%s", data_path.name, df_raw.shape)

    df = df_raw[df_raw["DVS"] < 1.0].copy().reset_index(drop=True)
    LOGGER.info("Applied DVS<1.0 filter | shape=%s", df.shape)

    if "split" not in df.columns:
        raise KeyError("Expected 'split' column in parquet file.")

    df["cv_group"] = df["district_name"].astype(str) + "_" + df["crop_name"].astype(str)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    LOGGER.info("Split sizes | train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Split column did not produce non-empty train/test sets.")

    same_district_multi_fold_risk = train_df.groupby("district_name")["cv_group"].nunique().gt(1).sum()
    LOGGER.info("Districts appearing in multiple cv_group labels: %d", int(same_district_multi_fold_risk))

    train_groups = set(train_df["cv_group"].unique())
    test_groups = set(test_df["cv_group"].unique())
    overlap = train_groups.intersection(test_groups)
    LOGGER.info("Group overlap train-test: %d", len(overlap))

    cat_maps = fit_category_maps(train_df)
    train_df = apply_category_maps(train_df, cat_maps)
    test_df = apply_category_maps(test_df, cat_maps)

    model_params = {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
    }

    features_with_roll = build_feature_list(train_df, remove_rolling=False)
    cv_folds_1, cv_agg_1 = run_group_cv(train_df, features_with_roll, "cv_group", model_params)
    LOGGER.info("CV fixed (with rolling) RMSE mean=%.3f std=%.3f", cv_agg_1["cv_rmse_mean"], cv_agg_1["cv_rmse_std"])

    base_model = LGBMRegressor(**model_params)
    base_model.fit(train_df[features_with_roll], train_df[TARGET])
    y_pred_base = base_model.predict(test_df[features_with_roll])
    eval_base = compute_metrics(test_df[TARGET].values, y_pred_base)

    fi = pd.DataFrame(
        {
            "feature": features_with_roll,
            "importance_gain": base_model.booster_.feature_importance(importance_type="gain"),
            "importance_split": base_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    fi["rank"] = np.arange(1, len(fi) + 1)
    top20 = fi.head(20).copy()
    top20.to_csv(MODEL_DIR / "feature_importance_top20.csv", index=False)

    rolling_rank = fi[fi["feature"].str.contains("_roll7|_roll30", regex=True)][["feature", "rank"]]
    rolling_top10 = bool((rolling_rank["rank"] <= 10).any()) if not rolling_rank.empty else False
    rolling_issue_level = "critical" if rolling_top10 else "moderate"

    if rolling_top10:
        LOGGER.warning("Rolling features appear in Top10 importance. Marked critical.")
    else:
        LOGGER.info("Rolling features are not in Top10.")

    # Sorun 4 fix: remove rolling features from final model to avoid fake inference filling.
    final_features = build_feature_list(train_df, remove_rolling=True)
    cv_folds_final, cv_agg_final = run_group_cv(train_df, final_features, "cv_group", model_params)
    cv_folds_final.to_csv(MODEL_DIR / "cv_fold_metrics.csv", index=False)

    final_model = LGBMRegressor(**model_params)
    final_model.fit(train_df[final_features], train_df[TARGET])
    y_pred_test = final_model.predict(test_df[final_features])
    eval_final = compute_metrics(test_df[TARGET].values, y_pred_test)

    ratio = eval_final.rmse / max(cv_agg_final["cv_rmse_mean"], 1e-6)
    LOGGER.info("Final ratio test_rmse/cv_rmse_mean=%.3f", ratio)

    leakage_df = validate_leakage(train_df, TARGET, final_features, COMBO)
    leakage_df.to_csv(MODEL_DIR / "leakage_risk_report.csv", index=False)

    test_df = test_df.copy()
    test_df["pred"] = y_pred_test

    per_crop = []
    for crop, g in test_df.groupby("crop_name"):
        if len(g) < 2:
            continue
        per_crop.append(
            {
                "crop_name": crop,
                "RMSE": float(np.sqrt(mean_squared_error(g[TARGET], g["pred"]))),
                "MAE": float(mean_absolute_error(g[TARGET], g["pred"])),
                "R2": float(r2_score(g[TARGET], g["pred"])),
                "n_samples": int(len(g)),
            }
        )
    per_crop_df = pd.DataFrame(per_crop).sort_values("RMSE", ascending=False).reset_index(drop=True)
    per_crop_df.to_csv(MODEL_DIR / "metrics_per_crop.csv", index=False)
    per_crop_df.head(5).to_csv(MODEL_DIR / "worst_5_crops.csv", index=False)

    per_group_df = evaluate_per_group(final_model, test_df[final_features], test_df[TARGET], test_df["cv_group"])
    per_group_df.to_csv(MODEL_DIR / "metrics_per_group.csv", index=False)

    joblib.dump(final_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(final_model, MODEL_DIR / "lgbm_yield_final.pkl")
    joblib.dump(final_features, MODEL_DIR / "feature_cols.pkl")
    joblib.dump(cat_maps["variety_name"], MODEL_DIR / "variety_map.pkl")
    joblib.dump(cat_maps, MODEL_DIR / "category_maps.pkl")

    align_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    align_df.to_parquet(MODEL_DIR / "alignment_reference.parquet", index=False)

    try:
        from src.inference_pipeline import YieldPredictor  # type: ignore
    except ImportError:
        from inference_pipeline import YieldPredictor  # type: ignore

    predictor = YieldPredictor()
    mismatch_df = check_inference_alignment(align_df, predictor)
    if mismatch_df.empty:
        LOGGER.info("Inference alignment check: PASSED")
    else:
        LOGGER.error("Inference alignment mismatch count: %d", len(mismatch_df))
        mismatch_df.to_csv(MODEL_DIR / "inference_alignment_mismatch.csv", index=False)

    # predict_batch benchmark on 1000 rows
    bench_df = test_df.head(1000).copy()
    bench_cols = [
        "district_name",
        "crop_name",
        "variety_name",
        "AIR_TEMP_mean",
        "AIR_TEMP_min",
        "AIR_TEMP_max",
        "AIR_HUMIDITY_mean",
        "AIR_HUMIDITY_min",
        "AIR_HUMIDITY_max",
        "PRECIP_sum",
        "SOIL_TEMP_0_7_mean",
        "SOIL_MOISTURE_0_7_mean",
        "DVS",
        "LAI",
        "TRA",
        "RD",
        "SM",
        "WWLOW",
        "RFTRA",
        "GDD_cumsum",
        "PRECIP_cumsum",
        "day_of_year",
        "month",
        "week",
        "latitude",
        "longitude",
    ]
    bench_input = bench_df[[c for c in bench_cols if c in bench_df.columns]].rename(
        columns={
            "AIR_TEMP_mean": "air_temp_mean",
            "AIR_TEMP_min": "air_temp_min",
            "AIR_TEMP_max": "air_temp_max",
            "AIR_HUMIDITY_mean": "air_humidity_mean",
            "AIR_HUMIDITY_min": "air_humidity_min",
            "AIR_HUMIDITY_max": "air_humidity_max",
            "PRECIP_sum": "precip_sum",
            "SOIL_TEMP_0_7_mean": "soil_temp_mean",
            "SOIL_MOISTURE_0_7_mean": "soil_moisture_mean",
            "DVS": "dvs",
            "LAI": "lai",
        }
    )

    t0 = time.perf_counter()
    _ = predictor.predict_batch(bench_input)
    batch_1000_sec = time.perf_counter() - t0

    cv_folds_final.to_csv(MODEL_DIR / "cv_fold_metrics.csv", index=False)

    summary_row = {
        "RMSE": round(eval_final.rmse, 2),
        "MAE": round(eval_final.mae, 2),
        "R2": round(eval_final.r2, 4),
        "MAPE%": round(eval_final.mape_raw, 2),
        "MAPE_raw": round(eval_final.mape_raw, 2),
        "MAPE_safe_100": round(eval_final.mape_safe_100, 2),
        "CV_RMSE_mean": round(cv_agg_final["cv_rmse_mean"], 2),
        "CV_RMSE_std": round(cv_agg_final["cv_rmse_std"], 2),
        "CV_MAE_mean": round(cv_agg_final["cv_mae_mean"], 2),
        "CV_R2_mean": round(cv_agg_final["cv_r2_mean"], 4),
        "cv_group_scheme": "district_name + '_' + crop_name",
        "test_cv_ratio": round(ratio, 3),
        "rolling_issue_level": rolling_issue_level,
        "rolling_top10": bool(rolling_top10),
        "alignment_ok": bool(mismatch_df.empty),
        "n_features": int(len(final_features)),
        "batch_1000_sec": round(batch_1000_sec, 4),
    }

    update_scoreboards(summary_row)

    LOGGER.info("Saved model artifacts and reports under %s", MODEL_DIR)
    LOGGER.info("Success criterion ratio<=2.0: %s", ratio <= 2.0)
    LOGGER.info("Success criterion safe MAPE in [20,80]: %s", 20 <= eval_final.mape_safe_100 <= 80)
    LOGGER.info("Success criterion alignment: %s", mismatch_df.empty)
    LOGGER.info("Success criterion batch<1s: %s", batch_1000_sec < 1.0)


if __name__ == "__main__":
    main()
