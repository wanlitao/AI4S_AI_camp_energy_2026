from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import catboost
import lightgbm
import xgboost
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FEATURE_PATH = DATA_DIR / "train" / "mengxi_boundary_anon_filtered_nc_attach.csv"
TRAIN_TARGET_PATH = DATA_DIR / "train" / "mengxi_node_price_selected.csv"
TEST_FEATURE_PATH = DATA_DIR / "test" / "test_in_feature_ori_nc_attach.csv"

TIME_COL = "times"
TARGET_COL = "A"

ACTUAL_FORECAST_PAIRS = [
    ("系统负荷实际值", "系统负荷预测值"),
    ("风光总加实际值", "风光总加预测值"),
    ("联络线实际值", "联络线预测值"),
    ("风电实际值", "风电预测值"),
    ("光伏实际值", "光伏预测值"),
    ("水电实际值", "水电预测值"),
    ("非市场化机组实际值", "非市场化机组预测值"),
]

FORECAST_COLS = [pair[1] for pair in ACTUAL_FORECAST_PAIRS] + [
    "u100_空间平均",
    "v100_空间平均",
    "ghi_空间平均",
    "tp_空间平均",
]
RAW_ACTUAL_COLS = [pair[0] for pair in ACTUAL_FORECAST_PAIRS]
WEATHER_COLS = ["u100_空间平均", "v100_空间平均", "ghi_空间平均", "tp_空间平均"]

EXOG_LAG_SOURCE_COLS = FORECAST_COLS
EXOG_LAGS = [1, 4, 96]
EXOG_ROLL_WINDOWS = [4, 16, 96]
PRICE_LAGS = [96, 192, 288, 672, 1344]
PRICE_ROLL_WINDOWS = [96, 192, 672]


@dataclass
class StageConfig:
    stage_name: str
    use_price_history: bool = False
    use_actual_reconstruction: bool = False
    use_spike_model: bool = False
    use_lgb_residual: bool = False
    use_xgb_residual: bool = False
    use_weighted_ensemble: bool = False
    use_gpu: bool = False
    gpu_devices: str = "0"
    random_seed: int = 20260502
    fold_months: list[int] = field(default_factory=lambda: [10, 11, 12])
    spike_quantile: float = 0.95
    extreme_quantile: float = 0.99
    actual_model_preference: str = "catboost"
    base_model_preference: str = "catboost"
    lgb_model_preference: str = "lightgbm"
    xgb_model_preference: str = "xgboost"


@dataclass
class ModelBundle:
    feature_cols: list[str]
    fill_values: dict[str, float]
    base_model: Any
    base_backend: str
    spike_classifier: Any | None = None
    spike_classifier_backend: str | None = None
    spike_model: Any | None = None
    spike_model_backend: str | None = None
    lgb_residual_model: Any | None = None
    lgb_backend: str | None = None
    xgb_residual_model: Any | None = None
    xgb_backend: str | None = None
    residual_feature_cols: list[str] | None = None
    residual_fill_values: dict[str, float] | None = None
    spike_threshold: float | None = None
    extreme_threshold: float | None = None


def log(message: str) -> None:
    print(message, flush=True)


def backend_report() -> dict[str, str]:
    return {
        "catboost": catboost.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
    }


def ensure_paths() -> None:
    for path in [TRAIN_FEATURE_PATH, TRAIN_TARGET_PATH, TEST_FEATURE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"未找到数据文件: {path}")


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.abs() + 1e-6)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    dt = df[TIME_COL].dt
    df["month"] = dt.month
    df["day"] = dt.day
    df["dayofyear"] = dt.dayofyear
    df["dayofweek"] = dt.dayofweek
    df["weekofyear"] = dt.isocalendar().week.astype(int)
    df["hour"] = dt.hour
    df["minute"] = dt.minute
    df["quarter_of_day"] = df["hour"] * 4 + (df["minute"] // 15)
    df["hour_float"] = df["hour"] + df["minute"] / 60.0
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([18, 19, 20]).astype(int)
    df["is_night_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    for col, period in [("quarter_of_day", 96), ("dayofweek", 7), ("month", 12), ("dayofyear", 366)]:
        radians = 2.0 * math.pi * df[col] / period
        df[f"{col}_sin"] = np.sin(radians)
        df[f"{col}_cos"] = np.cos(radians)
    return df


def add_exogenous_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    load_forecast = df["系统负荷预测值"]
    total_renew_forecast = df["风光总加预测值"]
    hydro_forecast = df["水电预测值"]
    non_market_forecast = df["非市场化机组预测值"]
    line_forecast = df["联络线预测值"]
    wind_forecast = df["风电预测值"]
    solar_forecast = df["光伏预测值"]

    df["预测净负荷"] = load_forecast - total_renew_forecast - hydro_forecast - non_market_forecast + line_forecast
    df["预测新能源渗透率"] = safe_ratio(total_renew_forecast, load_forecast)
    df["预测风电占比"] = safe_ratio(wind_forecast, total_renew_forecast + 1e-6)
    df["预测光伏占比"] = safe_ratio(solar_forecast, total_renew_forecast + 1e-6)
    df["预测联络线负荷比"] = safe_ratio(line_forecast, load_forecast)
    df["预测水电负荷比"] = safe_ratio(hydro_forecast, load_forecast)
    df["预测非市场化负荷比"] = safe_ratio(non_market_forecast, load_forecast)
    df["预测风速模长"] = np.sqrt(df["u100_空间平均"] ** 2 + df["v100_空间平均"] ** 2)
    df["预测辐照降水乘积"] = df["ghi_空间平均"] * df["tp_空间平均"]
    return df


def add_exogenous_temporal_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TIME_COL).copy()
    extra_features: dict[str, pd.Series] = {}
    for col in EXOG_LAG_SOURCE_COLS:
        for lag in EXOG_LAGS:
            lag_series = df[col].shift(lag)
            extra_features[f"{col}_lag_{lag}"] = lag_series
            extra_features[f"{col}_diff_{lag}"] = df[col] - lag_series
        for window in EXOG_ROLL_WINDOWS:
            shifted = df[col].shift(1)
            extra_features[f"{col}_roll_mean_{window}"] = shifted.rolling(window, min_periods=max(1, window // 2)).mean()
            extra_features[f"{col}_roll_std_{window}"] = shifted.rolling(window, min_periods=max(1, window // 2)).std()
    return pd.concat([df, pd.DataFrame(extra_features, index=df.index)], axis=1)


def preprocess_exogenous_frames(train_features: pd.DataFrame, test_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part = train_features.copy()
    train_part["dataset_flag"] = "train"
    test_part = test_features.copy()
    test_part["dataset_flag"] = "test"
    combined = pd.concat([train_part, test_part], axis=0, ignore_index=True)
    combined = add_calendar_features(combined)
    combined = add_exogenous_derived_features(combined)
    combined = add_exogenous_temporal_stats(combined)

    train_processed = combined[combined["dataset_flag"] == "train"].drop(columns=["dataset_flag"]).reset_index(drop=True)
    test_processed = combined[combined["dataset_flag"] == "test"].drop(columns=["dataset_flag"]).reset_index(drop=True)
    return train_processed, test_processed


def load_competition_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_paths()
    log("=" * 90)
    log("加载数据集并构建基础外生特征 ...")
    log("=" * 90)
    raw_train_features = pd.read_csv(TRAIN_FEATURE_PATH)
    raw_train_target = pd.read_csv(TRAIN_TARGET_PATH)
    raw_test_features = pd.read_csv(TEST_FEATURE_PATH)

    raw_train_features[TIME_COL] = pd.to_datetime(raw_train_features[TIME_COL])
    raw_train_target[TIME_COL] = pd.to_datetime(raw_train_target[TIME_COL])
    raw_test_features[TIME_COL] = pd.to_datetime(raw_test_features[TIME_COL])

    log(f"训练特征原始 shape: {raw_train_features.shape}")
    log(f"训练标签原始 shape: {raw_train_target.shape}")
    log(f"测试特征原始 shape: {raw_test_features.shape}")

    train_features, test_features = preprocess_exogenous_frames(raw_train_features, raw_test_features)
    train_df = train_features.merge(raw_train_target, on=TIME_COL, how="inner").sort_values(TIME_COL).reset_index(drop=True)
    test_df = test_features.sort_values(TIME_COL).reset_index(drop=True)

    log(f"按时间对齐后的训练集 shape: {train_df.shape}")
    log(f"测试集 shape: {test_df.shape}")
    log(f"训练时间范围: {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()}")
    log(f"测试时间范围: {test_df[TIME_COL].min()} -> {test_df[TIME_COL].max()}")
    return train_df, test_df


def get_base_exogenous_feature_cols(train_df: pd.DataFrame) -> list[str]:
    excluded = {TIME_COL, TARGET_COL, *RAW_ACTUAL_COLS}
    return [col for col in train_df.columns if col not in excluded]


def add_price_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TIME_COL).copy()
    y = df[TARGET_COL]
    extra_features: dict[str, pd.Series] = {}
    for lag in PRICE_LAGS:
        extra_features[f"price_lag_{lag}"] = y.shift(lag)
    for window in PRICE_ROLL_WINDOWS:
        shifted = y.shift(96)
        min_periods = max(24, window // 4)
        extra_features[f"price_roll_mean_{window}"] = shifted.rolling(window, min_periods=min_periods).mean()
        extra_features[f"price_roll_std_{window}"] = shifted.rolling(window, min_periods=min_periods).std()
        extra_features[f"price_roll_min_{window}"] = shifted.rolling(window, min_periods=min_periods).min()
        extra_features[f"price_roll_max_{window}"] = shifted.rolling(window, min_periods=min_periods).max()
    extra_features["price_lag_96_vs_192"] = extra_features["price_lag_96"] - extra_features["price_lag_192"]
    extra_features["price_lag_96_vs_672"] = extra_features["price_lag_96"] - extra_features["price_lag_672"]
    return pd.concat([df, pd.DataFrame(extra_features, index=df.index)], axis=1)


def get_price_history_feature_cols() -> list[str]:
    cols = [f"price_lag_{lag}" for lag in PRICE_LAGS]
    for window in PRICE_ROLL_WINDOWS:
        cols.extend(
            [
                f"price_roll_mean_{window}",
                f"price_roll_std_{window}",
                f"price_roll_min_{window}",
                f"price_roll_max_{window}",
            ]
        )
    cols.extend(["price_lag_96_vs_192", "price_lag_96_vs_672"])
    return cols


def is_likely_gpu_runtime_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    gpu_keywords = ["gpu", "cuda", "opencl", "device", "driver", "boost_compute", "nccl", "cudart"]
    return any(keyword in message for keyword in gpu_keywords)


def build_regressor(
    preference: str,
    seed: int,
    model_role: str,
    use_gpu: bool,
    gpu_devices: str,
) -> tuple[Any, str, Callable[[], Any] | None]:
    if preference == "catboost":
        base_kwargs = {
            "iterations": 1200,
            "depth": 8,
            "learning_rate": 0.03,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": False,
        }
        cpu_builder = lambda: CatBoostRegressor(**base_kwargs)
        if use_gpu:
            gpu_kwargs = {**base_kwargs, "task_type": "GPU", "devices": gpu_devices}
            return CatBoostRegressor(**gpu_kwargs), "catboost-gpu", cpu_builder
        return cpu_builder(), "catboost", None

    if preference == "lightgbm":
        base_kwargs = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": seed,
        }
        cpu_builder = lambda: LGBMRegressor(**base_kwargs)
        if use_gpu:
            gpu_kwargs = {**base_kwargs, "device": "gpu"}
            return LGBMRegressor(**gpu_kwargs), "lightgbm-gpu", cpu_builder
        return cpu_builder(), "lightgbm", None

    if preference == "xgboost":
        base_kwargs = {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "reg:absoluteerror",
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 0,
        }
        cpu_builder = lambda: XGBRegressor(**base_kwargs)
        if use_gpu:
            gpu_kwargs = {**base_kwargs, "tree_method": "hist", "device": "cuda"}
            return XGBRegressor(**gpu_kwargs), "xgboost-gpu", cpu_builder
        return cpu_builder(), "xgboost", None

    raise ValueError(f"未知回归模型偏好: {preference}, role={model_role}")


def build_classifier(
    preference: str,
    seed: int,
    model_role: str,
    use_gpu: bool,
    gpu_devices: str,
) -> tuple[Any, str, Callable[[], Any] | None]:
    if preference == "catboost":
        base_kwargs = {
            "iterations": 900,
            "depth": 7,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": False,
        }
        cpu_builder = lambda: CatBoostClassifier(**base_kwargs)
        if use_gpu:
            gpu_kwargs = {**base_kwargs, "task_type": "GPU", "devices": gpu_devices}
            return CatBoostClassifier(**gpu_kwargs), "catboost-gpu", cpu_builder
        return cpu_builder(), "catboost", None

    raise ValueError(f"未知分类模型偏好: {preference}, role={model_role}")


def fit_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
    backend_label: str | None = None,
    model_role: str | None = None,
    cpu_fallback_builder: Callable[[], Any] | None = None,
) -> tuple[Any, str | None]:
    def _fit(current_model: Any) -> Any:
        if sample_weight is None:
            current_model.fit(X, y)
        else:
            try:
                current_model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                current_model.fit(X, y)
        return current_model

    try:
        return _fit(model), backend_label
    except Exception as exc:
        if cpu_fallback_builder is None or not is_likely_gpu_runtime_error(exc):
            raise
        fallback_backend = backend_label.replace("-gpu", "") if backend_label else backend_label
        role_label = model_role or "model"
        log(f"[fit_model] {role_label} GPU 训练失败，回退 CPU | backend={backend_label} | error={type(exc).__name__}: {exc}")
        fallback_model = cpu_fallback_builder()
        return _fit(fallback_model), fallback_backend


def predict_proba_binary(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    raw = np.asarray(model.decision_function(X))
    return 1.0 / (1.0 + np.exp(-raw))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    q95 = float(np.quantile(y_true, 0.95))
    spike_mask = y_true >= q95
    spike_mae = float(mean_absolute_error(y_true[spike_mask], y_pred[spike_mask])) if spike_mask.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "spike_mae": spike_mae, "spike_threshold_eval": q95}


def compute_classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    result = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, prob)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        result["roc_auc"] = float("nan")
    return result


def pretty_metrics(prefix: str, metrics: dict[str, float]) -> None:
    log(prefix)
    for key, value in metrics.items():
        if isinstance(value, float):
            log(f"  - {key}: {value:.6f}")
        else:
            log(f"  - {key}: {value}")


def compute_price_sample_weights(y: pd.Series, spike_q: float, extreme_q: float) -> np.ndarray:
    q1 = float(y.quantile(spike_q))
    q2 = float(y.quantile(extreme_q))
    weights = np.ones(len(y), dtype=float)
    weights += 2.0 * (y >= q1).astype(float)
    weights += 4.0 * (y >= q2).astype(float)
    return weights


def get_actual_reconstruction_feature_cols(train_df: pd.DataFrame) -> list[str]:
    base_cols = get_base_exogenous_feature_cols(train_df)
    return [col for col in base_cols if not col.startswith("price_")]


def add_reconstructed_actual_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for actual_col, forecast_col in ACTUAL_FORECAST_PAIRS:
        hat_col = f"{actual_col}_hat"
        proxy_col = f"{actual_col}_偏差代理"
        if hat_col in df.columns:
            df[proxy_col] = df[hat_col] - df[forecast_col]

    required_hat_cols = [f"{actual_col}_hat" for actual_col in RAW_ACTUAL_COLS]
    if all(col in df.columns for col in required_hat_cols):
        df["重建净负荷"] = (
            df["系统负荷实际值_hat"]
            - df["风光总加实际值_hat"]
            - df["水电实际值_hat"]
            - df["非市场化机组实际值_hat"]
            + df["联络线实际值_hat"]
        )
        df["重建新能源渗透率"] = safe_ratio(df["风光总加实际值_hat"], df["系统负荷实际值_hat"])
        df["重建风电占比"] = safe_ratio(df["风电实际值_hat"], df["风光总加实际值_hat"] + 1e-6)
        df["重建光伏占比"] = safe_ratio(df["光伏实际值_hat"], df["风光总加实际值_hat"] + 1e-6)
    return df


def fit_actual_reconstruction(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    config: StageConfig,
    fit_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    train_aug = train_df.copy()
    pred_aug = pred_df.copy()
    feature_cols = get_actual_reconstruction_feature_cols(train_df)
    metrics_payload: list[dict[str, Any]] = []
    log(f"[{fit_label}] Stage A: 开始训练 7 个实际值重建模型")
    log(f"[{fit_label}] 实际值重建使用特征数: {len(feature_cols)}")

    X_train = train_aug[feature_cols].copy()
    fill_values = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(fill_values)
    X_pred = pred_aug[feature_cols].copy().fillna(fill_values)

    for actual_col, forecast_col in ACTUAL_FORECAST_PAIRS:
        model, backend, cpu_fallback_builder = build_regressor(
            config.actual_model_preference,
            config.random_seed,
            f"actual_reconstruction::{actual_col}",
            config.use_gpu,
            config.gpu_devices,
        )
        y_train = train_aug[actual_col]
        model, backend = fit_model(
            model,
            X_train,
            y_train,
            backend_label=backend,
            model_role=f"actual_reconstruction::{actual_col}",
            cpu_fallback_builder=cpu_fallback_builder,
        )

        train_hat = np.asarray(model.predict(X_train), dtype=float)
        pred_hat = np.asarray(model.predict(X_pred), dtype=float)
        train_aug[f"{actual_col}_hat"] = train_hat
        pred_aug[f"{actual_col}_hat"] = pred_hat

        train_mae = float(mean_absolute_error(train_aug[actual_col], train_hat))
        record = {
            "target": actual_col,
            "backend": backend,
            "train_mae": train_mae,
        }
        message = f"[{fit_label}] 实际值重建 {actual_col} | backend={backend} | train_mae={train_mae:.6f}"
        if actual_col in pred_aug.columns:
            pred_actual = pd.to_numeric(pred_aug[actual_col], errors="coerce")
            valid_mask = pred_actual.notna().to_numpy()
            if valid_mask.any():
                val_mae = float(mean_absolute_error(pred_actual.loc[valid_mask], pred_hat[valid_mask]))
                record["pred_mae"] = val_mae
                message += f" | pred_mae={val_mae:.6f}"
            else:
                message += " | pred_mae=skip(no_actuals)"
        log(message)
        metrics_payload.append(record)

    train_aug = add_reconstructed_actual_features(train_aug)
    pred_aug = add_reconstructed_actual_features(pred_aug)
    return train_aug, pred_aug, metrics_payload


def build_price_feature_cols(train_df: pd.DataFrame, config: StageConfig, exog_feature_cols: list[str]) -> list[str]:
    feature_cols = list(exog_feature_cols)
    if config.use_actual_reconstruction:
        hat_cols = [f"{actual_col}_hat" for actual_col in RAW_ACTUAL_COLS]
        proxy_cols = [f"{actual_col}_偏差代理" for actual_col in RAW_ACTUAL_COLS]
        feature_cols.extend([col for col in hat_cols + proxy_cols if col in train_df.columns])
        for derived_col in ["重建净负荷", "重建新能源渗透率", "重建风电占比", "重建光伏占比"]:
            if derived_col in train_df.columns:
                feature_cols.append(derived_col)
    if config.use_price_history:
        feature_cols.extend(get_price_history_feature_cols())
    seen: set[str] = set()
    deduped: list[str] = []
    for col in feature_cols:
        if col not in seen:
            seen.add(col)
            deduped.append(col)
    return deduped


def prepare_training_matrix(df: pd.DataFrame, feature_cols: list[str], fit_label: str) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    train_matrix = df.sort_values(TIME_COL).copy()
    before_rows = len(train_matrix)
    train_matrix = train_matrix.dropna(subset=[TARGET_COL]).copy()
    train_matrix = train_matrix.dropna(subset=feature_cols, how="any").copy()
    fill_values = train_matrix[feature_cols].median(numeric_only=True).to_dict()
    train_matrix[feature_cols] = train_matrix[feature_cols].fillna(fill_values)
    dropped_rows = before_rows - len(train_matrix)
    log(f"[{fit_label}] 价格训练样本: {len(train_matrix)} 行, 丢弃 {dropped_rows} 行")
    X = train_matrix[feature_cols].copy()
    y = train_matrix[TARGET_COL].copy()
    return X, y, fill_values


def build_residual_feature_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    stage4_pred: np.ndarray,
    spike_prob: np.ndarray | None,
) -> pd.DataFrame:
    residual_df = df[feature_cols].copy()
    residual_df["stage4_pred"] = stage4_pred
    residual_df["stage4_abs_proxy"] = np.abs(stage4_pred - residual_df.get("price_lag_96", pd.Series(np.zeros(len(df)), index=df.index)))
    residual_df["stage4_to_load_ratio"] = stage4_pred / (df["系统负荷预测值"].abs() + 1e-6)
    residual_df["spike_prob"] = 0.0 if spike_prob is None else spike_prob
    return residual_df


def search_best_weights(oof_df: pd.DataFrame, candidate_cols: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    best_weights: dict[str, float] = {}
    best_pred: np.ndarray | None = None
    best_mae = float("inf")

    step = 0.05
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    if len(candidate_cols) == 2:
        for w0 in grid:
            w1 = 1.0 - w0
            weights = [w0, w1]
            pred = sum(weights[idx] * oof_df[col].to_numpy(float) for idx, col in enumerate(candidate_cols))
            mae = mean_absolute_error(oof_df[TARGET_COL], pred)
            if mae < best_mae:
                best_mae = float(mae)
                best_pred = pred
                best_weights = {candidate_cols[0]: w0, candidate_cols[1]: w1}
    elif len(candidate_cols) == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                weights = [w0, w1, w2]
                pred = sum(weights[idx] * oof_df[col].to_numpy(float) for idx, col in enumerate(candidate_cols))
                mae = mean_absolute_error(oof_df[TARGET_COL], pred)
                if mae < best_mae:
                    best_mae = float(mae)
                    best_pred = pred
                    best_weights = {candidate_cols[i]: float(weights[i]) for i in range(3)}
    else:
        raise ValueError("权重搜索当前仅支持 2 或 3 个候选模型")

    assert best_pred is not None
    metrics = compute_regression_metrics(oof_df[TARGET_COL].to_numpy(float), best_pred)
    return best_weights, metrics


def build_validation_folds(train_df: pd.DataFrame, fold_months: list[int]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    train_df = train_df.sort_values(TIME_COL)
    for month in fold_months:
        val_mask = train_df[TIME_COL].dt.month == month
        train_mask = train_df[TIME_COL].dt.month < month
        if not val_mask.any() or not train_mask.any():
            continue
        val_start = train_df.loc[val_mask, TIME_COL].min()
        val_end = train_df.loc[val_mask, TIME_COL].max()
        folds.append(
            {
                "name": f"2025-{month:02d}",
                "train_mask": train_mask,
                "val_mask": val_mask,
                "val_start": val_start,
                "val_end": val_end,
            }
        )
    return folds


def fit_stage_models(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    config: StageConfig,
    fit_label: str,
    base_exog_feature_cols: list[str],
) -> tuple[pd.DataFrame, ModelBundle, dict[str, Any]]:
    log("-" * 90)
    log(f"[{fit_label}] 开始训练阶段模型: {config.stage_name}")
    log(f"[{fit_label}] 训练样本区间: {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()}, 行数={len(train_df)}")
    log(f"[{fit_label}] 预测样本区间: {pred_df[TIME_COL].min()} -> {pred_df[TIME_COL].max()}, 行数={len(pred_df)}")
    log(f"[{fit_label}] GPU 优先: {'ON' if config.use_gpu else 'OFF'} | devices={config.gpu_devices}")

    train_stage = train_df.sort_values(TIME_COL).copy()
    pred_stage = pred_df.sort_values(TIME_COL).copy()
    actual_metrics: list[dict[str, Any]] = []

    if config.use_actual_reconstruction:
        train_stage, pred_stage, actual_metrics = fit_actual_reconstruction(train_stage, pred_stage, config, fit_label)

    if config.use_price_history:
        train_stage = add_price_history_features(train_stage)

    feature_cols = build_price_feature_cols(train_stage, config, base_exog_feature_cols)
    X_train, y_train, fill_values = prepare_training_matrix(train_stage, feature_cols, fit_label)

    log(f"[{fit_label}] 价格主模型特征数: {len(feature_cols)}")
    log(f"[{fit_label}] 主模型 backend 偏好: {config.base_model_preference}")
    base_model, base_backend, base_cpu_fallback_builder = build_regressor(
        config.base_model_preference,
        config.random_seed,
        "base_price",
        config.use_gpu,
        config.gpu_devices,
    )

    sample_weight = None
    if config.use_spike_model:
        sample_weight = compute_price_sample_weights(y_train, config.spike_quantile, config.extreme_quantile)
        log(f"[{fit_label}] 启用 spike 加权训练, 权重均值={sample_weight.mean():.4f}, 最大值={sample_weight.max():.4f}")

    base_model, base_backend = fit_model(
        base_model,
        X_train,
        y_train,
        sample_weight=sample_weight,
        backend_label=base_backend,
        model_role="base_price",
        cpu_fallback_builder=base_cpu_fallback_builder,
    )
    base_train_pred = np.asarray(base_model.predict(X_train), dtype=float)
    base_train_metrics = compute_regression_metrics(y_train.to_numpy(float), base_train_pred)
    pretty_metrics(f"[{fit_label}] 主模型训练集指标 | backend={base_backend}", base_train_metrics)

    spike_classifier = None
    spike_classifier_backend = None
    spike_model = None
    spike_model_backend = None
    spike_threshold = None
    extreme_threshold = None
    train_spike_prob = None
    train_stage4_pred = base_train_pred.copy()

    if config.use_spike_model:
        spike_threshold = float(y_train.quantile(config.spike_quantile))
        extreme_threshold = float(y_train.quantile(config.extreme_quantile))
        spike_label = (y_train >= spike_threshold).astype(int)
        log(
            f"[{fit_label}] Spike 阈值: q{config.spike_quantile:.2f}={spike_threshold:.6f}, "
            f"extreme q{config.extreme_quantile:.2f}={extreme_threshold:.6f}, 正样本占比={spike_label.mean():.4f}"
        )

        spike_classifier, spike_classifier_backend, spike_classifier_cpu_fallback_builder = build_classifier(
            config.base_model_preference,
            config.random_seed,
            "spike_classifier",
            config.use_gpu,
            config.gpu_devices,
        )
        clf_sample_weight = np.where(spike_label == 1, 4.0, 1.0)
        spike_classifier, spike_classifier_backend = fit_model(
            spike_classifier,
            X_train,
            spike_label,
            sample_weight=clf_sample_weight,
            backend_label=spike_classifier_backend,
            model_role="spike_classifier",
            cpu_fallback_builder=spike_classifier_cpu_fallback_builder,
        )
        train_spike_prob = predict_proba_binary(spike_classifier, X_train)
        clf_metrics = compute_classification_metrics(spike_label.to_numpy(int), train_spike_prob)
        pretty_metrics(f"[{fit_label}] Spike 分类器训练集指标 | backend={spike_classifier_backend}", clf_metrics)

        spike_mask = y_train >= float(y_train.quantile(0.90))
        spike_train_X = X_train.loc[spike_mask].copy()
        spike_train_y = y_train.loc[spike_mask].copy()
        spike_weights = compute_price_sample_weights(spike_train_y, 0.75, 0.90)
        spike_model, spike_model_backend, spike_model_cpu_fallback_builder = build_regressor(
            config.base_model_preference,
            config.random_seed,
            "spike_regressor",
            config.use_gpu,
            config.gpu_devices,
        )
        spike_model, spike_model_backend = fit_model(
            spike_model,
            spike_train_X,
            spike_train_y,
            sample_weight=spike_weights,
            backend_label=spike_model_backend,
            model_role="spike_regressor",
            cpu_fallback_builder=spike_model_cpu_fallback_builder,
        )
        spike_train_pred = np.asarray(spike_model.predict(X_train), dtype=float)
        train_stage4_pred = (1.0 - train_spike_prob) * base_train_pred + train_spike_prob * spike_train_pred
        spike_train_metrics = compute_regression_metrics(y_train.to_numpy(float), train_stage4_pred)
        pretty_metrics(f"[{fit_label}] Spike 融合训练集指标 | spike_backend={spike_model_backend}", spike_train_metrics)
    else:
        train_spike_prob = np.zeros(len(X_train), dtype=float)

    residual_feature_cols = None
    residual_fill_values = None
    lgb_residual_model = None
    lgb_backend = None
    xgb_residual_model = None
    xgb_backend = None

    if config.use_lgb_residual or config.use_xgb_residual:
        residual_train_X = build_residual_feature_frame(X_train.join(train_stage.loc[X_train.index, [TIME_COL]]), feature_cols, train_stage4_pred, train_spike_prob)
        residual_feature_cols = residual_train_X.columns.tolist()
        residual_fill_values = residual_train_X.median(numeric_only=True).to_dict()
        residual_train_X = residual_train_X.fillna(residual_fill_values)
        residual_target = y_train.to_numpy(float) - train_stage4_pred
        log(f"[{fit_label}] 残差模型训练目标统计: mean={residual_target.mean():.6f}, std={residual_target.std():.6f}")

        if config.use_lgb_residual:
            lgb_residual_model, lgb_backend, lgb_cpu_fallback_builder = build_regressor(
                config.lgb_model_preference,
                config.random_seed,
                "lgb_residual",
                config.use_gpu,
                config.gpu_devices,
            )
            lgb_residual_model, lgb_backend = fit_model(
                lgb_residual_model,
                residual_train_X,
                pd.Series(residual_target, index=residual_train_X.index),
                backend_label=lgb_backend,
                model_role="lgb_residual",
                cpu_fallback_builder=lgb_cpu_fallback_builder,
            )
            train_lgb_pred = np.asarray(lgb_residual_model.predict(residual_train_X), dtype=float)
            metrics = compute_regression_metrics(y_train.to_numpy(float), train_stage4_pred + train_lgb_pred)
            pretty_metrics(f"[{fit_label}] LGB 残差修正训练集指标 | backend={lgb_backend}", metrics)

        if config.use_xgb_residual:
            xgb_residual_model, xgb_backend, xgb_cpu_fallback_builder = build_regressor(
                config.xgb_model_preference,
                config.random_seed,
                "xgb_residual",
                config.use_gpu,
                config.gpu_devices,
            )
            xgb_residual_model, xgb_backend = fit_model(
                xgb_residual_model,
                residual_train_X,
                pd.Series(residual_target, index=residual_train_X.index),
                backend_label=xgb_backend,
                model_role="xgb_residual",
                cpu_fallback_builder=xgb_cpu_fallback_builder,
            )
            train_xgb_pred = np.asarray(xgb_residual_model.predict(residual_train_X), dtype=float)
            metrics = compute_regression_metrics(y_train.to_numpy(float), train_stage4_pred + train_xgb_pred)
            pretty_metrics(f"[{fit_label}] XGB 残差修正训练集指标 | backend={xgb_backend}", metrics)

    bundle = ModelBundle(
        feature_cols=feature_cols,
        fill_values=fill_values,
        base_model=base_model,
        base_backend=base_backend,
        spike_classifier=spike_classifier,
        spike_classifier_backend=spike_classifier_backend,
        spike_model=spike_model,
        spike_model_backend=spike_model_backend,
        lgb_residual_model=lgb_residual_model,
        lgb_backend=lgb_backend,
        xgb_residual_model=xgb_residual_model,
        xgb_backend=xgb_backend,
        residual_feature_cols=residual_feature_cols,
        residual_fill_values=residual_fill_values,
        spike_threshold=spike_threshold,
        extreme_threshold=extreme_threshold,
    )

    artifacts = {
        "train_augmented": train_stage,
        "pred_augmented": pred_stage,
        "actual_reconstruction_metrics": actual_metrics,
        "base_train_metrics": base_train_metrics,
    }
    return pred_stage, bundle, artifacts


def build_future_price_history_features(history_df: pd.DataFrame, current_times: pd.DataFrame) -> pd.DataFrame:
    working = pd.concat(
        [
            history_df[[TIME_COL, TARGET_COL]].copy(),
            current_times[[TIME_COL]].copy().assign(**{TARGET_COL: np.nan}),
        ],
        axis=0,
        ignore_index=True,
    ).sort_values(TIME_COL)
    working = add_price_history_features(working)
    return working.loc[working[TIME_COL].isin(current_times[TIME_COL]), [TIME_COL] + get_price_history_feature_cols()].copy()


def sequential_predict(
    history_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    bundle: ModelBundle,
    config: StageConfig,
    fit_label: str,
) -> pd.DataFrame:
    history_work = history_df[[TIME_COL, TARGET_COL]].sort_values(TIME_COL).copy()
    pred_work = pred_df.sort_values(TIME_COL).copy()
    output_frames: list[pd.DataFrame] = []

    for day, day_df in pred_work.groupby(pred_work[TIME_COL].dt.date):
        day_frame = day_df.copy()
        if config.use_price_history:
            history_features = build_future_price_history_features(history_work, day_frame[[TIME_COL]])
            day_frame = day_frame.merge(history_features, on=TIME_COL, how="left")

        X_day = day_frame[bundle.feature_cols].copy().fillna(bundle.fill_values)
        base_pred = np.asarray(bundle.base_model.predict(X_day), dtype=float)
        stage4_pred = base_pred.copy()
        spike_prob = np.zeros(len(day_frame), dtype=float)

        if bundle.spike_classifier is not None and bundle.spike_model is not None:
            spike_prob = predict_proba_binary(bundle.spike_classifier, X_day)
            spike_pred = np.asarray(bundle.spike_model.predict(X_day), dtype=float)
            stage4_pred = (1.0 - spike_prob) * base_pred + spike_prob * spike_pred

        day_out = day_frame[[TIME_COL]].copy()
        if TARGET_COL in day_frame.columns:
            day_out[TARGET_COL] = day_frame[TARGET_COL].to_numpy(float)
        day_out["base_pred"] = base_pred
        day_out["spike_prob"] = spike_prob
        day_out["stage4_pred"] = stage4_pred
        day_out["stage5_pred"] = stage4_pred
        day_out["stage6_pred"] = stage4_pred

        if bundle.lgb_residual_model is not None:
            residual_day_X = build_residual_feature_frame(day_frame, bundle.feature_cols, stage4_pred, spike_prob)
            residual_day_X = residual_day_X[bundle.residual_feature_cols].copy().fillna(bundle.residual_fill_values)
            lgb_residual_pred = np.asarray(bundle.lgb_residual_model.predict(residual_day_X), dtype=float)
            day_out["lgb_residual_pred"] = lgb_residual_pred
            day_out["stage5_pred"] = stage4_pred + lgb_residual_pred
        else:
            day_out["lgb_residual_pred"] = 0.0

        if bundle.xgb_residual_model is not None:
            residual_day_X = build_residual_feature_frame(day_frame, bundle.feature_cols, stage4_pred, spike_prob)
            residual_day_X = residual_day_X[bundle.residual_feature_cols].copy().fillna(bundle.residual_fill_values)
            xgb_residual_pred = np.asarray(bundle.xgb_residual_model.predict(residual_day_X), dtype=float)
            day_out["xgb_residual_pred"] = xgb_residual_pred
            day_out["stage6_pred"] = stage4_pred + xgb_residual_pred
        else:
            day_out["xgb_residual_pred"] = 0.0

        history_append = day_out[[TIME_COL, "stage4_pred"]].rename(columns={"stage4_pred": TARGET_COL})
        history_work = pd.concat([history_work, history_append], axis=0, ignore_index=True).sort_values(TIME_COL)

        log(
            f"[{fit_label}] 顺序预测 {day} 完成 | 样本={len(day_out)} | "
            f"base_mean={day_out['base_pred'].mean():.6f} | final_hint_mean={day_out['stage6_pred'].mean():.6f}"
        )
        output_frames.append(day_out)

    result = pd.concat(output_frames, axis=0, ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    return result


def determine_final_column(config: StageConfig) -> str:
    if config.use_weighted_ensemble:
        return "ensemble_pred"
    if config.use_xgb_residual:
        return "stage6_pred"
    if config.use_lgb_residual:
        return "stage5_pred"
    return "stage4_pred"


def fit_and_predict_stage(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    config: StageConfig,
    fit_label: str,
    base_exog_feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_aug, bundle, artifacts = fit_stage_models(train_df, pred_df, config, fit_label, base_exog_feature_cols)
    prediction_df = sequential_predict(train_df[[TIME_COL, TARGET_COL]], pred_aug, bundle, config, fit_label)
    detail = {"bundle": bundle, "artifacts": artifacts}
    return prediction_df, detail


def save_prediction_csv(script_path: str, prediction_df: pd.DataFrame, final_col: str) -> Path:
    script_stem = Path(script_path).stem
    output_path = OUTPUT_DIR / f"{script_stem}_output.csv"
    submission = prediction_df[[TIME_COL, final_col]].copy()
    submission.columns = [TIME_COL, TARGET_COL]
    submission[TIME_COL] = pd.to_datetime(submission[TIME_COL]).dt.strftime("%Y-%m-%d %H:%M:%S")
    submission.to_csv(output_path, index=False)
    log(f"测试集预测已保存: {output_path}")
    return output_path


def save_metrics_json(script_path: str, payload: dict[str, Any]) -> Path:
    script_stem = Path(script_path).stem
    output_path = OUTPUT_DIR / f"{script_stem}_metrics.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)
    log(f"验证指标已保存: {output_path}")
    return output_path


def run_pipeline(config: StageConfig, script_path: str) -> None:
    log("#" * 90)
    log(f"启动训练脚本: {Path(script_path).name}")
    log(f"阶段名称: {config.stage_name}")
    log("后端依赖状态:")
    for name, status in backend_report().items():
        log(f"  - {name}: {status}")
    log("#" * 90)

    train_df, test_df = load_competition_data()
    base_exog_feature_cols = get_base_exogenous_feature_cols(train_df)
    log(f"基础外生特征数: {len(base_exog_feature_cols)}")

    folds = build_validation_folds(train_df, config.fold_months)
    fold_payloads: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    final_col = determine_final_column(config)

    for fold_idx, fold in enumerate(folds, start=1):
        log("=" * 90)
        log(f"开始第 {fold_idx}/{len(folds)} 个时序验证折: {fold['name']}")
        train_fold = train_df.loc[fold["train_mask"]].copy().sort_values(TIME_COL)
        val_fold = train_df.loc[fold["val_mask"]].copy().sort_values(TIME_COL)
        log(f"训练折行数: {len(train_fold)}, 验证折行数: {len(val_fold)}")
        log(f"验证区间: {fold['val_start']} -> {fold['val_end']}")

        val_pred_df, detail = fit_and_predict_stage(
            train_fold,
            val_fold,
            config,
            fit_label=f"fold_{fold['name']}",
            base_exog_feature_cols=base_exog_feature_cols,
        )

        component_metrics: dict[str, dict[str, float]] = {}
        for candidate in ["stage4_pred", "stage5_pred", "stage6_pred"]:
            if candidate in val_pred_df.columns:
                metrics = compute_regression_metrics(val_pred_df[TARGET_COL].to_numpy(float), val_pred_df[candidate].to_numpy(float))
                component_metrics[candidate] = metrics
                pretty_metrics(f"[fold_{fold['name']}] 验证指标 - {candidate}", metrics)

        fold_record = {
            "fold_name": fold["name"],
            "component_metrics": component_metrics,
            "actual_reconstruction_metrics": detail["artifacts"]["actual_reconstruction_metrics"],
        }
        fold_payloads.append(fold_record)
        oof_frames.append(val_pred_df)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    oof_metrics: dict[str, Any] = {}
    for candidate in ["stage4_pred", "stage5_pred", "stage6_pred"]:
        if candidate in oof_df.columns:
            oof_metrics[candidate] = compute_regression_metrics(oof_df[TARGET_COL].to_numpy(float), oof_df[candidate].to_numpy(float))
            pretty_metrics(f"[OOF] 汇总验证指标 - {candidate}", oof_metrics[candidate])

    ensemble_weights: dict[str, float] | None = None
    ensemble_metrics: dict[str, float] | None = None
    if config.use_weighted_ensemble:
        candidate_cols = ["stage4_pred"]
        if config.use_lgb_residual:
            candidate_cols.append("stage5_pred")
        if config.use_xgb_residual:
            candidate_cols.append("stage6_pred")
        ensemble_weights, ensemble_metrics = search_best_weights(oof_df, candidate_cols)
        oof_df["ensemble_pred"] = sum(weight * oof_df[col] for col, weight in ensemble_weights.items())
        pretty_metrics("[OOF] 加权集成验证指标", ensemble_metrics)
        log(f"[OOF] 最优集成权重: {ensemble_weights}")
        final_col = "ensemble_pred"

    log("=" * 90)
    log("开始使用全量 2025 数据训练，并对 2026-01/02 测试集输出预测")
    full_test_pred_df, full_detail = fit_and_predict_stage(
        train_df,
        test_df,
        config,
        fit_label="full_train",
        base_exog_feature_cols=base_exog_feature_cols,
    )

    if config.use_weighted_ensemble and ensemble_weights is not None:
        full_test_pred_df["ensemble_pred"] = sum(weight * full_test_pred_df[col] for col, weight in ensemble_weights.items())

    output_path = save_prediction_csv(script_path, full_test_pred_df, final_col)
    metrics_payload = {
        "script": Path(script_path).name,
        "config": asdict(config),
        "backend_report": backend_report(),
        "fold_metrics": fold_payloads,
        "oof_metrics": oof_metrics,
        "ensemble_weights": ensemble_weights,
        "ensemble_metrics": ensemble_metrics,
        "final_output_path": str(output_path),
        "final_prediction_column": final_col,
        "full_train_actual_reconstruction_metrics": full_detail["artifacts"]["actual_reconstruction_metrics"],
        "full_train_base_metrics": full_detail["artifacts"]["base_train_metrics"],
    }
    save_metrics_json(script_path, metrics_payload)
    log("#" * 90)
    log(f"脚本运行完成: {Path(script_path).name}")
    log(f"最终输出列: {final_col}")
    log("#" * 90)
