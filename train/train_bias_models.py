from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train" / "mengxi_boundary_anon_filtered_nc_attach.csv"
OUTPUT_DIR = ROOT / "output" / "bias_training"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


PAIR_SPECS = [
    ("load", 1, 2, "系统负荷"),
    ("renewable_total", 3, 4, "风光总加"),
    ("tie_line", 5, 6, "联络线"),
    ("wind", 7, 8, "风电"),
    ("solar", 9, 10, "光伏"),
    ("hydro", 11, 12, "水电"),
    ("non_market", 13, 14, "非市场化机组"),
]

WEATHER_SPECS = [
    ("u100", -5),
    ("v100", -4),
    ("ghi", -3),
    ("tp", -2),
    ("t2m", -1),
]


@dataclass(frozen=True)
class BiasTask:
    target: str
    display_name: str
    feature_cols: list[str]
    model_name: str = "CatBoost"


TASKS = [
    BiasTask(
        target="solar",
        display_name="光伏偏差",
        feature_cols=[
            "solar_pred",
            "ghi",
            "ghi_delta_3h",
            "ghi_delta_24h",
            "ghi_roll_1h",
            "hour_sin",
            "hour_cos",
        ],
    ),
    BiasTask(
        target="wind",
        display_name="风电偏差",
        feature_cols=[
            "wind_pred",
            "u100",
            "v100",
            "wind_speed",
            "wind_dir",
            "wind_speed_delta_1h",
            "wind_speed_delta_3h",
            "wind_speed_delta_24h",
        ],
    ),
    BiasTask(
        target="hydro",
        display_name="水电偏差",
        feature_cols=[
            "hydro_pred",
            "tp",
            "tp_delta_1h",
            "tp_roll_1h",
            "tp_roll_3h",
            "tp_roll_6h",
        ],
    ),
    BiasTask(
        target="non_market",
        display_name="非市场化机组偏差",
        feature_cols=[
            "non_market_pred",
            "renewable_share_pred",
            "dayofweek",
            "hour_cos",
            "dayofyear_sin",
            "dayofyear_cos",
        ],
    ),
    BiasTask(
        target="tie_line",
        display_name="联络线偏差",
        feature_cols=[
            "tie_line_pred",
            "renewable_share_pred",
            "netload_forecast",
            "t2m",
            "is_evening_peak_hour",
        ],
    ),
    BiasTask(
        target="load",
        display_name="系统负荷偏差",
        feature_cols=[
            "load_pred",
            "hour",
            "is_evening_peak_hour",
            "dayofyear_sin",
            "dayofyear_cos",
            "t2m_delta_1h",
            "t2m_roll_3h",
            "tp",
            "tp_delta_1h",
            "tp_roll_3h",
        ],
    ),
]


def rmse(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def load_dataset() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    rename_map = {"times": "times"}

    for alias, actual_idx, pred_idx, _ in PAIR_SPECS:
        rename_map[raw.columns[actual_idx]] = f"{alias}_actual"
        rename_map[raw.columns[pred_idx]] = f"{alias}_pred"

    for alias, idx in WEATHER_SPECS:
        rename_map[raw.columns[idx]] = alias

    df = raw.rename(columns=rename_map)
    df["times"] = pd.to_datetime(df["times"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["times"].dt.hour
    out["minute"] = out["times"].dt.minute
    out["dayofweek"] = out["times"].dt.dayofweek
    out["month"] = out["times"].dt.month
    out["dayofyear"] = out["times"].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365)
    out["is_evening_peak_hour"] = out["hour"].between(18, 21).astype(int)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["wind_speed"] = np.hypot(out["u100"], out["v100"])
    out["wind_dir"] = np.arctan2(out["v100"], out["u100"])
    out["is_daylight"] = (out["ghi"] > 0).astype(int)
    out["renewable_share_pred"] = safe_divide(out["renewable_total_pred"], out["load_pred"])
    out["renewable_share_pred"] = out["renewable_share_pred"].replace([np.inf, -np.inf], np.nan)
    out["netload_forecast"] = (
        out["load_pred"]
        - out["renewable_total_pred"]
        - out["hydro_pred"]
        - out["non_market_pred"]
    )

    for base_col in ["ghi", "wind_speed", "tp", "t2m"]:
        out[f"{base_col}_delta_1h"] = out[base_col] - out[base_col].shift(4)
        out[f"{base_col}_delta_3h"] = out[base_col] - out[base_col].shift(12)
        out[f"{base_col}_delta_24h"] = out[base_col] - out[base_col].shift(96)

    out["ghi_roll_1h"] = out["ghi"].rolling(4, min_periods=1).mean()
    out["wind_speed_roll_1h"] = out["wind_speed"].rolling(4, min_periods=1).mean()
    out["tp_roll_1h"] = out["tp"].rolling(4, min_periods=1).sum()
    out["tp_roll_3h"] = out["tp"].rolling(12, min_periods=1).sum()
    out["tp_roll_6h"] = out["tp"].rolling(24, min_periods=1).sum()
    out["t2m_roll_3h"] = out["t2m"].rolling(12, min_periods=1).mean()

    for alias, *_ in PAIR_SPECS:
        out[f"{alias}_bias"] = out[f"{alias}_actual"] - out[f"{alias}_pred"]

    return out


def build_validation_mask(df: pd.DataFrame, valid_ratio: float = 0.2) -> pd.Series:
    split_idx = int(len(df) * (1 - valid_ratio))
    split_time = df["times"].sort_values().iloc[split_idx]
    return df["times"] >= split_time


def make_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    )


def evaluate_target(
    y_true: pd.Series,
    baseline_pred: pd.Series,
    corrected_pred: pd.Series,
    target: str,
    display_name: str,
) -> dict[str, float | str]:
    baseline_bias = y_true - baseline_pred
    corrected_bias = y_true - corrected_pred
    baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
    corrected_mae = float(mean_absolute_error(y_true, corrected_pred))
    baseline_rmse = rmse(y_true, baseline_pred)
    corrected_rmse = rmse(y_true, corrected_pred)
    return {
        "target": target,
        "display_name": display_name,
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "mae_improve": baseline_mae - corrected_mae,
        "mae_improve_pct": (baseline_mae - corrected_mae) / baseline_mae * 100 if baseline_mae else 0.0,
        "baseline_rmse": baseline_rmse,
        "corrected_rmse": corrected_rmse,
        "rmse_improve": baseline_rmse - corrected_rmse,
        "rmse_improve_pct": (baseline_rmse - corrected_rmse) / baseline_rmse * 100 if baseline_rmse else 0.0,
        "baseline_bias_mean": float(baseline_bias.mean()),
        "corrected_bias_mean": float(corrected_bias.mean()),
    }


def train_task(df: pd.DataFrame, task: BiasTask, valid_mask: pd.Series) -> tuple[dict, pd.DataFrame]:
    target_col = f"{task.target}_bias"
    base_pred_col = f"{task.target}_pred"
    actual_col = f"{task.target}_actual"

    work = df.dropna(subset=task.feature_cols + [target_col, base_pred_col, actual_col]).copy()
    work["is_valid"] = valid_mask.reindex(work.index).fillna(False).astype(bool)

    train_df = work.loc[~work["is_valid"]].copy()
    valid_df = work.loc[work["is_valid"]].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"{task.target} train/valid split is empty")

    model = make_model()
    model.fit(
        train_df[task.feature_cols],
        train_df[target_col],
        eval_set=(valid_df[task.feature_cols], valid_df[target_col]),
        use_best_model=True,
    )

    valid_df[f"{task.target}_bias_hat"] = model.predict(valid_df[task.feature_cols])
    valid_df[f"{task.target}_corrected"] = valid_df[base_pred_col] + valid_df[f"{task.target}_bias_hat"]

    importance_df = model.get_feature_importance(prettified=True).rename(
        columns={"Feature Id": "feature", "Importances": "importance"}
    )
    importance_df = importance_df.head(10)

    metrics = evaluate_target(
        y_true=valid_df[actual_col],
        baseline_pred=valid_df[base_pred_col],
        corrected_pred=valid_df[f"{task.target}_corrected"],
        target=task.target,
        display_name=task.display_name,
    )
    metrics["train_rows"] = int(len(train_df))
    metrics["valid_rows"] = int(len(valid_df))

    metrics["top_features"] = importance_df[["feature", "importance"]].round(6).to_dict(orient="records")

    prediction_cols = [
        "times",
        actual_col,
        base_pred_col,
        target_col,
        f"{task.target}_bias_hat",
        f"{task.target}_corrected",
    ]
    return metrics, valid_df[prediction_cols].copy()


def merge_prediction_frames(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for frame in prediction_frames.values():
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on="times", how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("times").reset_index(drop=True)


def evaluate_renewable_total_from_components(df: pd.DataFrame, valid_mask: pd.Series) -> tuple[dict, pd.DataFrame]:
    cols = [
        "times",
        "renewable_total_actual",
        "renewable_total_pred",
        "renewable_total_bias",
        "wind_pred",
        "solar_pred",
        "wind_actual",
        "solar_actual",
    ]
    work = df.dropna(subset=cols + ["wind_bias", "solar_bias"]).copy()
    work["is_valid"] = valid_mask.reindex(work.index).fillna(False).astype(bool)
    valid_df = work.loc[work["is_valid"]].copy()
    if valid_df.empty:
        raise ValueError("renewable_total validation set is empty")

    wind_task_df = prediction_frames["wind"].set_index("times")
    solar_task_df = prediction_frames["solar"].set_index("times")
    valid_df = valid_df.set_index("times")
    valid_df["wind_bias_hat"] = wind_task_df.reindex(valid_df.index)["wind_bias_hat"]
    valid_df["solar_bias_hat"] = solar_task_df.reindex(valid_df.index)["solar_bias_hat"]
    valid_df = valid_df.dropna(subset=["wind_bias_hat", "solar_bias_hat"])

    valid_df["renewable_total_bias_hat"] = valid_df["wind_bias_hat"] + valid_df["solar_bias_hat"]
    valid_df["renewable_total_corrected"] = valid_df["renewable_total_pred"] + valid_df["renewable_total_bias_hat"]
    valid_df = valid_df.reset_index()

    metrics = evaluate_target(
        y_true=valid_df["renewable_total_actual"],
        baseline_pred=valid_df["renewable_total_pred"],
        corrected_pred=valid_df["renewable_total_corrected"],
        target="renewable_total",
        display_name="风光总加偏差(风电+光伏汇总)",
    )
    metrics["train_rows"] = None
    metrics["valid_rows"] = int(len(valid_df))
    metrics["top_features"] = [
        {"feature": "wind_bias_hat + solar_bias_hat", "importance": None},
    ]

    output_df = valid_df[
        [
            "times",
            "renewable_total_actual",
            "renewable_total_pred",
            "renewable_total_bias",
            "wind_bias_hat",
            "solar_bias_hat",
            "renewable_total_bias_hat",
            "renewable_total_corrected",
        ]
    ].copy()
    return metrics, output_df


if __name__ == "__main__":
    df = add_engineered_features(add_time_features(load_dataset()))
    valid_mask = build_validation_mask(df, valid_ratio=0.2)

    metrics_rows: list[dict] = []
    global prediction_frames
    prediction_frames: dict[str, pd.DataFrame] = {}

    for task in TASKS:
        metrics, pred_df = train_task(df, task, valid_mask)
        metrics_rows.append(metrics)
        prediction_frames[task.target] = pred_df

    renewable_metrics, renewable_pred_df = evaluate_renewable_total_from_components(df, valid_mask)
    metrics_rows.insert(2, renewable_metrics)
    prediction_frames["renewable_total"] = renewable_pred_df

    metrics_df = pd.DataFrame(
        [
            {k: v for k, v in row.items() if k != "top_features"}
            for row in metrics_rows
        ]
    )

    feature_report = {
        row["target"]: {
            "display_name": row["display_name"],
            "top_features": row["top_features"],
        }
        for row in metrics_rows
    }

    predictions_df = merge_prediction_frames(prediction_frames)

    metrics_path = OUTPUT_DIR / "bias_validation_metrics.csv"
    predictions_path = OUTPUT_DIR / "bias_validation_predictions.csv"
    feature_report_path = OUTPUT_DIR / "bias_feature_importance.json"
    summary_path = OUTPUT_DIR / "bias_training_summary.json"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    feature_report_path.write_text(json.dumps(feature_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "data_path": str(DATA_PATH),
        "validation_ratio": 0.2,
        "validation_start": str(df.loc[valid_mask, "times"].min()),
        "metrics": metrics_df.round(6).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validation start: {summary['validation_start']}")
    print("\nBias correction metrics:")
    print(
        metrics_df[
            [
                "target",
                "display_name",
                "baseline_mae",
                "corrected_mae",
                "mae_improve_pct",
                "baseline_rmse",
                "corrected_rmse",
                "rmse_improve_pct",
                "baseline_bias_mean",
                "corrected_bias_mean",
                "valid_rows",
            ]
        ].round(6).to_string(index=False)
    )
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved feature report to: {feature_report_path}")
