from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / "output" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


TRAIN_FEATURE_PATH = PROJECT_ROOT / "data" / "train" / "mengxi_boundary_anon_filtered_nc_attach.csv"
TRAIN_LABEL_PATH = PROJECT_ROOT / "data" / "train" / "mengxi_node_price_selected.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

TARGET_COL = "A"
TIME_COL = "times"

CV_ROUNDS = [
    {"train_end": "2025-09", "valid_month": "2025-10"},
    {"train_end": "2025-10", "valid_month": "2025-11"},
    {"train_end": "2025-11", "valid_month": "2025-12"},
]

BOUNDARY_FORECAST_COLS = [
    "系统负荷预测值",
    "风光总加预测值",
    "联络线预测值",
    "风电预测值",
    "光伏预测值",
    "水电预测值",
    "非市场化机组预测值",
]

TIME_FEATURE_COLS = [
    "month",
    "dayofweek",
    "is_weekend",
    "is_morning_peak_hour",
    "is_evening_peak_hour",
]

DERIVED_FORECAST_FEATURE_COLS = [
    "renew_share_forecast",
    "net_load_forecast",
    "ghi_rolling_mean_1h",
    "wind_speed_rolling_mean_1h",
    "t2m_rolling_mean_3h",
]

PRICE_FEATURE_COLS = [
    "price_lag_4",
    "price_lag_8",
    "price_lag_96",
    "price_rolling_mean_4",
    "price_rolling_mean_8",
]

FEATURE_COLS = BOUNDARY_FORECAST_COLS + TIME_FEATURE_COLS + DERIVED_FORECAST_FEATURE_COLS + PRICE_FEATURE_COLS


def load_training_data() -> pd.DataFrame:
    feature_df = pd.read_csv(TRAIN_FEATURE_PATH)
    label_df = pd.read_csv(TRAIN_LABEL_PATH)

    df = feature_df.merge(label_df, on=TIME_COL, how="inner")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df[TIME_COL].dt.hour
    df["month"] = df[TIME_COL].dt.month
    df["dayofweek"] = df[TIME_COL].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_morning_peak_hour"] = hour.between(8, 11).astype(int)
    df["is_evening_peak_hour"] = hour.between(18, 21).astype(int)
    return df


def add_boundary_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    load_forecast = df["系统负荷预测值"]
    renew_forecast = df["风光总加预测值"]
    hydro_forecast = df["水电预测值"]
    non_market_forecast = df["非市场化机组预测值"]

    df["renew_share_forecast"] = renew_forecast / load_forecast.replace(0, np.nan)
    df["net_load_forecast"] = load_forecast - renew_forecast - hydro_forecast - non_market_forecast

    wind_speed = np.sqrt(df["u100_空间平均"] ** 2 + df["v100_空间平均"] ** 2)
    df["ghi_rolling_mean_1h"] = df["ghi_空间平均"].rolling(window=4, min_periods=4).mean()
    df["wind_speed_rolling_mean_1h"] = wind_speed.rolling(window=4, min_periods=4).mean()
    df["t2m_rolling_mean_3h"] = df["t2m_空间平均"].rolling(window=12, min_periods=12).mean()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    shifted_4 = df[TARGET_COL].shift(4)
    df["price_lag_4"] = shifted_4
    df["price_lag_8"] = df[TARGET_COL].shift(8)
    shifted_96 = df[TARGET_COL].shift(96)
    df["price_lag_96"] = shifted_96
    df["price_rolling_mean_4"] = shifted_4.rolling(window=4).mean()
    df["price_rolling_mean_8"] = shifted_4.rolling(window=8).mean()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_boundary_derived_features(df)
    df = add_price_features(df)
    return df


def select_feature_ready_rows(df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    subset = FEATURE_COLS.copy()
    if require_target:
        subset.append(TARGET_COL)
    return df.dropna(subset=subset).reset_index(drop=True)


def get_round_data(
    feature_df: pd.DataFrame, raw_df: pd.DataFrame, train_end: str, valid_month: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_period = pd.Period(train_end, freq="M")
    valid_period = pd.Period(valid_month, freq="M")

    feature_months = feature_df[TIME_COL].dt.to_period("M")
    train_df = feature_df.loc[feature_months <= train_end_period].copy()
    valid_feature_df = feature_df.loc[feature_months == valid_period].copy()

    if train_df.empty:
        raise ValueError(f"empty training set for round train_end={train_end}")
    if valid_feature_df.empty:
        raise ValueError(f"empty validation set for round valid_month={valid_month}")

    valid_times = set(valid_feature_df[TIME_COL])
    history_df = raw_df.loc[raw_df[TIME_COL] < valid_feature_df[TIME_COL].min()].copy()
    valid_raw_df = raw_df.loc[raw_df[TIME_COL].isin(valid_times)].copy()

    return train_df, history_df, valid_raw_df


def recursive_predict_by_hour(
    model: CatBoostRegressor, history_df: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    history_df = history_df.sort_values(TIME_COL).reset_index(drop=True).copy()
    future_df = future_df.sort_values(TIME_COL).reset_index(drop=True).copy()
    future_df["hour_bucket"] = future_df[TIME_COL].dt.floor("h")

    prediction_chunks: list[pd.DataFrame] = []

    for current_hour, hour_future_df in future_df.groupby("hour_bucket", sort=True):
        hour_input_df = hour_future_df.drop(columns=[TARGET_COL], errors="ignore").copy()
        hour_input_df[TARGET_COL] = np.nan

        combined_df = pd.concat([history_df, hour_input_df], ignore_index=True)
        combined_feature_df = build_features(combined_df)

        hour_feature_df = combined_feature_df[combined_feature_df[TIME_COL].isin(hour_future_df[TIME_COL])].copy()
        hour_feature_df = hour_feature_df.sort_values(TIME_COL).reset_index(drop=True)

        if len(hour_feature_df) != len(hour_future_df):
            raise ValueError(f"feature rows missing for validation hour {current_hour}")

        hour_predictions = model.predict(hour_feature_df[FEATURE_COLS])

        hour_result_df = hour_future_df[[TIME_COL, TARGET_COL]].copy()
        hour_result_df["prediction"] = hour_predictions
        prediction_chunks.append(hour_result_df)

        history_append_df = hour_input_df.copy()
        history_append_df[TARGET_COL] = hour_predictions
        history_df = pd.concat([history_df, history_append_df], ignore_index=True)

    result_df = pd.concat(prediction_chunks, ignore_index=True)
    return result_df.sort_values(TIME_COL).reset_index(drop=True)


def plot_validation_result(val_result_df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(val_result_df[TIME_COL], val_result_df[TARGET_COL], label="Actual", linewidth=1.4)
    plt.plot(val_result_df[TIME_COL], val_result_df["prediction"], label="Predicted", linewidth=1.2)
    plt.title("Validation Set Hourly Recursive Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    if not TRAIN_FEATURE_PATH.exists():
        raise FileNotFoundError(f"missing training feature file: {TRAIN_FEATURE_PATH}")
    if not TRAIN_LABEL_PATH.exists():
        raise FileNotFoundError(f"missing training label file: {TRAIN_LABEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_training_data()
    feature_df = build_features(raw_df)
    feature_df = select_feature_ready_rows(feature_df, require_target=True)

    round_results: list[dict[str, object]] = []
    all_val_result_df_list: list[pd.DataFrame] = []

    for round_idx, round_cfg in enumerate(CV_ROUNDS, start=1):
        train_df, train_history_df, val_raw_df = get_round_data(
            feature_df=feature_df,
            raw_df=raw_df,
            train_end=round_cfg["train_end"],
            valid_month=round_cfg["valid_month"],
        )

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=100,
        )
        model.fit(X_train, y_train, verbose=100)

        val_result_df = recursive_predict_by_hour(model, train_history_df, val_raw_df)
        mae = mean_absolute_error(val_result_df[TARGET_COL], val_result_df["prediction"])
        rmse = np.sqrt(mean_squared_error(val_result_df[TARGET_COL], val_result_df["prediction"]))

        val_result_df["round"] = round_idx
        val_result_df["train_end_month"] = round_cfg["train_end"]
        val_result_df["valid_month"] = round_cfg["valid_month"]
        all_val_result_df_list.append(val_result_df)

        round_results.append(
            {
                "round": round_idx,
                "train_end_month": round_cfg["train_end"],
                "valid_month": round_cfg["valid_month"],
                "train_samples": len(train_df),
                "validation_samples": len(val_result_df),
                "mae": mae,
                "rmse": rmse,
            }
        )

    all_val_result_df = pd.concat(all_val_result_df_list, ignore_index=True)
    overall_mae = mean_absolute_error(all_val_result_df[TARGET_COL], all_val_result_df["prediction"])
    overall_rmse = np.sqrt(mean_squared_error(all_val_result_df[TARGET_COL], all_val_result_df["prediction"]))

    round_metrics_df = pd.DataFrame(round_results)

    val_result_path = OUTPUT_DIR / "catboost_val_predictions.csv"
    round_metric_path = OUTPUT_DIR / "catboost_round_metrics.csv"
    plot_path = OUTPUT_DIR / "catboost_val_prediction.png"
    all_val_result_df.to_csv(val_result_path, index=False)
    round_metrics_df.to_csv(round_metric_path, index=False)
    plot_validation_result(all_val_result_df, plot_path)

    print("=" * 60)
    print("CatBoost multi-round training finished")
    print("=" * 60)
    print(f"Raw merged samples: {len(raw_df)}")
    print(f"Samples after feature engineering: {len(feature_df)}")
    print(f"Features used: {FEATURE_COLS}")
    for round_result in round_results:
        print(
            f"Round {round_result['round']}: "
            f"train<= {round_result['train_end_month']}, "
            f"valid= {round_result['valid_month']}, "
            f"train_samples= {round_result['train_samples']}, "
            f"validation_samples= {round_result['validation_samples']}, "
            f"MAE= {round_result['mae']:.6f}, "
            f"RMSE= {round_result['rmse']:.6f}"
        )
    print(f"Overall validation samples: {len(all_val_result_df)}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Validation predictions saved to: {val_result_path}")
    print(f"Round metrics saved to: {round_metric_path}")
    print(f"Validation plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
