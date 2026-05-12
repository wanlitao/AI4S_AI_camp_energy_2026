# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = ROOT / "data" / "train" / "mengxi_boundary_anon_filtered_nc_attach.csv"
PRICE_PATH = ROOT / "data" / "train" / "mengxi_node_price_selected.csv"
OUTPUT_DIR = ROOT / "output" / "price_analysis"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["times"].dt.date
    df["hour"] = df["times"].dt.hour
    df["quarter"] = df["times"].dt.hour * 4 + (df["times"].dt.minute // 15)
    df["weekday"] = df["times"].dt.dayofweek
    df["month"] = df["times"].dt.month

    load_actual = df["系统负荷实际值"]
    load_forecast = df["系统负荷预测值"]
    renew_actual = df["风光总加实际值"]
    renew_forecast = df["风光总加预测值"]
    nonmarket_actual = df["非市场化机组实际值"]
    nonmarket_forecast = df["非市场化机组预测值"]

    df["renew_share"] = renew_actual / load_actual
    df["wind_share"] = df["风电实际值"] / load_actual
    df["solar_share"] = df["光伏实际值"] / load_actual
    df["load_gap"] = load_actual - load_forecast
    df["renew_gap"] = renew_actual - renew_forecast
    df["tie_gap"] = df["联络线实际值"] - df["联络线预测值"]
    df["wind_gap"] = df["风电实际值"] - df["风电预测值"]
    df["solar_gap"] = df["光伏实际值"] - df["光伏预测值"]
    df["hydro_gap"] = df["水电实际值"] - df["水电预测值"]
    df["nonmarket_gap"] = nonmarket_actual - nonmarket_forecast
    df["net_load"] = load_actual - renew_actual - nonmarket_actual
    df["net_load_forecast"] = load_forecast - renew_forecast - nonmarket_forecast
    df["residual_supply_gap"] = df["net_load"] - df["net_load_forecast"]

    df["price_diff"] = df["price"].diff()
    df["price_abs_diff"] = df["price_diff"].abs()
    df["price_pct_change"] = df["price"].pct_change()

    diff_cols = [
        "t2m_空间平均",
        "ghi_空间平均",
        "u100_空间平均",
        "v100_空间平均",
        "tp_空间平均",
        "系统负荷实际值",
        "风光总加实际值",
        "风电实际值",
        "光伏实际值",
        "net_load",
    ]
    for col in diff_cols:
        df[f"{col}_diff"] = df[col].diff()

    return df


def build_quantile_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    tmp = df[[feature, "price"]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[feature], 5, duplicates="drop")
    summary = (
        tmp.groupby("bin", observed=False)["price"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )
    summary.insert(0, "feature", feature)
    return summary


def save_event_summary(df: pd.DataFrame, threshold: float, label: str) -> None:
    event_mask = df["price_abs_diff"] >= threshold
    event_df = df.loc[event_mask].copy()
    normal_df = df.loc[~event_mask].copy()

    compare_cols = [
        "price",
        "price_diff",
        "系统负荷实际值",
        "load_gap",
        "风光总加实际值",
        "renew_share",
        "renew_gap",
        "net_load",
        "residual_supply_gap",
        "t2m_空间平均",
        "t2m_空间平均_diff",
        "ghi_空间平均",
        "ghi_空间平均_diff",
        "u100_空间平均",
        "u100_空间平均_diff",
        "tp_空间平均",
    ]

    summary = pd.DataFrame(
        {
            "event_mean": event_df[compare_cols].mean(),
            "normal_mean": normal_df[compare_cols].mean(),
        }
    )
    summary["delta"] = summary["event_mean"] - summary["normal_mean"]
    summary.to_csv(OUTPUT_DIR / f"{label.lower()}_event_vs_normal.csv", encoding="utf-8-sig")

    for sub_label, sub_df in [("up", event_df[event_df["price_diff"] > 0]), ("down", event_df[event_df["price_diff"] < 0])]:
        if sub_df.empty:
            continue
        sub_df[compare_cols].mean().to_frame("mean").to_csv(
            OUTPUT_DIR / f"{label.lower()}_{sub_label}_mean.csv",
            encoding="utf-8-sig",
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    boundary = pd.read_csv(BOUNDARY_PATH)
    price = pd.read_csv(PRICE_PATH).rename(columns={"A": "price"})

    boundary["times"] = pd.to_datetime(boundary["times"])
    price["times"] = pd.to_datetime(price["times"])

    df = boundary.merge(price, on="times", how="inner").sort_values("times").reset_index(drop=True)
    df = add_derived_features(df)

    meta = pd.Series(
        {
            "rows": len(df),
            "days": df["date"].nunique(),
            "start_time": df["times"].min(),
            "end_time": df["times"].max(),
            "price_mean": df["price"].mean(),
            "price_std": df["price"].std(),
            "price_p01": df["price"].quantile(0.01),
            "price_p05": df["price"].quantile(0.05),
            "price_p50": df["price"].quantile(0.50),
            "price_p95": df["price"].quantile(0.95),
            "price_p99": df["price"].quantile(0.99),
        }
    )
    meta.to_csv(OUTPUT_DIR / "meta_summary.csv", header=["value"], encoding="utf-8-sig")

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    missing_ratio.to_csv(OUTPUT_DIR / "missing_ratio.csv", header=["missing_ratio"], encoding="utf-8-sig")

    hourly_profile = df.groupby("hour")["price"].agg(mean="mean", median="median", std="std", count="count")
    hourly_profile.to_csv(OUTPUT_DIR / "hourly_profile.csv", encoding="utf-8-sig")

    quarter_profile = df.groupby("quarter")["price"].agg(mean="mean", median="median", std="std", count="count")
    quarter_profile.to_csv(OUTPUT_DIR / "quarter_profile.csv", encoding="utf-8-sig")

    weekday_profile = df.groupby("weekday")["price"].agg(mean="mean", median="median", std="std", count="count")
    weekday_profile.to_csv(OUTPUT_DIR / "weekday_profile.csv", encoding="utf-8-sig")

    month_profile = df.groupby("month")["price"].agg(mean="mean", median="median", std="std", count="count")
    month_profile.to_csv(OUTPUT_DIR / "month_profile.csv", encoding="utf-8-sig")

    daily_profile = df.groupby("date")["price"].agg(mean="mean", median="median", std="std", max="max", min="min")
    daily_profile.to_csv(OUTPUT_DIR / "daily_profile.csv", encoding="utf-8-sig")

    autocorr = pd.Series(
        {
            "lag_1": df["price"].autocorr(1),
            "lag_4": df["price"].autocorr(4),
            "lag_8": df["price"].autocorr(8),
            "lag_16": df["price"].autocorr(16),
            "lag_32": df["price"].autocorr(32),
            "lag_48": df["price"].autocorr(48),
            "lag_96": df["price"].autocorr(96),
            "lag_192": df["price"].autocorr(192),
            "lag_288": df["price"].autocorr(288),
            "lag_672": df["price"].autocorr(672),
        }
    )
    autocorr.to_csv(OUTPUT_DIR / "price_autocorr.csv", header=["autocorr"], encoding="utf-8-sig")

    corr_cols = [
        "系统负荷实际值",
        "系统负荷预测值",
        "风光总加实际值",
        "风光总加预测值",
        "联络线实际值",
        "联络线预测值",
        "风电实际值",
        "风电预测值",
        "光伏实际值",
        "光伏预测值",
        "水电实际值",
        "水电预测值",
        "非市场化机组实际值",
        "非市场化机组预测值",
        "u100_空间平均",
        "v100_空间平均",
        "ghi_空间平均",
        "tp_空间平均",
        "t2m_空间平均",
        "renew_share",
        "wind_share",
        "solar_share",
        "load_gap",
        "renew_gap",
        "tie_gap",
        "wind_gap",
        "solar_gap",
        "hydro_gap",
        "nonmarket_gap",
        "net_load",
        "net_load_forecast",
        "residual_supply_gap",
    ]
    correlation = (
        df[corr_cols + ["price"]]
        .corr(numeric_only=True)["price"]
        .drop("price")
        .to_frame("pearson_corr")
        .sort_values("pearson_corr", key=lambda s: s.abs(), ascending=False)
    )
    correlation.to_csv(OUTPUT_DIR / "feature_price_correlation.csv", encoding="utf-8-sig")

    quantile_tables = []
    for feature in [
        "t2m_空间平均",
        "renew_share",
        "ghi_空间平均",
        "系统负荷实际值",
        "net_load",
        "residual_supply_gap",
    ]:
        quantile_tables.append(build_quantile_summary(df, feature))
    pd.concat(quantile_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "quantile_price_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    time_windows = {
        "morning_peak": df["hour"].between(7, 10),
        "evening_peak": df["hour"].between(17, 21),
        "midday": df["hour"].between(11, 14),
        "overnight": df["hour"].between(0, 5),
    }
    time_window_summary = []
    for name, mask in time_windows.items():
        series = df.loc[mask, "price"]
        time_window_summary.append(
            {
                "window": name,
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "count": series.shape[0],
            }
        )
    pd.DataFrame(time_window_summary).to_csv(
        OUTPUT_DIR / "time_window_price_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vol95 = df["price_abs_diff"].quantile(0.95)
    vol99 = df["price_abs_diff"].quantile(0.99)
    pd.Series({"vol95_threshold": vol95, "vol99_threshold": vol99}).to_csv(
        OUTPUT_DIR / "volatility_thresholds.csv",
        header=["value"],
        encoding="utf-8-sig",
    )
    save_event_summary(df, vol95, "vol95")
    save_event_summary(df, vol99, "vol99")

    top_spikes = df.nlargest(
        30,
        "price_abs_diff",
    )[
        [
            "times",
            "price",
            "price_diff",
            "系统负荷实际值",
            "load_gap",
            "风光总加实际值",
            "renew_share",
            "net_load",
            "residual_supply_gap",
            "t2m_空间平均",
            "t2m_空间平均_diff",
            "ghi_空间平均",
            "ghi_空间平均_diff",
            "u100_空间平均",
            "u100_空间平均_diff",
        ]
    ]
    top_spikes.to_csv(OUTPUT_DIR / "top_price_spikes.csv", index=False, encoding="utf-8-sig")

    model_cols = [
        "quarter",
        "weekday",
        "month",
        "系统负荷实际值",
        "系统负荷预测值",
        "风光总加实际值",
        "风光总加预测值",
        "联络线实际值",
        "联络线预测值",
        "风电实际值",
        "风电预测值",
        "光伏实际值",
        "光伏预测值",
        "水电实际值",
        "水电预测值",
        "非市场化机组实际值",
        "非市场化机组预测值",
        "u100_空间平均",
        "v100_空间平均",
        "ghi_空间平均",
        "tp_空间平均",
        "t2m_空间平均",
        "renew_share",
        "load_gap",
        "renew_gap",
        "tie_gap",
        "wind_gap",
        "solar_gap",
        "hydro_gap",
        "nonmarket_gap",
        "net_load",
        "residual_supply_gap",
    ]
    model_df = df[model_cols + ["price"]].dropna().copy()
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(model_df[model_cols], model_df["price"])
    importance = permutation_importance(
        model,
        model_df[model_cols],
        model_df["price"],
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance_df = (
        pd.Series(importance.importances_mean, index=model_cols, name="permutation_importance")
        .sort_values(ascending=False)
        .to_frame()
    )
    importance_df.to_csv(OUTPUT_DIR / "model_feature_importance.csv", encoding="utf-8-sig")

    temp_shock_summary = []
    temp_drop_threshold = df["t2m_空间平均_diff"].quantile(0.05)
    temp_rise_threshold = df["t2m_空间平均_diff"].quantile(0.95)
    for name, mask in [
        ("temp_drop_5pct", df["t2m_空间平均_diff"] <= temp_drop_threshold),
        ("temp_rise_5pct", df["t2m_空间平均_diff"] >= temp_rise_threshold),
    ]:
        subset = df.loc[mask]
        temp_shock_summary.append(
            {
                "bucket": name,
                "count": len(subset),
                "price_mean": subset["price"].mean(),
                "price_diff_mean": subset["price_diff"].mean(),
                "load_mean": subset["系统负荷实际值"].mean(),
                "renew_share_mean": subset["renew_share"].mean(),
                "ghi_mean": subset["ghi_空间平均"].mean(),
                "u100_mean": subset["u100_空间平均"].mean(),
            }
        )
    pd.DataFrame(temp_shock_summary).to_csv(
        OUTPUT_DIR / "temperature_shock_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"analysis saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
