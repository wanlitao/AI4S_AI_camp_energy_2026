from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
NC_CLEAN_DIR = ROOT_DIR / "data" / "train" / "nc" / "clean"
TRAIN_INPUT_PATH = ROOT_DIR / "data" / "train" / "mengxi_boundary_anon_filtered.csv"
TEST_INPUT_PATH = ROOT_DIR / "data" / "test" / "test_in_feature_ori.csv"
TRAIN_OUTPUT_PATH = ROOT_DIR / "data" / "train" / "mengxi_boundary_anon_filtered_nc_attach.csv"
TEST_OUTPUT_PATH = ROOT_DIR / "data" / "test" / "test_in_feature_ori_nc_attach.csv"


def load_weather_data() -> pd.DataFrame:
    clean_paths = sorted(NC_CLEAN_DIR.glob("*.csv"))
    if not clean_paths:
        raise FileNotFoundError(f"No cleaned nc csv files found in: {NC_CLEAN_DIR}")

    frames: list[pd.DataFrame] = []
    for path in clean_paths:
        frame = pd.read_csv(path)
        if "times" not in frame.columns:
            raise KeyError(f"'times' column not found in weather file: {path}")
        frames.append(frame)

    weather_df = pd.concat(frames, ignore_index=True)
    weather_df["times"] = pd.to_datetime(weather_df["times"])

    if weather_df["times"].duplicated().any():
        duplicate_times = weather_df.loc[weather_df["times"].duplicated(), "times"].head(5).tolist()
        raise ValueError(f"Duplicate weather times found, sample: {duplicate_times}")

    return weather_df.sort_values("times").reset_index(drop=True)


def attach_weather(base_path: Path, output_path: Path, weather_df: pd.DataFrame, year: int) -> None:
    base_df = pd.read_csv(base_path)
    if "times" not in base_df.columns:
        raise KeyError(f"'times' column not found in base file: {base_path}")

    base_df = base_df.copy()
    base_df["_row_id"] = range(len(base_df))
    base_df["times"] = pd.to_datetime(base_df["times"])

    year_weather_df = weather_df.loc[weather_df["times"].dt.year == year].copy()
    weather_columns = [column for column in year_weather_df.columns if column != "times"]

    merged_df = base_df.merge(year_weather_df, on="times", how="left", sort=False)
    merged_df = merged_df.sort_values("_row_id").drop(columns="_row_id")
    merged_df["times"] = merged_df["times"].dt.strftime("%Y-%m-%d %H:%M:%S")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    matched_rows = merged_df[weather_columns].notna().any(axis=1).sum() if weather_columns else 0
    print(f"[DONE] {base_path.name} -> {output_path.name}")
    print(f"  Rows: {len(merged_df)}")
    print(f"  Weather columns attached: {len(weather_columns)}")
    print(f"  Matched rows: {matched_rows}")
    print(f"  Unmatched rows: {len(merged_df) - matched_rows}")


def main() -> None:
    weather_df = load_weather_data()
    attach_weather(TRAIN_INPUT_PATH, TRAIN_OUTPUT_PATH, weather_df, year=2025)
    attach_weather(TEST_INPUT_PATH, TEST_OUTPUT_PATH, weather_df, year=2026)


if __name__ == "__main__":
    main()
