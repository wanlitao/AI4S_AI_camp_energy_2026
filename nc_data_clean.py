from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "all_nc"
RAW_OUTPUT_DIR = ROOT_DIR / "data" / "train" / "nc" / "raw"
CLEAN_OUTPUT_DIR = ROOT_DIR / "data" / "train" / "nc" / "clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert NetCDF weather files into raw and cleaned CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing .nc files. Defaults to data/all_nc",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single .nc file to process instead of the whole directory.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip writing raw CSV files.",
    )
    return parser.parse_args()


def resolve_channel_name(channels: list[str], preferred_name: str) -> str:
    if preferred_name in channels:
        return preferred_name
    raise ValueError(f"Required channel '{preferred_name}' not found. Available: {channels}")


def resolve_ghi_channel(channels: list[str]) -> str:
    if "ghi" in channels:
        return "ghi"

    keywords = ("ghi", "irr", "radi", "solar", "srad", "ssrd")
    for channel in channels:
        normalized = channel.lower()
        if any(keyword in normalized for keyword in keywords):
            return channel

    raise ValueError(
        "No irradiance channel found. Expected something like 'ghi'. "
        f"Available: {channels}"
    )


def build_raw_dataframe(dataset: xr.Dataset) -> pd.DataFrame:
    raw_long_df = dataset.to_dataframe().reset_index()
    raw_wide_df = (
        raw_long_df.pivot_table(
            index=["time", "lead_time", "lat", "lon"],
            columns="channel",
            values="data",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    raw_wide_df["time"] = pd.to_datetime(raw_wide_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return raw_wide_df


def save_raw_csv(dataset: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df = build_raw_dataframe(dataset)
    raw_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_clean_dataframe(dataset: xr.Dataset) -> pd.DataFrame:
    channels = [str(channel) for channel in dataset["channel"].values.tolist()]
    u100_channel = resolve_channel_name(channels, "u100")
    v100_channel = resolve_channel_name(channels, "v100")
    ghi_channel = resolve_ghi_channel(channels)

    mean_df = (
        dataset["data"]
        .sel(channel=[u100_channel, v100_channel, ghi_channel])
        .mean(dim=["lat", "lon"], skipna=True)
        .to_dataframe(name="spatial_mean")
        .reset_index()
    )

    clean_df = (
        mean_df.pivot_table(
            index=["time", "lead_time"],
            columns="channel",
            values="spatial_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    base_dates = pd.to_datetime(clean_df["time"]).dt.normalize() + pd.Timedelta(days=1)
    clean_df["lead_time"] = clean_df["lead_time"].astype(int)
    clean_df["times"] = base_dates + pd.to_timedelta(clean_df["lead_time"], unit="h")

    clean_df = clean_df.rename(
        columns={
            u100_channel: "u100_空间平均",
            v100_channel: "v100_空间平均",
            ghi_channel: "ghi_空间平均",
        }
    )

    base_output = clean_df[["times", "u100_空间平均", "v100_空间平均", "ghi_空间平均"]].copy()

    expanded_frames = []
    for offset_minutes in (0, 15, 30, 45):
        frame = base_output.copy()
        frame["times"] = frame["times"] + pd.Timedelta(minutes=offset_minutes)
        expanded_frames.append(frame)

    result = pd.concat(expanded_frames, ignore_index=True).sort_values("times")
    result["times"] = pd.to_datetime(result["times"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return result.reset_index(drop=True)


def save_clean_csv(dataset: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = build_clean_dataframe(dataset)
    clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_file is not None:
        input_path = args.input_file
        if not input_path.is_absolute():
            input_path = ROOT_DIR / input_path
        input_path = input_path.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = ROOT_DIR / input_dir
    input_dir = input_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_paths = sorted(input_dir.glob("*.nc"))
    if not input_paths:
        raise FileNotFoundError(f"No .nc files found in: {input_dir}")
    return input_paths


def process_one_file(input_path: Path, *, write_raw: bool) -> None:
    raw_output_path = RAW_OUTPUT_DIR / f"{input_path.stem}_raw.csv"
    clean_output_path = CLEAN_OUTPUT_DIR / f"{input_path.stem}_clean.csv"

    with xr.open_dataset(input_path) as dataset:
        if write_raw:
            save_raw_csv(dataset, raw_output_path)
        save_clean_csv(dataset, clean_output_path)

    print(f"[DONE] {input_path.name}")
    if write_raw:
        print(f"  Raw CSV: {raw_output_path}")
    else:
        print("  Raw CSV: skipped (--no-raw)")
    print(f"  Clean CSV: {clean_output_path}")


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(args)

    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(input_paths)} nc file(s) to process.")
    for index, input_path in enumerate(input_paths, start=1):
        print(f"[{index}/{len(input_paths)}] Processing {input_path.name} ...")
        process_one_file(input_path, write_raw=not args.no_raw)


if __name__ == "__main__":
    main()
