#!/usr/bin/env python
"""
MLP test worker — train on all training years, predict on held-out test years.

Array layout (6 tasks, one per zone):
    SLURM_ARRAY_TASK_ID = zone_idx  (0–5)

Outputs one parquet per task to --outdir:
    mlp_test_preds_zone{zone_idx}.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib

# ── Constants (must match mlp_worker.py exactly) ─────────────────────────────
LRZ_IDS = ["MISO-0001", "MISO-0027", "MISO-0035", "MISO-0004", "MISO-0006", "MISO-8910"]
YEARS   = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
SEED        = 42
TEST_YEARS  = [2016, 2021, 2024]
TRAIN_YEARS = sorted(set(YEARS) - set(TEST_YEARS))

MODEL_FREQ = "6H"

# ── Holiday window settings ───────────────────────────────────────────────────
HOLIDAYS       = ["memorial_day", "july_4th", "labor_day", "christmas", "new_years"]
HOLIDAY_WINDOW = 0  # days on each side

CDD_BASE = 18.0  # °C (≈ 65°F)

GAM_COLS = [
    "cdd", "hdd",
    "cdd_lag6", "hdd_lag6",
    "cdd_lag12", "hdd_lag12",
    "cdd_lag18", "hdd_lag18",
    "cdd_lag24", "hdd_lag24",
    "hour", "month", "day_of_week", "year",
    "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
]


# ── Feature encoding ──────────────────────────────────────────────────────────
def encode_features(X_df):
    return X_df[GAM_COLS].copy()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MLP test worker")
    parser.add_argument(
        "--task-id", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="Zone index 0–5 (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing MISO historical loads/ and ERA5/ subdirectories",
    )
    parser.add_argument(
        "--outdir", default="mlp_test_results",
        help="Output directory for per-zone prediction parquets",
    )
    parser.add_argument(
        "--model-dir", default="mlp_tuned_results",
        help="Directory containing tuned mlp_tuned_zone*.joblib files",
    )
    return parser.parse_args()


# ── Holiday helpers ───────────────────────────────────────────────────────────
def get_holiday_dates(years):
    year_range = range(min(years) - 1, max(years) + 2)
    dates = {h: set() for h in HOLIDAYS}
    for year in year_range:
        may_31 = pd.Timestamp(year, 5, 31)
        dates["memorial_day"].add(may_31 - pd.Timedelta(days=may_31.dayofweek))

        dates["july_4th"].add(pd.Timestamp(year, 7, 4))

        sep_1 = pd.Timestamp(year, 9, 1)
        dates["labor_day"].add(sep_1 + pd.Timedelta(days=(7 - sep_1.dayofweek) % 7))

        dates["christmas"].add(pd.Timestamp(year, 12, 25))
        dates["new_years"].add(pd.Timestamp(year, 1, 1))

    return {h: frozenset(v) for h, v in dates.items()}


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features_raw(temp_series):
    t = temp_series

    X = pd.DataFrame(index=t.index)
    X["cdd"] = np.maximum(0.0, t - CDD_BASE)
    X["hdd"] = np.maximum(0.0, CDD_BASE - t)

    def create_lag(X, n):
        shift = n // 6 if MODEL_FREQ == "6H" else n
        X[f"cdd_lag{n}"] = X["cdd"].shift(shift)
        X[f"hdd_lag{n}"] = X["hdd"].shift(shift)
        return X

    X = create_lag(X, 6)
    X = create_lag(X, 12)
    X = create_lag(X, 18)
    X = create_lag(X, 24)

    # Zero out lag values that spuriously cross a gap in the time index
    step = pd.Timedelta("6h") if MODEL_FREQ == "6H" else pd.Timedelta("1h")
    gap_mask = X.index.to_series().diff() > step
    for col in [c for c in X.columns if "lag" in c]:
        X.loc[gap_mask, col] = np.nan

    X["hour"]        = t.index.hour
    X["day_of_week"] = t.index.day_of_week
    X["weekend"]     = (t.index.day_of_week > 4).astype(int)
    X["month"]       = t.index.month
    X["year"]        = t.index.year

    date_index = t.index.normalize()
    if date_index.tz is not None:
        date_index = date_index.tz_localize(None)

    holiday_dates = get_holiday_dates(t.index.year.unique())
    for holiday, day_set in holiday_dates.items():
        col = np.zeros(len(t.index), dtype=int)
        for level, offset in enumerate(range(-HOLIDAY_WINDOW, HOLIDAY_WINDOW + 1), start=1):
            shifted = frozenset(d + pd.Timedelta(days=offset) for d in day_set)
            col[date_index.isin(shifted)] = level
        X[holiday] = col

    return X


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(data_dir):
    load_df = pd.DataFrame(columns=["LoadResource Zone", "ActualLoad (MWh)"])
    for year in YEARS:
        load_ts = pd.read_excel(
            os.path.join(data_dir, "MISO historical loads", f"{year}1231_dfal_HIST.xls"),
            index_col=0, header=5,
        )
        load_ts = load_ts[:-2].drop(columns=["MTLF (MWh)"])
        load_ts = load_ts[~(load_ts.index == "MarketDay")]
        load_ts.index = pd.to_datetime(load_ts.index)
        load_ts.index = load_ts.index + pd.to_timedelta(load_ts["HourEnding"] - 1, unit="h")
        load_df = pd.concat([load_df, load_ts[["LoadResource Zone", "ActualLoad (MWh)"]]], axis=0)

    temp_data = pd.DataFrame(columns=LRZ_IDS)
    for year in YEARS:
        temp_ts = pd.read_csv(
            os.path.join(data_dir, "ERA5", f"MISO_t2m_{year}.csv"),
            index_col=0, header=0, parse_dates=True,
        )
        temp_data = pd.concat([temp_data, temp_ts], axis=0)

    load_wide = load_df.pivot_table(
        index=load_df.index, columns="LoadResource Zone", values="ActualLoad (MWh)"
    )
    load_wide["LRZ8_9_10"] = load_wide["LRZ8_9_10"].fillna(0) + load_wide["LRZ8_9"].fillna(0)
    load_wide = load_wide.drop(columns=["LRZ8_9", "MISO"])
    load_wide = load_wide.rename(columns=dict(zip(load_wide.columns, LRZ_IDS)))

    if MODEL_FREQ == "6H":
        # resample to 6-hourly, backwards looking to match CESM
        # so 06:00 timestamp represents 0:00-06:00 average
        temp_data = temp_data.tz_localize("UTC").resample("6h", closed="right", label="right").mean()
        load_wide.index = pd.to_datetime(load_wide.index).tz_localize("Etc/GMT+5").tz_convert("UTC")
        load_wide = load_wide.resample("6h", closed="right", label="right").mean()

        temp_data.index = temp_data.index.tz_convert("America/Indiana/Indianapolis")
        load_wide.index  = load_wide.index.tz_convert("America/Indiana/Indianapolis")
    else:
        temp_data.index = temp_data.index.tz_localize("UTC").tz_convert("America/Indiana/Indianapolis")
        load_wide.index = pd.to_datetime(load_wide.index).tz_localize("Etc/GMT+5").tz_convert("America/Indiana/Indianapolis")

    common_index = temp_data.index.intersection(load_wide.index)
    temp_data = temp_data.loc[common_index]
    load_wide = load_wide.loc[common_index]

    valid_rows = temp_data.notna().all(axis=1) & load_wide.notna().all(axis=1)
    temp_data = temp_data.loc[valid_rows]
    load_wide = load_wide.loc[valid_rows]

    return temp_data, load_wide


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    zone_idx = args.task_id
    if zone_idx >= len(LRZ_IDS):
        sys.exit(f"task_id {args.task_id} out of range (max {len(LRZ_IDS) - 1})")

    zone = LRZ_IDS[zone_idx]
    print(f"[task {args.task_id}] zone={zone} ({zone_idx})", flush=True)

    train_years = TRAIN_YEARS
    test_years  = TEST_YEARS
    print(f"[task {args.task_id}] train_years={train_years}", flush=True)
    print(f"[task {args.task_id}] test_years={test_years}", flush=True)

    print(f"[task {args.task_id}] Loading data from {args.data_dir} ...", flush=True)
    temp_data, load_wide = load_data(args.data_dir)

    test_mask = temp_data.index.year.isin(test_years)
    temp_test = temp_data.loc[test_mask]
    load_test = load_wide.loc[test_mask]

    X_te_all = build_features_raw(temp_test[zone]).dropna()
    y_te_all = load_test[zone].loc[X_te_all.index]
    X_te_enc = encode_features(X_te_all).values

    tuned_path = os.path.join(args.model_dir, f"mlp_tuned_zone{zone_idx}.joblib")
    print(f"[task {args.task_id}] Loading tuned model from {tuned_path} ...", flush=True)
    bundle = joblib.load(tuned_path)
    mlp    = bundle["model"]
    scaler = bundle["scaler"]
    print(f"[task {args.task_id}] Best params: {bundle['best_params']}", flush=True)

    X_te_scaled = scaler.transform(X_te_enc)
    preds = mlp.predict(X_te_scaled)
    print(f"[task {args.task_id}] Done. n_test={len(preds)}", flush=True)

    out = pd.DataFrame(
        {"actual": y_te_all.values, "predicted": preds},
        index=y_te_all.index,
    )
    out["zone"]       = zone
    out["zone_idx"]   = zone_idx
    out["test_years"] = str(test_years)

    out_path = os.path.join(args.outdir, f"mlp_test_preds_zone{zone_idx}.parquet")
    out.to_parquet(out_path)
    print(f"[task {args.task_id}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
