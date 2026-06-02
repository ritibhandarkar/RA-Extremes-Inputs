#!/usr/bin/env python
"""
GAM CV worker — one SLURM array task per (zone, fold).

Array layout (48 tasks, 6 zones × 8 folds):
    SLURM_ARRAY_TASK_ID = zone_idx * N_FOLDS + fold_idx

Outputs one parquet per task to --outdir:
    gam_preds_zone{zone_idx}_fold{fold_idx}.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from pygam import LinearGAM, s, f, te

# ── Constants (must match notebook exactly) ──────────────────────────────────
LRZ_IDS = ["MISO-0001", "MISO-0027", "MISO-0035", "MISO-0004", "MISO-0006", "MISO-8910"]
YEARS   = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
SEED    = 42
N_FOLDS = 8   # leave-one-year-out over the 8 training years

# ── Sample weights ───────────────────────────────────────────────────────────
# Top HIGH_LOAD_WEIGHT_PERCENTILE% of training load values receive HIGH_LOAD_WEIGHT
# weight; all others receive weight 1.
HIGH_LOAD_WEIGHT_PERCENTILE = 90
HIGH_LOAD_WEIGHT            = 1

# ── Toggle: exclude top-percentile load hours from training ───────────────────
# When True: train on hours ≤ p{EXTREME_PERCENTILE}, evaluate on hours > p{EXTREME_PERCENTILE}
EXCLUDE_EXTREMES   = True
EXTREME_PERCENTILE = 99

# ── Holiday window settings ───────────────────────────────────────────────────
# Each holiday gets an integer-encoded column: 0 = not in window,
# 1 = -2 days, 2 = -1 day, 3 = day-of, 4 = +1 day, 5 = +2 days.
HOLIDAYS       = ["memorial_day", "july_4th", "labor_day", "christmas", "new_years"]
HOLIDAY_WINDOW = 2  # days on each side


CDD_BASE = 18.0  # °C (≈ 65°F) — standard US utility base temperature


GAM_COLS = [
    "cdd", "hdd",
    "cdd_lag1", "hdd_lag1",
    "cdd_lag24", "hdd_lag24",
    "hour", "month", "day_of_week", "year",
    "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
]

# col indices: 0:cdd  1:hdd  2:cdd_lag1  3:hdd_lag1  4:cdd_lag24  5:hdd_lag24
#              6:hour  7:month  8:dow  9:year
#              10:memorial_day  11:july_4th  12:labor_day  13:christmas  14:new_years
EDGE_KNOT_HEADROOM = 0.10  # extend CDD/HDD boundary this fraction beyond training max


def build_gam_formula(X_tr_fit):
    """Build GAM formula with edge_knots set from training data + headroom."""
    cdd_max = X_tr_fit["cdd"].max() * (1 + EDGE_KNOT_HEADROOM)
    hdd_max = X_tr_fit["hdd"].max() * (1 + EDGE_KNOT_HEADROOM)
    return (
        s(0, n_splines=15, edge_knots=[0, cdd_max])   # cdd
        + s(1, n_splines=15, edge_knots=[0, hdd_max]) # hdd
        + s(2, n_splines=8)                            # cdd_lag1
        + s(3, n_splines=8)                            # hdd_lag1
        + s(4, n_splines=8)                            # cdd_lag24
        + s(5, n_splines=8)                            # hdd_lag24
        + te(6, 7, n_splines=[24, 12])                 # hour × month tensor product
        + f(8)                                         # day_of_week
        + s(9, n_splines=6)                            # year
        + f(10)                                        # memorial_day window
        + f(11)                                        # july_4th window
        + f(12)                                        # labor_day window
        + f(13)                                        # christmas window
        + f(14)                                        # new_years window
    )


def build_features_raw(temp_series):
    t   = temp_series
    t_f = t * 9 / 5 + 32

    X = pd.DataFrame(index=t.index)
    X["cdd"]      = np.maximum(0.0, t - CDD_BASE)
    X["hdd"]      = np.maximum(0.0, CDD_BASE - t)
    X["cdd_lag1"]  = X["cdd"].shift(1)
    X["hdd_lag1"]  = X["hdd"].shift(1)
    X["cdd_lag24"] = X["cdd"].shift(24)
    X["hdd_lag24"] = X["hdd"].shift(24)

    daily_avg_f = t_f.resample("D").mean().reindex(t.index, method="ffill")
    bin_edges   = np.arange(
        np.floor(daily_avg_f.min() / 10) * 10,
        np.ceil(daily_avg_f.max()  / 10) * 10 + 10,
        10,
    )
    X["temp_bin"]    = pd.cut(daily_avg_f, bins=bin_edges, labels=False).astype(float)
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


# GAM_COLS = [
#     "temp", "temp_squared", "temp_lag1", "temp_lag24",
#     "hour", "month", "day_of_week",  "year",
#     "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
# ]

# # column indices:
# #   0:temp  1:temp_sq  2:lag1  3:lag24  4:hour  5:month  6:dow  7:weekend  8:year
# #   9:memorial_day  10:july_4th  11:labor_day  12:christmas  13:new_years
# GAM_FORMULA = (
#     s(0, n_splines=20)              # temp
#     + s(1, n_splines=10)            # temp_squared
#     + s(2, n_splines=10)            # temp_lag1
#     + s(3, n_splines=10)            # temp_lag24
#     + te(4, 5, n_splines=[24, 12])  # hour × month tensor product
#     + f(6)                          # day_of_week
#     # + f(7)                          # weekend (factor)
#     + s(7, n_splines=6)             # year
#     + f(8)                          # memorial_day window (factor)
#     + f(9)                         # july_4th window (factor)
#     + f(10)                         # labor_day window (factor)
#     + f(11)                         # christmas window (factor)
#     + f(12)                         # new_years window (factor)
# )


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="GAM CV worker")
    parser.add_argument(
        "--task-id", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="Flat task index (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing MISO historical loads/ and ERA5/ subdirectories",
    )
    parser.add_argument(
        "--outdir", default="gam_results",
        help="Output directory for per-fold prediction parquets",
    )
    return parser.parse_args()


# ── Holiday helpers ───────────────────────────────────────────────────────────
def get_holiday_dates(years):
    """
    Return {holiday_name: frozenset of tz-naive pd.Timestamps at midnight}
    for Memorial Day, July 4th, Labor Day, Christmas, and New Year's Day.
    Generates dates for years-1 through years+1 so ±HOLIDAY_WINDOW day
    windows are covered at series boundaries.
    """
    year_range = range(min(years) - 1, max(years) + 2)
    dates = {h: set() for h in HOLIDAYS}
    for year in year_range:
        # Memorial Day: last Monday of May
        may_31 = pd.Timestamp(year, 5, 31)
        dates["memorial_day"].add(may_31 - pd.Timedelta(days=may_31.dayofweek))

        # Independence Day
        dates["july_4th"].add(pd.Timestamp(year, 7, 4))

        # Labor Day: first Monday of September
        sep_1 = pd.Timestamp(year, 9, 1)
        dates["labor_day"].add(sep_1 + pd.Timedelta(days=(7 - sep_1.dayofweek) % 7))

        # Christmas
        dates["christmas"].add(pd.Timestamp(year, 12, 25))

        # New Year's Day
        dates["new_years"].add(pd.Timestamp(year, 1, 1))

    return {h: frozenset(v) for h, v in dates.items()}


# # ── Feature engineering ──────────────────────────────────────────────────────
# def build_features_raw(temp_series):
#     """Build raw feature DataFrame for a single LRZ temperature series."""
#     t   = temp_series
#     t_f = t * 9 / 5 + 32

#     X = pd.DataFrame(index=t.index)
#     X["temp"]         = t
#     X["temp_squared"] = t ** 2
#     X["temp_lag1"]    = t.shift(1)
#     X["temp_lag24"]   = t.shift(24)

#     daily_avg_f = t_f.resample("D").mean().reindex(t.index, method="ffill")
#     bin_edges   = np.arange(
#         np.floor(daily_avg_f.min() / 10) * 10,
#         np.ceil(daily_avg_f.max()  / 10) * 10 + 10,
#         10,
#     )
#     X["temp_bin"]    = pd.cut(daily_avg_f, bins=bin_edges, labels=False).astype(float)
#     X["hour"]        = t.index.hour
#     X["day_of_week"] = t.index.day_of_week
#     X["weekend"]     = (t.index.day_of_week > 4).astype(int)
#     X["month"]       = t.index.month
#     X["year"]        = t.index.year

#     # Holiday window features
#     # Strip timezone so comparisons with tz-naive holiday dates work correctly.
#     date_index = t.index.normalize()
#     if date_index.tz is not None:
#         date_index = date_index.tz_localize(None)

#     holiday_dates = get_holiday_dates(t.index.year.unique())
#     # Encoding: 0 = not in window, 1 = -2 days, 2 = -1 day, 3 = day-of,
#     #           4 = +1 day, 5 = +2 days
#     for holiday, day_set in holiday_dates.items():
#         col = np.zeros(len(t.index), dtype=int)
#         for level, offset in enumerate(range(-HOLIDAY_WINDOW, HOLIDAY_WINDOW + 1), start=1):
#             shifted = frozenset(d + pd.Timedelta(days=offset) for d in day_set)
#             col[date_index.isin(shifted)] = level
#         X[holiday] = col

#     return X


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(data_dir):
    load_data = pd.DataFrame(columns=["LoadResource Zone", "ActualLoad (MWh)"])
    for year in YEARS:
        load_ts = pd.read_excel(
            os.path.join(data_dir, "MISO historical loads", f"{year}1231_dfal_HIST.xls"),
            index_col=0, header=5,
        )
        load_ts = load_ts[:-2].drop(columns=["MTLF (MWh)"])
        load_ts = load_ts[~(load_ts.index == "MarketDay")]
        load_ts.index = pd.to_datetime(load_ts.index)
        load_ts.index = load_ts.index + pd.to_timedelta(load_ts["HourEnding"] - 1, unit="h")
        load_data = pd.concat([load_data, load_ts[["LoadResource Zone", "ActualLoad (MWh)"]]], axis=0)

    temp_data = pd.DataFrame(columns=LRZ_IDS)
    for year in YEARS:
        temp_ts = pd.read_csv(
            os.path.join(data_dir, "ERA5", f"MISO_t2m_{year}.csv"),
            index_col=0, header=0, parse_dates=True,
        )
        temp_data = pd.concat([temp_data, temp_ts], axis=0)

    # Reshape load
    load_wide = load_data.pivot_table(
        index=load_data.index, columns="LoadResource Zone", values="ActualLoad (MWh)"
    )
    load_wide["LRZ8_9_10"] = load_wide["LRZ8_9_10"].fillna(0) + load_wide["LRZ8_9"].fillna(0)
    load_wide = load_wide.drop(columns=["LRZ8_9", "MISO"])
    load_wide = load_wide.rename(columns=dict(zip(load_wide.columns, LRZ_IDS)))

    # Align timezones
    temp_data.index = temp_data.index.tz_localize("UTC").tz_convert("Etc/GMT+5")
    load_wide.index = pd.to_datetime(load_wide.index).tz_localize("Etc/GMT+5")

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

    # Decode task ID
    zone_idx = args.task_id // N_FOLDS
    fold_idx = args.task_id  % N_FOLDS

    if zone_idx >= len(LRZ_IDS):
        sys.exit(f"task_id {args.task_id} out of range (max {len(LRZ_IDS) * N_FOLDS - 1})")

    zone = LRZ_IDS[zone_idx]
    print(f"[task {args.task_id}] zone={zone} ({zone_idx}), fold={fold_idx}", flush=True)

    # Reproducible train/test split (identical to notebook)
    rng = np.random.default_rng(SEED)
    train_years = sorted(rng.choice(YEARS, N_FOLDS, replace=False).tolist())
    print(f"[task {args.task_id}] train_years={train_years}", flush=True)

    # Load data
    print(f"[task {args.task_id}] Loading data from {args.data_dir} ...", flush=True)
    temp_data, load_wide = load_data(args.data_dir)

    # Filter to training years
    train_mask = temp_data.index.year.isin(train_years)
    temp  = temp_data.loc[train_mask]
    load  = load_wide.loc[train_mask]

    # Build features for this zone (lags computed on full training series)
    X_all = build_features_raw(temp[zone]).dropna()
    y_all = load[zone].loc[X_all.index]
    groups = pd.Series(X_all.index.year, index=X_all.index)

    # Get all LOGO splits and select the requested fold
    logo   = LeaveOneGroupOut()
    splits = list(logo.split(X_all, y_all, groups))

    if fold_idx >= len(splits):
        sys.exit(f"fold_idx {fold_idx} >= available folds ({len(splits)})")

    train_idx, test_idx = splits[fold_idx]
    held_out_year = int(groups.iloc[test_idx[0]])
    print(f"[task {args.task_id}] Held-out year: {held_out_year}", flush=True)

    X_tr = X_all[GAM_COLS].iloc[train_idx]
    X_te = X_all[GAM_COLS].iloc[test_idx]
    y_tr = y_all.iloc[train_idx]
    y_te = y_all.iloc[test_idx]

    # Optionally restrict training and evaluation to extremes
    if EXCLUDE_EXTREMES:
        train_thresh = np.percentile(y_tr, EXTREME_PERCENTILE)
        X_tr_fit = X_tr[y_tr <= train_thresh]
        y_tr_fit = y_tr[y_tr <= train_thresh]

        test_thresh = np.percentile(y_te, EXTREME_PERCENTILE)
        extreme_mask = y_te > test_thresh
        X_te_eval = X_te #[extreme_mask]
        y_te_eval = y_te #[extreme_mask]
        print(f"[task {args.task_id}] Extremes mode: "
              f"train n={len(y_tr_fit)} (≤{train_thresh:.0f} MWh), "
            #   f"eval n={len(y_te_eval)} (>{test_thresh:.0f} MWh)", flush=True
            )
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr
        X_te_eval, y_te_eval = X_te, y_te

    # Build sample weights: top HIGH_LOAD_WEIGHT_PERCENTILE% of training y → HIGH_LOAD_WEIGHT
    w_thresh = np.percentile(y_tr_fit, HIGH_LOAD_WEIGHT_PERCENTILE)
    weights  = np.where(y_tr_fit.values >= w_thresh, HIGH_LOAD_WEIGHT, 1.0)
    n_high   = (y_tr_fit.values >= w_thresh).sum()
    print(f"[task {args.task_id}] Weights: {n_high} high-load points (≥{w_thresh:.0f} MWh) "
          f"weighted {HIGH_LOAD_WEIGHT}x", flush=True)

    # Fit GAM with gridsearch
    print(f"[task {args.task_id}] Fitting GAM (gridsearch) ...", flush=True)
    gam = LinearGAM(build_gam_formula(X_tr_fit), max_iter=100)
    gam.gridsearch(X_tr_fit.values, y_tr_fit.values, weights=weights, progress=False)

    preds = gam.predict(X_te_eval.values)
    print(f"[task {args.task_id}] Done. n_test={len(preds)}", flush=True)

    # Save predictions
    out = pd.DataFrame(
        {"actual": y_te_eval.values, "predicted": preds},
        index=y_te_eval.index,
    )
    out["zone"]          = zone
    out["fold_idx"]      = fold_idx
    out["held_out_year"] = held_out_year

    out_path = os.path.join(
        args.outdir, f"gam_preds_zone{zone_idx}_fold{fold_idx}.parquet"
    )
    out.to_parquet(out_path)
    print(f"[task {args.task_id}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
