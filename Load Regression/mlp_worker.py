#!/usr/bin/env python
"""
MLP CV worker — one SLURM array task per (zone, fold).

Array layout (48 tasks, 6 zones × 8 folds):
    SLURM_ARRAY_TASK_ID = zone_idx * N_FOLDS + fold_idx

Outputs one parquet per task to --outdir:
    mlp_preds_zone{zone_idx}_fold{fold_idx}.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Constants (must match notebook exactly) ──────────────────────────────────
LRZ_IDS = ["MISO-0001", "MISO-0027", "MISO-0035", "MISO-0004", "MISO-0006", "MISO-8910"]
YEARS   = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
SEED    = 42
N_FOLDS = 8   # leave-one-year-out over the 8 training years

# ── Toggle: exclude top-percentile load hours from training ───────────────────
EXCLUDE_EXTREMES   = True
EXTREME_PERCENTILE = 99

MODEL_FREQ = "6H"

# ── Holiday window settings ───────────────────────────────────────────────────
HOLIDAYS       = ["memorial_day", "july_4th", "labor_day", "christmas", "new_years"]
HOLIDAY_WINDOW = 0  # days on each side

CDD_BASE = 18.0  # °C (≈ 65°F)

# Raw feature columns extracted from temperature series (same as GAM)
GAM_COLS =  [
    "cdd", "hdd",
    # "cdd_lag1", "hdd_lag1",
    "cdd_lag6", "hdd_lag6",
    "cdd_lag12", "hdd_lag12",
    "cdd_lag18", "hdd_lag18",
    "cdd_lag24", "hdd_lag24",
    "hour", "month", "day_of_week", "year",
    "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
]

# ── MLP hyperparameters ───────────────────────────────────────────────────────
MLP_HIDDEN_LAYERS = (128, 64, 32)
MLP_ACTIVATION    = "relu"
MLP_MAX_ITER      = 500
MLP_ALPHA         = 1e-4   # L2 weight decay
MLP_LR_INIT       = 1e-3


# ── Feature encoding ──────────────────────────────────────────────────────────
def encode_features(X_df):
    """
    Transform raw GAM_COLS into MLP-ready numeric features.

    All columns are passed through as raw integers/floats and z-scored by the
    caller. hour, month, and day_of_week are kept as integers rather than
    sin/cos pairs because load has multiple peaks (e.g. morning + evening) that
    a single sinusoid cannot represent.
    """
    return X_df[GAM_COLS].copy()


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MLP CV worker")
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
        "--outdir", default="mlp_results",
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


# ── Feature engineering ──────────────────────────────────────────────────────
def build_features_raw(temp_series):
    t   = temp_series
    t_f = t * 9 / 5 + 32

    X = pd.DataFrame(index=t.index)
    X["cdd"]      = np.maximum(0.0, t - CDD_BASE)
    X["hdd"]      = np.maximum(0.0, CDD_BASE - t)

    def create_lag(X, n):
        shift = n // 6 if MODEL_FREQ == "6H" else n
        X[f"cdd_lag{n}"] = X["cdd"].shift(shift)
        X[f"hdd_lag{n}"] = X["hdd"].shift(shift)
        return X

    X = create_lag(X, 6)
    X = create_lag(X, 12)
    X = create_lag(X, 18)
    X = create_lag(X, 24)

    # Fix bug where if year of data is missing, it will use hours from the year - 1
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

    if MODEL_FREQ == "6H":
        # resample to 6-hourly, backwards looking to match CESM
        # so 06:00 timestamp represents 0:00-06:00 average
        temp_data = temp_data.tz_localize("UTC").resample("6h", closed="right", label="right").mean()
        load_wide.index = pd.to_datetime(load_wide.index).tz_localize("Etc/GMT+5").tz_convert("UTC")
        load_wide = load_wide.resample("6h", closed="right", label="right").mean()

        # convert to DST
        temp_data.index = temp_data.index.tz_convert("America/Indiana/Indianapolis")
        load_wide.index  = load_wide.index.tz_convert("America/Indiana/Indianapolis")
    else: 
        # Align timezones (with daylight savings)
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

    # Optionally restrict training to non-extreme hours
    if EXCLUDE_EXTREMES:
        train_thresh = np.percentile(y_tr, EXTREME_PERCENTILE)
        X_tr_fit = X_tr[y_tr <= train_thresh]
        y_tr_fit = y_tr[y_tr <= train_thresh]
        print(f"[task {args.task_id}] Extremes mode: "
              f"train n={len(y_tr_fit)} (≤{train_thresh:.0f} MWh)", flush=True)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    X_te_eval, y_te_eval = X_te, y_te

    # Encode and scale features
    X_tr_enc = encode_features(X_tr_fit).values
    X_te_enc = encode_features(X_te_eval).values

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_te_scaled = scaler.transform(X_te_enc)

    # Fit MLP
    print(f"[task {args.task_id}] Fitting MLP {MLP_HIDDEN_LAYERS} ...", flush=True)
    mlp = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        activation=MLP_ACTIVATION,
        solver="adam",
        alpha=MLP_ALPHA,
        learning_rate_init=MLP_LR_INIT,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=SEED,
    )
    mlp.fit(X_tr_scaled, y_tr_fit.values)
    print(f"[task {args.task_id}] Converged after {mlp.n_iter_} iterations", flush=True)

    preds = mlp.predict(X_te_scaled)
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
        args.outdir, f"mlp_preds_zone{zone_idx}_fold{fold_idx}.parquet"
    )
    out.to_parquet(out_path)
    print(f"[task {args.task_id}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
