#!/usr/bin/env python
"""
MLP hyperparameter tuning worker — one PBS array task per zone (6 tasks).

For each zone:
  1. Random search over hidden_layer_sizes, alpha, learning_rate_init
     using 8-fold LOGO CV (leave-one-year-out) to score each config.
  2. Retrains a final model on ALL training years with the best config.
  3. Saves the model + scaler to disk (joblib).

Array layout (6 tasks):
    PBS_ARRAY_INDEX = zone_idx

Outputs per zone (to --outdir):
    mlp_tuned_zone{zone_idx}.joblib    — model, scaler, best params
    mlp_tuned_cv_zone{zone_idx}.csv   — RMSE per config × fold
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────
LRZ_IDS = ["MISO-0001", "MISO-0027", "MISO-0035", "MISO-0004", "MISO-0006", "MISO-8910"]
YEARS   = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
SEED    = 42
TEST_YEARS  = [2016, 2021, 2024]
TRAIN_YEARS = sorted(set(YEARS) - set(TEST_YEARS))

MODEL_FREQ = "6H"

EXCLUDE_EXTREMES   = False
EXTREME_PERCENTILE = 99

HOLIDAYS       = ["memorial_day", "july_4th", "labor_day", "christmas", "new_years"]
HOLIDAY_WINDOW = 0
CDD_BASE       = 18.0  # °C

GAM_COLS = [
    "cdd", "hdd",
    "cdd_lag6",  "hdd_lag6",
    "cdd_lag12", "hdd_lag12",
    "cdd_lag18", "hdd_lag18",
    "cdd_lag24", "hdd_lag24",
    "hour", "month", "day_of_week", "year",
    "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
]

# ── Hyperparameter search space ───────────────────────────────────────────────
N_ITER = 25   # number of random configs to try

HIDDEN_SIZES = [
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (256, 128),
    (256, 128, 64),
    (256, 128, 64, 32),
]
ALPHA_LOG_RANGE = (-5, -1)    # 10^-5 to 10^-1
LR_LOG_RANGE    = (-4, -2)    # 10^-4 to 10^-2

# Fixed MLP settings (not tuned)
FIXED_PARAMS = dict(
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=SEED,
)


# ── Hyperparameter sampling ───────────────────────────────────────────────────
def sample_configs(n, seed):
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({
            "hidden_layer_sizes": HIDDEN_SIZES[rng.integers(len(HIDDEN_SIZES))],
            "alpha":              float(10 ** rng.uniform(*ALPHA_LOG_RANGE)),
            "learning_rate_init": float(10 ** rng.uniform(*LR_LOG_RANGE)),
        })
    return configs


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MLP tuning worker")
    parser.add_argument(
        "--task-id", type=int,
        default=int(os.environ.get("PBS_ARRAY_INDEX",
                    os.environ.get("SLURM_ARRAY_TASK_ID", 0))),
        help="Zone index (0–5)",
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--outdir",   default="mlp_tuned_results")
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


# ── Feature engineering ──────────────────────────────────────────────────────
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


# ── Data loading ─────────────────────────────────────────────────────────────
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
    return temp_data.loc[valid_rows], load_wide.loc[valid_rows]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    zone_idx = args.task_id
    if zone_idx >= len(LRZ_IDS):
        sys.exit(f"task_id {zone_idx} out of range (max {len(LRZ_IDS) - 1})")

    zone = LRZ_IDS[zone_idx]
    print(f"[zone {zone_idx}] zone={zone}", flush=True)

    train_years = TRAIN_YEARS
    print(f"[zone {zone_idx}] train_years={train_years}", flush=True)
    print(f"[zone {zone_idx}] test_years={TEST_YEARS}", flush=True)

    print(f"[zone {zone_idx}] Loading data ...", flush=True)
    temp_data, load_wide = load_data(args.data_dir)

    train_mask = temp_data.index.year.isin(train_years)
    temp = temp_data.loc[train_mask]
    load = load_wide.loc[train_mask]

    X_all  = build_features_raw(temp[zone]).dropna()
    y_all  = load[zone].loc[X_all.index].astype(float)
    groups = pd.Series(X_all.index.year, index=X_all.index)

    cv_splits = list(LeaveOneGroupOut().split(X_all, y_all, groups))
    print(f"[zone {zone_idx}] n_samples={len(y_all)}, n_folds={len(cv_splits)}", flush=True)

    # Scale on all training data (scaler is fixed across configs for speed)
    scaler   = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all[GAM_COLS].values)

    # ── Random search ─────────────────────────────────────────────────────────
    configs = sample_configs(N_ITER, SEED)
    print(f"[zone {zone_idx}] Starting random search: {N_ITER} configs × "
          f"{len(cv_splits)} folds ...", flush=True)

    rows = []
    for cfg_idx, params in enumerate(configs):
        fold_rmses = []
        for fold_idx, (tr_idx, te_idx) in enumerate(cv_splits):
            X_tr_f = X_all_sc[tr_idx]
            y_tr_f = y_all.values[tr_idx]
            X_te_f = X_all_sc[te_idx]
            y_te_f = y_all.values[te_idx]

            if EXCLUDE_EXTREMES:
                thresh = np.percentile(y_tr_f, EXTREME_PERCENTILE)
                mask   = y_tr_f <= thresh
                X_tr_f = X_tr_f[mask]
                y_tr_f = y_tr_f[mask]

            model = MLPRegressor(**params, **FIXED_PARAMS)
            model.fit(X_tr_f, y_tr_f)
            rmse = root_mean_squared_error(y_te_f, model.predict(X_te_f))
            fold_rmses.append(rmse)

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))
        rows.append({
            "config_idx":        cfg_idx,
            "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
            "alpha":              params["alpha"],
            "learning_rate_init": params["learning_rate_init"],
            "mean_rmse":          mean_rmse,
            "std_rmse":           std_rmse,
            **{f"fold_{i}_rmse": fold_rmses[i] for i in range(len(fold_rmses))},
        })
        print(f"[zone {zone_idx}] config {cfg_idx+1:2d}/{N_ITER}  "
              f"layers={params['hidden_layer_sizes']}  "
              f"alpha={params['alpha']:.2e}  "
              f"lr={params['learning_rate_init']:.2e}  "
              f"→ RMSE={mean_rmse:.2f} ± {std_rmse:.2f}", flush=True)

    # ── Select best config ────────────────────────────────────────────────────
    cv_df      = pd.DataFrame(rows).sort_values("mean_rmse")
    best_row   = cv_df.iloc[0]
    best_params = configs[int(best_row["config_idx"])]

    print(f"\n[zone {zone_idx}] Best config: {best_params}  "
          f"RMSE={best_row['mean_rmse']:.2f}", flush=True)

    cv_path = os.path.join(args.outdir, f"mlp_tuned_cv_zone{zone_idx}.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"[zone {zone_idx}] CV results → {cv_path}", flush=True)

    # ── Retrain final model on all training data ──────────────────────────────
    print(f"[zone {zone_idx}] Retraining final model on all {len(y_all)} samples ...",
          flush=True)

    X_final = X_all_sc
    y_final = y_all.values

    if EXCLUDE_EXTREMES:
        thresh  = np.percentile(y_final, EXTREME_PERCENTILE)
        mask    = y_final <= thresh
        X_final = X_final[mask]
        y_final = y_final[mask]
        print(f"[zone {zone_idx}] Extremes excluded: n={mask.sum()} (≤{thresh:.0f} MWh)",
              flush=True)

    final_model = MLPRegressor(**best_params, **FIXED_PARAMS)
    final_model.fit(X_final, y_final)
    print(f"[zone {zone_idx}] Converged in {final_model.n_iter_} iterations", flush=True)

    # ── Save model ────────────────────────────────────────────────────────────
    bundle = {
        "model":       final_model,
        "scaler":      scaler,
        "best_params": best_params,
        "zone":        zone,
        "zone_idx":    zone_idx,
        "train_years": train_years,
        "gam_cols":    GAM_COLS,
    }
    model_path = os.path.join(args.outdir, f"mlp_tuned_zone{zone_idx}.joblib")
    joblib.dump(bundle, model_path)
    print(f"[zone {zone_idx}] Model saved → {model_path}", flush=True)


if __name__ == "__main__":
    main()
