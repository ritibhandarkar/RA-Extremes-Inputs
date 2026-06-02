#!/usr/bin/env python
"""
Lasso CV worker — one SLURM array task per (zone, fold).

Array layout (48 tasks, 6 zones × 8 folds):
    SLURM_ARRAY_TASK_ID = zone_idx * N_FOLDS + fold_idx

Outputs one parquet per task to --outdir:
    lasso_preds_zone{zone_idx}_fold{fold_idx}.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# ── Constants (must match notebook exactly) ──────────────────────────────────
LRZ_IDS = ["MISO-0001", "MISO-0027", "MISO-0035", "MISO-0004", "MISO-0006", "MISO-8910"]
YEARS   = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
SEED    = 42
N_FOLDS = 8   # leave-one-year-out over the 8 training years

# ── Sample weights ───────────────────────────────────────────────────────────
# Top HIGH_LOAD_WEIGHT_PERCENTILE% of training load values receive HIGH_LOAD_WEIGHT
# weight; all others receive weight 1.
HIGH_LOAD_WEIGHT_PERCENTILE = 90
HIGH_LOAD_WEIGHT            = 5

# ── Toggle: exclude top-percentile load hours from training ───────────────────
# When True: train on hours ≤ p{EXTREME_PERCENTILE}, evaluate on hours > p{EXTREME_PERCENTILE}
EXCLUDE_EXTREMES   = False
EXTREME_PERCENTILE = 99


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Lasso CV worker")
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
        "--outdir", default="lasso_results",
        help="Output directory for per-fold prediction parquets",
    )
    return parser.parse_args()


# ── Feature engineering ──────────────────────────────────────────────────────
def build_features(temp_series):
    """Build Lasso feature matrix for a single LRZ temperature series."""
    t = temp_series

    X = pd.DataFrame(index=t.index)

    # Base temperature features
    t = t - t.mean() # centered to avoid issues

    X["temp"]         = t
    X["temp_squared"] = t ** 2
    # X["temp_cubed"]   = t ** 3

    # # Lags
    # X["temp_lag1"]  = t.shift(1)
    # X["temp_lag24"] = t.shift(24)

    # Hour dummies (hour 0 = reference)
    hour_dummies = pd.get_dummies(
        t.index.hour, prefix="hour", drop_first=True
    ).set_index(t.index).astype(int)

    # # 10°F daily-average temperature bin dummies
    # daily_avg_f = t_f.resample("D").mean().reindex(t.index, method="ffill")
    # bin_edges = np.arange(
    #     np.floor(daily_avg_f.min() / 10) * 10,
    #     np.ceil(daily_avg_f.max()  / 10) * 10 + 10,
    #     10,
    # )
    # temp_bin_raw    = pd.cut(daily_avg_f, bins=bin_edges, labels=False)
    # temp_bin_dummies = pd.get_dummies(
    #     temp_bin_raw, prefix="tbin", drop_first=True
    # ).set_index(t.index).astype(int)

    # # Hour × temp-bin interactions (caused convergence issues: temp_bin is
    # # constant within each day, making these columns near-collinear)
    # interaction_frames = []
    # for h_col in hour_dummies.columns:
    #     for b_col in temp_bin_dummies.columns:
    #         s = (hour_dummies[h_col] * temp_bin_dummies[b_col]).rename(
    #             f"{h_col}_x_{b_col}"
    #         )
    #         interaction_frames.append(s)
    # interactions = pd.concat(interaction_frames, axis=1)

    # Hour × month interactions (month varies day-to-day, well-conditioned)
    month_dummies = pd.get_dummies(
        t.index.month, prefix="month", drop_first=True
    ).set_index(t.index).astype(int)

    interaction_frames = []
    for h_col in hour_dummies.columns:
        for m_col in month_dummies.columns:
            s = (hour_dummies[h_col] * month_dummies[m_col]).rename(
                f"{h_col}_x_{m_col}"
            )
            interaction_frames.append(s)
    interactions = pd.concat(interaction_frames, axis=1)

    X = pd.concat([X, interactions], axis=1)

    X["weekend"] = (t.index.day_of_week > 4).astype(int)
    X["year"]    = t.index.year

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
    temp = temp_data.loc[train_mask]
    load = load_wide.loc[train_mask]

    # Build features for this zone
    X_all = build_features(temp[zone]).dropna()
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

    X_tr = X_all.iloc[train_idx]
    X_te = X_all.iloc[test_idx]
    y_tr = y_all.iloc[train_idx]
    y_te = y_all.iloc[test_idx]

    # Optionally restrict training and evaluation to extremes
    if EXCLUDE_EXTREMES:
        train_thresh = np.percentile(y_tr, EXTREME_PERCENTILE)
        X_tr_fit = X_tr[y_tr <= train_thresh]
        y_tr_fit = y_tr[y_tr <= train_thresh]

        test_thresh  = np.percentile(y_te, EXTREME_PERCENTILE)
        extreme_mask = y_te > test_thresh
        X_te_eval    = X_te[extreme_mask]
        y_te_eval    = y_te[extreme_mask]
        print(f"[task {args.task_id}] Extremes mode: "
              f"train n={len(y_tr_fit)} (≤{train_thresh:.0f} MWh), "
              f"eval n={len(y_te_eval)} (>{test_thresh:.0f} MWh)", flush=True)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr
        X_te_eval, y_te_eval = X_te, y_te

    # Build sample weights: top HIGH_LOAD_WEIGHT_PERCENTILE% of training y → HIGH_LOAD_WEIGHT
    w_thresh  = np.percentile(y_tr_fit, HIGH_LOAD_WEIGHT_PERCENTILE)
    weights   = np.where(y_tr_fit.values >= w_thresh, HIGH_LOAD_WEIGHT, 1.0)
    n_high    = (y_tr_fit.values >= w_thresh).sum()
    print(f"[task {args.task_id}] Weights: {n_high} high-load points (≥{w_thresh:.0f} MWh) "
          f"weighted {HIGH_LOAD_WEIGHT}x", flush=True)

    # Inner LOGO CV for alpha selection (same as notebook)
    inner_groups = pd.Series(X_tr_fit.index.year, index=X_tr_fit.index)
    inner_logo   = LeaveOneGroupOut()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(max_iter=5000, tol=1e-3)),
    ])
    param_grid = {"lasso__alpha": np.logspace(-3, 2, 50)}

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=inner_logo,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    print(f"[task {args.task_id}] Fitting Lasso (GridSearchCV, {len(param_grid['lasso__alpha'])} alphas × {inner_logo.get_n_splits(groups=inner_groups)} inner folds) ...", flush=True)
    grid.fit(X_tr_fit, y_tr_fit, groups=inner_groups,
             lasso__sample_weight=weights)

    best_alpha = grid.best_estimator_.named_steps["lasso"].alpha
    print(f"[task {args.task_id}] Best alpha: {best_alpha:.4g}", flush=True)

    preds = grid.predict(X_te_eval)
    rmse  = root_mean_squared_error(y_te_eval, preds)
    print(f"[task {args.task_id}] Done. RMSE={rmse:,.1f}  n_test={len(preds)}", flush=True)

    # Save predictions
    out = pd.DataFrame(
        {"actual": y_te_eval.values, "predicted": preds},
        index=y_te_eval.index,
    )
    out["zone"]          = zone
    out["fold_idx"]      = fold_idx
    out["held_out_year"] = held_out_year
    out["best_alpha"]    = best_alpha

    out_path = os.path.join(
        args.outdir, f"lasso_preds_zone{zone_idx}_fold{fold_idx}.parquet"
    )
    out.to_parquet(out_path)
    print(f"[task {args.task_id}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
