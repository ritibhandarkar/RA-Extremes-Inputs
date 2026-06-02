#!/usr/bin/env python
"""
Tabular ResNet CV worker — one PBS array task per (zone, fold).

Array layout (48 tasks, 6 zones × 8 folds):
    PBS_ARRAY_INDEX = zone_idx * N_FOLDS + fold_idx

Outputs one parquet per task to --outdir:
    resnet_preds_zone{zone_idx}_fold{fold_idx}.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# ── Constants (must match notebook exactly) ──────────────────────────────────
LRZ_IDS = ["MISO-0001", "MISO-0027", "MISO-0035", "MISO-0004", "MISO-0006", "MISO-8910"]
YEARS   = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
SEED    = 42
N_FOLDS = 8

# ── Toggle: exclude top-percentile load hours from training ───────────────────
EXCLUDE_EXTREMES   = True
EXTREME_PERCENTILE = 99

# ── Holiday window settings ───────────────────────────────────────────────────
HOLIDAYS       = ["memorial_day", "july_4th", "labor_day", "christmas", "new_years"]
HOLIDAY_WINDOW = 2

CDD_BASE = 18.0  # °C

GAM_COLS = [
    "cdd", "hdd",
    "cdd_lag1", "hdd_lag1",
    "cdd_lag24", "hdd_lag24",
    "hour", "month", "day_of_week", "year",
    "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
]

# ── ResNet hyperparameters ────────────────────────────────────────────────────
HIDDEN_DIM  = 256
N_BLOCKS    = 4
DROPOUT     = 0.1
BATCH_SIZE  = 512
LR          = 1e-3
MAX_EPOCHS  = 300
PATIENCE    = 20       # early stopping patience (epochs)
VAL_FRAC    = 0.1      # fraction of training set held out for early stopping


# ── Model ─────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TabularResNet(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, X_tr, y_tr, device, task_id):
    """Train with Adam + early stopping on a held-out validation split."""
    rng = np.random.default_rng(SEED)
    n_val = max(1, int(len(y_tr) * VAL_FRAC))
    val_idx = rng.choice(len(y_tr), n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    def to_tensors(X, y):
        return (
            torch.tensor(X.astype(np.float64), dtype=torch.float32, device=device),
            torch.tensor(y.astype(np.float64), dtype=torch.float32, device=device),
        )

    X_t, y_t = to_tensors(X_tr[tr_idx],  y_tr[tr_idx])
    X_v, y_v = to_tensors(X_tr[val_idx], y_tr[val_idx])

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_rmse = loss_fn(model(X_v), y_v).item() ** 0.5

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[task {task_id}] Early stop at epoch {epoch} "
                      f"(best val RMSE={best_val:.2f})", flush=True)
                break

    model.load_state_dict(best_state)
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Tabular ResNet CV worker")
    parser.add_argument(
        "--task-id", type=int,
        default=int(os.environ.get("PBS_ARRAY_INDEX",
                    os.environ.get("SLURM_ARRAY_TASK_ID", 0))),
        help="Flat task index",
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--outdir",   default="resnet_results")
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
    t   = temp_series
    t_f = t * 9 / 5 + 32

    X = pd.DataFrame(index=t.index)
    X["cdd"]       = np.maximum(0.0, t - CDD_BASE)
    X["hdd"]       = np.maximum(0.0, CDD_BASE - t)
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
    return temp_data.loc[valid_rows], load_wide.loc[valid_rows]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    zone_idx = args.task_id // N_FOLDS
    fold_idx = args.task_id  % N_FOLDS

    if zone_idx >= len(LRZ_IDS):
        sys.exit(f"task_id {args.task_id} out of range (max {len(LRZ_IDS) * N_FOLDS - 1})")

    zone = LRZ_IDS[zone_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task {args.task_id}] zone={zone} fold={fold_idx} device={device}", flush=True)

    rng = np.random.default_rng(SEED)
    train_years = sorted(rng.choice(YEARS, N_FOLDS, replace=False).tolist())
    print(f"[task {args.task_id}] train_years={train_years}", flush=True)

    print(f"[task {args.task_id}] Loading data ...", flush=True)
    temp_data, load_wide = load_data(args.data_dir)

    train_mask = temp_data.index.year.isin(train_years)
    temp = temp_data.loc[train_mask]
    load = load_wide.loc[train_mask]

    X_all  = build_features_raw(temp[zone]).dropna()
    y_all  = load[zone].loc[X_all.index]
    groups = pd.Series(X_all.index.year, index=X_all.index)

    splits = list(LeaveOneGroupOut().split(X_all, y_all, groups))
    if fold_idx >= len(splits):
        sys.exit(f"fold_idx {fold_idx} >= available folds ({len(splits)})")

    train_idx, test_idx = splits[fold_idx]
    held_out_year = int(groups.iloc[test_idx[0]])
    print(f"[task {args.task_id}] Held-out year: {held_out_year}", flush=True)

    X_tr = X_all[GAM_COLS].iloc[train_idx]
    X_te = X_all[GAM_COLS].iloc[test_idx]
    y_tr = y_all.iloc[train_idx]
    y_te = y_all.iloc[test_idx]

    if EXCLUDE_EXTREMES:
        train_thresh = np.percentile(y_tr, EXTREME_PERCENTILE)
        X_tr = X_tr[y_tr <= train_thresh]
        y_tr = y_tr[y_tr <= train_thresh]
        print(f"[task {args.task_id}] Extremes mode: train n={len(y_tr)} "
              f"(≤{train_thresh:.0f} MWh)", flush=True)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr.values)
    X_te_sc  = scaler.transform(X_te.values)

    torch.manual_seed(SEED)
    model = TabularResNet(n_features=X_tr_sc.shape[1]).to(device)
    print(f"[task {args.task_id}] Fitting ResNet "
          f"({N_BLOCKS} blocks, dim={HIDDEN_DIM}) ...", flush=True)

    model = train(model, X_tr_sc, y_tr.values, device, args.task_id)

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te_sc, dtype=torch.float32, device=device)
        preds  = model(X_te_t).cpu().numpy()

    print(f"[task {args.task_id}] Done. n_test={len(preds)}", flush=True)

    out = pd.DataFrame(
        {"actual": y_te.values, "predicted": preds},
        index=y_te.index,
    )
    out["zone"]          = zone
    out["fold_idx"]      = fold_idx
    out["held_out_year"] = held_out_year

    out_path = os.path.join(
        args.outdir, f"resnet_preds_zone{zone_idx}_fold{fold_idx}.parquet"
    )
    out.to_parquet(out_path)
    print(f"[task {args.task_id}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
