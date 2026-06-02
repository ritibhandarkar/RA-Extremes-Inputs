"""
Generate MISO solar capacity factor profiles from ERA5 reanalysis data.
Intended to be run as a SLURM job array — one job per year.

Usage:  python generate_solar_profiles_era5.py --year 2020
Output: /glade/work/rbhandarkar/Inputs/Solar Profiles/ERA5/{profile_name}_{year}.csv

Available years: 2015–2025 (limited by ssrd coverage).
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import KDTree

# ── Configuration ──────────────────────────────────────────────────────────────
ERA5_DIR   = Path("/glade/work/rbhandarkar/ERA5")
OUTPUT_DIR = Path("/glade/work/rbhandarkar/Inputs/Solar Profiles/ERA5")
PUDL_CSV   = Path("Generator data/Processed_PUDL_data_byLRZ.csv")

# ── File path helpers ──────────────────────────────────────────────────────────
def era5_path(var, year):
    return ERA5_DIR / f"miso_{var}_{year}.nc"

# ── Physics ────────────────────────────────────────────────────────────────────
def calculate_cfs(fsds, u10, trefht_c):
    """
    fsds      : surface solar radiation in W/m² (ssrd / 3600)
    u10       : 10-m wind speed in m/s
    trefht_c  : 2-m air temperature in °C
    """
    c1, c2, c3, c4 = 4.3, 0.943, 0.028, 1.528
    g        = -0.005
    t_stc    = 25    # °C
    rsds_stc = 1000  # W/m²
    cell_temp = c1 + c2 * trefht_c + c3 * fsds + c4 * u10
    pr = 1 + g * (cell_temp - t_stc)
    cf = pr * (fsds / rsds_stc)
    return np.clip(cf, 0, 1)

# ── PUDL loading ───────────────────────────────────────────────────────────────
def load_solar_plants(pudl_csv):
    pudl = pd.read_csv(pudl_csv, index_col=0)

    conditions = [
        (pudl["BA"] == "MISO-0001"),
        (pudl["BA"] == "MISO-0027") & (pudl["state"] == "WI"),
        (pudl["BA"] == "MISO-0027") & (pudl["state"] == "MI"),
        (pudl["BA"] == "MISO-0035") & pudl["state"].isin(["IA", "MN", "IL"]),
        (pudl["BA"] == "MISO-0035"),
        (pudl["BA"] == "MISO-0004"),
        (pudl["BA"] == "MISO-0006"),
        (pudl["BA"] == "MISO-8910") & (pudl["state"] == "AR"),
        (pudl["BA"] == "MISO-8910") & (pudl["state"].isin(["LA", "TX"])),
        (pudl["BA"] == "MISO-8910") & (pudl["state"] == "MS"),
    ]
    choices = ["01", "02", "07", "03", "05", "04", "06", "08", "09", "10"]
    pudl["LRZ"] = np.select(conditions, choices, default=None)

    return (pudl[pudl["fuel_type_code_pudl"] == "solar"]
            [["latitude", "longitude", "LRZ"]]
            .dropna(subset=["latitude", "longitude", "LRZ"])
            .copy())

# ── Profile assignment (ERA5 regular grid) ────────────────────────────────────
def assign_profiles(solar, era5_lat_1d, era5_lon_1d):
    """
    Map each plant to its nearest ERA5 grid cell, then number unique grid
    points within each LRZ (N→S, W→E).  ERA5 longitudes are -180/180 — no
    % 360 conversion needed.
    Returns one row per unique profile.
    """
    lon_2d, lat_2d = np.meshgrid(era5_lon_1d, era5_lat_1d)
    lat_flat = lat_2d.flatten()
    lon_flat = lon_2d.flatten()

    tree = KDTree(np.column_stack([lat_flat, lon_flat]))
    _, flat_indices = tree.query(
        np.column_stack([solar["latitude"].values, solar["longitude"].values])
    )

    solar = solar.copy()
    solar["flat_idx"]  = flat_indices
    solar["era5_lat"]  = lat_flat[flat_indices]
    solar["era5_lon"]  = lon_flat[flat_indices]

    chunks = []
    for _, group in solar.groupby("LRZ"):
        pts = (group[["flat_idx", "era5_lat", "era5_lon"]]
               .drop_duplicates("flat_idx")
               .sort_values(["era5_lat", "era5_lon"], ascending=[False, True])
               .reset_index(drop=True))
        pts["grid_num"] = range(1, len(pts) + 1)
        chunks.append(group.merge(pts[["flat_idx", "grid_num"]], on="flat_idx", how="left"))
    solar = pd.concat(chunks, ignore_index=True)

    solar["profile_name"] = (
        "MISO_" + solar["LRZ"] + "_solar_"
        + solar["grid_num"].apply(lambda n: f"{int(n):03d}")
    )

    return (solar[["profile_name", "LRZ", "flat_idx", "era5_lat", "era5_lon"]]
            .drop_duplicates("profile_name")
            .reset_index(drop=True))

# ── Per-year worker ────────────────────────────────────────────────────────────
def process_year(year, profiles_list, unique_flats, flat_to_pos, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_flat(var):
        ds    = xr.open_dataset(era5_path(var, year))
        name  = list(ds.data_vars)[0]
        n_t   = ds.sizes["valid_time"]
        vals  = ds[name].values.reshape(n_t, -1)[:, unique_flats]
        times = ds["valid_time"].values
        ds.close()
        return vals, times

    # ssrd is accumulated J/m² per hour → divide by 3600 for W/m²
    ssrd_mem, times = load_flat("ssrd")
    fsds_mem        = ssrd_mem / 3600.0

    u10_mem,  _ = load_flat("u10")
    v10_mem,  _ = load_flat("v10")
    ws_mem      = np.sqrt(u10_mem ** 2 + v10_mem ** 2)
    t2m_mem,  _ = load_flat("t2m")
    trefht_mem  = t2m_mem - 273.15   # K → °C

    for profile_name, flat_idx in profiles_list:
        pos = flat_to_pos[flat_idx]
        cf  = calculate_cfs(fsds_mem[:, pos], ws_mem[:, pos], trefht_mem[:, pos])
        pd.DataFrame({"time": times, "cf": cf}).to_csv(
            output_dir / f"{profile_name}_{year}.csv", index=False
        )

    return year

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                        help="Year to process (2015–2025)")
    args = parser.parse_args()

    if args.year < 2015 or args.year > 2025:
        raise ValueError("ERA5 ssrd data only available 2015–2025")

    solar = load_solar_plants(PUDL_CSV)
    print(f"Loaded {len(solar)} solar plants")

    ref_ds      = xr.open_dataset(era5_path("ssrd", args.year))
    era5_lat_1d = ref_ds["latitude"].values
    era5_lon_1d = ref_ds["longitude"].values
    ref_ds.close()

    profiles = assign_profiles(solar, era5_lat_1d, era5_lon_1d)
    print(f"Assigned {len(profiles)} unique profiles")

    unique_flats  = np.unique(profiles["flat_idx"].values).tolist()
    flat_to_pos   = {int(f): pos for pos, f in enumerate(unique_flats)}
    profiles_list = list(zip(
        profiles["profile_name"],
        profiles["flat_idx"].tolist(),
    ))

    print(f"Processing year {args.year}...")
    process_year(args.year, profiles_list, unique_flats, flat_to_pos, str(OUTPUT_DIR))
    print(f"Done — output in {OUTPUT_DIR}")
