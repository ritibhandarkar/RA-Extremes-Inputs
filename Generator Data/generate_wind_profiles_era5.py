"""
Generate MISO wind capacity factor profiles from ERA5 reanalysis data.
Intended to be run as a SLURM job array — one job per year.

Usage:  python generate_wind_profiles_era5.py --year 2020
Output: /glade/work/rbhandarkar/Inputs/Wind Profiles/ERA5/{profile_name}_{year}.csv

Available years: 2015–2025 (limited by sp and v10 coverage).
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import KDTree

# ── Configuration ──────────────────────────────────────────────────────────────
ERA5_DIR   = Path("/glade/work/rbhandarkar/ERA5")
OUTPUT_DIR = Path("/glade/work/rbhandarkar/Inputs/Wind Profiles/ERA5")
PUDL_CSV   = Path("Generator data/Processed_PUDL_data_byLRZ.csv")

CLASS2_LRZS = {"01", "02", "04", "05", "06"}

# ── File path helpers ──────────────────────────────────────────────────────────
def era5_path(var, year):
    return ERA5_DIR / f"miso_{var}_{year}.nc"

# ── Physics ────────────────────────────────────────────────────────────────────
def huss_from_d2m(d2m_k, sp_pa):
    """Specific humidity from ERA5 dewpoint (K) and surface pressure (Pa).
    VP (hPa) = 6.112 * exp(17.67 * Td_C / (Td_C + 243.5))
    HUSS = 0.622 * VP / (0.01 * SP - 0.378 * VP)
    """
    td_c = d2m_k - 273.15
    vp   = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))   # hPa
    return 0.622 * vp / (0.01 * sp_pa - 0.378 * vp)

def calculate_cfs(t2m, huss, ws, sp, power_curve):
    """Moisture- and density-corrected capacity factor via power curve lookup."""
    r     = 287.05                                           # J kg⁻¹ K⁻¹
    rho_d = sp / (r * t2m)
    rho_m = rho_d * (1 + huss) / (1 + 1.609 * huss)
    w100  = ws * (100 / 10) ** (1 / 7) * (rho_m / 1.255) ** (1 / 3)
    return np.interp(w100, power_curve["Wind Speed (m/s)"], power_curve["Turbine Output"])

# ── PUDL loading ───────────────────────────────────────────────────────────────
def load_wind_plants(pudl_csv):
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

    return (pudl[pudl["fuel_type_code_pudl"] == "wind"]
            [["latitude", "longitude", "LRZ"]]
            .dropna(subset=["latitude", "longitude", "LRZ"])
            .copy())

# ── Profile assignment (ERA5 regular grid) ────────────────────────────────────
def assign_profiles(wind, era5_lat_1d, era5_lon_1d):
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
        np.column_stack([wind["latitude"].values, wind["longitude"].values])
    )

    wind = wind.copy()
    wind["flat_idx"]  = flat_indices
    wind["era5_lat"]  = lat_flat[flat_indices]
    wind["era5_lon"]  = lon_flat[flat_indices]

    chunks = []
    for _, group in wind.groupby("LRZ"):
        pts = (group[["flat_idx", "era5_lat", "era5_lon"]]
               .drop_duplicates("flat_idx")
               .sort_values(["era5_lat", "era5_lon"], ascending=[False, True])
               .reset_index(drop=True))
        pts["grid_num"] = range(1, len(pts) + 1)
        chunks.append(group.merge(pts[["flat_idx", "grid_num"]], on="flat_idx", how="left"))
    wind = pd.concat(chunks, ignore_index=True)

    wind["profile_name"] = (
        "MISO_" + wind["LRZ"] + "_wind_"
        + wind["grid_num"].apply(lambda n: f"{int(n):03d}")
    )

    return (wind[["profile_name", "LRZ", "flat_idx", "era5_lat", "era5_lon"]]
            .drop_duplicates("profile_name")
            .reset_index(drop=True))

# ── Per-year worker ────────────────────────────────────────────────────────────
def process_year(year, profiles_list, lrz_power_curves, unique_flats, flat_to_pos, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_flat(var):
        ds   = xr.open_dataset(era5_path(var, year))
        name = list(ds.data_vars)[0]          # e.g. "u10", "d2m", "sp", "t2m"
        n_t  = ds.sizes["valid_time"]
        vals = ds[name].values.reshape(n_t, -1)[:, unique_flats]
        times = ds["valid_time"].values
        ds.close()
        return vals, times

    u10_mem,  times = load_flat("u10")
    v10_mem,  _     = load_flat("v10")
    t2m_mem,  _     = load_flat("t2m")
    d2m_mem,  _     = load_flat("d2m")
    sp_mem,   _     = load_flat("ps")         # file is miso_ps_*.nc, var inside is "sp"

    ws_mem   = np.sqrt(u10_mem ** 2 + v10_mem ** 2)
    huss_mem = huss_from_d2m(d2m_mem, sp_mem)

    for profile_name, flat_idx, lrz in profiles_list:
        pos         = flat_to_pos[flat_idx]
        power_curve = lrz_power_curves[lrz]
        cf = calculate_cfs(
            t2m_mem[:, pos], huss_mem[:, pos],
            ws_mem[:, pos],  sp_mem[:, pos],
            power_curve,
        )
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
        raise ValueError("ERA5 sp and v10 data only available 2015–2025")

    wind = load_wind_plants(PUDL_CSV)
    print(f"Loaded {len(wind)} wind plants")

    # Use any available ERA5 file to get the grid
    ref_ds       = xr.open_dataset(era5_path("u10", args.year))
    era5_lat_1d  = ref_ds["latitude"].values
    era5_lon_1d  = ref_ds["longitude"].values
    ref_ds.close()

    curve_class2 = pd.read_csv("Generator data/wind_power_curve_class2.csv")
    curve_class3 = pd.read_csv("Generator data/wind_power_curve_class3.csv")
    lrz_power_curves = {
        **{lrz: curve_class2 for lrz in CLASS2_LRZS},
        **{lrz: curve_class3 for lrz in ("03", "07", "08", "09", "10")},
    }

    profiles = assign_profiles(wind, era5_lat_1d, era5_lon_1d)
    print(f"Assigned {len(profiles)} unique profiles")

    unique_flats  = np.unique(profiles["flat_idx"].values).tolist()
    flat_to_pos   = {int(f): pos for pos, f in enumerate(unique_flats)}
    profiles_list = list(zip(
        profiles["profile_name"],
        profiles["flat_idx"].tolist(),
        profiles["LRZ"].tolist(),
    ))

    print(f"Processing year {args.year}...")
    process_year(args.year, profiles_list, lrz_power_curves,
                 unique_flats, flat_to_pos, str(OUTPUT_DIR))
    print(f"Done — output in {OUTPUT_DIR}")
