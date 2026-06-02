"""
Generate MISO wind capacity factor profiles from CESM HR large ensemble data.
Intended to be run as a SLURM job array — one job per year.

Usage:  python generate_wind_profiles.py --year 2023
Output: Wind Profiles/{ensemble}_{rcp}/{profile_name}_{year}.csv
"""
import argparse
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
path_map = {"RCP85" : {"002" : "30-2006-2100.002",
            "003" : "31-2006-2100.003",
            "004" : "44-2006-2100.004",
            "005" : "44-2006-2100.005",
            "006" : "46-2006-2100.006",
            "007" : "46-2006-2100.007",
            "008" : "46-2006-2100.008",
            "009" : "46-2006-2100.009",
            "010" : "46-2006-2100.010"
            },
            "RCP60" : {"001" : "44-2006-2100.001",
            "002" : "44-2006-2100.002",
            "003" : "44-2006-2100.003",
            "004" : "46-2006-2100.004",
            "005" : "44-2006-2100.005",
            "006" : "46-2006-2100.006",
            "007" : "46-2006-2100.007",
            "008" : "46-2006-2100.008",
            "009" : "46-2006-2100.009",
            "010" : "46-2006-2100.010"}
            }
rcp_map = {"RCP85" : "d651009",
           "RCP60" : "d651008",
           }
PUDL_CSV   = Path(r"Generator data/Processed_PUDL_data_byLRZ.csv")

# ── File path helpers ──────────────────────────────────────────────────────────
def _nc_path(variable, h_stream, year, ensemble, rcp, cesm_base):
    fname = (
        f"b.e13.B{rcp}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[rcp][ensemble]}"
        f".cam.{h_stream}.{variable}.{year}010100-{year+1}010100.nc"
    )
    return cesm_base / fname

def qrefht_path(year, ensemble, rcp, cesm_base):   return _nc_path("QREFHT",   "h4", year, ensemble, rcp, cesm_base)
def u10_path(year, ensemble, rcp, cesm_base):    return _nc_path("U10",    "h2", year, ensemble, rcp, cesm_base)
def trefht_path(year, ensemble, rcp, cesm_base): return _nc_path("TREFHT", "h2", year, ensemble, rcp, cesm_base)
def ps_path(year, ensemble, rcp, cesm_base): return _nc_path("PS", "h2", year, ensemble, rcp, cesm_base)

# ── Physics ────────────────────────────────────────────────────────────────────
def calculate_cfs(trefht, qrefht, u10, ps, power_curve):
    r =  287.05 # J*kg^−1*K^−1
    rho_d = ps/(r*trefht)
    rho_m = rho_d*(1+qrefht)/(1+1.609*qrefht)

    w100_p = u10*(100/10)**(1/7)

    w100 = w100_p*(rho_m/1.255)**(1/3)

    return np.interp(w100, power_curve["Wind Speed (m/s)"], power_curve["Turbine Output"])


# ── PUDL loading ───────────────────────────────────────────────────────────────
def load_wind_plants(pudl_csv: Path) -> pd.DataFrame:
    """Return wind plants with LRZ assigned, dropping rows with missing lat/lon."""
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

    wind = (pudl[pudl["fuel_type_code_pudl"] == "wind"]
             [["latitude", "longitude", "LRZ"]]
             .dropna(subset=["latitude", "longitude", "LRZ"])
             .copy())
    return wind

# ── Profile assignment (requires CESM grid) ────────────────────────────────────
def assign_profiles(wind: pd.DataFrame, nc_lat: np.ndarray, nc_lon: np.ndarray) -> pd.DataFrame:
    """
    Map each plant to its nearest CESM ncol, then number unique ncol points
    within each LRZ (N→S, W→E by actual CESM coordinates).
    Returns one row per unique profile.
    """
    tree = KDTree(np.column_stack([nc_lat, nc_lon]))
    query_pts = np.column_stack([wind["latitude"].values, wind["longitude"].values % 360])
    _, ncol_indices = tree.query(query_pts)
    wind = wind.copy()
    wind["ncol_idx"]  = ncol_indices
    wind["cesm_lat"]  = nc_lat[ncol_indices]
    wind["cesm_lon"]  = nc_lon[ncol_indices]

    chunks = []
    for _, group in wind.groupby("LRZ"):
        pts = (group[["ncol_idx", "cesm_lat", "cesm_lon"]]
               .drop_duplicates("ncol_idx")
               .sort_values(["cesm_lat", "cesm_lon"], ascending=[False, True])
               .reset_index(drop=True))
        pts["grid_num"] = range(1, len(pts) + 1)
        chunks.append(group.merge(pts[["ncol_idx", "grid_num"]], on="ncol_idx", how="left"))
    wind = pd.concat(chunks, ignore_index=True)

    wind["profile_name"] = (
        "MISO_" + wind["LRZ"] + "_wind_"
        + wind["grid_num"].apply(lambda n: f"{int(n):03d}")
    )

    profiles = (wind[["profile_name", "LRZ", "ncol_idx", "cesm_lat", "cesm_lon"]]
                .drop_duplicates("profile_name")
                .reset_index(drop=True))
    return profiles

# ── Per-year worker ────────────────────────────────────────────────────────────
def process_year(year, profiles_list, lrz_power_curves, unique_ncols, ncol_to_pos, output_dir, ensemble, rcp, cesm_base):
    year_dir = Path(output_dir)
    year_dir.mkdir(parents=True, exist_ok=True)

    ps_ds   = xr.open_dataset(ps_path(year, ensemble, rcp, cesm_base))
    u10_ds    = xr.open_dataset(u10_path(year, ensemble, rcp, cesm_base))
    trefht_ds = xr.open_dataset(trefht_path(year, ensemble, rcp, cesm_base))
    qrefht_ds = xr.open_dataset(qrefht_path(year, ensemble, rcp, cesm_base))

    time_vals  = ps_ds["time"].values

    # .chunk() only on the float data variables — avoids object-dtype time coord issue
    qrefht_mem   = qrefht_ds["QREFHT"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values
    u10_mem    = u10_ds["U10"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values
    trefht_mem = trefht_ds["TREFHT"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values
    ps_mem = ps_ds["PS"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values


    qrefht_ds.close(); u10_ds.close(); trefht_ds.close(); ps_ds.close()

    for profile_name, ncol_idx, lrz in profiles_list:
        pos = ncol_to_pos[ncol_idx]
        power_curve = lrz_power_curves[lrz]
        cf = calculate_cfs(trefht_mem[:, pos], qrefht_mem[:, pos], u10_mem[:, pos], ps_mem[:, pos], power_curve)
        pd.DataFrame({"time": time_vals, "cf": cf}).to_csv(
        year_dir / f"{profile_name}_{year}.csv", index=False
        )

    return year

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Year to process (e.g. 2023)")
    parser.add_argument("--ensemble", type=str, required=True, help="Ensemble member")
    parser.add_argument("--rcp", type=str, required=True, help="RCP")
    args = parser.parse_args()

    wind = load_wind_plants(PUDL_CSV)
    print(f"Loaded {len(wind)} wind plants")

    ENSEMBLE   = args.ensemble
    RCP        = args.rcp
    label      = ENSEMBLE if RCP == "RCP60" else f"{int(ENSEMBLE) + 9:03d}"
    OUTPUT_DIR = Path(f"/glade/work/rbhandarkar/Inputs/Wind Profiles/{label}")
    CESM_BASE  = Path(f"/gdex/data/{rcp_map[RCP]}/b.e13.B{RCP}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[RCP][ENSEMBLE]}/atm/proc/tseries/hour_6")

    ref_ds = xr.open_dataset(ps_path(args.year, ENSEMBLE, RCP, CESM_BASE))
    nc_lat = ref_ds["lat"].values
    nc_lon = ref_ds["lon"].values
    ref_ds.close()

    curve_class2 = pd.read_csv("Generator data/wind_power_curve_class2.csv")
    curve_class3 = pd.read_csv("Generator data/wind_power_curve_class3.csv")
    lrz_power_curves = {
        **{lrz: curve_class2 for lrz in ("01", "02", "04", "05", "06")},
        **{lrz: curve_class3 for lrz in ("03", "07", "08", "09", "10")},
    }

    profiles = assign_profiles(wind, nc_lat, nc_lon)
    print(f"Assigned {len(profiles)} unique profiles")

    unique_ncols  = np.unique(profiles["ncol_idx"].values).tolist()
    ncol_to_pos   = {int(nc): pos for pos, nc in enumerate(unique_ncols)}
    profiles_list = list(zip(profiles["profile_name"], profiles["ncol_idx"].tolist(), profiles["LRZ"].tolist()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing year {args.year}...")
    process_year(args.year, profiles_list, lrz_power_curves, unique_ncols, ncol_to_pos, str(OUTPUT_DIR), ENSEMBLE, RCP, CESM_BASE)
    print("Done.")
