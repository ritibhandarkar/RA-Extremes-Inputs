"""
Generate MISO thermal derate profiles from CESM HR large ensemble data.
Intended to be run as a SLURM job array — one job per year.

Usage:  python generate_thermal_derates.py --year 2023
Output:  Thermal Derates/002/{year}/{profile_name}.csv
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
PUDL_CSV    = Path(r"Generator data/Processed_PUDL_data_byLRZ.csv")

# ── File path helpers ──────────────────────────────────────────────────────────
def _nc_path(variable, h_stream, year, ensemble, rcp, cesm_base):
    fname = (
        f"b.e13.B{rcp}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[rcp][ensemble]}"
        f".cam.{h_stream}.{variable}.{year}010100-{year+1}010100.nc"
    )
    return cesm_base / fname

def qrefht_path(year, ensemble, rcp, cesm_base):   return _nc_path("QREFHT",   "h4", year, ensemble, rcp, cesm_base)
def trefht_path(year, ensemble, rcp, cesm_base): return _nc_path("TREFHT", "h2", year, ensemble, rcp, cesm_base)
def ps_path(year, ensemble, rcp, cesm_base): return _nc_path("PS", "h2", year, ensemble, rcp, cesm_base)

# ── Physics ────────────────────────────────────────────────────────────────────
def frac_capacity(tech, cooling, trefht, qrefht, ps):
    es = 6.107*100*np.exp(17.27*(trefht-273.15)/(trefht-35.85))
    e = 0.622
    qsat = e*es/(ps-(1-e)*es)
    rh = qrefht/qsat*100 #convert to % RH
    trefht = (trefht-273.15)*9/5+32 # convert to F
    ps = ps*0.000145038 # convert to psia

    if tech in ["Natural Gas Steam Turbine", "Conventional Steam Coal"] and cooling == "RC":
        beta_T = 0.0107
        beta_RH = 0.0287
        beta_TRH = -0.000378
        alpha = 0.195
    elif tech == "Natural Gas Fired Combined Cycle" and cooling == "RC":
        beta_T = 0.0113
        beta_RH = 0.0266
        beta_TRH = -0.000336
        alpha = 0.12
    elif tech in ["Natural Gas Steam Turbine", "Conventional Steam Coal"] and cooling == "DC":
        beta_T = -0.0192
        beta_P = 0.0315
        alpha = 2.02
    elif tech == "Natural Gas Fired Combined Cycle" and cooling == "DC":
        beta_T = -0.0107
        beta_P = 0.0161
        alpha = 1.66
    elif tech == "Natural Gas Fired Combustion Turbine":
        frac_avail_capacity =-0.0083*trefht + 1.15
        return np.minimum(frac_avail_capacity, 1)
    else:
        raise ValueError
    
    if cooling == "RC":
        frac_avail_capacity = beta_T*trefht + beta_RH*rh + beta_TRH*(trefht*rh) + alpha
    elif cooling == "DC":
        frac_avail_capacity = beta_T*trefht + beta_P*ps + alpha
    else:
        raise TypeError
    return np.minimum(frac_avail_capacity, 1)


# ── PUDL loading ───────────────────────────────────────────────────────────────
def load_thermal_plants(pudl_csv: Path) -> pd.DataFrame:
    """Return thermal plants with LRZ assigned, dropping rows with missing lat/lon."""
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

    thermal_list = ["Natural Gas Fired Combustion Turbine", "Natural Gas Steam Turbine", "Natural Gas Fired Combined Cycle", "Conventional Steam Coal"]

    thermal = (pudl[pudl["technology_description"].isin(thermal_list)]
             [["latitude", "longitude", "LRZ", "plant_id_eia", "technology_description"]]
             .dropna(subset=["latitude", "longitude", "LRZ"])
             .copy())
    
    # Assign cooling types
    cooling_info = pd.read_parquet(r"Generator data/_core_eia860__cooling_equipment.parquet")
    
    thermal = thermal.merge(cooling_info[["plant_id_eia", "cooling_type_1"]], how="left", on="plant_id_eia")

    # Assume that any units that are not DC are RC
    thermal["cooling_type_1"] = thermal["cooling_type_1"].fillna("RC")
    thermal["cooling_type_1"] = np.select([(thermal["cooling_type_1"] != "DC"), (thermal["cooling_type_1"] == "DC")], ["RC", "DC"], default=thermal["cooling_type_1"])
    
    return thermal

# ── Profile assignment (requires CESM grid) ────────────────────────────────────
def assign_profiles(thermal: pd.DataFrame, nc_lat: np.ndarray, nc_lon: np.ndarray) -> pd.DataFrame:
    """
    Map each plant to its nearest CESM ncol, then number unique ncol points
    within each LRZ (N→S, W→E by actual CESM coordinates).
    Returns one row per unique profile.
    """
    tree = KDTree(np.column_stack([nc_lat, nc_lon]))
    query_pts = np.column_stack([thermal["latitude"].values, thermal["longitude"].values % 360])
    _, ncol_indices = tree.query(query_pts)
    thermal = thermal.copy()
    thermal["ncol_idx"]  = ncol_indices
    thermal["cesm_lat"]  = nc_lat[ncol_indices]
    thermal["cesm_lon"]  = nc_lon[ncol_indices]

    chunks = []
    for _, group in thermal.groupby("LRZ"):
        pts = (group[["ncol_idx", "cesm_lat", "cesm_lon"]]
               .drop_duplicates("ncol_idx")
               .sort_values(["cesm_lat", "cesm_lon"], ascending=[False, True])
               .reset_index(drop=True))
        pts["grid_num"] = range(1, len(pts) + 1)
        chunks.append(group.merge(pts[["ncol_idx", "grid_num"]], on="ncol_idx", how="left"))
    thermal = pd.concat(chunks, ignore_index=True)

    thermal["profile_name"] = "MISO_" + thermal["LRZ"] + "_" + thermal["technology_description"].str.replace(" ", "_") + "_" + thermal["cooling_type_1"] + "_" + thermal["grid_num"].apply(lambda n: f"{int(n):03d}")
    return thermal

# ── Per-year worker ────────────────────────────────────────────────────────────
def process_year(year, profiles_list, unique_ncols, ncol_to_pos, output_dir, ensemble, rcp, cesm_base):
    year_dir = Path(output_dir)
    year_dir.mkdir(parents=True, exist_ok=True)

    ps_ds   = xr.open_dataset(ps_path(year, ensemble, rcp, cesm_base))
    trefht_ds = xr.open_dataset(trefht_path(year, ensemble, rcp, cesm_base))
    qrefht_ds = xr.open_dataset(qrefht_path(year, ensemble, rcp, cesm_base))

    time_vals  = ps_ds["time"].values

    # .chunk() only on the float data variables — avoids object-dtype time coord issue
    qrefht_mem   = qrefht_ds["QREFHT"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values
    trefht_mem = trefht_ds["TREFHT"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values
    ps_mem = ps_ds["PS"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values


    qrefht_ds.close(); trefht_ds.close(); ps_ds.close()

    for profile_name, ncol_idx, tech, cooling in profiles_list:
        pos = ncol_to_pos[ncol_idx]
        cf = frac_capacity(tech, cooling, trefht_mem[:, pos], qrefht_mem[:, pos], ps_mem[:, pos])
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

    thermal = load_thermal_plants(PUDL_CSV)
    print(f"Loaded {len(thermal)} thermal plants")

    ENSEMBLE   = args.ensemble
    RCP        = args.rcp
    label      = ENSEMBLE if RCP == "RCP60" else f"{int(ENSEMBLE) + 9:03d}"
    OUTPUT_DIR = Path(f"/glade/work/rbhandarkar/Inputs/Thermal Derates/{label}")
    CESM_BASE  = Path(f"/gdex/data/{rcp_map[RCP]}/b.e13.B{RCP}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[RCP][ENSEMBLE]}/atm/proc/tseries/hour_6")

    ref_ds = xr.open_dataset(ps_path(args.year, ENSEMBLE, RCP, CESM_BASE))
    nc_lat = ref_ds["lat"].values
    nc_lon = ref_ds["lon"].values
    ref_ds.close()

    profiles = assign_profiles(thermal, nc_lat, nc_lon)
    print(f"Assigned {len(profiles)} unique profiles")

    unique_ncols  = np.unique(profiles["ncol_idx"].values).tolist()
    ncol_to_pos   = {int(nc): pos for pos, nc in enumerate(unique_ncols)}
    profiles_list = list(zip(profiles["profile_name"], profiles["ncol_idx"].tolist(), profiles["technology_description"], profiles["cooling_type_1"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing year {args.year}...")
    process_year(args.year, profiles_list, unique_ncols, ncol_to_pos, str(OUTPUT_DIR), ENSEMBLE, RCP, CESM_BASE)
    print("Done.")
