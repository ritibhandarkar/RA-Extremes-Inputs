"""
Generate MISO thermal derate profiles from CESM HR large ensemble data.
Intended to be run as a SLURM job array — one job per year.

Usage:  python generate_thermal_derates.py --year 2023
Output:  Thermal FORs/{ensemble}_{rcp}/{profile_name}_{year}.csv
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
FOR_TABLE = Path(r"./Generator data/temperature_dependent_outage_rates.csv")

# ── File path helpers ──────────────────────────────────────────────────────────
def _nc_path(variable, h_stream, year, ensemble, rcp, cesm_base):
    fname = (
        f"b.e13.B{rcp}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[rcp][ensemble]}"
        f".cam.{h_stream}.{variable}.{year}010100-{year+1}010100.nc"
    )
    return cesm_base / fname

def trefht_path(year, ensemble, rcp, cesm_base): return _nc_path("TREFHT", "h2", year, ensemble, rcp, cesm_base)

# ── Physics ────────────────────────────────────────────────────────────────────
def calculate_fors(tech, trefht, for_table):
    trefht_rounded = np.clip(5 * np.round(trefht / 5), -15, 35).astype(int)
    mapping = for_table.set_index("Temperature")[tech]
    return mapping[trefht_rounded].values


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

    thermal_list = ["Natural Gas Fired Combustion Turbine", 
                "Natural Gas Steam Turbine", 
                "Natural Gas Fired Combined Cycle", 
                "Conventional Steam Coal", 
                "Nuclear", 
                "Petroleum Liquids", 
                'Municipal Solid Waste',
                'Natural Gas Internal Combustion Engine', 
                'Other Gases',
                'Biomass',
                'Other Waste Biomass', 
                'Wood/Wood Waste Biomass',
                "Landfill Gas",
                "Petroleum Coke", 
                "Coal Integrated Gasification Combined Cycle"
                ]

    thermal_gen = pudl[pudl["technology_description"].isin(thermal_list)].dropna(subset=["latitude", "longitude", "LRZ"]).copy()
    thermal_gen = thermal_gen[["plant_id_eia", "plant_name_eia", "capacity_mw", "generator_operating_date", "fuel_type_code_pudl", "technology_description", "prime_mover_code", "latitude", "longitude", "LRZ"]]

    # Align types with FOR table codes
    tech_categories = {
        "CC" : ["Natural Gas Fired Combined Cycle", "Coal Integrated Gasification Combined Cycle"],
        "CT" : ["Natural Gas Fired Combustion Turbine", 'Natural Gas Internal Combustion Engine', "Landfill Gas", "Petroleum Liquids"],
        "ST" : ["Natural Gas Steam Turbine", "Conventional Steam Coal", 'Biomass', 'Other Waste Biomass', 'Wood/Wood Waste Biomass', "Petroleum Coke", 'Other Gases', 'Municipal Solid Waste',],
        "NU" : ["Nuclear"], 
    }

    tech_map = {tech: cat for cat, techs in tech_categories.items() for tech in techs}
    thermal_gen["tech_category"] = thermal_gen["technology_description"].map(tech_map)
    
    return thermal_gen

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

    thermal["profile_name"] = "MISO_" + thermal["LRZ"] + "_" + thermal["tech_category"].str.replace(" ", "_") + "_" + thermal["grid_num"].apply(lambda n: f"{int(n):03d}")
    return thermal

# ── Per-year worker ────────────────────────────────────────────────────────────
def process_year(year, profiles_list, unique_ncols, ncol_to_pos, output_dir, ensemble, rcp, cesm_base, for_table):
    year_dir = Path(output_dir)
    year_dir.mkdir(parents=True, exist_ok=True)

    trefht_ds = xr.open_dataset(trefht_path(year, ensemble, rcp, cesm_base))

    time_vals  = trefht_ds["time"].values

    # .chunk() only on the float data variables — avoids object-dtype time coord issue
    trefht_mem = trefht_ds["TREFHT"].chunk({"time": 43}).isel(ncol=unique_ncols).compute().values  -273.15
    trefht_ds.close(); 

    for profile_name, ncol_idx, tech in profiles_list:
        pos = ncol_to_pos[ncol_idx]
        outage_profile = calculate_fors(tech, trefht_mem[:, pos], for_table)
        pd.DataFrame({"time": time_vals, "FOR": outage_profile}).to_csv(
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
    OUTPUT_DIR = Path(f"/glade/work/rbhandarkar/Inputs/Thermal FORs/{label}")
    CESM_BASE  = Path(f"/gdex/data/{rcp_map[RCP]}/b.e13.B{RCP}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[RCP][ENSEMBLE]}/atm/proc/tseries/hour_6")

    ref_ds = xr.open_dataset(trefht_path(args.year, ENSEMBLE, RCP, CESM_BASE))
    nc_lat = ref_ds["lat"].values
    nc_lon = ref_ds["lon"].values
    ref_ds.close()

    for_table = pd.read_csv(FOR_TABLE)

    profiles = assign_profiles(thermal, nc_lat, nc_lon)
    print(f"Assigned {len(profiles)} unique profiles")

    unique_ncols  = np.unique(profiles["ncol_idx"].values).tolist()
    ncol_to_pos   = {int(nc): pos for pos, nc in enumerate(unique_ncols)}
    profiles_list = list(zip(profiles["profile_name"], profiles["ncol_idx"].tolist(), profiles["tech_category"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing year {args.year}...")
    process_year(args.year, profiles_list, unique_ncols, ncol_to_pos, str(OUTPUT_DIR), ENSEMBLE, RCP, CESM_BASE, for_table)
    print("Done.")
