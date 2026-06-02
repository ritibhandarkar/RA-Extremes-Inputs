"""
Process MESACLIP temperature data for a single year/RCP/ensemble combination.

Usage:
    python process_mesaclip.py --year 2020 --rcp RCP85 --ens 005

Batch submission: submit one job per (year, rcp, ens) combination.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths relative to this script (mirrors the notebook's working directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
GIS_DIR = BASE_DIR / "GIS"
POP_DIR = BASE_DIR / "Population data"
OUT_DIR = "/glade/work/rbhandarkar/CESM/load_weighted_temps/"

# ---------------------------------------------------------------------------
# MESACLIP path / file-stem mappings (from notebook)
# ---------------------------------------------------------------------------
PATH_MAP = {
    "RCP85": {
        "002": "30-2006-2100.002",
        "003": "31-2006-2100.003",
        "004": "44-2006-2100.004",
        "005": "44-2006-2100.005",
        "006": "46-2006-2100.006",
        "007": "46-2006-2100.007",
        "008": "46-2006-2100.008",
        "009": "46-2006-2100.009",
        "010": "46-2006-2100.010",
    },
    "RCP60": {
        "001": "44-2006-2100.001",
        "002": "44-2006-2100.002",
        "003": "44-2006-2100.003",
        "004": "46-2006-2100.004",
        "005": "44-2006-2100.005",
        "006": "46-2006-2100.006",
        "007": "46-2006-2100.007",
        "008": "46-2006-2100.008",
        "009": "46-2006-2100.009",
        "010": "46-2006-2100.010",
    },
}

RCP_MAP = {
    "RCP85": "d651009",
    "RCP60": "d651008",
}

POP_YEAR = 2025


def load_geo_data():
    pop = pd.read_csv(POP_DIR / "co-est2019-alldata.csv", encoding="latin1")
    pop = pop[
        [
            "STATE", "COUNTY", "STNAME", "CTYNAME",
            "POPESTIMATE2010", "POPESTIMATE2011", "POPESTIMATE2012",
            "POPESTIMATE2013", "POPESTIMATE2014", "POPESTIMATE2015",
            "POPESTIMATE2016", "POPESTIMATE2017", "POPESTIMATE2018",
            "POPESTIMATE2019",
        ]
    ]

    pop2 = pd.read_csv(POP_DIR / "co-est2025-alldata.csv", encoding="latin1")
    pop2 = pop2[
        [
            "STATE", "COUNTY",
            "POPESTIMATE2020", "POPESTIMATE2021", "POPESTIMATE2022",
            "POPESTIMATE2023", "POPESTIMATE2024", "POPESTIMATE2025",
        ]
    ]

    pop = pd.merge(pop, pop2, on=["STATE", "COUNTY"], how="left")
    pop = pop.rename(columns={"STATE": "STATEFP", "COUNTY": "COUNTYFP"})
    pop["STATEFP"] = pop["STATEFP"].astype(str).str.zfill(2)
    pop["COUNTYFP"] = pop["COUNTYFP"].astype(str).str.zfill(3)

    counties = gpd.read_file(GIS_DIR / "cb_2018_us_county_500k" / "cb_2018_us_county_500k.shp")
    lrz = gpd.read_file(GIS_DIR / "MISO_grouped_LRZ.shp")

    counties = counties.merge(pop, on=["STATEFP", "COUNTYFP"])

    counties["geometry_orig"] = counties.geometry
    counties_pt = counties.copy()
    counties_pt["centroid"] = counties_pt.geometry.to_crs("EPSG:5070").centroid.to_crs("EPSG:4326")
    counties_pt = counties_pt.set_geometry("centroid")

    counties_lrz = gpd.sjoin(
        counties_pt,
        lrz[["BA", "geometry"]].to_crs(counties_pt.crs),
        how="left",
        predicate="within",
    )
    counties_lrz = counties_lrz.set_geometry("geometry_orig")
    counties_lrz = counties_lrz[counties_lrz["BA"].notna()]

    return counties_lrz, lrz


def process(year: int, rcp: str, ens: str):
    counties_lrz, lrz = load_geo_data()

    stem = PATH_MAP[rcp][ens]
    cesm_base = Path(
        f"/gdex/data/{RCP_MAP[rcp]}"
        f"/b.e13.B{rcp}C5.ne120_t12.cesm-ihesp-hires1.0.{stem}"
        f"/atm/proc/tseries/hour_6"
    )
    nc_file = (
        cesm_base
        / f"b.e13.B{rcp}C5.ne120_t12.cesm-ihesp-hires1.0.{stem}"
          f".cam.h2.TREFHT.{year}010100-{year + 1}010100.nc"
    )

    print(f"Loading {nc_file}")
    # No dask chunks — we load all county data in one shot below
    ds = xr.open_dataset(nc_file)

    # MESACLIP is unstructured — build a point GeoDataFrame for the grid
    lon = ((ds["lon"] + 180) % 360) - 180
    grid = pd.DataFrame({"lat": ds["lat"].values, "lon": lon.values})
    gdf_grid = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(grid.lon, grid.lat),
        crs="EPSG:4326",
    )

    centroids = counties_lrz["centroid"]
    county_points = gpd.GeoDataFrame(
        counties_lrz[["centroid", f"POPESTIMATE{POP_YEAR}", "BA"]],
        geometry=centroids,
        crs="EPSG:4326",
    )

    county_to_grid = gpd.sjoin_nearest(
        county_points.to_crs("EPSG:5070"),
        gdf_grid.to_crs("EPSG:5070"),
        how="left",
    )

    grid_indices = county_to_grid["index_right"].values
    pop_weights = np.asarray(county_to_grid[f"POPESTIMATE{POP_YEAR}"])
    lrz_ids = np.asarray(county_to_grid["BA"])
    times = ds.time.values

    # Chunk along time so dask parallelises reads across small time batches,
    # then select only the needed ncol indices and load into memory
    print("Reading county temperature data...")
    data = ds["TREFHT"].chunk({"time": 43}).isel(ncol=grid_indices).compute().values  # (time, n_counties)
    ds.close()

    unique_lrz = lrz["BA"].unique()
    lrz_temp = np.zeros((len(times), len(unique_lrz)))

    for j, lrz_id in enumerate(unique_lrz):
        mask = lrz_ids == lrz_id
        w = pop_weights[mask]
        if w.sum() == 0:
            lrz_temp[:, j] = np.nan
        else:
            lrz_temp[:, j] = np.average(data[:, mask], weights=w, axis=1)

    result = pd.DataFrame(
        lrz_temp,
        index=pd.to_datetime([str(t) for t in times]),
        columns=unique_lrz,
    )
    result = result - 273.15
    result.index.name = "time"
    

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"MISO_{rcp}_{ens}_trefht_{year}.csv"
    result.to_csv(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Process MESACLIP temperature data.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--rcp", choices=["RCP60", "RCP85"], required=True)
    parser.add_argument("--ens", required=True, help="Ensemble ID, zero-padded (e.g. 005)")
    args = parser.parse_args()

    ens = args.ens.zfill(3)

    if args.rcp not in PATH_MAP:
        sys.exit(f"Unknown RCP: {args.rcp}")
    if ens not in PATH_MAP[args.rcp]:
        sys.exit(f"Ensemble {ens} not available for {args.rcp}")

    process(args.year, args.rcp, ens)


if __name__ == "__main__":
    main()
