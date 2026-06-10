import argparse
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import joblib
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

zone_model_map = {"MISO-0001" : "0",
                "MISO-0027" : "1", 
                "MISO-0035" : "2",
                "MISO-0004": "3",
                "MISO-0006" : "4",
                "MISO-8910" : "5",
                }

MODEL_FREQ = "6H"
POP_YEAR = 2025
ZONES = zone_model_map.keys()

# ── Holiday window settings ───────────────────────────────────────────────────
HOLIDAYS       = ["memorial_day", "july_4th", "labor_day", "christmas", "new_years"]
HOLIDAY_WINDOW = 0  # days on each side

CDD_BASE = 18.0  # °C (≈ 65°F)

GAM_COLS = [
    "cdd", "hdd",
    "cdd_lag6", "hdd_lag6",
    "cdd_lag12", "hdd_lag12",
    "cdd_lag18", "hdd_lag18",
    "cdd_lag24", "hdd_lag24",
    "hour", "month", "day_of_week", "year",
    "memorial_day", "july_4th", "labor_day", "christmas", "new_years",
]

# ── File path helpers ──────────────────────────────────────────────────────────
def _nc_path(variable, h_stream, year, ensemble, rcp, cesm_base):
    fname = (
        f"b.e13.B{rcp}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[rcp][ensemble]}"
        f".cam.{h_stream}.{variable}.{year}010100-{year+1}010100.nc"
    )
    return cesm_base / fname

def trefht_path(year, ensemble, rcp, cesm_base): return _nc_path("TREFHT", "h2", year, ensemble, rcp, cesm_base)

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
    t = temp_series

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
    # Noleap model calendar skips Feb 29; the resulting 30h gap makes Mar 1 00:00 UTC
    # appear to have invalid lags, but they are correct (positional lag = model 6H lag)
    utc_idx = X.index.tz_convert("UTC")
    gap_mask &= ~((utc_idx.month == 3) & (utc_idx.day == 1) & (utc_idx.hour == 0))
    for col in [c for c in X.columns if "lag" in c]:
        X.loc[gap_mask, col] = np.nan

    X["hour"]        = t.index.hour
    X["day_of_week"] = t.index.day_of_week
    X["weekend"]     = (t.index.day_of_week > 4).astype(int)
    X["month"]       = t.index.month
    X["year"]        = POP_YEAR     # do inference assuming 2025 fixed effects

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


# ── Per-year worker ────────────────────────────────────────────────────────────
def process_year(year, output_dir, ensemble, rcp, cesm_base):
    year_dir = Path(output_dir)
    year_dir.mkdir(parents=True, exist_ok=True)

    # Read in MESACLIP temperature data
    trefht_ds = xr.open_dataset(trefht_path(year, ensemble, rcp, cesm_base))
    temp_data = calc_weighted_temp(trefht_ds)
    trefht_ds.close()

    # Prepend prior year so lagged features are available at the start of target year
    try:
        trefht_prev = xr.open_dataset(trefht_path(year - 1, ensemble, rcp, cesm_base))
        temp_prev = calc_weighted_temp(trefht_prev)
        temp_combined = pd.concat([temp_prev, temp_data])
    except FileNotFoundError:
        temp_combined = temp_data

    # Convert to ET (with daylight savings)
    temp_combined = temp_combined.tz_localize("UTC").tz_convert("America/Indiana/Indianapolis")

    for z in ZONES:
        profile = infer_load(temp_combined[z], z)
        profile.index = profile.index.tz_convert("UTC")
        profile = profile[profile.index.year == year]
        profile.to_csv(year_dir / f"{z}_load_{year}.csv")

    return year

# ── County population weighted average ───────────────────────────────────────────────────────
def calc_weighted_temp(ds):
    # Load population data
    pop = pd.read_csv("../Weather Data Processing/Population data/co-est2019-alldata.csv", encoding="latin1")
    pop = pop[["STATE", "COUNTY", "STNAME", "CTYNAME", "POPESTIMATE2010", "POPESTIMATE2011","POPESTIMATE2012","POPESTIMATE2013","POPESTIMATE2014","POPESTIMATE2015","POPESTIMATE2016","POPESTIMATE2017","POPESTIMATE2018","POPESTIMATE2019"]]

    pop2 = pd.read_csv("../Weather Data Processing/Population data/co-est2025-alldata.csv", encoding="latin1")
    pop2 = pop2[['STATE', 'COUNTY','POPESTIMATE2020', 'POPESTIMATE2021',
        'POPESTIMATE2022', 'POPESTIMATE2023', 'POPESTIMATE2024', 'POPESTIMATE2025']]

    pop = pd.merge(pop, pop2, on=["STATE", "COUNTY"], how="left")
    pop = pop.rename(columns={"STATE" : "STATEFP", "COUNTY" : "COUNTYFP"})
    pop["STATEFP"] = pop["STATEFP"].astype(str).str.zfill(2)
    pop["COUNTYFP"] = pop["COUNTYFP"].astype(str).str.zfill(3)

    
    # Load county geographic boundaries
    counties = (
        gpd.read_file(r"../Weather Data Processing/GIS/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"))

    # Load grouped load resource zones
    lrz = gpd.read_file(r"../Weather Data Processing/GIS/MISO_grouped_LRZ.shp")

    # Merge population counts into counties dataset
    counties = counties.merge(pop, on =["STATEFP", "COUNTYFP"])

    # Spatial join: tag each county with the LRZ it falls in (based on centroid)
    counties["geometry_orig"] = counties.geometry
    counties_pt = counties.copy()
    counties_pt["centroid"] = counties_pt.geometry.to_crs("EPSG:5070").centroid.to_crs("EPSG:4326")

    counties_pt = counties_pt.set_geometry("centroid")
    counties_lrz = gpd.sjoin(counties_pt, lrz[["BA", "geometry"]].to_crs(counties_pt.crs), 
                            how="left", predicate="within")

    counties_lrz = counties_lrz.set_geometry("geometry_orig")
    counties_lrz = counties_lrz[counties_lrz["BA"].notna()]

    centroids = counties_lrz["centroid"]

    
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

    return result

# ── Infer 6-hourly load using MLP  ──────────────────────────────────────────────
def infer_load(temp_series, zone):
    # temp_series: tz-aware Series in °C at 6H resolution, ET with daylight savings
    X = build_features_raw(temp_series).dropna()

    bundle = joblib.load(f"mlp_tuned_results/mlp_tuned_zone{zone_model_map[zone]}.joblib")
    mlp, scaler = bundle["model"], bundle["scaler"]

    X_scaled = scaler.transform(encode_features(X).values)
    preds = mlp.predict(X_scaled)

    return pd.DataFrame({"predicted_load_mwh": preds}, index=X.index)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Year to process (e.g. 2023)")
    parser.add_argument("--ensemble", type=str, required=True, help="Ensemble member")
    parser.add_argument("--rcp", type=str, required=True, help="RCP")
    args = parser.parse_args()

    ENSEMBLE   = args.ensemble
    RCP        = args.rcp
    label      = ENSEMBLE if RCP == "RCP60" else f"{int(ENSEMBLE) + 9:03d}"
    OUTPUT_DIR = Path(f"/glade/work/rbhandarkar/Inputs/Load Profiles/{label}")
    CESM_BASE  = Path(f"/gdex/data/{rcp_map[RCP]}/b.e13.B{RCP}C5.ne120_t12.cesm-ihesp-hires1.0.{path_map[RCP][ENSEMBLE]}/atm/proc/tseries/hour_6")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing year {args.year}...")
    process_year(args.year, str(OUTPUT_DIR), ENSEMBLE, RCP, CESM_BASE)
    print("Done.")
