"""
Combine per-year profile CSVs into single per-profile CSVs.

Input layout:
    base_dir/
        {ensemble}/
            {profile_name}_{year}.csv   # columns: time, cf (or any cols)

Output layout:
    output_dir/
        {ensemble}/
            {profile_name}.csv          # all years concatenated, sorted by time
"""

import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


YEAR_SUFFIX = re.compile(r"^(.+)_(\d{4})$")


def combine_profiles(base_dir: str, output_dir: str, skip_pattern: str = None) -> None:
    """
    For each ensemble member directory under base_dir, concatenate all per-year
    CSVs that share a profile name and write one combined CSV per profile to
    output_dir/{ensemble}/{profile_name}.csv.

    Args:
        base_dir:     Path containing ensemble member subdirectories.
        output_dir:   Root directory for combined output (created if absent).
        skip_pattern: Optional regex; ensemble dirs whose names match are skipped
                      (e.g. r"002_og" to skip the 002_og directory).
    """
    base = Path(base_dir)
    out_root = Path(output_dir)

    skip_re = re.compile(skip_pattern) if skip_pattern else None

    ensemble_dirs = sorted(d for d in base.iterdir() if d.is_dir())

    for ens_dir in ensemble_dirs:
        if skip_re and skip_re.search(ens_dir.name):
            print(f"Skipping {ens_dir.name}")
            continue

        print(f"Processing ensemble {ens_dir.name} ...", flush=True)

        # Group files by profile name (strip _{year} suffix)
        profile_files: dict[str, list[Path]] = defaultdict(list)
        for f in sorted(ens_dir.glob("*.csv")):
            m = YEAR_SUFFIX.match(f.stem)
            if m:
                profile_name, year = m.group(1), int(m.group(2))
                profile_files[profile_name].append((year, f))

        out_ens = out_root / ens_dir.name
        out_ens.mkdir(parents=True, exist_ok=True)

        for profile_name, year_files in profile_files.items():
            # Sort by year so time series is chronological
            year_files.sort(key=lambda x: x[0])
            dfs = [pd.read_csv(f) for _, f in year_files]
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(out_ens / f"{profile_name}.csv", index=False)

        print(f"  Written {len(profile_files)} profiles to {out_ens}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine per-year profile CSVs into single per-profile CSVs.")
    parser.add_argument("--base-dir",    required=True,  help="Directory containing ensemble member subdirectories")
    parser.add_argument("--output-dir",  required=True,  help="Root directory for combined output")
    parser.add_argument("--ensemble",    required=True,  help="Ensemble member name (e.g. 001, 017)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_root = Path(args.output_dir)
    ens_dir = base / args.ensemble

    if not ens_dir.is_dir():
        raise FileNotFoundError(f"Ensemble directory not found: {ens_dir}")

    print(f"Processing ensemble {args.ensemble} ...", flush=True)

    profile_files: dict[str, list] = defaultdict(list)
    for f in sorted(ens_dir.glob("*.csv")):
        m = YEAR_SUFFIX.match(f.stem)
        if m:
            profile_name, year = m.group(1), int(m.group(2))
            profile_files[profile_name].append((year, f))

    out_ens = out_root / args.ensemble
    out_ens.mkdir(parents=True, exist_ok=True)

    for profile_name, year_files in profile_files.items():
        year_files.sort(key=lambda x: x[0])
        dfs = [pd.read_csv(f) for _, f in year_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(out_ens / f"{profile_name}.csv", index=False)

    print(f"  Written {len(profile_files)} profiles to {out_ens}")
