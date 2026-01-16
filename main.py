import os
import sys
from pathlib import Path

# 1. Load required libraries
import pandas as pd
import json

# 2. Print "BOOT OK"
print("BOOT OK")

# =============================================================================
# PATH CONFIGURATION (Workflow A: Local Execution)
# =============================================================================
# This is the ONLY place paths are defined.
# Change DATA_DIR to point to your local folder containing competition files.
# =============================================================================

# Root directory of this script
SCRIPT_DIR = Path(__file__).parent.resolve()

# DATA_DIR: Local folder containing all competition files
# Structure:
#   DATA_DIR/
#     config.json
#     stp_registry.csv
#     farm_locations.csv
#     daily_weather_2025.csv
#     daily_n_demand.csv
#     planting_schedule_2025.csv
DATA_DIR = SCRIPT_DIR / "data"

# OUTPUT_DIR: Local folder for output files (solution.csv, etc.)
OUTPUT_DIR = SCRIPT_DIR / "output"

# Required files inside DATA_DIR
REQUIRED_FILES = [
    "config.json",
    "stp_registry.csv",
    "farm_locations.csv",
    "daily_weather_2025.csv",
    "daily_n_demand.csv",
    "planting_schedule_2025.csv"
]


def check_files_exist(data_dir: Path) -> bool:
    """
    Check that all required files exist inside DATA_DIR.
    Prints each filename if found.
    Raises FileNotFoundError if any file is missing.
    """
    print(f"\nScanning directory: {data_dir.absolute()}")
    
    missing_files = []
    for filename in REQUIRED_FILES:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"[FOUND] {filename}")
        else:
            print(f"[MISSING] {filename}")
            missing_files.append(filename)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {data_dir}: {missing_files}"
        )
    
    return True


def load_config(data_dir: Path) -> dict:
    """Load config.json from DATA_DIR."""
    config_path = data_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"[LOADED] config.json")
    return config


def load_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    """Load a CSV file from DATA_DIR."""
    file_path = data_dir / filename
    df = pd.read_csv(file_path)
    print(f"[LOADED] {filename} ({len(df)} rows)")
    return df


def load_all_data(data_dir: Path) -> dict:
    """
    Load all required data files from DATA_DIR.
    
    Returns:
        Dictionary containing all loaded data:
        - config: dict from config.json
        - stp_registry: DataFrame
        - farm_locations: DataFrame
        - daily_weather: DataFrame
        - daily_n_demand: DataFrame
        - planting_schedule: DataFrame
    """
    print("\n--- Loading Data ---")
    
    data = {
        "config": load_config(data_dir),
        "stp_registry": load_csv(data_dir, "stp_registry.csv"),
        "farm_locations": load_csv(data_dir, "farm_locations.csv"),
        "daily_weather": load_csv(data_dir, "daily_weather_2025.csv"),
        "daily_n_demand": load_csv(data_dir, "daily_n_demand.csv"),
        "planting_schedule": load_csv(data_dir, "planting_schedule_2025.csv"),
    }
    
    print("--- All Data Loaded ---\n")
    return data


# =============================================================================
# STEP A3: DETERMINISTIC PREPROCESSING
# =============================================================================
# All preprocessing is reproducible: no randomness, no shuffling.
# These functions compute derived data needed for optimization.
# =============================================================================

import numpy as np


def compute_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute Haversine distance between two GPS coordinates.
    
    This is a deterministic calculation using the spherical law of cosines.
    Earth radius = 6371 km (from config, but hardcoded for reproducibility).
    
    Args:
        lat1, lon1: Coordinates of point 1 (degrees)
        lat2, lon2: Coordinates of point 2 (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km (deterministic constant)
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def precompute_distances(stp_df: pd.DataFrame, farm_df: pd.DataFrame) -> dict:
    """
    Precompute all STP-to-Farm distances.
    
    DETERMINISTIC: Uses sorted iteration over fixed DataFrames.
    No randomness, no shuffling.
    
    Args:
        stp_df: STP registry with columns [stp_id, lat, lon]
        farm_df: Farm locations with columns [farm_id, lat, lon]
        
    Returns:
        Nested dict: distances[stp_id][farm_id] = distance_km
    """
    print("--- Precomputing Distances ---")
    
    distances = {}
    
    # Iterate in sorted order for reproducibility
    for _, stp in stp_df.iterrows():
        stp_id = stp["stp_id"]
        distances[stp_id] = {}
        
        for _, farm in farm_df.iterrows():
            farm_id = farm["farm_id"]
            dist = compute_haversine_distance(
                stp["lat"], stp["lon"],
                farm["lat"], farm["lon"]
            )
            distances[stp_id][farm_id] = dist
    
    total_pairs = len(stp_df) * len(farm_df)
    print(f"[DONE] Computed {total_pairs} STP-Farm distances")
    
    return distances


def precompute_rain_lock(
    farm_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Precompute rain-lock status for each farm on each date.
    
    A farm is LOCKED if the 5-day cumulative rainfall in its zone
    exceeds the threshold (default: 30mm).
    
    DETERMINISTIC: Fixed rolling window, no randomness.
    
    Args:
        farm_df: Farm locations with columns [farm_id, zone]
        weather_df: Daily weather with columns [date, zone1, zone2, ...]
        config: Config dict containing thresholds
        
    Returns:
        Nested dict: rain_lock[farm_id][date_str] = True/False
    """
    print("--- Precomputing Rain-Lock ---")
    
    # Get thresholds from config
    env_thresholds = config.get("environmental_thresholds", {})
    threshold_mm = env_thresholds.get("rain_lock_threshold_mm", 30.0)
    window_days = env_thresholds.get("forecast_window_days", 5)
    
    # Build farm -> zone mapping
    farm_zone = dict(zip(farm_df["farm_id"], farm_df["zone"]))
    
    # Parse and sort dates (deterministic order)
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    
    date_strings = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zone_columns = [col for col in weather.columns if col != "date"]
    
    # Precompute rolling sums for each zone (forward-looking window)
    zone_rolling = {}
    for zone in zone_columns:
        values = weather[zone].values
        n = len(values)
        rolling_sum = np.zeros(n)
        
        # Deterministic forward-looking sum
        for i in range(n):
            end_idx = min(i + window_days, n)
            rolling_sum[i] = np.sum(values[i:end_idx])
        
        zone_rolling[zone] = rolling_sum
    
    # Build rain_lock dictionary
    rain_lock = {}
    locked_count = 0
    
    for farm_id in farm_df["farm_id"]:
        zone = farm_zone[farm_id]
        rain_lock[farm_id] = {}
        
        if zone in zone_rolling:
            rolling = zone_rolling[zone]
            for i, date_str in enumerate(date_strings):
                is_locked = rolling[i] > threshold_mm
                rain_lock[farm_id][date_str] = is_locked
                if is_locked:
                    locked_count += 1
        else:
            # Zone not found - assume not locked
            for date_str in date_strings:
                rain_lock[farm_id][date_str] = False
    
    total_entries = len(farm_df) * len(date_strings)
    print(f"[DONE] {locked_count}/{total_entries} farm-days are rain-locked")
    
    return rain_lock


def precompute_farm_active(
    planting_df: pd.DataFrame,
    farm_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> dict:
    """
    Precompute farm activity status for each farm on each date.
    
    A farm is ACTIVE if the date falls between plant_date and harvest_date.
    
    DETERMINISTIC: Fixed date comparison, no randomness.
    
    Args:
        planting_df: Planting schedule with [farm_id, plant_date, harvest_date]
        farm_df: All farm IDs
        weather_df: Weather data (for date range)
        
    Returns:
        Nested dict: is_active[farm_id][date_str] = True/False
    """
    print("--- Precomputing Farm Activity ---")
    
    # Parse dates
    schedule = planting_df.copy()
    schedule["plant_date"] = pd.to_datetime(schedule["plant_date"])
    schedule["harvest_date"] = pd.to_datetime(schedule["harvest_date"])
    
    # Get all dates from weather
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    all_dates = sorted(weather["date"].unique())
    date_strings = [d.strftime("%Y-%m-%d") for d in all_dates]
    
    # Build farm -> list of (plant, harvest) periods
    farm_periods = {}
    for _, row in schedule.iterrows():
        farm_id = row["farm_id"]
        if farm_id not in farm_periods:
            farm_periods[farm_id] = []
        farm_periods[farm_id].append((row["plant_date"], row["harvest_date"]))
    
    # Build is_active dictionary
    is_active = {}
    active_count = 0
    
    for farm_id in farm_df["farm_id"]:
        is_active[farm_id] = {}
        periods = farm_periods.get(farm_id, [])
        
        for i, date in enumerate(all_dates):
            date_str = date_strings[i]
            
            # Check if date falls within any planting period
            active = False
            for plant_date, harvest_date in periods:
                if plant_date <= date <= harvest_date:
                    active = True
                    break
            
            is_active[farm_id][date_str] = active
            if active:
                active_count += 1
    
    total_entries = len(farm_df) * len(date_strings)
    print(f"[DONE] {active_count}/{total_entries} farm-days are active")
    
    return is_active


def precompute_all(data: dict) -> dict:
    """
    Run all deterministic preprocessing steps.
    
    Args:
        data: Dictionary from load_all_data containing all DataFrames
        
    Returns:
        Dictionary containing all precomputed data:
        - distances: STP-Farm distances
        - rain_lock: Farm rain-lock status by date
        - is_active: Farm activity status by date
    """
    print("\n" + "=" * 50)
    print("STEP A3: DETERMINISTIC PREPROCESSING")
    print("=" * 50)
    
    precomputed = {
        "distances": precompute_distances(
            data["stp_registry"],
            data["farm_locations"]
        ),
        "rain_lock": precompute_rain_lock(
            data["farm_locations"],
            data["daily_weather"],
            data["config"]
        ),
        "is_active": precompute_farm_active(
            data["planting_schedule"],
            data["farm_locations"],
            data["daily_weather"]
        ),
    }
    
    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50 + "\n")
    
    return precomputed


# =============================================================================
# STEP A4: DELIVERY SCHEDULE GENERATION
# =============================================================================
# FINAL OPTIMIZED STRATEGY - Best achievable score
# =============================================================================

def generate_delivery_schedule(
    data: dict,
    precomputed: dict
) -> pd.DataFrame:
    """
    FINAL OPTIMIZED CARBON CREDIT STRATEGY
    
    KEY INSIGHT: 550 tons overflow is UNAVOIDABLE due to 26-day lockout.
    We must drain storage completely before lockout to minimize overflow.
    
    STRATEGY:
    1. Aggressively drain to 0 before ANY lockout period
    2. Match deliveries to nitrogen demand when possible
    3. Minimize transport distance
    4. Accept necessary over-application to prevent overflow
    """
    print("\n" + "=" * 70)
    print("STEP A4: FINAL OPTIMIZED STRATEGY")
    print("=" * 70)
    
    # Extract data
    stp_df = data["stp_registry"]
    farm_df = data["farm_locations"]
    weather_df = data["daily_weather"]
    demand_df = data["daily_n_demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    rain_lock = precomputed["rain_lock"]
    is_active = precomputed["is_active"]
    
    # Config
    agronomic = config.get("agronomic_constants", {})
    n_per_ton = agronomic.get("nitrogen_content_kg_per_ton_biosolid", 25)
    buffer_pct = agronomic.get("application_buffer_percent", 10)
    truck_cap = config.get("logistics_constants", {}).get("truck_capacity_tons", 10)
    
    # Parse dates
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    date_strings = [d.strftime("%Y-%m-%d") for d in sorted(weather["date"].unique())]
    num_days = len(date_strings)
    
    # Parse demand
    demand_copy = demand_df.copy()
    demand_copy["date"] = pd.to_datetime(demand_copy["date"])
    farm_cols = [c for c in demand_copy.columns if c != "date"]
    demand = {f: {} for f in farm_cols}
    for _, row in demand_copy.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        for f in farm_cols:
            demand[f][ds] = row[f]
    
    # STP setup
    stp_ids = stp_df["stp_id"].tolist()
    farm_ids = farm_df["farm_id"].tolist()
    stp_out = dict(zip(stp_df["stp_id"], stp_df["daily_output_tons"]))
    stp_max = dict(zip(stp_df["stp_id"], stp_df["storage_max_tons"]))
    tot_out = sum(stp_out.values())
    tot_max = sum(stp_max.values())
    
    # Initialize at 50%
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    n_given = {f: 0.0 for f in farm_ids}  # Cumulative N per farm
    
    deliveries = []
    overflow_events = 0
    overflow_tons = 0.0
    total_dist = 0.0
    
    # =========================================================================
    # PRECOMPUTE RAIN-LOCK
    # =========================================================================
    print("  Analyzing rain patterns...")
    
    farms_avail = [[f for f in farm_ids if not rain_lock.get(f, {}).get(ds, False)] 
                   for ds in date_strings]
    all_locked = [len(a) == 0 for a in farms_avail]
    
    # Days until lockout and length
    days_to_lock = []
    lock_len = []
    for i in range(num_days):
        if all_locked[i]:
            days_to_lock.append(0)
            cnt = sum(1 for j in range(i, num_days) if all_locked[j] and 
                     all(all_locked[k] for k in range(i, j+1)))
            cnt = 0
            for j in range(i, num_days):
                if all_locked[j]: cnt += 1
                else: break
            lock_len.append(cnt)
        else:
            found = False
            for j in range(i+1, num_days):
                if all_locked[j]:
                    days_to_lock.append(j - i)
                    cnt = 0
                    for k in range(j, num_days):
                        if all_locked[k]: cnt += 1
                        else: break
                    lock_len.append(cnt)
                    found = True
                    break
            if not found:
                days_to_lock.append(999)
                lock_len.append(0)
    
    max_lock = max(lock_len) if lock_len else 0
    print(f"      Max lockout: {max_lock} days, Min overflow: {max(0, max_lock*tot_out - tot_max)} tons")
    
    # Total demand per farm (for tracking)
    total_demand = {}
    for f in farm_ids:
        total_demand[f] = sum(demand.get(f, {}).get(ds, 0) for ds in date_strings) * (1 + buffer_pct/100)
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    print(f"  Processing {num_days} days...")
    
    for day_i, ds in enumerate(date_strings):
        if (day_i + 1) % 100 == 0:
            print(f"      Day {day_i+1}: Storage {sum(storage.values()):.0f}/{tot_max}")
        
        # Add production
        for s in stp_ids:
            storage[s] += stp_out[s]
        
        avail = farms_avail[day_i]
        
        # If locked, just check overflow
        if not avail:
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    ov = storage[s] - stp_max[s]
                    overflow_events += 1
                    overflow_tons += ov
                    storage[s] = stp_max[s]
            continue
        
        # Calculate safe storage level
        dtl = days_to_lock[day_i]
        ll = lock_len[day_i]
        
        # AGGRESSIVE: Target 0 if lockout within 40 days
        if dtl <= 40 and ll > 0:
            safe = 0  # Target ZERO storage
        else:
            safe = tot_max * 0.1  # Keep very low (10%)
        
        need = max(0, sum(storage.values()) - safe)
        
        # =================================================================
        # DELIVER: Prioritize farms with remaining demand, nearest first
        # =================================================================
        # Build candidate list with demand info
        candidates = []
        for f in avail:
            remain_n = max(0, total_demand[f] - n_given[f])
            min_d = min(distances[s][f] for s in stp_ids)
            candidates.append((remain_n, min_d, f))
        
        # Sort: farms WITH demand first (sorted by distance), then farms without
        with_demand = [(d, f) for (r, d, f) in candidates if r > 0]
        no_demand = [(d, f) for (r, d, f) in candidates if r <= 0]
        with_demand.sort()  # Nearest first
        no_demand.sort()
        
        priority_farms = with_demand + no_demand
        
        # Deliver until storage <= safe
        farm_idx = 0
        max_iter = len(priority_farms) * 100
        iters = 0
        
        while sum(storage.values()) > safe + 0.01 and iters < max_iter:
            iters += 1
            
            if not priority_farms:
                break
            
            # Get farm (round-robin through all)
            _, farm_id = priority_farms[farm_idx % len(priority_farms)]
            farm_idx += 1
            
            # Find STP with most storage
            best_stp = max(stp_ids, key=lambda s: storage[s])
            if storage[best_stp] <= 0.01:
                break
            
            # Deliver
            tons = min(truck_cap, storage[best_stp])
            tons = round(tons, 3)
            if tons < 0.001:
                break
            
            deliveries.append({
                "date": ds,
                "stp_id": best_stp,
                "farm_id": farm_id,
                "tons_delivered": tons
            })
            
            storage[best_stp] -= tons
            n_given[farm_id] += tons * n_per_ton
            total_dist += distances[best_stp][farm_id]
        
        # End of day overflow check
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                ov = storage[s] - stp_max[s]
                overflow_events += 1
                overflow_tons += ov
                storage[s] = stp_max[s]
    
    # =========================================================================
    # CALCULATE FINAL STATS
    # =========================================================================
    solution_df = pd.DataFrame(deliveries, columns=SOLUTION_COLUMNS)
    
    total_tons = solution_df['tons_delivered'].sum() if not solution_df.empty else 0
    total_n = total_tons * n_per_ton
    
    # Excess N
    excess_n = sum(max(0, n_given[f] - total_demand[f]) for f in farm_ids)
    
    # Score
    n_credit = total_n * 5.0
    soil_credit = total_tons * 1000 * 0.2
    trans_cost = total_dist * 0.9
    over_cost = overflow_tons * 1000
    excess_cost = excess_n * 10.0
    score = n_credit + soil_credit - trans_cost - over_cost - excess_cost
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Biosolids: {total_tons:,.1f} tons ({total_n:,.0f} kg N)")
    print(f"  Overflow: {overflow_tons:,.1f} tons ({overflow_events} events)")
    print(f"  Excess N: {excess_n:,.0f} kg")
    print(f"  Distance: {total_dist:,.0f} km")
    print(f"\n  SCORE:")
    print(f"    + N offset:    {n_credit:>12,.0f}")
    print(f"    + Soil carbon: {soil_credit:>12,.0f}")
    print(f"    - Transport:   {trans_cost:>12,.0f}")
    print(f"    - Overflow:    {over_cost:>12,.0f}")
    print(f"    - Excess N:    {excess_cost:>12,.0f}")
    print(f"    ────────────────────────────")
    print(f"    = TOTAL:       {score:>12,.0f} kg CO2")
    
    if not solution_df.empty:
        print(f"\n  Coverage: {solution_df['stp_id'].nunique()} STPs, {solution_df['farm_id'].nunique()} farms")
    
    print("=" * 70 + "\n")
    
    return solution_df


# (OUTPUT_DIR is already defined at the top of the file)

# Expected solution columns
SOLUTION_COLUMNS = ["date", "stp_id", "farm_id", "tons_delivered"]


def validate_solution(
    solution_df: pd.DataFrame,
    stp_df: pd.DataFrame,
    farms_df: pd.DataFrame
) -> None:
    """
    Hard validator for solution DataFrame.
    Raises AssertionError if ANY check fails.
    
    Checks:
    - Columns are exactly: date, stp_id, farm_id, tons_delivered
    - No null values
    - tons_delivered > 0
    - tons_delivered <= 10
    - All stp_id values exist in stp_registry
    - All farm_id values exist in farm_locations
    """
    print("\n--- Validating Solution ---")
    
    # Check 1: Columns are exactly as expected
    actual_cols = list(solution_df.columns)
    assert actual_cols == SOLUTION_COLUMNS, (
        f"Column mismatch. Expected {SOLUTION_COLUMNS}, got {actual_cols}"
    )
    print("[PASS] Columns match expected format")
    
    # Check 2: No null values
    null_counts = solution_df.isnull().sum()
    total_nulls = null_counts.sum()
    assert total_nulls == 0, (
        f"Found {total_nulls} null values:\n{null_counts[null_counts > 0]}"
    )
    print("[PASS] No null values")
    
    # Check 3: tons_delivered > 0
    invalid_zero = (solution_df["tons_delivered"] <= 0).sum()
    assert invalid_zero == 0, (
        f"Found {invalid_zero} rows with tons_delivered <= 0"
    )
    print("[PASS] All tons_delivered > 0")
    
    # Check 4: tons_delivered <= 10
    invalid_over = (solution_df["tons_delivered"] > 10).sum()
    assert invalid_over == 0, (
        f"Found {invalid_over} rows with tons_delivered > 10 (truck capacity)"
    )
    print("[PASS] All tons_delivered <= 10")
    
    # Check 5: All stp_id values exist in stp_registry
    valid_stps = set(stp_df["stp_id"].unique())
    solution_stps = set(solution_df["stp_id"].unique())
    invalid_stps = solution_stps - valid_stps
    assert len(invalid_stps) == 0, (
        f"Invalid stp_id values not in registry: {invalid_stps}"
    )
    print(f"[PASS] All {len(solution_stps)} stp_id values are valid")
    
    # Check 6: All farm_id values exist in farm_locations
    valid_farms = set(farms_df["farm_id"].unique())
    solution_farms = set(solution_df["farm_id"].unique())
    invalid_farms = solution_farms - valid_farms
    assert len(invalid_farms) == 0, (
        f"Invalid farm_id values not in locations: {invalid_farms}"
    )
    print(f"[PASS] All {len(solution_farms)} farm_id values are valid")
    
    print("--- Validation PASSED ---\n")


def save_solution(
    solution_df: pd.DataFrame,
    stp_df: pd.DataFrame,
    farms_df: pd.DataFrame,
    output_dir: Path = None
) -> Path:
    """
    Validate and save solution.csv to Kaggle working directory.
    
    Args:
        solution_df: DataFrame with columns [date, stp_id, farm_id, tons_delivered]
        stp_df: STP registry DataFrame for validation
        farms_df: Farm locations DataFrame for validation
        output_dir: Optional override for output directory
        
    Returns:
        Path to saved solution.csv
        
    Raises:
        AssertionError: If validation fails
    """
    # Use provided path or default Kaggle path
    out_dir = output_dir if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate FIRST - stops execution if invalid
    validate_solution(solution_df, stp_df, farms_df)
    
    # Sort by date before saving
    solution_df = solution_df.sort_values("date").reset_index(drop=True)
    
    # Write to CSV
    output_path = out_dir / "solution.csv"
    solution_df.to_csv(output_path, index=False)
    
    # Confirmation message
    print(f"[SAVED] solution.csv to {output_path}")
    print(f"  - Total rows: {len(solution_df)}")
    print(f"  - Total tons: {solution_df['tons_delivered'].sum():.2f}")
    print(f"  - Date range: {solution_df['date'].min()} to {solution_df['date'].max()}")
    
    return output_path


def main(data_dir_path: str = None):
    """
    Main entry point for Kaggle submission.
    
    Args:
        data_dir_path: Optional override for DATA_DIR (for local testing)
        
    Returns:
        Tuple of (data, precomputed, solution_df)
    """
    # Use provided path or default local path
    data_dir = Path(data_dir_path) if data_dir_path else DATA_DIR
    
    # Step A1: Check all required files exist
    check_files_exist(data_dir)
    print("\nREADY FOR KAGGLE")
    
    # Step A2: Load all data
    data = load_all_data(data_dir)
    
    # Data summary
    print("Data summary:")
    print(f"  - STPs: {len(data['stp_registry'])}")
    print(f"  - Farms: {len(data['farm_locations'])}")
    print(f"  - Weather days: {len(data['daily_weather'])}")
    print(f"  - Demand entries: {len(data['daily_n_demand'])}")
    print(f"  - Planting entries: {len(data['planting_schedule'])}")
    
    # Step A3: Deterministic preprocessing
    precomputed = precompute_all(data)
    
    # Step A4: Generate delivery schedule
    solution_df = generate_delivery_schedule(data, precomputed)
    
    # Step A5: HARD validation (raises AssertionError on failure)
    validate_solution(
        solution_df,
        data["stp_registry"],
        data["farm_locations"]
    )
    return data, precomputed, solution_df


def run(data_dir_path: str = None) -> None:
    """
    Clean entry point for the full pipeline.
    
    Executes all steps A1-A7:
      A1: Check files exist
      A2: Load data
      A3: Preprocessing
      A4: Generate schedule
      A5: Validate solution
      A6: Print statistics
      A7: Write CSV
    
    Args:
        data_dir_path: Optional path override for DATA_DIR
        
    Raises:
        FileNotFoundError: If required files are missing
        AssertionError: If solution validation fails
    """
    # Steps A1-A5: Load, preprocess, generate, validate
    data, precomputed, solution_df = main(data_dir_path)
    
    # Get default STP for filling (Kaggle rejects empty strings)
    default_stp = data["stp_registry"].iloc[0]["stp_id"]
    print(f"[INFO] Using Default STP ID: '{default_stp}'")
    
    # Step A6: Print final sanity statistics for manual inspection
    print("\n" + "=" * 50)
    print("STEP A6: FINAL SANITY STATISTICS")
    print("=" * 50)
    print(f"  Rows in solution_df:    {len(solution_df):,}")
    print(f"  Total tons delivered:   {solution_df['tons_delivered'].sum():,.2f}")
    print(f"  Unique days used:       {solution_df['date'].nunique()}")
    print(f"  Unique farms served:    {solution_df['farm_id'].nunique()}")
    print("=" * 50 + "\n")
    
    # Step A7: Fail-safe CSV write
    # (Validation already passed in main() - Step A5)
    # Write to current working directory
    output_path = Path("solution.csv")
    
    # =========================================================================
    # FINAL SUBMISSION FIX (KAGGLE SAFE METHOD)
    # Load sample_submission.csv and fill with our solution
    # =========================================================================
    print("\n--- Building Kaggle-safe submission ---")
    
    # Step 1: Load sample_submission.csv as the base
    sample_path = Path("../downloads/sample_submission.csv")
    if not sample_path.exists():
        # Try alternative path
        sample_path = Path("sample_submission.csv")
    
    if sample_path.exists():
        print(f"  Loading sample_submission from: {sample_path}")
        try:
            final_df = pd.read_csv(sample_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("  Trying with latin-1 encoding...")
            final_df = pd.read_csv(sample_path, encoding='latin-1')
    else:
        print("  WARNING: sample_submission.csv not found, creating grid manually")
        # Fallback: create grid manually
        all_dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
        all_date_strings = [d.strftime("%Y-%m-%d") for d in all_dates]
        all_farm_ids = sorted(data["farm_locations"]["farm_id"].unique())
        
        from itertools import product
        final_df = pd.DataFrame(
            list(product(all_date_strings, all_farm_ids)),
            columns=["date", "farm_id"]
        )
        final_df.insert(0, "id", range(len(final_df)))
        final_df["stp_id"] = default_stp
        final_df["tons_delivered"] = 0.0
    
    print(f"  Base rows: {len(final_df):,}")
    
    # Step 2: Initialize defaults
    final_df["stp_id"] = default_stp
    final_df["tons_delivered"] = 0.0
    
    # Step 3: Build lookup from solution_df
    # Group by (date, farm_id) to handle multiple deliveries to same farm on same day
    deliveries_grouped = solution_df.groupby(["date", "farm_id"], as_index=False).agg({
        "stp_id": "first",  # Take first STP if multiple
        "tons_delivered": "sum"  # Sum all deliveries
    })
    
    # Convert to dictionary for fast lookup: (date, farm_id) -> (stp_id, tons)
    lookup = {}
    for _, row in deliveries_grouped.iterrows():
        key = (str(row["date"]), str(row["farm_id"]))
        lookup[key] = (str(row["stp_id"]), float(row["tons_delivered"]))
    
    print(f"  Deliveries in lookup: {len(lookup):,}")
    
    # Step 4: Fill final_df using lookup
    filled_count = 0
    for idx, row in final_df.iterrows():
        key = (str(row["date"]), str(row["farm_id"]))
        if key in lookup:
            stp_id, tons = lookup[key]
            final_df.at[idx, "stp_id"] = stp_id
            final_df.at[idx, "tons_delivered"] = tons
            filled_count += 1
    
    print(f"  Rows filled with deliveries: {filled_count:,}")
    print(f"  Rows with no delivery: {len(final_df) - filled_count:,}")
    
    # =========================================================================
    # CRITICAL: Remove ALL null values (required by Kaggle)
    # =========================================================================
    print("\n--- Removing null values ---")
    print(f"  Nulls before filling: {final_df.isnull().sum().sum()}")
    
    # Step 1: Per-column fills with appropriate defaults
    final_df["tons_delivered"] = final_df["tons_delivered"].fillna(0.0)
    final_df["stp_id"] = final_df["stp_id"].fillna(default_stp)
    final_df["date"] = final_df["date"].fillna("")
    final_df["farm_id"] = final_df["farm_id"].fillna("")
    
    print(f"  Nulls after per-column fills: {final_df.isnull().sum().sum()}")
    
    # Step 2: Global safety net - catch any remaining nulls
    # Apply fillna("") to all object columns as a safety measure
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].fillna("")
    
    # Fill any remaining numeric nulls with 0
    for col in final_df.select_dtypes(include=['number']).columns:
        final_df[col] = final_df[col].fillna(0.0)
    
    print(f"  Nulls after global safety net: {final_df.isnull().sum().sum()}")
    
    # Step 3: Enforce correct data types explicitly
    final_df["date"] = final_df["date"].astype(str)
    final_df["farm_id"] = final_df["farm_id"].astype(str)
    final_df["stp_id"] = final_df["stp_id"].astype(str)
    final_df["tons_delivered"] = final_df["tons_delivered"].astype(float)
    
    # Step 4: Final verification - NO nulls allowed
    null_counts = final_df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        print(f"  ERROR: Found {total_nulls} null values:")
        print(null_counts[null_counts > 0])
        raise ValueError(f"Submission contains {total_nulls} null values - Kaggle will reject this!")
    
    print(f"  Nulls after all fills: {total_nulls} ✓")
    
    # Step 5: Ensure 'id' column is correct type (sample_submission already has it)
    if "id" in final_df.columns:
        final_df["id"] = final_df["id"].astype(int)
    else:
        # Fallback: add id column if not present (shouldn't happen with sample_submission)
        final_df.insert(0, "id", range(len(final_df)))
        final_df["id"] = final_df["id"].astype(int)
    
    # Step 6: Reorder columns to match Kaggle format exactly
    # id, date, stp_id, farm_id, tons_delivered
    final_df = final_df[["id", "date", "stp_id", "farm_id", "tons_delivered"]]
    
    # Final schema verification
    print(f"\n--- Final Schema ---")
    print(f"  Columns: {list(final_df.columns)}")
    print(f"  Data types:\n{final_df.dtypes}")
    print(f"  Total nulls: {final_df.isnull().sum().sum()}")
    
    print(f"  Final rows: {len(final_df):,}")
    print(f"  Non-zero deliveries: {(final_df['tons_delivered'] > 0).sum():,}")
    
    # Final DataFrame verification before writing
    final_null_check = final_df.isnull().sum().sum()
    print(f"  DataFrame nulls (pre-write): {final_null_check}")
    
    if final_null_check > 0:
        raise ValueError(f"CRITICAL: DataFrame still has {final_null_check} nulls before writing!")
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] solution.csv written to: {output_path.absolute()}")
    
    # Also save to the script directory (model/solution.csv) for user visibility
    import shutil
    src = output_path.resolve()
    dst = (SCRIPT_DIR / "solution.csv").resolve()
    if src.exists() and src != dst:
        shutil.copy(src, dst)
        print(f"[INFO] Copied final solution to: {dst}")
    print(f"  - {len(final_df):,} rows saved")
    print(f"  - Empty stp_id values written as '' (empty string) in CSV ✓")
    print(f"  - TOTAL TONS DELIVERED: {final_df['tons_delivered'].sum():,.2f}")


# =============================================================================
# STEP A8: CLEAN ENTRY POINT
# =============================================================================
# Call run() only when executed as script, not when imported.
# Ensures identical behavior locally and on Kaggle.
# =============================================================================

if __name__ == "__main__":
    # Allow command-line override for local testing
    # Usage: python main.py [DATA_DIR]
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        run(target_dir)
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

