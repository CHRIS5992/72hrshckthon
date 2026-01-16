"""
IMPROVED OPTIMIZER v2 - Maximum Throughput Strategy

KEY INSIGHT: Overflow penalty (-1000 CO2/ton) is FAR worse than excess N penalty (-250 CO2/ton).
Therefore, we MUST deliver ALL biosolids to farms, even if it exceeds N demand.

Strategy:
1. Prioritize farms with N demand (to get the +5 CO2/kg N credit)
2. Once demand is met, continue delivering to ANY available farm
3. Goal: ZERO overflow, maximize biosolid application
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

print("BOOT OK - Improved Optimizer v2")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

REQUIRED_FILES = [
    "config.json",
    "stp_registry.csv",
    "farm_locations.csv",
    "daily_weather_2025.csv",
    "daily_n_demand.csv",
    "planting_schedule_2025.csv"
]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(data_dir: Path) -> dict:
    """Load all required data files."""
    print("\n--- Loading Data ---")
    
    with open(data_dir / "config.json", 'r') as f:
        config = json.load(f)
    print("[LOADED] config.json")
    
    data = {
        "config": config,
        "stp_registry": pd.read_csv(data_dir / "stp_registry.csv"),
        "farm_locations": pd.read_csv(data_dir / "farm_locations.csv"),
        "daily_weather": pd.read_csv(data_dir / "daily_weather_2025.csv"),
        "daily_n_demand": pd.read_csv(data_dir / "daily_n_demand.csv"),
        "planting_schedule": pd.read_csv(data_dir / "planting_schedule_2025.csv"),
    }
    
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"[LOADED] {name} ({len(df)} rows)")
    
    return data


# =============================================================================
# HAVERSINE DISTANCE
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Compute distance in km between two coordinates."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# =============================================================================
# PRECOMPUTATION
# =============================================================================

def precompute(data: dict) -> dict:
    """Precompute distances, rain-lock, and farm activity."""
    print("\n--- Precomputing ---")
    
    stp_df = data["stp_registry"]
    farm_df = data["farm_locations"]
    weather_df = data["daily_weather"]
    planting_df = data["planting_schedule"]
    config = data["config"]
    
    # 1. Distances
    distances = {}
    for _, stp in stp_df.iterrows():
        stp_id = stp["stp_id"]
        distances[stp_id] = {}
        for _, farm in farm_df.iterrows():
            farm_id = farm["farm_id"]
            distances[stp_id][farm_id] = haversine(
                stp["lat"], stp["lon"], farm["lat"], farm["lon"]
            )
    print(f"[DONE] Computed {len(stp_df) * len(farm_df)} distances")
    
    # 2. Rain-lock (5-day forward cumulative > 30mm)
    threshold = config.get("environmental_thresholds", {}).get("rain_lock_threshold_mm", 30.0)
    window = config.get("environmental_thresholds", {}).get("forecast_window_days", 5)
    
    farm_zone = dict(zip(farm_df["farm_id"], farm_df["zone"]))
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    date_strings = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zone_cols = [c for c in weather.columns if c != "date"]
    
    # Forward rolling sum for each zone
    zone_rolling = {}
    for zone in zone_cols:
        vals = weather[zone].values
        n = len(vals)
        rolling = np.zeros(n)
        for i in range(n):
            rolling[i] = np.sum(vals[i:min(i+window, n)])
        zone_rolling[zone] = rolling
    
    rain_lock = {}
    for farm_id in farm_df["farm_id"]:
        zone = farm_zone[farm_id]
        rain_lock[farm_id] = {}
        if zone in zone_rolling:
            for i, ds in enumerate(date_strings):
                rain_lock[farm_id][ds] = zone_rolling[zone][i] > threshold
        else:
            for ds in date_strings:
                rain_lock[farm_id][ds] = False
    
    locked_count = sum(1 for f in rain_lock for d in rain_lock[f] if rain_lock[f][d])
    print(f"[DONE] {locked_count} farm-days rain-locked")
    
    # 3. Farm activity (between plant and harvest)
    schedule = planting_df.copy()
    schedule["plant_date"] = pd.to_datetime(schedule["plant_date"])
    schedule["harvest_date"] = pd.to_datetime(schedule["harvest_date"])
    
    all_dates = sorted(weather["date"].unique())
    
    farm_periods = {}
    for _, row in schedule.iterrows():
        fid = row["farm_id"]
        if fid not in farm_periods:
            farm_periods[fid] = []
        farm_periods[fid].append((row["plant_date"], row["harvest_date"]))
    
    is_active = {}
    for farm_id in farm_df["farm_id"]:
        is_active[farm_id] = {}
        periods = farm_periods.get(farm_id, [])
        for i, date in enumerate(all_dates):
            ds = date_strings[i]
            active = any(p <= date <= h for p, h in periods)
            is_active[farm_id][ds] = active
    
    active_count = sum(1 for f in is_active for d in is_active[f] if is_active[f][d])
    print(f"[DONE] {active_count} farm-days are active")
    
    return {
        "distances": distances,
        "rain_lock": rain_lock,
        "is_active": is_active,
        "date_strings": date_strings
    }


# =============================================================================
# IMPROVED OPTIMIZER - DRAIN EVERYTHING STRATEGY
# =============================================================================

def generate_schedule_v2(data: dict, precomputed: dict) -> pd.DataFrame:
    """
    IMPROVED STRATEGY: Maximize throughput, minimize overflow.
    
    Priority order:
    1. Farms with N demand (get +5 CO2/kg N credit)
    2. Active farms without rain-lock (any delivery is good)
    3. Any available farm (even inactive, if not rain-locked)
    
    Key: It's MUCH better to over-apply (-250 CO2/ton) than overflow (-1000 CO2/ton)
    """
    print("\n" + "=" * 70)
    print("IMPROVED OPTIMIZER v2: MAXIMUM THROUGHPUT STRATEGY")
    print("=" * 70)
    
    stp_df = data["stp_registry"]
    farm_df = data["farm_locations"]
    demand_df = data["daily_n_demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    rain_lock = precomputed["rain_lock"]
    is_active = precomputed["is_active"]
    date_strings = precomputed["date_strings"]
    
    # Config
    truck_cap = config.get("logistics_constants", {}).get("truck_capacity_tons", 10)
    n_per_ton = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    buffer_pct = config.get("agronomic_constants", {}).get("application_buffer_percent", 10)
    
    # Parse demand data
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
    stp_output = dict(zip(stp_df["stp_id"], stp_df["daily_output_tons"]))
    stp_max = dict(zip(stp_df["stp_id"], stp_df["storage_max_tons"]))
    
    total_output = sum(stp_output.values())
    total_max = sum(stp_max.values())
    
    print(f"  Daily STP output: {total_output} tons")
    print(f"  Total storage: {total_max} tons")
    print(f"  Annual production: {total_output * 365:,.0f} tons")
    
    # Initialize storage at 50%
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    
    # Track cumulative N delivered per farm (for scoring)
    n_delivered = {f: 0.0 for f in farm_ids}
    
    # Precompute total annual demand per farm (for prioritization)
    total_annual_demand = {}
    for f in farm_ids:
        total_annual_demand[f] = sum(demand.get(f, {}).values()) * (1 + buffer_pct / 100)
    
    deliveries = []
    overflow_tons = 0.0
    total_distance = 0.0
    
    # For each STP, precompute sorted farms by distance
    stp_farm_by_dist = {}
    for stp_id in stp_ids:
        sorted_farms = sorted(farm_ids, key=lambda f: distances[stp_id][f])
        stp_farm_by_dist[stp_id] = sorted_farms
    
    print(f"\n  Processing {len(date_strings)} days...")
    
    for day_i, ds in enumerate(date_strings):
        if (day_i + 1) % 50 == 0:
            curr_storage = sum(storage.values())
            print(f"    Day {day_i+1}: Storage {curr_storage:.0f}/{total_max}, Overflow so far: {overflow_tons:.0f}")
        
        # Add daily production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        # Find available farms (not rain-locked)
        available_farms = [f for f in farm_ids if not rain_lock.get(f, {}).get(ds, False)]
        
        if not available_farms:
            # All farms locked - overflow is inevitable
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    overflow_tons += storage[s] - stp_max[s]
                    storage[s] = stp_max[s]
            continue
        
        # Categorize farms:
        # Priority 1: Has remaining N demand today
        # Priority 2: Active (has crop) but no remaining demand
        # Priority 3: Inactive (will accept anyway to prevent overflow)
        
        farms_with_demand = []
        farms_active_no_demand = []
        farms_inactive = []
        
        for f in available_farms:
            day_demand = demand.get(f, {}).get(ds, 0)
            remaining_demand = max(0, total_annual_demand[f] - n_delivered[f])
            active = is_active.get(f, {}).get(ds, False)
            
            if day_demand > 0 or remaining_demand > 0:
                farms_with_demand.append(f)
            elif active:
                farms_active_no_demand.append(f)
            else:
                farms_inactive.append(f)
        
        # Deliver from each STP until storage is comfortable
        for stp_id in stp_ids:
            sorted_farms = stp_farm_by_dist[stp_id]
            
            # Target: keep storage below 80% to have buffer
            target_storage = stp_max[stp_id] * 0.3
            
            # Prioritize farms with demand, then active, then any
            priority_farms = (
                [f for f in sorted_farms if f in farms_with_demand] +
                [f for f in sorted_farms if f in farms_active_no_demand] +
                [f for f in sorted_farms if f in farms_inactive]
            )
            
            farm_idx = 0
            max_deliveries = len(priority_farms) * 10  # Safety limit
            delivery_count = 0
            
            while storage[stp_id] > target_storage and farm_idx < len(priority_farms) and delivery_count < max_deliveries:
                farm_id = priority_farms[farm_idx % len(priority_farms)]
                farm_idx += 1
                delivery_count += 1
                
                # Calculate tonnage
                tons = min(truck_cap, storage[stp_id] - target_storage + 0.1)
                tons = max(0, min(tons, storage[stp_id]))
                tons = round(tons, 3)
                
                if tons < 0.001:
                    break
                
                # Record delivery
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": tons
                })
                
                storage[stp_id] -= tons
                n_delivered[farm_id] += tons * n_per_ton
                total_distance += distances[stp_id][farm_id]
        
        # End of day overflow check
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_tons += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
    
    # =========================================================================
    # SCORING CALCULATION
    # =========================================================================
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR: No deliveries generated!")
        return solution_df
    
    total_tons = solution_df['tons_delivered'].sum()
    total_n = total_tons * n_per_ton
    
    # Calculate effective N (capped at demand + buffer) and excess
    effective_n = 0.0
    excess_n = 0.0
    for f in farm_ids:
        delivered = n_delivered[f]
        max_allowed = total_annual_demand[f]
        effective_n += min(delivered, max_allowed)
        if delivered > max_allowed:
            excess_n += delivered - max_allowed
    
    # Score components
    n_credit = effective_n * 5.0
    soil_credit = total_tons * 1000 * 0.2  # Convert tons to kg
    transport_penalty = total_distance * 0.9
    excess_penalty = excess_n * 10.0
    overflow_penalty = overflow_tons * 1000.0
    
    score = n_credit + soil_credit - transport_penalty - excess_penalty - overflow_penalty
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Biosolids delivered: {total_tons:,.1f} tons")
    print(f"  Overflow: {overflow_tons:,.1f} tons")
    print(f"  Effective N: {effective_n:,.0f} kg")
    print(f"  Excess N: {excess_n:,.0f} kg")
    print(f"  Total distance: {total_distance:,.0f} km")
    print(f"\n  SCORE BREAKDOWN:")
    print(f"    + N offset credit:    {n_credit:>15,.0f} CO2")
    print(f"    + Soil carbon:        {soil_credit:>15,.0f} CO2")
    print(f"    - Transport:          {transport_penalty:>15,.0f} CO2")
    print(f"    - Excess N penalty:   {excess_penalty:>15,.0f} CO2")
    print(f"    - Overflow penalty:   {overflow_penalty:>15,.0f} CO2")
    print(f"    ─────────────────────────────────")
    print(f"    = ESTIMATED SCORE:    {score:>15,.0f} CO2")
    print("=" * 70)
    
    return solution_df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run():
    """Main entry point."""
    print(f"\nData directory: {DATA_DIR}")
    
    # Check files
    for f in REQUIRED_FILES:
        if not (DATA_DIR / f).exists():
            raise FileNotFoundError(f"Missing: {f}")
    print("[OK] All files found")
    
    # Load data
    data = load_all_data(DATA_DIR)
    
    # Precompute
    precomputed = precompute(data)
    
    # Generate schedule
    solution_df = generate_schedule_v2(data, precomputed)
    
    if solution_df.empty:
        print("ERROR: Empty solution!")
        return
    
    # =========================================================================
    # BUILD KAGGLE SUBMISSION
    # =========================================================================
    print("\n--- Building Kaggle submission ---")
    
    # Load sample submission as base
    sample_path = DATA_DIR / "sample_submission.csv"
    if sample_path.exists():
        final_df = pd.read_csv(sample_path)
        print(f"  Loaded sample_submission: {len(final_df):,} rows")
    else:
        print("  WARNING: sample_submission.csv not found!")
        return
    
    default_stp = data["stp_registry"]["stp_id"].iloc[0]
    
    # Initialize with defaults
    final_df["stp_id"] = default_stp
    final_df["tons_delivered"] = 0.0
    
    # Group solution by (date, farm_id) and sum tons
    grouped = solution_df.groupby(["date", "farm_id"], as_index=False).agg({
        "stp_id": "first",
        "tons_delivered": "sum"
    })
    
    # Build lookup
    lookup = {}
    for _, row in grouped.iterrows():
        key = (str(row["date"]), str(row["farm_id"]))
        lookup[key] = (str(row["stp_id"]), float(row["tons_delivered"]))
    
    print(f"  Unique (date, farm) pairs in solution: {len(lookup):,}")
    
    # Fill final_df
    filled = 0
    for idx in range(len(final_df)):
        key = (str(final_df.at[idx, "date"]), str(final_df.at[idx, "farm_id"]))
        if key in lookup:
            stp_id, tons = lookup[key]
            final_df.at[idx, "stp_id"] = stp_id
            final_df.at[idx, "tons_delivered"] = tons
            filled += 1
    
    print(f"  Rows filled: {filled:,}")
    print(f"  Total tons in submission: {final_df['tons_delivered'].sum():,.1f}")
    
    # Ensure no nulls
    final_df["stp_id"] = final_df["stp_id"].fillna(default_stp).astype(str)
    final_df["tons_delivered"] = final_df["tons_delivered"].fillna(0.0).astype(float)
    final_df["date"] = final_df["date"].astype(str)
    final_df["farm_id"] = final_df["farm_id"].astype(str)
    
    # Ensure column order
    final_df = final_df[["id", "date", "stp_id", "farm_id", "tons_delivered"]]
    
    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "solution.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved to: {output_path}")
    print(f"  Rows: {len(final_df):,}")
    print(f"  Non-zero deliveries: {(final_df['tons_delivered'] > 0).sum():,}")


if __name__ == "__main__":
    run()
