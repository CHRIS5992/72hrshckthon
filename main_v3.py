"""
IMPROVED OPTIMIZER v3 - Balanced Distribution Strategy

Key insights:
1. Overflow penalty (-1000/ton) >> Excess N penalty (-250/ton theoretically)
2. But excess N penalty is -10/kg N = -250/ton which is still significant
3. We need to SPREAD deliveries across many farms to minimize per-farm excess
4. Prioritize farms that CAN absorb more N (large demand)

Strategy:
1. Deliver ALL biosolids (prevent overflow)
2. Spread across ALL available farms on each day
3. Limit per-farm delivery based on remaining capacity
4. Use distance-weighted allocation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

print("BOOT OK - Improved Optimizer v3")

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_all_data(data_dir):
    print("\n--- Loading Data ---")
    with open(data_dir / "config.json", 'r') as f:
        config = json.load(f)
    
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


def precompute(data):
    print("\n--- Precomputing ---")
    
    stp_df = data["stp_registry"]
    farm_df = data["farm_locations"]
    weather_df = data["daily_weather"]
    planting_df = data["planting_schedule"]
    config = data["config"]
    
    # Distances
    distances = {}
    for _, stp in stp_df.iterrows():
        stp_id = stp["stp_id"]
        distances[stp_id] = {}
        for _, farm in farm_df.iterrows():
            distances[stp_id][farm["farm_id"]] = haversine(
                stp["lat"], stp["lon"], farm["lat"], farm["lon"]
            )
    print(f"[DONE] Distances computed")
    
    # Rain-lock
    threshold = config.get("environmental_thresholds", {}).get("rain_lock_threshold_mm", 30.0)
    window = config.get("environmental_thresholds", {}).get("forecast_window_days", 5)
    
    farm_zone = dict(zip(farm_df["farm_id"], farm_df["zone"]))
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    date_strings = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zone_cols = [c for c in weather.columns if c != "date"]
    
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
    
    print(f"[DONE] Rain-lock computed")
    
    # Farm activity
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
            is_active[farm_id][ds] = any(p <= date <= h for p, h in periods)
    
    print(f"[DONE] Farm activity computed")
    
    return {
        "distances": distances,
        "rain_lock": rain_lock,
        "is_active": is_active,
        "date_strings": date_strings
    }


def generate_schedule_v3(data, precomputed):
    """
    BALANCED STRATEGY: Spread deliveries across all farms to minimize excess per farm.
    """
    print("\n" + "=" * 70)
    print("OPTIMIZER v3: BALANCED DISTRIBUTION")
    print("=" * 70)
    
    stp_df = data["stp_registry"]
    farm_df = data["farm_locations"]
    demand_df = data["daily_n_demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    rain_lock = precomputed["rain_lock"]
    is_active = precomputed["is_active"]
    date_strings = precomputed["date_strings"]
    
    truck_cap = config.get("logistics_constants", {}).get("truck_capacity_tons", 10)
    n_per_ton = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    buffer_pct = config.get("agronomic_constants", {}).get("application_buffer_percent", 10)
    
    # Parse demand
    demand_copy = demand_df.copy()
    demand_copy["date"] = pd.to_datetime(demand_copy["date"])
    farm_cols = [c for c in demand_copy.columns if c != "date"]
    
    # Daily demand per farm
    daily_demand = {}
    for _, row in demand_copy.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        daily_demand[ds] = {f: row[f] for f in farm_cols}
    
    stp_ids = stp_df["stp_id"].tolist()
    farm_ids = farm_df["farm_id"].tolist()
    stp_output = dict(zip(stp_df["stp_id"], stp_df["daily_output_tons"]))
    stp_max = dict(zip(stp_df["stp_id"], stp_df["storage_max_tons"]))
    
    total_daily = sum(stp_output.values())
    total_storage = sum(stp_max.values())
    
    print(f"  Daily output: {total_daily} tons")
    print(f"  Total storage: {total_storage} tons")
    
    # Initialize storage at 50%
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    
    # Track N delivered to each farm for the ENTIRE year
    cumulative_n = {f: 0.0 for f in farm_ids}
    
    # Calculate total annual demand per farm with buffer
    annual_demand = {}
    for f in farm_ids:
        total = sum(daily_demand[ds].get(f, 0) for ds in date_strings)
        annual_demand[f] = total * (1 + buffer_pct / 100)
    
    total_annual_n = sum(annual_demand.values())
    print(f"  Total annual N demand (with buffer): {total_annual_n:,.0f} kg")
    print(f"  Equivalent biosolids: {total_annual_n/n_per_ton:,.0f} tons")
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    # Pre-sort farms by distance from each STP
    stp_sorted_farms = {}
    for stp_id in stp_ids:
        stp_sorted_farms[stp_id] = sorted(farm_ids, key=lambda f: distances[stp_id][f])
    
    print(f"\n  Processing {len(date_strings)} days...")
    
    for day_idx, ds in enumerate(date_strings):
        if (day_idx + 1) % 50 == 0:
            curr = sum(storage.values())
            print(f"    Day {day_idx+1}: Storage {curr:.0f}/{total_storage}, Overflow: {overflow_total:.0f}")
        
        # Add daily production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        # Find available farms (not rain-locked)
        available = [f for f in farm_ids if not rain_lock.get(f, {}).get(ds, False)]
        
        if not available:
            # All farms locked - check overflow
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    overflow_total += storage[s] - stp_max[s]
                    storage[s] = stp_max[s]
            continue
        
        # Calculate how much we NEED to deliver today to prevent overflow
        # Target: keep storage at 30% to have buffer for locked days
        total_need = max(0, sum(storage.values()) - total_storage * 0.3)
        
        if total_need < 1:
            continue
        
        # Categorize farms by remaining capacity
        # Remaining = annual_demand - cumulative_n (can be negative)
        farm_remaining = {}
        for f in available:
            remaining = annual_demand[f] - cumulative_n[f]
            farm_remaining[f] = remaining
        
        # Sort by: 1) has positive remaining capacity, 2) nearest to any STP
        farms_with_capacity = [f for f in available if farm_remaining[f] > 0]
        farms_over_capacity = [f for f in available if farm_remaining[f] <= 0]
        
        # Allocate deliveries from each STP
        for stp_id in stp_ids:
            if storage[stp_id] <= stp_max[stp_id] * 0.3:
                continue  # This STP is fine
            
            to_deliver = storage[stp_id] - stp_max[stp_id] * 0.3
            
            # Preferred order: farms with capacity, sorted by distance
            sorted_available = stp_sorted_farms[stp_id]
            preferred = [f for f in sorted_available if f in farms_with_capacity]
            fallback = [f for f in sorted_available if f in farms_over_capacity]
            
            all_targets = preferred + fallback
            
            if not all_targets:
                continue
            
            # Distribute across farms - each farm gets a share proportional to remaining capacity
            # But limit to truck capacity per delivery
            farm_idx = 0
            while to_deliver > 0.01 and farm_idx < len(all_targets) * 5:
                farm_id = all_targets[farm_idx % len(all_targets)]
                farm_idx += 1
                
                # How much can this farm take?
                # If has remaining capacity, try to fill it
                # If over capacity, just give 1 truck load to spread it out
                remaining = farm_remaining.get(farm_id, 0)
                
                if remaining > 0:
                    max_this_farm = min(truck_cap, remaining / n_per_ton, to_deliver)
                else:
                    # Over capacity - give small amount to spread
                    max_this_farm = min(truck_cap / 2, to_deliver)
                
                if max_this_farm < 0.1:
                    continue
                
                tons = round(max_this_farm, 3)
                
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": tons
                })
                
                storage[stp_id] -= tons
                to_deliver -= tons
                n_added = tons * n_per_ton
                cumulative_n[farm_id] += n_added
                farm_remaining[farm_id] -= n_added
                distance_total += distances[stp_id][farm_id]
        
        # End of day overflow check
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
    
    # Build solution
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR: No deliveries!")
        return solution_df
    
    # Calculate score
    total_tons = solution_df["tons_delivered"].sum()
    total_n = total_tons * n_per_ton
    
    effective_n = sum(min(cumulative_n[f], annual_demand[f]) for f in farm_ids)
    excess_n = sum(max(0, cumulative_n[f] - annual_demand[f]) for f in farm_ids)
    
    n_credit = effective_n * 5.0
    soil_credit = total_tons * 1000 * 0.2
    transport_cost = distance_total * 0.9
    excess_cost = excess_n * 10.0
    overflow_cost = overflow_total * 1000.0
    
    score = n_credit + soil_credit - transport_cost - excess_cost - overflow_cost
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Total tons: {total_tons:,.1f}")
    print(f"  Overflow: {overflow_total:,.1f} tons")
    print(f"  Effective N: {effective_n:,.0f} kg")
    print(f"  Excess N: {excess_n:,.0f} kg")
    print(f"  Distance: {distance_total:,.0f} km")
    print(f"\n  SCORE:")
    print(f"    + N credit:     {n_credit:>15,.0f}")
    print(f"    + Soil carbon:  {soil_credit:>15,.0f}")
    print(f"    - Transport:    {transport_cost:>15,.0f}")
    print(f"    - Excess N:     {excess_cost:>15,.0f}")
    print(f"    - Overflow:     {overflow_cost:>15,.0f}")
    print(f"    --------------------------------")
    print(f"    = TOTAL:        {score:>15,.0f}")
    print("=" * 70)
    
    return solution_df


def run():
    print(f"\nData directory: {DATA_DIR}")
    
    for f in ["config.json", "stp_registry.csv", "farm_locations.csv", 
              "daily_weather_2025.csv", "daily_n_demand.csv", "planting_schedule_2025.csv"]:
        if not (DATA_DIR / f).exists():
            raise FileNotFoundError(f"Missing: {f}")
    
    data = load_all_data(DATA_DIR)
    precomputed = precompute(data)
    solution_df = generate_schedule_v3(data, precomputed)
    
    if solution_df.empty:
        print("ERROR: Empty solution!")
        return
    
    # Build submission
    print("\n--- Building submission ---")
    
    sample_path = DATA_DIR / "sample_submission.csv"
    final_df = pd.read_csv(sample_path)
    default_stp = data["stp_registry"]["stp_id"].iloc[0]
    
    final_df["stp_id"] = default_stp
    final_df["tons_delivered"] = 0.0
    
    grouped = solution_df.groupby(["date", "farm_id"], as_index=False).agg({
        "stp_id": "first",
        "tons_delivered": "sum"
    })
    
    lookup = {}
    for _, row in grouped.iterrows():
        key = (str(row["date"]), str(row["farm_id"]))
        lookup[key] = (str(row["stp_id"]), float(row["tons_delivered"]))
    
    for idx in range(len(final_df)):
        key = (str(final_df.at[idx, "date"]), str(final_df.at[idx, "farm_id"]))
        if key in lookup:
            stp_id, tons = lookup[key]
            final_df.at[idx, "stp_id"] = stp_id
            final_df.at[idx, "tons_delivered"] = tons
    
    final_df["stp_id"] = final_df["stp_id"].fillna(default_stp).astype(str)
    final_df["tons_delivered"] = final_df["tons_delivered"].fillna(0.0).astype(float)
    final_df = final_df[["id", "date", "stp_id", "farm_id", "tons_delivered"]]
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "solution.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Saved: {output_path}")
    print(f"  Rows: {len(final_df):,}")
    print(f"  Non-zero: {(final_df['tons_delivered'] > 0).sum():,}")
    print(f"  Total tons: {final_df['tons_delivered'].sum():,.1f}")


if __name__ == "__main__":
    run()
