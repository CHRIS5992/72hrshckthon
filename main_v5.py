"""
OPTIMIZER v5 - DISTANCE OPTIMIZED

Improvements over v4:
1. Minimize transport distance by strictly using nearest farms
2. Still drain before lockouts to minimize overflow
3. Better farm-STP assignment based on distance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v5 - DISTANCE OPTIMIZED")

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_data():
    print("\n--- Loading ---")
    with open(DATA_DIR / "config.json") as f:
        config = json.load(f)
    
    return {
        "config": config,
        "stp": pd.read_csv(DATA_DIR / "stp_registry.csv"),
        "farm": pd.read_csv(DATA_DIR / "farm_locations.csv"),
        "weather": pd.read_csv(DATA_DIR / "daily_weather_2025.csv"),
        "demand": pd.read_csv(DATA_DIR / "daily_n_demand.csv"),
        "planting": pd.read_csv(DATA_DIR / "planting_schedule_2025.csv"),
    }


def precompute(data):
    print("--- Precomputing ---")
    
    stp = data["stp"]
    farm = data["farm"]
    weather = data["weather"].copy()
    config = data["config"]
    
    # Distances
    distances = {}
    for _, s in stp.iterrows():
        distances[s["stp_id"]] = {
            f["farm_id"]: haversine(s["lat"], s["lon"], f["lat"], f["lon"])
            for _, f in farm.iterrows()
        }
    
    # Rain-lock
    threshold = config.get("environmental_thresholds", {}).get("rain_lock_threshold_mm", 30.0)
    window = 5
    
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    dates = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zones = [c for c in weather.columns if c != "date"]
    
    zone_rolling = {}
    for zone in zones:
        vals = weather[zone].values
        n = len(vals)
        zone_rolling[zone] = np.array([np.sum(vals[i:min(i+window, n)]) for i in range(n)])
    
    farm_zone = dict(zip(farm["farm_id"], farm["zone"]))
    
    rain_lock = {}
    for fid in farm["farm_id"]:
        zone = farm_zone[fid]
        if zone in zone_rolling:
            rain_lock[fid] = {dates[i]: zone_rolling[zone][i] > threshold for i in range(len(dates))}
        else:
            rain_lock[fid] = {d: False for d in dates}
    
    # Available farms per day
    farms_available = []
    for i, ds in enumerate(dates):
        avail = [f for f in farm["farm_id"] if not rain_lock[f][ds]]
        farms_available.append(avail)
    
    # Days to lockout
    days_to_lockout = []
    for i in range(len(dates)):
        if len(farms_available[i]) == 0:
            days_to_lockout.append(0)
        else:
            found = False
            for j in range(i+1, len(dates)):
                if len(farms_available[j]) == 0:
                    days_to_lockout.append(j - i)
                    found = True
                    break
            if not found:
                days_to_lockout.append(999)
    
    # Assign each farm to its NEAREST STP for distance optimization
    farm_nearest_stp = {}
    for fid in farm["farm_id"]:
        nearest = min(stp["stp_id"], key=lambda s: distances[s][fid])
        farm_nearest_stp[fid] = nearest
    
    # Group farms by their nearest STP
    stp_farms = {s: [] for s in stp["stp_id"]}
    for fid, sid in farm_nearest_stp.items():
        stp_farms[sid].append(fid)
    
    # Sort farms in each group by distance
    for sid in stp_farms:
        stp_farms[sid].sort(key=lambda f: distances[sid][f])
    
    print(f"  Farm assignments: {', '.join(f'{s}: {len(stp_farms[s])}' for s in stp_farms)}")
    
    return {
        "distances": distances,
        "dates": dates,
        "farms_available": farms_available,
        "days_to_lockout": days_to_lockout,
        "farm_nearest_stp": farm_nearest_stp,
        "stp_farms": stp_farms
    }


def optimize(data, precomputed):
    print("\n--- Optimizing ---")
    
    stp = data["stp"]
    farm = data["farm"]
    demand_df = data["demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    farms_available = precomputed["farms_available"]
    days_to_lockout = precomputed["days_to_lockout"]
    farm_nearest_stp = precomputed["farm_nearest_stp"]
    stp_farms = precomputed["stp_farms"]
    
    truck_cap = config.get("logistics_constants", {}).get("truck_capacity_tons", 10)
    n_per_ton = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    buffer_pct = config.get("agronomic_constants", {}).get("application_buffer_percent", 10)
    
    # Annual demand per farm
    demand_df = demand_df.copy()
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    
    annual_demand = {f: demand_df[f].sum() * (1 + buffer_pct / 100) for f in farm_cols}
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    total_daily = sum(stp_output.values())
    total_storage = sum(stp_max.values())
    
    print(f"  Daily output: {total_daily} tons, Storage: {total_storage} tons")
    
    # Initialize
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    cumulative_n = {f: 0.0 for f in farm_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 100 == 0:
            curr = sum(storage.values())
            print(f"    Day {day_i+1}: Storage {curr:.0f}, Overflow: {overflow_total:.0f}, Distance: {distance_total:.0f}")
        
        # Add daily production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        available = set(farms_available[day_i])
        
        if not available:
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    overflow_total += storage[s] - stp_max[s]
                    storage[s] = stp_max[s]
            continue
        
        # Target storage based on upcoming lockout
        dtl = days_to_lockout[day_i]
        if dtl <= 3:
            target_pct = 0.0  # Drain completely before lockout
        elif dtl <= 10:
            target_pct = 0.1
        else:
            target_pct = 0.3
        
        # Deliver from each STP to its nearest farms
        for stp_id in stp_ids:
            target = stp_max[stp_id] * target_pct
            to_deliver = storage[stp_id] - target
            
            if to_deliver < 1:
                continue
            
            # Get available farms assigned to this STP, sorted by distance
            my_farms = [f for f in stp_farms[stp_id] if f in available]
            
            # If this STP has no available farms, use any available farm (nearest first)
            if not my_farms:
                all_sorted = sorted(available, key=lambda f: distances[stp_id][f])
                my_farms = all_sorted[:50]  # Limit to 50 nearest
            
            if not my_farms:
                continue
            
            farm_idx = 0
            iterations = 0
            max_iter = len(my_farms) * 30
            
            while to_deliver > 0.01 and iterations < max_iter:
                iterations += 1
                farm_id = my_farms[farm_idx % len(my_farms)]
                farm_idx += 1
                
                remaining_cap = annual_demand.get(farm_id, 0) - cumulative_n[farm_id]
                
                if remaining_cap > 0:
                    tons = min(truck_cap, remaining_cap / n_per_ton, to_deliver, storage[stp_id])
                else:
                    # Over capacity - use smaller loads
                    tons = min(truck_cap / 2, to_deliver, storage[stp_id])
                
                if tons < 0.1:
                    continue
                
                tons = round(tons, 3)
                
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": tons
                })
                
                storage[stp_id] -= tons
                to_deliver -= tons
                cumulative_n[farm_id] += tons * n_per_ton
                distance_total += distances[stp_id][farm_id]
        
        # Overflow check
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
    
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR: No deliveries!")
        return solution_df
    
    # Calculate score
    total_tons = solution_df["tons_delivered"].sum()
    effective_n = sum(min(cumulative_n[f], annual_demand.get(f, 0)) for f in farm_ids)
    excess_n = sum(max(0, cumulative_n[f] - annual_demand.get(f, 0)) for f in farm_ids)
    
    n_credit = effective_n * 5.0
    soil_credit = total_tons * 1000 * 0.2
    transport_cost = distance_total * 0.9
    excess_cost = excess_n * 10.0
    overflow_cost = overflow_total * 1000.0
    score = n_credit + soil_credit - transport_cost - excess_cost - overflow_cost
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Total tons: {total_tons:,.1f}")
    print(f"  Overflow: {overflow_total:,.1f}")
    print(f"  Effective N: {effective_n:,.0f} kg")
    print(f"  Excess N: {excess_n:,.0f} kg")
    print(f"  Distance: {distance_total:,.0f} km")
    print(f"\n  SCORE:")
    print(f"    + N credit:    {n_credit:>12,.0f}")
    print(f"    + Soil carbon: {soil_credit:>12,.0f}")
    print(f"    - Transport:   {transport_cost:>12,.0f}")
    print(f"    - Excess N:    {excess_cost:>12,.0f}")
    print(f"    - Overflow:    {overflow_cost:>12,.0f}")
    print(f"    -------------------------")
    print(f"    = TOTAL:       {score:>12,.0f}")
    print("=" * 60)
    
    return solution_df


def build_submission(solution_df, data):
    print("\n--- Building submission ---")
    
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    default_stp = data["stp"]["stp_id"].iloc[0]
    
    sample["stp_id"] = default_stp
    sample["tons_delivered"] = 0.0
    
    grouped = solution_df.groupby(["date", "farm_id"], as_index=False).agg({
        "stp_id": "first",
        "tons_delivered": "sum"
    })
    
    lookup = {
        (str(r["date"]), str(r["farm_id"])): (str(r["stp_id"]), float(r["tons_delivered"]))
        for _, r in grouped.iterrows()
    }
    
    for idx in range(len(sample)):
        key = (str(sample.at[idx, "date"]), str(sample.at[idx, "farm_id"]))
        if key in lookup:
            sample.at[idx, "stp_id"] = lookup[key][0]
            sample.at[idx, "tons_delivered"] = lookup[key][1]
    
    sample["stp_id"] = sample["stp_id"].fillna(default_stp).astype(str)
    sample["tons_delivered"] = sample["tons_delivered"].fillna(0.0).astype(float)
    sample = sample[["id", "date", "stp_id", "farm_id", "tons_delivered"]]
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "solution.csv"
    sample.to_csv(output_path, index=False)
    
    print(f"[SAVED] {output_path}")
    print(f"  Rows: {len(sample):,}, Non-zero: {(sample['tons_delivered'] > 0).sum():,}")
    print(f"  Total: {sample['tons_delivered'].sum():,.1f} tons")


def main():
    data = load_data()
    precomputed = precompute(data)
    solution = optimize(data, precomputed)
    if not solution.empty:
        build_submission(solution, data)


if __name__ == "__main__":
    main()
