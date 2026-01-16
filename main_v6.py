"""
OPTIMIZER v6 - FINAL OPTIMIZED VERSION

Key optimizations:
1. FULL TRUCKLOADS ONLY - 10 tons per delivery to minimize trips
2. NEAREST FARMS ONLY - minimize transport distance
3. PRE-DRAIN before lockouts - minimize overflow
4. ROUND-ROBIN across farms - spread the excess N
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v6 - FINAL VERSION")

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
    stp = data["stp"]
    farm = data["farm"]
    weather = data["weather"].copy()
    config = data["config"]
    
    # Distances - compute for all pairs
    distances = {}
    for _, s in stp.iterrows():
        distances[s["stp_id"]] = {}
        for _, f in farm.iterrows():
            distances[s["stp_id"]][f["farm_id"]] = haversine(
                s["lat"], s["lon"], f["lat"], f["lon"]
            )
    
    # Rain-lock
    threshold = 30.0
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
    
    # Find farms available per day
    farms_available = []
    for i, ds in enumerate(dates):
        avail = []
        for fid in farm["farm_id"]:
            zone = farm_zone[fid]
            if zone in zone_rolling and zone_rolling[zone][i] <= threshold:
                avail.append(fid)
        farms_available.append(avail)
    
    # For each day, find the NEAREST farms from each STP
    stp_nearest_avail = {}
    for i, ds in enumerate(dates):
        stp_nearest_avail[ds] = {}
        for sid in stp["stp_id"]:
            # Sort available farms by distance to this STP
            avail = farms_available[i]
            sorted_farms = sorted(avail, key=lambda f: distances[sid][f])
            stp_nearest_avail[ds][sid] = sorted_farms
    
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
    
    print(f"  Locked days: {sum(1 for f in farms_available if len(f) == 0)}")
    
    return {
        "distances": distances,
        "dates": dates,
        "farms_available": farms_available,
        "days_to_lockout": days_to_lockout,
        "stp_nearest_avail": stp_nearest_avail
    }


def optimize(data, precomputed):
    stp = data["stp"]
    farm = data["farm"]
    demand_df = data["demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    farms_available = precomputed["farms_available"]
    days_to_lockout = precomputed["days_to_lockout"]
    stp_nearest_avail = precomputed["stp_nearest_avail"]
    
    TRUCK_CAP = 10  # ALWAYS full truckloads
    N_PER_TON = 25
    BUFFER = 0.10
    
    # Annual demand
    demand_df = demand_df.copy()
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    annual_demand = {f: demand_df[f].sum() * (1 + BUFFER) for f in farm_cols}
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    total_daily = sum(stp_output.values())
    total_storage = sum(stp_max.values())
    
    # Initialize
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    cum_n = {f: 0.0 for f in farm_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    # Track which farms have received most, for round-robin
    farm_deliveries_today = {f: 0 for f in farm_ids}
    
    for day_i, ds in enumerate(dates):
        # Reset daily counters
        for f in farm_ids:
            farm_deliveries_today[f] = 0
        
        if (day_i + 1) % 100 == 0:
            print(f"  Day {day_i+1}: Storage {sum(storage.values()):.0f}, Overflow: {overflow_total:.0f}")
        
        # Add production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        available = farms_available[day_i]
        
        if not available:
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    overflow_total += storage[s] - stp_max[s]
                    storage[s] = stp_max[s]
            continue
        
        # Target storage based on upcoming lockout
        dtl = days_to_lockout[day_i]
        if dtl <= 2:
            target_pct = 0.0
        elif dtl <= 5:
            target_pct = 0.05
        elif dtl <= 10:
            target_pct = 0.15
        else:
            target_pct = 0.25
        
        # Deliver from each STP
        for stp_id in stp_ids:
            target = stp_max[stp_id] * target_pct
            need = storage[stp_id] - target
            
            if need < TRUCK_CAP:
                continue
            
            # Get nearest farms for this STP
            nearest = stp_nearest_avail[ds][stp_id]
            
            if not nearest:
                continue
            
            # Use top 30 nearest farms in round-robin
            top_farms = nearest[:30]
            
            farm_idx = 0
            max_iter = (len(top_farms) * 50)
            iteration = 0
            
            while need >= TRUCK_CAP and iteration < max_iter:
                iteration += 1
                
                # Pick farm with fewest deliveries today (round-robin)
                farm_id = min(top_farms, key=lambda f: (farm_deliveries_today[f], distances[stp_id][f]))
                
                # Full truckload only
                tons = TRUCK_CAP
                
                if storage[stp_id] < tons:
                    break
                
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": tons
                })
                
                storage[stp_id] -= tons
                need -= tons
                cum_n[farm_id] += tons * N_PER_TON
                farm_deliveries_today[farm_id] += 1
                distance_total += distances[stp_id][farm_id]
        
        # Overflow check
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
    
    # Build solution
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR: No deliveries!")
        return solution_df
    
    total_tons = solution_df["tons_delivered"].sum()
    effective_n = sum(min(cum_n[f], annual_demand.get(f, 0)) for f in farm_ids)
    excess_n = sum(max(0, cum_n[f] - annual_demand.get(f, 0)) for f in farm_ids)
    
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
    print(f"  Total tons: {total_tons:,.0f}")
    print(f"  Overflow: {overflow_total:,.0f}")
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
    
    print(f"\n[SAVED] {output_path}")
    print(f"  Rows: {len(sample):,}")
    print(f"  Non-zero deliveries: {(sample['tons_delivered'] > 0).sum():,}")
    print(f"  Total tons: {sample['tons_delivered'].sum():,.0f}")


def main():
    print("\nLoading data...")
    data = load_data()
    
    print("Precomputing...")
    precomputed = precompute(data)
    
    print("\nOptimizing...")
    solution = optimize(data, precomputed)
    
    if not solution.empty:
        build_submission(solution, data)
        print("\nDONE!")


if __name__ == "__main__":
    main()
