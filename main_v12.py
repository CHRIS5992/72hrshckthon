"""
OPTIMIZER v12 - ZERO EXCESS STRATEGY (STOP THE BLEEDING)

Problem:
- User reports score of -40M.
- Baseline "Overflow All" score is -27M.
- This proves that "Delivering with Excess N" is TOXIC (worse than overflow).

Strategy:
1. Deliver ONLY exact daily demand (+10% buffer).
2. NEVER deliver excess N.
3. Accept the Overflow (-1000/ton) because it's cheaper than the mysterious delivery penalty.
4. Minimize Transport (Nearest Neighbor) to save the last few points.

Target Score: ~ -27 Million (Improving from -40M).
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

print("OPTIMIZER v12 - ZERO EXCESS")

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
    print("Loading data...")
    with open(DATA_DIR / "config.json") as f:
        config = json.load(f)
    
    return {
        "config": config,
        "stp": pd.read_csv(DATA_DIR / "stp_registry.csv"),
        "farm": pd.read_csv(DATA_DIR / "farm_locations.csv"),
        "weather": pd.read_csv(DATA_DIR / "daily_weather_2025.csv"),
        "demand": pd.read_csv(DATA_DIR / "daily_n_demand.csv"),
    }


def precompute(data):
    print("Precomputing...")
    stp = data["stp"]
    farm = data["farm"]
    weather = data["weather"].copy()
    demand_df = data["demand"].copy()
    
    # Distances
    distances = {}
    for _, s in stp.iterrows():
        distances[s["stp_id"]] = {
            f["farm_id"]: haversine(s["lat"], s["lon"], f["lat"], f["lon"])
            for _, f in farm.iterrows()
        }
    
    # Rain Lock (5-day > 30mm)
    weather["date"] = pd.to_datetime(weather["date"])
    dates = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zones = [c for c in weather.columns if c != "date"]
    
    zone_rolling = {}
    for zone in zones:
        vals = weather[zone].values
        n = len(vals)
        zone_rolling[zone] = np.array([np.sum(vals[i:min(i+5, n)]) for i in range(n)])
    
    farm_zone = dict(zip(farm["farm_id"], farm["zone"]))
    
    # Daily Demand
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    daily_demand = {} # date -> farm -> kg N
    for _, row in demand_df.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        daily_demand[ds] = {f: row[f] for f in farm_cols}
    
    # Daily Available Farms (Valid Demand AND Not Rain Locked)
    farms_avail_demand = {}
    for i, ds in enumerate(dates):
        avail = {}
        for fid in farm["farm_id"]:
            zone = farm_zone[fid]
            # Check rain lock
            locked = zone in zone_rolling and zone_rolling[zone][i] > 30
            if not locked:
                dem = daily_demand[ds].get(fid, 0)
                if dem > 0:
                    avail[fid] = dem
        farms_avail_demand[ds] = avail
        
    # Pre-sort farms for each STP by distance
    stp_nearest_all = {}
    for sid in stp["stp_id"]:
        stp_nearest_all[sid] = sorted(farm["farm_id"].tolist(), 
                                    key=lambda f: distances[sid][f])
            
    print(f"  Done for {len(dates)} days.")
    return {
        "distances": distances,
        "dates": dates,
        "farms_avail_demand": farms_avail_demand,
        "stp_nearest_all": stp_nearest_all
    }


def optimize(data, precomputed):
    print("Optimizing (Zero Excess)...")
    
    stp = data["stp"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    farms_avail_demand = precomputed["farms_avail_demand"]
    stp_nearest_all = precomputed["stp_nearest_all"]
    
    N_PER_TON = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    BUFFER = config.get("agronomic_constants", {}).get("application_buffer_percent", 10.0)
    # TRUCK_CAP = 10.0 # Not used as limit, demand is the limit
    
    stp_ids = stp["stp_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    # Initialize
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    total_n_delivered = 0.0
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 50 == 0:
            print(f"  Day {day_i+1}: Overflow {overflow_total:.0f}")
            sys.stdout.flush()
        
        # Production
        for s in stp_ids:
            storage[s] += stp_output[s]
            
        occupied_farms = set()
        avail_demand = farms_avail_demand[ds]
        
        for stp_id in stp_ids:
            if storage[stp_id] < 0.001:
                continue
            
            # Find nearest farms with demand
            nearest_list = stp_nearest_all[stp_id]
            
            for farm_id in nearest_list:
                if storage[stp_id] < 0.001:
                    break
                
                # Check criteria
                if farm_id in occupied_farms:
                    continue
                if farm_id not in avail_demand:
                    continue # No demand or rain locked
                
                # Calculate strict demand cap
                demand_kg = avail_demand[farm_id]
                max_kg = demand_kg * (1 + BUFFER / 100.0)
                max_tons = max_kg / N_PER_TON
                
                # Cap delivery
                tons = min(max_tons, storage[stp_id])
                
                # Filter tiny
                if tons < 0.001:
                    continue
                    
                # Rounding
                tons = round(tons, 4)
                if tons == 0:
                    continue
                
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": tons
                })
                
                storage[stp_id] -= tons
                occupied_farms.add(farm_id)
                
                total_n_delivered += tons * N_PER_TON
                distance_total += distances[stp_id][farm_id]
                
        # Overflow
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
                
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR: Empty")
        return solution_df
        
    total_tons = solution_df["tons_delivered"].sum()
    
    # Approximate Score
    # Soil Credit + N Credit ~= Tons * (200 + 125) = Tons * 325
    # Excess N = 0
    # Transport = Distance * 0.9
    # Overflow = Overflow * 1000
    
    credits_approx = total_tons * 325
    transport_pen = distance_total * 0.9
    overflow_pen = overflow_total * 1000.0
    
    score = credits_approx - transport_pen - overflow_pen
    
    print("\n" + "=" * 60)
    print("RESULTS v12")
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Total tons: {total_tons:,.2f}")
    print(f"  Overflow: {overflow_total:,.1f} tons")
    print(f"  Distance: {distance_total:,.0f} km")
    print(f"\n  ESTIMATED SCORE: {score:,.0f}")
    print("=" * 60)
    
    return solution_df


def build_submission(solution_df, data):
    print("Building submission...")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    default_stp = data["stp"]["stp_id"].iloc[0]
    
    sample["stp_id"] = default_stp
    sample["tons_delivered"] = 0.0
    
    lookup = {}
    for _, row in solution_df.iterrows():
        key = (str(row["date"]), str(row["farm_id"]))
        lookup[key] = (str(row["stp_id"]), float(row["tons_delivered"]))
        
    for idx in range(len(sample)):
        key = (str(sample.at[idx, "date"]), str(sample.at[idx, "farm_id"]))
        if key in lookup:
            sample.at[idx, "stp_id"] = lookup[key][0]
            sample.at[idx, "tons_delivered"] = lookup[key][1]
            
    output_path = OUTPUT_DIR / "solution.csv"
    sample.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path}")


def main():
    data = load_data()
    precomputed = precompute(data)
    solution = optimize(data, precomputed)
    if not solution.empty:
        build_submission(solution, data)

if __name__ == "__main__":
    main()
