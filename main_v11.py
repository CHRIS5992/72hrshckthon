"""
OPTIMIZER v11 - AGGRESSIVE SAFE DRAIN

Changes from v10:
1. ALWAYS target 0% storage (drain immediately)
   - Minimizes overflow during rain locks
   - Maximizes "Soil Carbon Credit" early
2. Retains strict "Safe" constraints (One Farm One STP, 10-ton cap)
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

print("OPTIMIZER v11 - AGGRESSIVE SAFE DRAIN")

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
    
    # Distances
    distances = {}
    for _, s in stp.iterrows():
        distances[s["stp_id"]] = {
            f["farm_id"]: haversine(s["lat"], s["lon"], f["lat"], f["lon"])
            for _, f in farm.iterrows()
        }
    
    # Rain Lock
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    dates = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zones = [c for c in weather.columns if c != "date"]
    
    zone_rolling = {}
    for zone in zones:
        vals = weather[zone].values
        n = len(vals)
        zone_rolling[zone] = np.array([np.sum(vals[i:min(i+5, n)]) for i in range(n)])
    
    farm_zone = dict(zip(farm["farm_id"], farm["zone"]))
    
    # Precompute nearest farms and rain lock
    stp_nearest_all = {}
    for sid in stp["stp_id"]:
        stp_nearest_all[sid] = sorted(farm["farm_id"].tolist(), 
                                    key=lambda f: distances[sid][f])
    
    rain_lock = {}
    for i, ds in enumerate(dates):
        rain_lock[ds] = {}
        for fid in farm["farm_id"]:
            zone = farm_zone[fid]
            locked = False
            if zone in zone_rolling and zone_rolling[zone][i] > 30:
                locked = True
            rain_lock[ds][fid] = locked
            
    print(f"  Done for {len(dates)} days.")
    return {
        "distances": distances,
        "dates": dates,
        "rain_lock": rain_lock,
        "stp_nearest_all": stp_nearest_all
    }


def optimize(data, precomputed):
    print("Optimizing...")
    
    stp = data["stp"]
    farm = data["farm"]
    demand_df = data["demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    rain_lock = precomputed["rain_lock"]
    stp_nearest_all = precomputed["stp_nearest_all"]
    
    N_PER_TON = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    TRUCK_CAP = config.get("logistics_constants", {}).get("truck_capacity_tons", 10.0)
    
    # Annual demand stats
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    annual_demand = {f: demand_df[f].sum() * 1.1 for f in farm_cols}
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    # Initialize
    # Assume 50% start
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    cum_n = {f: 0.0 for f in farm_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 50 == 0:
            print(f"  Day {day_i+1}: Overflow so far {overflow_total:.0f}")
            sys.stdout.flush()
        
        # Production
        for s in stp_ids:
            storage[s] += stp_output[s]
            
        occupied_farms = set()
        
        for stp_id in stp_ids:
            # AGGRESSIVE: Target = 0.0 always
            target = 0.0
            need = storage[stp_id] - target
            
            if need < 0.1:
                continue
                
            nearest_list = stp_nearest_all[stp_id]
            
            # Find candidates (Not locked, Not occupied)
            candidates = []
            count = 0
            for f in nearest_list:
                if f not in occupied_farms and not rain_lock[ds][f]:
                    candidates.append(f)
                    count += 1
                if count >= 40: 
                    break
            
            cand_idx = 0
            while need > 0.1 and cand_idx < len(candidates):
                farm_id = candidates[cand_idx]
                cand_idx += 1
                
                if storage[stp_id] < 0.1:
                    break
                
                tons = min(10.0, need, storage[stp_id])
                
                if tons < 0.1:
                    continue
                    
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": round(tons, 3)
                })
                
                storage[stp_id] -= tons
                need -= tons
                
                occupied_farms.add(farm_id)
                cum_n[farm_id] += tons * N_PER_TON
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
    effective_n = sum(min(cum_n[f], annual_demand[f]) for f in farm_ids)
    excess_n = sum(max(0, cum_n[f] - annual_demand[f]) for f in farm_ids)
    
    n_credit = effective_n * 5.0
    soil_credit = total_tons * 1000 * 0.2
    transport_cost = distance_total * 0.9
    excess_cost = excess_n * 10.0
    overflow_cost = overflow_total * 1000.0
    score = n_credit + soil_credit - transport_cost - excess_cost - overflow_cost
    
    print("\n" + "=" * 60)
    print("RESULTS v11")
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Total tons: {total_tons:,.1f}")
    print(f"  Overflow: {overflow_total:,.1f} tons")
    print(f"  Excess N: {excess_n:,.0f} kg")
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
            
    sample["tons_delivered"] = sample["tons_delivered"].clip(upper=10.0)
    
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
