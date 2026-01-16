"""
OPTIMIZER v7 - DEMAND-WEIGHTED DISTRIBUTION

Changes from v6:
1. Prioritize farms with HIGHEST annual demand
2. Each farm receives up to their demand + buffer before others
3. Only then spread excess to remaining farms
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v7 - DEMAND-WEIGHTED DISTRIBUTION")

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
    }


def precompute(data):
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
    
    # Annual demand per farm
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    annual_demand = {f: demand_df[f].sum() * 1.1 for f in farm_cols}  # with 10% buffer
    
    # Rain-lock
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
    
    farms_available = []
    for i in range(len(dates)):
        avail = [f for f in farm["farm_id"] 
                 if farm_zone[f] in zone_rolling and zone_rolling[farm_zone[f]][i] <= 30]
        farms_available.append(set(avail))
    
    # Days to lockout
    days_to_lockout = []
    for i in range(len(dates)):
        if len(farms_available[i]) == 0:
            days_to_lockout.append(0)
        else:
            dtl = 999
            for j in range(i+1, min(i+50, len(dates))):
                if len(farms_available[j]) == 0:
                    dtl = j - i
                    break
            days_to_lockout.append(dtl)
    
    # For each STP, rank farms by: 1) high demand, 2) short distance
    # Score = demand / distance (higher is better)
    stp_farm_priority = {}
    for sid in stp["stp_id"]:
        farms_scored = []
        for fid in farm["farm_id"]:
            demand = annual_demand.get(fid, 0)
            dist = distances[sid][fid]
            # Priority: high demand, low distance
            # We want farms where we get most N credit per km
            if dist > 0:
                score = demand / dist  # N per km
            else:
                score = float('inf')
            farms_scored.append((fid, score, demand, dist))
        
        # Sort by score descending (high demand per km first)
        farms_scored.sort(key=lambda x: -x[1])
        stp_farm_priority[sid] = [f[0] for f in farms_scored]
    
    print(f"  Locked days: {sum(1 for f in farms_available if len(f) == 0)}")
    print(f"  Total annual demand: {sum(annual_demand.values()):.0f} kg N")
    
    return {
        "distances": distances,
        "dates": dates,
        "farms_available": farms_available,
        "days_to_lockout": days_to_lockout,
        "annual_demand": annual_demand,
        "stp_farm_priority": stp_farm_priority
    }


def optimize(data, precomputed):
    stp = data["stp"]
    farm = data["farm"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    farms_available = precomputed["farms_available"]
    days_to_lockout = precomputed["days_to_lockout"]
    annual_demand = precomputed["annual_demand"]
    stp_farm_priority = precomputed["stp_farm_priority"]
    
    TRUCK_CAP = 10
    N_PER_TON = 25
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    total_storage = sum(stp_max.values())
    
    # Initialize
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    cum_n = {f: 0.0 for f in farm_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    for day_i, ds in enumerate(dates):
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
        
        # Target based on lockout
        dtl = days_to_lockout[day_i]
        if dtl <= 2:
            target_pct = 0.0
        elif dtl <= 5:
            target_pct = 0.05
        elif dtl <= 10:
            target_pct = 0.1
        else:
            target_pct = 0.2
        
        # Calculate remaining capacity for each farm
        farm_remaining = {f: max(0, annual_demand[f] - cum_n[f]) for f in farm_ids}
        
        for stp_id in stp_ids:
            target = stp_max[stp_id] * target_pct
            need = storage[stp_id] - target
            
            if need < TRUCK_CAP:
                continue
            
            # Get priority farms that are available
            priority = [f for f in stp_farm_priority[stp_id] if f in available]
            
            if not priority:
                continue
            
            # Phase 1: Deliver to farms with remaining capacity (by priority)
            farms_with_cap = [f for f in priority if farm_remaining[f] > 0]
            
            for farm_id in farms_with_cap:
                if need < TRUCK_CAP:
                    break
                if storage[stp_id] < TRUCK_CAP:
                    break
                
                # How many loads can this farm take?
                max_loads = int(farm_remaining[farm_id] / (TRUCK_CAP * N_PER_TON)) + 1
                loads_to_send = min(max_loads, int(need / TRUCK_CAP))
                
                for _ in range(loads_to_send):
                    if need < TRUCK_CAP or storage[stp_id] < TRUCK_CAP:
                        break
                    
                    deliveries.append({
                        "date": ds,
                        "stp_id": stp_id,
                        "farm_id": farm_id,
                        "tons_delivered": TRUCK_CAP
                    })
                    
                    storage[stp_id] -= TRUCK_CAP
                    need -= TRUCK_CAP
                    cum_n[farm_id] += TRUCK_CAP * N_PER_TON
                    farm_remaining[farm_id] -= TRUCK_CAP * N_PER_TON
                    distance_total += distances[stp_id][farm_id]
            
            # Phase 2: If still need to deliver, spread across nearest farms
            if need >= TRUCK_CAP:
                # Sort remaining available by distance
                nearest = sorted([f for f in priority[:50]], 
                               key=lambda f: distances[stp_id][f])
                
                farm_idx = 0
                while need >= TRUCK_CAP and farm_idx < len(nearest) * 10:
                    farm_id = nearest[farm_idx % len(nearest)]
                    farm_idx += 1
                    
                    if storage[stp_id] < TRUCK_CAP:
                        break
                    
                    deliveries.append({
                        "date": ds,
                        "stp_id": stp_id,
                        "farm_id": farm_id,
                        "tons_delivered": TRUCK_CAP
                    })
                    
                    storage[stp_id] -= TRUCK_CAP
                    need -= TRUCK_CAP
                    cum_n[farm_id] += TRUCK_CAP * N_PER_TON
                    distance_total += distances[stp_id][farm_id]
        
        # Overflow
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
    
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR!")
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
    print("RESULTS")
    print("=" * 60)
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Total tons: {total_tons:,.0f}")
    print(f"  Overflow: {overflow_total:,.0f}")
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
    print(f"  Non-zero: {(sample['tons_delivered'] > 0).sum():,}")
    print(f"  Total: {sample['tons_delivered'].sum():,.0f} tons")


def main():
    print("\nLoading...")
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
