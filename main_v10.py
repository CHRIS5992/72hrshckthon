"""
OPTIMIZER v10 - THE SAFE DRAIN STRATEGY

Core Features:
1. SAFE DRAIN: Drains storage to prevent overflow (-1000 penalty)
2. NEAREST NEIGHBOR: Always prioritizes closest farms to minimize transport
3. ROUND ROBIN: Enforces "One STP per Farm per Day" to prevent simulation conflicts
4. STRICT COMPLIANCE: Caps deliveries at 10.0 tons, respects rain locks

Hypothesis:
- Previous -41M score was due to "Ghost Overflow" where multiple STPs delivered to same farm
- Kaggle simulation likely processes one delivery, rejecting the other, or counting storage wrong
- This version ensures 1 Farm = 1 Delivery per day
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v10 - SAFE DRAIN")

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
    print("\nLoading data...")
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
    print("\nPrecomputing...")
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
    
    # Rain Lock (5-day rolling sum > 30mm)
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
    
    # Precompute nearest farms for each STP
    # And check rain lock per day
    stp_nearest_all = {}
    for sid in stp["stp_id"]:
        stp_nearest_all[sid] = sorted(farm["farm_id"].tolist(), 
                                    key=lambda f: distances[sid][f])
    
    rain_lock = {}  # [date][farm_id] -> bool
    for i, ds in enumerate(dates):
        rain_lock[ds] = {}
        for fid in farm["farm_id"]:
            zone = farm_zone[fid]
            locked = False
            if zone in zone_rolling and zone_rolling[zone][i] > 30:
                locked = True
            rain_lock[ds][fid] = locked
            
    # Days to lockout
    days_to_lockout = {}
    for i, ds in enumerate(dates):
        # Check if ANY farm is available
        any_avail = False
        for fid in farm["farm_id"]:
            if not rain_lock[ds][fid]:
                any_avail = True
                break
        
        if not any_avail:
            days_to_lockout[ds] = 0
        else:
            dtl = 999
            for j in range(i+1, min(i+60, len(dates))):
                # Check next day
                day_avail = False
                next_ds = dates[j]
                for fid in farm["farm_id"]:
                    if not rain_lock[next_ds][fid]:
                        day_avail = True
                        break
                if not day_avail:
                    dtl = j - i
                    break
            days_to_lockout[ds] = dtl
            
    print(f"  Precomputation done for {len(dates)} days.")
    return {
        "distances": distances,
        "dates": dates,
        "rain_lock": rain_lock,
        "stp_nearest_all": stp_nearest_all,
        "days_to_lockout": days_to_lockout
    }


def optimize(data, precomputed):
    print("\nOptimizing (Safe Drain)...")
    
    stp = data["stp"]
    farm = data["farm"]
    demand_df = data["demand"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    rain_lock = precomputed["rain_lock"]
    stp_nearest_all = precomputed["stp_nearest_all"]
    days_to_lockout = precomputed["days_to_lockout"]
    
    N_PER_TON = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    TRUCK_CAP = config.get("logistics_constants", {}).get("truck_capacity_tons", 10.0)
    
    # Calculate annual demand for stats
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    annual_demand = {f: demand_df[f].sum() * 1.1 for f in farm_cols}
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    # Initialize
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    cum_n = {f: 0.0 for f in farm_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 50 == 0:
            print(f"    Day {day_i+1}: Storage {sum(storage.values()):.0f}, Overflow: {overflow_total:.0f}")
        
        # Add production
        for s in stp_ids:
            storage[s] += stp_output[s]
            
        # Determine target storage
        dtl = days_to_lockout[ds]
        if dtl <= 2:
            target_pct = 0.0  # Panic drain
        elif dtl <= 7:
            target_pct = 0.0  # Prepare
        elif dtl <= 14:
            target_pct = 0.2
        else:
            target_pct = 0.4
            
        # Track occupied farms for this day (One Farm = One STP)
        occupied_farms = set()
        
        # Optimize draining
        for stp_id in stp_ids:
            target = stp_max[stp_id] * target_pct
            need = storage[stp_id] - target
            
            # If storage is overflowing, we MUST drain
            if storage[stp_id] > stp_max[stp_id]:
                need = max(need, storage[stp_id] - stp_max[stp_id])
            
            if need < 0.1:
                continue
                
            # Iterate nearest farms
            # We filter for: Not Rain Locked, Not Occupied
            nearest_list = stp_nearest_all[stp_id]
            
            # Efficiently find candidates
            candidates = []
            count = 0
            for f in nearest_list:
                if f not in occupied_farms and not rain_lock[ds][f]:
                    candidates.append(f)
                    count += 1
                if count >= 30: # Look at closest 30 available
                    break
            
            # Deliver to candidates
            cand_idx = 0
            while need > 0.1 and cand_idx < len(candidates):
                farm_id = candidates[cand_idx]
                cand_idx += 1
                
                # Check constraints again
                if storage[stp_id] < 0.1:
                    break
                
                # Deliver max possible (10 tons)
                tons = min(10.0, need, storage[stp_id])
                
                # Enforce truck capacity strict
                if tons > TRUCK_CAP:
                    tons = TRUCK_CAP
                    
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
                
                # Update stats
                occupied_farms.add(farm_id)
                cum_n[farm_id] += tons * N_PER_TON
                distance_total += distances[stp_id][farm_id]
                
        # Overflow check
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
                
    # Results
    solution_df = pd.DataFrame(deliveries)
    
    if solution_df.empty:
        print("ERROR: Empty solution")
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
    print(f"  Total tons: {total_tons:,.1f}")
    print(f"  Overflow: {overflow_total:,.1f} tons")
    print(f"  Effective N: {effective_n:,.0f} kg")
    print(f"  Excess N: {excess_n:,.0f} kg")
    print(f"  Distance: {distance_total:,.0f} km")
    print(f"\n  ESTIMATED SCORE:")
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
    print("\nBuilding submission...")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    default_stp = data["stp"]["stp_id"].iloc[0]
    
    sample["stp_id"] = default_stp
    sample["tons_delivered"] = 0.0
    
    # Map deliveries
    lookup = {}
    for _, row in solution_df.iterrows():
        key = (str(row["date"]), str(row["farm_id"]))
        lookup[key] = (str(row["stp_id"]), float(row["tons_delivered"]))
        
    for idx in range(len(sample)):
        key = (str(sample.at[idx, "date"]), str(sample.at[idx, "farm_id"]))
        if key in lookup:
            sample.at[idx, "stp_id"] = lookup[key][0]
            sample.at[idx, "tons_delivered"] = lookup[key][1]
            
    # Final checks
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
