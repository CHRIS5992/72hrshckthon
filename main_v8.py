"""
OPTIMIZER v8 - DEMAND-MATCHED DELIVERIES

CRITICAL FIX: Match deliveries to DAILY nitrogen demand!

Problem discovered:
- Max daily N demand per farm: ~2.2 kg
- Previous approach: 10 ton deliveries = 250 kg N
- This creates ~248 kg EXCESS per delivery = -2,480 CO2 penalty each!

New strategy:
- Deliver ONLY what each farm needs that day (+ 10% buffer)
- Convert N demand to tons: demand_kg / 25 = tons needed
- Small deliveries to many farms are better than large to few
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v8 - DEMAND-MATCHED DELIVERIES")

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
    
    data = {
        "config": config,
        "stp": pd.read_csv(DATA_DIR / "stp_registry.csv"),
        "farm": pd.read_csv(DATA_DIR / "farm_locations.csv"),
        "weather": pd.read_csv(DATA_DIR / "daily_weather_2025.csv"),
        "demand": pd.read_csv(DATA_DIR / "daily_n_demand.csv"),
    }
    
    print(f"  STPs: {len(data['stp'])}, Farms: {len(data['farm'])}")
    return data


def precompute(data):
    print("\nPrecomputing...")
    
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
    
    # Parse dates
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    dates = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zones = [c for c in weather.columns if c != "date"]
    
    # Rain-lock (5-day forward sum > 30mm)
    zone_rolling = {}
    for zone in zones:
        vals = weather[zone].values
        n = len(vals)
        zone_rolling[zone] = np.array([np.sum(vals[i:min(i+5, n)]) for i in range(n)])
    
    farm_zone = dict(zip(farm["farm_id"], farm["zone"]))
    
    # Daily demand lookup
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    
    daily_demand = {}  # daily_demand[date][farm_id] = kg N
    for _, row in demand_df.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        daily_demand[ds] = {f: row[f] for f in farm_cols}
    
    # Available farms per day and their demand
    farms_available = {}
    for i, ds in enumerate(dates):
        farms_available[ds] = {}
        for fid in farm["farm_id"]:
            zone = farm_zone[fid]
            if zone in zone_rolling and zone_rolling[zone][i] <= 30:
                # Farm is available - store its demand
                demand_n = daily_demand[ds].get(fid, 0)
                farms_available[ds][fid] = demand_n
    
    # Days to lockout
    days_to_lockout = {}
    for i, ds in enumerate(dates):
        if len(farms_available[ds]) == 0:
            days_to_lockout[ds] = 0
        else:
            dtl = 999
            for j in range(i+1, min(i+50, len(dates))):
                if len(farms_available[dates[j]]) == 0:
                    dtl = j - i
                    break
            days_to_lockout[ds] = dtl
    
    # Find nearest available farm from each STP for each day
    stp_nearest = {}
    for ds in dates:
        stp_nearest[ds] = {}
        avail = list(farms_available[ds].keys())
        for sid in stp["stp_id"]:
            if avail:
                sorted_farms = sorted(avail, key=lambda f: distances[sid][f])
                stp_nearest[ds][sid] = sorted_farms
            else:
                stp_nearest[ds][sid] = []
    
    total_demand = sum(sum(farms_available[ds].values()) for ds in dates)
    print(f"  Total annual N demand: {total_demand:.0f} kg")
    print(f"  Locked days: {sum(1 for ds in dates if len(farms_available[ds]) == 0)}")
    
    return {
        "distances": distances,
        "dates": dates,
        "farms_available": farms_available,
        "days_to_lockout": days_to_lockout,
        "stp_nearest": stp_nearest,
        "daily_demand": daily_demand
    }


def optimize(data, precomputed):
    print("\nOptimizing (demand-matched)...")
    
    stp = data["stp"]
    farm = data["farm"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    farms_available = precomputed["farms_available"]
    days_to_lockout = precomputed["days_to_lockout"]
    stp_nearest = precomputed["stp_nearest"]
    daily_demand = precomputed["daily_demand"]
    
    N_PER_TON = 25  # kg N per ton biosolid
    BUFFER = 0.10  # 10% buffer allowed
    TRUCK_CAP = 10  # max tons per trip
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    total_daily = sum(stp_output.values())  # 75 tons/day
    total_storage = sum(stp_max.values())   # 1400 tons
    
    print(f"  Daily output: {total_daily} tons, Storage: {total_storage} tons")
    
    # Initialize storage at 50%
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    excess_n_total = 0.0
    effective_n_total = 0.0
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 50 == 0:
            print(f"    Day {day_i+1}: Storage {sum(storage.values()):.0f}, Overflow: {overflow_total:.0f}")
        
        # Add daily production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        avail_farms = farms_available[ds]  # {farm_id: demand_n}
        
        if not avail_farms:
            # All farms locked - check overflow
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    overflow_total += storage[s] - stp_max[s]
                    storage[s] = stp_max[s]
            continue
        
        # Target storage based on lockout
        dtl = days_to_lockout[ds]
        if dtl <= 2:
            target_pct = 0.0
        elif dtl <= 5:
            target_pct = 0.05
        else:
            target_pct = 0.2
        
        # Track N delivered to each farm today
        farm_n_today = {f: 0.0 for f in avail_farms}
        
        # PHASE 1: Deliver to farms with demand (match their demand exactly)
        # Collect all candidate (stp, farm, demand) tuples
        candidates = []
        for stp_id in stp_ids:
            for farm_id in stp_nearest[ds][stp_id]:
                demand_n = avail_farms.get(farm_id, 0)
                if demand_n > 0:
                    dist = distances[stp_id][farm_id]
                    candidates.append((dist, stp_id, farm_id, demand_n))
        
        # Sort by distance (nearest first)
        candidates.sort(key=lambda x: x[0])
        
        # Deliver to meet demand
        for dist, stp_id, farm_id, demand_n in candidates:
            if storage[stp_id] <= 0.01:
                continue
            
            already = farm_n_today[farm_id]
            max_n = demand_n * (1 + BUFFER)  # Allow 10% buffer
            remaining_n = max(0, max_n - already)
            
            if remaining_n <= 0:
                continue
            
            # Convert N to tons
            tons_needed = remaining_n / N_PER_TON
            tons = min(tons_needed, storage[stp_id], TRUCK_CAP)
            
            if tons < 0.001:
                continue
            
            tons = round(tons, 3)
            
            deliveries.append({
                "date": ds,
                "stp_id": stp_id,
                "farm_id": farm_id,
                "tons_delivered": tons
            })
            
            storage[stp_id] -= tons
            n_delivered = tons * N_PER_TON
            farm_n_today[farm_id] += n_delivered
            effective_n_total += n_delivered  # All within demand limit
            distance_total += dist
        
        # PHASE 2: If storage still too high, spread SMALL amounts across farms
        # (This is necessary to prevent overflow during lockouts)
        for stp_id in stp_ids:
            target = stp_max[stp_id] * target_pct
            excess_storage = storage[stp_id] - target
            
            if excess_storage < 0.1:
                continue
            
            # Spread to nearest farms with small amounts
            nearest = stp_nearest[ds][stp_id][:30]
            
            if not nearest:
                continue
            
            # Calculate how to spread
            # To minimize excess N penalty per kg:
            # - Deliver small amounts to many farms
            # - Each farm can take (demand * 1.1 - already_delivered) before excess
            
            farm_idx = 0
            while excess_storage > 0.1 and farm_idx < len(nearest) * 5:
                farm_id = nearest[farm_idx % len(nearest)]
                farm_idx += 1
                
                # Check remaining demand capacity
                demand_n = avail_farms.get(farm_id, 0)
                max_n = demand_n * (1 + BUFFER)
                remaining_capacity = max(0, max_n - farm_n_today.get(farm_id, 0))
                
                if remaining_capacity > 0:
                    # Can deliver within demand - good!
                    tons = min(remaining_capacity / N_PER_TON, excess_storage, TRUCK_CAP)
                    excess_this = 0
                else:
                    # Need to overfill - use small amount
                    # Balance: excess penalty vs overflow penalty
                    # Excess: -10 per kg N = -250 per ton
                    # Overflow: -1000 per ton
                    # So delivering with excess is 4x better than overflow
                    tons = min(0.5, excess_storage, TRUCK_CAP)  # Small load
                    excess_this = tons * N_PER_TON
                
                if tons < 0.01:
                    continue
                
                tons = round(tons, 3)
                
                deliveries.append({
                    "date": ds,
                    "stp_id": stp_id,
                    "farm_id": farm_id,
                    "tons_delivered": tons
                })
                
                storage[stp_id] -= tons
                excess_storage -= tons
                n_delivered = tons * N_PER_TON
                farm_n_today[farm_id] = farm_n_today.get(farm_id, 0) + n_delivered
                
                if excess_this > 0:
                    excess_n_total += excess_this
                else:
                    effective_n_total += n_delivered
                
                distance_total += distances[stp_id][farm_id]
        
        # End of day overflow
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
    
    # Score calculation
    n_credit = effective_n_total * 5.0
    soil_credit = total_tons * 1000 * 0.2
    transport_cost = distance_total * 0.9
    excess_cost = excess_n_total * 10.0
    overflow_cost = overflow_total * 1000.0
    score = n_credit + soil_credit - transport_cost - excess_cost - overflow_cost
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Deliveries: {len(solution_df):,}")
    print(f"  Total tons: {total_tons:,.1f}")
    print(f"  Overflow: {overflow_total:,.1f} tons")
    print(f"  Effective N: {effective_n_total:,.0f} kg")
    print(f"  Excess N: {excess_n_total:,.0f} kg")
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
    
    # Group by (date, farm) and sum
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
    print(f"  Total: {sample['tons_delivered'].sum():,.1f} tons")


def main():
    data = load_data()
    precomputed = precompute(data)
    solution = optimize(data, precomputed)
    if not solution.empty:
        build_submission(solution, data)
        print("\nDONE!")


if __name__ == "__main__":
    main()
