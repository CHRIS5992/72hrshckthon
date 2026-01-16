"""
OPTIMIZER v9 - STRICT CONSTRAINTS VERSION

CRITICAL FIXES:
1. Cap ALL deliveries at exactly 10 tons (truck capacity)
2. Never exceed daily demand + 10% per farm per day
3. Never deliver to rain-locked farms
4. Drain storage efficiently to minimize overflow
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v9 - STRICT CONSTRAINTS")

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
    
    daily_demand = {}
    for _, row in demand_df.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        daily_demand[ds] = {f: row[f] for f in farm_cols}
    
    # Per-day: available farms with their daily demand
    farms_per_day = {}
    for i, ds in enumerate(dates):
        available = {}
        for fid in farm["farm_id"]:
            zone = farm_zone[fid]
            if zone in zone_rolling and zone_rolling[zone][i] <= 30:
                demand_n = daily_demand[ds].get(fid, 0)
                available[fid] = demand_n
        farms_per_day[ds] = available
    
    # Days to lockout
    days_to_lockout = {}
    for i, ds in enumerate(dates):
        if len(farms_per_day[ds]) == 0:
            days_to_lockout[ds] = 0
        else:
            dtl = 999
            for j in range(i+1, min(i+60, len(dates))):
                if len(farms_per_day[dates[j]]) == 0:
                    dtl = j - i
                    break
            days_to_lockout[ds] = dtl
    
    locked_days = sum(1 for ds in dates if len(farms_per_day[ds]) == 0)
    print(f"  Locked days: {locked_days}")
    
    return {
        "distances": distances,
        "dates": dates,
        "farms_per_day": farms_per_day,
        "days_to_lockout": days_to_lockout,
        "daily_demand": daily_demand
    }


def optimize(data, precomputed):
    print("\nOptimizing...")
    
    stp = data["stp"]
    farm = data["farm"]
    config = data["config"]
    
    distances = precomputed["distances"]
    dates = precomputed["dates"]
    farms_per_day = precomputed["farms_per_day"]
    days_to_lockout = precomputed["days_to_lockout"]
    daily_demand = precomputed["daily_demand"]
    
    N_PER_TON = 25
    BUFFER = 0.10
    TRUCK_CAP = 10.0  # STRICT limit
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    total_daily = sum(stp_output.values())
    total_storage = sum(stp_max.values())
    
    print(f"  Daily output: {total_daily} tons, Storage: {total_storage} tons")
    
    # Initialize storage at 50%
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    
    # Track deliveries: deliveries[date][farm_id] = (stp_id, tons)
    deliveries = {}
    for ds in dates:
        deliveries[ds] = {}
    
    overflow_total = 0.0
    distance_total = 0.0
    effective_n_total = 0.0
    excess_n_total = 0.0
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 50 == 0:
            curr_storage = sum(storage.values())
            print(f"    Day {day_i+1}: Storage {curr_storage:.0f}, Overflow: {overflow_total:.0f}")
        
        # Add daily production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        available_farms = farms_per_day[ds]  # {farm_id: demand_n}
        
        if not available_farms:
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
            target_pct = 0.1
        elif dtl <= 15:
            target_pct = 0.2
        else:
            target_pct = 0.3
        
        # For each available farm, calculate max allowed delivery
        # Max = (demand * (1 + buffer)) / N_PER_TON, capped at TRUCK_CAP
        farm_max_tons = {}
        for fid, demand_n in available_farms.items():
            max_n = demand_n * (1 + BUFFER)
            max_tons = min(max_n / N_PER_TON, TRUCK_CAP)
            if max_tons >= 0.01:
                farm_max_tons[fid] = max_tons
        
        # Sort farms by max capacity (larger capacity = more useful for draining)
        sorted_farms = sorted(farm_max_tons.keys(), key=lambda f: -farm_max_tons[f])
        
        # Deliver from each STP
        for stp_id in stp_ids:
            target = stp_max[stp_id] * target_pct
            need_to_drain = storage[stp_id] - target
            
            if need_to_drain <= 0:
                continue
            
            # Sort available farms by distance from this STP
            stp_farms = sorted(sorted_farms, key=lambda f: distances[stp_id][f])
            
            for farm_id in stp_farms:
                if need_to_drain <= 0:
                    break
                if storage[stp_id] <= 0:
                    break
                
                # Check if already delivered to this farm today
                if farm_id in deliveries[ds]:
                    continue
                
                max_tons = farm_max_tons.get(farm_id, 0)
                if max_tons < 0.01:
                    continue
                
                # Deliver (capped)
                tons = min(max_tons, need_to_drain, storage[stp_id], TRUCK_CAP)
                tons = round(tons, 6)
                
                if tons >= 0.001:
                    deliveries[ds][farm_id] = (stp_id, tons)
                    storage[stp_id] -= tons
                    need_to_drain -= tons
                    
                    n_delivered = tons * N_PER_TON
                    demand_n = available_farms[farm_id]
                    max_allowed_n = demand_n * (1 + BUFFER)
                    
                    effective = min(n_delivered, max_allowed_n)
                    excess = max(0, n_delivered - max_allowed_n)
                    
                    effective_n_total += effective
                    excess_n_total += excess
                    distance_total += distances[stp_id][farm_id]
        
        # End of day overflow
        for s in stp_ids:
            if storage[s] > stp_max[s]:
                overflow_total += storage[s] - stp_max[s]
                storage[s] = stp_max[s]
    
    # Calculate score
    total_tons = sum(tons for ds in deliveries for fid, (sid, tons) in deliveries[ds].items())
    
    n_credit = effective_n_total * 5.0
    soil_credit = total_tons * 1000 * 0.2
    transport_cost = distance_total * 0.9
    excess_cost = excess_n_total * 10.0
    overflow_cost = overflow_total * 1000.0
    score = n_credit + soil_credit - transport_cost - excess_cost - overflow_cost
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total deliveries: {sum(len(d) for d in deliveries.values()):,}")
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
    
    return deliveries


def build_submission(deliveries, data):
    print("\nBuilding submission...")
    
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    default_stp = data["stp"]["stp_id"].iloc[0]
    
    # Initialize
    sample["stp_id"] = default_stp
    sample["tons_delivered"] = 0.0
    
    # Build lookup
    lookup = {}
    for ds in deliveries:
        for fid, (sid, tons) in deliveries[ds].items():
            lookup[(ds, fid)] = (sid, tons)
    
    # Fill
    for idx in range(len(sample)):
        key = (sample.at[idx, "date"], sample.at[idx, "farm_id"])
        if key in lookup:
            stp_id, tons = lookup[key]
            sample.at[idx, "stp_id"] = stp_id
            sample.at[idx, "tons_delivered"] = tons
    
    # Validate
    sample["stp_id"] = sample["stp_id"].astype(str)
    sample["tons_delivered"] = sample["tons_delivered"].astype(float)
    
    # STRICT: Cap at 10 tons
    over_10 = (sample["tons_delivered"] > 10).sum()
    if over_10 > 0:
        print(f"  WARNING: {over_10} rows > 10 tons, capping...")
        sample.loc[sample["tons_delivered"] > 10, "tons_delivered"] = 10.0
    
    sample = sample[["id", "date", "stp_id", "farm_id", "tons_delivered"]]
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "solution.csv"
    sample.to_csv(output_path, index=False)
    
    print(f"\n[SAVED] {output_path}")
    print(f"  Rows: {len(sample):,}")
    print(f"  Non-zero: {(sample['tons_delivered'] > 0).sum():,}")
    print(f"  Total: {sample['tons_delivered'].sum():,.1f} tons")
    print(f"  Max: {sample['tons_delivered'].max():.3f} tons")


def main():
    data = load_data()
    precomputed = precompute(data)
    deliveries = optimize(data, precomputed)
    build_submission(deliveries, data)
    print("\nDONE!")


if __name__ == "__main__":
    main()
