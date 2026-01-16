"""
OPTIMIZER v4 - PREDICTIVE DRAIN STRATEGY

Key insights:
1. There are 86 days when ALL farms are rain-locked
2. Longest consecutive lockout: ~28 days
3. During lockout: 28 * 75 = 2100 tons produced, only 1400 can be stored
4. UNAVOIDABLE overflow: ~700 tons

Strategy:
1. PREDICT when lockouts are coming
2. DRAIN storage to near-zero before lockouts
3. Accept overflow during long lockouts (unavoidable)
4. Spread excess N across all farms
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("OPTIMIZER v4 - PREDICTIVE DRAIN STRATEGY")

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
    
    # Rain-lock calculation
    threshold = config.get("environmental_thresholds", {}).get("rain_lock_threshold_mm", 30.0)
    window = 5
    
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.sort_values("date").reset_index(drop=True)
    dates = weather["date"].dt.strftime("%Y-%m-%d").tolist()
    zones = [c for c in weather.columns if c != "date"]
    
    # Forward rolling sums
    zone_rolling = {}
    for zone in zones:
        vals = weather[zone].values
        n = len(vals)
        rolling = np.array([np.sum(vals[i:min(i+window, n)]) for i in range(n)])
        zone_rolling[zone] = rolling
    
    farm_zone = dict(zip(farm["farm_id"], farm["zone"]))
    
    rain_lock = {}
    for fid in farm["farm_id"]:
        zone = farm_zone[fid]
        if zone in zone_rolling:
            rain_lock[fid] = {dates[i]: zone_rolling[zone][i] > threshold 
                            for i in range(len(dates))}
        else:
            rain_lock[fid] = {d: False for d in dates}
    
    # Count available farms per day
    farms_available = []
    for i, ds in enumerate(dates):
        avail = [f for f in farm["farm_id"] if not rain_lock[f][ds]]
        farms_available.append(avail)
    
    # Track days until next full lockout and lockout duration
    days_to_lockout = []
    lockout_duration = []
    
    for i in range(len(dates)):
        if len(farms_available[i]) == 0:
            # Currently locked
            days_to_lockout.append(0)
            # Count consecutive locked days
            count = 1
            for j in range(i+1, len(dates)):
                if len(farms_available[j]) == 0:
                    count += 1
                else:
                    break
            lockout_duration.append(count)
        else:
            # Find next lockout
            found = False
            for j in range(i+1, len(dates)):
                if len(farms_available[j]) == 0:
                    days_to_lockout.append(j - i)
                    # Count lockout duration
                    count = 1
                    for k in range(j+1, len(dates)):
                        if len(farms_available[k]) == 0:
                            count += 1
                        else:
                            break
                    lockout_duration.append(count)
                    found = True
                    break
            if not found:
                days_to_lockout.append(999)
                lockout_duration.append(0)
    
    print(f"  Days with no available farms: {sum(1 for f in farms_available if len(f) == 0)}")
    print(f"  Longest lockout: {max(lockout_duration)} days")
    
    return {
        "distances": distances,
        "rain_lock": rain_lock,
        "dates": dates,
        "farms_available": farms_available,
        "days_to_lockout": days_to_lockout,
        "lockout_duration": lockout_duration
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
    lockout_duration = precomputed["lockout_duration"]
    
    truck_cap = config.get("logistics_constants", {}).get("truck_capacity_tons", 10)
    n_per_ton = config.get("agronomic_constants", {}).get("nitrogen_content_kg_per_ton_biosolid", 25)
    buffer_pct = config.get("agronomic_constants", {}).get("application_buffer_percent", 10)
    
    # Parse demand
    demand_df = demand_df.copy()
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    farm_cols = [c for c in demand_df.columns if c != "date"]
    
    # Annual demand per farm
    annual_demand = {}
    for f in farm_cols:
        annual_demand[f] = demand_df[f].sum() * (1 + buffer_pct / 100)
    
    stp_ids = stp["stp_id"].tolist()
    farm_ids = farm["farm_id"].tolist()
    stp_output = dict(zip(stp["stp_id"], stp["daily_output_tons"]))
    stp_max = dict(zip(stp["stp_id"], stp["storage_max_tons"]))
    
    total_daily = sum(stp_output.values())
    total_storage = sum(stp_max.values())
    
    print(f"  Daily output: {total_daily} tons")
    print(f"  Storage: {total_storage} tons")
    print(f"  Total annual demand (N): {sum(annual_demand.values()):.0f} kg")
    
    # Initialize
    storage = {s: stp_max[s] * 0.5 for s in stp_ids}
    cumulative_n = {f: 0.0 for f in farm_ids}
    
    # Precompute nearest farms for each STP
    stp_sorted_farms = {
        s: sorted(farm_ids, key=lambda f: distances[s][f])
        for s in stp_ids
    }
    
    deliveries = []
    overflow_total = 0.0
    distance_total = 0.0
    
    print(f"\n  Processing {len(dates)} days...")
    
    for day_i, ds in enumerate(dates):
        if (day_i + 1) % 50 == 0:
            curr = sum(storage.values())
            print(f"    Day {day_i+1}: Storage {curr:.0f}, Overflow: {overflow_total:.0f}")
        
        # Add daily production
        for s in stp_ids:
            storage[s] += stp_output[s]
        
        available = farms_available[day_i]
        
        if not available:
            # All farms locked - check overflow
            for s in stp_ids:
                if storage[s] > stp_max[s]:
                    overflow_total += storage[s] - stp_max[s]
                    storage[s] = stp_max[s]
            continue
        
        # Determine target storage level based on upcoming lockout
        dtl = days_to_lockout[day_i]
        duration = lockout_duration[day_i]
        
        if dtl == 0:
            # We're unlocking today - drain as much as possible
            target_pct = 0.1
        elif dtl <= 5:
            # Lockout coming soon - drain to zero
            target_pct = 0.0
        elif dtl <= 10:
            # Lockout in 5-10 days - start draining
            # Need to deliver (storage + dtl*daily - capacity_for_lockout)
            capacity_for_lockout = total_storage  # Want to be at max before lockout
            target_pct = 0.1
        else:
            # No immediate lockout - maintain comfortable level
            target_pct = 0.3
        
        # Calculate how much to deliver
        current_storage = sum(storage.values())
        target_storage = total_storage * target_pct
        need_to_deliver = max(0, current_storage - target_storage)
        
        if need_to_deliver < 1:
            continue
        
        # Categorize farms by remaining capacity
        farm_remaining = {f: annual_demand.get(f, 0) - cumulative_n[f] for f in available}
        
        # Sort: farms with most remaining capacity first
        sorted_farms = sorted(available, key=lambda f: -farm_remaining[f])
        
        # Deliver from each STP
        for stp_id in stp_ids:
            stp_need = max(0, storage[stp_id] - stp_max[stp_id] * target_pct)
            if stp_need < 1:
                continue
            
            # Get farms sorted by distance for this STP
            stp_farms = [f for f in stp_sorted_farms[stp_id] if f in available]
            
            # Prefer farms with remaining capacity
            with_cap = [f for f in stp_farms if farm_remaining.get(f, 0) > 0]
            without_cap = [f for f in stp_farms if farm_remaining.get(f, 0) <= 0]
            
            priority = with_cap + without_cap
            
            if not priority:
                continue
            
            farm_idx = 0
            iterations = 0
            max_iter = len(priority) * 20
            
            while stp_need > 0.01 and iterations < max_iter:
                iterations += 1
                farm_id = priority[farm_idx % len(priority)]
                farm_idx += 1
                
                remaining = farm_remaining.get(farm_id, 0)
                
                # Determine delivery size
                if remaining > 0:
                    tons = min(truck_cap, remaining / n_per_ton, stp_need, storage[stp_id])
                else:
                    # Spread among farms - smaller deliveries
                    tons = min(truck_cap / 2, stp_need, storage[stp_id])
                
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
                stp_need -= tons
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
    
    total_tons = solution_df["tons_delivered"].sum()
    effective_n = sum(min(cumulative_n[f], annual_demand.get(f, 0)) for f in farm_ids)
    excess_n = sum(max(0, cumulative_n[f] - annual_demand.get(f, 0)) for f in farm_ids)
    
    # Score calculation
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
    
    sample_path = DATA_DIR / "sample_submission.csv"
    final_df = pd.read_csv(sample_path)
    default_stp = data["stp"]["stp_id"].iloc[0]
    
    final_df["stp_id"] = default_stp
    final_df["tons_delivered"] = 0.0
    
    # Group and sum deliveries by (date, farm)
    grouped = solution_df.groupby(["date", "farm_id"], as_index=False).agg({
        "stp_id": "first",
        "tons_delivered": "sum"
    })
    
    lookup = {
        (str(r["date"]), str(r["farm_id"])): (str(r["stp_id"]), float(r["tons_delivered"]))
        for _, r in grouped.iterrows()
    }
    
    for idx in range(len(final_df)):
        key = (str(final_df.at[idx, "date"]), str(final_df.at[idx, "farm_id"]))
        if key in lookup:
            final_df.at[idx, "stp_id"] = lookup[key][0]
            final_df.at[idx, "tons_delivered"] = lookup[key][1]
    
    final_df["stp_id"] = final_df["stp_id"].fillna(default_stp).astype(str)
    final_df["tons_delivered"] = final_df["tons_delivered"].fillna(0.0).astype(float)
    final_df = final_df[["id", "date", "stp_id", "farm_id", "tons_delivered"]]
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "solution.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n[SAVED] {output_path}")
    print(f"  Rows: {len(final_df):,}")
    print(f"  Non-zero: {(final_df['tons_delivered'] > 0).sum():,}")
    print(f"  Total: {final_df['tons_delivered'].sum():,.1f} tons")
    
    return final_df


def main():
    data = load_data()
    precomputed = precompute(data)
    solution = optimize(data, precomputed)
    if not solution.empty:
        build_submission(solution, data)


if __name__ == "__main__":
    main()
