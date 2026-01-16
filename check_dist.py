import pandas as pd
import numpy as np

# Load solution
sol = pd.read_csv('output/solution.csv')
sol = sol[sol['tons_delivered'] > 0]

# Load data
stp = pd.read_csv('data/stp_registry.csv')
farm = pd.read_csv('data/farm_locations.csv')

# Calculate distances
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

stp_coords = {r['stp_id']: (r['lat'], r['lon']) for _, r in stp.iterrows()}
farm_coords = {r['farm_id']: (r['lat'], r['lon']) for _, r in farm.iterrows()}

distances = []
for _, row in sol.iterrows():
    s = stp_coords[row['stp_id']]
    f = farm_coords[row['farm_id']]
    d = haversine(s[0], s[1], f[0], f[1])
    distances.append(d)

sol['distance'] = distances

print('Total deliveries:', len(sol))
print('Total tons:', sol['tons_delivered'].sum())
print('Total distance:', sol['distance'].sum())
print('Avg distance:', sol['distance'].mean())
print('Weighted avg (by tons):', (sol['distance'] * sol['tons_delivered']).sum() / sol['tons_delivered'].sum())

print('\nBy STP:')
for sid in stp['stp_id']:
    stp_sol = sol[sol['stp_id'] == sid]
    if len(stp_sol) > 0:
        print(f'  {sid}: {len(stp_sol)} del, {stp_sol["tons_delivered"].sum():.0f}t, avg {stp_sol["distance"].mean():.1f}km')
