"""
Analysis of optimal delivery strategy
"""
n_per_ton = 25  # kg N per ton biosolid
annual_demand = 40365  # kg N (with buffer)
annual_production = 27375  # tons biosolids
storage = 1400  # tons max
daily_production = 75  # tons/day

# What the farms can absorb (converted to tons)
can_absorb_tons = annual_demand / n_per_ton
print(f'Farms can absorb: {can_absorb_tons:.0f} tons biosolids (for N demand)')
print(f'Annual production: {annual_production} tons')
print()

# Find optimal delivery amount
print('Finding optimal...')
results = []

for delivered in range(500, 28000, 500):
    # Overflow is unavoidable if we can't deliver enough
    overflow = max(0, annual_production - delivered)
    
    # N calculations
    n_delivered = delivered * n_per_ton
    excess_n = max(0, n_delivered - annual_demand)
    effective_n = min(n_delivered, annual_demand)
    
    # Credits and penalties
    soil = delivered * 1000 * 0.2
    n_cred = effective_n * 5.0
    trans = delivered * 7 * 0.9  # Rough estimate: 7km avg per ton
    excess_pen = excess_n * 10.0
    overflow_pen = overflow * 1000.0
    
    score = soil + n_cred - trans - excess_pen - overflow_pen
    
    results.append({
        'delivered': delivered,
        'overflow': overflow,
        'excess_n': excess_n,
        'score': score
    })

# Sort by score
results.sort(key=lambda x: -x['score'])

print("\nTop 10 scenarios by score:")
print("-" * 70)
for r in results[:10]:
    print(f"Deliver {r['delivered']:,} tons | Overflow: {r['overflow']:,} | Excess N: {r['excess_n']:,} kg | Score: {r['score']:,.0f}")

print("\n\nBottom 5 scenarios:")
print("-" * 70)
for r in results[-5:]:
    print(f"Deliver {r['delivered']:,} tons | Overflow: {r['overflow']:,} | Excess N: {r['excess_n']:,} kg | Score: {r['score']:,.0f}")

# Best scenario
best = results[0]
print("\n" + "=" * 70)
print(f"OPTIMAL: Deliver {best['delivered']:,} tons")
print(f"  Overflow: {best['overflow']:,} tons")
print(f"  Excess N: {best['excess_n']:,} kg")
print(f"  Score: {best['score']:,.0f}")
print("=" * 70)
