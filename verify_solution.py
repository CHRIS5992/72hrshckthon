import pandas as pd

# Read the solution
df = pd.read_csv('solution.csv')

print('=' * 60)
print('KAGGLE SUBMISSION VERIFICATION')
print('=' * 60)

print(f'\nTotal rows: {len(df):,}')
print(f'Expected: 91,250')
print(f'Match: {"✓" if len(df) == 91250 else "✗"}')

print(f'\nColumns: {list(df.columns)}')
print(f'Expected: [\'id\', \'date\', \'farm_id\', \'stp_id\', \'tons_delivered\']')

print(f'\nData types:')
print(df.dtypes)

print(f'\nNull counts (pandas default reading):')
print(df.isnull().sum())
print(f'Total nulls: {df.isnull().sum().sum()}')

print(f'\nNull counts (with keep_default_na=False):')
df2 = pd.read_csv('solution.csv', keep_default_na=False)
print(df2.isnull().sum())
print(f'Total nulls: {df2.isnull().sum().sum()}')

print(f'\nDelivery statistics:')
print(f'  Zero deliveries: {(df["tons_delivered"]==0).sum():,}')
print(f'  Non-zero deliveries: {(df["tons_delivered"]>0).sum():,}')

print(f'\nFirst 3 rows:')
print(df.head(3))

print(f'\nSample delivery row:')
print(df[df["tons_delivered"]>0].head(1))

print('\n' + '=' * 60)
print('VERIFICATION COMPLETE')
print('=' * 60)
