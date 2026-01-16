import pandas as pd

sol = pd.read_csv('output/solution.csv')
sample = pd.read_csv('data/sample_submission.csv')

# Check if id, date, farm_id match exactly
match_id = (sol['id'] == sample['id']).all()
match_date = (sol['date'] == sample['date']).all()
match_farm = (sol['farm_id'] == sample['farm_id']).all()

print("ID match:", match_id)
print("Date match:", match_date)  
print("Farm match:", match_farm)

# Check unique STPs used
print()
print("Unique STPs in solution:", sol["stp_id"].nunique())
print("STPs used:", list(sol['stp_id'].unique()))

# Check delivery statistics
del_rows = sol[sol['tons_delivered'] > 0]
print()
print("Rows with deliveries:", len(del_rows))
print("Total tons:", sol["tons_delivered"].sum())
print("Max tons in single row:", sol["tons_delivered"].max())
print("Rows > 10 tons:", (sol["tons_delivered"] > 10).sum())

# Check for any issues
print()
print("Null values:", sol.isnull().sum().sum())
print("Negative values:", (sol['tons_delivered'] < 0).sum())
