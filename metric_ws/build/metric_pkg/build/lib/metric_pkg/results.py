import pandas as pd
from scipy.stats import chi2_contingency
import glob

files = glob.glob("results/*.csv")

dfs = {f: pd.read_csv(f) for f in files}

for name, df in dfs.items():
    success_rate = df["success"].mean() * 100
    print(f"{name}: Success Rate = {success_rate:.2f}%")

# Example: compare two specific files
df5 = pd.read_csv("results/benchmark_5dof_task0.csv")
df6 = pd.read_csv("results/benchmark_6dof_task0.csv")

table = [
    [df5.success.sum(), len(df5) - df5.success.sum()],
    [df6.success.sum(), len(df6) - df6.success.sum()]
]

chi2, p, _, _ = chi2_contingency(table)

print("\nChi-Square Test")
print("p-value:", p)

if p < 0.05:
    print("Statistically significant difference.")
else:
    print("No significant difference.")