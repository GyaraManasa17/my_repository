import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kinematic_overlap_dataset.csv")

# remove NaNs
roll = df["joint5_roll_6"].dropna()

print("=================================")
print("Joint5 Roll Usage Statistics")
print("=================================")

print("Samples:", len(roll))
print("Min:", roll.min())
print("Max:", roll.max())
print("Mean:", roll.mean())
print("Std:", roll.std())

# --------------------------------------------------
# Histogram
# --------------------------------------------------

plt.hist(roll, bins=50)

plt.xlabel("Joint5 Roll Angle (rad)")
plt.ylabel("Frequency")

plt.title("Distribution of Wrist Roll Joint Usage")

plt.show()