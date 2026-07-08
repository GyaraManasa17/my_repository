import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

df5 = pd.read_csv("verified_kinematic_overlap_5dof.csv")
df6 = pd.read_csv("verified_kinematic_overlap_6dof.csv")

# --------------------------------------------------
# Extract categories
# --------------------------------------------------

# reachable by 5DOF
r5 = df5[df5["reachable_by_5dof"] == True]

# reachable by 6DOF
r6 = df6[df6["reachable_by_6dof"] == True]

# overlap
both = df6[
    (df6["source"] == "5dof") &
    (df6["reachable_by_6dof"] == True)
]

# 5DOF only
only5 = r5[
    (r5["source"] == "5dof")
]

# 6DOF only
only6 = r6[
    (r6["source"] == "6dof")
]

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    only5["x"], only5["y"], only5["z"],
    s=2, label="5DOF only"
)

ax.scatter(
    only6["x"], only6["y"], only6["z"],
    s=2, label="6DOF only"
)

ax.scatter(
    both["x"], both["y"], both["z"],
    s=4, label="Reachable by both"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Workspace Reachability Comparison")

ax.legend()

plt.show()