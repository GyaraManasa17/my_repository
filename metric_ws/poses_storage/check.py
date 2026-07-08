import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import numpy as np
# -----------------------------
# Load datasets
# -----------------------------

original = pd.read_csv("kinematic_overlap_dataset.csv")
sampled = pd.read_csv("shared_300_poses_for_both_arms.csv")

print("Original dataset:", len(original))
print("Sampled dataset:", len(sampled))



# -----------------------------
# Create 3D plot
# -----------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

points = sampled[["x","y","z"]].values

tree = KDTree(points)
dist, _ = tree.query(points, k=2)

mean_distance = np.mean(dist[:,1])

print("Mean nearest neighbor distance:", mean_distance)

# Original workspace points
ax.scatter(
    original["x"],
    original["y"],
    original["z"],
    s=1,
    alpha=0.1,
    label="Original 7000 poses"
)

# Sampled points
ax.scatter(
    sampled["x"],
    sampled["y"],
    sampled["z"],
    s=40,
    label="Sampled 300 poses"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

plt.title("Voxel Sampling Verification")

plt.show()








# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("shared_300_poses_for_both_arms.csv")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(df.x, df.y, df.z)

# plt.show()