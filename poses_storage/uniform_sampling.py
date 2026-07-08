import pandas as pd
import numpy as np

INPUT_FILE = "kinematic_overlap_dataset.csv"
OUTPUT_FILE = "shared_300_poses_for_both_arms.csv"

TARGET = 300
VOXEL_SIZE = 0.05   # 5 cm voxels


print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print("Total poses:", len(df))


# ---------------------------------------
# Compute voxel indices
# ---------------------------------------

df["vx"] = np.floor(df["x"] / VOXEL_SIZE)
df["vy"] = np.floor(df["y"] / VOXEL_SIZE)
df["vz"] = np.floor(df["z"] / VOXEL_SIZE)

df["voxel"] = list(zip(df.vx, df.vy, df.vz))


# ---------------------------------------
# Group by voxel
# ---------------------------------------

groups = df.groupby("voxel")

selected = []

for _, group in groups:

    # pick one random pose from voxel
    pose = group.sample(1)

    selected.append(pose)

selected_df = pd.concat(selected)


print("Unique voxels:", len(selected_df))


# ---------------------------------------
# If more than 300, randomly trim
# ---------------------------------------

if len(selected_df) > TARGET:
    selected_df = selected_df.sample(TARGET)


# ---------------------------------------
# If less than 300, fill randomly
# ---------------------------------------

elif len(selected_df) < TARGET:

    remaining = TARGET - len(selected_df)

    extra = df.sample(remaining)

    selected_df = pd.concat([selected_df, extra])


# ---------------------------------------
# Remove helper columns
# ---------------------------------------

selected_df = selected_df.drop(columns=["vx","vy","vz","voxel"])


# ---------------------------------------
# Save
# ---------------------------------------

selected_df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Final poses:", len(selected_df))