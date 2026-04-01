#!/usr/bin/env python3

import pandas as pd

print("Loading datasets...")

df5 = pd.read_csv("verified_kinematic_overlap_5dof.csv")
df6 = pd.read_csv("verified_kinematic_overlap_6dof.csv")

# --------------------------------------------------
# 5DOF source poses reachable by 6DOF
# --------------------------------------------------

subset_5 = df6[
    (df6["source"] == "5dof") &
    (df6["reachable_by_6dof"] == True)
]

# --------------------------------------------------
# 6DOF source poses reachable by 5DOF
# --------------------------------------------------

subset_6 = df5[
    (df5["source"] == "6dof") &
    (df5["reachable_by_5dof"] == True)
]

# --------------------------------------------------
# Combine results
# --------------------------------------------------

combined = pd.concat([subset_5, subset_6], ignore_index=True)

# --------------------------------------------------
# Select only required columns
# --------------------------------------------------

final_cols = [
    "x","y","z",
    "roll","pitch","yaw",
    "source",

    "joint1_5","joint2_5","joint3_5","joint4_5",

    "joint1_6","joint2_6","joint3_6","joint4_6","joint5_roll_6",

    "reachable_by_6dof"
]

final_df = combined[final_cols]

# --------------------------------------------------
# Save dataset
# --------------------------------------------------

output = "kinematic_overlap_dataset.csv"

final_df.to_csv(output,index=False)

print("\n======================================")
print("OVERLAP DATASET CREATED")
print("======================================")

print("Total poses reachable by BOTH robots:", len(final_df))

print("Saved to:", output)