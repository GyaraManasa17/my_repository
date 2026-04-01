#!/usr/bin/env python3

import pandas as pd

def main():
    # 1. Define your two generated files
    file_5dof_results = "verified_kinematic_overlap_5dof.csv"
    file_6dof_results = "verified_kinematic_overlap_6dof.csv"

    print("📥 Loading verification datasets...")
    try:
        df_5 = pd.read_csv(file_5dof_results)
        df_6 = pd.read_csv(file_6dof_results)
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return

    # 2. Extract the Reachability Columns
    # (This handles it even if you didn't change the column name in the 6DOF script)
    col_5 = 'reachable_by_5dof' if 'reachable_by_5dof' in df_5.columns else df_5.columns[-1]
    col_6 = 'reachable_by_6dof' if 'reachable_by_6dof' in df_6.columns else df_6.columns[-1]

    # Combine the data so we can filter it easily
    # Since both scripts processed the exact same 16,000 poses in the exact same order,
    # Row 1 in df_5 is the exact same pose as Row 1 in df_6.
    df_combined = df_5.copy()
    df_combined['reach_5dof'] = df_5[col_5]
    df_combined['reach_6dof'] = df_6[col_6]

    # 3. Filter for poses that are reachable by BOTH robots
    shared_poses = df_combined[(df_combined['reach_5dof'] == True) & (df_combined['reach_6dof'] == True)]

    print("\n" + "="*60)
    print(" 📊 SHARED WORKSPACE ANALYSIS")
    print("="*60)
    print(f"Total poses analyzed:       {len(df_combined)}")
    print(f"Poses reachable by BOTH:    {len(shared_poses)}")
    print("="*60)

    # 4. Extract exactly 300 random poses
    if len(shared_poses) >= 300:
        # random_state=42 ensures you get the exact same 300 poses every time you run this (good for research consistency!)
        final_300 = shared_poses.sample(n=300, random_state=42) 
        print(f"\n🎯 Successfully randomly sampled exactly 300 shared poses.")
    else:
        print(f"\n⚠️ Only found {len(shared_poses)} shared poses! Saving all of them instead of 300.")
        final_300 = shared_poses

    # Clean up the dataframe to just keep the original poses and the source
    # We drop the boolean columns since we already know they are all True
    columns_to_keep =['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'source']
    final_300_clean = final_300[columns_to_keep]

    # 5. Save to a new CSV
    out_file = "shared_300_poses_for_both_arms.csv"
    final_300_clean.to_csv(out_file, index=False)
    print(f"💾 Saved the final 300 poses to: {out_file}\n")

if __name__ == "__main__":
    main()